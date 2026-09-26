"""Proximal / gradient / projection atoms with closed forms.

All proxes are (1/2)-averaged operators [LSCOMO SS2.5]; gradients of
L-smooth convex functions are (1/L)-cocoercive [LSCOMO SS2.2.1,
Baillon-Haddad]. Conjugate proxes are obtained by the Moreau identity

    prox_{s h*}(v) = v - s prox_{h/s}(v/s)          [LSCOMO SS2.5]

which for h = lam*||.||_1 gives componentwise clipping to [-lam, lam].
"""
from __future__ import annotations

from dataclasses import dataclass, field

from typing import Any

import jax
import jax.numpy as jnp

from atlas.operators.base import PropSet

# ---------------------------------------------------------------------------
# Stateless functional forms
# ---------------------------------------------------------------------------


def grad_quadratic(x, y):
    """Gradient of f(x) = (1/2)||x - y||^2: grad f(x) = x - y.

    f is 1-smooth and 1-strongly convex; grad f is 1-Lipschitz and
    1-cocoercive [LSCOMO SS2.2.1].
    """
    return x - y


def prox_l1(v, step, lam=1.0):
    """prox_{step*lam*||.||_1}(v) = soft-threshold at step*lam [LSCOMO SS2.5]."""
    t = step * lam
    return jnp.sign(v) * jnp.maximum(jnp.abs(v) - t, 0.0)


def prox_l2sq(v, step, y=0.0):
    """prox of step * (1/2)||. - y||^2: (v + step*y)/(1 + step). Closed form."""
    return (v + step * y) / (1.0 + step)


def prox_box(v, lo, hi):
    """Projection onto the box [lo, hi] (prox of its indicator): clip."""
    return jnp.clip(v, lo, hi)


def prox_zero(v, step=None):
    """prox of the zero function: identity."""
    return v


def project_linf_ball(v, radius):
    """Projection onto {u : ||u||_inf <= radius}: componentwise clip."""
    return jnp.clip(v, -radius, radius)


def prox_conj(prox_fn, v, s):
    """Moreau identity: prox_{s h*}(v) = v - s * prox_{h/s}(v / s)."""
    return v - s * prox_fn(v / s, 1.0 / s)


def prox_scaled_l1_conj(v, s, lam=1.0):
    """prox_{s (lam||.||_1)*}(v) = clip(v, -lam, lam), any s > 0.

    Follows from the Moreau identity: (lam||.||_1)* = indicator of the
    lam-radius inf-ball, whose prox is the projection, independent of s.
    """
    return project_linf_ball(v, lam)


# ---------------------------------------------------------------------------
# Atom objects (carry PropSet tags for the type system)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QuadraticDataFidelity:
    """f(x) = (1/2)||x - y||^2. 1-smooth, 1-strongly convex, prox-friendly."""

    y: object
    props: PropSet = field(
        default=PropSet(
            lipschitz=1.0,
            cocoercivity=1.0,
            averagedness=0.5,
            prox_friendly=True,
            strong_convexity=1.0,
        )
    )

    def grad(self, x):
        return grad_quadratic(x, self.y)

    def prox(self, v, step):
        return prox_l2sq(v, step, self.y)

    def grad_conj(self, v):
        """Gradient of the conjugate f*(v) = (1/2)||v||^2 + <y, v>:
        grad f*(v) = y + v (affine, unit linear part). Since f is 1-strongly
        convex, f* is 1-smooth; grad f* inverts grad f [LSCOMO SS2.2]."""
        return self.y + v


@dataclass(frozen=True)
class ScaledL1:
    """h(z) = lam * ||z||_1 (anisotropic). Prox-friendly, nonsmooth."""

    lam: float
    props: PropSet = field(default=PropSet(averagedness=0.5, prox_friendly=True))

    def prox(self, v, step):
        return prox_l1(v, step, self.lam)

    def prox_conj(self, v, s):
        """prox_{s h*} = projection onto the lam inf-ball (Moreau)."""
        return prox_scaled_l1_conj(v, s, self.lam)


@dataclass(frozen=True)
class Zero:
    """g(x) = 0. prox = identity."""

    props: PropSet = field(default=PropSet(averagedness=0.5, prox_friendly=True))

    def prox(self, v, step):
        return prox_zero(v, step)


@dataclass(frozen=True)
class Box:
    """Indicator of [lo, hi]; prox = clip (a projection, (1/2)-averaged)."""

    lo: float
    hi: float
    props: PropSet = field(default=PropSet(averagedness=0.5, prox_friendly=True))

    def prox(self, v, step=None):
        return prox_box(v, self.lo, self.hi)


def project_simplex(v):
    """Euclidean projection onto {x >= 0, sum x = 1} (Duchi, Shalev-Shwartz, Singer, Chandra 2008).

    Sort descending, find the largest rho whose threshold keeps the entry positive, subtract it, clip.
    O(n log n) and jittable: the rho search is an argmax over a boolean mask rather than a loop.
    """
    n = v.shape[0]
    mu = jnp.sort(v)[::-1]
    css = jnp.cumsum(mu) - 1.0
    idx = jnp.arange(1, n + 1, dtype=v.dtype)
    cond = mu - css / idx > 0
    rho = jnp.maximum(jnp.sum(cond), 1)                 # at least one coordinate survives
    theta = css[rho - 1] / rho.astype(v.dtype)
    return jnp.maximum(v - theta, 0.0)


def project_box_hyperplane(v, lo, hi, a, b, iters=60):
    """Projection onto {lo <= x <= hi, a'x = b}, the continuous knapsack set.

    x(mu) = clip(v - mu*a, lo, hi) is nonincreasing in mu along a, so a'x(mu) - b has one sign change
    and bisection converges. Fixed iteration count so it is jittable and has no data-dependent control
    flow. This generalizes the simplex case: a = 1, b = 1, lo = 0, hi = +inf.
    """
    def phi(mu):
        return jnp.sum(a * jnp.clip(v - mu * a, lo, hi)) - b

    # bracket: mu = 0 plus a spread wide enough that clipping saturates at both ends
    scale = jnp.maximum(jnp.max(jnp.abs(v)) / jnp.maximum(jnp.max(jnp.abs(a)), 1e-30), 1.0)
    lo_mu, hi_mu = -1e3 * scale, 1e3 * scale

    def body(_, state):
        l_, h_ = state
        m = 0.5 * (l_ + h_)
        # phi is nonincreasing in mu: if phi(m) > 0 the root lies to the right
        return jnp.where(phi(m) > 0, m, l_), jnp.where(phi(m) > 0, h_, m)

    l_, h_ = jax.lax.fori_loop(0, iters, body, (lo_mu, hi_mu))
    return jnp.clip(v - 0.5 * (l_ + h_) * a, lo, hi)


@dataclass(frozen=True)
class EqualitySet:
    """Indicator of {v : v = c} for a VECTOR c. prox is the constant map v -> c.

    Box(lo, hi) takes scalars, so it can only express an equality whose right-hand side is a single
    repeated value; Box(0, 0) means Lx = 0. A family whose equality is Lx = c with c nonzero needs
    this. Using Box(0,0) there silently solves a DIFFERENT problem that still converges and is still
    feasible on every other row, which is how it escapes notice: measured objective gap 2.6e+06 with
    row feasibility at 3.6e-12.
    """

    c: object
    props: PropSet = field(default=PropSet(averagedness=0.5, prox_friendly=True))

    def prox(self, v, step=None):
        return jnp.broadcast_to(self.c, v.shape)


@dataclass(frozen=True)
class BoxHyperplaneProduct:
    """Indicator of {lo <= x <= hi, a'x = b} on the leading block, free on the rest.

    The general form of the simplex case. Exists to test whether the diagnose-and-relocate rule has
    more than one instance: a DENSE equality row in L sets ||L|| for the whole problem, and if the set
    it defines has a cheap projection it belongs here instead.
    """

    a: object                      # dense row coefficients, length n_x (a data field: it is an array)
    b: float                       # right-hand side
    n_x: int                       # static: slice length
    lo: float
    hi: float
    props: PropSet = field(default=PropSet(averagedness=0.5, prox_friendly=True))

    def prox(self, v, step=None):
        x, y = v[: self.n_x], v[self.n_x :]
        return jnp.concatenate([project_box_hyperplane(x, self.lo, self.hi, self.a, self.b), y])


@dataclass(frozen=True)
class SimplexProduct:
    """Indicator of {x in the probability simplex} x {y free}, prox = projection on the x block only.

    Exists because a dense budget row (sum x = 1) carried into the constraint operator L sets the whole
    step size. Measured on portfolio n=5000 after Ruiz: the top singular direction of the rescaled A
    puts 99.91% of its mass on that ONE row, ||A|| = 33.970, and dropping the row leaves 2.129, a 255x
    gain in the tau*sigma*||L||^2 bound. Diagonal equilibration cannot remove a dense rank-one row.
    Handling {x >= 0, sum x = 1} here as an exact simplex projection takes it out of L entirely.

    n_x: size of the leading block that is constrained to the simplex. The remainder passes through.
    """

    n_x: int
    props: PropSet = field(default=PropSet(averagedness=0.5, prox_friendly=True))

    def prox(self, v, step=None):
        x, y = v[: self.n_x], v[self.n_x :]
        return jnp.concatenate([project_simplex(x), y])


@dataclass(frozen=True)
class QuadraticForm:
    """f(x) = (1/2) x'Px + q'x + r, P symmetric positive semidefinite.

    Smooth with L_f = ||P||_2 (supplied by the loader as a certified bound).
    Deliberately has NO prox: the prox of a general quadratic is a linear solve
    with (I + tP). A scheme that needs prox_f as its primal block (PDHG) is
    therefore rejected at build time, while Condat-Vu, which needs only grad_f,
    admits it. That asymmetry is what makes the method axis live on QP, where
    on LP the two schemes coincide because f is affine.

    P is a static operator (identity-hashed SparseLinOp); q is a dynamic leaf.
    """

    P: Any
    q: Any
    r: float = 0.0
    props: PropSet = field(default=PropSet(prox_friendly=False))

    def grad(self, x):
        return self.P.forward(x) + self.q

    def value(self, x):
        return 0.5 * jnp.vdot(x, self.P.forward(x)) + jnp.vdot(self.q, x) + self.r


# ---------------------------------------------------------------------------
# Pytree registration: atom DATA (y, lam, lo, hi) are dynamic leaves, so
# compiled solver kernels take the problem data as runtime arguments (one
# compile per scheme per shape, not per instance); PropSet tags are static
# metadata. See atlas.wrappers.km for the kernel cache that relies on this.
# ---------------------------------------------------------------------------

jax.tree_util.register_dataclass(
    QuadraticDataFidelity, data_fields=["y"], meta_fields=["props"]
)
jax.tree_util.register_dataclass(
    ScaledL1, data_fields=["lam"], meta_fields=["props"]
)
jax.tree_util.register_dataclass(Zero, data_fields=[], meta_fields=["props"])
jax.tree_util.register_dataclass(
    SimplexProduct, data_fields=[], meta_fields=["n_x", "props"]
)
jax.tree_util.register_dataclass(
    BoxHyperplaneProduct, data_fields=["a", "b"],
    meta_fields=["n_x", "lo", "hi", "props"]
)
jax.tree_util.register_dataclass(
    EqualitySet, data_fields=["c"], meta_fields=["props"]
)
jax.tree_util.register_dataclass(
    Box, data_fields=["lo", "hi"], meta_fields=["props"]
)
jax.tree_util.register_dataclass(
    QuadraticForm, data_fields=["q"], meta_fields=["P", "r", "props"]
)
