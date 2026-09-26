"""The five base schemes on the E1 composite  min f(x) + g(x) + h(Lx).

GENERAL: every scheme solves any CompositeProblem exposing the pieces its
structure needs (grad via f.grad + f.props.lipschitz, proxes via
atom.prox / atom.prox_conj (Moreau fallback from atom.prox), the linear
map via L.forward / L.adjoint / L.norm_bound). The hard-coded TV dispatch
of the P0 build is gone; the TV instance is just the composite with
f = (1/2)||.-y||^2, g = 0, h = lam||.||_1, L = Grad2D, and every solver
reproduces the P0 numerics on it exactly.

Structure map:
- FBS  [LSCOMO SS2.7]: f smooth + ONE prox block (L absent), OR the DUAL
  of  f strongly convex + h(Lx)  (g absent): with mu = strong convexity
  of f, f* is (1/mu)-smooth and the dual  min_u f*(-L^T u) + h*(u)  is
  smooth + prox with L_dual = ||L||^2 / mu; the primal iterate is
  recovered exactly as x = grad f*(-L^T u). For TV (mu = 1,
  grad f*(v) = y + v) this is the projected-gradient dual solver of P0.
- DRS  [LSCOMO SS2.7.1]: two prox blocks (f absent, L absent), OR the same
  dual as FBS when f is a quadratic data fidelity (grad f* affine, so the
  resolvent of the smooth dual term is an (I + (a/mu) L L^T) solve by CG).
- PDHG [LSCOMO SS3.3]: ONE primal prox block (f XOR g) + h(Lx).
- Condat-Vu [LSCOMO SS3.3 eq. 3.13]: f smooth + g prox + h(Lx).
- DYS  [LSCOMO SS2.7 + SS13.4.2 Thm 28]: f smooth + g prox + h prox
  (L absent; the canonical three-operator splitting).

A structurally inapplicable (problem, scheme) pair raises TypeCheckError
(a construction failure, recorded as such by bench runners).

Termination metric: when the problem has the (quadratic + lam||.||_1 + L)
structure, the RIGOROUS relative duality gap (atlas.certify.residuals) is
the Level-3 certificate and the termination metric. Otherwise the
normalized fixed-point residual ||T(z) - z|| / (1 + ||z||) is used
(documented: no runtime gap certificate for such problems yet).

SPEED: stepsizes, problem data (atom pytree leaves) and wrapper knobs are
DYNAMIC arguments of a kernel compiled once per scheme per shape and
cached (atlas.wrappers.km); a new genome on the same shape pays zero
recompilation.
"""
from __future__ import annotations

import dataclasses
import os as _os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np

from atlas.certify.residuals import fp_residual, logreg_enet_rel_gap, lp_rel_gap, qp_rel_kkt, rel_gap
from atlas.ir.spec import CompositeProblem
from atlas.operators.prox import QuadraticDataFidelity, ScaledL1, Zero
from atlas.typing_ import (
    ConstructionCert,
    TypeCheckError,
    typecheck_condat_vu,
    typecheck_drs,
    typecheck_dys,
    typecheck_fbs,
    typecheck_halpern,
    typecheck_pdhg,
    typecheck_relaxation,
    typecheck_restart,
)
from atlas.wrappers.km import Budget, km_run


@dataclass
class SolveResult:
    x: np.ndarray
    u: Optional[np.ndarray]
    iters: int
    fp_residual_history: np.ndarray  # sampled every budget.sample_every
    rel_gap_history: np.ndarray  # sampled alongside (or normalized FP res)
    wall_time: float
    cert: Optional[ConstructionCert] = None
    extra: Optional[dict] = None  # e.g. switch-monitor accept/reject counts


# ---------------------------------------------------------------------------
# Params
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FBSParams:
    step: float
    rho: Optional[float] = None  # over-relaxation in (0, 2)
    halpern: bool = False
    restart_k: Optional[int] = None  # Halpern anchor restart (needs halpern)
    outer: Optional[str] = None  # registered outer wrapper (atlas.wrappers.registry)


@dataclass(frozen=True)
class DRSParams:
    step: float
    rho: Optional[float] = None
    halpern: bool = False
    restart_k: Optional[int] = None
    outer: Optional[str] = None  # registered outer wrapper (atlas.wrappers.registry)
    cg_tol: float = 1e-12
    cg_maxiter: int = 200


@dataclass(frozen=True)
class PDHGParams:
    t: float
    s: float
    rho: Optional[float] = None
    halpern: bool = False
    restart_k: Optional[int] = None
    outer: Optional[str] = None  # registered outer wrapper (atlas.wrappers.registry)


@dataclass(frozen=True)
class CondatVuParams:
    t: float
    s: float
    rho: Optional[float] = None
    halpern: bool = False
    restart_k: Optional[int] = None
    outer: Optional[str] = None  # registered outer wrapper (atlas.wrappers.registry)


@dataclass(frozen=True)
class CondatVuMetricParams:
    s: float
    metric: Any  # atlas.operators.primal_metric.PrimalMetric (the primal step T)
    rho: Optional[float] = None
    halpern: bool = False
    restart_k: Optional[int] = None
    outer: Optional[str] = None


@dataclass(frozen=True)
class FISTAParams:
    step: float
    restart: bool = True
    rho: Optional[float] = None
    halpern: bool = False
    restart_k: Optional[int] = None
    outer: Optional[str] = None


@dataclass(frozen=True)
class DYSParams:
    step: float
    rho: Optional[float] = None
    halpern: bool = False
    restart_k: Optional[int] = None
    outer: Optional[str] = None  # registered outer wrapper (atlas.wrappers.registry)


# ---------------------------------------------------------------------------
# Structure helpers
# ---------------------------------------------------------------------------


def _is_zero(atom) -> bool:
    return atom is None or isinstance(atom, Zero)


def _prox_conj(h, v, s):
    """prox_{s h*}(v): atom's prox_conj if present, else Moreau identity
    prox_{s h*}(v) = v - s * prox_{h/s}(v/s)  [LSCOMO SS2.5]."""
    if hasattr(h, "prox_conj"):
        return h.prox_conj(v, s)
    return v - s * h.prox(v / s, 1.0 / s)


def _gap_kind(problem: CompositeProblem) -> str:
    """'dual_gap' when the rigorous TV-form duality gap applies (quadratic
    data fidelity + lam||.||_1 through a linear map); 'lp_gap' for the
    standard-form LP composite (linear cost + nonneg orthant + affine
    equality through L, atlas.bench.lp_cell), certified by the PDLP-style
    lp_rel_gap; else 'fp_res'."""
    if (
        isinstance(problem.f, QuadraticDataFidelity)
        and isinstance(problem.h, ScaledL1)
        and problem.L is not None
        and _is_zero(problem.g)
    ):
        return "dual_gap"
    if (
        problem.name == "lp_standard"
        and problem.L is not None
        and hasattr(problem.f, "c")
        and hasattr(problem.h, "b")
    ):
        return "lp_gap"
    if (
        problem.name == "qp_standard"
        and problem.L is not None
        and hasattr(problem.f, "P")
        and hasattr(problem.h, "lo") and hasattr(problem.h, "hi")
    ):
        return "qp_kkt"
    if problem.name == "logreg_enet":
        return "logreg_gap"
    return "fp_res"


def _grad_lipschitz(f) -> float:
    """Lipschitz constant of grad f. L_f = 0 is VALID (affine f: grad
    constant, e.g. the LP linear cost); None/negative/non-finite is a
    construction failure."""
    Lf = getattr(getattr(f, "props", None), "lipschitz", None)
    if Lf is None or not np.isfinite(Lf) or Lf < 0:
        raise TypeCheckError(
            f"scheme needs a smooth f with a finite Lipschitz constant; got "
            f"{type(f).__name__} with lipschitz={Lf!r}"
        )
    return float(Lf)


def _strong_convexity(f) -> float:
    mu = getattr(getattr(f, "props", None), "strong_convexity", 0.0) or 0.0
    return float(mu)


def _dual_shape_zeros(problem: CompositeProblem):
    return jnp.zeros_like(problem.L.forward(jnp.zeros(problem.shape, dtype=jnp.float64)))


def _fp_metric(T):
    """Normalized fixed-point residual metric (no gap certificate)."""

    def metric(state, dyn):
        Tx = T(state, jnp.asarray(0), dyn)
        num = fp_residual(Tx, state)
        den = 1.0 + jnp.sqrt(
            sum(jnp.sum(l * l) for l in jax.tree_util.tree_leaves(state))
        )
        return num / den

    return metric


@dataclass(frozen=True)
class AffineTiltedProx:
    """The combined primal block  phi = affine + p  when `affine` is an
    affine atom (grad constant, L = 0, prox = translation).

    Exact prox identity (linear-tilt property):  for any convex p and
    affine q(x) = <c, x> + d,
        prox_{t(p + q)}(v) = prox_{t p}(v - t c) = p.prox(q.prox(v, t), t).
    This is what lets PDHG take the LP primal block  <c, x> + I_{x>=0}
    as ONE prox-friendly atom [LSCOMO SS2.5 prox calculus]."""

    affine: Any  # atom with .prox(v, t) = v - t*c (checked by the builder)
    p: Any  # any prox-friendly atom

    def prox(self, v, step):
        return self.p.prox(self.affine.prox(v, step), step)


jax.tree_util.register_dataclass(
    AffineTiltedProx, data_fields=["affine", "p"], meta_fields=[]
)


def _lp_arrays_dyn(problem: CompositeProblem) -> dict:
    """The (c, b) arrays the LP gap certificate needs, as dyn entries."""
    return {
        "lp_c": jnp.asarray(problem.data["c"], dtype=jnp.float64),
        "lp_b": jnp.asarray(problem.data["b"], dtype=jnp.float64),
    }


# ---------------------------------------------------------------------------
# BuiltScheme: the compiled-iteration seam shared by base solves, avg
# blends (atlas.schemes.compile) and the switch runtime.
# ---------------------------------------------------------------------------


@dataclass
class BuiltScheme:
    scheme: str
    T: Callable  # (state, k, dyn) -> state
    metric: Callable  # (state, dyn) -> scalar
    extract: Callable  # (state, dyn) -> (x, u or None)
    state0: Any
    dyn: dict  # dynamic inputs (atoms + stepsizes); runner adds tol/rho/...
    static_key: tuple
    cert: Optional[ConstructionCert]
    gap_kind: str
    # Optional admitted implementation of (1-rho)*state + rho*T(state).
    # The raw map T remains available for reference checks and certificates.
    relaxed_T: Optional[Callable] = None
    # (p, q, dyn) -> (<p,p>, <p,q>, <q,q>) in the metric in which the cert's
    # averagedness holds; None = Euclidean. Needed by outer steps that take
    # inner products (Haugazeau), never by the plain iteration.
    gram: Optional[Callable] = None


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _build_fbs(problem: CompositeProblem, params: FBSParams, check: bool) -> BuiltScheme:
    L = problem.L
    if problem.has_linear_map:
        # DUAL path: min_u f*(-L^T u) + h*(u), L_dual = ||L||^2 / mu.
        if not _is_zero(problem.g):
            raise TypeCheckError(
                "fbs on an h(Lx) composite requires g = 0 (got a nonzero g); "
                "use condat_vu"
            )
        f, h = problem.f, problem.h
        mu = _strong_convexity(f)
        if not hasattr(f, "grad_conj") or mu <= 0:
            raise TypeCheckError(
                "fbs dual path needs f strongly convex with grad_conj; got "
                f"{type(f).__name__} (mu={mu})"
            )
        L_dual = float(L.norm_bound) ** 2 / mu
        cert = typecheck_fbs(L_smooth=L_dual, step=params.step) if check else None
        zeros_p = jnp.zeros(problem.shape, dtype=jnp.float64)
        Dc = L.forward(f.grad_conj(zeros_p))  # = L @ argmin f  (Dy for TV)

        def T(u, k, dyn):
            # grad of F(u) = f*(-L^T u): grad f* is affine with linear part
            # (1/mu), so grad F(u) = L L^T u / mu - L grad f*(0). At mu = 1
            # (the only strong-convexity constant any grad_conj atom carries
            # today) this is bitwise the previous (L L^T u - Dc)/mu; the
            # previous form mis-scaled the constant term for mu != 1
            # (latent-generality fix found by tests/test_compiler_equivalence).
            grad = L.forward(L.adjoint(u)) / mu - dyn["Dc"]
            return _prox_conj(dyn["h"], u - dyn["a"] * grad, dyn["a"])

        gap_kind = _gap_kind(problem)
        if gap_kind == "dual_gap":

            def extract(u, dyn):
                lam = dyn["h"].lam
                uf = jnp.clip(u, -lam, lam)
                return dyn["f"].y - L.adjoint(uf), uf

            def metric(u, dyn):
                lam = dyn["h"].lam
                x = dyn["f"].y - L.adjoint(jnp.clip(u, -lam, lam))
                return rel_gap(x, u, dyn["f"].y, lam, L)

        else:

            def extract(u, dyn):
                return dyn["f"].grad_conj(-L.adjoint(u)), u

            metric = None  # filled with _fp_metric(T) below

        return BuiltScheme(
            scheme="fbs",
            T=T,
            metric=metric if metric is not None else _fp_metric(T),
            extract=extract,
            state0=_dual_shape_zeros(problem),
            dyn={"f": f, "h": h, "a": params.step, "Dc": Dc},
            static_key=("fbs", "dual", L, mu, gap_kind),
            cert=cert,
            gap_kind=gap_kind,
        )

    # PRIMAL path: f smooth + one prox block, no L.
    f = problem.f
    Lf = _grad_lipschitz(f)
    blocks = [a for a in (problem.g, problem.h) if not _is_zero(a)]
    if len(blocks) > 1:
        raise TypeCheckError(
            "fbs handles ONE prox block; this problem has two (g and h): use dys"
        )
    p = blocks[0] if blocks else Zero()
    cert = typecheck_fbs(L_smooth=Lf, step=params.step) if check else None

    def T(x, k, dyn):
        return dyn["p"].prox(x - dyn["a"] * dyn["f"].grad(x), dyn["a"])

    def extract(x, dyn):
        return x, None

    gap_kind = _gap_kind(problem)
    if gap_kind == "logreg_gap":
        def primal_metric(x, dyn):
            return logreg_enet_rel_gap(x, dyn["f"], dyn["p"])
    else:
        primal_metric = _fp_metric(T)

    x0 = (
        f.grad_conj(jnp.zeros(problem.shape, dtype=jnp.float64))
        if hasattr(f, "grad_conj")
        else jnp.zeros(problem.shape, dtype=jnp.float64)
    )
    return BuiltScheme(
        scheme="fbs",
        T=T,
        metric=primal_metric,
        extract=extract,
        state0=x0,
        dyn={"f": f, "p": p, "a": params.step},
        static_key=("fbs", "primal", Lf, gap_kind),
        cert=cert,
        gap_kind=gap_kind,
    )


def _build_drs(problem: CompositeProblem, params: DRSParams, check: bool) -> BuiltScheme:
    L = problem.L
    if problem.has_linear_map:
        # DUAL path: A = subdiff h*, B = grad of f*(-L^T .); J_aB is the
        # linear solve (I + (a/mu) L L^T) w = v + a L grad_f*(0) by CG,
        # exact because grad f* is affine (quadratic data fidelity).
        if not _is_zero(problem.g):
            raise TypeCheckError(
                "drs on an h(Lx) composite requires g = 0; use condat_vu"
            )
        f, h = problem.f, problem.h
        mu = _strong_convexity(f)
        if not isinstance(f, QuadraticDataFidelity):
            raise TypeCheckError(
                "drs dual path needs an affine grad_conj (quadratic data "
                f"fidelity); got {type(f).__name__}"
            )
        cert = typecheck_drs(step=params.step) if check else None
        zeros_p = jnp.zeros(problem.shape, dtype=jnp.float64)
        Dc = L.forward(f.grad_conj(zeros_p))

        def T(z, k, dyn):
            a = dyn["a"]
            u = _prox_conj(dyn["h"], z, a)  # J_aA (clip for the lam-ball)

            def matvec(w):
                return w + (a / mu) * L.forward(L.adjoint(w))

            v = 2.0 * u - z
            rhs = v + a * dyn["Dc"]
            w, _ = jax.scipy.sparse.linalg.cg(
                matvec, rhs, x0=v, tol=params.cg_tol, maxiter=params.cg_maxiter
            )
            return z + w - u

        gap_kind = _gap_kind(problem)
        if gap_kind == "dual_gap":

            def extract(z, dyn):
                lam = dyn["h"].lam
                u = jnp.clip(z, -lam, lam)
                return dyn["f"].y - L.adjoint(u), u

            def metric(z, dyn):
                lam = dyn["h"].lam
                u = jnp.clip(z, -lam, lam)
                x = dyn["f"].y - L.adjoint(u)
                return rel_gap(x, u, dyn["f"].y, lam, L)

        else:

            def extract(z, dyn):
                u = _prox_conj(dyn["h"], z, dyn["a"])
                return dyn["f"].grad_conj(-L.adjoint(u)), u

            metric = None

        built_T = T
        return BuiltScheme(
            scheme="drs",
            T=built_T,
            metric=metric if metric is not None else _fp_metric(built_T),
            extract=extract,
            state0=_dual_shape_zeros(problem),
            dyn={"f": f, "h": h, "a": params.step, "Dc": Dc},
            static_key=(
                "drs", "dual", L, mu, gap_kind, params.cg_tol, params.cg_maxiter
            ),
            cert=cert,
            gap_kind=gap_kind,
        )

    # PRIMAL two-prox path: f absent, g + h prox blocks.
    if not _is_zero(problem.f):
        raise TypeCheckError(
            "drs without a linear map requires f = 0 (two prox blocks); "
            "f smooth + two proxes is dys"
        )
    g, h = problem.g, problem.h
    if _is_zero(g) or _is_zero(h):
        raise TypeCheckError("drs needs two prox blocks (g and h)")
    cert = typecheck_drs(step=params.step) if check else None

    def T(z, k, dyn):
        a = dyn["a"]
        xg = dyn["g"].prox(z, a)
        xh = dyn["h"].prox(2.0 * xg - z, a)
        return z + xh - xg

    def extract(z, dyn):
        return dyn["g"].prox(z, dyn["a"]), None

    return BuiltScheme(
        scheme="drs",
        T=T,
        metric=_fp_metric(T),
        extract=extract,
        state0=jnp.zeros(problem.shape, dtype=jnp.float64),
        dyn={"g": g, "h": h, "a": params.step},
        static_key=("drs", "primal", "fp_res"),
        cert=cert,
        gap_kind="fp_res",
    )


def _saddle_gram(L):
    """Gram entries in the PDHG metric M = [[I/t, -L*], [-L, I/s]], in which
    the pdhg iteration x+ = prox(x - t L*u), u+ = prox*(u + s L(2x+ - x)) is a
    resolvent ((1/2)-averaged) and condat_vu is (1/delta)-averaged:
    <p, q>_M = <px,qx>/t + <pu,qu>/s - <L px, qu> - <L qx, pu>. Two matvecs."""

    def gram(p, q, dyn):
        (px, pu), (qx, qu) = p, q
        t, s = dyn["t"], dyn["s"]
        Lp, Lq = L.forward(px), L.forward(qx)
        pp = jnp.vdot(px, px) / t + jnp.vdot(pu, pu) / s - 2.0 * jnp.vdot(Lp, pu)
        qq = jnp.vdot(qx, qx) / t + jnp.vdot(qu, qu) / s - 2.0 * jnp.vdot(Lq, qu)
        pq = (jnp.vdot(px, qx) / t + jnp.vdot(pu, qu) / s
              - jnp.vdot(Lp, qu) - jnp.vdot(Lq, pu))
        return pp, pq, qq

    return gram


def _build_pdhg(problem: CompositeProblem, params: PDHGParams, check: bool) -> BuiltScheme:
    if not problem.has_linear_map:
        raise TypeCheckError("pdhg requires an h(Lx) block (L and h)")
    L, h = problem.L, problem.h
    if _is_zero(problem.g):
        phi = problem.f
    elif _is_zero(problem.f):
        phi = problem.g
    elif (
        getattr(getattr(problem.f, "props", None), "lipschitz", None) == 0.0
        and hasattr(problem.f, "prox")
        and hasattr(problem.g, "prox")
    ):
        # f affine (grad constant): the linear-tilt prox identity combines
        # f + g into one exact prox block (AffineTiltedProx docstring); this
        # is the LP primal block <c,x> + I_{x>=0}.
        phi = AffineTiltedProx(affine=problem.f, p=problem.g)
    else:
        raise TypeCheckError(
            "pdhg handles ONE primal block (f XOR g, or f affine + g prox "
            "via the tilt identity); both non-affine blocks present: use "
            "condat_vu"
        )
    if phi is None or not hasattr(phi, "prox"):
        raise TypeCheckError("pdhg needs a prox-friendly primal block")
    cert = (
        typecheck_pdhg(t=params.t, s=params.s, opnorm=float(L.norm_bound))
        if check
        else None
    )

    def T(state, k, dyn):
        x, u = state
        t, s = dyn["t"], dyn["s"]
        x_new = dyn["phi"].prox(x - t * L.adjoint(u), t)
        u_new = _prox_conj(dyn["h"], u + s * L.forward(2.0 * x_new - x), s)
        return (x_new, u_new)

    gap_kind = _gap_kind(problem)
    dyn = {"phi": phi, "h": h, "t": params.t, "s": params.s}
    if gap_kind == "dual_gap":

        def metric(state, dyn):
            x, u = state
            return rel_gap(x, u, dyn["phi"].y, dyn["h"].lam, L)

    elif gap_kind == "lp_gap":
        dyn.update(_lp_arrays_dyn(problem))

        def metric(state, dyn):
            x, u = state
            return lp_rel_gap(x, u, dyn["lp_c"], L, dyn["lp_b"])

    else:
        metric = None

    def extract(state, dyn):
        return state[0], state[1]

    x0 = (
        phi.grad_conj(jnp.zeros(problem.shape, dtype=jnp.float64))
        if hasattr(phi, "grad_conj")
        else jnp.zeros(problem.shape, dtype=jnp.float64)
    )
    built = BuiltScheme(
        scheme="pdhg",
        T=T,
        metric=metric,  # may be None, fixed below
        extract=extract,
        state0=(x0, _dual_shape_zeros(problem)),
        dyn=dyn,
        static_key=("pdhg", L, gap_kind),
        cert=cert,
        gap_kind=gap_kind,
    )
    if built.metric is None:
        built.metric = _fp_metric(T)
    built.gram = _saddle_gram(L)
    return built


def _build_condat_vu(
    problem: CompositeProblem, params: CondatVuParams, check: bool
) -> BuiltScheme:
    if not problem.has_linear_map:
        raise TypeCheckError("condat_vu requires an h(Lx) block (L and h)")
    L, h = problem.L, problem.h
    f = problem.f
    Lf = _grad_lipschitz(f)
    g = problem.g if problem.g is not None else Zero()
    cert = (
        typecheck_condat_vu(
            t=params.t, s=params.s, opnorm=float(L.norm_bound), L_smooth=Lf
        )
        if check
        else None
    )

    def T(state, k, dyn):
        x, u = state
        t, s = dyn["t"], dyn["s"]
        x_new = dyn["g"].prox(x - t * dyn["f"].grad(x) - t * L.adjoint(u), t)
        u_new = _prox_conj(dyn["h"], u + s * L.forward(2.0 * x_new - x), s)
        return (x_new, u_new)

    gap_kind = _gap_kind(problem)
    dyn = {"f": f, "g": g, "h": h, "t": params.t, "s": params.s}
    if gap_kind == "dual_gap":

        def metric(state, dyn):
            x, u = state
            return rel_gap(x, u, dyn["f"].y, dyn["h"].lam, L)

    elif gap_kind == "lp_gap":
        dyn.update(_lp_arrays_dyn(problem))

        def metric(state, dyn):
            x, u = state
            return lp_rel_gap(x, u, dyn["lp_c"], L, dyn["lp_b"])

    elif gap_kind == "qp_kkt":

        def metric(state, dyn):
            x, u = state
            return qp_rel_kkt(x, u, dyn["f"].P, dyn["f"].q, L, dyn["h"].lo, dyn["h"].hi)

    else:
        metric = None

    def extract(state, dyn):
        return state[0], state[1]

    x0 = (
        f.grad_conj(jnp.zeros(problem.shape, dtype=jnp.float64))
        if hasattr(f, "grad_conj")
        else jnp.zeros(problem.shape, dtype=jnp.float64)
    )
    built = BuiltScheme(
        scheme="condat_vu",
        T=T,
        metric=metric,
        extract=extract,
        state0=(x0, _dual_shape_zeros(problem)),
        dyn=dyn,
        static_key=("condat_vu", L, Lf, gap_kind),
        cert=cert,
        gap_kind=gap_kind,
    )
    if built.metric is None:
        built.metric = _fp_metric(T)
    built.gram = _saddle_gram(L)
    return built


def _build_dys(problem: CompositeProblem, params: DYSParams, check: bool) -> BuiltScheme:
    if problem.has_linear_map:
        raise TypeCheckError(
            "dys is the three-operator splitting for f + g + h WITHOUT a "
            "linear map; this problem has h(Lx) (prox of h∘L unavailable)"
        )
    f = problem.f
    Lf = _grad_lipschitz(f)
    g = problem.g if problem.g is not None else Zero()
    h = problem.h if problem.h is not None else Zero()
    cert = typecheck_dys(L_smooth=Lf, step=params.step) if check else None

    def T(z, k, dyn):
        a = dyn["a"]
        xg = dyn["g"].prox(z, a)
        xh = dyn["h"].prox(2.0 * xg - z - a * dyn["f"].grad(xg), a)
        return z + xh - xg

    def extract(z, dyn):
        return dyn["g"].prox(z, dyn["a"]), None

    return BuiltScheme(
        scheme="dys",
        T=T,
        metric=_fp_metric(T),
        extract=extract,
        state0=jnp.zeros(problem.shape, dtype=jnp.float64),
        dyn={"f": f, "g": g, "h": h, "a": params.step},
        static_key=("dys", Lf, "fp_res"),
        cert=cert,
        gap_kind="fp_res",
    )



def _build_pdhg_entropic(problem: CompositeProblem, params: PDHGParams, check: bool) -> BuiltScheme:
    """PDHG with an entropic primal step on the mass simplex the constraints
    imply (see atlas.typing_.rules.typecheck_pdhg_entropic). Standard-form LP
    composites only, and only those that declare data["simplex_mass"] (the
    transport families do: the marginals fix the total mass)."""
    from atlas.typing_.rules import typecheck_pdhg_entropic

    if problem.name != "lp_standard":
        raise TypeCheckError("pdhg_entropic needs the standard-form LP composite")
    mass = problem.data.get("simplex_mass")
    if mass is None:
        raise TypeCheckError(
            "pdhg_entropic REJECTED: negative entropy is not strongly convex on the "
            "unbounded orthant, so no Bregman step condition exists unless the "
            "constraints imply a simplex {x >= 0, sum x = mass}; this composite "
            "declares none (data['simplex_mass'])")
    L = problem.L
    meta = problem.data.get("meta", {})
    tail, head = getattr(L, "tail", None), getattr(L, "head", None)
    if meta.get("opnorm_1to2") is not None:
        n12 = float(meta["opnorm_1to2"])
    elif tail is not None and not bool(np.any(np.asarray(tail) == np.asarray(head))):
        n12 = float(np.sqrt(2.0))  # every incidence column is e_tail - e_head
    else:
        raise TypeCheckError("pdhg_entropic needs ||L||_{1->2} (incidence operator or meta['opnorm_1to2'])")
    b = np.asarray(problem.data["b"], dtype=np.float64)
    m, n = meta.get("m"), meta.get("n")
    if m is not None and abs(b[: int(m)].sum() - float(mass)) > 1e-9 * max(1.0, float(mass)):
        raise TypeCheckError("declared simplex_mass does not match the source marginals")
    cert = typecheck_pdhg_entropic(t=params.t, s=params.s, opnorm_1to2=n12, mass=float(mass)) if check else None
    # DUAL RECOVERY. The simplex constraint is redundant: sum(x) = <L^T w_S, x>
    # for a row combination w_S (the source indicator for transport). Its
    # multiplier therefore never enters u (both marginal sums of Lx - b stay
    # 0, so u's component along w_S is frozen) and at convergence the reduced
    # costs c + L^T u equal a constant on the support instead of 0. The LP dual
    # is recovered as u - min(c + L^T u) w_S, which makes the reduced costs
    # exactly >= 0; any u is a valid input to the certificate, so this changes
    # which dual point is certified, never what the certificate means.
    # (Without it the gap stays at 0.93 on a 6x5 instance for 1e5 iterations.)
    w_S = problem.data.get("simplex_dual_direction")
    if w_S is None and m is not None and n is not None:
        w_S = np.concatenate([np.ones(int(m)), np.zeros(int(n))])
    if w_S is None:
        raise TypeCheckError("pdhg_entropic needs the row combination w_S with L^T w_S = 1")
    w_S = jnp.asarray(w_S, dtype=jnp.float64)
    if float(jnp.max(jnp.abs(L.adjoint(w_S) - 1.0))) > 1e-12:
        raise TypeCheckError("data['simplex_dual_direction'] does not satisfy L^T w_S = 1")

    def T(state, k, dyn):
        x, u = state
        t, s, mass_ = dyn["t"], dyn["s"], dyn["mass"]
        w = jnp.log(jnp.maximum(x, 1e-300)) - (t / mass_) * (dyn["lp_c"] + L.adjoint(u))
        x_new = mass_ * jnp.exp(w - jax.scipy.special.logsumexp(w))
        u_new = u + s * (L.forward(2.0 * x_new - x) - dyn["lp_b"])
        return (x_new, u_new)

    dyn = {"t": params.t, "s": params.s, "mass": float(mass), "w_S": w_S}
    dyn.update(_lp_arrays_dyn(problem))

    def _recover(u, dyn):
        return u - jnp.min(dyn["lp_c"] + L.adjoint(u)) * dyn["w_S"]

    def metric(state, dyn):
        x, u = state
        return lp_rel_gap(x, _recover(u, dyn), dyn["lp_c"], L, dyn["lp_b"])

    def extract(state, dyn):
        return state[0], _recover(state[1], dyn)

    if m is not None and n is not None and int(m) * int(n) == int(problem.shape[0]):
        a_, bt = b[: int(m)], -b[int(m):]
        x0 = jnp.asarray(np.outer(a_, bt).ravel() / float(mass))  # independent coupling: feasible
    else:
        x0 = jnp.full(problem.shape, float(mass) / float(np.prod(problem.shape)))
    return BuiltScheme(
        scheme="pdhg_entropic", T=T, metric=metric, extract=extract,
        state0=(x0, _dual_shape_zeros(problem)), dyn=dyn,
        static_key=("pdhg_entropic", L, "lp_gap"), cert=cert, gap_kind="lp_gap",
    )


def _build_condat_vu_metric(problem: CompositeProblem, params: CondatVuMetricParams, check: bool) -> BuiltScheme:
    """Condat-Vu with a matrix primal step T = params.metric, for g = 0 and a
    quadratic f with an explicit Hessian operator (f.P): the QP cell. The gate
    (typecheck_condat_vu_metric) is checked against power-iteration estimates
    of lambda_max(T P) and ||A T A^T||."""
    from atlas.operators.primal_metric import gate_quantities
    from atlas.typing_.rules import typecheck_condat_vu_metric

    if not problem.has_linear_map:
        raise TypeCheckError("condat_vu_metric requires an h(Lx) block")
    f, L, h = problem.f, problem.L, problem.h
    P = getattr(f, "P", None)
    if P is None:
        raise TypeCheckError("condat_vu_metric needs a quadratic f with an explicit Hessian operator")
    g_zero = problem.g is None or _is_zero(problem.g)
    n = int(problem.shape[0])
    m_rows = int(_dual_shape_zeros(problem).shape[0])
    cert = None
    if check:
        lamP, nAT = gate_quantities(params.metric, P, L, n, m_rows)
        cert = typecheck_condat_vu_metric(float(params.s), lamP, nAT, g_zero)
    elif not g_zero:
        raise TypeCheckError("condat_vu_metric requires g = 0")

    def T(state, k, dyn):
        x, u = state
        x_new = x - dyn["Tm"].apply(dyn["f"].grad(x) + L.adjoint(u))
        u_new = _prox_conj(dyn["h"], u + dyn["s"] * L.forward(2.0 * x_new - x), dyn["s"])
        return (x_new, u_new)

    gap_kind = _gap_kind(problem)
    dyn = {"f": f, "h": h, "Tm": params.metric, "s": params.s}
    if gap_kind == "qp_kkt":
        def metric(state, dyn):
            x, u = state
            return qp_rel_kkt(x, u, dyn["f"].P, dyn["f"].q, L, dyn["h"].lo, dyn["h"].hi)
    else:
        metric = _fp_metric(T)

    def extract(state, dyn):
        return state[0], state[1]

    return BuiltScheme(
        scheme="condat_vu_metric", T=T, metric=metric, extract=extract,
        state0=(jnp.zeros(problem.shape, dtype=jnp.float64), _dual_shape_zeros(problem)), dyn=dyn,
        static_key=("condat_vu_metric", L, P, gap_kind, int(params.metric.S.shape[0])),
        cert=cert, gap_kind=gap_kind,
    )


def _build_fista(problem: CompositeProblem, params: FISTAParams, check: bool) -> BuiltScheme:
    """FISTA with optional gradient restart on the primal f + (one prox block),
    no linear map. State (x, y, t); the certificate is evaluated on x."""
    from atlas.typing_.rules import typecheck_fista

    if problem.has_linear_map:
        raise TypeCheckError("fista handles f + g without a linear map")
    f = problem.f
    Lf = _grad_lipschitz(f)
    blocks = [a for a in (problem.g, problem.h) if not _is_zero(a)]
    if len(blocks) > 1:
        raise TypeCheckError("fista handles ONE prox block")
    p = blocks[0] if blocks else Zero()
    cert = typecheck_fista(Lf, float(params.step), bool(params.restart)) if check else None
    restart = bool(params.restart)

    def T(state, k, dyn):
        x, yv, t = state
        a = dyn["a"]
        x_new = dyn["p"].prox(yv - a * dyn["f"].grad(yv), a)
        t_new = 0.5 * (1.0 + jnp.sqrt(1.0 + 4.0 * t * t))
        y_new = x_new + ((t - 1.0) / t_new) * (x_new - x)
        if restart:
            bad = jnp.vdot(yv - x_new, x_new - x) > 0.0
            t_new = jnp.where(bad, jnp.ones_like(t_new), t_new)
            y_new = jnp.where(bad, x_new, y_new)
        return (x_new, y_new, t_new)

    gap_kind = _gap_kind(problem)
    if gap_kind == "logreg_gap":
        def metric(state, dyn):
            return logreg_enet_rel_gap(state[0], dyn["f"], dyn["p"])
    else:
        def metric(state, dyn):
            x = state[0]
            xn = dyn["p"].prox(x - dyn["a"] * dyn["f"].grad(x), dyn["a"])
            return jnp.linalg.norm(xn - x) / jnp.maximum(jnp.linalg.norm(x), 1.0)

    def extract(state, dyn):
        return state[0], None

    x0 = jnp.zeros(problem.shape, dtype=jnp.float64)
    return BuiltScheme(
        scheme="fista", T=T, metric=metric, extract=extract,
        state0=(x0, x0, jnp.asarray(1.0, dtype=jnp.float64)),
        dyn={"f": f, "p": p, "a": params.step},
        static_key=("fista", Lf, gap_kind, restart), cert=cert, gap_kind=gap_kind,
    )

_BUILDERS = {
    "fbs": (_build_fbs, FBSParams),
    "drs": (_build_drs, DRSParams),
    "pdhg": (_build_pdhg, PDHGParams),
    "condat_vu": (_build_condat_vu, CondatVuParams),
    "dys": (_build_dys, DYSParams),
    "pdhg_entropic": (_build_pdhg_entropic, PDHGParams),
    "condat_vu_metric": (_build_condat_vu_metric, CondatVuMetricParams),
    "fista": (_build_fista, FISTAParams),
}


def build_scheme(problem: CompositeProblem, scheme: str, params, check: bool = True) -> BuiltScheme:
    """Build the (T, metric, state0, dyn, cert) bundle for one base scheme.

    check=False skips the Level-1 typecheck (ONLY for the aggressive branch
    of the safeguarded switch wrapper; everything else must certify)."""
    if scheme not in _BUILDERS:
        raise TypeCheckError(f"unknown scheme {scheme!r} (have {sorted(_BUILDERS)})")
    builder, _ = _BUILDERS[scheme]
    return builder(problem, params, check)


# ---------------------------------------------------------------------------
# Wrapper cert threading + the shared runner
# ---------------------------------------------------------------------------


def _wrap_cert(cert: ConstructionCert, rho, halpern, restart_k=None,
               outer=None) -> ConstructionCert:
    # rho + halpern is ADMITTED as of 2026-09-05: relax THEN anchor. The
    # anchor only needs a nonexpansive map, so the relaxation is checked in
    # its wider regime (rho in (0,2], rho*alpha <= 1), which admits the
    # reflection rho = 2. See typecheck_relaxation for the theorem.
    if restart_k is not None and not halpern:
        raise TypeCheckError(
            "restart_k requires the Halpern anchor (wrapper.halpern=true): "
            "only the anchor carries restartable state in this build"
        )
    if rho is not None:
        cert = typecheck_relaxation(cert, rho, with_anchor=bool(halpern))
    if halpern:
        cert = typecheck_halpern(cert)
    if restart_k is not None:
        cert = typecheck_restart(cert, restart_k)
    if outer is not None:
        # A registered outer wrapper is an outer step of its own: it replaces
        # the relaxation and the anchor rather than composing with them.
        if rho is not None or halpern or restart_k is not None:
            raise TypeCheckError(
                f"wrapper.kind={outer!r} does not combine with rho/halpern/restart")
        from atlas.wrappers.registry import gate as _outer_gate
        cert = _outer_gate(outer, cert, cert.scheme.split("+")[0])
    return cert


def _cast_tree(tree, dtype):
    """Cast every floating leaf of a pytree to dtype (ints/indices kept)."""

    def _cast(a):
        a = jnp.asarray(a)
        return a.astype(dtype) if jnp.issubdtype(a.dtype, jnp.floating) else a

    return jax.tree_util.tree_map(_cast, tree)


def _run_built_policy(built: BuiltScheme, cert, params, budget: Budget,
                      backend) -> SolveResult:
    """Execute a BuiltScheme under an atlas.backend precision policy.

    The wrapper genes (rho relaxation / Halpern anchor + restart) combine
    INSIDE step_factory: they are part of WHAT T is; the policy only decides
    HOW it is computed (iterate dtype, certificate cadence). Termination is
    decided ONLY by the f64 certified gap (atlas.backend.precision).
    TIMING CONTRACT (fixed 2026-08-26, sprint finding): the chunk runner is
    AOT-compiled BEFORE the exec timer and cached across solves keyed by
    (built.static_key, wrapper constants, K, shapes/dtypes), exactly like
    the km path's _EXEC_CACHE; problem data and stepsizes flow through the
    DYNAMIC dyn argument. The reported wall_time EXCLUDES compile
    (compile seconds recorded separately in extra)."""
    from atlas.backend.precision import get_policy

    if backend.precision == "schedule":
        theta = backend.theta
        if theta is None or theta <= budget.tol:
            raise TypeCheckError(
                f"backend.theta must exceed the budget tolerance "
                f"({budget.tol:g}) or the f32 phase of the schedule "
                f"degenerates; got theta={theta!r}"
            )
        policy = get_policy("schedule", K=int(backend.K), theta=float(theta))
    else:
        policy = get_policy(backend.precision, K=int(backend.K))

    dyn64 = jax.tree_util.tree_map(jnp.asarray, built.dyn)
    halpern = bool(params.halpern)
    rho = getattr(params, "rho", None)
    rk = getattr(params, "restart_k", None)

    from atlas.backend.precision import promote64
    from atlas.wrappers.km import _abstract, _compile_cached

    def dyn_factory(dtype):
        return _cast_tree(built.dyn, dtype)

    outer = getattr(params, "outer", None)
    augmented = halpern or outer == "haugazeau"
    if outer == "haugazeau":
        from atlas.wrappers.km import euclidean_gram, haugazeau_coefficients
        if cert is None or "haugazeau_alpha" not in cert.params:
            raise TypeCheckError("haugazeau needs its gate certificate (the averagedness)")
        alpha_v = float(cert.params["haugazeau_alpha"])
        w_v = 0.5 / alpha_v
        gram = built.gram if built.gram is not None else euclidean_gram
        static_key = ("policy", built.static_key, "haugazeau", alpha_v)

        def step_factory(dtype):
            def step(aug, k, dynd):
                z, anchor = aug
                Tz = built.T(z, k, dynd)
                Fz = jax.tree_util.tree_map(lambda s, t: s + w_v * (t - s), z, Tz)
                p = jax.tree_util.tree_map(jnp.subtract, anchor, z)
                q = jax.tree_util.tree_map(jnp.subtract, z, Fz)
                ca, cb, cc, _ = haugazeau_coefficients(*gram(p, q, dynd))
                new = jax.tree_util.tree_map(
                    lambda a, b, c: ca * a + cb * b + cc * c, anchor, z, Fz)
                return (new, anchor)

            return step

        def init_factory(dtype):
            s0 = _cast_tree(built.state0, dtype)
            return (s0, s0)

        def gap64(aug):
            return built.metric(aug[0], dyn64)

    elif not halpern:
        rho_v = 1.0 if rho is None else float(rho)
        static_key = ("policy", built.static_key, "km", rho_v)

        def step_factory(dtype):
            def step(state, k, dynd):
                if built.relaxed_T is not None:
                    return built.relaxed_T(state, k, dynd, rho_v)
                Tx = built.T(state, k, dynd)
                if rho_v == 1.0:
                    return Tx
                return jax.tree_util.tree_map(
                    lambda s, t: (1.0 - rho_v) * s + rho_v * t, state, Tx
                )

            return step

        def init_factory(dtype):
            return _cast_tree(built.state0, dtype)

        def gap64(state):
            return built.metric(state, dyn64)

    else:
        rk_v = int(rk) if rk else budget.max_iters + 2
        rho_v = 1.0 if rho is None else float(rho)
        static_key = ("policy", built.static_key, "halpern", rk_v, rho_v)

        def step_factory(dtype):
            def step(aug, k, dynd):
                z, anchor, j = aug
                if built.relaxed_T is not None:
                    Fx = built.relaxed_T(z, k, dynd, rho_v)
                else:
                    Tx = built.T(z, k, dynd)
                    Fx = jax.tree_util.tree_map(
                        lambda s, t: (1.0 - rho_v) * s + rho_v * t, z, Tx)
                lam_j = 1.0 / (j + 2.0)
                new = jax.tree_util.tree_map(
                    lambda a, t: lam_j * a + (1.0 - lam_j) * t, anchor, Fx
                )
                j_next = j + 1.0
                do_restart = j_next >= rk_v
                anchor = jax.tree_util.tree_map(
                    lambda a, n: jnp.where(do_restart, n, a), anchor, new
                )
                j_next = jnp.where(do_restart, jnp.zeros_like(j_next), j_next)
                return (new, anchor, j_next)

            return step

        def init_factory(dtype):
            s0 = _cast_tree(built.state0, dtype)
            return (s0, s0, jnp.zeros((), dtype=dtype))

        def gap64(aug):
            return built.metric(aug[0], dyn64)

    # Compile the f64 certificate ONCE (cached across solves, before the
    # timer) so the per-K gap evaluations run compiled, matching the km
    # path's in-graph metric checks. The certificate math is untouched:
    # jit of the same f64 ops.
    state64_ex = promote64(init_factory(jnp.float64))

    def _metric64(s64, d64):
        return built.metric(s64[0] if augmented else s64, d64)

    gap_key = ("policy_gap", static_key, _abstract(state64_ex), _abstract(dyn64))
    gap_exe = _compile_cached(gap_key, _metric64, state64_ex, dyn64)

    def gap64(s64, _exe=gap_exe):  # noqa: F811
        return _exe(s64, dyn64)

    t0 = time.perf_counter()
    if backend.precision in ("f64", "f32_cert64"):
        from atlas.backend.precision import run_device_policy
        dtype = jnp.float64 if backend.precision == "f64" else jnp.float32
        pr = run_device_policy(
            step_factory(dtype), init_factory(dtype), dyn_factory(dtype),
            _metric64, dyn64, K=int(backend.K),
            max_iters=int(budget.max_iters), tol=float(budget.tol),
            static_key=static_key)
    else:
        pr = policy.run(
            step_factory, init_factory, gap64, float(budget.tol),
            int(budget.max_iters),
            dyn_factory=dyn_factory, static_key=static_key,
        )
    wall_total = time.perf_counter() - t0
    # wall_time contract: compiled execution only, matching the km path
    # (AOT compile is cached and excluded; see atlas.backend.precision).
    wall = float(pr.exec_time_s)
    final = pr.state[0] if augmented else pr.state
    x, u = built.extract(final, dyn64)
    gap_h = (
        np.asarray(pr.gap_history, dtype=np.float64)
        if len(pr.gap_history)
        else np.asarray([np.inf])
    )
    return SolveResult(
        x=np.asarray(x),
        u=None if u is None else np.asarray(u),
        iters=int(pr.iters),
        fp_residual_history=np.asarray([np.nan]),  # policy path certifies
        rel_gap_history=gap_h,  # by the f64 gap only (per K iterations)
        wall_time=wall,
        cert=cert,
        extra={
            "backend": {
                "precision": backend.precision,
                "kernel": getattr(backend, "kernel", "jax"),
                "kernel_evidence": getattr(built, "kernel_evidence", None),
                "K": int(backend.K),
                "theta": backend.theta,
                "batch": (
                    "declared, inactive" if getattr(backend, "batch", False)
                    else False
                ),
                "iterate_dtypes": list(pr.iterate_dtypes),
                "switch_iter": pr.switch_iter,
                "certificate_dtype": pr.certificate_dtype,
                "gap_iters": np.asarray(pr.gap_iters).tolist(),
                "certificate_checks": len(pr.gap_iters),
                "execution": ("device_loop" if backend.precision != "schedule"
                              else "host_precision_phases"),
            },
            "wall_includes_compile": False,
            "compile_time_s": float(pr.compile_time_s),
            "wall_total_incl_compile_s": float(wall_total),
        },
    )



def _resolve_device(name: str):
    """Map a placement gene to a jax device, refusing a silent fallback."""
    platform = {"gpu": "cuda", "cpu": "cpu"}[name]
    try:
        return jax.devices(platform)[0]
    except RuntimeError as e:
        raise RuntimeError(
            f"backend.device={name!r} but this process has no {platform} backend "
            f"(JAX_PLATFORMS={_os.environ.get('JAX_PLATFORMS')!r}); start it with "
            f"JAX_PLATFORMS=cuda,cpu") from e

def run_built(built: BuiltScheme, cert: Optional[ConstructionCert], params,
              budget: Budget, backend=None) -> SolveResult:
    """Run a BuiltScheme through the cached-kernel KM loop with the wrapper
    genes (rho / halpern / restart_k) as DYNAMIC arguments.

    backend: an optional genome Backend gene. None (absent block) is the
    exact P0 km_run path; a present block routes through the corresponding
    atlas.backend precision policy (_run_built_policy)."""
    if backend is not None:
        device = getattr(backend, "device", None)
        if device is None:
            return _run_built_policy(built, cert, params, budget, backend)
        dev = _resolve_device(device)
        # Commit the problem data and the initial state to the requested device
        # and make it the default for everything created during the solve. The
        # compile caches key on placement, so this cannot reuse a program
        # compiled for another device.
        with jax.default_device(dev):
            placed = dataclasses.replace(
                built, dyn=jax.device_put(built.dyn, dev),
                state0=jax.device_put(built.state0, dev))
            return _run_built_policy(placed, cert, params, budget, backend)
    outer = getattr(params, "outer", None)
    mode = "haugazeau" if outer == "haugazeau" else ("halpern" if params.halpern else "km")
    dyn = dict(built.dyn)
    dyn["tol"] = float(budget.tol)
    if mode == "haugazeau":
        if cert is None or "haugazeau_alpha" not in cert.params:
            raise TypeCheckError("haugazeau needs its gate certificate (the averagedness)")
        dyn["haug_alpha"] = float(cert.params["haugazeau_alpha"])
    # rho is needed in BOTH modes: "halpern" anchors onto the relaxed map
    # F_rho = (1-rho) I + rho T (relax-then-anchor). Omitting it here is what
    # made rho a silent no-op under the anchor.
    dyn["rho"] = 1.0 if params.rho is None else float(params.rho)
    if mode == "halpern":
        rk = params.restart_k if getattr(params, "restart_k", None) else None
        dyn["restart_k"] = int(rk) if rk else budget.max_iters + 2
    state, iters, res_h, gap_h, wall = km_run(
        built.T, built.state0, budget, built.metric, dyn,
        mode=mode, static_key=built.static_key, gram=built.gram,
    )
    x, u = built.extract(state, dyn)
    return SolveResult(
        x=np.asarray(x),
        u=None if u is None else np.asarray(u),
        iters=iters,
        fp_residual_history=res_h,
        rel_gap_history=gap_h,
        wall_time=wall,
        cert=cert,
    )


def _solve(problem: CompositeProblem, scheme: str, params, budget: Budget) -> SolveResult:
    built = build_scheme(problem, scheme, params, check=True)  # Level 1, mandatory
    cert = _wrap_cert(
        built.cert, params.rho, params.halpern, getattr(params, "restart_k", None)
    )
    return run_built(built, cert, params, budget)


# ---------------------------------------------------------------------------
# Public per-scheme API (P0-compatible signatures)
# ---------------------------------------------------------------------------


def typecheck_fbs_solve(problem: CompositeProblem, params: FBSParams) -> ConstructionCert:
    built = build_scheme(problem, "fbs", params, check=True)
    return _wrap_cert(built.cert, params.rho, params.halpern, params.restart_k)


def solve_fbs(problem: CompositeProblem, params: FBSParams, budget: Budget) -> SolveResult:
    """x+ = prox_{a g}(x - a grad f(x)); on h(Lx) composites: the dual
    projected gradient  u+ = prox_{a h*}(u - a grad f*(-L^T u) pullback),
    primal recovery x = grad f*(-L^T u)."""
    return _solve(problem, "fbs", params, budget)


def typecheck_drs_solve(problem: CompositeProblem, params: DRSParams) -> ConstructionCert:
    built = build_scheme(problem, "drs", params, check=True)
    return _wrap_cert(built.cert, params.rho, params.halpern, params.restart_k)


def solve_drs(problem: CompositeProblem, params: DRSParams, budget: Budget) -> SolveResult:
    """z+ = z + J_aB(2 J_aA(z) - z) - J_aA(z); any a > 0; (1/2)-averaged."""
    return _solve(problem, "drs", params, budget)


def typecheck_pdhg_solve(problem: CompositeProblem, params: PDHGParams) -> ConstructionCert:
    built = build_scheme(problem, "pdhg", params, check=True)
    return _wrap_cert(built.cert, params.rho, params.halpern, params.restart_k)


def solve_pdhg(problem: CompositeProblem, params: PDHGParams, budget: Budget) -> SolveResult:
    """x+ = prox_{t phi}(x - t L^T u); u+ = prox_{s h*}(u + s L(2x+ - x));
    side condition t*s*||L||^2 < 1 [LSCOMO SS3.3]."""
    return _solve(problem, "pdhg", params, budget)


def typecheck_condat_vu_solve(
    problem: CompositeProblem, params: CondatVuParams
) -> ConstructionCert:
    built = build_scheme(problem, "condat_vu", params, check=True)
    return _wrap_cert(built.cert, params.rho, params.halpern, params.restart_k)


def solve_condat_vu(
    problem: CompositeProblem, params: CondatVuParams, budget: Budget
) -> SolveResult:
    """x+ = prox_{t g}(x - t grad f(x) - t L^T u);
    u+ = prox_{s h*}(u + s L(2x+ - x));
    side condition t*L_f/2 + t*s*||L||^2 < 1 [LSCOMO SS3.3 eq. 3.13]."""
    return _solve(problem, "condat_vu", params, budget)


def typecheck_dys_solve(problem: CompositeProblem, params: DYSParams) -> ConstructionCert:
    built = build_scheme(problem, "dys", params, check=True)
    return _wrap_cert(built.cert, params.rho, params.halpern, params.restart_k)


def solve_dys(problem: CompositeProblem, params: DYSParams, budget: Budget) -> SolveResult:
    """Davis-Yin three-operator splitting for f smooth + g + h (no L):
    x_g = prox_{a g}(z); x_h = prox_{a h}(2 x_g - z - a grad f(x_g));
    z+ = z + x_h - x_g;  a in (0, 2/L_f), theta = 2/(4 - a*L_f)-averaged
    [LSCOMO SS2.7 + SS13.4.2 Thm 28; Davis & Yin 2017]."""
    return _solve(problem, "dys", params, budget)
