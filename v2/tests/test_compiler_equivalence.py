"""THEOREM-TO-COMPILER EQUIVALENCE battery.

Closes the appendix gap that Lemma 1 (base-scheme certificates) certifies
the textbook operators rather than the emitted code: for every implemented
base scheme (fbs, drs, pdhg, condat_vu, dys) and every wrapper gene
(relaxation rho, Halpern anchor, fixed-k anchor restart) this module holds
an INDEPENDENT hand-written reference update (pure numpy f64, written
straight from the LSCOMO formulas, no reuse of atlas.schemes code) and
asserts that the compiled update from atlas/schemes matches it
iterate-by-iterate for 50 iterations on 3 random small problems per
scheme, to 1e-10 relative error.

Problem pool (small instances):
  - TV 16x16            (atlas.ir.spec.tv_denoise_problem; fbs/drs run on
                         the DUAL, pdhg/condat_vu on the saddle form)
  - tiny netflow LP     (atlas.bench.lp_cell / netflow, degree-2 regular;
                         pdhg/condat_vu only: f linear kills the fbs/drs
                         dual paths and h(Lx) kills dys)
  - random dense composites (per-scheme shapes: f+g for primal fbs, g+h
                         for primal drs, f+g+h for dys, f(+g)+h(Lx) with a
                         dense random L for pdhg/condat_vu)
Structurally inapplicable (scheme, problem) pairs are filled with extra
random seeds of the applicable families, keeping 3 instances per scheme.

Two layers:
  A. Iterate-by-iterate: the jit-compiled fixed-point map built.T from
     atlas.schemes.base_schemes.build_scheme is stepped 50 times and every
     state is compared against the reference trajectory (full solver
     state: dual u for fbs, auxiliary z for drs/dys, saddle pair (x, u)
     for pdhg/condat_vu), plus the decoded primal at the end.
  B. Production path: atlas.schemes.compile.solve_genome (Genome ->
     phenotype, the evolution entry point, including the km/halpern loop
     of atlas.wrappers.km with the wrapper genes) is run to fixed
     iteration counts (tol=0) and the decoded iterate is compared against
     the reference trajectory with the SAME wrapper applied in numpy:
       plain            z+ = T z
       relaxation       z+ = (1-rho) z + rho T z
       halpern          z+ = anchor/(j+2) + (j+1)/(j+2) T z
       halpern+restart  anchor <- z, j <- 0 every restart_k iterations

State decoders (what maps solver state to the primal iterate), asserted
against BuiltScheme.extract / SolveResult.x:
  fbs  (dual, TV form): x = y - L^T clip(u, -lam, lam)   [x = grad f*(-L^T u)]
  fbs  (primal):        x = state (identity)
  drs  (dual, TV form): u = clip(z, -lam, lam) = prox_{a h*}(z);
                        x = y - L^T u
  drs  (primal):        x = prox_{a g}(z)
  pdhg / condat_vu:     x = first saddle coordinate
  dys:                  x = prox_{a g}(z)

NOTE (finding, fixed): the FBS dual-path gradient in base_schemes read
(L L^T u - Dc)/mu with Dc = L grad_conj(0); the correct affine-conjugate
gradient is L L^T u / mu - Dc. Both are bitwise identical at mu = 1 (the
only strong-convexity constant any current atom carries), so no recorded
result is affected; the fix future-proofs the path and this battery pins
the correct form.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

import atlas  # noqa: F401  (enforces JAX_PLATFORMS=cpu + float64)
import jax
import jax.numpy as jnp

from atlas.bench.lp_cell import from_netflow, MatrixLinOp, to_composite
from atlas.bench.netflow import gen_regular_flow
from atlas.genome.schema import Genome, GenomeParams, Wrapper
from atlas.ir.spec import CompositeProblem, lasso_box_problem, tv_denoise_problem
from atlas.operators.prox import Box, QuadraticDataFidelity, ScaledL1, Zero
from atlas.schemes.base_schemes import (
    CondatVuParams,
    CondatVuMetricParams,
    FISTAParams,
    DRSParams,
    DYSParams,
    FBSParams,
    PDHGParams,
    build_scheme,
)
from atlas.schemes.compile import solve_genome
from atlas.bench import ot as ot_bench
from atlas.wrappers.km import Budget

TOL = 1e-10
N_ITERS = 50
CHECKPOINTS = (1, 2, 8, 50)  # 8 crosses the restart_k=7 anchor reset


# ---------------------------------------------------------------------------
# Numpy reference building blocks (independent closed forms)
# ---------------------------------------------------------------------------


def _soft(v, t):
    """prox of t*||.||_1: soft threshold [LSCOMO SS2.5]."""
    return np.sign(v) * np.maximum(np.abs(v) - t, 0.0)


def _dense_from_linop(L, in_shape):
    """Materialize a LinOp as a dense (m, n) matrix by probing basis
    vectors (the operator is problem DATA; the reference reimplements the
    scheme formulas, not the stencil)."""
    n = int(np.prod(in_shape))
    cols = []
    for j in range(n):
        e = np.zeros(n)
        e[j] = 1.0
        cols.append(np.asarray(L.forward(jnp.asarray(e.reshape(in_shape)))).ravel())
    return np.stack(cols, axis=1)


class Ref:
    """A reference iteration: step (plain map T), initial state, decoder."""

    def __init__(self, step, state0, decode):
        self.step = step
        self.state0 = state0
        self.decode = decode


def _tmap(f, *states):
    if isinstance(states[0], tuple):
        return tuple(f(*parts) for parts in zip(*states))
    return f(*states)


def run_reference(ref: Ref, n_iters: int, rho=None, halpern=False, restart_k=None):
    """Reference trajectory z_1..z_n with the wrapper applied in numpy,
    mirroring the LSCOMO wrapper formulas (NOT the atlas code):
      KM/relaxation [SS2.5]:  z+ = (1-rho) z + rho T z   (rho=None -> T z)
      Halpern/OHM  [SS12.2]:  z+ = anchor/(j+2) + (j+1)/(j+2) T z,
      fixed-k restart: every restart_k iterations anchor <- z+, j <- 0."""
    z = _tmap(np.array, ref.state0)
    traj = []
    if not halpern:
        for _ in range(n_iters):
            Tz = ref.step(z)
            if rho is None:
                z = Tz
            else:
                z = _tmap(lambda a, b: (1.0 - rho) * a + rho * b, z, Tz)
            traj.append(z)
        return traj
    anchor = z
    j = 0
    rk = int(restart_k) if restart_k else n_iters + 2
    for _ in range(n_iters):
        Tz = ref.step(z)
        lam_j = 1.0 / (j + 2.0)
        z = _tmap(lambda a, t: lam_j * a + (1.0 - lam_j) * t, anchor, Tz)
        j += 1
        if j >= rk:
            anchor = z
            j = 0
        traj.append(z)
    return traj


def to_flat(state):
    if isinstance(state, tuple):
        return tuple(np.asarray(p, dtype=np.float64).ravel() for p in state)
    return np.asarray(state, dtype=np.float64).ravel()


def rel_err(a, b):
    """max-norm relative error, tuple-aware."""
    if isinstance(b, tuple):
        return max(rel_err(x, y) for x, y in zip(a, b))
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    assert a.shape == b.shape
    return float(np.max(np.abs(a - b)) / (1.0 + np.max(np.abs(b))))


# ---------------------------------------------------------------------------
# Case construction: (CompositeProblem, scheme, params, Ref)
# ---------------------------------------------------------------------------


def _tv_problem(seed, lam):
    rng = np.random.default_rng(seed)
    y = rng.standard_normal((16, 16))
    return tv_denoise_problem(y, lam)


def _case_fbs_tv(seed, lam=0.15, a=0.2):
    """FBS on the TV dual  min_u f*(-L^T u) + h*(u)  [LSCOMO SS2.7 + SS3.2]:
    u+ = prox_{a h*}(u - a grad F(u)),  grad F(u) = L L^T u / mu - L grad f*(0)
    (mu = 1, grad f*(0) = y),  prox_{a h*} = clip to the lam inf-ball."""
    p = _tv_problem(seed, lam)
    y = np.asarray(p.data["y"]).ravel()
    D = _dense_from_linop(p.L, p.shape)
    Dy = D @ y

    def step(u):
        grad = D @ (D.T @ u) - Dy  # mu = 1
        return np.clip(u - a * grad, -lam, lam)

    def decode(u):
        uf = np.clip(u, -lam, lam)
        return y - D.T @ uf, uf

    ref = Ref(step, np.zeros(D.shape[0]), decode)
    return p, "fbs", FBSParams(step=a), ref


def _case_fbs_primal(seed, lam=0.3, a=1.5):
    """FBS primal on  min_x (1/2)||x-y||^2 + lam||x||_1  (no L):
    x+ = prox_{a g}(x - a grad f(x)) = soft(x - a(x - y), a lam)."""
    rng = np.random.default_rng(seed)
    y = rng.standard_normal(40)
    p = CompositeProblem(
        f=QuadraticDataFidelity(y=jnp.asarray(y)),
        g=ScaledL1(lam=lam),
        h=None,
        L=None,
        data={"y": y, "lam": lam},
        shape=(40,),
        name="dense_fg",
    )

    def step(x):
        return _soft(x - a * (x - y), a * lam)

    ref = Ref(step, y.copy(), lambda x: (x, None))  # state0 = grad f*(0) = y
    return p, "fbs", FBSParams(step=a), ref


def _case_drs_tv(seed, lam=0.12, a=0.7):
    """DRS on the TV dual [LSCOMO SS2.7.1]: A = subdiff h*, B = grad F.
    z+ = z + J_aB(2 J_aA(z) - z) - J_aA(z); J_aA = clip to the lam ball;
    J_aB(v) solves (I + (a/mu) L L^T) w = v + a L grad f*(0) (exact numpy
    solve in the reference; the compiled path uses CG at cg_tol)."""
    p = _tv_problem(seed, lam)
    y = np.asarray(p.data["y"]).ravel()
    D = _dense_from_linop(p.L, p.shape)
    m = D.shape[0]
    M = np.eye(m) + a * (D @ D.T)  # mu = 1
    Dy = D @ y

    def step(z):
        u = np.clip(z, -lam, lam)
        v = 2.0 * u - z
        w = np.linalg.solve(M, v + a * Dy)
        return z + w - u

    def decode(z):
        u = np.clip(z, -lam, lam)
        return y - D.T @ u, u

    ref = Ref(step, np.zeros(m), decode)
    return p, "drs", DRSParams(step=a), ref


def _case_drs_primal(seed, lam=0.25, lo=0.3, hi=1.5, a=0.7):
    """DRS primal two-prox split  min_x lam||x||_1 + I_[lo,hi](x):
    z+ = z + prox_{a h}(2 prox_{a g}(z) - z) - prox_{a g}(z)."""
    rng = np.random.default_rng(seed)
    n = 40
    # random shift enters through the box bounds; z0 = 0 moves immediately
    lo = float(lo + 0.1 * rng.uniform())
    p = CompositeProblem(
        f=Zero(),
        g=ScaledL1(lam=lam),
        h=Box(lo=lo, hi=hi),
        L=None,
        data={"lam": lam, "lo": lo, "hi": hi},
        shape=(n,),
        name="two_prox",
    )

    def step(z):
        xg = _soft(z, a * lam)
        xh = np.clip(2.0 * xg - z, lo, hi)
        return z + xh - xg

    ref = Ref(step, np.zeros(n), lambda z: (_soft(z, a * lam), None))
    return p, "drs", DRSParams(step=a), ref


def _lp_problem(seed):
    return to_composite(from_netflow(gen_regular_flow(n_nodes=8, seed=seed, degree=2)))


def _ref_pd(prox_phi, prox_hconj, A, t, s, x0, u0):
    """The shared PDHG/Condat-Vu saddle template [LSCOMO SS3.3]:
    x+ = prox_phi(x - t A^T u);  u+ = prox_hconj(u + s A(2x+ - x))."""

    def step(state):
        x, u = state
        xn = prox_phi(x - t * (A.T @ u), t)
        un = prox_hconj(u + s * (A @ (2.0 * xn - x)), s)
        return (xn, un)

    return Ref(step, (x0, u0), lambda st: (st[0], st[1]))


def _case_pdhg_tv(seed, lam=0.1, t=0.25, s=0.45):
    """PDHG on TV: phi = f = (1/2)||.-y||^2 (prox (v + t y)/(1 + t)),
    prox_{s h*} = clip to the lam ball."""
    p = _tv_problem(seed, lam)
    y = np.asarray(p.data["y"]).ravel()
    D = _dense_from_linop(p.L, p.shape)
    ref = _ref_pd(
        lambda v, tt: (v + tt * y) / (1.0 + tt),
        lambda v, ss: np.clip(v, -lam, lam),
        D, t, s, y.copy(), np.zeros(D.shape[0]),
    )
    return p, "pdhg", PDHGParams(t=t, s=s), ref


def _case_pdhg_lp(seed):
    """PDHG on the standard-form LP: phi = <c, .> + I_{x>=0} via the
    linear-tilt identity prox_{t phi}(v) = max(v - t c, 0);
    prox_{s h*}(v) = v - s b  (h = I_{z=b}, h* = <b, .>)."""
    p = _lp_problem(seed)
    c = np.asarray(p.data["c"])
    b = np.asarray(p.data["b"])
    A = np.asarray(p.data["A_dense"])
    nb = float(p.L.norm_bound)
    t = s = 0.9 / nb
    ref = _ref_pd(
        lambda v, tt: np.maximum(v - tt * c, 0.0),
        lambda v, ss: v - ss * b,
        A, t, s, np.zeros(c.shape[0]), np.zeros(b.shape[0]),
    )
    return p, "pdhg", PDHGParams(t=t, s=s), ref


def _case_pdhg_dense(seed, lam=0.2):
    """PDHG on a random dense composite  min (1/2)||x-y||^2 + lam||Ax||_1."""
    rng = np.random.default_rng(seed)
    n, m = 30, 20
    y = rng.standard_normal(n)
    A = rng.standard_normal((m, n)) / np.sqrt(n)
    L = MatrixLinOp.from_dense(A)
    p = CompositeProblem(
        f=QuadraticDataFidelity(y=jnp.asarray(y)),
        g=None,
        h=ScaledL1(lam=lam),
        L=L,
        data={"y": y, "lam": lam},
        shape=(n,),
        name="dense_saddle",
    )
    nb = float(L.norm_bound)
    t = s = 0.9 / nb
    ref = _ref_pd(
        lambda v, tt: (v + tt * y) / (1.0 + tt),
        lambda v, ss: np.clip(v, -lam, lam),
        A, t, s, y.copy(), np.zeros(m),
    )
    return p, "pdhg", PDHGParams(t=t, s=s), ref


def _ref_cv(grad_f, prox_g, prox_hconj, A, t, s, x0, u0):
    """Condat-Vu template [LSCOMO SS3.3 eq. 3.13]:
    x+ = prox_g(x - t grad_f(x) - t A^T u); u+ = prox_hconj(u + s A(2x+ - x))."""

    def step(state):
        x, u = state
        xn = prox_g(x - t * grad_f(x) - t * (A.T @ u), t)
        un = prox_hconj(u + s * (A @ (2.0 * xn - x)), s)
        return (xn, un)

    return Ref(step, (x0, u0), lambda st: (st[0], st[1]))


def _case_cv_tv(seed, lam=0.2, t=0.2, s=0.5):
    """Condat-Vu on TV: g = 0 (prox identity), grad f = x - y."""
    p = _tv_problem(seed, lam)
    y = np.asarray(p.data["y"]).ravel()
    D = _dense_from_linop(p.L, p.shape)
    ref = _ref_cv(
        lambda x: x - y,
        lambda v, tt: v,
        lambda v, ss: np.clip(v, -lam, lam),
        D, t, s, y.copy(), np.zeros(D.shape[0]),
    )
    return p, "condat_vu", CondatVuParams(t=t, s=s), ref


def _case_cv_lp(seed):
    """Condat-Vu on the LP (L_f = 0: the vanilla-PDLP iteration):
    x+ = max(x - t c - t A^T u, 0); u+ = u + s(A(2x+ - x) - b)."""
    p = _lp_problem(seed)
    c = np.asarray(p.data["c"])
    b = np.asarray(p.data["b"])
    A = np.asarray(p.data["A_dense"])
    nb = float(p.L.norm_bound)
    t = s = 0.9 / nb
    ref = _ref_cv(
        lambda x: c,
        lambda v, tt: np.maximum(v, 0.0),
        lambda v, ss: v - ss * b,
        A, t, s, np.zeros(c.shape[0]), np.zeros(b.shape[0]),
    )
    return p, "condat_vu", CondatVuParams(t=t, s=s), ref


def _case_cv_dense(seed, lam_g=0.15, lam_h=0.25, t=0.2):
    """Condat-Vu on a random dense three-block composite
    min (1/2)||x-y||^2 + lam_g||x||_1 + lam_h||Ax||_1."""
    rng = np.random.default_rng(seed)
    n, m = 30, 20
    y = rng.standard_normal(n)
    A = rng.standard_normal((m, n)) / np.sqrt(n)
    L = MatrixLinOp.from_dense(A)
    nb = float(L.norm_bound)
    s = 0.9 * (1.0 - t * 1.0 / 2.0) / (t * nb**2)
    p = CompositeProblem(
        f=QuadraticDataFidelity(y=jnp.asarray(y)),
        g=ScaledL1(lam=lam_g),
        h=ScaledL1(lam=lam_h),
        L=L,
        data={"y": y},
        shape=(n,),
        name="dense_three",
    )
    ref = _ref_cv(
        lambda x: x - y,
        lambda v, tt: _soft(v, tt * lam_g),
        lambda v, ss: np.clip(v, -lam_h, lam_h),
        A, t, s, y.copy(), np.zeros(m),
    )
    return p, "condat_vu", CondatVuParams(t=t, s=s), ref


def _ref_entropic(c, b, A, t, s, mass, w_S, x0, u0):
    """Entropic (Bregman) PDHG on the mass simplex, from the formula
    [Chambolle & Pock 2016, Bregman proximal steps; kernel mass * sum x log x]:
      x+ = argmin <c + A^T u, x> + (mass/t) KL(x, x_k) over {x >= 0, sum x = mass}
         = mass * softmax(log x_k - (t/mass)(c + A^T u))
      u+ = u + s (A(2x+ - x) - b)
    Decoder recovers the LP dual u - min(c + A^T u) w_S (A^T w_S = 1)."""

    def step(state):
        x, u = state
        w = np.log(np.maximum(x, 1e-300)) - (t / mass) * (c + A.T @ u)
        top = w.max()
        xn = mass * np.exp(w - (top + np.log(np.exp(w - top).sum())))
        un = u + s * (A @ (2.0 * xn - x) - b)
        return (xn, un)

    def decode(state):
        x, u = state
        return (x, u - np.min(c + A.T @ u) * w_S)

    return Ref(step, (x0, u0), decode)


def _case_entropic_ot(seed, m, n, t=0.4):
    """pdhg_entropic on a small transport LP (the simplex sum x = mass is
    implied by the marginals; ||A||_{1->2} = sqrt(2), so t*s*2 < 1)."""
    p = ot_bench.transport_composite(m, n, seed)
    b = np.asarray(p.data["b"])
    mass = float(b[:m].sum())
    p = dataclasses.replace(p, data={**p.data, "simplex_mass": mass,
                                     "meta": {**p.data["meta"], "m": m, "n": n}})
    c, A = np.asarray(p.data["c"]), np.asarray(p.data["A_dense"])
    s = 0.98 / (2.0 * t)
    w_S = np.concatenate([np.ones(m), np.zeros(n)])
    x0 = np.outer(b[:m], -b[m:]).ravel() / mass
    ref = _ref_entropic(c, b, A, t, s, mass, w_S, x0, np.zeros(m + n))
    return p, "pdhg_entropic", PDHGParams(t=t, s=s), ref


def _bfgs_inverse_dense(d, S, Y, c):
    """H = c * (BFGS inverse updates of diag(d) over the pairs), explicit formula
    H+ = (I - r s y^T) H (I - r y s^T) + r s s^T (Nocedal and Wright, eq. 6.17)."""
    H = np.diag(d)
    I = np.eye(len(d))
    for s_, y_ in zip(S, Y):
        r = 1.0 / (y_ @ s_)
        H = (I - r * np.outer(s_, y_)) @ H @ (I - r * np.outer(y_, s_)) + r * np.outer(s_, s_)
    return c * H


def _case_cv_metric(seed, n=6, m=4, pairs=3):
    """condat_vu_metric on a random dense QP  min 1/2 x'Px + q'x  s.t. lo <= Ax <= hi."""
    import scipy.sparse as sp
    from atlas.bench import qp_cell
    from atlas.operators.primal_metric import lbfgs_metric, scale_for_gate
    rng = np.random.default_rng(seed)
    B = rng.normal(size=(n, n)); P = B.T @ B + 0.1 * np.eye(n)
    A = rng.normal(size=(m, n)); q = rng.normal(size=n)
    lo, hi = -rng.uniform(0.5, 1.5, size=m), rng.uniform(0.5, 1.5, size=m)
    prob = qp_cell.to_composite({"P": sp.csc_matrix(P), "q": q, "r": 0.0, "A": sp.csc_matrix(A), "l": lo, "u": hi})
    S = rng.normal(size=(pairs, n)); Y = S @ P.T + 0.3 * S
    met, s, _, _ = scale_for_gate(lbfgs_metric(S, Y), prob.f.P, prob.L, n, m)
    H = _bfgs_inverse_dense(np.asarray(met.d), np.asarray(met.S), np.asarray(met.Y), float(met.c))

    def step(state):
        x, u = state
        xn = x - H @ (P @ x + q + A.T @ u)
        v = u + s * (A @ (2.0 * xn - x))
        return (xn, v - s * np.clip(v / s, lo, hi))

    ref = Ref(step, (np.zeros(n), np.zeros(m)), lambda st: (st[0], st[1]))
    return prob, "condat_vu_metric", CondatVuMetricParams(s=s, metric=met), ref


def _case_fista(seed, n=120, p=15, alpha=0.6, restart=True):
    """fista (with gradient restart) on a small elastic-net logistic regression."""
    from atlas.bench import logreg_cell as lc
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p)) / np.sqrt(p)
    y = np.sign(X @ rng.normal(size=p) + 0.2 * rng.normal(size=n)); y[y == 0] = 1
    lam = 0.05 * lc.lam_max(X, y, alpha)
    prob = lc.composite(X, y, lam=lam, alpha=alpha)
    a = 1.0 / prob.f.props.lipschitz

    def grad(w):
        z = y * (X @ w)
        return -(X.T @ (y / (1.0 + np.exp(z)))) / n

    def prox(v):
        return np.sign(v) * np.maximum(np.abs(v) - a * lam * alpha, 0.0) / (1.0 + a * lam * (1.0 - alpha))

    def step(state):
        x, yv, t = state
        xn = prox(yv - a * grad(yv))
        tn = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        yn = xn + ((t - 1.0) / tn) * (xn - x)
        if restart and (yv - xn) @ (xn - x) > 0:
            tn, yn = 1.0, xn
        return (xn, yn, np.asarray(tn))

    ref = Ref(step, (np.zeros(p), np.zeros(p), np.asarray(1.0)), lambda st: (st[0], None))
    return prob, "fista", FISTAParams(step=a, restart=restart), ref


def _case_dys(seed, lam=0.2, lo=0.1, hi=1.2, a=1.2):
    """Davis-Yin on lasso_box  min (1/2)||x-y||^2 + lam||x||_1 + I_[lo,hi]:
    x_g = prox_{a g}(z); x_h = prox_{a h}(2 x_g - z - a grad f(x_g));
    z+ = z + x_h - x_g  [LSCOMO SS2.7 + SS13.4.2 Thm 28]."""
    rng = np.random.default_rng(seed)
    y = rng.standard_normal(40)
    p = lasso_box_problem(jnp.asarray(y), lam, lo=lo, hi=hi)

    def step(z):
        xg = _soft(z, a * lam)
        xh = np.clip(2.0 * xg - z - a * (xg - y), lo, hi)
        return z + xh - xg

    ref = Ref(step, np.zeros(40), lambda z: (_soft(z, a * lam), None))
    return p, "dys", DYSParams(step=a), ref


# 3 random small problems per scheme; index 0 also carries the wrapper
# battery. Structurally inapplicable pool members (LP for fbs/drs/dys, TV
# for dys) are replaced by fresh random seeds of applicable families.
CASES = {
    "fbs-tv16": lambda: _case_fbs_tv(101),
    "fbs-dense1": lambda: _case_fbs_primal(102),
    "fbs-dense2": lambda: _case_fbs_primal(103, lam=0.12, a=0.8),
    "drs-tv16": lambda: _case_drs_tv(201),
    "drs-twoprox1": lambda: _case_drs_primal(202),
    "drs-twoprox2": lambda: _case_drs_primal(203, lam=0.4, a=1.3),
    "pdhg-tv16": lambda: _case_pdhg_tv(301),
    "pdhg-netflow": lambda: _case_pdhg_lp(302),
    "pdhg-dense": lambda: _case_pdhg_dense(303),
    "cv-tv16": lambda: _case_cv_tv(401),
    "cv-netflow": lambda: _case_cv_lp(402),
    "cv-dense": lambda: _case_cv_dense(403),
    "dys-lassobox1": lambda: _case_dys(501),
    "dys-lassobox2": lambda: _case_dys(502, lam=0.35, a=0.6),
    "dys-lassobox3": lambda: _case_dys(503, lam=0.1, lo=-0.4, hi=0.9, a=1.8),
    "entropic-ot1": lambda: _case_entropic_ot(601, 6, 5),
    "entropic-ot2": lambda: _case_entropic_ot(602, 5, 7, t=0.2),
    "entropic-ot3": lambda: _case_entropic_ot(603, 4, 4, t=0.6),
    "cvm-qp1": lambda: _case_cv_metric(701),
    "cvm-qp2": lambda: _case_cv_metric(702, n=8, m=5, pairs=4),
    "cvm-qp3": lambda: _case_cv_metric(703, n=5, m=6, pairs=2),
    "fista-logreg1": lambda: _case_fista(801),
    "fista-logreg2": lambda: _case_fista(802, n=200, p=25, alpha=0.3),
    "fista-logreg3": lambda: _case_fista(803, restart=False),
}

# Wrapper battery per scheme, applied to the scheme's first case. rho
# values satisfy rho * averagedness < 1 for that case's certificate.
WRAPPER_CASE = {
    "fbs": ("fbs-tv16", 1.1),
    "drs": ("drs-tv16", 1.5),
    "pdhg": ("pdhg-tv16", 1.5),
    "condat_vu": ("cv-tv16", 1.3),
    "dys": ("dys-lassobox1", 1.3),
}

RESTART_K = 7

# Schemes whose certificate carries no averagedness (the Bregman step is not an
# averaged map in a Hilbert metric): every wrapper gene must be REJECTED at the
# gate, which test_wrapper_inadmissible_schemes_reject_every_wrapper checks.
# They are also not genome scheme genes yet, so Layer B does not apply.
WRAPPER_INADMISSIBLE = {"pdhg_entropic"}
# Reachable through build_scheme only (not genome scheme genes yet): no Layer B.
NOT_GENOME_SCHEMES = {"pdhg_entropic", "condat_vu_metric", "fista"}
GENOME_CASES = sorted(c for c in CASES if not c.startswith(("entropic-", "cvm-", "fista-")))


# ---------------------------------------------------------------------------
# Layer A: compiled T vs reference, iterate by iterate (plain map)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_id", sorted(CASES))
def test_compiled_update_matches_reference_iterates(case_id):
    problem, scheme, params, ref = CASES[case_id]()
    built = build_scheme(problem, scheme, params, check=True)
    assert built.cert is not None and built.cert.satisfied
    dyn = jax.tree_util.tree_map(jnp.asarray, built.dyn)
    step = jax.jit(built.T)
    state = jax.tree_util.tree_map(jnp.asarray, built.state0)

    # Initial states agree (including the grad_conj warm start x0 = y).
    err0 = rel_err(to_flat(state), to_flat(ref.state0))
    assert err0 <= TOL, f"{case_id}: state0 mismatch {err0:.3e}"

    traj = run_reference(ref, N_ITERS)
    worst = 0.0
    for k in range(N_ITERS):
        state = step(state, jnp.asarray(k), dyn)
        err = rel_err(to_flat(state), to_flat(traj[k]))
        worst = max(worst, err)
        assert err <= TOL, (
            f"{case_id}: compiled iterate {k + 1} deviates from the "
            f"textbook reference by {err:.3e} (> {TOL:g})"
        )

    # Decoder: BuiltScheme.extract equals the reference decoder.
    x, u = built.extract(state, dyn)
    xr, ur = ref.decode(traj[-1])
    assert rel_err(to_flat(x), to_flat(xr)) <= TOL, f"{case_id}: decoder x"
    assert (u is None) == (ur is None)
    if u is not None:
        assert rel_err(to_flat(u), to_flat(ur)) <= TOL, f"{case_id}: decoder u"


# ---------------------------------------------------------------------------
# Layer B: full production path (Genome -> solve_genome -> km/halpern loop)
# ---------------------------------------------------------------------------


def _genome(scheme, params, rho=None, halpern=False, restart_k=None):
    if isinstance(params, (PDHGParams, CondatVuParams)):
        tau, sigma = params.t, params.s
    else:
        tau, sigma = params.step, None
    return Genome(
        scheme=scheme,
        params=GenomeParams(tau=tau, sigma=sigma, rho=rho),
        wrapper=Wrapper(
            halpern=halpern,
            restart="fixed_k" if restart_k else "none",
            restart_k=restart_k,
        ),
    )


def _production_vs_reference(case_id, rho=None, halpern=False, restart_k=None):
    problem, scheme, params, ref = CASES[case_id]()
    genome = _genome(scheme, params, rho=rho, halpern=halpern, restart_k=restart_k)
    traj = run_reference(
        ref, max(CHECKPOINTS), rho=rho, halpern=halpern, restart_k=restart_k
    )
    for k in CHECKPOINTS:
        res = solve_genome(
            problem, genome, Budget(max_iters=k, tol=0.0, sample_every=1)
        )
        if res.iters < k:
            # tol=0.0 only stops early on an EXACT fixed point (metric == 0;
            # possible for piecewise-affine maps, e.g. the two-prox DRS
            # split). Equivalence then requires the reference trajectory to
            # be stationary from that iteration on.
            err_stat = rel_err(
                to_flat(traj[k - 1]), to_flat(traj[res.iters - 1])
            )
            assert err_stat <= TOL, (
                f"{case_id}: compiled path stopped at an exact fixed point "
                f"after {res.iters} iterations but the reference still "
                f"moves (drift {err_stat:.3e})"
            )
        else:
            assert res.iters == k, (
                f"{case_id}: expected {k} iterations, got {res.iters}"
            )
        assert res.cert is not None and res.cert.satisfied
        xr, ur = ref.decode(traj[k - 1])
        err = rel_err(to_flat(res.x), to_flat(xr))
        assert err <= TOL, (
            f"{case_id} wrapper(rho={rho}, halpern={halpern}, "
            f"restart_k={restart_k}): decoded x at iteration {k} deviates "
            f"by {err:.3e} (> {TOL:g})"
        )
        if res.u is not None and ur is not None:
            erru = rel_err(to_flat(res.u), to_flat(ur))
            assert erru <= TOL, (
                f"{case_id} wrapper(rho={rho}, halpern={halpern}, "
                f"restart_k={restart_k}): decoded u at iteration {k} "
                f"deviates by {erru:.3e}"
            )


@pytest.mark.parametrize("case_id", GENOME_CASES)
def test_solve_genome_plain(case_id):
    _production_vs_reference(case_id)


@pytest.mark.parametrize("scheme", sorted(WRAPPER_CASE))
def test_solve_genome_relaxation(scheme):
    case_id, rho = WRAPPER_CASE[scheme]
    _production_vs_reference(case_id, rho=rho)


@pytest.mark.parametrize("scheme", sorted(WRAPPER_CASE))
def test_solve_genome_halpern(scheme):
    case_id, _ = WRAPPER_CASE[scheme]
    _production_vs_reference(case_id, halpern=True)


@pytest.mark.parametrize("scheme", sorted(WRAPPER_CASE))
def test_solve_genome_halpern_restart(scheme):
    case_id, _ = WRAPPER_CASE[scheme]
    _production_vs_reference(case_id, halpern=True, restart_k=RESTART_K)


# ---------------------------------------------------------------------------
# Battery completeness: every implemented base scheme is covered, so a new
# scheme cannot ship without an equivalence reference (drift guard).
# ---------------------------------------------------------------------------


def test_battery_covers_every_implemented_scheme():
    from atlas.schemes.base_schemes import _BUILDERS

    per_scheme = {}
    for c in CASES:
        s = CASES[c]()[1]
        per_scheme[s] = per_scheme.get(s, 0) + 1
    assert set(per_scheme) == set(_BUILDERS), (
        f"equivalence battery covers {sorted(per_scheme)} but the compiler "
        f"implements {sorted(_BUILDERS)}: add a reference implementation "
        "and 3 problem instances for the missing scheme(s)"
    )
    assert set(WRAPPER_CASE) == set(_BUILDERS) - WRAPPER_INADMISSIBLE - NOT_GENOME_SCHEMES
    assert all(v == 3 for v in per_scheme.values()), per_scheme


@pytest.mark.parametrize("case_id", ["entropic-ot1"])
def test_wrapper_inadmissible_schemes_reject_every_wrapper(case_id):
    from atlas.schemes.base_schemes import _wrap_cert
    from atlas.typing_ import TypeCheckError
    problem, scheme, params, _ = CASES[case_id]()
    assert scheme in WRAPPER_INADMISSIBLE
    cert = build_scheme(problem, scheme, params, check=True).cert
    assert cert.satisfied
    for kw in ({"rho": 1.2, "halpern": False}, {"rho": None, "halpern": True},
               {"rho": None, "halpern": False, "outer": "haugazeau"}):
        with pytest.raises(TypeCheckError):
            _wrap_cert(cert, **kw)
