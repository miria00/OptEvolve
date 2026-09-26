"""Hawkeye-pattern per-atom kernel unit tests (backend lane).

Every atom the schemes are compiled from has exactly one entry in REGISTRY;
each entry knows how to generate random inputs and holds an independent
float64 numpy reference implementation. The harness

    check(backend_impl, reference_impl, make_inputs, tolerance)

runs the backend implementation against the reference on seeded random
inputs and reports the worst normalized error. ANY future kernel variant
(f32 kernel, fused kernel, Pallas kernel, vmapped batch kernel) must pass
the same five checks at its declared tolerance BEFORE it may enter a
genome's phenotype; this is the mechanism behind the paper invariant that
backend genes change how T is computed, never what T is.

The Grad2D check is the adversarial one: the adjoint reference is the exact
matrix transpose of the forward map (built column-by-column from basis
vectors on small non-square shapes), plus the inner-product identity
<D x, u> == <x, D^T u>; this is the check class that caught v1-style silent
adjoint bugs.

Error metric: max|a - b| / (1 + max|ref|), a hybrid absolute/relative error
stable near zero. Default tolerance for f64 kernels is 1e-12; an f32 kernel
variant should declare about 1e-6.

Wired into pytest by tests/test_backend_units.py, which parametrizes over
this registry against the current atlas.operators implementations.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

DEFAULT_TOL_F64 = 1e-12
DEFAULT_TOL_F32 = 1e-6

_LINOP_SHAPES = [(9, 7), (12, 13), (16, 16)]  # non-square first, on purpose


# ---------------------------------------------------------------------------
# Reference implementations (independent, pure numpy, float64)
# ---------------------------------------------------------------------------


def ref_grad_quadratic(x, y):
    return np.asarray(x, np.float64) - np.asarray(y, np.float64)


def ref_prox_l1(v, step, lam=1.0):
    v = np.asarray(v, np.float64)
    t = float(step) * float(lam)
    return np.sign(v) * np.maximum(np.abs(v) - t, 0.0)


def ref_prox_box(v, lo, hi):
    return np.clip(np.asarray(v, np.float64), lo, hi)


def ref_project_linf_ball(v, radius):
    r = float(radius)
    return np.clip(np.asarray(v, np.float64), -r, r)


def ref_grad2d_forward(x):
    """Forward differences, Neumann boundary; independent of atlas code."""
    x = np.asarray(x, np.float64)
    d0 = np.zeros_like(x)
    d0[:-1, :] = x[1:, :] - x[:-1, :]
    d1 = np.zeros_like(x)
    d1[:, :-1] = x[:, 1:] - x[:, :-1]
    return np.stack([d0, d1], axis=0)


_GRAD2D_MATRIX_CACHE: dict = {}


def _grad2d_matrix(shape):
    """Dense matrix M with vec(forward(x)) = M @ vec(x), built column by
    column from basis vectors. The adjoint reference is then EXACTLY M.T."""
    if shape in _GRAD2D_MATRIX_CACHE:
        return _GRAD2D_MATRIX_CACHE[shape]
    H, W = shape
    n = H * W
    M = np.zeros((2 * n, n))
    for j in range(n):
        e = np.zeros(n)
        e[j] = 1.0
        M[:, j] = ref_grad2d_forward(e.reshape(H, W)).ravel()
    _GRAD2D_MATRIX_CACHE[shape] = M
    return M


def ref_grad2d_adjoint(u):
    """Exact matrix transpose of the forward map: M.T @ vec(u)."""
    u = np.asarray(u, np.float64)
    _, H, W = u.shape
    M = _grad2d_matrix((H, W))
    return (M.T @ u.ravel()).reshape(H, W)


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


@dataclass
class CheckReport:
    name: str
    n_trials: int
    max_err: float
    tolerance: float
    passed: bool

    def __str__(self):
        tag = "PASS" if self.passed else "FAIL"
        return (f"[{tag}] {self.name}: max_err={self.max_err:.3e} "
                f"tol={self.tolerance:.1e} over {self.n_trials} trials")


def _err(a, b) -> float:
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    if a.shape != b.shape:
        return float("inf")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        return float("inf")
    denom = 1.0 + float(np.max(np.abs(b))) if b.size else 1.0
    return float(np.max(np.abs(a - b))) / denom if a.size else 0.0


def check(backend_impl: Callable, reference_impl: Callable,
          make_inputs: Callable, tolerance: float,
          n_trials: int = 8, seed: int = 1234,
          name: str = "atom") -> CheckReport:
    """Run backend_impl against reference_impl on seeded random inputs.

    make_inputs(rng) returns the positional argument tuple passed to both
    implementations. Returns a CheckReport; passed = worst error <= tol."""
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(n_trials):
        args = make_inputs(rng)
        worst = max(worst, _err(backend_impl(*args), reference_impl(*args)))
    return CheckReport(name=name, n_trials=n_trials, max_err=worst,
                       tolerance=tolerance, passed=worst <= tolerance)


def check_linop(forward_impl: Callable, adjoint_impl: Callable,
                tolerance: float, shapes=None, n_trials: int = 4,
                seed: int = 1234, name: str = "linop") -> CheckReport:
    """Three-way linear-operator check: forward vs reference, adjoint vs the
    exact matrix transpose, and the inner-product identity
    <D x, u> == <x, D^T u> on random x, u."""
    shapes = shapes or _LINOP_SHAPES
    rng = np.random.default_rng(seed)
    worst = 0.0
    trials = 0
    for shape in shapes:
        H, W = shape
        for _ in range(n_trials):
            x = rng.standard_normal((H, W)) * rng.uniform(0.1, 10.0)
            u = rng.standard_normal((2, H, W)) * rng.uniform(0.1, 10.0)
            fwd_b = np.asarray(forward_impl(x), np.float64)
            adj_b = np.asarray(adjoint_impl(u), np.float64)
            worst = max(worst, _err(fwd_b, ref_grad2d_forward(x)))
            worst = max(worst, _err(adj_b, ref_grad2d_adjoint(u)))
            lhs = float(np.sum(fwd_b * u))
            rhs = float(np.sum(x * adj_b))
            worst = max(worst, abs(lhs - rhs) / (1.0 + abs(lhs)))
            trials += 1
    return CheckReport(name=name, n_trials=trials, max_err=worst,
                       tolerance=tolerance, passed=worst <= tolerance)


# ---------------------------------------------------------------------------
# Batched (vmap) kernel variants: per-atom checks for the batched backend
# ---------------------------------------------------------------------------
#
# The batched vmap backend (atlas.backend.batched) advances a stacked batch
# of instances with vmapped atom kernels. Per the registry contract, those
# vmapped kernels are atoms in their own right and must pass the same
# reference checks BEFORE entering a phenotype: each batched check stacks
# B lanes of random inputs (per-lane scalars become (B,) arrays) and the
# reference is the plain numpy single-lane reference applied lane by lane.

BATCH_LANES = 5


def _ref_lanes(ref: Callable) -> Callable:
    """Lift a single-lane numpy reference to a stacked batch: apply it lane
    by lane along axis 0 (scalars arrive as (B,) arrays and are unwrapped)."""

    def batched_ref(*args):
        B = np.asarray(args[0]).shape[0]

        def lane(a, i):
            a = np.asarray(a)
            return float(a[i]) if a[i].ndim == 0 else a[i]

        return np.stack([ref(*(lane(a, i) for a in args)) for i in range(B)])

    return batched_ref


def _mk_batched_grad_quadratic(rng):
    n = int(rng.integers(3, 24))
    scale = rng.uniform(0.1, 100.0)
    return (rng.standard_normal((BATCH_LANES, n, n)) * scale,
            rng.standard_normal((BATCH_LANES, n, n)) * scale)


def _mk_batched_prox_l1(rng):
    n = int(rng.integers(5, 200))
    v = rng.standard_normal((BATCH_LANES, n)) * rng.uniform(0.1, 10.0)
    return (v, rng.uniform(0.01, 2.0, size=BATCH_LANES),
            rng.uniform(0.05, 3.0, size=BATCH_LANES))


def _mk_batched_prox_box(rng):
    n = int(rng.integers(5, 200))
    v = rng.standard_normal((BATCH_LANES, n)) * rng.uniform(0.1, 10.0)
    lo = rng.uniform(-2.0, 0.0, size=BATCH_LANES)
    return (v, lo, lo + rng.uniform(0.1, 3.0, size=BATCH_LANES))


def _mk_batched_project_linf_ball(rng):
    n = int(rng.integers(5, 200))
    v = rng.standard_normal((BATCH_LANES, n)) * rng.uniform(0.1, 10.0)
    return (v, rng.uniform(0.05, 3.0, size=BATCH_LANES))


def _ref_incidence_forward(tail, head, n_nodes, x):
    """Numpy scatter-add reference for one incidence forward lane."""
    y = np.zeros(int(n_nodes))
    np.add.at(y, np.asarray(tail, np.int64), np.asarray(x, np.float64))
    np.add.at(y, np.asarray(head, np.int64), -np.asarray(x, np.float64))
    return y


def _ref_incidence_adjoint(tail, head, y):
    y = np.asarray(y, np.float64)
    return y[np.asarray(tail, np.int64)] - y[np.asarray(head, np.int64)]


def check_batched_grad2d(forward_impl: Callable, adjoint_impl: Callable,
                         tolerance: float, n_trials: int = 4,
                         seed: int = 1234,
                         name: str = "batched_grad2d") -> CheckReport:
    """Batched Grad2D check: the vmapped forward/adjoint on a (B, H, W) /
    (B, 2, H, W) stack must match the single-lane exact-transpose reference
    LANE BY LANE, plus the per-lane inner-product identity."""
    rng = np.random.default_rng(seed)
    worst = 0.0
    trials = 0
    for shape in _LINOP_SHAPES:
        H, W = shape
        for _ in range(n_trials):
            x = rng.standard_normal((BATCH_LANES, H, W)) * rng.uniform(0.1, 10.0)
            u = rng.standard_normal((BATCH_LANES, 2, H, W)) * rng.uniform(0.1, 10.0)
            fwd_b = np.asarray(forward_impl(x), np.float64)
            adj_b = np.asarray(adjoint_impl(u), np.float64)
            for i in range(BATCH_LANES):
                worst = max(worst, _err(fwd_b[i], ref_grad2d_forward(x[i])))
                worst = max(worst, _err(adj_b[i], ref_grad2d_adjoint(u[i])))
                lhs = float(np.sum(fwd_b[i] * u[i]))
                rhs = float(np.sum(x[i] * adj_b[i]))
                worst = max(worst, abs(lhs - rhs) / (1.0 + abs(lhs)))
            trials += 1
    return CheckReport(name=name, n_trials=trials, max_err=worst,
                       tolerance=tolerance, passed=worst <= tolerance)


def check_batched_incidence(forward_impl: Callable, adjoint_impl: Callable,
                            tolerance: float, n_trials: int = 4,
                            seed: int = 1234,
                            name: str = "batched_incidence") -> CheckReport:
    """Batched incidence-operator check (the LP cell's LinOp under vmap):
    per-lane random graphs (DIFFERENT tail/head per lane, the case the
    batched LP backend actually runs), forward vs numpy scatter-add,
    adjoint vs numpy gather, plus the per-lane inner-product identity.

    forward_impl(tail, head, n_nodes, x): (B,E)x(B,E)x int x(B,E) -> (B,N)
    adjoint_impl(tail, head, n_nodes, y): ... x (B,N) -> (B,E)
    """
    rng = np.random.default_rng(seed)
    worst = 0.0
    trials = 0
    for (N, E) in [(7, 12), (13, 20), (30, 60)]:
        for _ in range(n_trials):
            tail = rng.integers(0, N, size=(BATCH_LANES, E))
            head = (tail + 1 + rng.integers(0, N - 1, size=(BATCH_LANES, E))) % N
            x = rng.standard_normal((BATCH_LANES, E)) * rng.uniform(0.1, 10.0)
            y = rng.standard_normal((BATCH_LANES, N)) * rng.uniform(0.1, 10.0)
            fwd_b = np.asarray(forward_impl(tail, head, N, x), np.float64)
            adj_b = np.asarray(adjoint_impl(tail, head, N, y), np.float64)
            for i in range(BATCH_LANES):
                worst = max(worst, _err(
                    fwd_b[i], _ref_incidence_forward(tail[i], head[i], N, x[i])))
                worst = max(worst, _err(
                    adj_b[i], _ref_incidence_adjoint(tail[i], head[i], y[i])))
                lhs = float(np.sum(fwd_b[i] * y[i]))
                rhs = float(np.sum(x[i] * adj_b[i]))
                worst = max(worst, abs(lhs - rhs) / (1.0 + abs(lhs)))
            trials += 1
    return CheckReport(name=name, n_trials=trials, max_err=worst,
                       tolerance=tolerance, passed=worst <= tolerance)


# ---------------------------------------------------------------------------
# Registry: the five per-atom checks + the batched-kernel variants
# ---------------------------------------------------------------------------


def _mk_grad_quadratic(rng):
    shape = (int(rng.integers(3, 40)),) * 2
    scale = rng.uniform(0.1, 100.0)
    return (rng.standard_normal(shape) * scale,
            rng.standard_normal(shape) * scale)


def _mk_prox_l1(rng):
    v = rng.standard_normal(int(rng.integers(5, 500))) * rng.uniform(0.1, 10.0)
    return (v, float(rng.uniform(0.01, 2.0)), float(rng.uniform(0.05, 3.0)))


def _mk_prox_box(rng):
    v = rng.standard_normal(int(rng.integers(5, 500))) * rng.uniform(0.1, 10.0)
    lo = float(rng.uniform(-2.0, 0.0))
    return (v, lo, lo + float(rng.uniform(0.1, 3.0)))


def _mk_project_linf_ball(rng):
    v = rng.standard_normal(int(rng.integers(5, 500))) * rng.uniform(0.1, 10.0)
    return (v, float(rng.uniform(0.05, 3.0)))


def ref_lp_csr_update(A, v, old, base, cost, step, dual):
    if dual:
        return base + step * (A @ (2*v-old)) - step*cost
    return np.maximum(base - step*(A @ v) - step*cost, 0.)


def _mk_lp_csr_update(rng):
    A = rng.normal(size=(37, 53)) * (rng.random((37, 53)) < .2)
    A[3] = 0.
    A[:, 8] = 0.
    return (A, rng.normal(size=53), rng.normal(size=53), rng.normal(size=37),
            rng.normal(size=37), float(rng.uniform(.01, 2.)), bool(rng.integers(2)))


def ref_tv_update(x, u, y, t, s, lam, proximal, rho):
    """Independent small-grid TV map; adjoint uses an explicit matrix transpose."""
    adj = ref_grad2d_adjoint(u)
    xp = ((x-t*adj)+t*y)/(1.+t) if proximal else (x-t*(x-y))-t*adj
    up = np.clip(u+s*ref_grad2d_forward(2.*xp-x), -lam, lam)
    return np.concatenate((((1.-rho)*x+rho*xp).ravel(), ((1.-rho)*u+rho*up).ravel()))


def _mk_tv_update(rng):
    shape = [(1, 1), (1, 7), (9, 1), (9, 7), (17, 13)][int(rng.integers(5))]
    return (rng.normal(size=shape), rng.normal(size=(2, *shape)), rng.normal(size=shape),
            float(rng.uniform(.01, .25)), float(rng.uniform(.01, 1.)),
            float(rng.uniform(.001, 2.)), bool(rng.integers(2)), float(rng.uniform(.1, 1.9)))


@dataclass(frozen=True)
class CheckSpec:
    name: str
    kind: str                       # "pointwise" | "linop"
    reference: Optional[Callable]   # pointwise only
    make_inputs: Optional[Callable] # pointwise only
    tol: float = DEFAULT_TOL_F64


REGISTRY = {
    "tv_update": CheckSpec("tv_update", "pointwise", ref_tv_update, _mk_tv_update),
    "lp_csr_update": CheckSpec("lp_csr_update", "pointwise", ref_lp_csr_update,
                               _mk_lp_csr_update),
    "grad_quadratic": CheckSpec("grad_quadratic", "pointwise",
                                ref_grad_quadratic, _mk_grad_quadratic),
    "prox_l1": CheckSpec("prox_l1", "pointwise", ref_prox_l1, _mk_prox_l1),
    "prox_box": CheckSpec("prox_box", "pointwise", ref_prox_box, _mk_prox_box),
    "project_linf_ball": CheckSpec("project_linf_ball", "pointwise",
                                   ref_project_linf_ball,
                                   _mk_project_linf_ball),
    "grad2d": CheckSpec("grad2d", "linop", None, None),
    # Batched (vmap) kernel variants — the batched backend's atoms.
    "batched_grad_quadratic": CheckSpec(
        "batched_grad_quadratic", "pointwise",
        _ref_lanes(ref_grad_quadratic), _mk_batched_grad_quadratic),
    "batched_prox_l1": CheckSpec(
        "batched_prox_l1", "pointwise",
        _ref_lanes(ref_prox_l1), _mk_batched_prox_l1),
    "batched_prox_box": CheckSpec(
        "batched_prox_box", "pointwise",
        _ref_lanes(ref_prox_box), _mk_batched_prox_box),
    "batched_project_linf_ball": CheckSpec(
        "batched_project_linf_ball", "pointwise",
        _ref_lanes(ref_project_linf_ball), _mk_batched_project_linf_ball),
    "batched_grad2d": CheckSpec("batched_grad2d", "batched_grad2d", None, None),
    "batched_incidence": CheckSpec(
        "batched_incidence", "batched_incidence", None, None),
}


def default_impls() -> dict:
    """The current production atoms (atlas.operators), i.e. what the schemes
    compile against today. grad2d maps to a (forward, adjoint) pair."""
    from atlas.operators import prox as P
    from atlas.operators.linear import Grad2D

    def tv_update(x, u, y, t, s, lam, proximal, rho):
        import jax.numpy as jnp
        x, u, y = map(jnp.asarray, (x, u, y))
        op = Grad2D(tuple(x.shape))
        adj = op.adjoint(u)
        xp = ((x-t*adj)+t*y)/(1.+t) if proximal else (x-t*(x-y))-t*adj
        up = jnp.clip(u+s*op.forward(2.*xp-x), -lam, lam)
        return jnp.concatenate((((1.-rho)*x+rho*xp).ravel(), ((1.-rho)*u+rho*up).ravel()))

    def lp_csr_update(A, v, old, base, cost, step, dual):
        import jax.numpy as jnp
        A, v, old, base, cost = map(jnp.asarray, (A, v, old, base, cost))
        if dual:
            return base + step*(A @ (2*v-old)) - step*cost
        return jnp.maximum(base - step*(A @ v) - step*cost, 0.)

    def g2d_forward(x):
        import jax.numpy as jnp
        x = jnp.asarray(x, jnp.float64)
        return Grad2D(shape=tuple(x.shape)).forward(x)

    def g2d_adjoint(u):
        import jax.numpy as jnp
        u = jnp.asarray(u, jnp.float64)
        return Grad2D(shape=tuple(u.shape[1:])).adjoint(u)

    import jax as _jax
    import jax.numpy as _jnp

    def _v(fn):
        def batched(*args):
            return _jax.vmap(fn)(*[_jnp.asarray(a) for a in args])

        return batched

    def bg2d_forward(x):
        x = _jnp.asarray(x, _jnp.float64)
        op = Grad2D(shape=tuple(x.shape[1:]))
        return _jax.vmap(op.forward)(x)

    def bg2d_adjoint(u):
        u = _jnp.asarray(u, _jnp.float64)
        op = Grad2D(shape=tuple(u.shape[2:]))
        return _jax.vmap(op.adjoint)(u)

    def binc_forward(tail, head, n_nodes, x):
        from atlas.backend.batched import _IncOp

        return _jax.vmap(
            lambda t, h, xx: _IncOp(t, h, int(n_nodes)).forward(xx)
        )(_jnp.asarray(tail), _jnp.asarray(head), _jnp.asarray(x, _jnp.float64))

    def binc_adjoint(tail, head, n_nodes, y):
        from atlas.backend.batched import _IncOp

        return _jax.vmap(
            lambda t, h, yy: _IncOp(t, h, int(n_nodes)).adjoint(yy)
        )(_jnp.asarray(tail), _jnp.asarray(head), _jnp.asarray(y, _jnp.float64))

    return {
        "tv_update": tv_update,
        "lp_csr_update": lp_csr_update,
        "grad_quadratic": P.grad_quadratic,
        "prox_l1": P.prox_l1,
        "prox_box": P.prox_box,
        "project_linf_ball": P.project_linf_ball,
        "grad2d": (g2d_forward, g2d_adjoint),
        "batched_grad_quadratic": _v(P.grad_quadratic),
        "batched_prox_l1": _v(P.prox_l1),
        "batched_prox_box": _v(P.prox_box),
        "batched_project_linf_ball": _v(P.project_linf_ball),
        "batched_grad2d": (bg2d_forward, bg2d_adjoint),
        "batched_incidence": (binc_forward, binc_adjoint),
    }


def run_check(name: str, backend_impl=None,
              tolerance: Optional[float] = None) -> CheckReport:
    """Run one registry check. backend_impl defaults to the production atom;
    a kernel variant passes its own implementation (and, if lower precision,
    its declared tolerance)."""
    spec = REGISTRY[name]
    tol = spec.tol if tolerance is None else tolerance
    if backend_impl is None:
        backend_impl = default_impls()[name]
    if spec.kind == "linop":
        fwd, adj = backend_impl
        return check_linop(fwd, adj, tol, name=name)
    if spec.kind == "batched_grad2d":
        fwd, adj = backend_impl
        return check_batched_grad2d(fwd, adj, tol, name=name)
    if spec.kind == "batched_incidence":
        fwd, adj = backend_impl
        return check_batched_incidence(fwd, adj, tol, name=name)
    return check(backend_impl, spec.reference, spec.make_inputs, tol,
                 name=name)


def run_all(impls: Optional[dict] = None,
            tolerances: Optional[dict] = None) -> list:
    """Run all five checks against an implementation set (default: the
    production atoms). Returns the list of CheckReports; the caller decides
    whether a failure is fatal (pytest asserts, the Stage-0.5 kernel gate
    rejects the variant)."""
    impls = impls or default_impls()
    tolerances = tolerances or {}
    return [run_check(name, impls.get(name), tolerances.get(name))
            for name in sorted(REGISTRY)]
