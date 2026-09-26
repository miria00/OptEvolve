"""Batched vmap backend: one compiled kernel advances a whole SAME-SHAPE
batch of problem instances in lockstep (the physical substrate of the
backend gene `batch`; see INTEGRATION.md in this directory).

Design
------
The per-instance fixed-point step T (mirroring atlas.schemes.base_schemes
bit-for-bit: the same atom methods in the same order) is written against a
per-instance data pytree `per` plus a shared parameter pytree `shared`,
then vmapped over the leading batch axis of `per` and of the state. The
vmapped step is wrapped in a K-iteration fori_loop chunk runner, AOT
compiled BEFORE the wall-clock timer and cached across solves (the km-path
convention: everything semantic is either in the dynamic arguments or in
the cache key).

Per-instance stopping is done by MASKING: the whole batch keeps iterating,
but a lane whose own float64 certificate has reached tol is FROZEN with
jnp.where on a done-mask. Frozen lanes still burn FLOPs until the last
lane finishes or the budget caps out; that is the standard lockstep-batch
tradeoff and it is deliberate (the alternative, compacting the batch,
recompiles per survivor-set shape and loses the whole point).

Certificates: the per-instance duality gap is evaluated on the batch at
EXACT float64 (iterates promoted, problem data kept in a separate f64
copy), every K iterations. A lane terminates ONLY on its own f64 gap;
no f32 quantity ever decides termination (atlas.backend.precision rules).

TIMING CONVENTION (the factorial's backend-arm contract): batched_solve
reports ONE wall clock for the whole batch, `batch_wall_s`, covering
compiled chunk execution plus the f64 certificate evaluations and
EXCLUDING AOT compile (recorded separately in `compile_time_s`). The
convenience number `per_instance_effective_s = batch_wall_s / n_instances`
is derived, clearly labeled, and is NOT a per-instance measurement: the
hardware factorial compares backends on `batch_wall_s`.

Scope (deliberate): schemes pdhg / condat_vu (the saddle family, covering
both cells), problem kinds tv_denoise and lp_standard (netflow incidence
or dense matrix), precisions f64 and f32_cert64. The `schedule` policy is
refused here: its f32->f64 switch is a per-lane event and a lockstep batch
has no sound shared switch point yet. Mix genomes are refused (the switch
monitor is host-side control flow).
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Optional

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np

from atlas.backend.precision import PrecisionError
from atlas.certify.residuals import lp_rel_gap, rel_gap
from atlas.ir.spec import CompositeProblem
from atlas.typing_ import (
    ConstructionCert,
    TypeCheckError,
    typecheck_condat_vu,
    typecheck_halpern,
    typecheck_pdhg,
    typecheck_relaxation,
    typecheck_restart,
)
from atlas.wrappers.km import Budget, _abstract

SUPPORTED_SCHEMES = ("pdhg", "condat_vu")
SUPPORTED_PRECISIONS = ("f64", "f32_cert64")


# ---------------------------------------------------------------------------
# Result record
# ---------------------------------------------------------------------------


@dataclass
class BatchedSolveResult:
    """Result of one batched solve. Per-lane arrays have leading dim B.

    iters[i] is the batch iteration count at the FIRST certificate check
    where lane i's f64 gap was <= tol (granularity K), or the total count
    run if the lane never converged. batch_iters is the number of
    iterations the batch actually executed (every lane burned them,
    frozen or not).
    """

    x: np.ndarray                 # (B, *primal_shape), float64
    u: Optional[np.ndarray]       # (B, *dual_shape), float64
    iters: np.ndarray             # (B,) int64, per-lane (cadence K)
    batch_iters: int              # iterations the lockstep batch executed
    converged: np.ndarray         # (B,) bool, own-f64-gap <= tol
    certified_gap: np.ndarray     # (B,) float64 final gap per lane
    gap_history: np.ndarray       # (n_checks, B) float64
    gap_iters: np.ndarray         # (n_checks,) int64
    batch_wall_s: float           # ONE clock, whole batch, compile EXCLUDED
    per_instance_effective_s: float  # batch_wall_s / B (derived label ONLY)
    compile_time_s: float
    n_instances: int
    precision: str
    K: int
    scheme: str
    cert: Optional[ConstructionCert]
    iterate_dtype: str = "float64"
    certificate_dtype: str = "float64"


# ---------------------------------------------------------------------------
# Compile cache (km-pattern)
# ---------------------------------------------------------------------------

_BATCH_CACHE: dict = {}
_BATCH_COMPILE_LOG: list = []


def batched_compile_count() -> int:
    return len(_BATCH_COMPILE_LOG)


def clear_batched_cache() -> None:
    _BATCH_CACHE.clear()
    _BATCH_COMPILE_LOG.clear()


def _compile_cached(key, fn, *args):
    exe = _BATCH_CACHE.get(key)
    if exe is None:
        exe = jax.jit(fn).lower(*args).compile()
        _BATCH_CACHE[key] = exe
        _BATCH_COMPILE_LOG.append(key)
    return exe


def _cast_floats(tree, dtype):
    """Cast floating leaves to dtype; integer leaves (arc indices) kept."""

    def _c(a):
        a = jnp.asarray(a)
        return a.astype(dtype) if jnp.issubdtype(a.dtype, jnp.floating) else a

    return jax.tree_util.tree_map(_c, tree)


def _promote64(tree):
    return _cast_floats(tree, jnp.float64)


def _freeze(done, old, new):
    """tree_map jnp.where(done, old, new) with done (B,) broadcast to each
    (B, ...) leaf. Frozen lanes keep burning FLOPs; this only discards the
    result (documented lockstep tradeoff)."""

    def _w(o, n):
        mask = done.reshape((done.shape[0],) + (1,) * (o.ndim - 1))
        return jnp.where(mask, o, n)

    return jax.tree_util.tree_map(_w, old, new)


# ---------------------------------------------------------------------------
# Genome / params adapter (duck-typed: atlas.genome is being reworked, so
# only attribute names of the CURRENT public schema are touched, guarded)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Plan:
    scheme: str
    t: float
    s: float
    rho: Optional[float]
    halpern: bool
    restart_k: Optional[int]
    precision: str
    K: int


def _extract_plan(params_or_genome, precision_policy, K) -> _Plan:
    p = params_or_genome
    if hasattr(p, "scheme") and hasattr(p, "params"):  # Genome duck-type
        if getattr(p, "mix", None) is not None or p.scheme == "mix":
            raise NotImplementedError(
                "batched_solve does not run mix genomes (switch_k's prefix "
                "handoff and finiteness checks are host-side per-lane control "
                "flow); solve them sequentially via solve_genome"
            )
        if getattr(p, "metric", None) is not None:
            raise NotImplementedError(
                "batched_solve does not run metric (rescale) genomes yet: the "
                "Ruiz/Pock-Chambolle diagonals are per-instance data, so a "
                "lockstep batch would need per-lane ScaledLinOps and per-lane "
                "operator-norm bounds; solve them sequentially via solve_genome"
            )
        scheme = str(p.scheme)
        gp = p.params
        t = float(gp.tau)
        s = getattr(gp, "sigma", None)
        rho = getattr(gp, "rho", None)
        w = getattr(p, "wrapper", None)
        halpern = bool(getattr(w, "halpern", False))
        rk = getattr(w, "restart_k", None) if halpern else None
        be = getattr(p, "backend", None)
        prec = precision_policy or (getattr(be, "precision", None) or "f64")
        cadence = int(K or getattr(be, "K", None) or 100)
    elif hasattr(p, "t") and hasattr(p, "s"):  # PDHGParams / CondatVuParams
        scheme = "condat_vu" if type(p).__name__.startswith("CondatVu") else "pdhg"
        t, s = float(p.t), float(p.s)
        rho = getattr(p, "rho", None)
        halpern = bool(getattr(p, "halpern", False))
        rk = getattr(p, "restart_k", None) if halpern else None
        prec = precision_policy or "f64"
        cadence = int(K or 100)
    else:
        raise TypeError(
            "params_or_genome must be a Genome or a PDHGParams/CondatVuParams; "
            f"got {type(p).__name__}"
        )
    if scheme not in SUPPORTED_SCHEMES:
        raise NotImplementedError(
            f"batched_solve supports schemes {SUPPORTED_SCHEMES}; got {scheme!r}"
        )
    if s is None:
        raise TypeCheckError(f"{scheme} needs both stepsizes (t, s); sigma missing")
    if prec == "schedule":
        raise NotImplementedError(
            "the 'schedule' precision policy is not defined for the lockstep "
            "batch (the f32->f64 switch is a per-lane event); use 'f64' or "
            "'f32_cert64'"
        )
    if prec not in SUPPORTED_PRECISIONS:
        raise PrecisionError(
            f"batched_solve supports precisions {SUPPORTED_PRECISIONS}; "
            f"got {prec!r} (certificates are f64-only, always)"
        )
    if rho is not None and halpern:
        # BACKEND LIMITATION, not a construction failure. The scalar path
        # admits relax-then-anchor as of 2026-09-05 (see
        # typecheck_relaxation: rho in (0,2] with an anchored step following,
        # which includes the reflection rho=2). The lockstep batched kernel
        # has not been extended to that composition, so it refuses rather
        # than silently running a different iteration than the certificate
        # describes. Use the scalar path for relax-then-anchor genomes.
        raise TypeCheckError(
            "batched backend limitation: relax-then-anchor (rho + halpern) "
            "is admitted by the gate but not implemented in the lockstep "
            "batch kernel; run this genome on the scalar path"
        )
    if rk is not None and not halpern:
        raise TypeCheckError("restart_k requires the Halpern anchor")
    return _Plan(scheme, float(t), float(s), rho, halpern, rk, prec, cadence)


# ---------------------------------------------------------------------------
# Problem stacking (same-shape enforcement) + per-instance steps
# ---------------------------------------------------------------------------


class _IncOp:
    """Per-lane incidence operator built from traced (tail, head) arrays;
    arithmetic identical to atlas.bench.lp_cell.IncidenceLinOp."""

    def __init__(self, tail, head, n_nodes: int):
        self.tail, self.head, self.n_nodes = tail, head, n_nodes

    def forward(self, x):
        y = jnp.zeros(self.n_nodes, dtype=x.dtype)
        return y.at[self.tail].add(x).at[self.head].add(-x)

    def adjoint(self, y):
        return y[self.tail] - y[self.head]


class _DenseOp:
    def __init__(self, A):
        self.A = A

    def forward(self, x):
        return self.A @ x

    def adjoint(self, u):
        return self.A.T @ u


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(f"batched_solve same-shape contract violated: {msg}")


def _stack_problems(problems) -> tuple:
    """Validate the batch and build (kind, per64, statics). per64 is the
    per-instance f64 data pytree with leading dim B; statics carries the
    shape-level constants shared by every lane."""
    _require(len(problems) >= 1, "empty problem list")
    names = {p.name for p in problems}
    _require(len(names) == 1, f"mixed problem kinds {sorted(names)}")
    name = problems[0].name
    shapes = {tuple(p.shape) for p in problems}
    _require(len(shapes) == 1, f"mixed primal shapes {sorted(shapes)}")

    if name == "tv_denoise":
        ys = np.stack([np.asarray(p.data["y"], dtype=np.float64) for p in problems])
        lams = np.asarray([float(p.data["lam"]) for p in problems], dtype=np.float64)
        per = {"y": jnp.asarray(ys), "lam": jnp.asarray(lams)}
        statics = {
            "shape": problems[0].shape,
            "L": problems[0].L,  # Grad2D: shape-only, identical across lanes
            "opnorm": float(problems[0].L.norm_bound),
        }
        return "tv", per, statics

    if name == "lp_standard":
        Ls = [p.L for p in problems]
        opnorm = max(float(L.norm_bound) for L in Ls)
        cs = np.stack([np.asarray(p.data["c"], dtype=np.float64) for p in problems])
        bs = np.stack([np.asarray(p.data["b"], dtype=np.float64) for p in problems])
        if all(hasattr(L, "tail") for L in Ls):
            n_arcs = {int(np.asarray(L.tail).shape[0]) for L in Ls}
            n_nodes = {int(L.n_nodes) for L in Ls}
            _require(len(n_arcs) == 1, f"mixed arc counts {sorted(n_arcs)}")
            _require(len(n_nodes) == 1, f"mixed node counts {sorted(n_nodes)}")
            per = {
                "c": jnp.asarray(cs),
                "b": jnp.asarray(bs),
                "tail": jnp.stack([jnp.asarray(L.tail) for L in Ls]),
                "head": jnp.stack([jnp.asarray(L.head) for L in Ls]),
            }
            statics = {
                "shape": problems[0].shape,
                "lp_form": "incidence",
                "n_nodes": int(next(iter(n_nodes))),
                "opnorm": opnorm,
            }
            return "lp", per, statics
        if all(hasattr(L, "A") for L in Ls):
            Ashapes = {tuple(np.asarray(L.A).shape) for L in Ls}
            _require(len(Ashapes) == 1, f"mixed A shapes {sorted(Ashapes)}")
            per = {
                "c": jnp.asarray(cs),
                "b": jnp.asarray(bs),
                "A": jnp.stack([jnp.asarray(L.A) for L in Ls]),
            }
            statics = {
                "shape": problems[0].shape,
                "lp_form": "dense",
                "n_nodes": None,
                "opnorm": opnorm,
            }
            return "lp", per, statics
        raise ValueError(
            "batched_solve lp_standard needs a uniform LinOp family "
            "(all IncidenceLinOp or all MatrixLinOp)"
        )

    raise NotImplementedError(
        f"batched_solve supports tv_denoise and lp_standard; got {name!r}"
    )


def _lane_linop(kind: str, per_i: dict, statics: dict):
    if kind == "tv":
        return statics["L"]
    if statics["lp_form"] == "incidence":
        return _IncOp(per_i["tail"], per_i["head"], statics["n_nodes"])
    return _DenseOp(per_i["A"])


def _make_step_and_metric(kind: str, scheme: str, statics: dict):
    """(step_single, metric_single, x0_single) mirroring base_schemes.

    step_single(state_i, per_i, shared) -> state_i     (any float dtype)
    metric_single(state_i, per_i) -> scalar            (evaluated at f64)
    The atom arithmetic is EXACTLY the sequential path's: the same atom
    objects are constructed per lane from the per-instance data and their
    methods called in the same order as base_schemes' T closures, so the
    batched result matches the sequential result to roundoff (tested to
    1e-10 at f64 in tests/test_batched_backend.py).
    """
    if kind == "tv":
        from atlas.operators.prox import QuadraticDataFidelity, ScaledL1

        def step_pdhg(state, per, shared):
            f = QuadraticDataFidelity(y=per["y"])
            h = ScaledL1(lam=per["lam"])
            L = _lane_linop(kind, per, statics)
            x, u = state
            t, s = shared["t"], shared["s"]
            x_new = f.prox(x - t * L.adjoint(u), t)
            u_new = h.prox_conj(u + s * L.forward(2.0 * x_new - x), s)
            return (x_new, u_new)

        def step_cv(state, per, shared):
            f = QuadraticDataFidelity(y=per["y"])
            h = ScaledL1(lam=per["lam"])
            L = _lane_linop(kind, per, statics)
            x, u = state
            t, s = shared["t"], shared["s"]
            # g = Zero: prox is the identity (base_schemes condat_vu on TV)
            x_new = x - t * f.grad(x) - t * L.adjoint(u)
            u_new = h.prox_conj(u + s * L.forward(2.0 * x_new - x), s)
            return (x_new, u_new)

        def metric(state, per):
            x, u = state
            return rel_gap(x, u, per["y"], per["lam"], statics["L"])

        def x0(per_i):
            # base_schemes: x0 = phi.grad_conj(0) = y for the TV saddle
            return per_i["y"]

        return (step_pdhg if scheme == "pdhg" else step_cv), metric, x0

    # kind == "lp"
    from atlas.bench.lp_cell import AffineEqualitySet, LinearCost, NonnegOrthant

    def step_lp_pdhg(state, per, shared):
        fc = LinearCost(c=per["c"])
        g = NonnegOrthant()
        h = AffineEqualitySet(b=per["b"])
        L = _lane_linop(kind, per, statics)
        x, u = state
        t, s = shared["t"], shared["s"]
        # AffineTiltedProx(affine=f, p=g): p.prox(affine.prox(v, t), t)
        x_new = g.prox(fc.prox(x - t * L.adjoint(u), t), t)
        u_new = h.prox_conj(u + s * L.forward(2.0 * x_new - x), s)
        return (x_new, u_new)

    def step_lp_cv(state, per, shared):
        fc = LinearCost(c=per["c"])
        g = NonnegOrthant()
        h = AffineEqualitySet(b=per["b"])
        L = _lane_linop(kind, per, statics)
        x, u = state
        t, s = shared["t"], shared["s"]
        x_new = g.prox(x - t * fc.grad(x) - t * L.adjoint(u), t)
        u_new = h.prox_conj(u + s * L.forward(2.0 * x_new - x), s)
        return (x_new, u_new)

    def metric(state, per):
        x, u = state
        A = _lane_linop(kind, per, statics)
        return lp_rel_gap(x, u, per["c"], A, per["b"])

    def x0(per_i):
        return jnp.zeros(statics["shape"], dtype=jnp.float64)

    return (step_lp_pdhg if scheme == "pdhg" else step_lp_cv), metric, x0


# ---------------------------------------------------------------------------
# The batched solve
# ---------------------------------------------------------------------------


def batched_solve(
    problems,
    params_or_genome,
    budget: Budget,
    precision_policy: Optional[str] = None,
    K: Optional[int] = None,
) -> BatchedSolveResult:
    """Solve a SAME-SHAPE batch of instances with one vmapped kernel.

    problems:  list of CompositeProblems, all the same kind and shapes
               (tv_denoise: same (H, W); lp_standard: same (m, n) and,
               for netflow, the same arc/node counts). Mismatch raises
               ValueError.
    params_or_genome: a Genome (non-mix) or PDHGParams / CondatVuParams.
    budget:    Budget(max_iters, tol, ...); sample_every is ignored, the
               certificate cadence is K.
    precision_policy: "f64" | "f32_cert64" | None (None: the genome's
               backend gene, else "f64"). "schedule" is refused.
    K:         certificate cadence in iterations (None: genome backend.K,
               else 100).

    Wrapper genes rho / halpern+restart_k are applied per lane, inside the
    vmapped step, with the identical algebra of the sequential path; the
    Level-1 construction certificate is checked ONCE against the WORST
    operator norm in the batch (so it covers every lane).

    See the module docstring for the timing and masking contracts.
    """
    plan = _extract_plan(params_or_genome, precision_policy, K)
    kind, per64, statics = _stack_problems(problems)
    B = len(problems)

    # Level-1 certificate against the worst-case opnorm (covers all lanes).
    if plan.scheme == "pdhg":
        cert = typecheck_pdhg(t=plan.t, s=plan.s, opnorm=statics["opnorm"])
    else:
        L_smooth = 1.0 if kind == "tv" else 0.0
        cert = typecheck_condat_vu(
            t=plan.t, s=plan.s, opnorm=statics["opnorm"], L_smooth=L_smooth
        )
    if plan.rho is not None:
        cert = typecheck_relaxation(cert, plan.rho)
    if plan.halpern:
        cert = typecheck_halpern(cert)
        if plan.restart_k is not None:
            cert = typecheck_restart(cert, plan.restart_k)

    step_single, metric_single, x0_single = _make_step_and_metric(
        kind, plan.scheme, statics
    )
    rho_v = 1.0 if plan.rho is None else float(plan.rho)
    rk_v = int(plan.restart_k) if plan.restart_k else int(budget.max_iters) + 2

    if not plan.halpern:

        def lane_step(state, per, shared):
            Tx = step_single(state, per, shared)
            if rho_v == 1.0:
                return Tx
            return jax.tree_util.tree_map(
                lambda a, b: (1.0 - rho_v) * a + rho_v * b, state, Tx
            )

        def lane_metric(state, per):
            return metric_single(state, per)

        def lane_init(per_i, dtype):
            z = (x0_single(per_i), _dual_zeros(kind, per_i, statics))
            return _cast_floats(z, dtype)

    else:

        def lane_step(aug, per, shared):
            z, anchor, j = aug
            Tx = step_single(z, per, shared)
            lam_j = 1.0 / (j + 2.0)
            new = jax.tree_util.tree_map(
                lambda a, t_: lam_j * a + (1.0 - lam_j) * t_, anchor, Tx
            )
            j_next = j + 1.0
            do_restart = j_next >= rk_v
            anchor = jax.tree_util.tree_map(
                lambda a, n: jnp.where(do_restart, n, a), anchor, new
            )
            j_next = jnp.where(do_restart, jnp.zeros_like(j_next), j_next)
            return (new, anchor, j_next)

        def lane_metric(aug, per):
            return metric_single(aug[0], per)

        def lane_init(per_i, dtype):
            z = (x0_single(per_i), _dual_zeros(kind, per_i, statics))
            z = _cast_floats(z, dtype)
            return (z, z, jnp.zeros((), dtype=dtype))

    batched_step = jax.vmap(lane_step, in_axes=(0, 0, None))
    batched_metric64 = jax.vmap(lane_metric, in_axes=(0, 0))

    dtype = jnp.float64 if plan.precision == "f64" else jnp.float32
    per_run = _cast_floats(per64, dtype)
    shared_run = _cast_floats(
        {"t": jnp.asarray(plan.t), "s": jnp.asarray(plan.s)}, dtype
    )
    state = jax.vmap(lambda p: lane_init(p, dtype))(per_run)
    done0 = jnp.zeros((B,), dtype=bool)

    Kc = int(plan.K)

    def run_chunk(state_, done_, per_, shared_):
        def body(i, st):
            new = batched_step(st, per_, shared_)
            return _freeze(done_, st, new)

        return jax.lax.fori_loop(0, Kc, body, state_)

    def gap_batch(state64_, per64_):
        return batched_metric64(state64_, per64_)

    static_core = (
        "batched", kind, plan.scheme, statics.get("lp_form"),
        statics.get("n_nodes"), rho_v, plan.halpern, rk_v, Kc,
    )
    t_c0 = time.perf_counter()
    chunk_key = (static_core, "chunk", _abstract(state), _abstract(per_run))
    chunk_exe = _compile_cached(chunk_key, run_chunk, state, done0, per_run,
                                shared_run)
    state64_ex = _promote64(state)
    gap_key = (static_core, "gap", _abstract(state64_ex), _abstract(per64))
    gap_exe = _compile_cached(gap_key, gap_batch, state64_ex, per64)
    compile_s = time.perf_counter() - t_c0

    done = done0
    iters_lane = np.zeros((B,), dtype=np.int64)
    conv = np.zeros((B,), dtype=bool)
    gaps_hist: list = []
    gap_iters: list = []
    k = 0
    exec_s = 0.0
    tol = float(budget.tol)
    max_iters = int(budget.max_iters)
    while k < max_iters:
        t0 = time.perf_counter()
        state = chunk_exe(state, done, per_run, shared_run)
        jax.block_until_ready(state)
        k += Kc
        g = np.asarray(jax.device_get(gap_exe(_promote64(state), per64)))
        exec_s += time.perf_counter() - t0
        if g.dtype != np.float64:
            raise PrecisionError(
                f"batched certificate evaluated at {g.dtype}; certificates "
                "must be exact float64"
            )
        gaps_hist.append(g)
        gap_iters.append(k)
        newly = (~conv) & (g <= tol)
        iters_lane[newly] = k
        conv |= newly
        done = jnp.asarray(conv)
        if bool(conv.all()):
            break
    iters_lane[~conv] = k

    final64 = _promote64(state)
    if plan.halpern:
        final64 = final64[0]
    x, u = final64
    g_final = gaps_hist[-1] if gaps_hist else np.full((B,), np.inf)
    return BatchedSolveResult(
        x=np.asarray(x),
        u=np.asarray(u),
        iters=iters_lane,
        batch_iters=int(k),
        converged=conv,
        certified_gap=np.asarray(g_final, dtype=np.float64),
        gap_history=np.asarray(gaps_hist, dtype=np.float64).reshape(-1, B),
        gap_iters=np.asarray(gap_iters, dtype=np.int64),
        batch_wall_s=float(exec_s),
        per_instance_effective_s=float(exec_s) / B,
        compile_time_s=float(compile_s),
        n_instances=B,
        precision=plan.precision,
        K=Kc,
        scheme=plan.scheme,
        cert=cert,
        iterate_dtype=str(dtype.__name__ if hasattr(dtype, "__name__")
                          else jnp.dtype(dtype).name),
    )


def _dual_zeros(kind: str, per_i: dict, statics: dict):
    if kind == "tv":
        H, W = statics["shape"]
        return jnp.zeros((2, H, W), dtype=jnp.float64)
    return jnp.zeros(per_i["b"].shape, dtype=jnp.float64)
