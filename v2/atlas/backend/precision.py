"""Execution precision policies (backend lane, bounded Hawkeye scope).

Policies are first-class objects deciding the dtype in which a fixed-point
iteration runs, and when the convergence certificate is evaluated. The
backend invariant (see CONTRACT.md):

    backend policies change HOW T is computed, never WHAT T is.

Two rules are enforced mechanically here, not by convention:

1. The certificate (duality gap) is ALWAYS evaluated in exact float64 on
   promoted copies of the iterate, regardless of iterate dtype. certified_gap
   raises PrecisionError if the supplied gap function returns anything else.
2. Termination is decided ONLY by the float64 certificate value, never by an
   f32 quantity.

The three policies:

- F64Policy: baseline. Iterate f64, certify f64 every K iterations.
- F32Cert64Policy: iterate f32; every K iterations promote a copy of the
  iterate to f64 and evaluate the duality gap in f64; stop when that f64 gap
  reaches tol. On consumer GPUs (f64 at roughly 1/60 of f32 throughput) this
  buys the f32 iteration speed while keeping the certificate rigorous.
- SchedulePolicy: start f32; switch to f64 when the f64-evaluated gap either
  crosses the threshold theta or plateaus (relative improvement below
  plateau_rtol for plateau_patience consecutive checks); the f64 phase then
  runs to tol. This is the discoverable f32-iterate / f64-finish precision
  schedule gene.

Why there is deliberately NO bf16 (or f16) policy: the duality gap is a
difference of two near-equal numbers (primal value minus dual value), the
textbook catastrophic-cancellation setting. bf16 carries 8 mantissa bits
(about 2 to 3 decimal digits): a 1e-4 relative gap is below its resolution
entirely, and even f32 summation noise at n around 1e6 elements is the size
of the gap itself; the community 1e-8 tolerance tier is f64-only. A bf16
"certificate" would be noise dressed as a proof, so get_policy refuses to
construct one. bf16 ITERATES could in principle be certified by an f64 gap,
but on this problem class the iterate quantization error alone keeps the
true gap above any useful tolerance, so the policy is excluded until a use
case with commensurate tolerances exists.

The runner is deliberately self-contained: it takes dtype-parametric
factories rather than importing atlas.schemes (which is being rewritten).
    step_factory(dtype) -> step(state, k) -> state   (constants cast to dtype)
    init_factory(dtype) -> state0                    (pytree of arrays)
    gap64(state_f64)    -> scalar float64 relative gap
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np


class PrecisionError(RuntimeError):
    """A precision-policy contract violation (e.g. non-f64 certificate)."""


# ---------------------------------------------------------------------------
# Certificate evaluation: exact float64, always
# ---------------------------------------------------------------------------


def promote64(state):
    """Promote every leaf of a pytree state to float64 (a copy; the running
    low-precision iterate is not touched)."""
    return jax.tree_util.tree_map(lambda a: jnp.asarray(a, jnp.float64), state)


def certified_gap(gap64: Callable, state) -> np.float64:
    """Evaluate the duality-gap certificate at exact float64.

    The iterate is promoted to f64 first, so every arithmetic operation in
    gap64 (whose problem data must already be f64) runs at f64. Raises
    PrecisionError if the result comes back at any other dtype.
    """
    g = np.asarray(jax.device_get(gap64(promote64(state))))
    if g.dtype != np.float64:
        raise PrecisionError(
            f"certificate evaluated at {g.dtype}; certificates must be exact "
            "float64 (cancellation destroys the gap below f64)"
        )
    return np.float64(g)


# ---------------------------------------------------------------------------
# Result record
# ---------------------------------------------------------------------------


@dataclass
class PolicyResult:
    state: Any                    # final iterate, promoted to f64
    iters: int                    # total fixed-point iterations run
    converged: bool               # f64 gap <= tol
    certified_gap: np.float64     # final f64 gap
    gap_history: np.ndarray       # f64 gap at every certificate evaluation
    gap_iters: np.ndarray         # iteration count at each evaluation
    switch_iter: Optional[int]    # SCHEDULE: iteration of the f32 -> f64 switch
    iterate_dtypes: tuple         # dtypes the iteration actually ran at
    certificate_dtype: str = "float64"
    compile_time_s: float = 0.0   # XLA compile seconds (EXCLUDED from exec)
    exec_time_s: float = 0.0      # chunk execution + f64 certificate seconds


# ---------------------------------------------------------------------------
# Shared phase machinery
# ---------------------------------------------------------------------------


def _leaf_dtype(state) -> str:
    return str(jax.tree_util.tree_leaves(state)[0].dtype)


# Compiled-chunk cache: maps (static_key, K, abstract(state), abstract(dyn))
# to the AOT-compiled executable, exactly like atlas.wrappers.km._EXEC_CACHE.
# Without it every solve re-traced and re-compiled the chunk runner (the
# policy path paid a constant per-evaluation compile cost the km path did
# not, biasing fitness against backend genomes; sprint finding, 2026-08-26).
_CHUNK_CACHE: dict = {}
_CHUNK_COMPILE_LOG: list = []


def chunk_compile_count() -> int:
    """Number of chunk-runner compiles since import / last clear (test hook)."""
    return len(_CHUNK_COMPILE_LOG)


def clear_chunk_cache() -> None:
    _CHUNK_CACHE.clear()
    _CHUNK_COMPILE_LOG.clear()


# Shared with atlas.wrappers.km. The device-blind copy that stood here made a
# GPU solve receive a CPU-compiled program in a process holding both backends.
from atlas._cache_keys import abstract as _abstract  # noqa: E402


def _get_chunk_runner(step3, K, state, dyn, static_key):
    """AOT lower+compile the K-step chunk runner; cached by static_key when
    one is given. Returns (executable, compile_seconds). step3 takes
    (state, k, dyn) with dyn a DYNAMIC pytree argument, so a new genome on
    the same problem shape reuses the executable (the km-path convention:
    callers MUST route every semantic constant either through dyn or through
    static_key)."""

    def run_chunk(s, k0, d):
        def body(i, ss):
            return step3(ss, k0 + i, d)

        return jax.lax.fori_loop(0, K, body, s)

    k0_ex = jnp.asarray(0)
    if static_key is None:
        t0 = time.perf_counter()
        exe = jax.jit(run_chunk).lower(state, k0_ex, dyn).compile()
        return exe, time.perf_counter() - t0
    key = ("precision_chunk", static_key, int(K), _abstract(state), _abstract(dyn))
    exe = _CHUNK_CACHE.get(key)
    if exe is not None:
        return exe, 0.0
    t0 = time.perf_counter()
    exe = jax.jit(run_chunk).lower(state, k0_ex, dyn).compile()
    _CHUNK_CACHE[key] = exe
    _CHUNK_COMPILE_LOG.append(key)
    return exe, time.perf_counter() - t0


def _run_phase(step, state, k, K, max_iters, gap64, tol,
               gaps: list, gap_iters: list,
               stop_extra: Optional[Callable[[list], bool]] = None,
               *, dyn=None, static_key=None):
    """Run K-step chunks until the f64 gap <= tol, stop_extra fires, or the
    iteration budget is exhausted. Returns (state, k, converged, extra_fired,
    compile_s, exec_s). Termination consults ONLY the f64 certificate.

    Compile is AOT (before the exec timer) and cached across solves when
    static_key is given with the dyn form; exec_s covers chunk execution
    plus the f64 certificate evaluations, NEVER compile."""
    if dyn is None:
        # Legacy closure form step(state, k): adapt to the 3-arg contract
        # with a placeholder dyn; the closure carries the constants, so the
        # executable is NEVER shared across callers (static_key forced None).
        def step3(s, kk, d, _step=step):
            return _step(s, kk)

        dyn = {"_": jnp.zeros(())}
        static_key = None
    else:
        step3 = step
    if K < 1 or max_iters < 0:
        raise ValueError("K must be positive and max_iters nonnegative")
    exe, compile_s = _get_chunk_runner(step3, K, state, dyn, static_key)
    # Compile the final partial chunk before timing. Never run past the
    # declared iteration budget, including when a precision phase starts
    # at a nonzero iteration.
    remainder = (max_iters - k) % K
    last_exe = exe
    if remainder:
        last_exe, last_compile = _get_chunk_runner(
            step3, remainder, state, dyn, static_key)
        compile_s += last_compile
    exec_s = 0.0
    phase_gaps: list = []
    while k < max_iters:
        t0 = time.perf_counter()
        count = min(K, max_iters - k)
        runner = exe if count == K else last_exe
        state = runner(state, jnp.asarray(k), dyn)
        jax.block_until_ready(state)
        k += count
        g = certified_gap(gap64, state)
        exec_s += time.perf_counter() - t0
        gaps.append(g)
        gap_iters.append(k)
        phase_gaps.append(g)
        if bool(np.isfinite(g) and 0 <= g <= tol):
            return state, k, True, False, compile_s, exec_s
        if not np.isfinite(g) or g < 0:
            return state, k, False, True, compile_s, exec_s
        if stop_extra is not None and stop_extra(phase_gaps):
            return state, k, False, True, compile_s, exec_s
    return state, k, False, False, compile_s, exec_s


def _finalize(state, k, converged, gaps, gap_iters, switch_iter, dtypes,
              compile_s=0.0, exec_s=0.0):
    return PolicyResult(
        state=promote64(state),
        iters=k,
        converged=converged,
        certified_gap=np.float64(gaps[-1]) if len(gaps) else np.float64(np.inf),
        gap_history=np.asarray(gaps, dtype=np.float64),
        gap_iters=np.asarray(gap_iters, dtype=np.int64),
        switch_iter=switch_iter,
        iterate_dtypes=tuple(dtypes),
        compile_time_s=float(compile_s),
        exec_time_s=float(exec_s),
    )


def run_device_policy(step, state, dyn, gap_fn, gap_dyn, *, K, max_iters,
                      tol, static_key) -> PolicyResult:
    """One compiled device loop with certificate checks between step blocks.

    Both iteration data and original-coordinate f64 certificate data are
    explicit dynamic arguments. This prevents stale-data cache reuse across
    instances. Checks occur at K, 2K, ... and exactly at max_iters. No host
    synchronization occurs between checks; the final result is synchronized
    before ending the execution timer. Float64 is numerical arithmetic, not
    an interval-arithmetic or machine-checked proof of floating-point error.
    """
    if isinstance(K, bool) or int(K) != K or K < 1 or max_iters < 1:
        raise ValueError("positive integer cadence and iteration cap required")
    K, max_iters = int(K), int(max_iters)
    n_checks = (max_iters + K - 1) // K
    state = jax.tree_util.tree_map(jnp.asarray, state)
    dyn = jax.tree_util.tree_map(jnp.asarray, dyn)
    gap_dyn = jax.tree_util.tree_map(jnp.asarray, gap_dyn)

    def run(s0, d, gd, threshold):
        def cond(c):
            _, k, n, g, _ = c
            return (k < max_iters) & ((n == 0) |
                   (jnp.isfinite(g) & (g >= 0) & (g > threshold)))

        def body(c):
            s, k, n, _, hist = c
            end = jnp.minimum(k + K, max_iters)
            s = jax.lax.fori_loop(k, end, lambda i, z: step(z, i, d), s)
            g = gap_fn(promote64(s), gd)
            if g.dtype != jnp.float64 or g.shape != ():
                raise PrecisionError("certificate must be a scalar float64")
            return s, end, n + 1, g, hist.at[n].set(g)

        return jax.lax.while_loop(cond, body, (
            s0, jnp.asarray(0), jnp.asarray(0),
            jnp.asarray(jnp.inf, jnp.float64),
            jnp.full((n_checks,), jnp.nan, jnp.float64)))

    key = ("device_policy", static_key, K, max_iters,
           _abstract(state), _abstract(dyn), _abstract(gap_dyn))
    threshold = jnp.asarray(tol, jnp.float64)
    # A None static_key means the caller has NOT promised that every semantic
    # constant reaches this function through `dyn`. `step` is a closure, so two
    # callers can differ (a different relaxation rho, say) while producing an
    # identical key, and the second would silently execute the first's program.
    # _get_chunk_runner already refuses to cache in that case; match it here.
    # Measured before this fix: rho=1.0 returned rho=1.9's trajectory, 640
    # iterations instead of 1152.
    cacheable = static_key is not None
    exe = _CHUNK_CACHE.get(key) if cacheable else None
    compile_s = 0.0
    if exe is None:
        t0 = time.perf_counter()
        exe = jax.jit(run).lower(state, dyn, gap_dyn, threshold).compile()
        compile_s = time.perf_counter() - t0
        if cacheable:
            _CHUNK_CACHE[key] = exe
            _CHUNK_COMPILE_LOG.append(key)
    t0 = time.perf_counter()
    out = exe(state, dyn, gap_dyn, threshold)
    jax.block_until_ready(out)
    exec_s = time.perf_counter() - t0
    final, k, n, g, hist = out
    k, n, g = int(k), int(n), float(g)
    gaps = np.asarray(hist[:n])
    gap_iters = np.minimum(np.arange(1, n + 1) * K, max_iters)
    return _finalize(final, k, np.isfinite(g) and 0 <= g <= tol,
                     gaps, gap_iters, None, [_leaf_dtype(state)], compile_s,
                     exec_s)


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class F64Policy:
    """Baseline: iterate f64, certify f64 every K iterations."""

    K: int = 100
    name: str = field(default="f64", init=False)

    def run(self, step_factory, init_factory, gap64, tol, max_iters,
            dyn_factory=None, static_key=None) -> PolicyResult:
        state = init_factory(jnp.float64)
        gaps: list = []
        gap_iters: list = []
        dyn = None if dyn_factory is None else dyn_factory(jnp.float64)
        state, k, conv, _, c_s, e_s = _run_phase(
            step_factory(jnp.float64), state, 0, self.K, max_iters,
            gap64, tol, gaps, gap_iters, dyn=dyn, static_key=static_key,
        )
        return _finalize(state, k, conv, gaps, gap_iters, None, ["float64"],
                         c_s, e_s)


@dataclass(frozen=True)
class F32Cert64Policy:
    """Iterate f32; every K iterations certify at f64 on a promoted copy.

    Termination is decided ONLY by the f64 gap. The f32 iterate limits the
    reachable gap to roughly f32 resolution times the problem scale; use for
    the 1e-4 tolerance tier, not 1e-8."""

    K: int = 100
    name: str = field(default="f32_cert64", init=False)

    def run(self, step_factory, init_factory, gap64, tol, max_iters,
            dyn_factory=None, static_key=None) -> PolicyResult:
        state = init_factory(jnp.float32)
        gaps: list = []
        gap_iters: list = []
        dyn = None if dyn_factory is None else dyn_factory(jnp.float32)
        state, k, conv, _, c_s, e_s = _run_phase(
            step_factory(jnp.float32), state, 0, self.K, max_iters,
            gap64, tol, gaps, gap_iters, dyn=dyn, static_key=static_key,
        )
        return _finalize(state, k, conv, gaps, gap_iters, None, ["float32"],
                         c_s, e_s)


@dataclass(frozen=True)
class SchedulePolicy:
    """Start f32; switch to f64 when the f64-evaluated gap crosses theta or
    plateaus; finish at f64. theta and K are the policy parameters exposed
    as backend genes (see CONTRACT.md)."""

    K: int = 100
    theta: float = 1e-3
    plateau_rtol: float = 0.01
    plateau_patience: int = 2
    name: str = field(default="schedule", init=False)

    def _should_switch(self, phase_gaps: list) -> bool:
        if phase_gaps[-1] <= self.theta:
            return True
        if len(phase_gaps) > self.plateau_patience:
            recent = phase_gaps[-(self.plateau_patience + 1):]
            stalls = [
                (recent[i] - recent[i + 1]) / max(recent[i], 1e-300)
                < self.plateau_rtol
                for i in range(len(recent) - 1)
            ]
            return all(stalls)
        return False

    def run(self, step_factory, init_factory, gap64, tol, max_iters,
            dyn_factory=None, static_key=None) -> PolicyResult:
        gaps: list = []
        gap_iters: list = []
        dtypes = ["float32"]
        state = init_factory(jnp.float32)
        dyn32 = None if dyn_factory is None else dyn_factory(jnp.float32)
        # Reserve at least one certificate block for f64 refinement, even
        # when neither the threshold nor the plateau heuristic fires.
        prefix_cap = max(0, max_iters - min(self.K, max_iters))
        state, k, conv, switched, c_s, e_s = _run_phase(
            step_factory(jnp.float32), state, 0, self.K, prefix_cap,
            gap64, tol, gaps, gap_iters, stop_extra=self._should_switch,
            dyn=dyn32, static_key=static_key,
        )
        switch_iter = None
        if not conv and (switched or k < max_iters):
            switch_iter = k
            dtypes.append("float64")
            state = promote64(state)
            dyn64 = None if dyn_factory is None else dyn_factory(jnp.float64)
            state, k, conv, _, c2, e2 = _run_phase(
                step_factory(jnp.float64), state, k, self.K, max_iters,
                gap64, tol, gaps, gap_iters, dyn=dyn64, static_key=static_key,
            )
            c_s += c2
            e_s += e2
        return _finalize(state, k, conv, gaps, gap_iters, switch_iter, dtypes,
                         c_s, e_s)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

POLICIES = {
    "f64": F64Policy,
    "f32_cert64": F32Cert64Policy,
    "schedule": SchedulePolicy,
}

_SUB_F32_NAMES = ("bf16", "bfloat16", "f16", "float16", "half")


def get_policy(name: str, **kwargs):
    """Construct a policy by its genome name. Refuses sub-f32 precisions."""
    if name in _SUB_F32_NAMES:
        raise PrecisionError(
            f"no '{name}' policy is provided, deliberately: the duality-gap "
            "certificate is a difference of near-equal numbers and bf16/f16 "
            "cannot represent a 1e-4 relative gap at all (catastrophic "
            "cancellation); certificates are evaluated at f64 only, and "
            "sub-f32 iterates cannot reach the tolerance tiers we certify"
        )
    if name not in POLICIES:
        raise ValueError(
            f"unknown precision policy '{name}'; valid: {sorted(POLICIES)}"
        )
    return POLICIES[name](**kwargs)
