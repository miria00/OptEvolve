"""Krasnoselskii-Mann iteration loop, compiled ONCE per scheme per shape.

- Plain KM / over-relaxation: x_{k+1} = (1-rho) x_k + rho T(x_k) with rho
  a DYNAMIC argument (rho = 1.0 is exactly the plain iteration: for finite
  states (1-1)*x + 1*T(x) == T(x) bitwise); the relaxed map is
  (rho*alpha)-averaged [LSCOMO SS2.5].
- Halpern/OHM anchor [LSCOMO SS12.2] with fixed-k anchor RESTART:
  x_{k+1} = anchor/(j+2) + (j+1)/(j+2) T(x_k), where j counts iterations
  since the last restart; every restart_k iterations the anchor is reset
  to the current iterate and j restarts. restart_k is DYNAMIC; passing
  restart_k > max_iters reproduces the classic anchored iteration
  (anchor = x0, j = k) exactly.

SPEED LEVER (one compile per scheme per shape): T and metric_fn take the
full dynamic input pytree `dyn` (problem atoms with their data as pytree
leaves, stepsizes, rho / restart_k, tol) as an ARGUMENT. The lowered +
AOT-compiled executable is cached keyed by (static_key, mode, budget
statics, pytree structure + leaf shapes/dtypes of (state0, dyn)), so a new
genome on the same problem shape reuses the executable and only the
runtime arguments change. Callers MUST route every semantic constant
either through `dyn` or through `static_key` - anything baked into the T
closure but absent from both would silently alias cache entries.

wall_time_s measures ONLY the compiled execution (AOT compile happens
before the timer; cache hits pay no compile at all).
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from atlas.certify.residuals import fp_residual


@dataclass(frozen=True)
class Budget:
    max_iters: int = 100_000
    tol: float = 1e-6  # on the metric returned by metric_fn (rel gap)
    sample_every: int = 100


# ---------------------------------------------------------------------------
# Kernel cache
# ---------------------------------------------------------------------------

_EXEC_CACHE: dict = {}
_COMPILE_LOG: list = []  # one entry per actual XLA compile (for tests)


def kernel_compile_count() -> int:
    """Number of kernel compiles since import / last clear (test hook)."""
    return len(_COMPILE_LOG)


def clear_kernel_cache() -> None:
    _EXEC_CACHE.clear()
    _COMPILE_LOG.clear()


def _asarray_tree(tree):
    return jax.tree_util.tree_map(jnp.asarray, tree)


from atlas._cache_keys import abstract as _abstract  # noqa: E402
from atlas._cache_keys import placement as _placement  # noqa: E402,F401


def _compile_cached(key, fn, *abstract_args):
    """AOT lower+compile fn for the given example args, cached by key."""
    exe = _EXEC_CACHE.get(key)
    if exe is None:
        exe = jax.jit(fn).lower(*abstract_args).compile()
        _EXEC_CACHE[key] = exe
        _COMPILE_LOG.append(key)
    return exe


# ---------------------------------------------------------------------------
# The KM loop
# ---------------------------------------------------------------------------


def euclidean_gram(p, q, dyn=None):
    """(<p,p>, <p,q>, <q,q>) summed over the leaves of two state pytrees."""
    lp, lq = jax.tree_util.tree_leaves(p), jax.tree_util.tree_leaves(q)
    pp = sum(jnp.vdot(a, a) for a in lp)
    pq = sum(jnp.vdot(a, b) for a, b in zip(lp, lq))
    qq = sum(jnp.vdot(b, b) for b in lq)
    return pp, pq, qq


def haugazeau_coefficients(pp, pq, qq):
    """Q(a, b, c) = ca*a + cb*b + cc*c, the projection of a onto
    H(a, b) ∩ H(b, c), from the Gram entries of p = a - b and q = b - c
    (Bauschke & Combettes, closed form of Haugazeau's Q):

        rho = pp*qq - pq^2
        Q = c                                  if rho = 0 and pq >= 0
        Q = a + (1 + pq/qq)(c - b)             if rho > 0 and pq*qq >= rho
        Q = b + (qq/rho)(pq (a - b) + pp (c - b))  if rho > 0 and pq*qq < rho

    rho <= 1e-12 * pp * qq is treated as 0. rho = 0 with pq < 0 means the
    intersection is empty, which cannot happen when Fix T is nonempty; the
    step then falls back to c and the returned flag is set."""
    rho = pp * qq - pq * pq
    deg = rho <= 1e-12 * pp * qq
    safe_rho = jnp.where(deg, 1.0, rho)
    safe_qq = jnp.where(qq > 0, qq, 1.0)
    in_first = jnp.logical_and(~deg, pq * qq >= rho)
    k1 = 1.0 + pq / safe_qq
    ca2, cc2 = qq * pq / safe_rho, qq * pp / safe_rho
    ca = jnp.where(deg, 0.0, jnp.where(in_first, 1.0, ca2))
    cc = jnp.where(deg, 1.0, jnp.where(in_first, k1, cc2))
    cb = jnp.where(deg, 0.0, jnp.where(in_first, -k1, 1.0 - ca2 - cc2))
    return ca, cb, cc, jnp.logical_and(deg, pq < 0)


def km_run(T, state0, budget: Budget, metric_fn, dyn=None, *, mode: str = "km",
           static_key=(), rho=None, halpern: bool = False, gram=None):
    """Run the (wrapped) KM loop until metric <= tol or budget exhausted.

    T:         (state, k, dyn) -> state (the fixed-point map).
    metric_fn: (state, dyn) -> scalar termination metric.
    dyn:       pytree of dynamic inputs. Required: dyn["tol"]. Mode "km"
               additionally needs dyn["rho"] (1.0 = plain); mode "halpern"
               ALSO needs dyn["rho"]: it anchors onto F_rho, not onto T
               needs dyn["restart_k"] (> max_iters = never restart).
    static_key: hashable fingerprint of EVERYTHING semantic that lives in
               the closures of T / metric_fn (scheme name, linear operator,
               structural flags, static solver constants).

    LEGACY FORM (P0 API, kept for callers outside atlas.schemes): dyn=None
    with T(state, k), metric_fn(state) and the wrapper genes passed as
    rho= / halpern= keyword floats. This path compiles per call (no
    kernel cache - closures carry the constants), exactly the P0 behavior.

    Returns (final_state, iters, res_history, metric_history, wall_time_s)
    with histories trimmed to the sampled entries actually written.
    """
    use_cache = dyn is not None
    if dyn is None:
        T_legacy, metric_legacy = T, metric_fn
        T = lambda s, k, d: T_legacy(s, k)  # noqa: E731
        metric_fn = lambda s, d: metric_legacy(s)  # noqa: E731
        dyn = {"tol": budget.tol}
        if halpern:
            mode = "halpern"
            dyn["restart_k"] = budget.max_iters + 2
            dyn["rho"] = 1.0 if rho is None else float(rho)
        else:
            mode = "km"
            dyn["rho"] = 1.0 if rho is None else float(rho)
    if mode not in ("km", "halpern", "haugazeau"):
        raise ValueError(f"unknown mode {mode!r}")
    state0 = _asarray_tree(state0)
    dyn = _asarray_tree(dyn)
    n_samples = budget.max_iters // budget.sample_every + 1
    max_iters = budget.max_iters
    sample_every = budget.sample_every

    def run(state0_, dyn_):
        tol = dyn_["tol"]

        def cond(carry):
            k, metric = carry[1], carry[3]
            return jnp.logical_and(
                k < max_iters, jnp.logical_or(k == 0, metric > tol)
            )

        if mode == "km":

            def body(carry):
                state, k, _, _, res_h, met_h = carry
                Tx = T(state, k, dyn_)
                rho = dyn_["rho"]
                new_state = jax.tree_util.tree_map(
                    lambda s, t: (1.0 - rho) * s + rho * t, state, Tx
                )
                res = fp_residual(new_state, state)
                metric = metric_fn(new_state, dyn_)
                idx = k // sample_every
                res_h = res_h.at[idx].set(res)
                met_h = met_h.at[idx].set(metric)
                return new_state, k + 1, res, metric, res_h, met_h

            carry0 = (
                state0_,
                jnp.asarray(0),
                jnp.asarray(jnp.inf, dtype=jnp.float64),
                jnp.asarray(jnp.inf, dtype=jnp.float64),
                jnp.full((n_samples,), jnp.nan, dtype=jnp.float64),
                jnp.full((n_samples,), jnp.nan, dtype=jnp.float64),
            )
            out = jax.lax.while_loop(cond, body, carry0)
            state, k, _, _, res_h, met_h = out
            return state, k, res_h, met_h

        if mode == "haugazeau":
            # x_{k+1} = Q(x_0, x_k, F x_k), F = I + (T - I)/(2 alpha), with Q
            # evaluated in the scheme's averagedness metric (`gram`; see
            # atlas.typing_.rules.typecheck_haugazeau). Carry holds the anchor.
            gram_fn = euclidean_gram if gram is None else gram

            def body_q(carry):
                state, k, _, _, res_h, met_h, anchor = carry
                Tx = T(state, k, dyn_)
                w = 0.5 / dyn_["haug_alpha"]
                Fx = jax.tree_util.tree_map(lambda s, t: s + w * (t - s), state, Tx)
                p = jax.tree_util.tree_map(jnp.subtract, anchor, state)
                q = jax.tree_util.tree_map(jnp.subtract, state, Fx)
                ca, cb, cc, _ = haugazeau_coefficients(*gram_fn(p, q, dyn_))
                new_state = jax.tree_util.tree_map(
                    lambda a, b, c: ca * a + cb * b + cc * c, anchor, state, Fx)
                res = fp_residual(new_state, state)
                metric = metric_fn(new_state, dyn_)
                idx = k // sample_every
                res_h = res_h.at[idx].set(res)
                met_h = met_h.at[idx].set(metric)
                return new_state, k + 1, res, metric, res_h, met_h, anchor

            carry0 = (
                state0_,
                jnp.asarray(0),
                jnp.asarray(jnp.inf, dtype=jnp.float64),
                jnp.asarray(jnp.inf, dtype=jnp.float64),
                jnp.full((n_samples,), jnp.nan, dtype=jnp.float64),
                jnp.full((n_samples,), jnp.nan, dtype=jnp.float64),
                state0_,
            )
            out = jax.lax.while_loop(cond, body_q, carry0)
            state, k, _, _, res_h, met_h = out[:6]
            return state, k, res_h, met_h

        # mode == "halpern": carry additionally holds (anchor, j_since)
        def cond_h(carry):
            k, metric = carry[1], carry[3]
            return jnp.logical_and(
                k < max_iters, jnp.logical_or(k == 0, metric > tol)
            )

        def body_h(carry):
            state, k, _, _, res_h, met_h, anchor, j = carry
            Tx = T(state, k, dyn_)
            # RELAX THEN ANCHOR (2026-09-05). The anchored step is applied to
            # F_rho = (1-rho) I + rho T, not to T. rho = 1 recovers the plain
            # anchored iteration exactly, so this is backward compatible.
            # Before this, "halpern" mode silently DROPPED rho: the gate
            # certified relax-then-anchor while the executed iteration was
            # anchor-only, so rho=2 (the reflection) and rho=1 produced
            # bit-identical trajectories. That is the certificate describing
            # a different iteration than the one that runs.
            rho_h = dyn_["rho"]
            Fx = jax.tree_util.tree_map(
                lambda s, t: (1.0 - rho_h) * s + rho_h * t, state, Tx
            )
            lam_j = 1.0 / (j.astype(jnp.float64) + 2.0)
            new_state = jax.tree_util.tree_map(
                lambda a, t: lam_j * a + (1.0 - lam_j) * t, anchor, Fx
            )
            j_next = j + 1
            do_restart = j_next >= dyn_["restart_k"]
            anchor = jax.tree_util.tree_map(
                lambda a, n: jnp.where(do_restart, n, a), anchor, new_state
            )
            j_next = jnp.where(do_restart, 0, j_next)
            res = fp_residual(new_state, state)
            metric = metric_fn(new_state, dyn_)
            idx = k // sample_every
            res_h = res_h.at[idx].set(res)
            met_h = met_h.at[idx].set(metric)
            return new_state, k + 1, res, metric, res_h, met_h, anchor, j_next

        carry0 = (
            state0_,
            jnp.asarray(0),
            jnp.asarray(jnp.inf, dtype=jnp.float64),
            jnp.asarray(jnp.inf, dtype=jnp.float64),
            jnp.full((n_samples,), jnp.nan, dtype=jnp.float64),
            jnp.full((n_samples,), jnp.nan, dtype=jnp.float64),
            state0_,
            jnp.asarray(0),
        )
        out = jax.lax.while_loop(cond_h, body_h, carry0)
        state, k, _, _, res_h, met_h = out[:6]
        return state, k, res_h, met_h

    if use_cache:
        key = (
            "km",
            mode,
            static_key,
            max_iters,
            sample_every,
            _abstract(state0),
            _abstract(dyn),
        )
        compiled = _compile_cached(key, run, state0, dyn)
    else:  # legacy path: closures carry constants; compile per call
        compiled = jax.jit(run).lower(state0, dyn).compile()

    t0 = time.perf_counter()
    state, k, res_h, met_h = compiled(state0, dyn)
    jax.block_until_ready((state, k, res_h, met_h))
    wall_time_s = time.perf_counter() - t0

    iters = int(k)
    n_written = max((iters - 1) // sample_every + 1, 1) if iters > 0 else 1
    return (
        state,
        iters,
        np.asarray(res_h[:n_written]),
        np.asarray(met_h[:n_written]),
        wall_time_s,
    )
