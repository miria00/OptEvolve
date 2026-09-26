"""Safeguarded switching runtime (the certified controller wrapper).

Two fixed-point maps on the SAME state space:
- T_agg: the AGGRESSIVE map. May violate every side condition (that is the
  design); it only has to be finite arithmetic.
- T_fb:  the FALLBACK map. MUST carry a Level-1 construction certificate
  (enforced by the Stage-0 gate before this runtime is ever built).

Per iteration k:
  1. candidate = T_agg(state);  gap = metric(candidate)  (metric is the
     RIGOROUS relative duality gap, the Level-3 runtime certificate).
  2. ACCEPT the candidate iff gap < the MINIMUM certified gap over the
     last `window` certified evaluations (for k < window the buffer is
     filled with the starting gap). Otherwise take the certified fallback
     step from the CURRENT state (= the last accepted state; rejected
     aggressive candidates are never kept, so the iterate sequence only
     ever leaves the fallback trajectory through certified strict merit
     decrease below the recent best).
Convergence is inherited from the fallback KM iteration; the monitor makes
divergent aggressive maps harmless (every step rejected => pure fallback).

Compiled once per (static_key, window, shapes) exactly like km_run; the
window ring buffer's size makes `window` static.
"""
from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np

from atlas.certify.residuals import fp_residual
from atlas.wrappers.km import Budget, _abstract, _asarray_tree, _compile_cached


def switch_run(T_agg, T_fb, state0, budget: Budget, metric_fn, dyn, *,
               window: int, static_key=()):
    """Run the safeguarded switch loop until metric <= tol or budget out.

    T_agg/T_fb: (state, k, dyn) -> state; metric_fn: (state, dyn) -> gap.
    dyn must contain "tol". Returns (state, iters, res_history,
    metric_history, wall_time_s, n_accepted, n_rejected).
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    state0 = _asarray_tree(state0)
    dyn = _asarray_tree(dyn)
    n_samples = budget.max_iters // budget.sample_every + 1
    max_iters = budget.max_iters
    sample_every = budget.sample_every

    def run(state0_, dyn_):
        tol = dyn_["tol"]
        gap0 = metric_fn(state0_, dyn_)
        buf0 = jnp.full((window,), jnp.inf, dtype=jnp.float64).at[:].set(gap0)

        def cond(carry):
            k, metric = carry[1], carry[3]
            return jnp.logical_and(
                k < max_iters, jnp.logical_or(k == 0, metric > tol)
            )

        def body(carry):
            state, k, _, _, res_h, met_h, buf, n_acc, n_rej = carry
            candidate = T_agg(state, k, dyn_)
            gap_agg = metric_fn(candidate, dyn_)
            idx = jnp.mod(k, window)
            # Reference = the BEST certified gap over the last `window`
            # certified evaluations, not the single slot from exactly
            # `window` ago: the single-slot form ratchets (one accepted
            # expansive step inflates the fallback gaps written into the
            # buffer, and later aggressive candidates keep "improving" on
            # those polluted references all the way to overflow, observed
            # on the LP cell). min(buf) makes accepted gaps quasi-monotone
            # within window spans.
            gap_ref = jnp.min(buf)
            accept = jnp.logical_and(jnp.isfinite(gap_agg), gap_agg < gap_ref)

            def take_aggressive():
                return candidate, gap_agg

            def take_fallback():
                s = T_fb(state, k, dyn_)
                return s, metric_fn(s, dyn_)

            new_state, new_gap = jax.lax.cond(
                accept, take_aggressive, take_fallback
            )
            buf = buf.at[idx].set(new_gap)
            res = fp_residual(new_state, state)
            sidx = k // sample_every
            res_h = res_h.at[sidx].set(res)
            met_h = met_h.at[sidx].set(new_gap)
            n_acc = n_acc + accept.astype(jnp.int64)
            n_rej = n_rej + (1 - accept.astype(jnp.int64))
            return new_state, k + 1, res, new_gap, res_h, met_h, buf, n_acc, n_rej

        carry0 = (
            state0_,
            jnp.asarray(0),
            jnp.asarray(jnp.inf, dtype=jnp.float64),
            jnp.asarray(jnp.inf, dtype=jnp.float64),
            jnp.full((n_samples,), jnp.nan, dtype=jnp.float64),
            jnp.full((n_samples,), jnp.nan, dtype=jnp.float64),
            buf0,
            jnp.asarray(0, dtype=jnp.int64),
            jnp.asarray(0, dtype=jnp.int64),
        )
        out = jax.lax.while_loop(cond, body, carry0)
        state, k, _, _, res_h, met_h, _, n_acc, n_rej = out
        return state, k, res_h, met_h, n_acc, n_rej

    key = (
        "switch",
        static_key,
        window,
        max_iters,
        sample_every,
        _abstract(state0),
        _abstract(dyn),
    )
    compiled = _compile_cached(key, run, state0, dyn)

    t0 = time.perf_counter()
    state, k, res_h, met_h, n_acc, n_rej = compiled(state0, dyn)
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
        int(n_acc),
        int(n_rej),
    )
