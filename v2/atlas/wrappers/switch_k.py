"""Finite-prefix runtime for the switch_k gene (the prefix half only).

switch_k semantics: z_{k+1} = T_aggressive(z_k) for k < K, then the
certified tail runs from wherever the prefix left off. The tail is an
ordinary certified KM iteration executed by atlas.wrappers.km.km_run;
this module runs ONLY the aggressive prefix, with the non-finiteness
guard the typing rule promises (atlas.typing_.typecheck_switch_k):

- The prefix tracks the LAST FINITE state, recorded at a cheap periodic
  check every CHECK_EVERY iterations (the initial point counts as the
  first finite record, so there is always a finite handoff candidate).
- If a periodic check finds a non-finite state, the prefix ABORTS (NaNs
  do not recover under these iterations) and reports where.
- At the handoff (prefix exhausted or aborted) the final state is checked
  once more; if it is non-finite the handoff state is the last finite
  state instead (the initial point if none was recorded), so the
  certified tail always restarts from a finite point. Whether this reset
  fired, and at which iteration, is returned for result.extra.

The aggressive map is EXEMPT from convergence side conditions by design;
this guard is what keeps the exemption harmless: the worst a divergent
prefix can do is waste K iterations.

Compiled once per (static_key, CHECK_EVERY, shapes) exactly like km_run;
K is a DYNAMIC argument (dyn["K"]), so every prefix length reuses the
same executable.
"""
from __future__ import annotations

import time

import jax
import jax.numpy as jnp

from atlas.wrappers.km import _abstract, _asarray_tree, _compile_cached

CHECK_EVERY = 25  # cheap periodic finiteness-check cadence (static)


def _tree_finite(tree):
    """Guard notion of "finite": the state's squared norm is finite in f64.

    This implies every entry is finite (NaN/inf propagate through the
    sum) AND the norm is below the f64 overflow threshold (~1e154).
    A pre-overflow state of magnitude 1e200 passes an entrywise isfinite
    test but is unusable by every downstream f64 computation (the duality
    gap is quadratic in the state), so the guard counts it as non-finite;
    a violently divergent prefix therefore falls back to the initial
    point instead of handing the tail an astronomically large "finite"
    state."""
    leaves = jax.tree_util.tree_leaves(tree)
    sq = sum(jnp.sum(jnp.square(leaf)) for leaf in leaves)
    return jnp.isfinite(sq)


def switch_k_prefix(T_agg, state0, K: int, dyn_agg, *, static_key=(),
                    check_every: int = CHECK_EVERY):
    """Run K iterations of the (exempt) aggressive map with the guard.

    T_agg: (state, k, dyn) -> state. K: prefix length (dynamic argument).
    dyn_agg: dynamic-input pytree of the aggressive map.

    Returns (handoff_state, prefix_iters, reset_fired, reset_iter,
    handoff_from_iter, wall_time_s):
    - handoff_state: finite state the tail starts from.
    - prefix_iters: aggressive iterations actually run (< K on abort).
    - reset_fired (bool): the non-finite guard replaced the state.
    - reset_iter: iteration at which non-finiteness was detected (equals
      prefix_iters when only the final handoff check caught it); only
      meaningful when reset_fired.
    - handoff_from_iter: the prefix iteration the handoff state was
      recorded at (0 = the initial point).
    """
    if K < 1:
        raise ValueError("switch_k prefix needs K >= 1")
    state0 = _asarray_tree(state0)
    dyn = {"agg": _asarray_tree(dyn_agg),
           "K": jnp.asarray(int(K), dtype=jnp.int64)}

    def run(state0_, dyn_):
        Kv = dyn_["K"]

        def cond(carry):
            _state, k, _lf, _lfk, dead, _dk = carry
            return jnp.logical_and(k < Kv, jnp.logical_not(dead))

        def body(carry):
            state, k, lf, lfk, dead, dk = carry
            new = T_agg(state, k, dyn_["agg"])
            k1 = k + 1
            is_check = jnp.logical_or(
                jnp.mod(k1, check_every) == 0, k1 == Kv
            )
            finite = _tree_finite(new)
            record = jnp.logical_and(is_check, finite)
            kill = jnp.logical_and(is_check, jnp.logical_not(finite))
            lf = jax.tree_util.tree_map(
                lambda a, b: jnp.where(record, b, a), lf, new
            )
            lfk = jnp.where(record, k1, lfk)
            dk = jnp.where(kill, k1, dk)
            dead = jnp.logical_or(dead, kill)
            return new, k1, lf, lfk, dead, dk

        zero = jnp.asarray(0, dtype=jnp.int64)
        carry0 = (state0_, zero, state0_, zero, jnp.asarray(False), zero)
        state, k, lf, lfk, dead, dk = jax.lax.while_loop(cond, body, carry0)
        # Handoff check: covers non-multiples of check_every and the abort.
        final_finite = _tree_finite(state)
        use_fallback = jnp.logical_or(dead, jnp.logical_not(final_finite))
        handoff = jax.tree_util.tree_map(
            lambda a, b: jnp.where(use_fallback, a, b), lf, state
        )
        handoff_iter = jnp.where(use_fallback, lfk, k)
        reset_iter = jnp.where(dead, dk, k)
        return handoff, k, use_fallback, reset_iter, handoff_iter

    key = (
        "switch_k_prefix",
        static_key,
        int(check_every),
        _abstract(state0),
        _abstract(dyn),
    )
    compiled = _compile_cached(key, run, state0, dyn)

    t0 = time.perf_counter()
    handoff, k, reset, reset_iter, handoff_iter = compiled(state0, dyn)
    jax.block_until_ready((handoff, k, reset, reset_iter, handoff_iter))
    wall_time_s = time.perf_counter() - t0
    return (
        handoff,
        int(k),
        bool(reset),
        int(reset_iter),
        int(handoff_iter),
        wall_time_s,
    )
