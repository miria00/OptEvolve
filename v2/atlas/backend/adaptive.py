"""Certified adaptive step ratio for PDHG / Condat-Vu (a variable metric schedule).

The iteration runs in chunks of K steps with the certificate evaluated after
each chunk, the same cadence as the policy runner. Every `epoch_chunks`
chunks the controller may move the primal weight omega = sqrt(s/t):

  rule "fixed": never (the comparison arm, same runner, same execution);
  rule "pdlp":  omega <- exp(smooth*log(|dy|/|dx|) + (1-smooth)*log(omega)),
                |dx|, |dy| the movement since the last update (PDLP's
                primal-weight update, Applegate et al. 2021).

(t, s) is always the point with ratio omega on the admissible boundary
t L_f/2 + t s ||L||^2 = fill, and a move is accepted only while the variable
metric budget allows it (atlas.typing_.rules.typecheck_metric_schedule).
Wall time counts the whole host loop, controller included, and excludes the
one AOT compile, which is reported separately.
"""
from __future__ import annotations

import math
import time

import jax
import jax.numpy as jnp
import numpy as np

from atlas.typing_ import TypeCheckError

RULES = ("fixed", "pdlp", "diag")


def boundary_steps(omega: float, opnorm: float, L_f: float, fill: float) -> tuple[float, float]:
    """(t, s) with s/t = omega^2 and t L_f/2 + t s opnorm^2 = fill."""
    a, b = (opnorm * omega) ** 2, L_f / 2.0
    t = (-b + math.sqrt(b * b + 4.0 * a * fill)) / (2.0 * a) if a > 0 else fill / b
    return t, omega * omega * t


def ratio_change_cost(t0: float, s0: float, t1: float, s1: float, opnorm: float) -> float:
    """eta with M_{n+1} <= (1 + eta) M_n for the saddle metric (see the gate rule)."""
    theta = math.sqrt(t0 * s0) * opnorm
    if not theta < 1.0:
        raise TypeCheckError(f"saddle metric not positive definite: theta = {theta:.6g}")
    return max(0.0, t0 / t1 - 1.0, s0 / s1 - 1.0) / (1.0 - theta)


def _diag_theta(L, T, S, iters: int = 12, seed: int = 0):
    """theta = ||Sigma^{1/2} L T^{1/2}|| by power iteration on the rescaled operator, with a safety factor."""
    import numpy as _np
    rng = _np.random.default_rng(seed)
    v = rng.standard_normal(int(_np.size(T)))
    sT, sS = _np.sqrt(_np.asarray(T)), _np.sqrt(_np.asarray(S))
    lam = 0.0
    v = v / max(float(_np.linalg.norm(v)), 1e-300)
    for _ in range(iters):
        # z = (Sigma^{1/2} L T^{1/2})^T (Sigma^{1/2} L T^{1/2}) v, so ||z|| -> lambda_max of the Gram operator
        w = sS * _np.asarray(L.forward(jnp.asarray(sT * v)))
        z = sT * _np.asarray(L.adjoint(jnp.asarray(sS * w)))
        lam = float(_np.linalg.norm(z))
        if not _np.isfinite(lam) or lam == 0.0:
            return float("inf")
        v = z / lam
    return math.sqrt(lam) * 1.02          # theta is the operator norm: the square root of lambda_max


def _diag_update(dx, dy, T, S, Lf, opnorm, fill, smooth, lo_hi, L):
    """Propose new per-coordinate steps from the movement of each coordinate over an epoch.

    The scalar PDLP rule moves the single weight omega towards ||dy|| / ||dx||; per coordinate the same idea
    scales each entry by its own movement relative to the epoch's mean movement, which is the diagonal analogue,
    then rescales globally so the proposal sits on the gate boundary.
    """
    import numpy as _np
    # Per-coordinate factors, normalised by their GEOMETRIC mean so the proposal is scale free, and the RATIO is
    # clamped before it is applied. Clamping the result instead let a coordinate that barely moved take a factor
    # of up to 1e8 (epoch movement is very skewed: mean |dx| 0.128 against max 4.7), and sqrt of such a step
    # overflowed the Gram product, so theta came back infinite and every proposal was rejected.
    ax, ay = _np.abs(_np.asarray(dx)), _np.abs(_np.asarray(dy))
    floor_x = max(float(_np.median(ax)) * 1e-3, 1e-300)
    floor_y = max(float(_np.median(ay)) * 1e-3, 1e-300)
    ax, ay = _np.maximum(ax, floor_x), _np.maximum(ay, floor_y)
    gx = _np.exp(smooth * (_np.log(ax) - _np.mean(_np.log(ax))))
    gy = _np.exp(smooth * (_np.log(ay) - _np.mean(_np.log(ay))))
    MAX_STEP_RATIO = 4.0                                  # one epoch may move any entry by at most this factor
    gx = _np.clip(gx, 1.0 / MAX_STEP_RATIO, MAX_STEP_RATIO)
    gy = _np.clip(gy, 1.0 / MAX_STEP_RATIO, MAX_STEP_RATIO)
    Tp = _np.asarray(T) / gx                              # a coordinate that moves a lot gets a smaller step
    Sp = _np.asarray(S) / gy
    (tlo, thi), (slo, shi) = lo_hi
    Tp = _np.clip(Tp, tlo, thi); Sp = _np.clip(Sp, slo, shi)
    if not (_np.all(_np.isfinite(Tp)) and _np.all(_np.isfinite(Sp)) and Tp.min() > 0.0 and Sp.min() > 0.0):
        return None, None, float("inf")
    # The Condat-Vu gate is max_i T_i L_f / 2 + theta^2 <= fill. Enforce it BY CONSTRUCTION rather than
    # rejecting afterwards: first shrink the primal steps so the smooth term takes at most half the budget
    # (spreading the entries can push max(T) above t0, which previously made room negative and rejected every
    # proposal), then set the global scale from theta so the coupling term fits the rest.
    SMOOTH_SHARE = 0.5
    if Lf > 0.0:
        cap = SMOOTH_SHARE * fill * 2.0 / Lf
        if float(_np.max(Tp)) > cap:
            Tp = Tp * (cap / float(_np.max(Tp)))
    room = fill - float(_np.max(Tp)) * Lf / 2.0
    if room <= 0.0:
        return None, None, float("inf")
    theta = _diag_theta(L, Tp, Sp)
    if not _np.isfinite(theta) or theta <= 0.0:
        return None, None, float("inf")
    if theta * theta > room:                              # scale the pair onto the gate boundary
        scale = (room ** 0.5) / theta
        Tp, Sp, theta = Tp * scale, Sp * scale, theta * scale
    return Tp, Sp, theta


_CHUNK_CACHE: dict = {}


def run_adaptive(built, cert, *, scheme: str, tol: float, max_iters: int, K: int = 64,
                 rule: str = "pdlp", epoch_chunks: int = 8, smooth: float = 0.5,
                 eta_budget: float = 1e4, omega_range: float = 1e4, fill: float = 0.98,
                 max_updates=None, omega0=None, linop=None) -> dict:
    """`linop` is the linear map the iteration actually applies, required by rule="diag": BuiltScheme does not
    carry it (its builder closes over it), and under a metric gene the operator is the RESCALED one, so the
    caller must pass the same L whose norm the gate is checked against."""
    from atlas.typing_.rules import typecheck_metric_schedule

    if rule not in RULES:
        raise ValueError(f"rule must be one of {RULES}")
    if rule == "diag":
        from atlas.typing_.rules import typecheck_diag_metric_schedule
        cert = typecheck_diag_metric_schedule(cert, scheme, eta_budget, omega_range, fill, max_updates)
    else:
        cert = typecheck_metric_schedule(cert, scheme, eta_budget, omega_range, fill, max_updates)
    p = cert.params
    opnorm, L_f = float(p["opnorm"]), float(p.get("L_smooth", 0.0))
    omega = float(omega0) if omega0 is not None else math.sqrt(float(p["s"]) / float(p["t"]))
    lo, hi = omega / omega_range, omega * omega_range
    t, s = boundary_steps(omega, opnorm, L_f, fill)
    dyn = dict(built.dyn)
    dyn["t"], dyn["s"] = jnp.asarray(t, dtype=jnp.float64), jnp.asarray(s, dtype=jnp.float64)
    Td = Sd = None
    bounds = ((0.0, float("inf")), (0.0, float("inf")))
    theta_cur = math.sqrt(t * s) * opnorm
    if rule == "diag":
        if linop is None:
            raise ValueError('rule="diag" needs linop=<the (rescaled) problem L the iteration applies>')
        n_prim = int(np.size(np.asarray(jax.tree_util.tree_leaves(built.state0)[0])))
        n_dual = int(np.size(np.asarray(jax.tree_util.tree_leaves(built.state0)[1])))
        Td, Sd = np.full(n_prim, t), np.full(n_dual, s)
        bounds = ((t / omega_range, t * omega_range), (s / omega_range, s * omega_range))
        dyn["t"], dyn["s"] = jnp.asarray(Td, dtype=jnp.float64), jnp.asarray(Sd, dtype=jnp.float64)

    def chunk(state, dyn_):
        state = jax.lax.fori_loop(0, K, lambda k, z: built.T(z, k, dyn_), state)
        return state, built.metric(state, dyn_)

    state = jax.tree_util.tree_map(jnp.asarray, built.state0)
    # COMPILE CACHE. `chunk` is a fresh closure on every call, so jax.jit sees a new function object
    # each time and misses its own cache even though the computation is byte-identical. A tolerance
    # ladder calls run_adaptive several times on the SAME `built` and paid full compilation on every
    # rung: measured on huber n_var 140,000, 1.011 s then 0.938 s, which was 51% of that solve's
    # total wall clock. `tol` never enters the compiled function (it is tested in Python on the
    # returned metric), so rungs differ in no way that XLA can see.
    # Keyed on id(built) with the object held alive in the value, so an id cannot be recycled onto a
    # different scheme. A new `built` (one per repetition, from prepare_base) gets a new entry, which
    # preserves the harnesses' cold-compile-per-repetition policy.
    ck = (id(built), int(K), scheme, rule)
    hit = _CHUNK_CACHE.get(ck)
    if hit is not None and hit[0] is built:
        exe, compile_s = hit[1], 0.0
    else:
        c0 = time.perf_counter()
        exe = jax.jit(chunk).lower(state, dyn).compile()
        compile_s = time.perf_counter() - c0
        # BOUND AT 2 ENTRIES. The value holds `built` alive to make the id check sound, and `built`
        # owns GPU-resident operators, so a large cache PINS GPU MEMORY for every problem instance it
        # has seen. A new `built` is created per repetition, so an earlier 64-entry bound retained
        # dozens of instances and produced CUDA_ERROR_OUT_OF_MEMORY on svm at n_var 110,000 on the
        # 4090 (which shares its 24 GB with a resident vLLM). The ladder only ever needs the entry
        # for the CURRENT `built`, so two is already one more than required.
        while len(_CHUNK_CACHE) >= 2:
            _CHUNK_CACHE.pop(next(iter(_CHUNK_CACHE)))
        _CHUNK_CACHE[ck] = (built, exe)

    iters, chunks, spent, n_updates, frozen = 0, 0, 0.0, 0, False
    history = [(0, omega, t, s)]
    mark = state
    w0 = time.perf_counter()
    while True:
        state, m = exe(state, dyn)
        m = float(m)
        iters += K
        chunks += 1
        if m <= tol or iters >= max_iters:
            break
        if rule == "diag" and not frozen and chunks % epoch_chunks == 0:
            from atlas.typing_.rules import diag_change_cost
            dx = np.asarray(state[0]) - np.asarray(mark[0])
            dy = np.asarray(state[1]) - np.asarray(mark[1])
            mark = state
            Tp, Sp, theta_new = _diag_update(dx, dy, Td, Sd, L_f, opnorm, fill, smooth, bounds, linop)
            if Tp is not None and theta_new < 1.0:
                cost = diag_change_cost(Td, Sd, Tp, Sp, theta_cur)
                if spent + cost <= eta_budget and (max_updates is None or n_updates < max_updates):
                    Td, Sd, theta_cur = Tp, Sp, theta_new
                    spent, n_updates = spent + cost, n_updates + 1
                    dyn["t"] = jnp.asarray(Td, dtype=jnp.float64)
                    dyn["s"] = jnp.asarray(Sd, dtype=jnp.float64)
                    history.append((iters, float(np.max(Sd) / np.max(Td)) ** 0.5, float(np.mean(Td)), float(np.mean(Sd))))
                elif spent + cost > eta_budget or (max_updates is not None and n_updates >= max_updates):
                    frozen = True                        # the budget is spent: no further change is certified
            # a proposal rejected by the gate is simply skipped; the next epoch may propose an admissible one
        elif rule == "pdlp" and not frozen and chunks % epoch_chunks == 0:
            dx = float(jnp.linalg.norm(state[0] - mark[0]))
            dy = float(jnp.linalg.norm(state[1] - mark[1]))
            mark = state
            if dx > 0.0 and dy > 0.0 and math.isfinite(dx) and math.isfinite(dy):
                w = math.exp(smooth * math.log(dy / dx) + (1.0 - smooth) * math.log(omega))
                w = min(max(w, lo), hi)
                t1, s1 = boundary_steps(w, opnorm, L_f, fill)
                cost = ratio_change_cost(t, s, t1, s1, opnorm)
                if spent + cost <= eta_budget and (max_updates is None or n_updates < max_updates):
                    omega, t, s, spent, n_updates = w, t1, s1, spent + cost, n_updates + 1
                    dyn["t"], dyn["s"] = jnp.asarray(t, dtype=jnp.float64), jnp.asarray(s, dtype=jnp.float64)
                    history.append((iters, omega, t, s))
                else:
                    frozen = True
    jax.block_until_ready(state)
    wall = time.perf_counter() - w0
    x, u = built.extract(state, dyn)
    return {"x": np.asarray(x), "u": None if u is None else np.asarray(u), "iters": iters,
            "wall_s": wall, "compile_s": compile_s, "metric": m, "converged": m <= tol,
            "omega_history": history, "eta_spent": spent, "n_updates": n_updates,
            "frozen": frozen, "cert": cert}
