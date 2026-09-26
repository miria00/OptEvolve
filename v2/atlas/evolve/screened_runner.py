"""Certificate-driven structure reduction for elastic-net logistic regression: FISTA or
forward-backward with Gap Safe screening and mid-solve re-planning.

At every certificate check (every K steps) the FULL problem's duality gap is computed
(atlas.certify.residuals.logreg_enet_gap_parts on all columns) and Gap Safe screening
(logreg_enet_gap_safe_screen) marks features that are provably zero at every optimum.
When the surviving set is small enough, the problem is rebuilt on those columns and the
solve continues. Screened features are zero at the optimum, so the reduced problem has the
same solution; the step 1/L of the full problem stays valid because ||X_A|| <= ||X||.
Termination and the reported certificate are always the full-problem gap, so neither
re-planning option below can affect correctness, only speed.

Re-planning genes:
  momentum = "reset"  restart FISTA at the restricted iterate (t = 1);
             "keep"   restrict x and y to the surviving columns and keep t (the gradient
                      restart test still guards the momentum).
  working_set = True  (celer / skglm style; implies buckets) start on the bucket of features with the
                      largest dual scores |x_j^T u| at w = 0 instead of on all p; at every check add
                      features outside the set that violate the optimality condition |x_j^T u| <= lam alpha
                      (at most doubling the set), and drop Gap Safe screened ones for good. Unsafe as a
                      reduction, safe as a solver: termination is still the full-problem gap.
  buckets  = False    rebuild at the exact surviving size when it has shrunk to `shrink`;
             True     pad to the smallest bucket p, p/2, p/4, ... that holds the survivors
                      (padded columns are zero and their coefficients stay zero), re-plan
                      only when the bucket drops, and compile every bucket up front.

Step-size gene (step_rule). FISTA and forward-backward need a Lipschitz constant L of the gradient:
  "precomputed"  L = lipschitz_bound(X) (100 host power iterations) computed OUTSIDE the timed region;
                 a reference only, since the baselines pay for all of their setup;
  "power"        30 power iterations on the device on the current columns, inside the timed region,
                 repeated at every re-plan (safety factor 1.02, as lipschitz_bound);
  "gram"         the exact largest eigenvalue of X_W^T X_W / (4n) on the current columns, inside the
                 timed region; cheap on a small working set, and never below the true constant.
A re-plan that increases the step (smaller L on fewer columns) resets FISTA's momentum.

Time accounting: `wall_s` is the whole host loop (steps, certificate checks, screening, column
gathers). Compilation that depends only on (n, p) (the full width and, with buckets, every bucket
shape) runs before the timed region, like the baselines' untimed warm-up; compilation for
data-dependent shapes (exact-size re-plans without buckets) runs inside it and IS in `wall_s`.
`compile_s` totals all compilation; `wall_incl_compile_s` = wall_s + the ahead-of-time part.
"""
from __future__ import annotations

import time
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from atlas.certify.residuals import logreg_enet_gap_parts, logreg_enet_gap_safe_screen


def _bucket_sizes(p: int, min_cols: int = 16):
    sizes, s = [], p
    while s >= min_cols:
        sizes.append(s); s = -(-s // 2)
        if sizes[-1] == s:
            break
    return sizes


# Compiled executables, reused ACROSS calls. Until 2026-09-13 the cache below was a LOCAL dict inside
# run_screened, so every call re-traced and re-compiled every bucket program (~3 s on epsilon). That cost
# is paid on each point of a regularisation path and is excluded from wall_s, which made the reported
# "warm" time unobtainable by any caller. JAX's persistent cache cannot help: it covers the backend
# compile, not the tracing and lowering, and lam/alpha/y are baked into the HLO so every lam is a miss.
#
# THE KEY MUST CARRY EVERYTHING THE HLO DEPENDS ON. step() closes over the LABELS (yd) as well as lam and
# alpha, so a key on shapes alone would hand back an executable holding another dataset's labels: two
# datasets with the same n would silently collide. y is therefore fingerprinted.
_EXE_CACHE: dict = {}
_EXE_CACHE_MAX = 8                      # each entry holds every bucket program for one signature


def _exe_cache_for(sig):
    """The executable dict for this problem signature, evicting oldest-first when full."""
    import os as _os
    if _os.environ.get("ATLAS_SCREENED_EXE_CACHE") == "0":
        return {}                       # disabled: every call compiles, the pre-2026-09-13 behaviour
    d = _EXE_CACHE.get(sig)
    if d is None:
        while len(_EXE_CACHE) >= _EXE_CACHE_MAX:
            _EXE_CACHE.pop(next(iter(_EXE_CACHE)))
        d = _EXE_CACHE[sig] = {}
    return d


def run_screened(X, y, lam: float, alpha: float, *, base: str = "fista", tol: float = 1e-6,
                 max_evals: int = 200000, K: int = 64, shrink: float = 0.8, screen: bool = True,
                 momentum: str = "reset", buckets: bool = False, working_set: bool = False,
                 ws_init: float = 0.125, step_rule: str = "precomputed", L: float | None = None,
                 power_iters: int = 30, dtype=jnp.float64):
    from atlas.bench import logreg_cell as lc
    assert momentum in ("reset", "keep") and base in ("fista", "fbs") and step_rule in ("precomputed", "power", "gram")
    if working_set:
        screen, buckets = True, True
    Xf = jnp.asarray(X, dtype=jnp.float64)
    yf = jnp.asarray(y, dtype=jnp.float64)
    colnorm = jnp.linalg.norm(Xf, axis=0)                     # once: screening reuses it at every check
    n, p = Xf.shape
    g = lc.ElasticNet(lam=jnp.asarray(lam, dtype=jnp.float64), alpha=jnp.asarray(alpha, dtype=jnp.float64))
    L_pre = None
    if step_rule == "precomputed":
        L_pre = float(L) if L is not None else lc.lipschitz_bound(np.asarray(X, dtype=np.float64))

    yd = yf.astype(dtype)                                     # labels in the iterate dtype: a float64 y would
                                                              # promote float32 iterates and break the loop carry
    def step(state, XA, a, lm, al):
        x, yv, t = state
        z = yd * (XA @ yv)
        r = yd * jax.nn.sigmoid(-z)
        # The transposed product is where float32 collapses on this device. Measured on the 4090
        # (100000x2000): XLA's f32 X^T v runs at 298 GB/s against 896 for f64 on HALF the bytes, so in
        # wall time f32 is SLOWER than f64 (2.591 ms against 1.717 ms). It is not a layout problem:
        # v @ X, einsum and a materialised contiguous transpose all give the same 309 GB/s. The cost is
        # the accumulation over the length-n reduction axis. Reducing an elementwise product over rows
        # instead is fused by XLA, materialises no n-by-p temporary (0.858 ms for 0.8 GB; materialising
        # would cost ~2.4 ms), and reaches 932 GB/s, 3.0x the direct form, with relative error 1.2e-7
        # against the direct form's 2.9e-4. float64 is already at 90% of peak, so it keeps the direct
        # form. The default f32 path was Pareto-dominated: slower AND no more accurate.
        gsum = jnp.sum(XA * r[:, None], axis=0) if jnp.dtype(dtype) == jnp.float32 else XA.T @ r
        grad = -gsum / n
        v = yv - a * grad
        xn = jnp.sign(v) * jnp.maximum(jnp.abs(v) - a * lm * al, 0.0) / (1.0 + a * lm * (1.0 - al))
        if base == "fbs":
            return (xn, xn, t)
        tn = 0.5 * (1.0 + jnp.sqrt(1.0 + 4.0 * t * t))
        yn = xn + ((t - 1.0) / tn) * (xn - x)
        bad = jnp.vdot(yv - xn, xn - x) > 0.0
        return (xn, jnp.where(bad, xn, yn), jnp.where(bad, 1.0, tn))

    def chunk(state, XA, a, lm, al):
        return jax.lax.fori_loop(0, K, lambda _, s: step(s, XA, a, lm, al), state)

    def certify_(w_full, Xc, yc, cn, lm, al):
        # X and y are arguments, not closures: a closed-over X is baked into the executable as a
        # constant (1.6 GB on epsilon), which exhausted GPU memory next to vLLM. lam and alpha are
        # arguments for the same reason: as constants they made every penalty weight a fresh program.
        fc = SimpleNamespace(X=Xc, y=yc)
        gg = lc.ElasticNet(lam=lm, alpha=al)
        P, D, v, _ = logreg_enet_gap_parts(w_full, fc, gg)
        mask, _, _ = logreg_enet_gap_safe_screen(w_full, fc, gg, colnorm=cn)
        return (P - D) / jnp.maximum(jnp.abs(P), 1e-300), mask, v

    # Every operation in the loop is compiled ahead of time for shapes fixed by the width, so the
    # timed loop never compiles: eager gathers and scatters on a new active size each compile
    # on first use, which on the GPU cost more than screening saved.
    S, f64, i32 = jax.ShapeDtypeStruct, jnp.float64, jnp.int32
    import hashlib as _hashlib
    _yfp = _hashlib.blake2b(np.ascontiguousarray(np.asarray(y, dtype=np.float64)).tobytes(),
                            digest_size=8).hexdigest()
    # lam and alpha are NO LONGER in the key: they are runtime arguments now, so one executable serves an
    # entire regularisation path. L_pre is not in it either, because the step size is passed in as well.
    _sig = (int(n), int(p), str(jnp.dtype(dtype)), base, int(K), step_rule, int(power_iters), _yfp)
    compiled, compile_s = _exe_cache_for(_sig), 0.0

    def aot(key, fn, *specs):
        nonlocal compile_s
        if key not in compiled:
            c0 = time.perf_counter()
            compiled[key] = jax.jit(fn).lower(*specs).compile()
            compile_s += time.perf_counter() - c0
        return compiled[key]

    def chunk_exe(w):
        return aot(("chunk", w), chunk, (S((w,), dtype), S((w,), dtype), S((), dtype)), S((n, w), dtype),
                   S((), dtype), S((), dtype), S((), dtype))

    def power_L(XA):
        v = jnp.ones(XA.shape[1], dtype=f64) / np.sqrt(XA.shape[1])

        def it(_, c):
            u = (XA.T @ (XA @ c[0].astype(XA.dtype))).astype(f64)
            nu = jnp.linalg.norm(u)
            return (u / jnp.maximum(nu, 1e-300), nu)
        _, top = jax.lax.fori_loop(0, power_iters, it, (v, jnp.asarray(0.0, dtype=f64)))
        return 1.02 * top / (4.0 * n)

    def gram_L(XA):
        X64 = XA.astype(f64)
        return jnp.linalg.eigvalsh(X64.T @ X64)[-1] / (4.0 * n)

    def L_exe(w):
        fn = {"power": power_L, "gram": gram_L}[step_rule]
        return aot((step_rule, w), fn, S((n, w), dtype))

    def lipschitz(w, XA):
        return L_pre if step_rule == "precomputed" else float(L_exe(w)(XA))

    def scatter_exe(w):          # iterate on the active columns -> full-length float64 vector (pads dropped)
        return aot(("scatter", w), lambda x, idx: jnp.zeros(p, f64).at[idx].set(x.astype(f64), mode="drop"),
                   S((w,), dtype), S((w,), i32))

    def gather_exe(w):           # X on the active columns, pads filled with zero columns
        return aot(("gather", w), lambda Xc, idx: jnp.take(Xc, idx, axis=1, mode="fill", fill_value=0).astype(dtype),
                   S((n, p), f64), S((w,), i32))

    def remap_exe(w_old, w_new):  # new vector from old positions; out-of-range positions give 0
        return aot(("remap", w_old, w_new), lambda v, sel: jnp.take(v, sel, mode="fill", fill_value=0),
                   S((w_old,), dtype), S((w_new,), i32))

    certify_exe = aot(("certify",), certify_, S((p,), f64), S((n, p), f64), S((n,), f64), S((p,), f64),
                      S((), f64), S((), f64))
    lam_s, alpha_s = jnp.asarray(lam, dtype=dtype), jnp.asarray(alpha, dtype=dtype)      # step precision
    lam_d, alpha_d = jnp.asarray(lam, dtype=f64), jnp.asarray(alpha, dtype=f64)          # certificate
    bsizes = _bucket_sizes(p)
    chunk_exe(p); scatter_exe(p)
    if step_rule != "precomputed" and not working_set:    # a working set never takes a full-width step
        L_exe(p)
    if buckets and screen:       # data-independent warm-up: every bucket and every bucket-to-bucket remap
        for i, b in enumerate(bsizes):
            chunk_exe(b); scatter_exe(b); gather_exe(b)
            if step_rule != "precomputed" and (b < p or not working_set):
                L_exe(b)
            for b2 in (bsizes if working_set else bsizes[i + 1:]):
                remap_exe(b, b2)

    def zeros(m):
        return jnp.zeros(m, dtype=dtype)

    def bucket(k):
        return min([b for b in bsizes if b >= k], default=p)

    one = jnp.asarray(1.0, dtype=dtype)
    thr = lam * alpha
    safe_out = np.zeros(p, dtype=bool)                          # Gap Safe screened: zero at every optimum
    evals, replans, adds, rel = 0, 0, 0, float("inf")
    w0 = time.perf_counter()
    if working_set:                                             # inside the timed region: one full pass
        _, mask0, v0 = certify_exe(jnp.zeros(p, f64), Xf, yf, colnorm, lam_d, alpha_d)
        safe_out |= np.asarray(mask0)
        order = np.argsort(-np.abs(np.asarray(v0)), kind="stable")
        order = order[~safe_out[order]]
        width = bucket(max(1, int(np.ceil(ws_init * p))))
        active = order[:width]
        idx = np.full(width, p, dtype=np.int32); idx[:len(active)] = active
        idx_dev = jnp.asarray(idx)
        XA = gather_exe(width)(Xf, idx_dev)
        state = (zeros(width), zeros(width), one)
    else:
        active = np.arange(p); width = p
        idx_dev = jnp.asarray(np.arange(p), dtype=i32)
        state = (zeros(p), zeros(p), one)
        XA = Xf.astype(dtype)
    Lc = lipschitz(width, XA)                                    # timed: the step is part of the solve
    Ls = [Lc]
    sizes, replan_at = [width], []
    while evals < max_evals:
        state = chunk_exe(width)(state, XA, jnp.asarray(1.0 / Lc, dtype=dtype), lam_s, alpha_s)
        evals += K
        w_full = scatter_exe(width)(state[0], idx_dev)
        rel, mask, v = certify_exe(w_full, Xf, yf, colnorm, lam_d, alpha_d)
        rel = float(rel)
        if rel <= tol:
            break
        if not screen:
            continue
        nA = len(active)
        mask_h = np.asarray(mask)
        safe_out |= mask_h
        keep = ~mask_h[active]
        k = int(keep.sum())
        add = np.zeros(0, dtype=np.int64)
        if working_set:
            score = np.abs(np.asarray(v)) - thr
            inW = np.zeros(p, dtype=bool); inW[active[keep]] = True
            cand = np.nonzero((score > 0) & ~inW & ~safe_out)[0]
            if cand.size:
                add = cand[np.argsort(-score[cand], kind="stable")][:max(k, 1)]   # at most double the set
        k_new = k + len(add)
        if k_new == 0:
            continue
        if buckets:
            new_width = bucket(k_new)
            if len(add) == 0 and new_width >= width:
                continue
        else:
            if k > shrink * nA:
                continue
            new_width = k
        sel = np.full(new_width, width, dtype=np.int32); sel[:k] = np.nonzero(keep)[0]   # new/pad slots -> 0
        sel_dev = jnp.asarray(sel)
        R = remap_exe(width, new_width)
        xa = R(state[0], sel_dev)
        active = np.concatenate([active[keep], add])
        idx = np.full(new_width, p, dtype=np.int32); idx[:len(active)] = active          # p is out of range
        idx_dev = jnp.asarray(idx)
        XA = gather_exe(new_width)(Xf, idx_dev)
        L_new = lipschitz(new_width, XA)
        keep_mom = momentum == "keep" and base == "fista" and not L_new < Lc * (1.0 - 1e-12)
        state = (xa, R(state[1], sel_dev), state[2]) if keep_mom else (xa, xa, one)
        Lc = L_new; Ls.append(Lc)
        width = new_width
        replans += 1; adds += len(add); sizes.append(width); replan_at.append(evals)
    jax.block_until_ready(state)
    wall = time.perf_counter() - w0
    w_full = np.asarray(scatter_exe(width)(state[0], idx_dev))
    return {"x": w_full, "evals": evals, "wall_s": wall, "compile_s": compile_s, "wall_incl_compile_s": wall + compile_s,
            "gap": rel, "converged": rel <= tol, "replans": replans, "ws_adds": adds, "L_values": Ls, "step_rule": step_rule, "active_sizes": sizes, "replan_at": replan_at,
            "nnz": int((np.abs(w_full) > 0).sum())}
