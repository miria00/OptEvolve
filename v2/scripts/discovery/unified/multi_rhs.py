"""Many right hand sides, ONE design matrix: the case where the specialist has nothing to warm start from.

This isolates the mechanism. On a lambda PATH, coordinate descent is cheap per lambda not because the solves are
cheap but because consecutive lambdas are close, so every solve after the first starts next to its answer. Batching
cannot beat that: it only amortizes per-iteration overhead, and on a saturated GPU there is none left to amortize.

Remove the chain and the picture should change. Q unrelated targets b_1..b_Q over the same A are Q COLD solves for a
sequential solver, while for a batched proximal step they are one sparse product with Q columns, exactly as before.
If batching ever wins, it wins here. If it does not win here either, the mechanism claim is dead in general and not
just for the lasso path.

Targets are drawn from a matched sparse linear model, b_q = A x_q + noise with ||noise|| set from the requested SNR,
and each is regularized at lam_q = frac * ||A^T b_q||_inf so no column is trivially zero or trivially dense. Both
arms are scored by the same per-column lasso duality gap.
"""
from __future__ import annotations

import argparse, json, time

import numpy as np
import scipy.sparse as sp


def make_targets(A, Q, density=0.01, snr=3.0, seed=11):
    rng = np.random.default_rng(seed)
    m, p = A.shape
    s = max(1, int(density * p))
    B = np.zeros((m, Q))
    for q in range(Q):
        x = np.zeros(p)
        idx = rng.choice(p, s, replace=False)
        x[idx] = rng.standard_normal(s)
        sig = A @ x
        noise = rng.standard_normal(m)
        noise *= np.linalg.norm(sig) / (snr * np.linalg.norm(noise))
        B[:, q] = sig + noise
    return B


def certify_np(A, B, X, lams):
    R = A @ X - B
    P = 0.5 * np.sum(R * R, axis=0) + lams * np.sum(np.abs(X), axis=0)
    r = -R
    scale = np.maximum(lams, np.max(np.abs(A.T @ r), axis=0))
    theta = r / scale[None, :]
    resid = B - lams[None, :] * theta
    D = 0.5 * np.sum(B * B, axis=0) - 0.5 * np.sum(resid * resid, axis=0)
    return P, np.maximum(P - D, 0.0) / (1.0 + np.abs(P))


def _lipschitz(A, iters=60, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(A.shape[1]); v /= np.linalg.norm(v)
    s = 1.0
    for _ in range(iters):
        w = A.T @ (A @ v)
        s = np.linalg.norm(w)
        if s == 0:
            return 1.0
        v = w / s
    return float(s) * 1.001


def solve_ours(A, B, lams, bars, dtype="float64", max_iters=60000, check_every=25, deadline_s=600.0):
    import jax, jax.numpy as jnp
    from jax.experimental import sparse as jsp
    if dtype == "float64":
        jax.config.update("jax_enable_x64", True)
    dt = jnp.float32 if dtype == "float32" else jnp.float64
    dev = jax.devices()[0]
    m, p = A.shape
    Q = B.shape[1]

    t_setup = time.perf_counter()
    Acoo = sp.coo_matrix(A)
    Aj = jsp.BCOO((jnp.asarray(Acoo.data, dt), jnp.asarray(np.stack([Acoo.row, Acoo.col], 1), jnp.int32)),
                  shape=(m, p), indices_sorted=False, unique_indices=True)
    Atj = jsp.BCOO((jnp.asarray(Acoo.data, dt), jnp.asarray(np.stack([Acoo.col, Acoo.row], 1), jnp.int32)),
                   shape=(p, m), indices_sorted=False, unique_indices=True)
    Bj = jnp.asarray(B, dt)
    lamj = jnp.asarray(lams, dt)[None, :]
    step = 1.0 / float(_lipschitz(A))

    @jax.jit
    def body(state):
        X, Z, t, k = state
        V = Z - step * (Atj @ (Aj @ Z - Bj))
        Xn = jnp.sign(V) * jnp.maximum(jnp.abs(V) - step * lamj, 0.0)
        tn = 0.5 * (1.0 + jnp.sqrt(1.0 + 4.0 * t * t))
        return (Xn, Xn + ((t - 1.0) / tn) * (Xn - X), tn, k + 1)

    @jax.jit
    def gaps(X):
        r = Bj - Aj @ X
        P = 0.5 * jnp.sum(r * r, axis=0) + lamj[0] * jnp.sum(jnp.abs(X), axis=0)
        scale = jnp.maximum(lamj[0], jnp.max(jnp.abs(Atj @ r), axis=0))
        resid = Bj - lamj * (r / scale[None, :])
        D = 0.5 * jnp.sum(Bj * Bj, axis=0) - 0.5 * jnp.sum(resid * resid, axis=0)
        return P, jnp.maximum(P - D, 0.0)

    X0 = jnp.zeros((p, Q), dt)
    state = (X0, X0, jnp.asarray(1.0, dt), jnp.asarray(0, jnp.int32))
    body(state)[0].block_until_ready()
    gaps(X0)[0].block_until_ready()
    setup_s = time.perf_counter() - t_setup

    t0 = time.perf_counter()
    it, best, tight = 0, {b: (0, 0.0, 0) for b in bars}, min(bars)
    while it < max_iters:
        for _ in range(check_every):
            state = body(state)
        it += check_every
        P, g = gaps(state[0])
        P = np.asarray(P, float); g = np.asarray(g, float)
        rel = g / (1.0 + np.abs(P))
        el = time.perf_counter() - t0
        for bar in bars:
            c = int((rel <= bar).sum())
            if c > best[bar][0]:
                best[bar] = (c, el, it)
        if np.all(rel <= tight) or el > deadline_s:
            break
    state[0].block_until_ready()
    solve_s = time.perf_counter() - t0
    P, g = gaps(state[0])
    P = np.asarray(P, float); g = np.asarray(g, float)
    rel = g / (1.0 + np.abs(P))
    return {"X": np.asarray(state[0], float), "rel": rel, "iters": it, "setup_s": setup_s,
            "solve_s": solve_s, "total_s": setup_s + solve_s, "device": str(dev), "dtype": dtype,
            "bars": {f"{b:.0e}": {"cells": best[b][0], "t_total_s": setup_s + best[b][1],
                                  "iters": best[b][2]} for b in bars}}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", required=True)
    ap.add_argument("--Q", type=int, default=200)
    ap.add_argument("--frac", type=float, default=0.1, help="lam_q = frac * ||A^T b_q||_inf")
    ap.add_argument("--density", type=float, default=0.01)
    ap.add_argument("--snr", type=float, default=3.0)
    ap.add_argument("--bars", default="1e-4,1e-6,1e-8")
    ap.add_argument("--arm", required=True, choices=("ours", "specialists"))
    ap.add_argument("--dtype", default="float64")
    ap.add_argument("--solvers", default="celer,sklearn")
    ap.add_argument("--tols", default="1e-6,1e-8,1e-10")
    ap.add_argument("--homotopy", type=int, default=0,
                    help="if >0, give the specialist a warm-start chain of this many lambdas per target")
    ap.add_argument("--max-iters", type=int, default=60000)
    ap.add_argument("--deadline-s", type=float, default=600.0)
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    A = sp.load_npz(a.stem + ".A.npz").tocsc()
    m, p = A.shape
    B = make_targets(A, a.Q, a.density, a.snr)
    lams = np.array([a.frac * float(np.max(np.abs(A.T @ B[:, q]))) for q in range(a.Q)])
    bars = [float(x) for x in a.bars.split(",")]
    print(f"m={m} p={p} nnz={A.nnz} Q={a.Q} frac={a.frac} arm={a.arm}", flush=True)
    out = {"stem": a.stem, "m": m, "p": p, "nnz": int(A.nnz), "Q": a.Q, "frac": a.frac,
           "density": a.density, "snr": a.snr, "arm": a.arm, "runs": []}

    if a.arm == "ours":
        r = solve_ours(A, B, lams, bars, dtype=a.dtype, max_iters=a.max_iters, deadline_s=a.deadline_s)
        rec = {k: v for k, v in r.items() if k != "X"}
        rec["rel"] = r["rel"].tolist(); rec["solver"] = f"ours[{a.dtype}]"
        out["runs"].append(rec)
        print(json.dumps({k: v for k, v in rec.items() if k != "rel"}), flush=True)
    else:
        from celer import celer_path
        from sklearn.linear_model import lasso_path
        from joblib import Parallel, delayed

        def one(q, name, tol):
            if a.homotopy > 0:                      # the specialist's own trick: walk down to lam_q
                lmax = float(np.max(np.abs(A.T @ B[:, q])))
                grid = lmax * np.logspace(0.0, np.log10(a.frac), a.homotopy)
            else:
                grid = np.array([lams[q]])
            al = grid / m
            if name == "celer":
                _, co, _ = celer_path(A, B[:, q], pb="lasso", alphas=al, tol=tol,
                                      max_iter=a.max_iters, verbose=0)
            else:
                _, co, _ = lasso_path(A, B[:, q], alphas=al, tol=tol, max_iter=a.max_iters)
            return np.asarray(co, float)[:, -1]     # alphas are descending; last is lam_q

        for name in a.solvers.split(","):
            for tol in [float(t) for t in a.tols.split(",")]:
                try:
                    t0 = time.perf_counter()
                    if a.n_jobs > 1:
                        cols = Parallel(n_jobs=a.n_jobs, backend="loky")(
                            delayed(one)(q, name, tol) for q in range(a.Q))
                    else:
                        cols = [one(q, name, tol) for q in range(a.Q)]
                    dt = time.perf_counter() - t0
                    X = np.stack(cols, axis=1)
                    P, rel = certify_np(A, B, X, lams)
                    rec = {"solver": name, "tol": tol, "homotopy": a.homotopy, "n_jobs": a.n_jobs,
                           "total_s": dt, "rel": rel.tolist(), "max_rel": float(np.max(rel))}
                    for bar in bars:
                        rec[f"cert_{bar:.0e}"] = int((rel <= bar).sum())
                except Exception as e:
                    rec = {"solver": name, "tol": tol, "homotopy": a.homotopy, "error": repr(e)[:300]}
                out["runs"].append(rec)
                print(json.dumps({k: v for k, v in rec.items() if k != "rel"}), flush=True)
                json.dump(out, open(a.out, "w"))
    json.dump(out, open(a.out, "w"))
    print("MRHS_DONE", a.out, flush=True)
