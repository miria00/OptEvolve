"""Does the batching argument survive past the lasso? An elastic net path, both arms, one certificate.

The mechanism claim is not about the lasso. It is about workloads where ONE operator is shared across many
right hand sides or many regularization values: coordinate descent has to walk them, a batched proximal step does
not. The elastic net is the cleanest test of that claim, because it changes the objective without changing the
structure: the same design matrix, the same L values, the same sparse product, only the prox and the certificate move.

  min_x  0.5||Ax - b||^2 + lam1||x||_1 + (lam2/2)||x||^2       lam1 = n a r,  lam2 = n a (1 - r)   [sklearn's (a, r)]

The prox is soft(v, s lam1) / (1 + s lam2), one extra scalar divide per element. The certificate is the exact
Fenchel-Young residual, which for lam2 > 0 needs NO dual rescaling because the dual is unconstrained:

  r = b - A x,  v = A^T r,  P = 0.5||r||^2 + lam1||x||_1 + (lam2/2)||x||^2
  D = -0.5||r||^2 + b^T r - sum_j soft(|v_j|, lam1)^2 / (2 lam2),   gap = P - D >= 0

gap = sum_j [ g(x_j) + g*(v_j) - x_j v_j ] with g the elastic net penalty, so it is zero exactly at the solution and
positive everywhere else. Scored as rel = gap / (1 + |P|), the same bar the lasso runs use.
"""
from __future__ import annotations

import argparse, json, os, time

import numpy as np
import scipy.sparse as sp


def enet_grid(A, b, L=100, eps=0.01):
    """sklearn's _alpha_grid for the elastic net, expressed in the raw (unscaled) lam1."""
    lam1_max = float(np.max(np.abs(A.T @ b)))
    return lam1_max * np.logspace(0.0, np.log10(eps), L), lam1_max


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


def certify_enet_np(A, b, X, lam1, lam2):
    R = A @ X - b[:, None]
    r = -R
    P = 0.5 * np.sum(r * r, axis=0) + lam1 * np.sum(np.abs(X), axis=0) + 0.5 * lam2 * np.sum(X * X, axis=0)
    v = A.T @ r
    sft = np.maximum(np.abs(v) - lam1[None, :], 0.0)
    D = -0.5 * np.sum(r * r, axis=0) + (b @ r) - np.sum(sft * sft, axis=0) / (2.0 * lam2)
    g = np.maximum(P - D, 0.0)
    return P, g / (1.0 + np.abs(P))


def solve_enet_ours(A, b, lam1, lam2, bars, dtype="float64", max_iters=60000, check_every=25,
                    deadline_s=600.0):
    import jax, jax.numpy as jnp
    from jax.experimental import sparse as jsp
    if dtype == "float64":
        jax.config.update("jax_enable_x64", True)
    dt = jnp.float32 if dtype == "float32" else jnp.float64
    dev = jax.devices()[0]
    m, p = A.shape
    L = lam1.size

    t_setup = time.perf_counter()
    Acoo = sp.coo_matrix(A)
    Aj = jsp.BCOO((jnp.asarray(Acoo.data, dt), jnp.asarray(np.stack([Acoo.row, Acoo.col], 1), jnp.int32)),
                  shape=(m, p), indices_sorted=False, unique_indices=True)
    Atj = jsp.BCOO((jnp.asarray(Acoo.data, dt), jnp.asarray(np.stack([Acoo.col, Acoo.row], 1), jnp.int32)),
                   shape=(p, m), indices_sorted=False, unique_indices=True)
    bj = jnp.asarray(b, dt)
    l1 = jnp.asarray(lam1, dt)[None, :]
    l2 = jnp.asarray(lam2, dt)[None, :]
    step = 1.0 / float(_lipschitz(A))

    @jax.jit
    def body(state):
        X, Z, t, k = state
        V = Z - step * (Atj @ (Aj @ Z - bj[:, None]))
        Xn = (jnp.sign(V) * jnp.maximum(jnp.abs(V) - step * l1, 0.0)) / (1.0 + step * l2)
        tn = 0.5 * (1.0 + jnp.sqrt(1.0 + 4.0 * t * t))
        Zn = Xn + ((t - 1.0) / tn) * (Xn - X)
        return (Xn, Zn, tn, k + 1)

    @jax.jit
    def gaps(X):
        r = bj[:, None] - Aj @ X
        P = (0.5 * jnp.sum(r * r, axis=0) + l1[0] * jnp.sum(jnp.abs(X), axis=0)
             + 0.5 * l2[0] * jnp.sum(X * X, axis=0))
        v = Atj @ r
        sft = jnp.maximum(jnp.abs(v) - l1, 0.0)
        D = -0.5 * jnp.sum(r * r, axis=0) + (bj @ r) - jnp.sum(sft * sft, axis=0) / (2.0 * l2[0])
        return P, jnp.maximum(P - D, 0.0)

    X0 = jnp.zeros((p, L), dt)
    state = (X0, X0, jnp.asarray(1.0, dt), jnp.asarray(0, jnp.int32))
    body(state)[0].block_until_ready()
    gaps(X0)[0].block_until_ready()
    setup_s = time.perf_counter() - t_setup

    t0 = time.perf_counter()
    it, hist = 0, []
    best = {bar: (0, 0.0, 0) for bar in bars}
    tight = min(bars)
    while it < max_iters:
        for _ in range(check_every):
            state = body(state)
        it += check_every
        P, g = gaps(state[0])
        P = np.asarray(P, float); g = np.asarray(g, float)
        rel = g / (1.0 + np.abs(P))
        el = time.perf_counter() - t0
        row = {"iters": it, "t_s": el, "max_rel": float(np.max(rel))}
        for bar in bars:
            c = int((rel <= bar).sum())
            row[f"cert_{bar:.0e}"] = c
            if c > best[bar][0]:
                best[bar] = (c, el, it)
        hist.append(row)
        if np.all(rel <= tight):
            break
        if time.perf_counter() - t0 > deadline_s:
            break
    state[0].block_until_ready()
    solve_s = time.perf_counter() - t0
    P, g = gaps(state[0])
    P = np.asarray(P, float); g = np.asarray(g, float)
    rel = g / (1.0 + np.abs(P))
    return {"X": np.asarray(state[0], float), "rel": rel, "obj": P, "iters": it, "setup_s": setup_s,
            "solve_s": solve_s, "total_s": setup_s + solve_s, "hist": hist, "device": str(dev),
            "dtype": dtype,
            "bars": {f"{bar:.0e}": {"cells": best[bar][0], "t_solve_s": best[bar][1],
                                    "t_total_s": setup_s + best[bar][1], "iters": best[bar][2]}
                     for bar in bars}}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", required=True)
    ap.add_argument("--n-lam", type=int, default=100)
    ap.add_argument("--eps", type=float, default=0.01)
    ap.add_argument("--l1-ratio", type=float, default=0.8)
    ap.add_argument("--bars", default="1e-4,1e-6,1e-8")
    ap.add_argument("--arm", required=True, choices=("ours", "specialists"))
    ap.add_argument("--dtype", default="float64", choices=("float32", "float64"))
    ap.add_argument("--solvers", default="celer,sklearn")
    ap.add_argument("--tols", default="1e-4,1e-6,1e-8")
    ap.add_argument("--max-iters", type=int, default=60000)
    ap.add_argument("--deadline-s", type=float, default=600.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    A = sp.load_npz(a.stem + ".A.npz").tocsc()
    b = np.asarray(np.load(a.stem + ".meta.npz")["b"], float)
    m, p = A.shape
    lam1, lam1_max = enet_grid(A, b, a.n_lam, a.eps)
    lam2 = lam1 * (1.0 - a.l1_ratio) / a.l1_ratio
    bars = [float(x) for x in a.bars.split(",")]
    print(f"m={m} p={p} nnz={A.nnz} L={lam1.size} l1_ratio={a.l1_ratio} lam1_max={lam1_max:.6g}", flush=True)
    out = {"stem": a.stem, "m": m, "p": p, "nnz": int(A.nnz), "L": int(lam1.size), "eps": a.eps,
           "l1_ratio": a.l1_ratio, "lam1": lam1.tolist(), "arm": a.arm, "runs": []}

    if a.arm == "ours":
        r = solve_enet_ours(A, b, lam1, lam2, bars, dtype=a.dtype, max_iters=a.max_iters,
                            deadline_s=a.deadline_s)
        np.save(a.out + ".X.npy", r["X"])
        rec = {k: v for k, v in r.items() if k != "X"}
        rec["rel"] = r["rel"].tolist(); rec["obj"] = r["obj"].tolist()
        rec["solver"] = f"ours[{a.dtype}]"
        out["runs"].append(rec)
        print(json.dumps({"solver": rec["solver"], "iters": r["iters"], "setup_s": r["setup_s"],
                          "solve_s": r["solve_s"], "total_s": r["total_s"], "bars": r["bars"],
                          "max_rel": float(np.max(r["rel"]))}), flush=True)
    else:
        alphas = lam1 / (m * a.l1_ratio)                      # sklearn/celer (alpha, l1_ratio) convention
        from celer import celer_path
        from sklearn.linear_model import enet_path
        for name in a.solvers.split(","):
            for tol in [float(t) for t in a.tols.split(",")]:
                try:
                    t0 = time.perf_counter()
                    if name == "celer":
                        _, coefs, _ = celer_path(A, b, pb="lasso", alphas=alphas, l1_ratio=a.l1_ratio,
                                                 tol=tol, max_iter=a.max_iters, verbose=0)
                    else:
                        _, coefs, _ = enet_path(A, b, alphas=alphas, l1_ratio=a.l1_ratio, tol=tol,
                                                max_iter=a.max_iters)
                    dt = time.perf_counter() - t0
                    X = np.asarray(coefs, float)
                    P, rel = certify_enet_np(A, b, X, lam1, lam2)
                    rec = {"solver": name, "tol": tol, "t_s": dt, "total_s": dt, "rel": rel.tolist(),
                           "max_rel": float(np.max(rel)),
                           "support_max": int(np.max((X != 0).sum(axis=0)))}
                    for bar in bars:
                        rec[f"cert_{bar:.0e}"] = int((rel <= bar).sum())
                    np.save(a.out + f".{name}_{tol:.0e}.X.npy", X)
                except Exception as e:
                    rec = {"solver": name, "tol": tol, "error": repr(e)[:300]}
                out["runs"].append(rec)
                print(json.dumps({k: v for k, v in rec.items() if k not in ("rel", "obj")}), flush=True)
                json.dump(out, open(a.out, "w"))
    json.dump(out, open(a.out, "w"))
    print("ENET_DONE", a.out, flush=True)
