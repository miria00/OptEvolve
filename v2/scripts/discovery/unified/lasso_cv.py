"""K-fold cross-validation over a lasso path: K x L problems sharing ONE design matrix, batched on the GPU.

Why this is the decisive unit of work. A single lasso solve is won by coordinate descent with working sets, and not
narrowly (368x at n = 100,000). That advantage is SEQUENTIAL: celer and scikit-learn walk the lambda grid one value at
a time, warm-starting, and K-fold CV multiplies that walk by K. A proximal-gradient step is one sparse product whose
right hand side can carry every (fold, lambda) column at once, so the whole K x L field costs one product per
iteration instead of K * L of them.

The fold restriction is a ROW MASK on the residual, which is exact and costs nothing:

  grad_k(x) = A_k^T (A_k x - b_k) = A^T [ m_k * (A x - b) ]        m_k the 0/1 training-row mask of fold k

and ||A_k||_2 <= ||A||_2 for every k because masking only deletes rows, so the single step size 1 / ||A||_2^2 computed
once on the full matrix is admissible for every fold. Columns are laid out as (p, K, L), the residual is reshaped to
(m, K, L) and multiplied by the mask (m, K, 1), so the mask costs one elementwise product per iteration.

Certification is per CELL and reference-free, by the lasso duality gap of that cell's own masked subproblem:

  r = m_k * (b - A x),  theta = r / max(lam, ||A^T r||_inf),  P = 0.5||r||^2 + lam||x||_1,
  D = 0.5||m_k b||^2 - 0.5||m_k b - lam theta||^2,  rel = (P - D) / (1 + |P|)

Held out rows contribute nothing to either term because r and m_k b both vanish there. Nobody is credited for their
own stopping rule: the same formula scores the specialists in --arm specialists.

Our arm is measured ANYTIME: one run to the tightest bar, recording at every check the wall time and how many cells
are certified at each bar, so the time reported for a bar is the first moment that bar was met. The specialists are
run at several tolerance settings and scored virtual-best over settings, the same convention as lasso_path_score.py.
"""
from __future__ import annotations

import argparse, json, os, time

import numpy as np
import scipy.sparse as sp


def kfold_masks(m, K, seed=0):
    """Training-row masks, one column per fold, from a shuffled equal split (sklearn KFold(shuffle=True))."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(m)
    folds = np.array_split(perm, K)
    M = np.ones((m, K))
    for k in range(K):
        M[folds[k], k] = 0.0
    return M, [np.sort(f) for f in folds]


def lambda_grid(A, b, L=100, eps=0.01):
    """One grid for the whole field, taken from the FULL data, as LassoCV does."""
    lam_max = float(np.max(np.abs(A.T @ b)))
    return lam_max * np.logspace(0.0, np.log10(eps), L), lam_max


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


def solve_cv_ours(A, b, lams, M, bars, dtype="float64", max_iters=40000, check_every=25,
                  deadline_s=900.0, device=None):
    import jax, jax.numpy as jnp
    from jax.experimental import sparse as jsp
    if dtype == "float64":
        jax.config.update("jax_enable_x64", True)
    dt = jnp.float32 if dtype == "float32" else jnp.float64
    dev = jax.devices(device)[0] if device else jax.devices()[0]

    m, p = A.shape
    K = M.shape[1]
    L = lams.size
    KL = K * L
    tol_tight = min(bars)

    t_setup = time.perf_counter()
    Acoo = sp.coo_matrix(A)
    Aj = jsp.BCOO((jnp.asarray(Acoo.data, dt),
                   jnp.asarray(np.stack([Acoo.row, Acoo.col], 1), jnp.int32)),
                  shape=(m, p), indices_sorted=False, unique_indices=True)
    Atj = jsp.BCOO((jnp.asarray(Acoo.data, dt),
                    jnp.asarray(np.stack([Acoo.col, Acoo.row], 1), jnp.int32)),
                   shape=(p, m), indices_sorted=False, unique_indices=True)
    bj = jnp.asarray(b, dt)
    Mj = jnp.asarray(M, dt)                                  # (m, K)
    bM = bj[:, None] * Mj                                    # (m, K), the masked targets
    lamj = jnp.asarray(np.tile(lams, K), dt)[None, :]        # (1, KL), column k*L + l
    step = 1.0 / float(_lipschitz(A))                        # valid for every fold: masking only removes rows

    def mask(R):
        return (R.reshape(m, K, L) * Mj[:, :, None]).reshape(m, KL)

    @jax.jit
    def body(state):
        X, Z, t, k = state
        V = Z - step * (Atj @ mask(Aj @ Z - bj[:, None]))
        Xn = jnp.sign(V) * jnp.maximum(jnp.abs(V) - step * lamj, 0.0)
        tn = 0.5 * (1.0 + jnp.sqrt(1.0 + 4.0 * t * t))
        Zn = Xn + ((t - 1.0) / tn) * (Xn - X)
        return (Xn, Zn, tn, k + 1)

    @jax.jit
    def gaps(X):
        r = mask(bj[:, None] - Aj @ X)                       # masked residual, (m, KL)
        P = 0.5 * jnp.sum(r * r, axis=0) + lamj[0] * jnp.sum(jnp.abs(X), axis=0)
        AtR = Atj @ r
        scale = jnp.maximum(lamj[0], jnp.max(jnp.abs(AtR), axis=0))
        theta = r / scale[None, :]
        resid = (bM[:, :, None] - lamj.reshape(1, K, L) * theta.reshape(m, K, L)).reshape(m, KL)
        bsq = jnp.sum(bM * bM, axis=0)                       # (K,), one per fold
        D = 0.5 * jnp.repeat(bsq, L) - 0.5 * jnp.sum(resid * resid, axis=0)
        return P, jnp.maximum(P - D, 0.0)

    @jax.jit
    def val_mse(X):
        """Held-out mean squared error of every cell, the number CV actually selects on."""
        R = Aj @ X - bj[:, None]
        V = (R.reshape(m, K, L) * (1.0 - Mj)[:, :, None])
        n_val = jnp.sum(1.0 - Mj, axis=0)                    # (K,)
        return (jnp.sum(V * V, axis=0) / n_val[:, None])     # (K, L)

    X0 = jnp.zeros((p, KL), dt)
    state = (X0, X0, jnp.asarray(1.0, dt), jnp.asarray(0, jnp.int32))
    body(state)[0].block_until_ready()                       # compile OUTSIDE the iteration clock
    gaps(X0)[0].block_until_ready()
    val_mse(X0).block_until_ready()
    setup_s = time.perf_counter() - t_setup

    t0 = time.perf_counter()
    it, hist = 0, []
    best = {bar: (0, 0.0, 0) for bar in bars}                # bar -> (cells, first time reaching it, iters)
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
        if np.all(rel <= tol_tight):
            break
        if time.perf_counter() - t0 > deadline_s:
            break
    state[0].block_until_ready()
    solve_s = time.perf_counter() - t0
    P, g = gaps(state[0])
    P = np.asarray(P, float); g = np.asarray(g, float)
    rel = g / (1.0 + np.abs(P))
    V = np.asarray(val_mse(state[0]), float)
    Xf = np.asarray(state[0], float)
    return {"X": Xf, "rel": rel, "obj": P, "iters": it, "setup_s": setup_s, "solve_s": solve_s,
            "total_s": setup_s + solve_s, "hist": hist, "val_mse": V,
            "bars": {f"{bar:.0e}": {"cells": best[bar][0], "t_solve_s": best[bar][1],
                                    "t_total_s": setup_s + best[bar][1], "iters": best[bar][2]}
                     for bar in bars},
            "device": str(dev), "dtype": dtype, "K": K, "L": L, "step": step}


def certify_numpy(A, b, X, lams, rows):
    """The same certificate, in numpy, for one fold's columns. `rows` are that fold's TRAINING rows."""
    Ak = A[rows]
    bk = b[rows]
    R = Ak @ X - bk[:, None]
    P = 0.5 * np.sum(R * R, axis=0) + lams * np.sum(np.abs(X), axis=0)
    r = -R
    AtR = Ak.T @ r
    scale = np.maximum(lams, np.max(np.abs(AtR), axis=0))
    theta = r / scale[None, :]
    resid = bk[:, None] - lams[None, :] * theta
    D = 0.5 * float(bk @ bk) - 0.5 * np.sum(resid * resid, axis=0)
    return P, np.maximum(P - D, 0.0) / (1.0 + np.abs(P))


def _fold_job(args):
    """One fold of the path, in the caller's process or a joblib worker."""
    name, Acsr, b, tr, lams, tol, max_iter = args
    import time as _t
    t0 = _t.perf_counter()
    Ak = Acsr[tr].tocsc()
    bk = np.ascontiguousarray(b[tr])
    alphas = lams / tr.size                                  # sklearn/celer scale the loss by 1/n_train
    if name == "celer":
        from celer import celer_path
        _, coefs, _ = celer_path(Ak, bk, pb="lasso", alphas=alphas, tol=tol, max_iter=max_iter, verbose=0)
    else:
        from sklearn.linear_model import lasso_path
        _, coefs, _ = lasso_path(Ak, bk, alphas=alphas, tol=tol, max_iter=max_iter)
    return np.asarray(coefs, float), _t.perf_counter() - t0


def solve_cv_specialist(A, b, lams, folds, name, tol, max_iter=100000, backend="seq", n_jobs=1):
    """K-fold CV the way a practitioner runs it: the specialist's own path routine, once per fold.

    Folds may be run sequentially or in parallel across cores (joblib), and the parallel arm is reported too, so the
    specialist is not handicapped by being forced to stay on one core.
    """
    m = A.shape[0]
    t_setup = time.perf_counter()
    Acsr = A.tocsr()
    setup_s = time.perf_counter() - t_setup
    jobs = []
    for f in folds:
        tr = np.setdiff1d(np.arange(m), f)
        jobs.append((name, Acsr, b, tr, lams, tol, max_iter))
    t0 = time.perf_counter()
    if backend == "seq":
        res = [_fold_job(j) for j in jobs]
    else:
        from joblib import Parallel, delayed
        res = Parallel(n_jobs=n_jobs, backend=backend)(delayed(_fold_job)(j) for j in jobs)
    solve_s = time.perf_counter() - t0
    Xs = [r[0] for r in res]
    return {"X": Xs, "setup_s": setup_s, "solve_s": solve_s, "total_s": setup_s + solve_s,
            "fold_s": [r[1] for r in res], "trains": [np.setdiff1d(np.arange(m), f) for f in folds]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", required=True)
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--n-lam", type=int, default=100)
    ap.add_argument("--eps", type=float, default=0.01)
    ap.add_argument("--bars", default="1e-4,1e-6,1e-8")
    ap.add_argument("--arm", required=True, choices=("ours", "specialists"))
    ap.add_argument("--dtype", default="float64", choices=("float32", "float64"))
    ap.add_argument("--solvers", default="celer,sklearn")
    ap.add_argument("--tols", default="1e-4,1e-6,1e-8")
    ap.add_argument("--backend", default="seq", choices=("seq", "loky", "threading"))
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--max-iters", type=int, default=60000)
    ap.add_argument("--check-every", type=int, default=25)
    ap.add_argument("--deadline-s", type=float, default=900.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-x", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    A = sp.load_npz(a.stem + ".A.npz").tocsc()
    meta = np.load(a.stem + ".meta.npz")
    b = np.asarray(meta["b"], float)
    m, p = A.shape
    lams, lam_max = lambda_grid(A, b, a.n_lam, a.eps)
    M, folds = kfold_masks(m, a.K, a.seed)
    bars = [float(x) for x in a.bars.split(",")]
    print(f"m={m} p={p} nnz={A.nnz} K={a.K} L={lams.size} cells={a.K*lams.size} "
          f"lam_max={lam_max:.6g} lam_min={lams[-1]:.6g}", flush=True)

    out = {"stem": a.stem, "m": m, "p": p, "nnz": int(A.nnz), "K": a.K, "L": int(lams.size),
           "cells": int(a.K * lams.size), "eps": a.eps, "lam_max": lam_max, "lams": lams.tolist(),
           "arm": a.arm, "seed": a.seed, "runs": []}

    if a.arm == "ours":
        r = solve_cv_ours(A, b, lams, M, bars, dtype=a.dtype, max_iters=a.max_iters,
                          check_every=a.check_every, deadline_s=a.deadline_s)
        if a.save_x:
            np.save(a.out + ".X.npy", r["X"])
        np.save(a.out + ".valmse.npy", r["val_mse"])
        rec = {k: v for k, v in r.items() if k not in ("X", "val_mse")}
        rec["rel"] = r["rel"].tolist(); rec["obj"] = r["obj"].tolist()
        rec["solver"] = f"ours[{a.dtype}]"
        out["runs"].append(rec)
        print(json.dumps({"solver": rec["solver"], "iters": r["iters"], "setup_s": r["setup_s"],
                          "solve_s": r["solve_s"], "total_s": r["total_s"], "bars": r["bars"],
                          "max_rel": float(np.max(r["rel"]))}), flush=True)
    else:
        Acsc = A
        for name in a.solvers.split(","):
            for tol in [float(t) for t in a.tols.split(",")]:
                try:
                    s = solve_cv_specialist(Acsc, b, lams, folds, name, tol, a.max_iters,
                                            backend=a.backend, n_jobs=a.n_jobs)
                    rel = np.concatenate([certify_numpy(Acsc, b, s["X"][k], lams, s["trains"][k])[1]
                                          for k in range(a.K)])
                    rec = {"solver": name, "tol": tol, "backend": a.backend, "n_jobs": a.n_jobs,
                           "setup_s": s["setup_s"], "solve_s": s["solve_s"], "total_s": s["total_s"],
                           "fold_s": s["fold_s"], "rel": rel.tolist(), "max_rel": float(np.max(rel))}
                    for bar in bars:
                        rec[f"cert_{bar:.0e}"] = int((rel <= bar).sum())
                    if a.save_x:
                        np.save(a.out + f".{name}_{tol:.0e}.X.npy", np.stack(s["X"]))
                except Exception as e:
                    rec = {"solver": name, "tol": tol, "error": repr(e)[:300]}
                out["runs"].append(rec)
                print(json.dumps({k: v for k, v in rec.items() if k != "rel"}), flush=True)
                json.dump(out, open(a.out, "w"))
    json.dump(out, open(a.out, "w"))
    print("CV_DONE", a.out, flush=True)
