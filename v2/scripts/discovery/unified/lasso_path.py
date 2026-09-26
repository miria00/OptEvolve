"""Batched lasso regularization path on the GPU: every lambda solved at once, each one certified on its own.

Why this is the right unit of work. A single lasso solve is won by coordinate descent with working sets, and not
narrowly: celer and scikit-learn beat our operator by 368x at n = 100,000 because the solution has 2,807 nonzeros and
they touch only those columns. That advantage is sequential. The workload a practitioner actually runs is not one
solve, it is a path over ~100 lambdas, or K-fold cross-validation over that path, all sharing ONE design matrix.
Coordinate descent must walk the path one lambda at a time, warm-starting from the last. A proximal-gradient step is
the same sparse product for every lambda at once, so the whole path costs one matrix product per iteration instead of
L of them, and the GPU's disadvantage (it touches all p columns) is amortized across the path.

  X in R^{p x L},  R = A X - b,  G = A^T R,  X <- soft(X - s G, s * lam)     with lam broadcast along the path

Certification is per lambda and reference-free, by the standard lasso duality gap: for residual r = b - A x, the dual
point theta = r / max(lam, ||A^T r||_inf) is feasible by construction, so

  P = 0.5||Ax-b||^2 + lam||x||_1,   D = 0.5||b||^2 - 0.5||b - lam theta||^2,   gap = P - D >= 0

bounds the suboptimality of that lambda's column without anyone solving it first. A column is accepted at tol when
gap / (1 + |P|) <= tol, the same relative convention the router's acceptance check uses.
"""
from __future__ import annotations

import argparse, json, os, time

import numpy as np
import scipy.sparse as sp


def lambda_grid(A, b, L=100, eps=0.01):
    """glmnet's grid: L points log-spaced from the smallest lambda that zeroes x down to eps * that."""
    lam_max = float(np.max(np.abs(A.T @ b)))
    return lam_max * np.logspace(0.0, np.log10(eps), L), lam_max


def solve_path(A, b, lams, tol=1e-6, max_iters=20000, check_every=25, dtype="float32", device=None,
               deadline_s=None):
    import jax, jax.numpy as jnp
    from jax.experimental import sparse as jsp
    dt = jnp.float32 if dtype == "float32" else jnp.float64
    if dtype == "float64":
        jax.config.update("jax_enable_x64", True)
    dev = jax.devices(device)[0] if device else jax.devices()[0]

    t_setup = time.perf_counter()
    Acoo = sp.coo_matrix(A)
    Aj = jsp.BCOO((jnp.asarray(Acoo.data, dt),
                   jnp.asarray(np.stack([Acoo.row, Acoo.col], 1), jnp.int32)),
                  shape=A.shape, indices_sorted=False, unique_indices=True)
    Atj = jsp.BCOO((jnp.asarray(Acoo.data, dt),
                    jnp.asarray(np.stack([Acoo.col, Acoo.row], 1), jnp.int32)),
                   shape=(A.shape[1], A.shape[0]), indices_sorted=False, unique_indices=True)
    bj = jnp.asarray(b, dt)
    lamj = jnp.asarray(lams, dt)[None, :]
    step = 1.0 / float(_lipschitz(A))                       # 1/||A||_2^2, shared by every lambda

    @jax.jit
    def body(state):
        X, Z, t, k = state
        R = Aj @ Z - bj[:, None]
        G = Atj @ R
        Xn = _soft(Z - step * G, step * lamj)
        tn = 0.5 * (1.0 + jnp.sqrt(1.0 + 4.0 * t * t))
        Zn = Xn + ((t - 1.0) / tn) * (Xn - X)
        return (Xn, Zn, tn, k + 1)

    @jax.jit
    def gaps(X):
        """Per-lambda primal value and certified duality gap."""
        R = Aj @ X - bj[:, None]                            # = A x - b
        P = 0.5 * jnp.sum(R * R, axis=0) + lamj[0] * jnp.sum(jnp.abs(X), axis=0)
        r = -R                                              # b - A x
        AtR = Atj @ r
        scale = jnp.maximum(lamj[0], jnp.max(jnp.abs(AtR), axis=0))
        theta = r / scale[None, :]
        resid = bj[:, None] - lamj[0][None, :] * theta
        D = 0.5 * jnp.sum(bj * bj) - 0.5 * jnp.sum(resid * resid, axis=0)
        return P, jnp.maximum(P - D, 0.0)

    p, L = A.shape[1], lams.size
    X = jnp.zeros((p, L), dt)
    state = (X, X, jnp.asarray(1.0, dt), jnp.asarray(0, jnp.int32))
    body(state)[0].block_until_ready()                      # compile outside the iteration clock
    gaps(X)[0].block_until_ready()
    setup_s = time.perf_counter() - t_setup

    t0 = time.perf_counter()
    it, hist = 0, []
    while it < max_iters:
        for _ in range(check_every):
            state = body(state)
        it += check_every
        P, g = gaps(state[0])
        P, g = np.asarray(P, float), np.asarray(g, float)
        rel = g / (1.0 + np.abs(P))
        hist.append((it, float(np.max(rel)), int((rel <= tol).sum())))
        if np.all(rel <= tol):
            break
        if deadline_s is not None and time.perf_counter() - t0 > deadline_s:
            break
    state[0].block_until_ready()
    solve_s = time.perf_counter() - t0
    Xf = np.asarray(state[0], float)
    P, g = gaps(state[0])
    P, g = np.asarray(P, float), np.asarray(g, float)
    return {"X": Xf, "obj": P, "gap": g, "rel": g / (1.0 + np.abs(P)), "iters": it,
            "setup_s": setup_s, "solve_s": solve_s, "total_s": setup_s + solve_s,
            "converged": bool(np.all(g / (1.0 + np.abs(P)) <= tol)), "hist": hist,
            "device": str(dev), "dtype": dtype}


def _soft(x, t):
    import jax.numpy as jnp
    return jnp.sign(x) * jnp.maximum(jnp.abs(x) - t, 0.0)


def _lipschitz(A, iters=60, seed=0):
    """||A||_2^2 by power iteration on A^T A (bounded cost, unlike an ARPACK call)."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(A.shape[1])
    v /= np.linalg.norm(v)
    s = 1.0
    for _ in range(iters):
        w = A.T @ (A @ v)
        s = np.linalg.norm(w)
        if s == 0:
            return 1.0
        v = w / s
    return float(s) * 1.001                                  # small margin so the step stays admissible


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", required=True, help="prefix written by build_lasso.py (.A.npz / .meta.npz)")
    ap.add_argument("--n-lam", type=int, default=100)
    ap.add_argument("--eps", type=float, default=0.01)
    ap.add_argument("--tols", default="1e-4,1e-6,1e-8")
    ap.add_argument("--dtype", default="float32", choices=("float32", "float64"))
    ap.add_argument("--max-iters", type=int, default=40000)
    ap.add_argument("--deadline-s", type=float, default=900.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    A = sp.load_npz(a.stem + ".A.npz").tocsc()
    meta = np.load(a.stem + ".meta.npz")
    b = np.asarray(meta["b"], float)
    lams, lam_max = lambda_grid(A, b, a.n_lam, a.eps)
    print(f"n={A.shape[0]} p={A.shape[1]} nnz={A.nnz} L={lams.size} lam_max={lam_max:.6g} "
          f"lam_min={lams[-1]:.6g}", flush=True)
    out = {"stem": a.stem, "n": A.shape[0], "p": A.shape[1], "nnz": int(A.nnz), "n_lam": int(lams.size),
           "eps": a.eps, "lam_max": lam_max, "lams": lams.tolist(), "dtype": a.dtype, "runs": []}
    for tol in [float(t) for t in a.tols.split(",")]:
        r = solve_path(A, b, lams, tol=tol, max_iters=a.max_iters, dtype=a.dtype, deadline_s=a.deadline_s)
        np.save(a.out + f".X_{tol:.0e}.npy", r["X"].astype(np.float64))
        rec = {k: v for k, v in r.items() if k != "X"}
        rec["obj"] = r["obj"].tolist(); rec["gap"] = r["gap"].tolist(); rec["rel"] = r["rel"].tolist()
        rec["tol"] = tol
        rec["n_certified"] = int((r["rel"] <= tol).sum())
        out["runs"].append(rec)
        print(f"tol={tol:.0e} iters={r['iters']} setup={r['setup_s']:.2f}s solve={r['solve_s']:.2f}s "
              f"total={r['total_s']:.2f}s certified={rec['n_certified']}/{lams.size} "
              f"max_rel={float(np.max(r['rel'])):.3e} converged={r['converged']}", flush=True)
        json.dump(out, open(a.out, "w"))
    print("PATH_DONE", a.out, flush=True)
