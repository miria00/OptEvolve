"""How each arm's cost scales with the WIDTH of the workload: L lambdas over one shared design matrix.

This is the mechanism the whole comparison turns on. Coordinate descent with working sets walks the path one lambda
at a time, warm-starting from the previous one, so its cost is close to linear in L. A batched proximal step is ONE
sparse product whose right-hand side has L columns, so its cost is close to flat in L until the arithmetic saturates
the device. If both hold there is a crossover width, and cross-validation (K folds x L lambdas) sits far past it.

Run the two arms separately (different interpreters, different devices) and join on (n, L, tol).
"""
import argparse, json, os, time

import numpy as np
import scipy.sparse as sp

ap = argparse.ArgumentParser()
ap.add_argument("--stem", required=True)
ap.add_argument("--widths", default="25,50,100,200,400,800")
ap.add_argument("--tol", type=float, default=1e-6, help="certification bar")
ap.add_argument("--solver-tols", default="", help="tolerance SETTINGS for the specialists; their tol is gap <= tol*||y||^2, a different scale from our bar, so it must be swept separately")
ap.add_argument("--eps", type=float, default=0.01)
ap.add_argument("--arm", required=True, choices=("ours", "specialists"))
ap.add_argument("--dtype", default="float64")
ap.add_argument("--solvers", default="celer,sklearn")
ap.add_argument("--max-iters", type=int, default=60000)
ap.add_argument("--deadline-s", type=float, default=600.0)
ap.add_argument("--out", required=True)
a = ap.parse_args()

A = sp.load_npz(a.stem + ".A.npz").tocsc()
meta = np.load(a.stem + ".meta.npz")
b = np.asarray(meta["b"], float)
m = A.shape[0]
lam_max = float(np.max(np.abs(A.T @ b)))

out = {"stem": a.stem, "n": A.shape[0], "p": A.shape[1], "nnz": int(A.nnz), "tol": a.tol,
       "arm": a.arm, "dtype": a.dtype, "rows": []}
for L in [int(x) for x in a.widths.split(",")]:
    lams = lam_max * np.logspace(0.0, np.log10(a.eps), L)
    if a.arm == "ours":
        import sys
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from lasso_path import solve_path
        r = solve_path(A, b, lams, tol=a.tol, max_iters=a.max_iters, dtype=a.dtype,
                       deadline_s=a.deadline_s)
        rec = {"solver": f"ours[{a.dtype}]", "L": L, "t_s": r["total_s"], "setup_s": r["setup_s"],
               "solve_s": r["solve_s"], "iters": r["iters"],
               "n_certified": int((r["rel"] <= a.tol).sum()), "max_rel": float(np.max(r["rel"]))}
        out["rows"].append(rec)
        print(json.dumps({k: v for k, v in rec.items() if k != "rel"}), flush=True)
    else:
        from celer import celer_path
        from sklearn.linear_model import lasso_path
        alphas = lams / m
        stols = [float(x) for x in a.solver_tols.split(",")] if a.solver_tols else [a.tol]
        for name, stol in [(nm, st) for nm in a.solvers.split(",") for st in stols]:
            try:
                t0 = time.perf_counter()
                if name == "celer":
                    _, coefs, _ = celer_path(A, b, pb="lasso", alphas=alphas, tol=stol,
                                             max_iter=100000, verbose=0)
                else:
                    _, coefs, _ = lasso_path(A, b, alphas=alphas, tol=stol, max_iter=100000)
                dt = time.perf_counter() - t0
                X = np.asarray(coefs, float)
                R = A @ X - b[:, None]
                P = 0.5 * np.sum(R * R, axis=0) + lams * np.sum(np.abs(X), axis=0)
                r_ = -R
                scale = np.maximum(lams, np.max(np.abs(A.T @ r_), axis=0))
                resid = b[:, None] - lams[None, :] * (r_ / scale[None, :])
                D = 0.5 * float(b @ b) - 0.5 * np.sum(resid * resid, axis=0)
                rel = np.maximum(P - D, 0.0) / (1.0 + np.abs(P))
                rec = {"solver": name, "L": L, "t_s": dt, "solver_tol": stol,
                       "n_certified": int((rel <= a.tol).sum()), "max_rel": float(np.max(rel)),
                       "rel": rel.tolist()}
            except Exception as e:
                rec = {"solver": name, "L": L, "solver_tol": stol, "error": repr(e)[:200]}
            out["rows"].append(rec)
            print(json.dumps({k: v for k, v in rec.items() if k != "rel"}), flush=True)
    json.dump(out, open(a.out, "w"))
print("WIDTH_SWEEP_DONE", a.out, flush=True)
