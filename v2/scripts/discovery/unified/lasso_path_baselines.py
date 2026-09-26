"""The lasso path, computed by the solvers actually built for it. This is the completed field, not a stand-in.

celer and scikit-learn both expose a path routine that walks the lambda grid with warm starts, which is the fair and
strong way to run them: it is what a practitioner runs and it is much faster than L independent solves. Every returned
path is scored by OUR certificate (the lasso duality gap, per lambda) so both sides are judged on the same number and
neither is credited for its own stopping rule.
"""
import argparse, json, os, time

import numpy as np
import scipy.sparse as sp

ap = argparse.ArgumentParser()
ap.add_argument("--stem", required=True)
ap.add_argument("--n-lam", type=int, default=100)
ap.add_argument("--eps", type=float, default=0.01)
ap.add_argument("--tols", default="1e-4,1e-6,1e-8")
ap.add_argument("--solvers", default="celer,sklearn")
ap.add_argument("--max-iter", type=int, default=100000)
ap.add_argument("--out", required=True)
a = ap.parse_args()

A = sp.load_npz(a.stem + ".A.npz").tocsc()
meta = np.load(a.stem + ".meta.npz")
b = np.asarray(meta["b"], float)
m = A.shape[0]
lam_max = float(np.max(np.abs(A.T @ b)))
lams = lam_max * np.logspace(0.0, np.log10(a.eps), a.n_lam)
alphas = lams / m                                   # sklearn/celer convention

import celer, sklearn
from celer import celer_path
from sklearn.linear_model import lasso_path


def certify(X):
    """Per-lambda primal value and duality gap, identical formula to lasso_path.py's."""
    R = A @ X - b[:, None]
    P = 0.5 * np.sum(R * R, axis=0) + lams * np.sum(np.abs(X), axis=0)
    r = -R
    AtR = A.T @ r
    scale = np.maximum(lams, np.max(np.abs(AtR), axis=0))
    theta = r / scale[None, :]
    resid = b[:, None] - lams[None, :] * theta
    D = 0.5 * float(b @ b) - 0.5 * np.sum(resid * resid, axis=0)
    g = np.maximum(P - D, 0.0)
    return P, g, g / (1.0 + np.abs(P))


out = {"stem": a.stem, "n": A.shape[0], "p": A.shape[1], "nnz": int(A.nnz), "n_lam": int(lams.size),
       "eps": a.eps, "lam_max": lam_max, "lams": lams.tolist(),
       "versions": {"celer": celer.__version__, "sklearn": sklearn.__version__},
       "threads": os.environ.get("OMP_NUM_THREADS"), "runs": []}
for name in a.solvers.split(","):
    for tol in [float(t) for t in a.tols.split(",")]:
        try:
            t0 = time.perf_counter()
            if name == "celer":
                _, coefs, _ = celer_path(A, b, pb="lasso", alphas=alphas, tol=tol, max_iter=a.max_iter, verbose=0)
            else:
                _, coefs, _ = lasso_path(A, b, alphas=alphas, tol=tol, max_iter=a.max_iter)
            dt = time.perf_counter() - t0
            X = np.asarray(coefs, float)                 # (p, L) in the order of `alphas`
            P, g, rel = certify(X)
            rec = {"solver": name, "tol": tol, "t_s": dt, "obj": P.tolist(), "rel": rel.tolist(),
                   "n_certified": int((rel <= tol).sum()), "max_rel": float(np.max(rel)),
                   "support_max": int(np.max((X != 0).sum(axis=0)))}
        except Exception as e:
            rec = {"solver": name, "tol": tol, "error": repr(e)[:300]}
        out["runs"].append(rec)
        print(json.dumps({k: v for k, v in rec.items() if k not in ("obj", "rel")}), flush=True)
        json.dump(out, open(a.out, "w"))
print("BASELINE_PATH_DONE", a.out, flush=True)
