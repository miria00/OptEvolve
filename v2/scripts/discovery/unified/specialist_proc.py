"""Worker for specialist.py: fits one lasso solver in the specialist environment and writes the coefficients.

Runs under an interpreter that has celer/skglm/scikit-learn and nothing of ours, so the two environments stay
disjoint. Imports are done before the clock starts, matching the baseline runners: a first `import` inside the timed
region once added 0.2 s to a baseline and made a loss look like a win.

scikit-learn's convention is $\\tfrac{1}{2m}\\|Ax-b\\|^2 + \\alpha\\|x\\|_1$, ours is $\\tfrac12\\|Ax-b\\|^2 +
\\lambda\\|x\\|_1$, so $\\alpha = \\lambda / m$. celer and skglm follow scikit-learn.
"""
import argparse, json, os, time

import numpy as np
import scipy.sparse as sp

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True)
ap.add_argument("--solver", required=True, choices=("celer", "skglm", "sklearn"))
ap.add_argument("--tol", type=float, required=True)
ap.add_argument("--max-iter", type=int, default=10000)
a = ap.parse_args()

A = sp.load_npz(os.path.join(a.dir, "A.npz")).tocsc()
meta = np.load(os.path.join(a.dir, "meta.npz"))
b = np.asarray(meta["b"], float)
lam = float(meta["lam"])
alpha = lam / A.shape[0]

if a.solver == "celer":
    from celer import Lasso as Est
elif a.solver == "skglm":
    from skglm import Lasso as Est
else:
    from sklearn.linear_model import Lasso as Est

kw = {"alpha": alpha, "tol": a.tol, "max_iter": a.max_iter, "fit_intercept": False}
est = Est(**kw)
t0 = time.perf_counter()
est.fit(A, b)
dt = time.perf_counter() - t0
x = np.asarray(est.coef_, float).ravel()
np.save(os.path.join(a.dir, "x.npy"), x)
print(json.dumps({"solver": a.solver, "tol": a.tol, "t_s": dt, "nnz_x": int((x != 0).sum())}))
