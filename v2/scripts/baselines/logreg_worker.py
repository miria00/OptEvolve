"""One elastic-net logistic solve by author code, inside that solver's own environment.

Objective (all solvers): (1/n) sum log(1+exp(-y x^T w)) + lam (alpha ||w||_1 + (1-alpha)/2 ||w||^2),
no intercept. Mappings: skglm SparseLogisticRegression(alpha=lam, l1_ratio=alpha); scikit-learn and cuML
LogisticRegression(penalty="elasticnet", C=1/(n lam), l1_ratio=alpha); adelie grpnet binomial with a
lambda path from lam_max down to lam (glmnet style, the path is timed). Writes w to --w-out and prints
one JSON line {solver, seconds, n_iter}. Solve time excludes data loading.
"""
import argparse, json, time
import numpy as np
ap = argparse.ArgumentParser()
for k, t in (("--solver", str), ("--data", str), ("--lam", float), ("--alpha", float), ("--tol", float),
             ("--max-iter", int), ("--w-out", str)):
    ap.add_argument(k, type=t, required=True)
ap.add_argument("--precision", default="float32", choices=("float32", "float64"),
                help="cuML only: the dtype cuML fits in (its QN solver is templated on both)")
a = ap.parse_args()
X = np.load(a.data + "/X.npy", mmap_mode="r"); y = np.load(a.data + "/y.npy").astype(np.float64)
n = X.shape[0]; y01 = (y > 0).astype(np.float64)
# warm-up fits on a small slice, untimed: JIT (numba, CUDA) is excluded, as for our own solver
wi = slice(0, min(n, 256))
if a.solver == "skglm":
    from skglm import SparseLogisticRegression
    Xd = np.ascontiguousarray(X, dtype=np.float64)
    SparseLogisticRegression(alpha=a.lam, l1_ratio=a.alpha, max_iter=2, fit_intercept=False).fit(Xd[wi], y[wi])
    m = SparseLogisticRegression(alpha=a.lam, l1_ratio=a.alpha, tol=a.tol, max_iter=a.max_iter, fit_intercept=False)
    t0 = time.perf_counter(); m.fit(Xd, y); t = time.perf_counter() - t0
    w, it = np.asarray(m.coef_).ravel(), int(getattr(m, "n_iter_", -1))
elif a.solver == "sklearn_saga":
    from sklearn.linear_model import LogisticRegression
    Xd = np.ascontiguousarray(X, dtype=np.float64)
    m = LogisticRegression(penalty="elasticnet", solver="saga", C=1.0 / (n * a.lam), l1_ratio=a.alpha,
                           fit_intercept=False, tol=a.tol, max_iter=a.max_iter)
    t0 = time.perf_counter(); m.fit(Xd, y); t = time.perf_counter() - t0
    w, it = m.coef_.ravel(), int(np.max(m.n_iter_))
elif a.solver == "adelie":
    import adelie as ad
    Xd = np.asfortranarray(X, dtype=np.float64)
    lmax = float(np.max(np.abs(Xd.T @ (y01 - 0.5))) / n / max(a.alpha, 1e-12))
    path = np.geomspace(lmax, a.lam, 30)
    t0 = time.perf_counter()
    st = ad.grpnet(X=Xd, glm=ad.glm.binomial(y=y01), alpha=a.alpha, lmda_path=path, intercept=False,
                   tol=a.tol, progress_bar=False)
    t = time.perf_counter() - t0
    w, it = np.asarray(st.betas[-1].toarray()).ravel(), len(path)
elif a.solver == "cuml":
    import cupy as cp
    from cuml.linear_model import LogisticRegression
    dt = np.float32 if a.precision == "float32" else np.float64
    Xg, yg = cp.asarray(np.ascontiguousarray(X, dtype=dt)), cp.asarray(y01, dtype=dt)
    LogisticRegression(penalty="elasticnet", C=1.0, l1_ratio=a.alpha, fit_intercept=False, max_iter=5).fit(Xg[wi], yg[wi])
    m = LogisticRegression(penalty="elasticnet", C=1.0 / (n * a.lam), l1_ratio=a.alpha, fit_intercept=False,
                           tol=a.tol, max_iter=a.max_iter)
    cp.cuda.Stream.null.synchronize(); t0 = time.perf_counter(); m.fit(Xg, yg); cp.cuda.Stream.null.synchronize()
    t = time.perf_counter() - t0
    w, it = cp.asnumpy(m.coef_).ravel().astype(np.float64), int(np.max(cp.asnumpy(cp.asarray(m.n_iter_))))
else:
    raise SystemExit(f"unknown solver {a.solver}")
np.save(a.w_out, w)
print(json.dumps({"solver": a.solver, "seconds": t, "n_iter": it, "precision": a.precision if a.solver == "cuml" else "float64"}))
