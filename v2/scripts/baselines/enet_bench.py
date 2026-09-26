"""Elastic net: author-code solvers on real LIBSVM data, one common objective.

    P(w) = (1/(2n))||y - Xw||^2 + a*r*||w||_1 + (a*(1-r)/2)||w||^2
Every solver's coefficients are re-scored on this P; a different P means a
different problem was solved. All solvers are run single-threaded so the
comparison is core for core.
"""
import sys, time, json, argparse, inspect
import numpy as np, scipy.sparse as sp
from sklearn.datasets import load_svmlight_file
ap = argparse.ArgumentParser()
ap.add_argument("path"); ap.add_argument("frac", type=float); ap.add_argument("r", type=float)
ap.add_argument("tol", type=float); ap.add_argument("--out", default="")
ap.add_argument("--center", action="store_true",
                help="center y; without it a target with a large mean (E2006) is solved by one near-constant feature acting as an intercept")
A = ap.parse_args()
t0 = time.perf_counter(); X, y = load_svmlight_file(A.path); t_load = time.perf_counter() - t0
X = sp.csc_matrix(X); y = np.asarray(y, float); n, d = X.shape
if A.center: y = y - y.mean()
lam_max = np.abs(X.T @ y).max() / n; a = A.frac * lam_max; r = A.r
def P(w):
    res = y - X @ w
    return 0.5 * res @ res / n + a * r * np.abs(w).sum() + 0.5 * a * (1 - r) * w @ w
name = A.path.split("/")[-1]
print(f"{name}: n={n} d={d} nnz={X.nnz} a={a:.3e} r={r} tol={A.tol} (load {t_load:.1f}s)", flush=True)
out = {}
def run(k, fn):
    try:
        t0 = time.perf_counter(); w = fn(); dt = time.perf_counter() - t0
        w = np.asarray(w, float).ravel()
        out[k] = {"t_s": dt, "P": float(P(w)), "nnz_w": int((np.abs(w) > 1e-12).sum())}
    except Exception as e:
        out[k] = {"error": f"{type(e).__name__}: {str(e)[:220]}"}
    v = out[k]
    print(f"  {k:8s} " + (f"{v['t_s']:8.3f}s  P={v['P']:.10f}  nnz(w)={v['nnz_w']}" if "P" in v else v["error"]), flush=True)
from celer import ElasticNet as CeEN
run("celer", lambda: CeEN(alpha=a, l1_ratio=r, fit_intercept=False, tol=A.tol, max_iter=1000).fit(X, y).coef_)
from skglm import ElasticNet as SgEN
run("skglm", lambda: SgEN(alpha=a, l1_ratio=r, fit_intercept=False, tol=A.tol, max_iter=1000).fit(X, y).coef_)
from sklearn.linear_model import ElasticNet as SkEN
run("sklearn", lambda: SkEN(alpha=a, l1_ratio=r, fit_intercept=False, tol=A.tol, max_iter=100000).fit(X, y).coef_)
import adelie as ad
def adelie_fit(tol=None):
    tol = A.tol if tol is None else tol
    kw = {}
    sig = inspect.signature(ad.matrix.sparse)
    if "n_threads" in sig.parameters: kw["n_threads"] = 1
    Xs = ad.matrix.sparse(X, **kw)
    gsig = inspect.signature(ad.grpnet)
    gk = dict(X=Xs, glm=ad.glm.gaussian(y=y), lmda_path=np.array([a]), alpha=r,
              intercept=False, tol=tol, progress_bar=False)
    if "n_threads" in gsig.parameters: gk["n_threads"] = 1
    st = ad.grpnet(**{k: v for k, v in gk.items() if k in gsig.parameters or k == "X"})
    return st.betas[-1].toarray().ravel()
run("adelie", adelie_fit)
# adelie stopped short of the others on rcv1 and news20; tighten its own
# tolerance until it reaches the same objective, so its time is at matched accuracy
for t in (1e-9, 1e-12):
    run(f"adelie@{t:g}", lambda t=t: adelie_fit(t))
if "error" in out.get("adelie", {}):
    print("   adelie.matrix:", [m for m in dir(ad.matrix) if not m.startswith("_")][:20])
    print("   grpnet params:", list(inspect.signature(ad.grpnet).parameters)[:30])
best = min((v["P"] for v in out.values() if "P" in v), default=float("nan"))
for k, v in out.items():
    if "P" in v: v["P_minus_best"] = v["P"] - best
print(f"  best P = {best:.12f}  (" + ", ".join(f"{k} {v['P_minus_best']:+.1e}" for k, v in out.items() if "P" in v) + ")")
if A.out:
    json.dump({"dataset": name, "n": n, "d": d, "nnz": int(X.nnz), "a": a, "r": r, "tol": A.tol,
               "threads": 1, "solvers": out, "P_best": best}, open(A.out, "w"), indent=1)
