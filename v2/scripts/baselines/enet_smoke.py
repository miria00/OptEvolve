"""Elastic net on real LIBSVM data against author code, on ONE common objective.

Each package parameterises elastic net slightly differently, so every solver's
returned coefficients are re-scored on the SAME objective:
    P(w) = (1/(2n))||y - Xw||^2 + a*r*||w||_1 + (a*(1-r)/2)||w||^2
A different P means a different problem was solved, and the timing is void.
"""
import sys, time, json
import numpy as np, scipy.sparse as sp
from sklearn.datasets import load_svmlight_file

path = sys.argv[1]; frac = float(sys.argv[2]); r = float(sys.argv[3]); tol = float(sys.argv[4])
X, y = load_svmlight_file(path); X = sp.csc_matrix(X); y = np.asarray(y, float)
n, d = X.shape
lam_max = np.abs(X.T @ y).max() / n
a = frac * lam_max
def P(w):
    res = y - X @ w
    return 0.5 * res @ res / n + a * r * np.abs(w).sum() + 0.5 * a * (1 - r) * w @ w
print(f"{path.split('/')[-1]}: n={n} d={d} nnz={X.nnz}  a={a:.3e} (= {frac} lam_max)  l1_ratio={r}  tol={tol}")

out = {}
def run(name, fn):
    try:
        t0 = time.perf_counter(); w = fn(); dt = time.perf_counter() - t0
        w = np.asarray(w, float).ravel()
        out[name] = {"t_s": dt, "P": float(P(w)), "nnz_w": int((np.abs(w) > 1e-12).sum())}
    except Exception as e:
        out[name] = {"error": f"{type(e).__name__}: {str(e)[:150]}"}

from sklearn.linear_model import ElasticNet as SkEN
run("sklearn", lambda: SkEN(alpha=a, l1_ratio=r, fit_intercept=False, tol=tol, max_iter=100000).fit(X, y).coef_)
from celer import ElasticNet as CeEN
run("celer", lambda: CeEN(alpha=a, l1_ratio=r, fit_intercept=False, tol=tol, max_iter=1000).fit(X, y).coef_)
from skglm import ElasticNet as SgEN
run("skglm", lambda: SgEN(alpha=a, l1_ratio=r, fit_intercept=False, tol=tol, max_iter=1000).fit(X, y).coef_)
def adelie_fit():
    import adelie as ad
    Xd = np.asfortranarray(X.toarray()) if X.shape[0] * X.shape[1] <= 4e8 else None
    if Xd is None: raise RuntimeError("dense copy too large for this smoke test")
    st = ad.grpnet(X=Xd, glm=ad.glm.gaussian(y=y), lmda_path=np.array([a]), alpha=r,
                   intercept=False, tol=tol, progress_bar=False)
    return st.betas[-1].toarray().ravel()
run("adelie", adelie_fit)

best = min((v["P"] for v in out.values() if "P" in v), default=np.nan)
for k, v in out.items():
    if "P" in v:
        print(f"  {k:8s} {v['t_s']:8.3f}s   P={v['P']:.10f}   P-best={v['P']-best:+.2e}   nnz(w)={v['nnz_w']}")
    else:
        print(f"  {k:8s} {v['error']}")
