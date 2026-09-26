"""Baseline: per-pixel simplex-constrained least squares through CVXPY.

Runs in its own interpreter and its own venv.  Setup (canonicalization) is
inside the clock, as it is for our arm.  Every solver is offered a ladder of
its own tolerance settings; the reported time is the fastest setting whose
returned X passes OUR acceptance check, never the solver's own claim.
"""
import argparse, json, os, sys, time
import numpy as np, scipy.io as sio
import cvxpy as cp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fcls_core import fw_gap_and_feas

ap = argparse.ArgumentParser()
ap.add_argument("--mat", default="/home/miria/data/hsi/Cuprite.mat")
ap.add_argument("--solver", default="CLARABEL")
ap.add_argument("--pixels", type=int, default=200)
ap.add_argument("--eps", type=float, default=1e-8)
ap.add_argument("--budget", type=float, default=600.0)
ap.add_argument("--out", default="")
a = ap.parse_args()

d = sio.loadmat(a.mat)
D = np.asarray(d["D"], dtype=np.float64)
Y = np.asarray(d["Y"], dtype=np.float64)[:, : a.pixels]
n, m = D.shape[1], Y.shape[1]

t0 = time.perf_counter()
x = cp.Variable(n)
yp = cp.Parameter(D.shape[0])
prob = cp.Problem(cp.Minimize(0.5 * cp.sum_squares(D @ x - yp)),
                  [x >= 0, cp.sum(x) == 1])
X = np.zeros((n, m))
solved, failed = 0, 0
for i in range(m):
    if time.perf_counter() - t0 > a.budget:
        break
    yp.value = Y[:, i]
    try:
        kw = {}
        if a.solver in ("CLARABEL",):
            kw = dict(tol_gap_abs=a.eps, tol_gap_rel=a.eps, tol_feas=a.eps)
        elif a.solver in ("SCS",):
            kw = dict(eps_abs=a.eps, eps_rel=a.eps)
        elif a.solver in ("ECOS",):
            kw = dict(abstol=a.eps, reltol=a.eps, feastol=a.eps)
        prob.solve(solver=a.solver, **kw)
        if x.value is not None:
            X[:, i] = np.maximum(x.value, 0.0); solved += 1
        else:
            X[:, i] = 1.0 / n; failed += 1
    except Exception:
        X[:, i] = 1.0 / n; failed += 1
el = time.perf_counter() - t0
done = solved + failed
gw, gmed, feas, fsum = fw_gap_and_feas(D, Y[:, :done], X[:, :done]) if done else (None,)*4
r = {"solver": a.solver, "eps": a.eps, "pixels_requested": m, "pixels_done": done,
     "solved": solved, "failed": failed, "wall": el,
     "per_pixel_s": el / max(done, 1), "worst_gap": gw, "median_gap": gmed,
     "feas": feas, "obj": fsum,
     "projected_full_scene_s": el / max(done, 1) * 47750}
print(json.dumps(r, indent=1))
if a.out: open(a.out, "w").write(json.dumps(r, indent=1))
