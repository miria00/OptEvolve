"""Generic conic baseline for TV deblurring.

There is no packaged TV-deblurring solver in the standard imaging stack, so
the field for this problem is generic conic solvers driven through CVXPY.
This records where they stop being able to form the problem at all, which is a
reachability statement and is independent of machine load.
"""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deblur_core import make_kernel, blur, primal

ap = argparse.ArgumentParser()
ap.add_argument("--npy", required=True)
ap.add_argument("--lam", type=float, default=0.02)
ap.add_argument("--mu", type=float, default=1e-6)
ap.add_argument("--solver", default="SCS")
ap.add_argument("--budget", type=float, default=300.0)
ap.add_argument("--out", default="")
a = ap.parse_args()

clean = np.load(a.npy.replace("_noisy", "_clean"))
Kf = make_kernel(clean.shape, sigma=2.0)
rng = np.random.default_rng(0)
y = blur(clean, Kf) + 0.01 * rng.standard_normal(clean.shape)
n0, n1 = y.shape

t0 = time.perf_counter()
res = {"solver": a.solver, "shape": [n0, n1], "lam": a.lam, "mu": a.mu}
try:
    import cvxpy as cp
    import scipy.sparse as sp
    # Sparse circular blur operator: the kernel has 15x15 support, so each row
    # has 225 nonzeros.  This is the baseline's best form; a dense operator
    # would be a strawman.
    K = np.fft.irfft2(Kf, s=(n0, n1))
    nz = np.argwhere(np.abs(K) > 1e-15)
    vals = K[nz[:, 0], nz[:, 1]]
    idx = np.arange(n0 * n1).reshape(n0, n1)
    rows, cols, data = [], [], []
    ii, jj = np.divmod(np.arange(n0 * n1), n1)
    for (di, dj), v in zip(nz, vals):
        src = idx[(ii - di) % n0, (jj - dj) % n1]
        rows.append(np.arange(n0 * n1)); cols.append(src)
        data.append(np.full(n0 * n1, v))
        if time.perf_counter() - t0 > a.budget:
            raise TimeoutError("operator assembly exceeded budget")
    A = sp.csr_matrix((np.concatenate(data),
                       (np.concatenate(rows), np.concatenate(cols))),
                      shape=(n0 * n1, n0 * n1))
    xv = cp.Variable(n0 * n1)
    X = cp.reshape(xv, (n0, n1), order="C")
    obj = (0.5 * cp.sum_squares(A @ xv - y.ravel())
           + 0.5 * a.mu * cp.sum_squares(xv) + a.lam * cp.tv(X))
    prob = cp.Problem(cp.Minimize(obj))
    prob.solve(solver=a.solver, verbose=False)
    el = time.perf_counter() - t0
    xs = np.array(xv.value).reshape(n0, n1)
    res.update({"status": prob.status, "wall": el, "nnz": int(A.nnz),
                "P": primal(xs, y, Kf, a.lam, a.mu)})
except Exception as e:
    res.update({"status": "FAILED", "error": "%s: %s" % (type(e).__name__, str(e)[:200]),
                "wall": time.perf_counter() - t0})
print(json.dumps(res, indent=1))
if a.out: open(a.out, "w").write(json.dumps(res, indent=1))
