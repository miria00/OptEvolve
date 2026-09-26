"""Run our arm and the out-of-the-box baselines on cell-type deconvolution.

Same arm as the hyperspectral class, unchanged: the certified box-and-
hyperplane projection does not care whether the columns are mineral spectra or
immune-cell gene signatures.  Baselines run exactly as shipped.
"""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fcls_core import fw_gap_and_feas

ap = argparse.ArgumentParser()
ap.add_argument("--npz", default="/home/miria/data/deconv/deconv_lm22.npz")
ap.add_argument("--arm", required=True, choices=["ours", "nnls", "cvxpy"])
ap.add_argument("--device", default="gpu"); ap.add_argument("--dtype", default="f64")
ap.add_argument("--samples", type=int, default=0)
ap.add_argument("--solver", default="CLARABEL")
ap.add_argument("--budget", type=float, default=600.0)
ap.add_argument("--out", default="")
a = ap.parse_args()
d = np.load(a.npz, allow_pickle=True)
A = d["A"].astype(np.float64); B = d["B"].astype(np.float64)
Xtrue = d["Xtrue"].astype(np.float64)
if a.samples: B = B[:, : a.samples]; Xtrue = Xtrue[:, : a.samples]
n, m = A.shape[1], B.shape[1]
res = {"arm": a.arm, "genes": A.shape[0], "celltypes": n, "samples": m,
       "signature": os.path.basename(a.npz)}

if a.arm == "ours":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from ours_v5 import run
    r = run(A, B, a.device, a.dtype, [1e-3, 1e-4, 1e-6, 1e-8], budget_s=a.budget)
    res.update(r)
else:
    t0 = time.perf_counter()
    X = np.zeros((n, m))
    if a.arm == "nnls":
        # the naive approach the deconvolution literature uses: per sample,
        # sum-to-one imposed by the standard augmented row.  scipy defaults.
        from scipy.optimize import nnls
        w = 1e3
        Aa = np.vstack([A, w * np.ones((1, n))])
        for i in range(m):
            if time.perf_counter() - t0 > a.budget: break
            ba = np.concatenate([B[:, i], [w]])
            X[:, i] = nnls(Aa, ba)[0]
            done = i + 1
    else:
        import cvxpy as cp
        for i in range(m):
            if time.perf_counter() - t0 > a.budget: break
            x = cp.Variable(n)
            prob = cp.Problem(cp.Minimize(0.5 * cp.sum_squares(A @ x - B[:, i])),
                              [x >= 0, cp.sum(x) == 1])
            prob.solve(solver=a.solver)          # DEFAULTS
            X[:, i] = np.maximum(x.value, 0.0) if x.value is not None else 1.0 / n
            done = i + 1
    el = time.perf_counter() - t0
    gw, gmed, feas, _ = fw_gap_and_feas(A, B[:, :done], X[:, :done])
    res.update({"wall": el, "done": done, "s_per_sample": el / done,
                "worst_gap": gw, "median_gap": gmed, "feas": feas,
                "full_10k_s": el / done * m})
print(json.dumps(res, indent=1))
if a.out: open(a.out, "w").write(json.dumps(res, indent=1))
