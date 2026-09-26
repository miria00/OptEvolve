"""Every baseline, run strictly out of the box.

RULE (Mert, 2026-09-24): nothing may be done to make a baseline more
performant.  No tolerance overrides, no iteration-cap changes, no swapping a
solver's linear algebra, no amortizing canonicalization across instances, no
warm starts we supply.  Whatever the default constructor gives is what runs.

This file re-runs the baselines that earlier violated that rule and records
the out-of-the-box numbers as the headline.  Where an earlier, faster form was
measured it is kept only as a disclosed control, never as the reported number.
"""
import argparse, json, sys, time
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--regime", required=True, choices=["unmix", "tvdenoise"])
ap.add_argument("--pixels", type=int, default=50)
ap.add_argument("--out", default="")
a = ap.parse_args()
res = {"rule": "defaults only, no tuning, naive usage"}

if a.regime == "unmix":
    sys.path.insert(0, "/home/miria/OptEvolve/v2/scripts/discovery/hsi")
    import scipy.io as sio, cvxpy as cp
    from fcls_core import fw_gap_and_feas
    d = sio.loadmat("/home/miria/data/hsi/Cuprite.mat")
    D = np.asarray(d["D"], float); Y = np.asarray(d["Y"], float)[:, : a.pixels]
    n, m = D.shape[1], Y.shape[1]
    res["problem"] = {"bands": D.shape[0], "library": n, "pixels_full_scene": 47750}
    for solver in ("CLARABEL", "ECOS", "SCS", "OSQP"):
        t0 = time.perf_counter(); X = np.zeros((n, m)); ok = 0
        try:
            for i in range(m):
                # NAIVE: the problem is rebuilt per pixel.  No cp.Parameter,
                # because amortizing canonicalization is an optimization.
                x = cp.Variable(n)
                prob = cp.Problem(cp.Minimize(0.5 * cp.sum_squares(D @ x - Y[:, i])),
                                  [x >= 0, cp.sum(x) == 1])
                prob.solve(solver=solver)          # DEFAULT SETTINGS ONLY
                X[:, i] = np.maximum(x.value, 0.0) if x.value is not None else 1.0 / n
                ok += x.value is not None
            el = time.perf_counter() - t0
            gw, gmed, feas, _ = fw_gap_and_feas(D, Y, X)
            res[solver] = {"s_per_pixel": el / m, "solved": ok, "worst_gap": gw,
                           "median_gap": gmed, "feas": feas,
                           "full_scene_s": el / m * 47750}
        except Exception as e:
            res[solver] = {"error": "%s: %s" % (type(e).__name__, str(e)[:120]),
                           "s_per_pixel": (time.perf_counter() - t0) / max(ok, 1)}

elif a.regime == "tvdenoise":
    sys.path.insert(0, "/home/miria/OptEvolve/v2/scripts/discovery/tv")
    from tv_core import primal
    from skimage.restoration import denoise_tv_chambolle
    import skimage
    res["skimage_version"] = skimage.__version__
    for S in (128, 256, 512, 1024, 1356):
        f = f"/home/miria/data/tv/0801_{S}_noisy.npy"
        try:
            y = np.load(f)
        except FileNotFoundError:
            continue
        pstar = json.load(open(
            f"/home/miria/OptEvolve/v2/results/tv_20260924/ours_4090f64_{S}.json"))["primal"]
        t = time.perf_counter()
        x = denoise_tv_chambolle(y, weight=0.1)    # ONLY the weight; all else default
        el = time.perf_counter() - t
        P = primal(x, y, 0.1)
        res[str(S)] = {"wall": el, "rel_gap": (P - pstar) / (1.0 + abs(pstar)),
                       "defaults": {"eps": 2e-4, "max_num_iter": 200}}

print(json.dumps(res, indent=1))
if a.out:
    open(a.out, "w").write(json.dumps(res, indent=1))
