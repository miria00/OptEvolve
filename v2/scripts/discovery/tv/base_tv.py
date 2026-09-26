"""Baselines for isotropic TV denoising, each graded by OUR certificate.

scikit-image's denoise_tv_chambolle is the standard packaged solver.  Its
weight convention is CHECKED against our objective rather than assumed: a
wrong mapping does not raise, it silently makes the baseline solve a different
problem, which is the most dishonest failure available in a comparison.

Each baseline is offered a ladder of its own tolerance settings and scored by
the fastest setting whose returned image passes our relative primal-dual gap.
"""
import argparse, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tv_core import certify, dual_from_x, primal


def check_convention(y, lam):
    """Find which weight makes skimage minimize the same functional we do."""
    from skimage.restoration import denoise_tv_chambolle
    out = {}
    for name, w in (("lam", lam), ("lam_over_n", lam / y.size), ("2lam", 2 * lam)):
        x = denoise_tv_chambolle(y, weight=w, max_num_iter=2000, eps=1e-12)
        out[name] = primal(x, y, lam)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--npy", required=True)
    ap.add_argument("--lam", type=float, default=0.1)
    ap.add_argument("--check-convention", action="store_true")
    ap.add_argument("--budget", type=float, default=600.0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    y = np.load(a.npy)
    if a.check_convention:
        print(json.dumps(check_convention(y, a.lam), indent=1)); raise SystemExit

    from skimage.restoration import denoise_tv_chambolle
    rows = []
    for eps in (1e-4, 1e-6, 1e-8, 1e-10, 1e-12):
        for mi in (200, 2000, 20000, 200000):
            t = time.perf_counter()
            x = denoise_tv_chambolle(y, weight=a.lam, max_num_iter=mi, eps=eps)
            el = time.perf_counter() - t
            g, P, G = dual_from_x(x, y, a.lam)
            rows.append({"eps": eps, "max_iter": mi, "wall": el, "gap": g, "P": P})
            if el > a.budget:
                break
        if el > a.budget:
            break
    best = {}
    for tgt in (1e-3, 1e-4, 1e-6, 1e-8):
        ok = [r for r in rows if r["gap"] <= tgt]
        if ok:
            b = min(ok, key=lambda r: r["wall"])
            best[str(tgt)] = {"t": b["wall"], "eps": b["eps"], "max_iter": b["max_iter"],
                              "gap": b["gap"]}
    r = {"arm": "skimage_tv_chambolle", "shape": list(y.shape), "lam": a.lam,
         "ladder": rows, "hit": best}
    print(json.dumps(r, indent=1))
    if a.out: open(a.out, "w").write(json.dumps(r, indent=1))
