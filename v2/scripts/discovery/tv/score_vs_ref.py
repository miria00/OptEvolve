"""Score every arm against a high-accuracy reference objective.

An earlier version graded a competitor by a dual point CONSTRUCTED from its
own iterate.  That is unfair: a weak dual construction reports a large gap for
a nearly optimal image, so the competitor is charged for our bound rather than
for its answer.  The protocol this project uses elsewhere is the right one:
compute a reference objective once, outside every timed region, and grade
every arm by its objective gap relative to 1 + |P*|.

P* here comes from our own run certified to 1e-8 by a genuine primal-dual gap,
so P* is itself known to that accuracy and the grading is sound for anything
above it.
"""
import argparse, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tv_core import primal, certify

ap = argparse.ArgumentParser()
ap.add_argument("--npy", required=True)
ap.add_argument("--lam", type=float, default=0.1)
ap.add_argument("--pstar", type=float, required=True)
ap.add_argument("--budget", type=float, default=120.0)
ap.add_argument("--out", default="")
a = ap.parse_args()
y = np.load(a.npy)
from skimage.restoration import denoise_tv_chambolle

rows = []
for eps in (1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 1e-14):
    for mi in (200, 1000, 5000, 25000, 100000):
        t = time.perf_counter()
        x = denoise_tv_chambolle(y, weight=a.lam, max_num_iter=mi, eps=eps)
        el = time.perf_counter() - t
        P = primal(x, y, a.lam)
        rel = (P - a.pstar) / (1.0 + abs(a.pstar))
        rows.append({"eps": eps, "max_iter": mi, "wall": el, "P": P, "rel": rel})
        if el > a.budget:
            break
    if el > a.budget:
        break
best = {}
for tgt in (1e-3, 1e-4, 1e-6, 1e-8):
    ok = [r for r in rows if r["rel"] <= tgt]
    if ok:
        b = min(ok, key=lambda r: r["wall"])
        best[str(tgt)] = {"t": b["wall"], "eps": b["eps"], "max_iter": b["max_iter"],
                          "rel": b["rel"]}
res = {"arm": "skimage_tv_chambolle", "shape": list(y.shape), "lam": a.lam,
       "pstar": a.pstar, "best_rel": min(r["rel"] for r in rows),
       "ladder": rows, "hit": best}
print(json.dumps(res, indent=1))
if a.out: open(a.out, "w").write(json.dumps(res, indent=1))
