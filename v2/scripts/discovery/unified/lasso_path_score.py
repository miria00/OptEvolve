"""Join the path runs and score them the way the corrected protocol requires.

Two rules, both of which matter and both of which cut against us as often as for us:

  * VIRTUAL BEST OVER SETTINGS. A solver's tolerance parameter means different things in different libraries, so no
    arm is scored at the setting whose name matches the bar. For each arm and each bar tau, we take the best result
    over every setting that arm was run at: the largest number of certified cells, and among settings achieving that,
    the smallest wall time.
  * ONE CERTIFICATE FOR EVERYONE. A lambda cell counts as certified at tau only if the lasso duality gap of the
    returned column satisfies gap / (1 + |P|) <= tau. Nobody is credited for their own stopping rule.
"""
import argparse, glob, json, os

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True)
ap.add_argument("--bars", default="1e-4,1e-6,1e-8")
ap.add_argument("--sizes", default="10000,50000,100000")
a = ap.parse_args()
BARS = [float(x) for x in a.bars.split(",")]


def load(path, arm):
    if not os.path.exists(path):
        return []
    d = json.load(open(path))
    out = []
    for r in d["runs"]:
        if "rel" not in r:
            continue
        out.append({"arm": r.get("solver", arm), "setting": r["tol"], "t_s": r.get("t_s", r.get("total_s")),
                    "rel": np.asarray(r["rel"], float)})
    return out


for n in [int(x) for x in a.sizes.split(",")]:
    runs = load(os.path.join(a.dir, f"path_base_n{n}.json"), "spec")
    runs += load(os.path.join(a.dir, f"path_base_tight_n{n}.json"), "spec")  # 1e-10/-12/-14 settings
    for tag, arm in (("path_ours_f64_n%d.json" % n, "ours[H100 fp64]"),
                     ("path_ours_n%d.json" % n, "ours[4090 fp32]")):
        p = os.path.join(a.dir, tag)
        if os.path.exists(p):
            d = json.load(open(p))
            for r in d["runs"]:
                runs.append({"arm": arm, "setting": r["tol"], "t_s": r["total_s"],
                             "rel": np.asarray(r["rel"], float)})
    if not runs:
        continue
    L = runs[0]["rel"].size
    arms = sorted({r["arm"] for r in runs})
    print(f"\n=== n={n}  L={L} lambdas   (certified cells at bar / time of the cheapest setting reaching it)")
    print(f"{'bar':>8} " + " ".join(f"{x:>22s}" for x in arms))
    for bar in BARS:
        cells = []
        for arm in arms:
            best_c, best_t = -1, None
            for r in runs:
                if r["arm"] != arm:
                    continue
                c = int((r["rel"] <= bar).sum())
                if c > best_c or (c == best_c and r["t_s"] < best_t):
                    best_c, best_t = c, r["t_s"]
            cells.append(f"{best_c:>3d}/{L:<3d} {best_t:>8.2f}s" if best_c >= 0 else "          --")
        print(f"{bar:>8.0e} " + " ".join(f"{c:>22s}" for c in cells))
