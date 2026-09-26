"""Join the K-fold CV runs and score every arm on the one certificate, at the three bars.

Two rules, unchanged from lasso_path_score.py and both cutting against us as often as for us:

  * ONE CERTIFICATE FOR EVERYONE. A (fold, lambda) CELL counts at bar tau only if the lasso duality gap of that
    cell's own masked subproblem satisfies gap / (1 + |P|) <= tau. Nobody is credited for their own stopping rule.
  * VIRTUAL BEST OVER SETTINGS. The specialists are run at three tolerances and three fold-parallelism backends
    (one fold at a time, folds across threads, folds across processes) and scored at their best: for each bar, the
    most cells, and among settings reaching that count, the least wall time. Our arm is anytime, so its number for
    a bar is the first moment that bar was met in a single run.

Setup is timed on both sides: ours pays for the sparse upload, the power iteration and the JIT compile, theirs pays
for the CSR conversion and the per-fold row slicing.
"""
import argparse, glob, json, os, re

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True)
ap.add_argument("--bars", default="1e-4,1e-6,1e-8")
ap.add_argument("--sizes", default="10000,50000,100000")
ap.add_argument("--ks", default="5,10")
a = ap.parse_args()
BARS = [float(x) for x in a.bars.split(",")]


def ours(n, K):
    p = os.path.join(a.dir, f"cv_ours_n{n}_K{K}.json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    r = d["runs"][0]
    return {"cells": d["cells"], "bars": r["bars"], "total_s": r["total_s"], "setup_s": r["setup_s"],
            "iters": r["iters"], "max_rel": float(np.max(np.asarray(r["rel"], float))),
            "solver": r["solver"]}


def specs(n, K):
    rows = []
    for p in glob.glob(os.path.join(a.dir, f"cv_spec_n{n}_K{K}_*.json")):
        d = json.load(open(p))
        for r in d["runs"]:
            if "rel" not in r:
                continue
            rows.append({"solver": r["solver"], "tol": r["tol"], "backend": r["backend"],
                         "total_s": r["total_s"], "rel": np.asarray(r["rel"], float)})
    return rows


def vbest(rows, bar):
    """Most certified cells; among settings tied there, the cheapest."""
    best = None
    for r in rows:
        c = int((r["rel"] <= bar).sum())
        if best is None or c > best[0] or (c == best[0] and r["total_s"] < best[1]):
            best = (c, r["total_s"], f"{r['solver']}/{r['tol']:.0e}/{r['backend']}")
    return best


print("K-FOLD CROSS-VALIDATION: K folds x L=100 lambdas over ONE design matrix, scored by certified CELLS")
for n in [int(x) for x in a.sizes.split(",")]:
    for K in [int(x) for x in a.ks.split(",")]:
        o, sp_ = ours(n, K), specs(n, K)
        if o is None and not sp_:
            continue
        cells = o["cells"] if o else K * 100
        print(f"\n=== n={n}  K={K}  cells={cells} " + "=" * 30)
        hdr = f"{'bar':>6} | {'ours (GPU fp64)':>26} | {'celer best':>30} | {'sklearn best':>30} | speedup"
        print(hdr); print("-" * len(hdr))
        for bar in BARS:
            ob = o["bars"][f"{bar:.0e}"] if o else None
            oc, ot = (ob["cells"], ob["t_total_s"]) if ob else (-1, float("nan"))
            cel = vbest([r for r in sp_ if r["solver"] == "celer"], bar)
            skl = vbest([r for r in sp_ if r["solver"] == "sklearn"], bar)
            def fmt(x):
                return f"{x[0]:>4d}/{cells:<4d} {x[1]:>7.2f}s {x[2]:<14s}" if x else " " * 30
            # speedup is only meaningful against a specialist that completed the same field
            sp_full = [x for x in (cel, skl) if x and x[0] == cells]
            if oc == cells and sp_full:
                bt = min(x[1] for x in sp_full)
                spd = f"{bt/ot:>6.2f}x"
            elif oc == cells:
                bestc = max([x[0] for x in (cel, skl) if x] or [0])
                spd = f" none complete ({bestc}/{cells} best)"
            else:
                spd = "  --"
            print(f"{bar:>6.0e} | {oc:>4d}/{cells:<4d} {ot:>7.2f}s{'':<7s} | {fmt(cel)} | {fmt(skl)} | {spd}")
        if o:
            print(f"       ours: setup {o['setup_s']:.2f}s, {o['iters']} iters, final max_rel {o['max_rel']:.2e}")
        # every specialist setting, for the record
        for r in sorted(sp_, key=lambda r: (r["solver"], r["backend"], r["tol"])):
            cs = " ".join(f"{bar:.0e}:{int((r['rel']<=bar).sum()):>4d}" for bar in BARS)
            print(f"       {r['solver']:<8s} tol={r['tol']:.0e} {r['backend']:<10s} "
                  f"{r['total_s']:>8.2f}s  cells[{cs}]  max_rel {r['rel'].max():.2e}")
