"""Recompute the headroom assay's basin width against the RIGHT baseline.

BUG being corrected. headroom_assay.py computes W10 by comparing each SCREENED
setting (20k-iteration budget) against B0 measured at the FULL budget (200k).
That is not a basin width, it is a budget mismatch: almost nothing at 20k can
beat a 200k baseline, so W10 collapses to ~0% by construction and says nothing
about how broad the winning region is.

The fix is to compare like with like. This script runs B0 once per tolerance at
the SCREENING budget, then recomputes, per family:

  W10  = fraction of sampled settings that beat the screening-budget B0
         (more certified solves, or equal solves with >=10% lower SGM time)
  W0   = fraction that beat it at all (any improvement), for context
  best = the best screened setting under the same budget

It also reports the UNION of instances solved across families at full budget,
which the per-family counts understate: if relaxation rescues one instance and
restart rescues a different one, a search that can compose genes has more
headroom available than any single family's count reveals.

Nothing is re-tuned and no membership is touched; this only re-reads the saved
assay artifact and adds one baseline measurement per tolerance.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cuda")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from atlas.bench import mps_cell  # noqa: E402
from atlas.genome import genome_from_dict  # noqa: E402
from atlas.schemes.compile import solve_genome  # noqa: E402
from atlas.typing_ import TypeCheckError  # noqa: E402
from atlas.wrappers.km import Budget  # noqa: E402

PHASE3 = os.path.join(ROOT, "results", "sprint", "phase3")
ASSAY = os.path.join(PHASE3, "headroom_assay.json")
SPLIT = os.path.join(PHASE3, "cell_split.json")
OUT = os.path.join(PHASE3, "headroom_assay_corrected.json")
SGM_SHIFT = 10.0


def sgm(vals, shift=SGM_SHIFT):
    v = np.asarray([x for x in vals if np.isfinite(x)], dtype=float)
    return float(np.exp(np.mean(np.log(v + shift))) - shift) if v.size else float("inf")


def run(probs, d, tol, max_iters, cap_s):
    g = genome_from_dict(d)
    rows = {}
    for name, prob in probs.items():
        t0 = time.time()
        try:
            r = solve_genome(prob, g, Budget(max_iters=max_iters, tol=tol,
                                             sample_every=250))
            el = time.time() - t0
            h = r.rel_gap_history[np.isfinite(r.rel_gap_history)]
            gap = float(h[-1]) if h.size else float("nan")
            rows[name] = {"ok": bool(gap <= tol) and el <= cap_s,
                          "wall": el, "gap": gap}
        except TypeCheckError:
            rows[name] = {"ok": False}
    solved = sorted(n for n, r in rows.items() if r["ok"])
    times = [rows[n]["wall"] if rows[n]["ok"] else cap_s for n in rows]
    return {"n_solved": len(solved), "solved": solved, "sgm_s": sgm(times)}


def beats(c, b, margin):
    if not c or c.get("rejected") or b.get("rejected"):
        return False
    if c["n_solved"] > b["n_solved"]:
        return True
    return (c["n_solved"] == b["n_solved"]
            and c["sgm_s"] <= (1.0 - margin) * b["sgm_s"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--margin", type=float, default=0.10)
    args = ap.parse_args()

    assay = json.load(open(ASSAY))
    scr_iters = assay["config"]["screen_max_iters"]
    cap_s = assay["config"]["cap_s"]
    b0 = assay["B0"]
    split = json.load(open(SPLIT))

    probs = {}
    for e in split["calibration"]["instances"]:
        try:
            std = mps_cell.load_standard_form(
                os.path.join(ROOT, "data", "lp_suites", e["file"]))
            probs[e["name"]] = mps_cell.to_composite(std)
        except Exception:  # noqa: BLE001
            pass

    from atlas.schemes.rescale import lp_substrate_metric
    b0_genome = {"scheme": "pdhg",
                 "params": {"tau": b0["eta"] / b0["omega"],
                            "sigma": b0["eta"] * b0["omega"]},
                 "metric": lp_substrate_metric(pc_alpha=b0["alpha"])}

    out = {"source": os.path.basename(ASSAY), "margin": args.margin,
           "screen_max_iters": scr_iters, "tolerances": {}}
    for tolkey, fams in assay["tolerances"].items():
        tol = float(tolkey)
        print(f"\n=== tolerance {tol:g} ===", flush=True)
        base_scr = run(probs, b0_genome, tol, scr_iters, cap_s)
        base_full = fams["G0"]["base"]
        print(f"B0 @screen({scr_iters}): {base_scr['n_solved']}/{len(probs)} "
              f"sgm={base_scr['sgm_s']:.2f}s   |   "
              f"B0 @full: {base_full['n_solved']}/{len(probs)} "
              f"sgm={base_full['sgm_s']:.2f}s", flush=True)
        blk = {"b0_screen": base_scr,
               "b0_full": {k: base_full[k] for k in
                           ("n_solved", "solved", "sgm_s")},
               "families": {}}
        union = set(base_full["solved"])
        print(f"{'family':8s}{'best@full':>12s}{'W10':>7s}{'W_any':>8s}"
              f"{'rescued vs B0':>16s}", flush=True)
        for fname, f in fams.items():
            if fname in ("G0", "_union"):
                continue
            scr = [s for s in f.get("screened", []) if not s.get("rejected")]
            w10 = (sum(1 for s in scr if beats(s, base_scr, args.margin))
                   / len(scr)) if scr else 0.0
            wany = (sum(1 for s in scr if beats(s, base_scr, 0.0))
                    / len(scr)) if scr else 0.0
            conf = f.get("confirmed_full")
            rescued = []
            if conf:
                union |= set(conf["solved"])
                rescued = sorted(set(conf["solved"]) - set(base_full["solved"]))
            blk["families"][fname] = {
                "W10_vs_screen_baseline": w10, "W_any_vs_screen_baseline": wany,
                "n_screened": len(scr),
                "full_n_solved": conf["n_solved"] if conf else None,
                "full_sgm_s": conf["sgm_s"] if conf else None,
                "rescued_vs_B0": rescued,
                "lost_vs_B0": sorted(set(base_full["solved"])
                                     - set(conf["solved"])) if conf else [],
            }
            print(f"{fname:8s}{(conf['n_solved'] if conf else -1):>8d}/"
                  f"{len(probs):<3d}{w10:>6.0%}{wany:>8.0%}"
                  f"{len(rescued):>10d} new", flush=True)
        blk["union"] = {"n": len(union), "instances": sorted(union),
                        "b0_alone": base_full["n_solved"]}
        print(f"UNION across families: {len(union)}/{len(probs)} "
              f"(B0 alone {base_full['n_solved']})", flush=True)
        out["tolerances"][tolkey] = blk

    json.dump(out, open(OUT, "w"), indent=1, default=float)
    print(f"\nwrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
