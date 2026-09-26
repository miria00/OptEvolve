"""Merge Phase 2 holdout outputs into the three canonical files + tables.

- holdout_within.json: rep1 (all configs, canonical) + N=5 fresh-process
  wall-repeat statistics for the headline configs.
- holdout_ood_small.json: REMOTE part1 (champions+vanilla+per-instance)
  merged with REMOTE part2 (tuned + textbook best); rows re-aggregated.
- holdout_ood_medium.json: passed through (already canonical).
- Prints summary tables (SGM + solve rate per config per tier per tol,
  iterations and wall).
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
from phase2_common import aggregate_holdout  # noqa: E402

P2 = "/home/miria/OptimizationAtlas/v2/results/sprint/phase2"

HEADLINE = ("llm_seed43", "random_seed43", "tuned_pdhg", "vanilla_pdhg")


def load(name):
    with open(os.path.join(P2, name)) as fh:
        return json.load(fh)


def dump(doc, name):
    with open(os.path.join(P2, name), "w") as fh:
        json.dump(doc, fh, indent=2, allow_nan=False)
        fh.write("\n")
    print(f"wrote {name}")


def within():
    reps = [load(f"holdout_within_rep{i}.json") for i in range(1, 6)]
    base = reps[0]
    # N=5 fresh-process wall stats per (config, instance, tol), headline only
    wall = {}
    for rep_i, doc in enumerate(reps, 1):
        for r in doc["rows"]:
            if r["config"] not in HEADLINE:
                continue
            key = (r["config"], r["instance"], f"{r['tol']:g}")
            wall.setdefault(key, []).append(
                {
                    "rep": rep_i,
                    "status": r["status"],
                    "time_to_cert_s": r["time_to_cert_s"],
                    "iters_to_cert": r["iters_to_cert"],
                }
            )
    wall_stats = {}
    for (cfg, inst, tol), entries in sorted(wall.items()):
        ts = [e["time_to_cert_s"] for e in entries
              if e["status"] == "converged"]
        its = [e["iters_to_cert"] for e in entries
               if e["status"] == "converged"]
        wall_stats[f"{cfg}|{inst}|tol={tol}"] = {
            "n_repeats": len(entries),
            "n_converged": len(ts),
            "wall_median_s": float(np.median(ts)) if ts else None,
            "wall_min_s": float(np.min(ts)) if ts else None,
            "wall_max_s": float(np.max(ts)) if ts else None,
            "iters_median": float(np.median(its)) if its else None,
            "iters_min": int(np.min(its)) if its else None,
            "iters_max": int(np.max(its)) if its else None,
        }
    # per (config, tol) SGM-of-medians across the 5 fresh processes
    sgm_reps = {}
    for cfg in HEADLINE:
        for tol in ("0.0001", "1e-06"):
            per_rep = []
            for rep_i, doc in enumerate(reps, 1):
                agg = doc["aggregates"].get(
                    f"{cfg}|netflow_within|tol={tol}"
                )
                if agg:
                    per_rep.append(agg["sgm10_time_s_par1_at_cap"])
            if per_rep:
                sgm_reps[f"{cfg}|tol={tol}"] = {
                    "sgm10_per_rep": per_rep,
                    "median": float(np.median(per_rep)),
                    "min": float(np.min(per_rep)),
                    "max": float(np.max(per_rep)),
                }
    base["wall_repeats_n5_fresh_process"] = {
        "note": (
            "N=5 fresh-process repeats (rep1 = the all-config canonical "
            "run; reps 2-5 headline configs only); per-instance and "
            "per-config-SGM dispersion for wall claims"
        ),
        "per_instance": wall_stats,
        "sgm10_across_reps": sgm_reps,
    }
    dump(base, "holdout_within.json")
    return base


def ood_small():
    p1 = load("holdout_ood_small_part1.json")
    p2 = load("holdout_ood_small_part2.json")
    assert p1["config"]["ood_selection"]["small_selected"] == \
        p2["config"]["ood_selection"]["small_selected"], "selection mismatch"
    rows = p1["rows"] + p2["rows"]
    rows.sort(key=lambda r: (r["config"], r["tier"], r["tol"], r["instance"]))
    doc = dict(p1)
    doc["merged_from"] = ["holdout_ood_small_part1.json",
                         "holdout_ood_small_part2.json"]
    doc["config"]["configs"].update(p2["config"]["configs"])
    doc["rows"] = rows
    doc["aggregates"] = aggregate_holdout(rows)
    doc["total_wall_s"] = p1["total_wall_s"] + p2["total_wall_s"]
    dump(doc, "holdout_ood_small.json")
    return doc


def table(doc, tier, title):
    print(f"\n== {title} ==")
    hdr = (f"{'config':28s} {'tol':>6s} {'solve':>7s} {'SGM10 wall s':>12s} "
           f"{'SGM10 iters':>11s} {'cert':>9s} {'capT':>4s} {'capI':>4s} "
           f"{'rej':>4s}")
    print(hdr)
    for key, a in sorted(doc["aggregates"].items()):
        if a["tier"] != tier:
            continue
        it = a["sgm10_iters_certified_only"]
        print(
            f"{a['config']:28s} {a['tol']:>6g} {a['solve_rate']:>7.2f} "
            f"{a['sgm10_time_s_par1_at_cap']:>12.4g} "
            f"{(f'{it:.0f}' if it is not None else '-'):>11s} "
            f"{a['n_certified']:>4d}/{a['n_instances']:<4d} "
            f"{a['n_cap_time']:>4d} {a['n_cap_iters']:>4d} "
            f"{a['n_typecheck_rejected']:>4d}"
        )


def main():
    w = within()
    s = ood_small()
    m = load("holdout_ood_medium.json")
    table(w, "netflow_within", "WITHIN-distribution (10 fresh netflow, LOCAL)")
    table(s, "ood_small", "OOD small tier (48 Mittelmann+MIPLIB, REMOTE)")
    table(m, "ood_medium", "OOD medium tier (19 Mittelmann+MIPLIB, LOCAL)")
    print("\n== WITHIN N=5 fresh-process SGM10 wall dispersion (headline) ==")
    for k, v in w["wall_repeats_n5_fresh_process"][
            "sgm10_across_reps"].items():
        print(f"  {k:34s} median {v['median']:.4g}s "
              f"[{v['min']:.4g}, {v['max']:.4g}] n={len(v['sgm10_per_rep'])}")


if __name__ == "__main__":
    main()
