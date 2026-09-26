"""Merge QP head-to-head shards and summarise at matched accuracy.

Every solver is judged by the same certificate (qp_rel_kkt <= tol). Time is a
shifted geometric mean (shift 1 ms) over the instances a solver solves, and
over all instances with failures charged a penalty time; head-to-head ratios
are computed only on instances both solvers solve.
"""
import glob, json, sys
import numpy as np

files = sorted(glob.glob(sys.argv[1] if len(sys.argv) > 1 else "results/qp_20260910/full_shard*.json"))
rows = [r for f in files for r in json.load(open(f))["rows"]]
arms = sorted({k for r in rows for k in r if k.startswith("ours_")})
solvers = ["osqp", "scs", "clarabel"] + arms
T = lambda v: v.get("t_s", v.get("t_exec_s"))
SHIFT, PEN = 1e-3, 60.0

def sgm(ts):
    ts = np.asarray(ts, float)
    return float(np.exp(np.mean(np.log(ts + SHIFT))) - SHIFT) if len(ts) else float("nan")

print(f"{len(rows)} instances from {len(files)} shard files")
print(f"{'solver':14s} solved  sgm(solved)  sgm(all, fail={PEN:.0f}s)  obj-rel-err>1e-3")
for s in solvers:
    ok = [r for r in rows if r.get(s, {}).get("passed")]
    bad_obj = sum(1 for r in ok if "obj" in r.get("ref", {}) and
                  abs(r[s]["obj"] - r["ref"]["obj"]) > 1e-3 * max(1.0, abs(r["ref"]["obj"])))
    all_t = [T(r[s]) if r.get(s, {}).get("passed") else PEN for r in rows if s in r]
    print(f"{s:14s} {len(ok):4d}/{len(rows)}  {sgm([T(r[s]) for r in ok]):9.4f}s  {sgm(all_t):12.4f}s  {bad_obj:6d}")

print("\nhead to head on instances both solve (ratio = ours/baseline time; <1 means ours faster)")
for arm in arms:
    for b in ("osqp", "scs", "clarabel"):
        both = [r for r in rows if r.get(arm, {}).get("passed") and r.get(b, {}).get("passed")]
        if not both:
            continue
        rat = np.array([T(r[arm]) / max(T(r[b]), 1e-7) for r in both])
        wins = [r["name"] for r in both if T(r[arm]) < T(r[b])]
        print(f"  {arm:12s} vs {b:9s} n={len(both):3d} median {np.median(rat):8.2f}x  "
              f"geo {np.exp(np.mean(np.log(rat))):8.2f}x  ours faster on {len(wins)}: {wins[:8]}")

best = []
for r in rows:
    ts = [T(r[a]) for a in arms if r.get(a, {}).get("passed")]
    best.append(min(ts) if ts else None)
solved_any = sum(t is not None for t in best)
print(f"\nours (best arm per instance, oracle choice): solved {solved_any}/{len(rows)}")
only_ours = [r["name"] for r in rows if any(r.get(a, {}).get("passed") for a in arms) and not r.get("osqp", {}).get("passed")]
print("solved by ours but not OSQP:", only_ours)
err = [(r["name"], r[a].get("error")) for r in rows for a in arms if r.get(a, {}).get("error")]
print("our errors:", err[:10])
