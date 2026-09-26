"""Summarize the two synthesis arms from their SkyDiscover checkpoints and export the top programs as operator files.

Spec arm programs carry certificate metrics; black-box arm programs carry end-to-end metrics. Top programs of BOTH arms
are exported so they can be put through the SAME independent checks afterwards (full-contract certificate, in-solver
decision flips on the ladder, and the generalization test on unseen instances).
"""
import argparse, glob, json, os

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--export", required=True)
ap.add_argument("--top", type=int, default=3)
a = ap.parse_args()
os.makedirs(a.export, exist_ok=True)
summary = {}
for arm in ("sky27b", "skybb"):
    cks = sorted(glob.glob(os.path.join(a.root, arm, "out", "checkpoints", "checkpoint_*")),
                 key=lambda p: int(p.rsplit("_", 1)[1]))
    if not cks:
        print(arm, "no checkpoint yet"); continue
    progs = [json.load(open(f)) for f in glob.glob(os.path.join(cks[-1], "programs", "*.json"))]
    progs.sort(key=lambda p: p.get("iteration_found", 0))
    ms = [p["metrics"] for p in progs]
    if arm == "sky27b":
        cats = {"crash": sum(m.get("combined_score", 0) <= 0.05 for m in ms),
                "not_exact": sum(0.05 < m.get("combined_score", 0) < 1.0 for m in ms),
                "certified_below_spec": sum(1.0 <= m.get("combined_score", 0) < 2.0 for m in ms),
                "certified_meets_spec": sum(m.get("combined_score", 0) >= 2.0 for m in ms)}
        best_t = sorted((m.get("cand_us_target") or 9e9, p["id"]) for p, m in zip(progs, ms)
                        if m.get("certified", 0) >= 1.0 and m.get("cand_us_target"))
        cats["best_certified_us_target"] = best_t[0][0] if best_t else None
    else:
        cats = {"crash": sum(m.get("combined_score", 0) <= 0.05 for m in ms),
                "not_accepted": sum(0.05 < m.get("combined_score", 0) and not m.get("accepted") for m in ms),
                "accepted": sum(bool(m.get("accepted")) for m in ms),
                "beats_T_fixed": sum(bool(m.get("accepted")) and m.get("combined_score", 0) > 1.0 for m in ms)}
        acc = sorted((m.get("total_s"), p["id"]) for p, m in zip(progs, ms) if m.get("accepted"))
        cats["best_total_s"] = acc[0][0] if acc else None
    cats["programs"] = len(progs)
    cats["checkpoint"] = os.path.basename(cks[-1])
    best_so_far, curve = -1, []
    for p, m in zip(progs, ms):
        best_so_far = max(best_so_far, m.get("combined_score", 0))
        curve.append((p.get("iteration_found", 0), round(best_so_far, 3)))
    cats["best_score_curve"] = curve[:: max(1, len(curve) // 12)]
    summary[arm] = cats
    for rank, p in enumerate(sorted(progs, key=lambda p: -p["metrics"].get("combined_score", 0))[: a.top]):
        path = os.path.join(a.export, f"{arm}_top{rank}.py")
        open(path, "w").write(p["solution"])
        cats.setdefault("exported", []).append({"file": path, "iteration": p.get("iteration_found"),
                                                "metrics": p["metrics"]})
print(json.dumps(summary, indent=1, default=str))
