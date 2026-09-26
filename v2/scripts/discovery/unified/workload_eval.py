"""BETTER THAN EITHER ALONE? Time every candidate on a workload so three policies can be compared on the same numbers.

Policies (analyzed by workload_policies.py from this file's output):
  TOOL ALONE   the principled tool with its shipped library: per instance it picks the fastest admissible candidate
               among std_raw, std_ruiz, reloc:bisect (variant: + the hand-written breakpoint and newton12)
  AGENT ALONE  the black-box arm's evolved operator, used the way that arm optimized it (relocated formulation), on
               every instance; if it fails on an instance, that instance counts as failed
  MERGED       the tool with its library extended by the operators the spec arm got CERTIFIED, deciding per instance
Workload: the knapsack spec ladder (where the tool generated specs) plus instances NO arm saw: other families, a
larger size, and real Maros-Meszaros instances. Timings on a quiet device: medians of interleaved repetitions.
"""
import argparse, json, os, sys, traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from uni import make_instance, permute_instance, diagnose, plan_block, build_formulation, reference, measure, time_clean
from realdata import load_mm

ap = argparse.ArgumentParser()
ap.add_argument("--ops", default="", help="name=path,... extra operator files")
ap.add_argument("--out", required=True)
ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--cases", default="knapsack_ridge:1500:0.1,knapsack_ridge:1500:0.03,knapsack_ridge:1500:0.01,"
                "knapsack_ridge:1500:0.005,knapsack_ridge:1500:0.003,knapsack_ridge:1500:0.002,"
                "H:svm_dual:800,H:budget_band:2000,H:portfolio:2000,H:knapsack_ridge:6000:0.1,H:knapsack_ridge:3000:0.003,"
                "MM:DUAL1,MM:DUAL2,MM:DUAL3,MM:DUAL4,MM:VALUES")
a = ap.parse_args()
extra = [tuple(e.split("=", 1)) for e in a.ops.split(",") if e]
done = set()
if os.path.exists(a.out):
    for line in open(a.out):
        r = json.loads(line); done.add((r["case"], r["cand"]))
out = open(a.out, "a")
for case in a.cases.split(","):
    parts = case.split(":")
    try:
        if parts[0] == "MM":
            d, heldout = load_mm(parts[1]), True
        else:
            heldout = parts[0] == "H"
            parts = parts[1:] if heldout else parts
            kw = {"rho": float(parts[2])} if len(parts) > 2 else {}
            seed, pseed = (3, 11) if heldout else (0, 7)          # ladder instances match close_loop.py exactly
            d = permute_instance(make_instance(parts[0], int(parts[1]), seed=seed, **kw), pseed)
        plan, why = plan_block(d, diagnose(d))
        obj_ref, _ = reference(d)
    except Exception:
        print(case, "setup failed", traceback.format_exc()[-300:], flush=True); continue
    cands = [("std_raw", "std_raw", {}), ("std_ruiz", "std_ruiz", {})]
    if plan:
        cands += [("reloc:bisect", "reloc", {"impl": "bisect"})]
        if plan["kind"] != "simplex":
            cands += [("reloc:breakpoint", "reloc", {"impl": "breakpoint"}),
                      ("reloc:newton12", "reloc", {"impl": "newton", "iters": 12})]
        cands += [(f"reloc:{nm}", "reloc", {"impl": "file:" + pth}) for nm, pth in extra]
    print(f"== {case} heldout={heldout} kind={plan['kind'] if plan else why}", flush=True)
    for name, cand, kw in cands:
        if (case, name) in done:
            continue
        rec = {"case": case, "heldout": heldout, "cand": name, "kind": plan["kind"] if plan else None}
        lk = open(os.environ.get("SPEC_LOCK", "/tmp/claude-1000/opcheck.lock"), "w")
        import fcntl
        fcntl.flock(lk, fcntl.LOCK_EX)                 # never time while a synthesis arm is timing on this device
        try:
            K = 10 if cand == "reloc" else 50
            m = measure(build_formulation(d, cand, plan=plan, **kw), d, obj_ref, tol=1e-6, cap=200000, K=K)
            rec.update({"N": m["N"], "rv": m["row_violation"], "gap": m["obj_gap"]})
            if m["N"]:
                ts = [time_clean(build_formulation(d, cand, plan=plan, **kw), m["N"], K=K) for _ in range(a.reps)]
                rec.update({"total_s": float(np.median([t["total_s"] for t in ts])),
                            "setup_s": float(np.median([t["setup_s"] for t in ts])),
                            "per_iter_s": float(np.median([t["per_iter_s"] for t in ts]))})
        except Exception:
            rec["error"] = traceback.format_exc()[-300:]
        finally:
            fcntl.flock(lk, fcntl.LOCK_UN); lk.close()
        out.write(json.dumps(rec, default=float) + "\n"); out.flush()
        print(f"   {name:28s} N={rec.get('N')} total={rec.get('total_s')} {rec.get('error', '')[-100:]}", flush=True)
print("WORKLOAD_DONE", flush=True)
