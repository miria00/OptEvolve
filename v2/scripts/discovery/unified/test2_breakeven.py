"""TEST 2 - does the cost model predict the decision? (run on a QUIET machine: this one is about wall clock)

For an instance where the fixed formulation does converge, measure the best fixed formulation's total time and the
relocated formulation's N. Then make the projection deliberately more expensive by re-applying an exact projection
r extra times per iteration: the output is unchanged (projections are idempotent), so N is unchanged and only t
grows. The additive model predicts the break-even
    setup_rel + N_rel * (t_base + (1 + r*) * t_proj) = T_fixed
from the r = 0 and r = 1 timings alone. The measured flip comes from the full r sweep. Agreement means the model can
hand an operator-synthesis search a checkable per-call budget c* before any code is written.
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from uni import (make_instance, permute_instance, diagnose, plan_block, build_formulation, reference, measure,
                 time_clean)

ap = argparse.ArgumentParser()
ap.add_argument("--cases", default="svm_dual:800,knapsack_ridge:1500:0.001,knapsack_ridge:1500:0.01,budget_band:800")
ap.add_argument("--pads", default="0,1,2,4,8,16,32,64,128,256,512")
ap.add_argument("--cap", type=int, default=400000)
ap.add_argument("--out", required=True)
a = ap.parse_args()

out = open(a.out, "w")
for case in a.cases.split(","):
    parts = case.split(":")
    fam, size = parts[0], int(parts[1])
    rho = float(parts[2]) if len(parts) > 2 else 0.0
    d = permute_instance(make_instance(fam, size, seed=0, rho=rho), 7)
    plan, why = plan_block(d, diagnose(d))
    obj_ref, _ = reference(d)
    print(f"\n== {case}  kind={plan['kind'] if plan else why}", flush=True)
    if plan is None or obj_ref is None:
        continue
    fixed = {}
    for cand in ("std_raw", "std_ruiz"):
        f = build_formulation(d, cand)
        m = measure(f, d, obj_ref, tol=1e-6, cap=a.cap, K=50)
        if m["N"]:
            tc = time_clean(build_formulation(d, cand), m["N"], K=50)
            fixed[cand] = {"N": m["N"], **tc}
        print(f"  fixed {cand:9s} N={m['N'] or '>' + str(m['iters_run'])}"
              + (f"  total={fixed[cand]['total_s']:.3f}s" if cand in fixed else ""), flush=True)
    if not fixed:
        print("  no fixed formulation converges within cap; break-even is unbounded here", flush=True)
        continue
    best_fixed = min(fixed.values(), key=lambda r: r["total_s"])
    impl = "sort" if plan["kind"] == "simplex" else "bisect"
    m_rel = measure(build_formulation(d, "reloc", plan=plan, impl=impl), d, obj_ref, tol=1e-6, cap=a.cap, K=10)
    N_rel = m_rel["N"]
    print(f"  relocated ({impl}) N={N_rel}", flush=True)
    if not N_rel:
        continue
    sweep = []
    for r in [int(x) for x in a.pads.split(",")]:
        tc = time_clean(build_formulation(d, "reloc", plan=plan, impl=impl, pad=r), N_rel, K=10)
        sweep.append({"pad": r, **tc})
        loses = tc["total_s"] >= best_fixed["total_s"]
        print(f"  pad={r:<4d} per_iter={1e6 * tc['per_iter_s']:.1f}us  total={tc['total_s']:.3f}s  "
              f"{'loses' if loses else 'WINS'}", flush=True)
        # total time is monotone in the padding, so the first losing pad ends the sweep; pads 0 and 1 always run
        # because the prediction is made from exactly those two timings
        if loses and r >= 1:
            break
    t0, t1 = sweep[0]["per_iter_s"], sweep[1]["per_iter_s"]
    t_proj = max(t1 - t0, 1e-12)
    t_base = t0 - t_proj
    budget = (best_fixed["total_s"] - sweep[0]["setup_s"]) / N_rel        # seconds per iteration allowed
    r_pred = (budget - t_base) / t_proj - 1.0
    c_star = budget - t_base                                              # seconds of projection per iteration
    r_meas = None
    for lo_, hi_ in zip(sweep[:-1], sweep[1:]):
        if lo_["total_s"] < best_fixed["total_s"] <= hi_["total_s"]:
            f = (best_fixed["total_s"] - lo_["total_s"]) / max(hi_["total_s"] - lo_["total_s"], 1e-12)
            r_meas = lo_["pad"] + f * (hi_["pad"] - lo_["pad"])
            break
    print(f"  best fixed total={best_fixed['total_s']:.3f}s  t_base={1e6*t_base:.1f}us  t_proj={1e6*t_proj:.1f}us  "
          f"c*={1e6*c_star:.1f}us/iter", flush=True)
    print(f"  PREDICTED break-even pad r*={r_pred:.1f}   MEASURED r*={'beyond sweep' if r_meas is None else f'{r_meas:.1f}'}",
          flush=True)
    out.write(json.dumps({"case": case, "kind": plan["kind"], "fixed": fixed, "N_rel": N_rel, "sweep": sweep,
                          "t_base": t_base, "t_proj": t_proj, "c_star": c_star, "r_pred": r_pred,
                          "r_meas": r_meas}) + "\n")
    out.flush()
print("TEST2_DONE", flush=True)
