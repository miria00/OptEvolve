"""GENERALIZATION TEST for synthesized operators: does an operator found against one evaluator work where it was not
searched? Each operator file is put in the solver on instances NEITHER search arm ever evaluated on:
  svm_dual 800          box-hyperplane with MIXED-SIGN coefficients (the contract a positive-only operator breaks)
  budget_band 2000      box-slab (the operator is called with the clipped target inside the band)
  portfolio 2000        simplex: hi = +inf, a = 1 (unbounded-above contract)
  knapsack_ridge 6000   same family, 4x larger block
  Maros-Meszaros        real instances where the plan applies (DUAL1-4, VALUES fully absorbed; QSTANDAT, EXDATA)
For each: did the solve reach original-problem acceptance, and at the SAME N as the library bisection (an exact
operator must not change N)? A wrong operator shows up as non-convergence, a different N, or a rejected answer.
"""
import argparse, json, os, sys, traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from uni import make_instance, permute_instance, diagnose, plan_block, build_formulation, reference, measure
from realdata import load_mm

ap = argparse.ArgumentParser()
ap.add_argument("--ops", required=True, help="name=path,...")
ap.add_argument("--out", required=True)
ap.add_argument("--cases", default="svm_dual:800,budget_band:2000,portfolio:2000,knapsack_ridge:6000:0.1,"
                                  "MM:DUAL1,MM:DUAL2,MM:DUAL3,MM:DUAL4,MM:VALUES,MM:QSTANDAT,MM:EXDATA")
ap.add_argument("--cap", type=int, default=100000)
a = ap.parse_args()
ops = [tuple(e.split("=", 1)) for e in a.ops.split(",") if e]
out = open(a.out, "a")
for case in a.cases.split(","):
    parts = case.split(":")
    try:
        if parts[0] == "MM":
            d = load_mm(parts[1])
        else:
            kw = {"rho": float(parts[2])} if len(parts) > 2 else {}
            d = permute_instance(make_instance(parts[0], int(parts[1]), seed=3, **kw), 11)
        plan, why = plan_block(d, diagnose(d))
        if plan is None:
            print(case, "no plan:", why, flush=True); continue
        obj_ref, _ = reference(d)
        ref = measure(build_formulation(d, "reloc", plan=plan, impl="bisect"), d, obj_ref, tol=1e-6, cap=a.cap, K=10)
        print(f"== {case} kind={plan['kind']} n_x={plan['n_x']} bisect N={ref['N']}", flush=True)
    except Exception:
        print(case, "setup failed", traceback.format_exc()[-300:], flush=True); continue
    for name, path in ops:
        rec = {"case": case, "kind": plan["kind"], "op": name, "path": path, "N_bisect": ref["N"]}
        try:
            m = measure(build_formulation(d, "reloc", plan=plan, impl="file:" + path), d, obj_ref, tol=1e-6,
                        cap=max(4 * (ref["N"] or a.cap), 2000), K=10)
            rec.update({"N": m["N"], "accepted": m["N"] is not None, "same_N": m["N"] == ref["N"],
                        "rv": m["row_violation"], "gap": m["obj_gap"]})
        except Exception:
            rec.update({"accepted": False, "error": traceback.format_exc()[-300:]})
        out.write(json.dumps(rec, default=float) + "\n"); out.flush()
        print(f"   {name:24s} accepted={rec.get('accepted')} N={rec.get('N')} same_N={rec.get('same_N')} "
              f"{rec.get('error', '')[-120:]}", flush=True)
print("GEN_TEST_DONE", flush=True)
