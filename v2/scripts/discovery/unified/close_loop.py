"""CLOSING THE LOOP - does an operator that meets the cost model's spec actually flip the formulation decision?

For each instance, on THIS machine:
  1. the fixed formulation (best of std_raw, std_ruiz) gives the time to beat;
  2. the relocated formulation with the library operator (60-step bisection) gives N_rel and, from pads 0 and 1, the
     per-iteration base cost and the per-call projection cost, hence the budget c* = (T_fixed - setup)/N_rel - t_base;
  3. the SPEC handed to operator synthesis is then: per-call projection cost <= c* (impossible if c* <= 0);
  4. each candidate operator (breakpoint sort, safeguarded Newton-12) is put in the solver, its iteration count is
     checked to be IDENTICAL to bisection (exact operators must not change N), and its per-call cost is measured;
  5. PREDICTION: relocation wins with that operator iff its per-call cost <= c*.  ACTUAL: its measured total time.
Timings are medians of interleaved repetitions on a machine with no other GPU work.
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from uni import (make_instance, permute_instance, diagnose, plan_block, build_formulation, reference, measure,
                 time_clean)

ap = argparse.ArgumentParser()
ap.add_argument("--cases", default="knapsack_ridge:1500:0.1,knapsack_ridge:1500:0.01,knapsack_ridge:1500:0.001,"
                                   "svm_dual:800,knapsack_ridge:6000:0.1")
ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--extra-ops", default="", help="name=path,... synthesized operator files put in the solver as well")
ap.add_argument("--out", required=True)
a = ap.parse_args()


def med_time(builder, iters, K, reps):
    vals = [time_clean(builder(), iters, K=K) for _ in range(reps)]
    return {"total_s": float(np.median([v["total_s"] for v in vals])),
            "per_iter_s": float(np.median([v["per_iter_s"] for v in vals])),
            "setup_s": float(np.median([v["setup_s"] for v in vals]))}


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
        m = measure(build_formulation(d, cand), d, obj_ref, tol=1e-6, cap=400000, K=50)
        if m["N"]:
            fixed[cand] = {"N": m["N"], **med_time(lambda c=cand: build_formulation(d, c), m["N"], 50, a.reps)}
            print(f"  fixed {cand:9s} N={m['N']:<7d} total={fixed[cand]['total_s']:.3f}s", flush=True)
        else:
            print(f"  fixed {cand:9s} N>{m['iters_run']}", flush=True)
    if not fixed:
        print("  no fixed formulation converges: any exact operator wins; nothing to test here", flush=True)
        continue
    T_fixed = min(v["total_s"] for v in fixed.values())
    m_b = measure(build_formulation(d, "reloc", plan=plan, impl="bisect"), d, obj_ref, tol=1e-6, cap=400000, K=10)
    N_rel = m_b["N"]
    if not N_rel:
        print("  relocated formulation did not converge", flush=True); continue
    p0 = med_time(lambda: build_formulation(d, "reloc", plan=plan, impl="bisect", pad=0), N_rel, 10, a.reps)
    p1 = med_time(lambda: build_formulation(d, "reloc", plan=plan, impl="bisect", pad=1), N_rel, 10, a.reps)
    t_proj_b = max(p1["per_iter_s"] - p0["per_iter_s"], 1e-12)
    t_base = p0["per_iter_s"] - t_proj_b
    c_star = (T_fixed - p0["setup_s"]) / N_rel - t_base
    spec = ("IMPOSSIBLE (c* <= 0: even a free projection loses)" if c_star <= 0
            else f"per-call projection <= {1e6 * c_star:.1f} us, i.e. >= {t_proj_b / c_star:.2f}x faster than bisection")
    print(f"  relocated N={N_rel}; bisection per-call {1e6*t_proj_b:.1f}us, base {1e6*t_base:.1f}us/iter; "
          f"T_fixed={T_fixed:.3f}s", flush=True)
    print(f"  SPEC from the cost model: {spec}", flush=True)
    # CALIBRATION MODEL (device-robust): no pad decomposition. Base = the relocated iteration with a NULL block map,
    # timed directly; each operator's per-iteration cost and setup come from a SHORT clean run (CAL iterations); the
    # prediction is T_hat = setup_op + N_rel * t_iter_op(short). On the H100 the pad decomposition gave NEGATIVE
    # per-call costs (an extra projection's marginal cost is not the first projection's cost there), so it declared
    # rungs impossible that several operators actually won.
    CAL = 300
    nul = med_time(lambda: build_formulation(d, "reloc", plan=plan, impl="null"), CAL, 10, a.reps)
    t_base_null = nul["per_iter_s"]
    print(f"  calibration: null-map relocated iteration {1e6*t_base_null:.1f} us/iter (pad estimate {1e6*t_base:.1f})",
          flush=True)
    rec = {"case": case, "kind": plan["kind"], "fixed": fixed, "T_fixed": T_fixed, "N_rel": N_rel,
           "t_base_null": t_base_null,
           "t_proj_bisect": t_proj_b, "t_base": t_base, "c_star": c_star, "ops": {}}
    print("  operator     N      same_N  per_call_us  meets_spec(pred)  total_s  wins(actual)  prediction_correct  "
          "| setup_s  own-setup budget_us  pred2  correct2", flush=True)
    extra = [(nm, {"impl": "file:" + pth}) for nm, pth in (e.split("=", 1) for e in a.extra_ops.split(",") if e)]
    for op, kw in [("bisect", {"impl": "bisect"}), ("breakpoint", {"impl": "breakpoint"}),
                   ("newton12", {"impl": "newton", "iters": 12})] + extra:
        m = measure(build_formulation(d, "reloc", plan=plan, **kw), d, obj_ref, tol=1e-6, cap=400000, K=10)
        if not m["N"]:
            print(f"  {op:12s} did not converge (N>{m['iters_run']})", flush=True)
            rec["ops"][op] = {"N": None}
            continue
        tm = med_time(lambda kw=kw: build_formulation(d, "reloc", plan=plan, **kw), m["N"], 10, a.reps)
        per_call = tm["per_iter_s"] - t_base
        meets = c_star > 0 and per_call <= c_star
        wins = tm["total_s"] < T_fixed
        # REFINED MODEL: T_setup depends on the operator (a 60-step bisection compiles far more XLA than a short
        # Newton loop), so the break-even budget is operator-specific: c*_op = (T_fixed - setup_op)/N_rel - t_base.
        # setup_op is measurable from one compile, without running the N iterations.
        cal = med_time(lambda kw=kw: build_formulation(d, "reloc", plan=plan, **kw), CAL, 10, a.reps)
        T_hat = cal["setup_s"] + N_rel * cal["per_iter_s"]
        per_call3 = cal["per_iter_s"] - t_base_null
        c3 = (T_fixed - cal["setup_s"]) / N_rel - t_base_null
        pred3 = T_hat < T_fixed
        c_own = (T_fixed - tm["setup_s"]) / N_rel - t_base
        meets2 = c_own > 0 and per_call <= c_own
        print(f"  {op:12s} {m['N']:<6d} {str(m['N'] == N_rel):7s} {1e6*per_call:<12.1f} {str(meets):17s} "
              f"{tm['total_s']:<8.3f} {str(wins):13s} {str(meets == wins):19s} | {tm['setup_s']:<8.3f} "
              f"{1e6*c_own:<18.1f} {str(meets2):6s} {str(meets2 == wins):8s} | cal: per_call {1e6*per_call3:.1f} "
              f"budget {1e6*c3:.1f} T_hat {T_hat:.3f} pred3 {pred3} correct3 {pred3 == wins}", flush=True)
        rec["ops"][op] = {"N": m["N"], "same_N": m["N"] == N_rel, "per_call_s": per_call, **tm,
                          "meets_spec": bool(meets), "wins": bool(wins), "c_star_own_setup": c_own,
                          "meets_spec_own_setup": bool(meets2), "cal": cal, "T_hat": T_hat, "per_call_cal": per_call3,
                          "c_star_cal": c3, "pred3": bool(pred3)}
    out.write(json.dumps(rec) + "\n"); out.flush()
print("CLOSE_LOOP_DONE", flush=True)
