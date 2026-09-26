"""Smoke test: (1) plan survives row/column permutation on positives and still refuses on negatives;
(2) every formulation decodes correctly, i.e. reaches original-problem acceptance at a finite N."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from uni import (make_instance, permute_instance, diagnose, plan_block, build_formulation, reference, measure)

print("== (1) permutation robustness ==", flush=True)
print("  family          encoding  detector  gap     plan", flush=True)
cases = [("portfolio", 3000), ("svm_dual", 3000), ("meb_dual", 3000), ("knapsack_ridge", 3000),
         ("budget_band", 3000), ("lasso", 3000), ("svm", 3000), ("huber", 3000), ("control", 300)]
for fam, sz in cases:
    d0 = make_instance(fam, sz, seed=0)
    for tag, d in (("original", d0), ("permuted", permute_instance(d0, 11))):
        try:
            g = diagnose(d)
            plan, why = plan_block(d, g)
            print("  %-15s %-9s %-9s %-7.1f %s" % (fam, tag, "FIRES" if g["fires"] else "silent",
                  g["singular_gap"], (plan["kind"] if plan else why)), flush=True)
        except Exception as e:
            print("  %-15s %-9s ERROR %s" % (fam, tag, str(e)[:80]), flush=True)

print("\n== (2) decode + acceptance on PERMUTED instances ==", flush=True)
print("  family          cand             N        rv        gap       setup_s  t_iter_us", flush=True)
for fam, sz in (("portfolio", 1000), ("svm_dual", 1000), ("meb_dual", 1000), ("knapsack_ridge", 800),
                ("budget_band", 1000)):
    d = permute_instance(make_instance(fam, sz, seed=1, rho=1e-2), 5)
    g = diagnose(d)
    plan, why = plan_block(d, g)
    obj_ref, info = reference(d)
    if obj_ref is None:
        print("  %-15s reference failed: %s" % (fam, info)); continue
    cands = [("std_raw", {}), ("std_ruiz", {}), ("reloc", {"impl": "bisect"})]
    if plan and plan["kind"] == "simplex":
        cands.append(("reloc", {"impl": "sort"}))
    for cand, kw in cands:
        if cand == "reloc" and plan is None:
            continue
        try:
            form = build_formulation(d, cand, plan=plan, **kw)
            m = measure(form, d, obj_ref, tol=1e-6, cap=60000, K=50)
            print("  %-15s %-16s %-8s %-9.1e %-9.1e %-8.2f %.1f" % (fam, form.name,
                  m["N"] if m["N"] else ">%d" % m["iters_run"], m["row_violation"] or -1, m["obj_gap"] or -1,
                  m["setup_s"], 1e6 * (m["t_iter"] or 0)), flush=True)
        except Exception as e:
            import traceback
            print("  %-15s %-16s ERROR %s" % (fam, cand, str(e)[:160]), flush=True)
print("SMOKE_DONE")
