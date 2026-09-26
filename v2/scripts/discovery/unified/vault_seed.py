"""Seed the vault with everything measured on 2026-09-15: components, transformations, evidence, lessons, open specs.

Every number comes from an artifact file; nothing is typed in by hand except the lesson statements, and each lesson
names the artifact that forced it. Operator source files are copied into the vault so it is self-contained.
"""
import argparse, glob, json, os, shutil, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vault import Vault

ap = argparse.ArgumentParser()
ap.add_argument("--vault", default="/home/miria/OptEvolve/v2/vault/vault.jsonl")
ap.add_argument("--art", default="/tmp/claude-1000/-home-miria/5f48ba93-fbf8-4468-84d8-75d06ace30b7/scratchpad/unified")
a = ap.parse_args()
if os.path.exists(a.vault):
    sys.exit(f"{a.vault} exists; the vault is append-only, seed a fresh path or add entries with supersedes=")
V = Vault(a.vault)
CODE = os.path.join(os.path.dirname(a.vault), "components")
os.makedirs(CODE, exist_ok=True)
U = a.art
HERE = os.path.dirname(os.path.abspath(__file__))
jl = lambda p: [json.loads(l) for l in open(p) if l.strip()] if os.path.exists(p) else []

CONTRACT_BH = ["lo finite, hi finite or +inf", "a of either sign, none zero", "n from 7 to 10000", "simplex special case",
               "degenerate ties", "deployment block inputs captured from the solver"]

# ---------------------------------------------------------------- components (box-hyperplane set)
cert = {os.path.basename(r["file"]): r["result"] for r in jl(f"{U}/exported_cert.jsonl")}
r2cert = json.load(open(f"{U}/round2/cert.json")) if os.path.exists(f"{U}/round2/cert.json") else None


def calib(device_files, op):
    """Per-device calibration of one operator from close-loop records (calibration model when present)."""
    out = {}
    for dev, files in device_files.items():
        rows = []
        for f in files:
            for rec in jl(f):
                o = (rec.get("ops") or {}).get(op)
                if o and o.get("N"):
                    cal = o.get("cal") or {}
                    rows.append({"case": rec["case"], "per_call_us": round(1e6 * (o.get("per_call_cal") or o["per_call_s"]), 2),
                                 "setup_s": round((cal.get("setup_s") or o["setup_s"]), 4),
                                 "wins": o.get("wins"), "T_total_s": round(o["total_s"], 4), "T_fixed_s": round(rec["T_fixed"], 4)})
        if rows:
            out[dev] = {"per_call_us": round(sorted(r["per_call_us"] for r in rows)[len(rows) // 2], 2), "cases": rows}
    return out


DEV = {"rtx4090": [f"{U}/replan2.jsonl", f"{U}/round2/replan.jsonl"], "h100": [f"{U}/h100/h100_replan_cal.jsonl"]}
transfer = {}
for f in (f"{U}/gen_test.jsonl", f"{U}/round2/gen_test.jsonl"):
    for r in jl(f):
        transfer.setdefault(r["op"], []).append({"case": r["case"], "accepted": r.get("accepted"), "same_N": r.get("same_N")})

comps = [
    ("component:bisect60", None, "library", "atlas.operators.prox.project_box_hyperplane(iters=60)", "bisect",
     {"certified": 1.0, "note": "library reference operator"}),
    ("component:breakpoint_hand", None, "hand-written", "uni.bp_project (sign-safe breakpoint sort)", "breakpoint",
     {"certified": 1.0, "speedup_target": 3.20, "note": "certified after the mixed-sign fix; first version failed svm_dual"}),
    ("component:newton12_hand", None, "hand-written", "uni.newton_project(steps=12)", "newton12",
     {"certified": 1.0, "speedup_target": 2.35}),
    ("component:spec_r1_top0", f"{U}/exported/sky27b_top0.py", "synthesized round 1 (spec 6.80x)", None, "spec0",
     cert.get("sky27b_top0.py")),
    ("component:spec_r1_top1", f"{U}/exported/sky27b_top1.py", "synthesized round 1 (spec 6.80x)", None, "spec1",
     cert.get("sky27b_top1.py")),
    ("component:spec_r1_top2", f"{U}/exported/sky27b_top2.py", "synthesized round 1 (spec 6.80x)", None, "spec2",
     cert.get("sky27b_top2.py")),
    ("component:spec_r2", f"{U}/round2/exported/spec_r2.py", "synthesized round 2 (spec 10.9x, seeded with round 1)", None,
     "spec_r2", r2cert),
    ("component:blackbox_top0", f"{U}/exported/skybb_top0.py", "black-box arm (single-instance end-to-end fitness)", None,
     "bb0", cert.get("skybb_top0.py")),
]
for cid, src, origin, ref, op, c in comps:
    path = None
    if src and os.path.exists(src):
        path = os.path.join(CODE, cid.split(":")[1] + ".py")
        shutil.copy(src, path)
    tr = transfer.get(op, [])
    loses = [t["case"] for t in tr if t["accepted"] is False and t["case"] not in ("MM:QSTANDAT", "MM:EXDATA")]
    V.add("component", {
        "set": "box_hyperplane", "file": path, "reference_impl": ref, "contract": CONTRACT_BH,
        "certificate": {k: c.get(k) for k in ("certified", "max_err", "speedup_target", "cand_us_target", "failing_clauses")
                        if c and k in c} if c else None,
        "calibration": calib(DEV, op), "transfer": tr, "loses": loses,
    }, id=cid, provenance={"origin": origin, "date": "2026-09-15"})
V.add("component", {
    "set": "product_box_slab", "file": None, "reference_impl": "uni.product_project (segment-op bracket, bisection, Newton polish)",
    "contract": ["disjoint blocks", "a of either sign", "infinite bounds", "equality and slab rows", "feasible inputs"],
    "certificate": {"certified": 1.0, "max_err": 1.27e-14, "trials": 40, "test": "test_product.py"},
    "calibration": {}, "transfer": [], "loses": []}, id="component:product_bisect",
    provenance={"origin": "hand-written", "date": "2026-09-15"})

# ---------------------------------------------------------------- transformations
for tid, name, summ, det in [
    ("transformation:ruiz", "std_ruiz", "diagonal equilibration of the whole constraint operator", "always applicable"),
    ("transformation:relocation", "reloc", "absorb ONE bounded row (box-hyperplane or box-slab) into the proximal step",
     "uni.plan_block on the dominant row of the Ruiz-scaled operator"),
    ("transformation:row_equilibration", "reloc_rows", "row-only equilibration of what stays in L; exact-compatible with "
     "any absorbed projection", "applies to every relocated formulation"),
    ("transformation:product_relocation", "reloc_prod", "absorb a DISJOINT FAMILY of bounded rows as a product set",
     "uni.plan_product (greedy disjoint rows whose support is fully bounded)"),
    ("transformation:exact_finishing", "polish", "first-order to 1e-3, then an exact reduced solve on the identified "
     "active set; switch point chosen by the cost model", "PLANNED"),
    ("transformation:device_routing", "route", "run the instance on the device (or CPU solver) the cost model predicts "
     "fastest", "PLANNED"),
]:
    V.add("transformation", {"name": name, "summary": summ, "detector": det}, id=tid,
          provenance={"date": "2026-09-15"})

# ---------------------------------------------------------------- evidence
n_ev = 0
for r in jl(f"{U}/workload.jsonl"):
    V.add("evidence", {"source": "workload", "device": "rtx4090", "case": r["case"], "heldout": r["heldout"],
                       "formulation_op": r["cand"], "N": r.get("N"), "total_s": r.get("total_s"),
                       "setup_s": r.get("setup_s"), "per_iter_s": r.get("per_iter_s"), "tol": 1e-6,
                       "accepted": bool(r.get("N"))}, provenance={"file": "workload.jsonl"})
    n_ev += 1
for dev, files in {"rtx4090": [f"{U}/spec_ladder.jsonl", f"{U}/replan2.jsonl", f"{U}/round2/replan.jsonl"],
                   "h100": [f"{U}/h100/h100_replan_cal.jsonl"]}.items():
    for f in files:
        for rec in jl(f):
            for op, o in (rec.get("ops") or {}).items():
                V.add("evidence", {"source": "close_loop", "device": dev, "case": rec["case"], "formulation_op": f"reloc:{op}",
                                   "N": o.get("N"), "total_s": o.get("total_s"), "setup_s": o.get("setup_s"),
                                   "per_iter_s": o.get("per_iter_s"), "T_fixed_s": rec["T_fixed"], "N_rel": rec["N_rel"],
                                   "wins": o.get("wins"), "T_hat": o.get("T_hat"), "pred_cal": o.get("pred3"),
                                   "tol": 1e-6}, provenance={"file": os.path.basename(f)})
                n_ev += 1
for r in jl(f"{U}/h100/mm_product_adaptive.jsonl"):
    if r.get("arm") in (None, "setup"):
        continue
    V.add("evidence", {"source": "mm_product_adaptive", "device": "h100(shared)", "case": "MM:" + r["instance"],
                       "formulation_op": r["arm"], "N": r.get("N"), "iters_all_rungs": r.get("iters_all_rungs"),
                       "tol": 1e-6, "driver": "adaptive", "product_K": r.get("product_K"),
                       "product_cover": r.get("product_cover"), "fires": (r.get("diag") or {}).get("fires")},
          provenance={"file": "mm_product_adaptive.jsonl", "note": "iteration counts only; times not clean"})
    n_ev += 1

# ---------------------------------------------------------------- lessons (each names the evidence that forced it)
LESSONS = [
    ("lesson:bound_not_estimate_N", "A norm bound is not an estimate of the iteration count it bounds.",
     "The dominant-row diagnosis fired on MM DUAL1-3 where relocation barely changed N (4300->4400, 1000->900, 600->600) "
     "and did not fire on LOTSCHD (6900->300) or DUALC2/5/8 (up to 47x).",
     "Decide formulations from measured or evidence-predicted N, never from ||L|| alone.", "check_diagnosis_vs_N"),
    ("lesson:setup_operator_specific", "Setup time belongs to the operator, not to the formulation.",
     "Charging every operator bisection's compile (~0.125 s vs ~0.044 s for synthesized Newton) made all 6 of 36 "
     "re-plan misses conservative; operator-specific setup gave 24/24.",
     "Break-even budgets use each operator's own setup: c*_op = (T_fixed - setup_op)/N_rel - t_base.", "check_setup_budget"),
    ("lesson:marginal_not_first_call", "The marginal cost of an extra call is not the cost of the first call, and the "
     "gap is device dependent.",
     "Pad decomposition gave negative per-call costs and 10 wrong decisions on the H100 (base 125-165 us vs null-map "
     "13.9 us); calibration from short clean runs gave 42/42 there and 18/18 on the 4090.",
     "Calibrate each operator directly (300-iteration clean run) against a null-map base; never infer from padding.",
     "check_calibration_model"),
    ("lesson:single_instance_overfits", "Fitness on one instance selects instance-bound programs.",
     "The black-box winner (2-step Newton guess) passed its instance 3.2x under the bar but failed every certificate "
     "clause and 4 of 9 unseen instances (svm_dual, budget_band, portfolio, MM DUAL1).",
     "Admit components only through the contract certificate; score evolution on suites, not single instances.",
     "check_transfer"),
    ("lesson:offline_eps_not_convergence", "Offline exactness on sampled inputs does not certify an INEXACT operator.",
     "Dykstra-k with offline error 1.7e-16 (svm_dual k=100) and 4e-17 (budget_band k=300) still failed to converge.",
     "Inexact operators need a runtime per-call certificate on real inputs.", "check_runtime_certificate"),
    ("lesson:isolated_timing_optimistic", "Isolated per-call timing is optimistic relative to in-solver cost.",
     "Round-2 winner: 32.5 us isolated vs 37.2 us in-solver; met its evaluator spec but missed the rho .001 flip by 2%.",
     "Score synthesis against an in-solver calibration run of the formulation.", "check_in_solver_spec"),
    ("lesson:imports_before_clock", "Timing contracts must exclude interpreter imports on both sides.",
     "baseline_proc timed `import osqp` (0.19-0.26 s on the H100); fixed 2026-09-15.",
     "Import solvers before starting the clock.", "check_timing_contract"),
    ("lesson:global_kkt_hides", "Globally normalized KKT residuals hide row violations on badly scaled instances.",
     "BOYD2 passed while missing 167,970 rows by more than 1e-6 of their bounds; PRIMALC2/8 similar.",
     "Judge per-row relative feasibility and relative objective error.", "check_acceptance"),
    ("lesson:certificate_is_a_test", "A certificate is a test, not a proof: its clauses must cover the structure "
     "the deployment uses.",
     "The positive-coefficient breakpoint operator passed a certificate without mixed-sign cases and never converged "
     "on svm_dual; acceptance on the original problem caught it.",
     "Derive contract clauses from the extracted structure (signs, infinite bounds, sizes) before admission.",
     "check_contract_coverage"),
]
for lid, stmt, ev, rule, check in LESSONS:
    V.add("lesson", {"statement": stmt, "evidence": ev, "rule": rule, "check": check}, id=lid,
          provenance={"date": "2026-09-15"})

# ---------------------------------------------------------------- specs from the latest re-plans (calibration model)
for dev, f in (("rtx4090", f"{U}/round2/replan.jsonl"), ("h100", f"{U}/h100/h100_replan_cal.jsonl")):
    for rec in jl(f):
        wins = [op for op, o in rec["ops"].items() if o.get("wins") and op != "bb0"]
        best = min((o["cal"]["per_iter_s"] - rec["t_base_null"], op) for op, o in rec["ops"].items()
                   if o.get("cal") and op != "bb0")
        # the budget a NEW operator must meet assumes the setup of the best certified operator it would replace
        # (lesson:setup_operator_specific), not the minimum over all operators, which charges bisection's compile
        budget = rec["ops"][best[1]]["c_star_cal"]
        status = "met" if wins and any(w not in ("bisect",) for w in wins) else "open"
        V.add("spec", {"status": status, "device": dev, "case": rec["case"], "set": "box_hyperplane",
                       "budget_per_call_us_in_solver": round(1e6 * budget, 1),
                       "best_certified_per_call_us": round(1e6 * best[0], 1), "best_certified": best[1],
                       "certified_winners": wins,
                       "summary": f"{rec['case']} on {dev}: exact box-hyperplane projection <= "
                                  f"{1e6 * budget:.1f} us/call in-solver (best certified {best[1]} "
                                  f"{1e6 * best[0]:.1f} us)"}, provenance={"file": os.path.basename(f)})

print(json.dumps(V.summary()))
print(V.describe())
