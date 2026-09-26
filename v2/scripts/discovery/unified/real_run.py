"""REAL-INSTANCE RUN: the formulation rule, exact-operator invariance and baselines on Maros-Meszaros / QPLIB.

Per instance (encoding as distributed, no generator involved):
  reference   Clarabel at 1e-9 (objective for acceptance)
  baselines   OSQP (eps 1e-3, the RLQP-style default, and 1e-6), Clarabel 1e-6, SCS 1e-6, each in its own isolated
              interpreter; every returned point is judged by the SAME original-problem check as ours
  ours        std_raw, std_ruiz and, if a plan exists, the relocated formulation with every exact operator that fits
              the block kind; N is iterations to original-problem acceptance (row violation and objective gap <= tol)
Records whether the diagnosis fired, so the rule's prediction (relocate only when it fires) can be tested on controls.
N does not depend on machine load; per-iteration times here are NOT clean (the GPU is shared) and are re-timed later.
"""
import argparse, json, os, sys, time, traceback, socket
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from realdata import load_mm, load_qplib
from uni import diagnose, plan_block, plan_product, build_formulation, reference, accept, measure, measure_ladder, LADDER
from atlas.bench.baseline_proc import run_baseline_proc

ap = argparse.ArgumentParser()
ap.add_argument("--set", choices=("mm", "qplib"), required=True)
ap.add_argument("--instances", required=True)
ap.add_argument("--qplib-dir", default="")
ap.add_argument("--out", required=True)
ap.add_argument("--tol", type=float, default=1e-6)
ap.add_argument("--cap", type=int, default=100000)
ap.add_argument("--K", type=int, default=100)
ap.add_argument("--no-baselines", action="store_true")
ap.add_argument("--no-single", dest="single", action="store_false", help="skip the dominant-row relocation candidates")
ap.add_argument("--ladder", action="store_true", help="record every accuracy rung (1e-2, 1e-3, 1e-4, 1e-6) in one run; "
                "baselines are run at each rung and judged by the same check")
ap.add_argument("--driver", choices=("plain", "adaptive"), default="plain",
                help="plain: fixed gate steps (isolates the formulation); adaptive: the production certified adaptive "
                     "step ratio with a charged tolerance ladder, as in qp_relocate_reps.py")
a = ap.parse_args()

done = {}
if os.path.exists(a.out):
    for line in open(a.out):
        try:
            r = json.loads(line); done[(r["instance"], r["arm"])] = r
        except Exception:
            pass
out = open(a.out, "a")
SOLVER_KW = {"rule": "pdlp", "eta_budget": 1e4, "fill": 0.99}


def solve_adaptive(d, cand, plan, kw, obj_ref):
    """Production driver. Every ladder rung is charged; N is the accepted rung's iteration count."""
    from atlas.backend.adaptive import run_adaptive
    from atlas.backend.hardware import prepare_base
    from atlas.bench import qp_cell
    from atlas.genome import Backend, Genome, GenomeParams, Metric
    from atlas.schemes.rescale import rescale_qp_problem
    t0 = time.perf_counter()
    prescaled, metric, decode = None, None, (lambda z: z)
    if cand == "std_ruiz":
        prob = qp_cell.to_composite(d, power_iters=100)
        pre = rescale_qp_problem(prob, "ruiz", iters=10, power_iters=100)
        prescaled = {"kind": "ruiz", "iters": 10, "alpha": None, "value": pre}
        steps_from, metric = pre[0], Metric(kind="ruiz", iters=10)
    elif cand == "std_raw":
        prob = qp_cell.to_composite(d, power_iters=100)
        steps_from = prob
    else:
        form = build_formulation(d, cand, plan=plan, **kw)
        prob, steps_from, decode = form.prob, form.prob, form.decode
    Lf, nb = float(steps_from.f.props.lipschitz), float(steps_from.L.norm_bound)
    t, s = 1.0 / Lf, 0.49 / ((1.0 / Lf) * nb ** 2)
    built, cert, _, _ = prepare_base(prob, Genome("condat_vu", GenomeParams(tau=t, sigma=s), metric=metric,
                                                  backend=Backend(device="gpu")), prescaled=prescaled)
    setup = time.perf_counter() - t0
    ladder, total_iters, wall = [], 0, setup
    for j in range(4):
        r = run_adaptive(built, cert, scheme="condat_vu", tol=a.tol / 10 ** j, max_iters=a.cap, K=64, **SOLVER_KW)
        x = decode(np.asarray(r["x"]))
        ok, rv, gap = accept(x, d, obj_ref, a.tol)
        total_iters += int(r["iters"]); wall += float(r["wall_s"]) + float(r["compile_s"])
        ladder.append({"rung": j, "iters": int(r["iters"]), "rv": rv, "gap": gap})
        if ok and r["iters"] < a.cap:
            return {"N": int(r["iters"]), "iters_all_rungs": total_iters, "rv": rv, "gap": gap, "ladder": ladder,
                    "t_total_dirty": wall, "normL": nb}
    return {"N": None, "iters_all_rungs": total_iters, "rv": rv, "gap": gap, "ladder": ladder, "normL": nb}


def emit(rec):
    out.write(json.dumps(rec, default=float) + "\n"); out.flush()
    print("  %-10s %-22s %s" % (rec["instance"], rec["arm"],
          {k: rec[k] for k in ("N", "iters_run", "t_s", "accepted", "rv", "gap", "error") if k in rec}), flush=True)


for key in a.instances.split(","):
    base = {"instance": key, "set": a.set, "host": socket.gethostname(), "tol": a.tol}
    try:
        d = load_mm(key) if a.set == "mm" else load_qplib(os.path.join(a.qplib_dir, f"QPLIB_{key}.qplib"))
        diag = diagnose(d)
        plan, why = plan_block(d, diag)
        pplan, pwhy = plan_product(d)
        obj_ref, ref_t = reference(d, timeout_s=900)
        base.update({"n": int(d["P"].shape[0]), "m": int(d["A"].shape[0]), "diag": diag,
                     "plan": plan["kind"] if plan else None, "plan_why": why, "obj_ref": obj_ref, "ref_s": ref_t,
                     "plan_n_x": plan["n_x"] if plan else None,
                     "fully_absorbed": plan.get("fully_absorbed") if plan else None,
                     "product_K": pplan["K"] if pplan else None, "product_cover": pplan["cover"] if pplan else None,
                     "product_fully_absorbed": pplan["fully_absorbed"] if pplan else None})
        print(f"== {key} n={base['n']} m={base['m']} fires={diag['fires']} plan={base['plan']} obj_ref={obj_ref}",
              flush=True)
        if obj_ref is None:
            emit({**base, "arm": "reference", "error": str(ref_t)}); continue
    except Exception:
        emit({**base, "arm": "setup", "error": traceback.format_exc()[-500:]}); continue
    if not a.no_baselines:
        pairs = ([(sv, e) for sv in ("osqp", "clarabel", "scs") for e in LADDER] if a.ladder
                 else [("osqp", 1e-3), ("osqp", 1e-6), ("clarabel", 1e-6), ("scs", 1e-6)])
        for solver, eps in pairs:
            arm = f"{solver}@{eps:g}"
            if (key, arm) in done:
                continue
            try:
                rb = run_baseline_proc(d, solver, eps, timeout_s=600, scratch=os.environ.get("OPTEVOLVE_SCRATCH"))
                rec = {**base, "arm": arm, "t_s": rb.get("t_s"), "status": rb.get("status")}
                if "error" in rb or rb.get("x") is None:
                    rec["error"] = str(rb.get("error"))[:300]
                else:
                    ok, rv, gap = accept(np.asarray(rb["x"]), d, obj_ref, a.tol)
                    rec.update({"accepted": bool(ok), "rv": rv, "gap": gap,
                                "accepted_at": {f"{t:g}": bool(rv <= t and gap <= t) for t in LADDER}})
            except Exception:
                rec = {**base, "arm": arm, "error": traceback.format_exc()[-300:]}
            emit(rec)
    cands = [("std_raw", {}, None, "std_raw"), ("std_ruiz", {}, None, "std_ruiz")]
    if plan and a.single:
        impls = ["sort", "bisect"] if plan["kind"] == "simplex" else ["bisect", "breakpoint", "newton"]
        for variant in ("reloc", "reloc_rows"):
            cands += [(variant, {"impl": i, **({"iters": 12} if i == "newton" else {})}, plan, f"{variant}:{i}")
                      for i in impls]
    if pplan:
        cands += [(variant, {"impl": "bisect"}, pplan, f"{variant}_prod:bisect") for variant in ("reloc", "reloc_rows")]
    unconverged = set()
    for cand, kw, plan_c, arm in cands:
        fam = arm.split(":")[0]
        if (key, arm) in done:
            if cand.startswith("reloc") and "N" in done[(key, arm)] and done[(key, arm)]["N"] is None:
                unconverged.add(fam)
            continue
        if fam in unconverged:           # an exact operator cannot rescue a formulation the first exact one could not
            continue
        try:
            if a.driver == "adaptive":
                m = solve_adaptive(d, cand, plan_c, kw, obj_ref)
                rec = {**base, "arm": arm, "driver": "adaptive", **m}
            elif a.ladder:
                form = build_formulation(d, cand, plan=plan_c, **kw)
                m = measure_ladder(form, d, obj_ref, cap=a.cap, K=a.K)
                rec = {**base, "arm": arm, "N_at": m["N_at"], "N": m["N_at"].get(f"{a.tol:g}"),
                       "iters_run": m["iters_run"], "setup_s": m["setup_s"], "t_iter_dirty": m["t_iter"],
                       "rv": m["row_violation"], "gap": m["obj_gap"], "normL": m["normL"], "Lf": m["Lf"]}
            else:
                form = build_formulation(d, cand, plan=plan_c, **kw)
                m = measure(form, d, obj_ref, tol=a.tol, cap=a.cap, K=a.K, probe_iters=200)
                rec = {**base, "arm": arm, "N": m["N"], "iters_run": m["iters_run"], "setup_s": m["setup_s"],
                       "t_iter_dirty": m["t_iter"], "rv": m["row_violation"], "gap": m["obj_gap"],
                       "normL": m["normL"], "Lf": m["Lf"]}
            if m["N"] is None and cand.startswith("reloc"):
                unconverged.add(fam)
        except Exception:
            rec = {**base, "arm": arm, "error": traceback.format_exc()[-500:]}
        emit(rec)
print("REAL_RUN_DONE", flush=True)
