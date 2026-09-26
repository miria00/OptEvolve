"""Runner for the RouterBench budgeted-routing LP track (2026-09-15).

Stages (each writes one JSON line per record to --out):
  sizes      n, m, nnz at every rung of the size ladder
  verify     LP optimum vs the Lagrangian brute force (a knapsack dual, no solver), plus the routing baselines
  structure  what uni.plan_product / uni.plan_block / uni.diagnose say, and the two absorbable families
  solvers    HiGHS / PDLP / Clarabel / cuPDLP at 1e-3 and 1e-4, judged by uni.accept against a reference
  ours       measure_ladder for std_ruiz and the two product relocations, and router.solve at a wall budget

Solvers run ONLY through atlas.bench.baseline_proc.run_baseline_proc in their isolated envs.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_V2 = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
if _V2 not in sys.path:
    sys.path.insert(0, _V2)

os.environ.setdefault("OPTEVOLVE_BASELINE_ENV_ROOT",
                      "/home/miria/baseline_envs" if os.path.isdir("/home/miria/baseline_envs")
                      else "/workspace:/root/baseline_envs")

import numpy as np
import scipy.sparse as sp

import routerbench as RB


def _emit(rec, out):
    print(json.dumps(rec, default=float), flush=True)
    if out:
        with open(out, "a") as fh:
            fh.write(json.dumps(rec, default=float) + "\n")


# ----------------------------------------------------------------------------------------------------------------
def _make(args, size, frac):
    """One instance of the requested variant. 'plain' is the budgeted routing LP whose optimum is closed form;
    'cap' adds per-model capacity rows, which removes the closed form (see routerbench.make_routing_capacitated)."""
    if args.variant == "cap":
        return RB.make_routing_capacitated(size, frac, cap_frac=args.cap_frac, permute=args.permute)
    return RB.make_routing(size, frac, budget_ref=args.budget_ref, permute=args.permute)


def _plans(args):
    return RB.PLANS_CAP if args.variant == "cap" else RB.PLANS


def _reference(args, d0, timeout_s):
    """(objective in the dict's MINIMIZATION sign, how it was obtained, extra fields).

    The plain variant is referenced by the closed-form dual, so acceptance never depends on a solver being right.
    The capacitated variant has more than one coupling row and no closed form, so it falls back to HiGHS at 1e-9
    (a simplex method, exact up to rounding on an LP) with Clarabel as the cross-check."""
    d = {k: d0[k] for k in ("P", "q", "r", "A", "l", "u")}
    if args.variant != "cap":
        q, _, _, cert = RB.brute_force_lp(d0["quality"], d0["cost"], d0["meta"]["budget"])
        return -q, "bruteforce", {"ref_rel_primal_dual_gap": cert["rel_primal_dual_gap"]}
    from atlas.bench.baseline_proc import run_baseline_proc
    info = {}
    obj = None
    for solver in ("highs", "clarabel"):
        t0 = time.perf_counter()
        rb = run_baseline_proc(d, solver, 1e-9, timeout_s=timeout_s, scratch=os.environ.get("OPTEVOLVE_SCRATCH"))
        if "error" in rb or rb.get("x") is None:
            info[f"ref_{solver}"] = {"error": str(rb.get("error"))[:200]}
            continue
        x = np.asarray(rb["x"])
        o = float(np.asarray(d["q"]) @ x)
        info[f"ref_{solver}"] = {"obj": o, "t_s": rb.get("t_s"), "wall_s": time.perf_counter() - t0,
                                 "row_violation": _row_violation(d, x), "status": rb.get("status")}
        if obj is None:
            obj = o
    if obj is None:
        return None, "none", info
    a, b = info.get("ref_highs", {}).get("obj"), info.get("ref_clarabel", {}).get("obj")
    if a is not None and b is not None:
        info["ref_agreement"] = abs(a - b) / (1.0 + abs(a))
    return obj, "highs1e-9", info


# ----------------------------------------------------------------------------------------------------------------
def stage_sizes(args, out):
    for s in args.sizes:
        for f in args.fracs:
            d = _make(args, s, f)
            _emit({"stage": "sizes", "size": s, "budget_frac": f, "budget_ref": args.budget_ref,
                   "variant": args.variant, **RB.instance_stats(d)}, out)


# ----------------------------------------------------------------------------------------------------------------
def stage_verify(args, out):
    """LP optimum (isolated solver) vs the closed-form dual, and the baseline-quality table."""
    from atlas.bench.baseline_proc import run_baseline_proc

    for s in args.sizes:
        for f in args.fracs:
            d = RB.make_routing(s, f, budget_ref=args.budget_ref)
            Q, C, B = d["quality"], d["cost"], d["meta"]["budget"]
            rec = {"stage": "verify", "size": s, "budget_frac": f, "budget_ref": args.budget_ref,
                   "budget": B, **{k: d["meta"][k] for k in ("cost_cheapest", "cost_dearest",
                                                             "cost_quality_optimal", "frac_of_max", "frac_of_opt")}}
            t0 = time.perf_counter()
            bf_obj, bf_cost, n_frac, cert = RB.brute_force_lp(Q, C, B)
            rec.update({"brute_force_quality": bf_obj, "brute_force_cost": bf_cost, "n_fractional": n_frac,
                        "dual_value": cert["dual"], "lambda_star": cert["lambda"],
                        "rel_primal_dual_gap": cert["rel_primal_dual_gap"],
                        "n_tied_prompts": cert["n_tied_prompts"],
                        "brute_force_s": time.perf_counter() - t0})
            for solver in args.solvers:
                rb = run_baseline_proc(d, solver, 1e-9, timeout_s=args.timeout,
                                       scratch=os.environ.get("OPTEVOLVE_SCRATCH"))
                if "error" in rb or rb.get("x") is None:
                    rec[f"lp_{solver}"] = {"error": str(rb.get("error"))[:200]}
                    continue
                x = np.asarray(rb["x"])
                qual = float(-np.asarray(d["q"]) @ x)
                rec[f"lp_{solver}"] = {"quality": qual, "cost": float(C.ravel() @ x), "t_s": rb.get("t_s"),
                                       "status": rb.get("status"),
                                       "row_violation": _row_violation(d, x),
                                       "rel_gap_vs_brute_force": abs(qual - bf_obj) / (1.0 + abs(bf_obj)),
                                       "rel_gap_vs_dual": abs(qual - cert["dual"]) / (1.0 + abs(cert["dual"]))}
            base = RB.baselines(Q, C, B)
            rec["baselines"] = {k: {"quality": q, "cost": c, "feasible": ok} for k, (q, c, ok) in base.items()}
            rec["models"] = RB.load_table()["models"]
            rec["n_prompts"] = int(Q.shape[0])
            _emit(rec, out)


def _row_violation(d, x):
    Ax = sp.csc_matrix(d["A"]) @ x
    l, u = np.asarray(d["l"]), np.asarray(d["u"])
    vl = np.where(np.isfinite(l), np.maximum(l - Ax, 0) / (1 + np.abs(np.where(np.isfinite(l), l, 0))), 0)
    vu = np.where(np.isfinite(u), np.maximum(Ax - u, 0) / (1 + np.abs(np.where(np.isfinite(u), u, 0))), 0)
    return float(np.max(np.maximum(vl, vu)))


# ----------------------------------------------------------------------------------------------------------------
def stage_structure(args, out):
    import uni
    import lp_families

    for s in args.sizes:
        for f in args.fracs:
            d0 = _make(args, s, f)
            d = {k: d0[k] for k in ("P", "q", "r", "A", "l", "u")}
            rec = {"stage": "structure", "size": s, "budget_frac": f, "permuted": bool(args.permute),
                   "variant": args.variant, **RB.instance_stats(d0)}
            t0 = time.perf_counter()
            pplan, pmsg = uni.plan_product(d)
            rec["plan_product"] = {"status": pmsg, "s": time.perf_counter() - t0}
            if pplan is not None:
                nb = np.bincount(pplan["bid"])
                rec["plan_product"].update({
                    "K": int(pplan["K"]), "cover": pplan["cover"], "n_x": pplan["n_x"],
                    "rows_kept": int(pplan["keep"].size), "block_sizes": sorted(set(nb.tolist()))[:6],
                    "absorbed_rows_are_equalities": bool(np.all(RB._row_is_equality(d)[pplan["rows"]])),
                    "fully_absorbed": pplan["fully_absorbed"],
                    "which_family": _which(d, pplan)})
            t0 = time.perf_counter()
            diag = lp_families.lp_diagnose(d)
            rec["diagnose"] = {**diag, "s": time.perf_counter() - t0}
            bplan, bmsg = uni.plan_block(d, diag)
            rec["plan_block"] = {"status": bmsg} if bplan is None else {
                "status": bmsg, "kind": bplan["kind"], "n_x": bplan["n_x"],
                "rows_kept": int(bplan["keep"].size), "dense_row": int(bplan["dense_row"]),
                "dense_row_is_equality": bool(RB._row_is_equality(d)[bplan["dense_row"]])}
            for name, fn in _plans(args).items():
                pl, msg = fn(d)
                if pl is None:
                    rec[f"plan_{name}"] = {"status": msg}
                    continue
                Ak = sp.csc_matrix(sp.csc_matrix(d["A"])[pl["keep"]][:, pl["perm"]])
                rec[f"plan_{name}"] = {"status": msg, "K": int(pl["K"]), "cover": pl["cover"],
                                       "n_x": pl["n_x"], "rows_kept": int(pl["keep"].size),
                                       "kept_nnz": int(Ak.nnz), "normL_kept": _norm(Ak)}
            A = sp.csc_matrix(d["A"])
            rec["normL_std_raw"] = _norm(A)
            _emit(rec, out)


def _which(d, pl):
    eq = RB._row_is_equality(d)
    n_eq = int(eq[pl["rows"]].sum())
    return "simplex" if n_eq == pl["K"] else ("budget" if n_eq == 0 else "mixed")


def _norm(M):
    from scipy.sparse.linalg import svds
    M = sp.csc_matrix(M).astype(float)
    if min(M.shape) <= 2:
        return float(np.linalg.norm(M.toarray(), 2))
    return float(svds(M, k=1, return_singular_vectors=False)[0])


# ----------------------------------------------------------------------------------------------------------------
def stage_solvers(args, out):
    """HiGHS / PDLP / Clarabel / cuPDLP at the requested accuracy targets, judged by uni.accept."""
    import uni
    from atlas.bench.baseline_proc import run_baseline_proc

    for s in args.sizes:
        for f in args.fracs:
            d0 = _make(args, s, f)
            d = {k: d0[k] for k in ("P", "q", "r", "A", "l", "u")}
            obj_ref, how, info = _reference(args, d0, args.timeout)
            rec0 = {"stage": "solvers", "size": s, "budget_frac": f, "permuted": bool(args.permute),
                    "variant": args.variant, "obj_ref": obj_ref, "obj_ref_source": how, **info,
                    **RB.instance_stats(d0)}
            if obj_ref is None:
                _emit(dict(rec0, accepted=False, error="no reference"), out)
                continue
            for solver in args.solvers:
                for tol in args.tols:
                    t0 = time.perf_counter()
                    rb = run_baseline_proc(d, solver, tol, timeout_s=args.timeout,
                                           scratch=os.environ.get("OPTEVOLVE_SCRATCH"))
                    wall = time.perf_counter() - t0
                    r = dict(rec0, solver=solver, tol=tol, wall_s=wall)
                    if "error" in rb or rb.get("x") is None:
                        r.update({"accepted": False, "error": str(rb.get("error"))[:300],
                                  "status": rb.get("status")})
                        _emit(r, out)
                        continue
                    x = np.asarray(rb["x"])
                    ok, rv, gap = uni.accept(x, d, obj_ref, tol)
                    r.update({"accepted": bool(ok), "row_violation": rv, "obj_gap": gap,
                              "t_solver_s": rb.get("t_s"), "status": rb.get("status"),
                              "iters": rb.get("iters") or rb.get("n_iter")})
                    _emit(r, out)


# ----------------------------------------------------------------------------------------------------------------
def stage_ours(args, out):
    """measure_ladder for std_ruiz and the two relocations, then router.solve at a wall budget."""
    import uni
    import lp_families

    for s in args.sizes:
        for f in args.fracs:
            d0 = _make(args, s, f)
            d = {k: d0[k] for k in ("P", "q", "r", "A", "l", "u")}
            obj_ref, how, info = _reference(args, d0, args.timeout)
            rec = {"stage": "ours", "size": s, "budget_frac": f, "permuted": bool(args.permute),
                   "variant": args.variant, "obj_ref": obj_ref, "obj_ref_source": how, **info,
                   **RB.instance_stats(d0)}
            if obj_ref is None:
                _emit(rec, out)
                continue
            plans = {"greedy": uni.plan_product(d)[0]}
            for name, fn in _plans(args).items():
                plans[name] = fn(d)[0]
            for cand, plname in args.arms:
                pl = plans.get(plname) if plname else None
                if plname and pl is None:
                    continue
                key = f"{cand}[{plname or '-'}]"
                try:
                    t0 = time.perf_counter()
                    form = lp_families.lp_formulation(d, cand, plan=pl, impl="bisect", omega=args.omega)
                    m = uni.measure_ladder(form, d, obj_ref, tols=tuple(args.tols), cap=args.cap,
                                           deadline=time.perf_counter() + args.ladder_budget)
                    rec[key] = {"N_at": m["N_at"], "iters_run": m["iters_run"], "normL": m["normL"],
                                "t_iter_s": m["t_iter"], "setup_s": m["setup_s"],
                                "row_violation": m["row_violation"], "obj_gap": m["obj_gap"],
                                "wall_s": time.perf_counter() - t0}
                except Exception as e:                      # one arm failing must not lose the others
                    rec[key] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
                _emit({**{k: rec[k] for k in ("stage", "size", "budget_frac")}, "arm": key, **rec[key]}, None)
            _emit(rec, out)


def stage_router(args, out):
    import router
    from vault import Vault

    v = Vault(args.vault)
    for s in args.sizes:
        for f in args.fracs:
            d0 = _make(args, s, f)
            d = {k: d0[k] for k in ("P", "q", "r", "A", "l", "u")}
            obj_ref, how, info = _reference(args, d0, args.timeout)
            case = f"RB:{args.variant}:{s}:{f:g}"
            t0 = time.perf_counter()
            res = router.solve(case, v, args.device, tol=args.tol, budget_s=args.router_budget, d=d,
                               obj_ref=obj_ref, record_evidence=args.record)
            res.update({"stage": "router", "size": s, "budget_frac": f, "variant": args.variant,
                        "obj_ref_source": how, "wall_s": time.perf_counter() - t0, **RB.instance_stats(d0)})
            _emit(res, out)


STAGES = {"sizes": stage_sizes, "verify": stage_verify, "structure": stage_structure, "solvers": stage_solvers,
          "ours": stage_ours, "router": stage_router}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stages", nargs="+", choices=sorted(STAGES))
    ap.add_argument("--sizes", nargs="*", type=int, default=[256, 1024, 4096])
    ap.add_argument("--fracs", nargs="*", type=float, default=[0.5])
    ap.add_argument("--budget-ref", default="opt")
    ap.add_argument("--variant", default="plain", choices=("plain", "cap"))
    ap.add_argument("--cap-frac", type=float, default=0.25)
    ap.add_argument("--solvers", nargs="*", default=["highs", "clarabel", "pdlp"])
    ap.add_argument("--tols", nargs="*", type=float, default=[1e-2, 1e-3, 1e-4])
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--cap", type=int, default=200000)
    ap.add_argument("--ladder-budget", type=float, default=300.0)
    ap.add_argument("--omega", type=float, default=1.0)
    ap.add_argument("--permute", action="store_true")
    ap.add_argument("--clarabel-ref", action="store_true")
    ap.add_argument("--arms", nargs="*", default=["std_ruiz:", "reloc_rows:greedy", "reloc_rows:simplex"])
    ap.add_argument("--vault", default="/workspace/optevolve/vault/vault.jsonl")
    ap.add_argument("--device", default="h100")
    ap.add_argument("--tol", type=float, default=1e-3)
    ap.add_argument("--router-budget", type=float, default=120.0)
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    args.arms = [tuple(a.split(":", 1)) if ":" in a else (a, "") for a in args.arms]
    for st in args.stages:
        STAGES[st](args, args.out)


if __name__ == "__main__":
    main()
