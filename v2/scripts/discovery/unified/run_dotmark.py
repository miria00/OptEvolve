"""Gap analysis for the DOTmark optimal-transport track (2026-09-15).

Stages, each writing one JSON line per instance to --out:

  build      sizes, memory of the explicit operator, mass balance, x_feas row violation
  ref        reference objective: POT's network simplex in a dedicated venv, cross-checked against HiGHS/Clarabel
             through atlas.bench.baseline_proc's isolated interpreters
  structure  uni.plan_product (K, cover, which marginal family), uni.diagnose + uni.plan_block, and the
             greedy's tie structure (every general row of a square instance has the same nnz)
  gap        router.solve at the requested tol and budget: solved, choice, binding factor, spec, uncovered
  ladder     iterations to 1e-2 and 1e-3 for std_ruiz, source-absorbed and sink-absorbed, the rows-vs-columns
             planner experiment. Same acceptance check on the ORIGINAL problem for all three.
  opcost     per-iteration cost of the product projection as K grows, and the per-iteration budget an operator
             would have to meet for the relocated formulation to beat the chosen one

H100 WALL-CLOCK IS CONTAMINATED (a vLLM server holds 70 GB and the SMs). Iteration counts and structure are the
primary numbers; every second reported here is indicative only.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(_HERE))))

import numpy as np
import scipy.sparse as sp

import dotmark_ot

POT_PY = os.environ.get("OPTEVOLVE_POT_PYTHON", "/root/baseline_envs/venv_pot/bin/python")


def _jsonable(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist() if o.size <= 32 else {"shape": list(o.shape), "head": o.ravel()[:8].tolist()}
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)


def emit(out, rec):
    line = json.dumps(rec, default=_jsonable)
    with open(out, "a") as fh:
        fh.write(line + "\n")
    print(line, flush=True)


def build(cls, pair, res, permute=True):
    t0 = time.perf_counter()
    d = dotmark_ot.make_ot(cls, pair, res, permute=permute)
    return d, time.perf_counter() - t0


def _mem(d):
    A = sp.csc_matrix(d["A"])
    return {"A_rows": int(A.shape[0]), "A_cols": int(A.shape[1]), "A_nnz": int(A.nnz),
            "A_bytes": int(A.data.nbytes + A.indices.nbytes + A.indptr.nbytes),
            "q_bytes": int(np.asarray(d["q"]).nbytes),
            "lu_bytes": int(np.asarray(d["l"]).nbytes + np.asarray(d["u"]).nbytes)}


# ----------------------------------------------------------------------------------------------------------------
# Reference objective
# ----------------------------------------------------------------------------------------------------------------
def ref_pot(cls, pair, res, p=2.0, timeout_s=7200):
    """Exact network-simplex optimum from POT, in its own venv (POT is NOT in any shared environment)."""
    script = os.path.join(_HERE, "dotmark_ref_pot.py")
    cmd = [POT_PY, script, cls, str(pair), str(res), str(p)]
    t0 = time.perf_counter()
    try:
        pr = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s,
                            env=dict(os.environ, OMP_NUM_THREADS="16"))
    except subprocess.TimeoutExpired:
        return {"solver": "pot_network_simplex", "error": f"timeout after {timeout_s}s"}
    if pr.returncode != 0:
        return {"solver": "pot_network_simplex", "error": (pr.stderr or pr.stdout).strip()[-400:]}
    rec = json.loads(pr.stdout.strip().splitlines()[-1])
    rec["wall_s"] = time.perf_counter() - t0
    return rec


def ref_engine(d, solver, eps=1e-9, timeout_s=3600):
    from atlas.bench.baseline_proc import run_baseline_proc
    t0 = time.perf_counter()
    rb = run_baseline_proc(dotmark_ot.solver_dict(d), solver, eps, timeout_s=timeout_s,
                           scratch=os.environ.get("OPTEVOLVE_SCRATCH"))
    if "error" in rb or rb.get("x") is None:
        return {"solver": solver, "error": str(rb.get("error"))[:300], "wall_s": time.perf_counter() - t0}
    x = np.asarray(rb["x"])
    import lp_families
    return {"solver": solver, "obj": float(np.asarray(d["q"]) @ x), "t_solver_s": rb.get("t_s"),
            "row_violation": lp_families.row_violation(d, x), "wall_s": time.perf_counter() - t0}


# ----------------------------------------------------------------------------------------------------------------
# Structure
# ----------------------------------------------------------------------------------------------------------------
def structure(d):
    import uni
    A = sp.csr_matrix(d["A"])
    fam = np.asarray(d["meta"]["row_family"])
    nnz = np.diff(A.indptr)
    gen = np.flatnonzero(fam != 2)
    rec = {"general_rows": int(gen.size), "general_row_nnz": sorted(set(nnz[gen].tolist()))[:6],
           "n_distinct_general_nnz": int(len(set(nnz[gen].tolist()))),
           "lowest_indexed_general_row": int(gen.min()),
           "family_of_lowest_general_row": ["source", "sink"][int(fam[gen.min()])]}
    t0 = time.perf_counter()
    pplan, pmsg = uni.plan_product(d)
    rec["plan_product_s"] = time.perf_counter() - t0
    if pplan is None:
        rec["plan_product"] = {"status": pmsg}
    else:
        side = dotmark_ot.plan_side_of(d, pplan)
        rec["plan_product"] = {"status": pmsg, "kind": pplan["kind"], "K": int(pplan["K"]),
                               "cover": float(pplan["cover"]), "n_x": int(pplan["n_x"]),
                               "rows_kept": int(pplan["keep"].size),
                               "block_sizes": sorted(set(np.bincount(pplan["bid"]).tolist()))[:6],
                               "fully_absorbed": bool(pplan["fully_absorbed"]), "side": side}
        for s in ("source", "sink"):
            t0 = time.perf_counter()
            ps = dotmark_ot.plan_side(d, s)
            rec[f"plan_{s}"] = {"K": int(ps["K"]), "cover": float(ps["cover"]), "n_x": int(ps["n_x"]),
                               "rows_kept": int(ps["keep"].size), "build_s": time.perf_counter() - t0,
                               "agrees_with_greedy": dotmark_ot.plans_agree(ps, pplan),
                               "b_lo_min": float(np.min(ps["b_lo"])), "b_lo_max": float(np.max(ps["b_lo"])),
                               "zero_rhs_blocks": int((np.asarray(ps["b_lo"]) == 0).sum())}
    t0 = time.perf_counter()
    diag = uni.diagnose(d)
    rec["diagnose"] = dict(diag, wall_s=time.perf_counter() - t0)
    bplan, bmsg = uni.plan_block(d, diag)
    rec["plan_block"] = {"status": bmsg} if bplan is None else {
        "status": bmsg, "kind": bplan["kind"], "n_x": int(bplan["n_x"]), "rows_kept": int(bplan["keep"].size),
        "dense_row": int(bplan["dense_row"]),
        "family_of_absorbed_row": ["source", "sink", "bound"][int(np.asarray(d["meta"]["row_family"])[bplan["dense_row"]])]}
    return rec


# ----------------------------------------------------------------------------------------------------------------
# Gap analysis and ladders
# ----------------------------------------------------------------------------------------------------------------
def gap(d, case, obj_ref, vault_path, device, tol, budget_s):
    import router
    from vault import Vault
    v = Vault(vault_path)
    t0 = time.perf_counter()
    r = router.solve(case, v, device, tol=tol, budget_s=budget_s, d=d, obj_ref=obj_ref)
    r["wall_total_measured_s"] = time.perf_counter() - t0
    return r


def ladder(d, obj_ref, cands, cap, deadline_s, tols=(1e-2, 1e-3)):
    """Iterations to each tol for a list of (label, cand, plan) triples, one measure_ladder run each."""
    import uni
    out = {}
    for label, cand, plan in cands:
        t0 = time.perf_counter()
        try:
            form = uni.build_formulation(d, cand, plan=plan, impl="bisect")
            m = uni.measure_ladder(form, d, obj_ref, tols=tuple(tols), cap=cap,
                                   deadline=None if deadline_s is None else time.perf_counter() + deadline_s)
            out[label] = {"N_at": m["N_at"], "iters_run": m["iters_run"], "normL": m["normL"], "Lf": m["Lf"],
                          "setup_s": m["setup_s"], "t_iter_s": m["t_iter"], "row_violation": m["row_violation"],
                          "obj_gap": m["obj_gap"], "wall_s": time.perf_counter() - t0,
                          "absorbed": int(form.absorbed), "form": form.name,
                          "L_shape": list(sp.csc_matrix(form.L_csc).shape), "L_nnz": int(sp.csc_matrix(form.L_csc).nnz)}
        except Exception as e:                                # an OOM on one candidate must not lose the others
            out[label] = {"error": f"{type(e).__name__}: {e}"[:400], "wall_s": time.perf_counter() - t0}
        print(json.dumps({label: out[label]}, default=_jsonable), flush=True)
    return out


def opcost(d, plan, reps=5, iters=300, rounds=3):
    """Per-call cost of uni.product_project on this instance's K blocks, and the per-iteration cost of the whole
    relocated iteration and of std_ruiz, so the projection's share is separable from the rest of the iteration.

    The three quantities are timed ROUND-ROBIN and reported by their MINIMUM over rounds. On this H100 a vLLM server
    holds 70 GB and the SMs, and back-to-back measurements of the same kernel differ by up to 3x between contention
    epochs; a single sequential pass therefore charges whichever formulation happened to run in the busy window.
    Interleaving plus a minimum makes the RATIO between the three (the number the cost model needs) far less
    sensitive to that, though no absolute second here is trustworthy.
    """
    import jax
    import jax.numpy as jnp
    import uni
    lo, hi = jnp.asarray(plan["lo"]), jnp.asarray(plan["hi"])
    a, bid = jnp.asarray(plan["a"]), jnp.asarray(plan["bid"])
    b_lo, b_hi = jnp.asarray(plan["b_lo"]), jnp.asarray(plan["b_hi"])
    K = int(plan["K"])
    rng = np.random.default_rng(0)
    v = jnp.asarray(rng.standard_normal(int(plan["n_x"])))
    f = jax.jit(lambda z: uni.product_project(z, lo, hi, a, bid, b_lo, b_hi, K, iters=60))
    t0 = time.perf_counter()
    jax.block_until_ready(f(v))
    compile_s = time.perf_counter() - t0
    form = uni.build_formulation(d, "reloc_rows", plan=plan, impl="bisect")
    fs = uni.build_formulation(d, "std_ruiz")
    proj, rel, std, rel_setup, std_setup = [], [], [], [], []
    for _ in range(rounds):
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            jax.block_until_ready(f(v))
            ts.append(time.perf_counter() - t0)
        proj.append(float(np.min(ts)))
        c = uni.time_clean(form, iters, K=50)
        rel.append(c["per_iter_s"]); rel_setup.append(c["setup_s"])
        cs = uni.time_clean(fs, iters, K=50)
        std.append(cs["per_iter_s"]); std_setup.append(cs["setup_s"])
    return {"K": K, "n_x": int(plan["n_x"]), "block_size": int(plan["n_x"] // max(K, 1)), "compile_s": compile_s,
            "reps": reps, "rounds": rounds, "timing_iters": iters,
            "proj_per_call_s": float(np.min(proj)), "proj_per_call_s_all": proj,
            "reloc_per_iter_s": float(np.min(rel)), "reloc_per_iter_s_all": rel,
            "std_ruiz_per_iter_s": float(np.min(std)), "std_ruiz_per_iter_s_all": std,
            "reloc_setup_s": float(np.min(rel_setup)), "std_ruiz_setup_s": float(np.min(std_setup))}


# ----------------------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=("list", "build", "ref", "structure", "gap", "ladder", "opcost", "all"))
    ap.add_argument("--class", dest="cls", default="ClassicImages")
    ap.add_argument("--pair", type=int, default=0)
    ap.add_argument("--res", type=int, nargs="*", default=[16, 32])
    ap.add_argument("--out", default="/dev/shm/dotmark/results.jsonl")
    ap.add_argument("--vault", default="/workspace/optevolve/vault/vault.jsonl")
    ap.add_argument("--device", default="h100")
    ap.add_argument("--tol", type=float, default=1e-3)
    ap.add_argument("--budget", type=float, default=120.0)
    ap.add_argument("--cap", type=int, default=200000)
    ap.add_argument("--ladder-deadline", type=float, default=900.0)
    ap.add_argument("--ref-json", default=None, help="jsonl of earlier ref records, to avoid re-solving")
    ap.add_argument("--no-permute", action="store_true")
    ap.add_argument("--opcost-iters", type=int, default=300)
    ap.add_argument("--opcost-rounds", type=int, default=3)
    ap.add_argument("--cands", nargs="*", default=None,
                    help="ladder candidates to run (std_ruiz absorb_source_rows absorb_sink_cols); default all. "
                         "One per process keeps two formulations' operators off the GPU at once, and lets a "
                         "candidate that the deadline censored be re-run alone with a longer one.")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    if args.stage == "list":
        emit(args.out, {"stage": "list", "root": dotmark_ot.root(), "listing": dotmark_ot.listing()})
        return

    refs = {}
    if args.ref_json and os.path.exists(args.ref_json):
        for ln in open(args.ref_json):
            r = json.loads(ln)
            if r.get("stage") == "ref" and r.get("obj_ref") is not None:
                refs[(r["class"], r["pair"], r["res"])] = r["obj_ref"]

    for res in args.res:
        tag = {"stage": args.stage, "class": args.cls, "pair": args.pair, "res": res}
        d, build_s = build(args.cls, args.pair, res, permute=not args.no_permute)
        case = f"DOTMARK:{args.cls}:{args.pair}:{res}"
        base = dict(tag, case=case, name=d["name"], build_s=build_s, meta={k: v for k, v in d["meta"].items()
                    if k not in ("row_family", "col_perm", "row_perm")}, memory=_mem(d),
                    feasibility=dotmark_ot.feasibility(d))
        if args.stage == "build":
            emit(args.out, base)
            continue
        if args.stage == "ref":
            rec = dict(base, stage="ref")
            rec["pot"] = ref_pot(args.cls, args.pair, res, p=d["meta"]["p"])
            for eng in ("highs", "clarabel"):
                rec[eng] = ref_engine(d, eng, eps=1e-9, timeout_s=3600)
            cands = [rec["pot"].get("obj")] + [rec[e].get("obj") for e in ("highs", "clarabel")]
            good = [c for c in cands if c is not None]
            rec["obj_ref"] = good[0] if good else None
            if len(good) > 1:
                rec["ref_spread"] = float(max(good) - min(good))
                rec["ref_spread_rel"] = float((max(good) - min(good)) / (1 + abs(good[0])))
            emit(args.out, rec)
            continue
        if args.stage == "structure":
            emit(args.out, dict(base, **structure(d)))
            continue

        obj_ref = refs.get((args.cls, args.pair, res))
        if obj_ref is None:
            emit(args.out, dict(base, error="no reference objective: run --stage ref first"))
            continue
        base["obj_ref"] = obj_ref
        if args.stage == "gap":
            emit(args.out, dict(base, result=gap(d, case, obj_ref, args.vault, args.device, args.tol, args.budget)))
            continue
        if args.stage == "ladder":
            p_src, p_snk = dotmark_ot.plan_side(d, "source"), dotmark_ot.plan_side(d, "sink")
            cands = [("std_ruiz", "std_ruiz", None), ("absorb_source_rows", "reloc_rows", p_src),
                     ("absorb_sink_cols", "reloc_rows", p_snk)]
            if args.cands:
                cands = [c for c in cands if c[0] in args.cands]
            emit(args.out, dict(base, ladders=ladder(d, obj_ref, cands, args.cap, args.ladder_deadline)))
            continue
        if args.stage == "opcost":
            p_src = dotmark_ot.plan_side(d, "source")
            emit(args.out, dict(base, opcost=opcost(d, p_src, iters=args.opcost_iters, rounds=args.opcost_rounds)))
            continue


if __name__ == "__main__":
    main()
