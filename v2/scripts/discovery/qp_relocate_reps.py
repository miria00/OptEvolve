"""Paired repetitions of the head-to-head for the DIAGNOSE-AND-RELOCATE form of a QP.

Why this is a separate file from qp_lasso_reps.py. That harness produced the four headline family
numbers (huber, lasso, svm, control) and its timing contract is the thing those numbers mean;
editing it to carry a second formulation would put the headline at risk for no gain. This is a
sibling: same rep-major interleaving, same cold-compile policy, same paired_ratio and the same
cross-instance t interval, and it imports both from atlas.bench.paired rather than reimplementing.

WHAT IS DIFFERENT, and why each difference is forced.

1. NO RUIZ RESCALE ON OUR ARM. The relocated composite carries a SimplexProduct in g, and diagonal
   rescaling does not preserve that atom (scaling x moves the set {x>=0, 1'x=1} off itself). The
   typed construction gate already refuses it: rescale.py:306 raises TypeCheckError for any problem
   that is not lp_standard. So the genome is built with metric=None, which prepare_base handles by
   skipping the metric gene entirely. This is not a concession, it is the gate working; the standard
   arm pays a rescale because it needs one and the relocated arm does not because it does not.

2. ACCEPTANCE IS OBJECTIVE-GAP, NOT KKT-ON-THE-REDUCED-OPERATOR. qp_rel_kkt against the RELOCATED
   problem would measure the KKT residual of an operator that no longer contains the budget row or
   the bounds, because those now live in g. A small value there would say nothing about the original
   problem and could accept a wrong answer. So the gate is, on the ORIGINAL (P,q,A,l,u):
       row_violation <= tol   AND   |obj - obj_ref| / |obj_ref| <= tol
   with obj_ref from an untimed Clarabel solve at 1e-9 per instance. For a convex QP, feasible plus
   matching objective is optimality, and this is strictly stronger than a residual on a reduced
   operator. This is the check that catches a wrong-problem bug: on 2026-09-14 an equality encoded
   as Box(0,0) instead of EqualitySet(g) converged, stayed feasible to 3.6e-12 on every row in the
   prox, and was caught ONLY by an objective gap of 2.6e+06.

3. THE RELOCATION IS DIAGNOSED, NOT ASSUMED. --relocate auto runs the detector and relocates only
   if it fires. The detector's output is recorded in the artifact either way, so the artifact can
   answer "did the framework choose this, or did the operator?".

The reference solve is OUTSIDE the timed region and outside the rep loop: it is a measuring
instrument, not an arm, and it runs at 1e-9 while the arms are compared at --tol.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import scipy.sparse as sp

ap = argparse.ArgumentParser()
ap.add_argument("--family", default="portfolio")
ap.add_argument("--size", type=int, default=20000)
ap.add_argument("--k", type=int, default=None, help="factor count for portfolio; default sqrt(n)")
ap.add_argument("--seeds", default="0")
ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--tol", type=float, default=1e-6)
ap.add_argument("--cap", type=int, default=200000)
ap.add_argument("--K", type=int, default=64)
ap.add_argument("--baselines", default="clarabel")
ap.add_argument("--baseline-timeout", type=float, default=1800.0)
ap.add_argument("--device", default=None, choices=("cpu", "gpu"))
ap.add_argument("--power-iters", type=int, default=100)
ap.add_argument("--relocate", default="auto", choices=("auto", "off", "force"),
                help="auto runs the detector and relocates only if it fires")
ap.add_argument("--ref-tol", type=float, default=1e-9,
                help="tolerance of the untimed reference solve that supplies obj_ref")
ap.add_argument("--out", required=True)
a = ap.parse_args()

# Same cold-compile policy as qp_lasso_reps: persistent cache off, clear_caches before each of our
# repetitions, so our arm pays compile on every rep exactly as each baseline subprocess does.
os.environ["ATLAS_JAX_CACHE_DIR"] = ""
_cache = "disabled: persistent cache off, jax.clear_caches() before each of our repetitions"

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from scipy.sparse.linalg import svds

from atlas.backend.adaptive import run_adaptive
from atlas.backend.hardware import prepare_base
from atlas.bench import qp_cell
from atlas.bench.baseline_proc import run_baseline_proc
from atlas.bench.paired import _tcrit, paired_ratio
from atlas.bench.qp_gen import FAMILIES, portfolio
from atlas.bench.mps_cell import SparseLinOp
from atlas.certify.residuals import qp_accuracy_judge
from atlas.genome import Backend, Genome, GenomeParams, Metric
from atlas.ir.spec import CompositeProblem
from atlas.operators.base import PropSet
from atlas.operators.prox import (Box, BoxHyperplaneProduct, EqualitySet,
                                  QuadraticForm, SimplexProduct)
from atlas.schemes.rescale import rescale_qp_problem

SOLVER_KW = {"rule": "pdlp", "eta_budget": 1e4, "fill": 0.99}


def diagnose(prob):
    """Does one dense row carry the top singular direction of the rescaled constraint operator?

    Returns the row's share of the top left singular vector, the singular gap, and that row's
    density. Fires when a single row holds more than half the top direction AND the operator is
    effectively rank-one at the top (gap >= 3). Measured on the five families: fires on portfolio
    alone (row mass 0.986, gap 16.0) against row densities 0.0005 to 0.09 elsewhere.
    """
    resc, _, _, _ = rescale_qp_problem(prob, "ruiz", iters=10, power_iters=a.power_iters)
    A = sp.csc_matrix(resc.data["A"]).astype(np.float64)
    U, S, _ = svds(A, k=2)
    order = np.argsort(S)[::-1]
    S, U = S[order], U[:, order]
    mass = U[:, 0] ** 2
    j = int(np.argmax(mass))
    Ar = sp.csr_matrix(A)
    density = float(Ar.indptr[j + 1] - Ar.indptr[j]) / A.shape[1]
    gap = float(S[0] / max(S[1], 1e-30))
    return {"top_row": j, "row_mass": float(mass[j]), "singular_gap": gap,
            "row_density": density, "fires": bool(mass[j] > 0.5 and gap >= 3.0)}


def relocation_plan(d, diag):
    """DERIVE the relocated form from the problem data, not from the family name.

    Given the dense row j the detector picked out, read the transform off (A, l, u):
      - row j must be an equality (l[j] == u[j]) whose coefficient vector a is supported on the
        first n_x variables, with n_x inferred as the width of that support;
      - the variables it touches must also carry simple bounds, supplied by identity rows;
      - everything else stays in L, and its right-hand side must be constant (Box) or vector
        (EqualitySet).
    Then {lo <= x <= hi, a'x = b} is exactly a box-hyperplane projection, and the special case
    a = 1, b = 1, lo = 0, hi = inf is the probability simplex, which has a cheaper exact projection.

    Returning None means "not relocatable by this rule", which is a legitimate answer and is what
    keeps the rule falsifiable rather than a family lookup table.
    """
    A = sp.csc_matrix(d["A"]); l = np.asarray(d["l"], dtype=float); u = np.asarray(d["u"], dtype=float)
    j = int(diag["top_row"])
    if not np.isclose(l[j], u[j]):
        return None                                   # the dense row is not an equality
    Ar = sp.csr_matrix(A)
    arow = np.asarray(Ar[j].todense()).ravel()
    support = np.flatnonzero(arow)
    if support.size == 0:
        return None
    n_x = int(support.max()) + 1
    if not np.allclose(arow[n_x:], 0.0):
        return None                                   # support is not a prefix block
    a_vec = arow[:n_x]
    b_val = float(l[j])

    # Find the identity rows that supply the bounds on those n_x variables.
    lo = np.full(n_x, -np.inf); hi = np.full(n_x, np.inf); bound_rows = []
    for r in range(A.shape[0]):
        if r == j:
            continue
        s_, e_ = Ar.indptr[r], Ar.indptr[r + 1]
        if e_ - s_ != 1:
            continue
        c = int(Ar.indices[s_])
        if c >= n_x or not np.isclose(Ar.data[s_], 1.0):
            continue
        lo[c], hi[c] = max(lo[c], l[r]), min(hi[c], u[r])
        bound_rows.append(r)
    if len(bound_rows) != n_x:
        return None                                   # not every relocated variable is bounded
    if not (np.allclose(lo, lo[0]) and np.allclose(hi, hi[0])):
        return None                                   # BoxHyperplaneProduct takes scalar bounds

    keep = np.array(sorted(set(range(A.shape[0])) - set(bound_rows) - {j}), dtype=int)
    if keep.size == 0:
        return None
    lk, uk = l[keep], u[keep]
    if not np.allclose(lk, uk):
        return None                                   # residual rows must be equalities
    simplex = (np.allclose(a_vec, 1.0) and np.isclose(b_val, 1.0)
               and np.isclose(lo[0], 0.0) and np.isinf(hi[0]))
    return {"n_x": n_x, "a": a_vec, "b": b_val, "lo": float(lo[0]), "hi": float(hi[0]),
            "keep": keep, "rhs": lk, "simplex": bool(simplex),
            "rhs_zero": bool(np.allclose(lk, 0.0))}


def build_relocated(d, plan):
    """The bounds and the dense equality leave L; only `keep` rows remain."""
    P = sp.csc_matrix(d["P"]); q = np.asarray(d["q"])
    A_eq = sp.csc_matrix(sp.csc_matrix(d["A"])[plan["keep"]])
    L = SparseLinOp.from_scipy(A_eq, power_iters=a.power_iters)
    P_op = SparseLinOp.from_scipy(P, power_iters=a.power_iters)
    Lf = max(float(P_op.norm_bound), 1e-12)
    f = QuadraticForm(P=P_op, q=jnp.asarray(q), r=0.0,
                      props=PropSet(lipschitz=Lf, cocoercivity=1.0 / Lf,
                                    prox_friendly=False, strong_convexity=0.0))
    g = (SimplexProduct(n_x=plan["n_x"]) if plan["simplex"]
         else BoxHyperplaneProduct(a=jnp.asarray(plan["a"]), b=plan["b"], n_x=plan["n_x"],
                                   lo=plan["lo"], hi=plan["hi"]))
    # Box(0,0) is the indicator of {0} and is correct ONLY for a zero right-hand side. A family whose
    # residual equality is Lx = c with c nonzero needs EqualitySet; using Box there silently solves a
    # different problem that still converges and stays feasible on every row living in the prox.
    h = Box(lo=0.0, hi=0.0) if plan["rhs_zero"] else EqualitySet(c=jnp.asarray(plan["rhs"]))
    return CompositeProblem(
        f=f, g=g, h=h, L=L,
        data={"P": P, "q": q, "r": 0.0, "A": A_eq, "l": plan["rhs"], "u": plan["rhs"]},
        shape=(P.shape[0],), name="qp_relocated")


def accept(x, d, obj_ref):
    """On the ORIGINAL problem. Feasible AND matching objective; see the module docstring."""
    j = qp_accuracy_judge(x, d["P"], d["q"], d["r"], d["A"], d["l"], d["u"],
                          ref_obj=None, feas_tol=a.tol)
    obj = float(0.5 * x @ (sp.csc_matrix(d["P"]) @ x) + np.asarray(d["q"]) @ x)
    gap = abs(obj - obj_ref) / max(abs(obj_ref), 1e-300)
    return (j["row_violation"] <= a.tol and gap <= a.tol), j["row_violation"], gap, obj


def run_ours(d, plan, obj_ref):
    """Every repetition pays the whole setup: operator norms, compile, solve. Same span as the
    sibling harness, minus the rescale when the relocated form forbids one.

    plan is None for the standard formulation and a relocation_plan dict otherwise."""
    t0 = time.perf_counter()
    prescaled = None
    if plan is not None:
        prob = build_relocated(d, plan)
        metric = None
    else:
        # DOUBLE-RESCALE TRAP. prepare_base with metric="ruiz" rescales internally, so handing it an
        # ALREADY-rescaled problem equilibrates twice and silently solves a different problem: huber
        # at n_var 140,000 then converges to row_violation 4.32e+00, identical at every rung of the
        # tolerance ladder, which is the signature of a malformed problem rather than a loose one.
        # qp_lasso_reps avoids this by passing the ORIGINAL problem together with prescaled=, and
        # this mirrors that exactly. The relocated branch is unaffected because metric is None there
        # and no rescale happens at all.
        prob = qp_cell.to_composite(d, power_iters=a.power_iters)
        _pre = rescale_qp_problem(prob, "ruiz", iters=10, power_iters=a.power_iters)
        prescaled = {"kind": "ruiz", "iters": 10, "alpha": None, "value": _pre}
        prob_for_steps = _pre[0]
        metric = Metric(kind="ruiz", iters=10)
    _steps_from = prob if plan is not None else prob_for_steps
    Lf, nb = float(_steps_from.f.props.lipschitz), float(_steps_from.L.norm_bound)
    t, s = 1.0 / Lf, 0.49 / ((1.0 / Lf) * nb ** 2)
    t_build = time.perf_counter() - t0
    t0 = time.perf_counter()
    built, cert, _, _ = prepare_base(
        prob, Genome("condat_vu", GenomeParams(tau=t, sigma=s), metric=metric,
                     backend=Backend(device=a.device)), prescaled=prescaled)
    t_prep = time.perf_counter() - t0
    ladder = []
    t_solve_cum, t_compile_cum = 0.0, 0.0
    for j in range(4):
        r = run_adaptive(built, cert, scheme="condat_vu", tol=a.tol / 10 ** j,
                         max_iters=a.cap, K=a.K, **SOLVER_KW)
        x = np.asarray(r["x"])
        ok, rv, gap, obj = accept(x, d, obj_ref)
        # Record every rung, including the failing ones. Without this a failure cannot be diagnosed:
        # "hit the iteration cap" and "converged to something slightly infeasible" look identical in
        # the artifact, and on svm_dual nsamp=10000 two of three seeds failed at row_violation 3.5e-06
        # with an objective gap of 1.7e-09, which is the second case and needs a different fix.
        # CHARGE EVERY RUNG, not just the accepted one. The solver cannot know in advance which
        # tolerance will certify, so walking the ladder IS the work. Billing only the final rung
        # understates our cost whenever acceptance happens late, and on svm_dual EVERY run accepted
        # at the last rung (j=3), so the understatement was systematic on that family.
        t_solve_cum += float(r["wall_s"]); t_compile_cum += float(r["compile_s"])
        ladder.append({"target": a.tol / 10 ** j, "iters": int(r["iters"]),
                       "hit_cap": bool(r["iters"] >= a.cap),
                       "row_violation": rv, "obj_gap": gap,
                       "wall_s": float(r["wall_s"]), "compile_s": float(r["compile_s"])})
        if ok and r["iters"] < a.cap:
            return {"passed": True, "iters": int(r["iters"]),
                    "iters_all_rungs": sum(L["iters"] for L in ladder),
                    "row_violation": rv, "obj_gap": gap, "obj": obj, "norm_bound": nb,
                    "ladder_step": j, "ladder": ladder,
                    "t_build_s": t_build, "t_prepare_s": t_prep,
                    "compile_s": t_compile_cum, "t_solve_s": t_solve_cum,
                    "compile_s_accepted_rung": float(r["compile_s"]),
                    "t_solve_s_accepted_rung": float(r["wall_s"]),
                    "t_total_s": t_build + t_prep + t_compile_cum + t_solve_cum,
                    "t_total_s_accepted_rung_only": (t_build + t_prep + float(r["compile_s"])
                                                     + float(r["wall_s"]))}
    return {"passed": False, "row_violation": rv, "obj_gap": gap, "norm_bound": nb,
            "ladder": ladder, "iters": ladder[-1]["iters"] if ladder else None,
            "hit_cap": bool(ladder and ladder[-1]["hit_cap"])}


arms = ["ours"] + [b for b in a.baselines.split(",") if b]
out = {"args": vars(a), "solver_kw": SOLVER_KW, "xla_cache": _cache,
       "acceptance": "row_violation AND relative objective gap, both on the ORIGINAL (P,q,A,l,u)",
       "regime": "first_solve: every rep pays full setup on our side and full setup per baseline",
       "device": str(jax.devices()[0]), "instances": []}

for seed in [int(v) for v in a.seeds.split(",")]:
    if a.family == "portfolio":
        k = a.k or max(2, int(np.sqrt(a.size)))
        d = portfolio(a.size, k=k, seed=seed)
    else:
        d = FAMILIES[a.family](a.size, seed=seed)
        k = a.k
    n, m = d["P"].shape[0], d["A"].shape[0]
    print(f"  {a.family}(size={a.size}, k={k}, seed={seed}): n_var={n} n_con={m}", flush=True)

    diag = diagnose(qp_cell.to_composite(d, power_iters=a.power_iters))
    print(f"    detector: row {diag['top_row']} mass={diag['row_mass']:.4f} "
          f"gap={diag['singular_gap']:.1f} density={diag['row_density']:.4f} "
          f"-> {'FIRES' if diag['fires'] else 'silent'}", flush=True)
    relocate = (a.relocate == "force") or (a.relocate == "auto" and diag["fires"])
    plan = relocation_plan(d, diag) if relocate else None
    if relocate and plan is None:
        print("    detector fired but the rule could not derive a transform; running standard",
              flush=True)
    if plan is not None:
        print(f"    plan: n_x={plan['n_x']} bounds=[{plan['lo']}, {plan['hi']}] b={plan['b']:g} "
              f"residual_rows={len(plan['keep'])} "
              f"{'SIMPLEX' if plan['simplex'] else 'BOX-HYPERPLANE'} "
              f"{'Box(0,0)' if plan['rhs_zero'] else 'EqualitySet'}", flush=True)
    print(f"    formulation: {'RELOCATED' if plan is not None else 'standard'}", flush=True)

    ref = run_baseline_proc(d, "clarabel", a.ref_tol, timeout_s=a.baseline_timeout,
                            scratch=os.environ.get("OPTEVOLVE_SCRATCH"))
    if "error" in ref:
        print(f"    reference solve FAILED: {ref['error'][:80]}; skipping instance", flush=True)
        continue
    zr = ref["x"]
    obj_ref = float(0.5 * zr @ (sp.csc_matrix(d["P"]) @ zr) + np.asarray(d["q"]) @ zr)
    print(f"    reference obj={obj_ref:.12g} (clarabel @ {a.ref_tol:g}, untimed, not an arm)",
          flush=True)

    times = {kk: [] for kk in arms}
    recs = {kk: [] for kk in arms}
    for rep in range(a.reps):                      # REP-MAJOR, so every arm sees the same state
        for arm in arms:
            if arm == "ours":
                jax.clear_caches()
                r = run_ours(d, plan, obj_ref)
                ok, t = r.get("passed"), r.get("t_total_s")
            else:
                r = run_baseline_proc(d, arm, a.tol, timeout_s=a.baseline_timeout,
                                      scratch=os.environ.get("OPTEVOLVE_SCRATCH"))
                ok = "error" not in r
                t = r.get("t_s")
                if ok:
                    x = r["x"]
                    b_ok, b_rv, b_gap, b_obj = accept(x, d, obj_ref)
                    r = {"passed": True, "t_s": t, "iters": r.get("iters"), "obj": b_obj,
                         "status": r.get("status"), "row_violation": b_rv, "obj_gap": b_gap,
                         "meets_same_bar": b_ok}
            if ok:
                times[arm].append(float(t))
            recs[arm].append({kk: v for kk, v in r.items() if kk not in ("x", "y")})
            extra = ""
            if ok and arm == "ours":
                extra = f" (iters={r['iters']} rv={r['row_violation']:.1e} gap={r['obj_gap']:.1e})"
            elif ok:
                extra = f" (rv={r['row_violation']:.1e} gap={r['obj_gap']:.1e})"
            print(f"    rep {rep}: {arm:9s} {'%.3f s' % t if ok else 'FAIL'}{extra}", flush=True)

    inst = {"seed": seed, "n_var": n, "n_con": m, "k": k,
            "relocated": plan is not None,
            "plan": (None if plan is None else
                     {kk: (int(vv) if isinstance(vv, (int, np.integer)) else
                           (len(vv) if hasattr(vv, "__len__") else vv))
                      for kk, vv in plan.items()}),
            "detector": diag, "obj_ref": obj_ref, "times": times, "records": recs, "ratios": {}}
    for arm in arms[1:]:
        if len(times["ours"]) == len(times[arm]) == a.reps and a.reps > 1:
            inst["ratios"][arm] = paired_ratio(times["ours"], times[arm])
            print(f"    speedup of ours over {arm}: {inst['ratios'][arm]}", flush=True)
        elif times["ours"] and times[arm]:
            inst["ratios"][arm] = {
                "note": "unpaired: an arm failed a repetition",
                "median_ratio": statistics.median(times[arm]) / statistics.median(times["ours"])}
    out["instances"].append(inst)

if len(out["instances"]) > 1:
    out["cross_instance"] = {}
    for arm in arms[1:]:
        gs = [i["ratios"][arm]["geo_ratio"] for i in out["instances"]
              if arm in i["ratios"] and "geo_ratio" in i["ratios"][arm]]
        if len(gs) < 2:
            continue
        lg = [math.log(g) for g in gs]
        mean, sd = statistics.mean(lg), statistics.stdev(lg)
        half = _tcrit(len(lg) - 1, 0.95) * sd / math.sqrt(len(lg))
        out["cross_instance"][arm] = {
            "n_seeds": len(gs), "geo_ratio": math.exp(mean),
            "lo": math.exp(mean - half), "hi": math.exp(mean + half),
            "per_seed": gs, "min": min(gs), "max": max(gs),
            "note": "interval is over SEEDS (cross-instance), not over repetitions of one instance"}
        r = out["cross_instance"][arm]
        print(f"  cross-instance over {r['n_seeds']} seeds vs {arm}: {r['geo_ratio']:.2f}x "
              f"[{r['lo']:.2f}, {r['hi']:.2f}]  per-seed {min(gs):.2f}-{max(gs):.2f}x", flush=True)

json.dump(out, open(a.out, "w"), indent=1, default=str)
print("QP_RELOCATE_REPS_DONE")
