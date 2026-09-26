"""Let the search pick the QP recipe, instead of us hand-picking condat_vu + Ruiz + PDLP.

The two QP wins so far (lasso 2.98x, huber 1.33x over Clarabel) came from a configuration WE chose in
qp_scale_sweep.run_ours. That is a measured win but not a discovered one. This drives
atlas.evolve.diagnose.evolve over the composer's operator genes (anchor, restart, rho, anderson,
step_frac, ratio) on a QP family, against a baseline time measured out of process, so the chain that
wins is one the loop found.

LIMITATION, stated because it changes what the number means: the composer is run on the ORIGINAL
composite, not the Ruiz-rescaled one. run_recipe certifies the problem it is handed, and only the
metric-gene path (_prepare_with_metric) maps a rescaled solution back to the original. So recipes here
do NOT get Ruiz, and their times are not directly comparable to run_ours's 5.268 s. What this answers
is which OPERATOR CHAIN wins; folding in the metric gene is a separate step.
"""
from __future__ import annotations

import argparse
import json
import os
import math
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--family", default="lasso", help="any key of atlas.bench.qp_gen.FAMILIES")
ap.add_argument("--size", type=int, default=20000)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--tol", type=float, default=1e-4)
ap.add_argument("--baseline-solver", default="clarabel")
ap.add_argument("--baseline-timeout", type=float, default=900.0)
ap.add_argument("--rounds", type=int, default=6)
ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--K", type=int, default=64)
ap.add_argument("--eval-cap", type=int, default=200000)
ap.add_argument("--composed-cap", type=int, default=20,
                help="cut a recipe off at this multiple of the best certified evaluation count so far; "
                     "without it a hopeless recipe runs to convergence before the loop can judge it")
ap.add_argument("--no-ruiz", action="store_true",
                help="run recipes on the UNSCALED composite (the old behaviour). Kept only for\n"
                     "comparison: without Ruiz, lasso n_var 50,000 reached cert 9.5e-2 at 4,032\n"
                     "evaluations and needs order 3.8M to certify, so no real-size search finishes.")
ap.add_argument("--power-iters", type=int, default=None,
                help="cap on the norm-bound power iteration (default 500); see qp_setup_profile.")
ap.add_argument("--step-frac", type=float, default=0.9,
                help="starting step fraction. MUST be < 1: the Condat-Vu gate requires\n"
                     "t*L_f/2 + t*s*||L||^2 < 1 STRICTLY, and step_frac=1.0 makes the left side "
                     "exactly 1, so the start node is rejected and the search has nothing to expand. "
                     "The hand-configured arm respects the same boundary with sigma=0.49/(tau*nb^2).")
ap.add_argument("--device", default=None, choices=("cpu", "gpu"))
ap.add_argument("--out", required=True)
a = ap.parse_args()

os.environ["ATLAS_JAX_CACHE_DIR"] = ""          # every evaluation pays its own compile
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from atlas.backend.device import detect_device
from atlas.bench import qp_cell
from atlas.bench.baseline_proc import run_baseline_proc
from atlas.bench.qp_gen import FAMILIES
from atlas.certify.residuals import qp_accuracy_judge, qp_rel_kkt
from atlas.evolve.diagnose import MUTATIONS, config_to_recipe, evolve
from atlas.evolve.kb_composer import run_recipe
from atlas.schemes.rescale import rescale_qp_problem
from atlas.typing_ import TypeCheckError

d = FAMILIES[a.family](a.size, seed=a.seed)
prob = qp_cell.to_composite(d, power_iters=a.power_iters)
# Recipes run on the RUIZ-RESCALED composite and are judged on the ORIGINAL, the same split
# _prepare_with_metric uses (x = D2 x', u = D1 u'). Without this the composer sees the raw condition
# number and cannot certify at real size. The rescale is setup WE pay, so it goes on our clock.
_t_rescale = 0.0
if a.no_ruiz:
    run_on, _d1, _d2 = prob, None, None
else:
    _t0 = time.perf_counter()
    _resc, _d1, _d2, _nb = rescale_qp_problem(prob, "ruiz", iters=10, power_iters=a.power_iters)
    _t_rescale = time.perf_counter() - _t0
    run_on = _resc


def _to_orig(xs, us):
    """Map a rescaled iterate back to original variables before any certificate is computed."""
    if _d2 is None:
        return np.asarray(xs), np.asarray(us)
    return _d2 * np.asarray(xs), _d1 * np.asarray(us)
n, m = d["P"].shape[0], d["A"].shape[0]
dev = detect_device()
print(f"  {a.family}({a.size}) n_var={n} n_con={m} on {jax.devices()[0]} ({dev['kind']})", flush=True)

b = run_baseline_proc(d, a.baseline_solver, a.tol, timeout_s=a.baseline_timeout,
                      scratch=os.environ.get("OPTEVOLVE_SCRATCH"))
if "error" in b:
    raise SystemExit(f"baseline {a.baseline_solver} did not run: {b['error']}")
baseline = {"seconds": float(b["t_s"]), "evals": int(b.get("iters") or 0)}
print(f"  baseline {a.baseline_solver}: {baseline['seconds']:.3f} s ({baseline['evals']} iters)", flush=True)

# CHAIN_RULES read these; they are recorded so a reader can tell a measured number from a nominal one.
hw = {"step_GBps": float(dev.get("tflops_f64", 0.0)), "peak_GBps": float(dev.get("tflops_f32", 0.0)),
      "f32_XTv_GBps": float(dev.get("tflops_f32", 0.0)), "f64_XTv_GBps": float(dev.get("tflops_f64", 0.0)),
      "NOTE": "TFLOPS stand-ins from detect_device, NOT measured bandwidths; bandwidth-gated rules "
              "will fire on the wrong scale until a real probe is supplied"}

records, best_evals = [], [None]


def _persist(partial=True):
    """Write what we have after EVERY candidate.

    The final json.dump only runs if the search completes. The previous real-size run was killed after
    7:46 and lost every evaluation it had already paid for; the sweep avoids this by dumping per size.
    """
    json.dump({"args": vars(a), "device": dev, "hw": hw, "baseline": {a.baseline_solver: baseline},
               "records": records, "partial": partial},
              open(a.out, "w"), indent=1, default=str)


def evaluate(cfg):
    """Every QP configuration goes to the composer: there is no screened engine for QP."""
    rec = config_to_recipe(cfg)
    # Give the first node the FULL budget: with nothing certified yet there is no sensible multiple
    # to take, and bootstrapping from 200 capped round 0 at 4000 evaluations (it stopped at 4032,
    # cert 9.5e-2). Once something certifies, tighten to composed_cap times that count.
    cap = a.eval_cap if best_evals[0] is None else min(a.eval_cap, max(2000, a.composed_cap * best_evals[0]))
    try:
        # Walk a tolerance ladder, exactly as qp_scale_sweep.run_ours does: ask the recipe for a
        # TIGHTER internal tolerance until the per-row feasibility judge accepts at feas_tol=a.tol.
        # With a single tol the round-0 node came back cert 8.8e-5 (inside 1e-4) but row_violation
        # outside it, so it was never certified, and evolve only expands CERTIFIED nodes: the search
        # had nothing to expand and stopped at round 1 every time.
        rs, accepted_spans, ladder = None, [], []
        for k in range(4):
            target = a.tol / 10 ** k
            trial, spans = [], []
            for _ in range(a.reps):
                jax.clear_caches()               # each rep pays compile, matching the cold baseline
                _t0 = time.perf_counter()
                trial.append(run_recipe(run_on, rec, tol=target, max_evals=cap, K=int(cfg.get("K", a.K))))
                # TIMING CONTRACT. run_recipe's wall_s is the SOLVE LOOP ONLY, while the baseline is
                # timed from before its own setup. Reporting wall_s against it claimed 92.38x on lasso
                # n_var 50,000 where the like-for-like sweep says 10.57x. Measure the whole call, so
                # typecheck, build and compile are on our clock too.
                spans.append(time.perf_counter() - _t0 + _t_rescale)
            xk, _ = _to_orig(trial[0]["x"], trial[0].get("u") if trial[0].get("u") is not None else np.zeros(m))
            jk = qp_accuracy_judge(xk, d["P"], d["q"], d["r"], d["A"], d["l"], d["u"],
                                   ref_obj=None, feas_tol=a.tol)
            ladder.append({"target": target, "row_violation": jk["row_violation"],
                           "converged": all(z["converged"] for z in trial)})
            rs, accepted_spans = trial, spans
            if jk["row_violation"] <= a.tol and all(z["converged"] for z in trial):
                break
    except TypeCheckError as e:
        records.append({"config": cfg, "gate_rejected": str(e)[:240]}); _persist()
        print(f"    {rec.name()}: gate rejected: {str(e)[:110]}", flush=True)
        return {"seconds": float("inf"), "evals": 0, "converged": False, "p": n, "nnz": n}
    except Exception as e:
        records.append({"config": cfg, "error": f"{type(e).__name__}: {str(e)[:240]}"}); _persist()
        print(f"    {rec.name()}: {type(e).__name__}: {str(e)[:110]}", flush=True)
        return {"seconds": float("inf"), "evals": 0, "converged": False, "p": n, "nnz": n}
    r0 = rs[0]
    x, u = _to_orig(r0["x"], r0["u"] if r0.get("u") is not None else np.zeros(m))
    cert = float(qp_rel_kkt(jnp.asarray(x), jnp.asarray(u), prob.f.P, prob.f.q, prob.L, prob.h.lo, prob.h.hi))
    j = qp_accuracy_judge(x, d["P"], d["q"], d["r"], d["A"], d["l"], d["u"], ref_obj=None, feas_tol=a.tol)
    ok = all(z["converged"] for z in rs) and cert <= a.tol and j["row_violation"] <= a.tol
    # diagnose.py's CHAIN_RULES read "p" and "nnz" (the sparse_solution rule is o["nnz"]/o["p"] < 0.3).
    # The GLM driver supplies them from the screened runner; for a QP they are the variable count and
    # the number of nonzero entries of the iterate. Omitting them crashes evolve with KeyError: 'nnz'.
    out = {"seconds": statistics.median(accepted_spans), "t_solve_only_s": statistics.median(z["wall_s"] for z in rs),
           "timed_span": "typecheck+build+compile+solve of the accepted ladder rung; to_composite is NOT included "
                         "(the baseline has no analogue) so this still flatters us slightly",
           "evals": int(r0.get("evals", 0)),
           "converged": bool(ok), "cert": cert, "row_violation": j["row_violation"],
           "p": n, "nnz": int((np.abs(x) > 0).sum()),
           "obj": float(0.5 * x @ (d["P"] @ x) + d["q"] @ x), "recipe": rec.name(),
           "walls": [z["wall_s"] for z in rs], "ladder": ladder}
    if ok and out["evals"]:
        best_evals[0] = min(best_evals[0] or out["evals"], out["evals"])
    records.append({"config": cfg, **out}); _persist()
    print(f"    {rec.name():<44s} {out['seconds']:8.3f}s (solve {out['t_solve_only_s']:.3f}s) "
          f"evals={out['evals']:<7d} cert={cert:.1e} {'OK' if ok else 'uncertified'}", flush=True)
    return out


start = {"base": "condat_vu", "step_frac": float(a.step_frac)}
# On QP the stock vocabulary is unreachable: halpern_anchor and anderson both force base="fbs", and fbs
# is inadmissible for a qp_standard composite (f is a QuadraticForm with mu=0, no grad_conj). Dropping
# the base override lets those operators combine with the CURRENT base (condat_vu), which is exactly the
# "old idea plus existing algorithm" combination the project is trying to discover.
QP_MUTATIONS = {**MUTATIONS, "halpern_anchor": {"anchor": True}, "anderson": {"anderson": 5}}
res = evolve(evaluate, start, baseline, hw, max_rounds=a.rounds, mutations=QP_MUTATIONS,
             log=lambda s: print(s, flush=True))
print("\nCHAIN")
for c in res["chain"]:
    print(f"  round {c['round']}: {c['seconds']:8.3f}s "
          + (f"{c['symptom']} -> {c['delta']}" if c.get("symptom") else "start"), flush=True)
best = res["best"]
certified = bool(best.get("converged")) and math.isfinite(best.get("seconds", float("inf")))
if not certified:
    print(f"\n  NO CERTIFIED CONFIGURATION. best attempt {best['config']} did not converge "
          f"(cert={best.get('cert')}, evals={best.get('evals')}); no ratio is reported, because a time "
          f"from an uncertified point is not a result.")
else:
    print(f"\n  best {best['config']} {best['seconds']:.3f}s vs {a.baseline_solver} {baseline['seconds']:.3f}s: "
          f"{baseline['seconds'] / best['seconds']:.2f}x "
          f"({'WIN' if res['beats_baseline'] else 'still behind'})")
json.dump({"args": vars(a), "device": dev, "hw": hw, "baseline": {a.baseline_solver: baseline},
           "chain": res["chain"], "beats_baseline": bool(res["beats_baseline"]) and certified,
           "best_certified": certified, "records": records},
          open(a.out, "w"), indent=1, default=str)
print("QP_DISCOVERY_DONE")
