"""Where do the factorisation-based QP solvers lose? A size sweep on two standard scalable families.

Maros-Meszaros tops out near 10^4 variables, every instance factorises, and there we do not beat OSQP or
Clarabel. Clarabel is an interior-point method (a KKT factorisation per iteration) and OSQP factorises once and
reuses it, so their cost grows with fill-in, while our Condat-Vu arm only applies P and A. This script sweeps the
size of two families from the OSQP benchmark set and times every solver on the same instance at the same
certificate AND the per-row judge: a solve counts only when qp_rel_kkt <= --tol and no row is violated by more
than --tol of its own bound (qp_accuracy_judge). The global certificate alone is not enough: on the first smoke
instances our point passed at 3.7e-5 while violating a row by 1.5e-4, and SCS did the same, so "certified" times
would compare unequal answers. Each solver walks its own ladder (baselines tighten eps, we tighten the target)
until both tests pass; the reported time is that of the accepted run.

Families (both standard; generated here so no download is needed, seeds recorded):
  lasso  min 0.5||y||^2 + lam ||t||_1  s.t. y = A x - b, -t <= x <= t   (variables x, y, t)
  svm    min 0.5||w||^2 + C 1^T xi     s.t. xi >= 1 - d_i (a_i^T w), xi >= 0

Ours: Ruiz equilibration then Condat-Vu with the certified adaptive step ratio (the best QP arm measured on
Maros-Meszaros), invoked exactly as scripts/baselines/qp_adaptive.py does. Baselines run on the CPU in their own
tolerance ladder until our certificate passes, as in qp_headtohead.py.
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# --cold-cache must be honoured BEFORE atlas is imported: atlas/__init__ reads ATLAS_JAX_CACHE_DIR at
# import time and an empty value disables the persistent compilation cache. Without this the sweep can
# measure a WARM cache while qp_lasso_reps.py (which clears per rep) measures a cold one, which is why
# the sweep reported svm 2.24x where 5 repetitions give 1.35x [1.24, 1.48].
if "--cold-cache" in sys.argv:
    os.environ["ATLAS_JAX_CACHE_DIR"] = ""
import numpy as np
import scipy.sparse as sp
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from atlas.backend.adaptive import run_adaptive
from atlas.backend.hardware import prepare_base
from atlas.bench import qp_cell
from atlas.bench.baseline_proc import run_baseline_proc
from atlas.bench.qp_gen import FAMILIES
from atlas.certify.residuals import qp_rel_kkt, qp_accuracy_judge
from atlas.genome import Backend, Genome, GenomeParams, Metric
from atlas.schemes.rescale import rescale_qp_problem

ap = argparse.ArgumentParser()
ap.add_argument("--family", default="lasso", choices=tuple(FAMILIES))
ap.add_argument("--sizes", default="1000,3000,10000,30000,100000", help="n for lasso (features) or samples for svm")
ap.add_argument("--density", type=float, default=1e-3)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--tol", type=float, default=1e-4)
ap.add_argument("--cap", type=int, default=200000)
ap.add_argument("--K", type=int, default=64)
ap.add_argument("--eta-budget", type=float, default=1e4)
ap.add_argument("--baselines", default="osqp,scs,clarabel")
ap.add_argument("--baseline-timeout", type=float, default=900.0, help="seconds per baseline attempt")
ap.add_argument("--device", default=None, choices=("cpu", "gpu"))
ap.add_argument("--cold-cache", action="store_true",
                help="disable the persistent XLA cache and clear JAX caches before our arm, so a "
                     "single-shot sweep pays compile like the repetition harness does. Without it "
                     "sweep ratios are an UPPER BOUND.")
ap.add_argument("--gpu-threshold", type=int, default=None,
                help="nnz at or above which the norm-bound power iteration runs on the GPU. "
                     "None (default) = never. Measured 1.40x on composite+rescale at 5.45M nnz "
                     "and 1.00x below the gate; norm_bound is bit-identical either way.")
ap.add_argument("--power-iters", type=int, default=None,
                help="cap on the norm-bound power iteration (default 500). 100 changes ||A|| by "
                     "9e-10 relative on lasso, far inside the 1.02 safety inflation, and the "
                     "iteration is 74%% of the rescale cost. 20 is NOT safe (3.5e-2 > 2e-2).")
ap.add_argument("--reuse-rescale", action="store_true",
                help="hand the rescale run_ours already computed to prepare_base instead of "
                     "letting _prepare_with_metric recompute it. Equivalent by construction "
                     "(rescale_qp_problem is deterministic and its power iteration is seeded). "
                     "OFF by default so results stay comparable with runs made before this flag.")
ap.add_argument("--arms", default="ruiz+pdlp", help="our arms: <ruiz|none>+<pdlp|fixed>, comma separated. The "
                "ablation that isolates the certified adaptive ratio (pdlp) from equilibration (ruiz) inside the "
                "regime we win, so the win can be attributed rather than assumed")
ap.add_argument("--out", required=True)
a = ap.parse_args()
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)


def cert_of(d, prob, x, y):
    return float(qp_rel_kkt(jnp.asarray(x), jnp.asarray(y), prob.f.P, prob.f.q, prob.L, prob.h.lo, prob.h.hi))


def run_baseline(name, d, prob):
    """Loosest tolerance in the ladder whose point passes our certificate; timed per attempt."""
    tries = []
    for k in range(4):
        eps = a.tol / 10 ** k
        try:
            rec = run_baseline_proc(d, name, eps, timeout_s=a.baseline_timeout,
                                    scratch=os.environ.get("OPTEVOLVE_SCRATCH"))
            if "error" in rec:
                raise RuntimeError(f"{rec['error']} (interpreter {rec.get('solver_python')})")
            x, y, st = rec["x"], rec["y"], rec["status"]
        except Exception as ex:
            tries.append({"eps": eps, "error": f"{type(ex).__name__}: {str(ex)[:160]}"}); break
        t = float(rec["t_s"])          # timed inside the baseline's own interpreter
        c = cert_of(d, prob, x, y)
        j = qp_accuracy_judge(x, d["P"], d["q"], d["r"], d["A"], d["l"], d["u"], ref_obj=None, feas_tol=a.tol)
        tries.append({"eps": eps, "t_s": t, "cert": c, "status": str(st), "row_violation": j["row_violation"]})
        if c <= a.tol and j["row_violation"] <= a.tol:
            return {"passed": True, "t_s": t, "t_total_s": t, "timed_span": "setup+solve",
                    "eps": eps, "cert": c, "status": str(st), "tries": tries,
                    "obj": float(0.5 * x @ (d["P"] @ x) + d["q"] @ x), "verdict": j["verdict"],
                    "row_violation": j["row_violation"]}
    return {"passed": False, "tries": tries}


def run_ours(d, prob, arm="ruiz+pdlp"):
    """Condat-Vu with (optionally) Ruiz equilibration and (optionally) the certified adaptive step ratio.

    TIMING CONTRACT (fixed 2026-09-13). The baselines are timed from BEFORE their own setup
    (osqp.setup, SCS's cone build, Clarabel's presolve), so ours must be too. Until this date our
    equilibration ran before the clock started, prepare and compile were reported in separate fields,
    and comparisons quoted the solve loop alone: at 25k that turned a 1.22x loss to Clarabel into a
    reported 6.4x win. `t_s` is the solve loop and is kept for continuity; `t_total_s` is the
    like-for-like number and is the ONLY one a comparison may use.
    """
    if a.cold_cache:
        jax.clear_caches()          # match the repetition harness: this run pays its own compile
    met, rule = arm.split("+")
    metric = Metric(kind="ruiz", iters=10) if met == "ruiz" else None
    t0 = time.perf_counter()
    _pre = (None if metric is None else
            rescale_qp_problem(prob, "ruiz", iters=10, power_iters=a.power_iters,
                               gpu_threshold=a.gpu_threshold))
    run_on = prob if metric is None else _pre[0]
    Lf, nb = float(run_on.f.props.lipschitz), float(run_on.L.norm_bound)
    t, s = 1.0 / Lf, 0.49 / ((1.0 / Lf) * nb ** 2)
    t_rescale = time.perf_counter() - t0
    t0 = time.perf_counter()
    _ps = ({"kind": "ruiz", "iters": 10, "alpha": None, "value": _pre}
           if (a.reuse_rescale and _pre is not None) else None)
    built, cert, _, _ = prepare_base(prob, Genome("condat_vu", GenomeParams(tau=t, sigma=s),
                                                  metric=metric, backend=Backend(device=a.device)),
                                     prescaled=_ps)
    t_prep = time.perf_counter() - t0
    tries = []
    for k in range(4):                      # our ladder: tighten the target until the judge passes too
        target = a.tol / 10 ** k
        r = run_adaptive(built, cert, scheme="condat_vu", tol=target, max_iters=a.cap, K=a.K, rule=rule,
                         eta_budget=a.eta_budget, fill=0.99)
        x = np.asarray(r["x"])
        c = cert_of(d, prob, r["x"], r["u"])
        j = qp_accuracy_judge(x, d["P"], d["q"], d["r"], d["A"], d["l"], d["u"], ref_obj=None, feas_tol=a.tol)
        tries.append({"target": target, "t_s": float(r["wall_s"]), "iters": int(r["iters"]), "cert": c,
                      "row_violation": j["row_violation"]})
        if c <= a.tol and j["row_violation"] <= a.tol and r["iters"] < a.cap:
            return {"passed": True, "t_s": float(r["wall_s"]), "compile_s": float(r["compile_s"]),
                    "t_prepare_s": t_prep, "t_rescale_s": t_rescale,
                    "t_total_s": t_rescale + t_prep + float(r["compile_s"]) + float(r["wall_s"]),
                    "timed_span": "rescale+prepare+compile+solve (like-for-like with the baselines)",
                    "target": target, "iters": int(r["iters"]), "cert": c,
                    "verdict": j["verdict"], "row_violation": j["row_violation"], "tries": tries,
                    "obj": float(0.5 * x @ (d["P"] @ x) + d["q"] @ x)}
        if r["iters"] >= a.cap:
            break
    return {"passed": False, "t_prepare_s": t_prep, "t_rescale_s": t_rescale, "tries": tries}


print(f"device {jax.devices()[0]}; family {a.family}; sizes {a.sizes}; density {a.density}; tol {a.tol}", flush=True)
out = {"args": vars(a), "rows": []}
for size in [int(v) for v in a.sizes.split(",")]:
    d = FAMILIES[a.family](size, seed=a.seed, density=a.density)
    nvar, ncon, nnz = d["P"].shape[0], d["A"].shape[0], d["P"].nnz + d["A"].nnz
    prob = qp_cell.to_composite(d, power_iters=a.power_iters, gpu_threshold=a.gpu_threshold)
    rec = {"size": size, "n_var": nvar, "n_con": ncon, "nnz": nnz}
    for arm in a.arms.split(","):
        key = "ours" if arm == "ruiz+pdlp" else "ours:" + arm
        try:
            rec[key] = run_ours(d, prob, arm)
        except Exception as ex:
            rec[key] = {"passed": False, "error": f"{type(ex).__name__}: {str(ex)[:200]}"}
    for b in [s for s in a.baselines.split(",") if s]:
        rec[b] = run_baseline(b, d, prob)
    out["rows"].append(rec); json.dump(out, open(a.out, "w"), indent=1)
    f = lambda v: (f"{v.get('t_total_s', v['t_s']):8.3f}s total {v.get('verdict', '')}" if v.get("passed") else
                   # our own tries carry target/cert/row_violation and NO status, so the old form
                   # evaluated None[:28] and crashed the sweep on the first failure of one of our arms
                   f"  FAIL {str(v.get('error') or (v['tries'][-1].get('status') if v.get('tries') else '') or 'no certificate within the ladder')[:28]}")
    arms_txt = " | ".join(f"{k} {f(rec[k])}" for k in rec if isinstance(rec[k], dict) and k.startswith("ours"))
    print(f"n_var {nvar:8d} con {ncon:8d} nnz {nnz:9d} | {arms_txt} | "
          + " | ".join(f"{b} {f(rec[b])}" for b in a.baselines.split(",") if b), flush=True)
print("SCALE_SWEEP_DONE")
