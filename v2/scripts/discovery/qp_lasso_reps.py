"""Paired repetitions of the QP head-to-head, so a ratio can carry an interval instead of being n=1.

The lasso win at n_var 50,000 (ours 5.268 s vs Clarabel 15.719 s, 2026-09-13) was a single
measurement. This runs the arms REP-MAJOR and interleaved (rep 1: ours, clarabel, scs, osqp; rep 2:
same order; ...) so that the k-th measurement of every arm sees the same machine state and the paired
label is earned rather than assumed. Ratios are reported as a geometric mean with a t-interval on the
log ratios, via atlas.bench.paired.

Baselines run in their own interpreters (atlas.bench.baseline_proc), so this respects the rule that a
baseline never shares an environment with our solver, and OSQP stays on CPU.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--family", default="lasso")
ap.add_argument("--size", type=int, default=20000)
ap.add_argument("--seeds", default="0", help="comma separated; each seed is a distinct instance")
ap.add_argument("--reps", type=int, default=5)
ap.add_argument("--tol", type=float, default=1e-4)
ap.add_argument("--cap", type=int, default=200000)
ap.add_argument("--K", type=int, default=64,
                help="certificate cadence. 64 is qp_scale_sweep's default and is what the "
                     "5.268 s lasso result used; cadence is the only gene in both factors of "
                     "T = N x t, so changing it changes the algorithm being measured.")
ap.add_argument("--baselines", default="clarabel,scs,osqp")
ap.add_argument("--baseline-timeout", type=float, default=900.0)
ap.add_argument("--device", default=None, choices=("cpu", "gpu"))
ap.add_argument("--gpu-threshold", type=int, default=None,
                help="nnz at or above which the norm-bound power iteration runs on the GPU. "
                     "None (default) = never. Measured 1.40x on composite+rescale at 5.45M nnz "
                     "and 1.00x below the gate; norm_bound is bit-identical either way.")
ap.add_argument("--power-iters", type=int, default=None,
                help="cap on the norm-bound power iteration (default 500). 100 moves ||L|| by "
                     "5e-10 on lasso and leaves the Condat-Vu gate quantity at 0.990000, while the "
                     "iteration is 74%% of the rescale cost. 20 is NOT safe.")
ap.add_argument("--reuse-rescale", action="store_true",
                help="hand the rescale run_ours already computed to prepare_base rather than "
                     "letting _prepare_with_metric recompute it. Verified bit-identical "
                     "(max|dx|=max|du|=0, same iteration count). OFF by default so the interval "
                     "belongs to the same configuration as the headline number.")
ap.add_argument("--xla-cache", default="isolated", choices=("isolated", "shared"),
                help="isolated (default) points ATLAS_JAX_CACHE_DIR at a fresh directory so every "
                     "repetition truly pays compile. The persistent cache at ~/.cache/atlas_jax "
                     "OUTLIVES the process, so a shared cache would collapse compile_s toward zero "
                     "for our arm while each baseline subprocess still pays full setup.")
ap.add_argument("--out", required=True)
a = ap.parse_args()

if a.xla_cache == "isolated":                      # must precede the atlas import
    # Disable the PERSISTENT cache outright rather than pointing it at a fresh directory. A fresh
    # directory is still shared across repetitions WITHIN a run, so rep 0 compiled and rep 1 reused
    # it (measured: ours 0.354 s then 0.170 s), while every baseline runs in a fresh subprocess and
    # is cold on EVERY rep. atlas/__init__ treats an empty value as "no persistent cache".
    _cache = "disabled: persistent cache off, jax.clear_caches() before each of our repetitions"
    os.environ["ATLAS_JAX_CACHE_DIR"] = ""
else:
    _cache = os.environ.get("ATLAS_JAX_CACHE_DIR", "~/.cache/atlas_jax")

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from atlas.backend.adaptive import run_adaptive
from atlas.backend.hardware import prepare_base
from atlas.bench import qp_cell
from atlas.bench.baseline_proc import run_baseline_proc
from atlas.bench.paired import _tcrit, paired_ratio
from atlas.bench.qp_gen import FAMILIES
from atlas.certify.residuals import qp_accuracy_judge, qp_rel_kkt
from atlas.genome import Backend, Genome, GenomeParams, Metric
from atlas.schemes.rescale import rescale_qp_problem


# The exact run_adaptive configuration that produced the 5.268 s lasso result (qp_scale_sweep
# passes these; run_adaptive's own defaults differ, fill defaults to 0.98 not 0.99, so omitting them
# would measure a DIFFERENT algorithm while looking like a valid repetition).
SOLVER_KW = {"rule": "pdlp", "eta_budget": 1e4, "fill": 0.99}


def run_ours(d, prob):
    """Same timed span as the sweep: rescale + prepare + compile + solve, all inside the clock."""
    t0 = time.perf_counter()
    _pre = rescale_qp_problem(prob, "ruiz", iters=10, power_iters=a.power_iters,
                              gpu_threshold=a.gpu_threshold)
    run_on = _pre[0]
    Lf, nb = float(run_on.f.props.lipschitz), float(run_on.L.norm_bound)
    t, s = 1.0 / Lf, 0.49 / ((1.0 / Lf) * nb ** 2)
    t_rescale = time.perf_counter() - t0
    t0 = time.perf_counter()
    _ps = ({"kind": "ruiz", "iters": 10, "alpha": None, "value": _pre} if a.reuse_rescale else None)
    built, cert, _, _ = prepare_base(
        prob, Genome("condat_vu", GenomeParams(tau=t, sigma=s),
                     metric=Metric(kind="ruiz", iters=10), backend=Backend(device=a.device)),
        prescaled=_ps)
    t_prep = time.perf_counter() - t0
    for k in range(4):
        target = a.tol / 10 ** k
        r = run_adaptive(built, cert, scheme="condat_vu", tol=target, max_iters=a.cap, K=a.K,
                         **SOLVER_KW)
        x = np.asarray(r["x"])
        c = float(qp_rel_kkt(jnp.asarray(r["x"]), jnp.asarray(r["u"]), prob.f.P, prob.f.q,
                             prob.L, prob.h.lo, prob.h.hi))
        j = qp_accuracy_judge(x, d["P"], d["q"], d["r"], d["A"], d["l"], d["u"],
                              ref_obj=None, feas_tol=a.tol)
        if c <= a.tol and j["row_violation"] <= a.tol and r["iters"] < a.cap:
            return {"passed": True, "iters": int(r["iters"]), "cert": c,
                    "row_violation": j["row_violation"],
                    "t_rescale_s": t_rescale, "t_prepare_s": t_prep,
                    "compile_s": float(r["compile_s"]), "t_solve_s": float(r["wall_s"]),
                    "t_total_s": t_rescale + t_prep + float(r["compile_s"]) + float(r["wall_s"]),
                    "obj": float(0.5 * x @ (d["P"] @ x) + d["q"] @ x)}
    return {"passed": False}


arms = ["ours"] + [b for b in a.baselines.split(",") if b]
out = {"args": vars(a), "solver_kw": SOLVER_KW, "xla_cache": _cache, "regime": "first_solve: every rep pays rescale+prepare+compile+solve on our side and full setup on each baseline side", "device": str(jax.devices()[0]), "instances": []}
for seed in [int(v) for v in a.seeds.split(",")]:
    d = FAMILIES[a.family](a.size, seed=seed)
    prob = qp_cell.to_composite(d, power_iters=a.power_iters, gpu_threshold=a.gpu_threshold)
    n, m = d["P"].shape[0], d["A"].shape[0]
    print(f"  {a.family}(size={a.size}, seed={seed}): n_var={n} n_con={m}", flush=True)
    times = {k: [] for k in arms}
    recs = {k: [] for k in arms}
    # NO untimed warm-up for our arm. Each baseline runs in a FRESH subprocess every repetition, so a
    # baseline can never be warm; priming our compile cache would time us in the repeated-solve regime
    # against baselines in the first-solve regime and inflate the ratio. Both sides pay their own setup
    # every repetition, which is the same contract the sweep's 5.268 s number was measured under.
    for rep in range(a.reps):                           # REP-MAJOR: every arm sees the same state
        for arm in arms:
            if arm == "ours":
                if a.xla_cache == "isolated":
                    jax.clear_caches()          # every rep pays compile, like the cold baselines
                r = run_ours(d, prob)
                ok, t = r.get("passed"), r.get("t_total_s")
            else:
                r = run_baseline_proc(d, arm, a.tol, timeout_s=a.baseline_timeout,
                                      scratch=os.environ.get("OPTEVOLVE_SCRATCH"))
                ok = "error" not in r
                t = r.get("t_s")
                if ok:
                    # Record the BASELINE's accuracy too. Without this the artifact cannot answer
                    # "was the win accuracy-matched?", and settling that for control cost a separate
                    # 111 s run. Twelve earlier artifacts show n/r in the baseline columns for want of
                    # these four lines.
                    x, y = r["x"], r["y"]
                    b_cert = float(qp_rel_kkt(jnp.asarray(x), jnp.asarray(y), prob.f.P, prob.f.q,
                                              prob.L, prob.h.lo, prob.h.hi))
                    b_judge = qp_accuracy_judge(x, d["P"], d["q"], d["r"], d["A"], d["l"], d["u"],
                                                ref_obj=None, feas_tol=a.tol)
                    r = {"passed": True, "t_s": t, "iters": r.get("iters"),
                         "obj": float(0.5 * x @ (d["P"] @ x) + d["q"] @ x), "status": r.get("status"),
                         "cert": b_cert, "row_violation": b_judge["row_violation"]}
            if ok:
                times[arm].append(float(t))
            recs[arm].append({k: v for k, v in r.items() if k not in ("x", "y")})
            print(f"    rep {rep}: {arm:9s} {'%.3f s' % t if ok else 'FAIL'}", flush=True)
    inst = {"seed": seed, "n_var": n, "n_con": m, "times": times, "records": recs, "ratios": {}}
    for arm in arms[1:]:
        if len(times["ours"]) == len(times[arm]) == a.reps and a.reps > 1:
            inst["ratios"][arm] = paired_ratio(times["ours"], times[arm])
            g = inst["ratios"][arm]
            print(f"    speedup of ours over {arm}: {g}", flush=True)
        elif times["ours"] and times[arm]:
            inst["ratios"][arm] = {"note": "unpaired: an arm failed a repetition",
                                   "median_ratio": statistics.median(times[arm]) / statistics.median(times["ours"])}
    out["instances"].append(inst)
# Cross-instance aggregate. The per-instance intervals above are over REPETITIONS of one instance;
# this is the geometric mean over SEEDS, which is the quantity to quote when several seeds were run.
# It was previously computed ad hoc outside the harness, so the artifact could not reproduce it.
if len(out["instances"]) > 1:
    out["cross_instance"] = {}
    for arm in arms[1:]:
        gs = [i["ratios"][arm]["geo_ratio"] for i in out["instances"]
              if arm in i["ratios"] and "geo_ratio" in i["ratios"][arm]]
        if len(gs) < 2:
            continue
        lg = [math.log(g) for g in gs]
        mean = statistics.mean(lg)
        sd = statistics.stdev(lg)
        # Use the SAME t interval paired.py uses for the within-instance case. A normal 1.96 at
        # n=5 seeds (df=4, t=2.776) understates the width by ~42% and would put two intervals with
        # different conventions in one artifact.
        half = _tcrit(len(lg) - 1, 0.95) * sd / math.sqrt(len(lg))
        out["cross_instance"][arm] = {
            "n_seeds": len(gs), "geo_ratio": math.exp(mean),
            "lo": math.exp(mean - half), "hi": math.exp(mean + half),
            "per_seed": gs, "min": min(gs), "max": max(gs),
            "note": "interval is over SEEDS (cross-instance), not over repetitions of one instance",
        }
        r = out["cross_instance"][arm]
        print(f"  cross-instance over {r['n_seeds']} seeds vs {arm}: {r['geo_ratio']:.2f}x "
              f"[{r['lo']:.2f}, {r['hi']:.2f}]  per-seed {min(gs):.2f}-{max(gs):.2f}x", flush=True)
json.dump(out, open(a.out, "w"), indent=1, default=str)
print("QP_REPS_DONE")
