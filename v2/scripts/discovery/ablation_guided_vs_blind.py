"""Guided versus blind mutation at a matched evaluation budget, the control CAKE runs for kernels
(typed IR with compiler diagnostics against raw code with timing only). Same gene space
(atlas.evolve.diagnose.GENE_SPACE), same start, same beam, same certificate (full-problem duality gap),
same budget of evaluated configurations. The guided arm chooses mutations from the diagnosis rules;
the blind arm chooses random single-gene changes (seeded) and sees only certified wall time.
Timings are medians of --reps solves after an untimed warm-up solve; the step size is timed."""
import argparse, json, statistics, sys
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
sys.path.insert(0, ".")
from atlas.bench import logreg_cell as lc
from atlas.evolve.diagnose import GENE_SPACE, GENE_SPACE_LARGE, evolve
from atlas.evolve.screened_runner import run_screened

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True); ap.add_argument("--baseline-json", required=True)
ap.add_argument("--baseline", default="cuml"); ap.add_argument("--bandwidth-json", required=True)
ap.add_argument("--peak-gbps", type=float, required=True)
ap.add_argument("--alpha", type=float, default=0.5); ap.add_argument("--lam-frac", type=float, default=0.1)
ap.add_argument("--tol", type=float, default=1e-6); ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--budget", type=int, default=24); ap.add_argument("--per-node", type=int, default=4)
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--arms", default="guided,blind")
ap.add_argument("--space", default="small", choices=("small", "large"), help="gene space of the blind arm")
ap.add_argument("--guided-cap", type=int, default=0, help="guided arm: at most this many proposals per expanded "
                "configuration, dominant factor of T = N x t first (0 = all proposals)")
ap.add_argument("--out", required=True)
a = ap.parse_args()
assert jax.devices()[0].platform == "gpu", f"not on a GPU: {jax.devices()}"
X = np.load(f"{a.data}/X.npy").astype(np.float64); y = np.load(f"{a.data}/y.npy").astype(np.float64)
lam = a.lam_frac * lc.lam_max(X, y, a.alpha)
bj = json.load(open(a.baseline_json)); b = bj[a.baseline]
assert b["passed"] and abs(bj["lam"] - lam) <= 1e-12 * lam, "baseline must solve the same instance"
baseline = {"seconds": b["seconds"], "evals": b["n_iter"]}
bw = {(r["op"], r["dtype"]): r["GBps"] for r in json.load(open(a.bandwidth_json))["rows"]}
hw = {"step_GBps": bw[("grad", "float64")], "peak_GBps": a.peak_gbps,
      "f32_XTv_GBps": bw[("XTv", "float32")], "f64_XTv_GBps": bw[("XTv", "float64")]}
DT = {"float32": jnp.float32, "float64": jnp.float64}
print(f"device {jax.devices()[0]}; n={X.shape[0]} p={X.shape[1]} lam={lam:.3e}; baseline {b['seconds']:.3f}s; "
      f"budget {a.budget}; seed {a.seed}", flush=True)


def make_eval(records):
    def evaluate(cfg):
        kw = {k: (DT[v] if k == "dtype" else v) for k, v in cfg.items()}
        try:
            run_screened(X, y, lam, a.alpha, tol=a.tol, **kw)
            rs = [run_screened(X, y, lam, a.alpha, tol=a.tol, **kw) for _ in range(a.reps)]
        except Exception as e:
            err = f"{type(e).__name__}: {str(e)[:200]}"
            records.append({"config": cfg, "error": err}); jax.clear_caches()
            return {"seconds": float("inf"), "evals": 0, "converged": False, "error": err, "p": X.shape[1], "nnz": X.shape[1]}
        r = dict(rs[0]); r.pop("x")
        r.update(seconds=statistics.median(x["wall_s"] for x in rs), p=X.shape[1], converged=all(x["converged"] for x in rs))
        records.append({"config": cfg, "seconds": r["seconds"], "evals": r["evals"], "converged": r["converged"]})
        jax.clear_caches()
        return r
    return evaluate


out = {"data": a.data, "lam": lam, "seed": a.seed, "budget": a.budget, "space": a.space, "guided_cap": a.guided_cap, "baseline": b, "arms": {}}
for arm in a.arms.split(","):
    records = []
    res = evolve(make_eval(records), {"K": 64, "screen": False, "step_rule": "power"}, baseline, hw,
                 max_rounds=100, patience=100, beam=2, proposer=arm, seed=a.seed, per_node=a.per_node,
                 budget=a.budget, space=GENE_SPACE_LARGE if a.space == "large" else GENE_SPACE,
                 guided_cap=a.guided_cap or None,
                 log=lambda m: print(f"[{arm}] {m}", flush=True))
    best = res["best"]
    out["arms"][arm] = {"trace": res["trace"], "evaluated": res["evaluated"], "best_seconds": best["seconds"],
                        "best_config": best["config"], "chain": res["chain"], "records": records}
    print(f"[{arm}] best {best['seconds']:.3f}s after {res['evaluated']} evaluated configs: {best['config']}", flush=True)
    json.dump(out, open(a.out, "w"), indent=1, default=str)
print("ABLATION_DONE")
