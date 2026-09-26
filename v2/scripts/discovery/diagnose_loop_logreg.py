"""The automatic chain on one elastic-net logistic instance: start from plain FISTA, diagnose
why it loses to a measured baseline (atlas.evolve.diagnose), apply one knowledge-base mutation
per round, re-run with the full-problem duality-gap certificate, keep the fastest, repeat.
Timings are medians of --reps solves after an untimed warm-up solve, like the baselines'."""
import argparse, json, statistics, sys
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
sys.path.insert(0, ".")
from atlas.bench import logreg_cell as lc
from atlas.evolve.diagnose import config_to_recipe, evolve, route
from atlas.evolve.screened_runner import _EXE_CACHE, run_screened

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/enet_libsvm/epsilon_test")
ap.add_argument("--baseline-json", required=True, help="logreg_headtohead output holding the baseline")
ap.add_argument("--baseline", default="cuml")
ap.add_argument("--bandwidth-json", required=True, help="scripts/diagnostics/logreg_bandwidth.py output")
ap.add_argument("--peak-gbps", type=float, default=1008.0, help="device's nominal memory bandwidth")
ap.add_argument("--alpha", type=float, default=0.5); ap.add_argument("--lam-frac", type=float, default=0.1)
ap.add_argument("--tol", type=float, default=1e-6); ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--rounds", type=int, default=6); ap.add_argument("--out", required=True)
ap.add_argument("--composed-cap", type=int, default=20,
                help="a composed recipe is cut off after this multiple of the best certified evaluation "
                     "count seen so far. Without it a hopeless operator recipe runs to convergence before "
                     "the loop can judge it: the Halpern anchor took 84,096 evaluations and 292 s against "
                     "0.143 s for the best screened configuration, and that is paid once per repetition.")
ap.add_argument("--allow-failing-baseline", action="store_true",
                help="tune even when the baseline never reached the target; its time then only feeds the rules")
a = ap.parse_args()
assert jax.devices()[0].platform == "gpu", f"not on a GPU: {jax.devices()}"
X = np.load(f"{a.data}/X.npy").astype(np.float64); y = np.load(f"{a.data}/y.npy").astype(np.float64)
lam = a.lam_frac * lc.lam_max(X, y, a.alpha)
b = json.load(open(a.baseline_json))[a.baseline]
assert b["passed"] or a.allow_failing_baseline, "baseline did not reach the target"
assert abs(json.load(open(a.baseline_json))["lam"] - lam) <= 1e-12 * lam, "baseline must solve the same instance"
baseline = {"seconds": b["seconds"], "evals": b["n_iter"]}
bw = {(r["op"], r["dtype"]): r["GBps"] for r in json.load(open(a.bandwidth_json))["rows"]}
hw = {"step_GBps": bw[("grad", "float64")], "peak_GBps": a.peak_gbps,
      "f32_XTv_GBps": bw[("XTv", "float32")], "f64_XTv_GBps": bw[("XTv", "float64")]}
print(f"device {jax.devices()[0]}; n={X.shape[0]} p={X.shape[1]} lam={lam:.3e}; baseline {a.baseline} {b['seconds']:.3f}s "
      f"({b['n_iter']} iters, gap {b['gap']:.1e}); hw {hw}", flush=True)
records = []


_prob = [None]
_best_evals = [None]                 # best certified evaluation count so far, for the composed cut-off


def _composite():
    """The same instance as a composite problem, for configurations the composer runs."""
    if _prob[0] is None:
        _prob[0] = lc.composite(X, y, lam=lam, alpha=a.alpha)
    return _prob[0]


def _failed(cfg, err, note=None):
    if note is None and ("RESOURCE_EXHAUSTED" in err or "OUT_OF_MEMORY" in err):
        note = "resource failure, NOT a slow configuration: this candidate was never measured"
    records.append({"config": cfg, "error": err, **({"note": note} if note else {})})
    jax.clear_caches(); _EXE_CACHE.clear()
    print(f"  candidate {cfg} failed: {err[:150]}", flush=True)
    return {"seconds": float("inf"), "evals": 0, "converged": False, "error": err, "p": X.shape[1], "nnz": X.shape[1]}


def evaluate(cfg):
    """Two engines, one configuration space: operator genes name a composed recipe (run by the KB
    composer, whose gate may refuse it), everything else names the screened runner. `route` decides,
    and reports the genes the chosen engine cannot honour so no run claims a gene it ignored."""
    rt = route(cfg)
    if rt["engine"] == "composed":
        from atlas.evolve.kb_composer import run_recipe
        from atlas.typing_ import TypeCheckError
        rec = config_to_recipe(cfg)
        K = int(cfg.get("K", 64))
        # Judge competitiveness on iterations before paying for convergence: a recipe that cannot certify
        # within `composed_cap` times the best certified count is not going to win on time either.
        cap = min(200000, max(2000, a.composed_cap * (_best_evals[0] or 200)))
        try:
            run_recipe(_composite(), rec, tol=a.tol, max_evals=cap, K=K)                     # untimed warm-up
            rs = [run_recipe(_composite(), rec, tol=a.tol, max_evals=cap, K=K) for _ in range(a.reps)]
        except TypeCheckError as e:      # the gate refusing a composition is an answer, not a crash
            return _failed(cfg, f"gate rejected: {str(e)[:260]}", note="typecheck")
        except Exception as e:
            return _failed(cfg, f"{type(e).__name__}: {str(e)[:300]}")
        r = dict(rs[0])
        nnz = int((np.abs(np.asarray(r["x"])) > 0).sum()); r.pop("x"); r.pop("u", None)
        r.update(seconds=statistics.median(x["wall_s"] for x in rs), walls=[x["wall_s"] for x in rs],
                 p=X.shape[1], nnz=nnz, engine="composed", recipe=rec.name(), ignored_genes=rt["ignored"],
                 converged=all(x["converged"] for x in rs))
        if rt["ignored"]:
            print(f"  composed recipe {rec.name()} ignores {rt['ignored']}", flush=True)
        if not r["converged"]:
            print(f"  composed recipe {rec.name()} did not certify within {cap} evaluations "
                  f"({a.composed_cap}x the best certified count); not competitive on this cell", flush=True)
        records.append({"config": cfg, "eval_cap": cap, **r}); jax.clear_caches(); _EXE_CACHE.clear()
        return r
    kw = {k: (jnp.float32 if v == "float32" else v) if k == "dtype" else v for k, v in cfg.items()}
    try:
        run_screened(X, y, lam, a.alpha, tol=a.tol, **kw)                      # untimed warm-up
        rs = [run_screened(X, y, lam, a.alpha, tol=a.tol, **kw) for _ in range(a.reps)]
    except Exception as e:                                                   # a failed candidate is uncertified, not fatal
        return _failed(cfg, f"{type(e).__name__}: {str(e)[:300]}")
    r = dict(rs[0]); r.pop("x")
    r.update(seconds=statistics.median(x["wall_s"] for x in rs), walls=[x["wall_s"] for x in rs], p=X.shape[1],
             engine="screened", converged=all(x["converged"] for x in rs))
    records.append({"config": cfg, **r}); jax.clear_caches(); _EXE_CACHE.clear()
    return r


def evaluate_tracked(cfg):
    """evaluate(), remembering the best certified evaluation count so the composed cut-off can adapt."""
    r = evaluate(cfg)
    if r.get("converged") and r.get("evals"):
        _best_evals[0] = min(_best_evals[0] or r["evals"], int(r["evals"]))
    return r


# the step size is computed inside the timed region in every configuration (power iteration to start)
out = evolve(evaluate_tracked, {"K": 64, "screen": False, "step_rule": "power"}, baseline, hw, max_rounds=a.rounds,
             log=lambda m: print(m, flush=True))
print("CHAIN")
for c in out["chain"]:
    print(f"  round {c['round']}: {c['seconds']:.3f}s {c['evals']:4d} evals  "
          + (f"{c['symptom']} -> {c['delta']} ({c.get('why', '')})" if c["symptom"] else "start"), flush=True)
print(f"best {out['best']['config']} {out['best']['seconds']:.3f}s vs {a.baseline} {baseline['seconds']:.3f}s: "
      f"{'WIN' if out['beats_baseline'] else 'still behind'} ({baseline['seconds'] / out['best']['seconds']:.2f}x)")
# Two different facts, previously printed as one list: an ingredient with NO implementation, and a move
# that IS implemented, was run, and was abandoned as uncompetitive. Reporting them together would let a
# reader conclude the anchor was never tried, when in fact it was tried and lost.
_stubs = sorted({s["ingredient"] for s in out["not_runnable"]
                 if s.get("ingredient") and s.get("delta") is None})
_tried = sorted({s["ingredient"] or str(s.get("symptom")) for s in out["not_runnable"]
                 if s.get("delta") is not None})
print("no implementation in either engine:", _stubs)
print("implemented, tried, abandoned as uncompetitive:", _tried)
json.dump({"lam": lam, "baseline": {a.baseline: b}, "baseline_passed": bool(b["passed"]), "hw": hw, "chain": out["chain"], "beats_baseline": out["beats_baseline"],
           "not_runnable": _stubs, "tried_and_abandoned": _tried, "records": records},
          open(a.out, "w"), indent=1, default=str)
print("DIAGNOSE_LOOP_DONE")
