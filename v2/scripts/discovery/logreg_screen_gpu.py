"""Epsilon on one GPU: plain FISTA against the diagnosed chain FISTA + Gap Safe screening with
mid-solve re-planning (momentum kept, bucketed shapes), at matched full-problem relative
duality gap. Median of --reps timed solves after an untimed warm-up solve per variant; compile
time is reported separately, as the baselines' warm-up solves are untimed."""
import argparse, json, statistics, sys, time
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
sys.path.insert(0, ".")
from atlas.bench import logreg_cell as lc
from atlas.evolve.screened_runner import run_screened

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/enet_libsvm/epsilon_test")
ap.add_argument("--alpha", type=float, default=0.5)
ap.add_argument("--lam-frac", type=float, default=0.1)
ap.add_argument("--tol", type=float, default=1e-6)
ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--out", default=None)
a = ap.parse_args()
X = np.load(f"{a.data}/X.npy").astype(np.float64); y = np.load(f"{a.data}/y.npy").astype(np.float64)
lam = a.lam_frac * lc.lam_max(X, y, a.alpha)
assert jax.devices()[0].platform == "gpu", f"not on a GPU: {jax.devices()}"
print(f"device {jax.devices()[0]}; n={X.shape[0]} p={X.shape[1]} alpha={a.alpha} lam={lam:.3e} target rel gap {a.tol}", flush=True)
L_host = lc.lipschitz_bound(X)          # once, untimed: used ONLY by the labelled "L precomputed" reference row
VARIANTS = [
    # reference: step from a host Lipschitz bound computed outside the timed region (as in all earlier rows)
    ("fista K=64, L precomputed (excluded)", dict(K=64, screen=False, L=L_host)),
    # honest rows: the step size is computed inside the timed region
    ("fista K=64, power L", dict(K=64, screen=False, step_rule="power")),
    ("fista K=16, power L", dict(K=16, screen=False, step_rule="power")),
    ("screen keep+buckets K=16, power L", dict(K=16, momentum="keep", buckets=True, step_rule="power")),
    ("working set p/16 keep K=16, power L", dict(K=16, momentum="keep", working_set=True, ws_init=0.0625, step_rule="power")),
    ("working set p/16 keep K=16, gram L", dict(K=16, momentum="keep", working_set=True, ws_init=0.0625, step_rule="gram")),
    ("working set p/8 keep K=16, gram L", dict(K=16, momentum="keep", working_set=True, step_rule="gram")),
    ("working set p/16 keep K=32, gram L", dict(K=32, momentum="keep", working_set=True, ws_init=0.0625, step_rule="gram")),
    ("working set p/16 keep K=8, gram L", dict(K=8, momentum="keep", working_set=True, ws_init=0.0625, step_rule="gram")),
    ("working set p/16 keep K=16, gram L, f32", dict(K=16, momentum="keep", working_set=True, ws_init=0.0625, step_rule="gram",
                                                     dtype=jnp.float32)),
]
rows = []


def dump():
    if a.out:
        json.dump({"n": X.shape[0], "p": X.shape[1], "lam": lam, "alpha": a.alpha, "tol": a.tol, "rows": rows},
                  open(a.out, "w"), indent=1)


for name, kw in VARIANTS:
    try:
        run_screened(X, y, lam, a.alpha, tol=a.tol, **kw)                      # untimed warm-up
        rs = [run_screened(X, y, lam, a.alpha, tol=a.tol, **kw) for _ in range(a.reps)]
    except Exception as e:                                                   # one failed variant must not lose the rest
        rows.append({"variant": name, "error": f"{type(e).__name__}: {str(e)[:300]}"})
        print(f"  {name:38s} ERROR {rows[-1]['error'][:150]}", flush=True); dump(); jax.clear_caches(); continue
    r = rs[0]; med = statistics.median(x["wall_s"] for x in rs)
    rows.append({"variant": name, "median_wall_s": med, "walls": [x["wall_s"] for x in rs], "evals": r["evals"],
                 "gap": r["gap"], "converged": all(x["converged"] for x in rs), "nnz": r["nnz"], "compile_s": r["compile_s"],
                 "replan_at": r["replan_at"], "active_sizes": r["active_sizes"], "L_values": r["L_values"], "step_rule": r["step_rule"]})
    print(f"  {name:38s} {med:7.3f}s (runs {', '.join(f'{x:.3f}' for x in rows[-1]['walls'])}) evals {r['evals']:4d} gap {r['gap']:.1e} "
          f"nnz {r['nnz']} compile {r['compile_s']:.2f}s at {r['replan_at']} sizes {r['active_sizes']} L {[round(v, 5) for v in r['L_values']]}", flush=True)
    dump(); jax.clear_caches()
print("SCREEN_GPU_DONE")
