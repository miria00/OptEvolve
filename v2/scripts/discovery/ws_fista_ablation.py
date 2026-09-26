"""The candidate new algorithm, isolated: working-set FISTA whose step comes from the Lipschitz constant
of the CURRENT working set, with Nesterov momentum CARRIED ACROSS working-set changes.

The literature sweep put exactly these two ingredients in the unoccupied gap. The screening/working-set
lane is almost entirely coordinate descent, where steps are per-feature (1/||x_j||^2, precomputed on the
full matrix) so a reduced-set Lipschitz constant is vacuous; the two places that do run a full-gradient
method on a shrinking set decline to exploit it (AS-ISTA normalises a global ||Phi||=1; dynamic screening
sets L by backtracking, which is monotone non-increasing and structurally cannot capitalise on a smaller
||D_W||). And every working-set method there RESTARTS its accelerator when the set changes: skglm re-inits
Anderson ("# re init AA at every iter to consider ws"), celer's dual extrapolation is gated per inner
solve, AS-ISTA re-initialises its inner sequence. A FISTA momentum vector lives in the full coordinate
space and is simply zero outside W, so carrying it is well defined here in a way it is not for them.

2x2 factorial over momentum {reset, keep} x step {full-matrix L, reduced-set L}, plus plain FISTA as the
floor. EVERY repetition is kept so arms are compared as paired differences with an interval, and both the
warm (solve loop) and cold (incl. compile) numbers are recorded, per the timing contract.
"""
import argparse, json, os, statistics, sys, time
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from atlas.bench import logreg_cell as lc
from atlas.bench.paired import paired_interval, paired_ratio
from atlas.evolve.screened_runner import _EXE_CACHE, run_screened

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/enet_libsvm/epsilon_test")
ap.add_argument("--alpha", type=float, default=0.5)
ap.add_argument("--lam-frac", type=float, default=0.1)
ap.add_argument("--tol", type=float, default=1e-6)
ap.add_argument("--reps", type=int, default=7)
ap.add_argument("--ws-init", type=float, default=0.0625)
ap.add_argument("--K", type=int, default=32)
ap.add_argument("--baseline-json", default=None, help="a logreg_headtohead output holding cuML for this instance")
ap.add_argument("--dtype", default="float64", choices=("float64", "float32"),
                help="ITERATE storage precision. NOTE this does NOT shrink the design matrix: run_screened "
                     "holds X as float64 on device regardless (Xf = jnp.asarray(X, dtype=jnp.float64)), so "
                     "float32 will not make a matrix fit that did not fit before. It exists so the arms can "
                     "be matched against cuML's float32, which is its winning precision on most instances.")
ap.add_argument("--allow-cpu", action="store_true")
ap.add_argument("--out", required=True)
a = ap.parse_args()
assert a.allow_cpu or jax.devices()[0].platform == "gpu", f"not on a GPU: {jax.devices()}"

X = np.load(f"{a.data}/X.npy").astype(np.float64)
y = np.load(f"{a.data}/y.npy").astype(np.float64)
lam = a.lam_frac * lc.lam_max(X, y, a.alpha)
L_full = float(lc.lipschitz_bound(X))
base = dict(K=a.K, working_set=True, buckets=True, screen=True, ws_init=a.ws_init)
_dt = {"float64": jnp.float64, "float32": jnp.float32}[a.dtype]
_p = {} if a.dtype == "float64" else {"dtype": _dt}       # every arm shares the precision, including the floor
ARMS = {
    "plain_fista":            dict(K=a.K, screen=False, step_rule="power", **_p),
    "ws_fullL_reset":         dict(base, momentum="reset", step_rule="precomputed", L=L_full, **_p),
    "ws_fullL_keep":          dict(base, momentum="keep",  step_rule="precomputed", L=L_full, **_p),
    "ws_reducedL_reset":      dict(base, momentum="reset", step_rule="gram", **_p),
    "ws_reducedL_keep":       dict(base, momentum="keep",  step_rule="gram", **_p),
}
print(f"device {jax.devices()[0]}; n={X.shape[0]} p={X.shape[1]} alpha={a.alpha} lam={lam:.4e} "
      f"({a.lam_frac} lam_max); L_full={L_full:.5f}; tol={a.tol:g}; reps={a.reps}", flush=True)

out = {"data": a.data, "dtype": a.dtype,
       "n": int(X.shape[0]), "p": int(X.shape[1]), "alpha": a.alpha, "lam_frac": a.lam_frac,
       "lam": lam, "L_full": L_full, "tol": a.tol, "reps": a.reps, "K": a.K, "ws_init": a.ws_init, "arms": {}}
# PHASE 1, arm-major by necessity: the first solve is about an EMPTY cache, so each arm needs the cache
# cleared immediately before it. (reset and keep share a signature, momentum not being part of the HLO.)
first_total, first_compile, failed = {}, {}, {}
for name, cfg in ARMS.items():
    _EXE_CACHE.clear()
    try:
        _c0 = time.perf_counter()
        _f = run_screened(X, y, lam, a.alpha, tol=a.tol, **cfg)
        first_total[name] = time.perf_counter() - _c0
        first_compile[name] = float(_f["compile_s"])
    except Exception as e:
        failed[name] = f"{type(e).__name__}: {str(e)[:300]}"
        out["arms"][name] = {"error": failed[name]}
        print(f"{name:20s} FAILED {str(e)[:120]}", flush=True)
    jax.clear_caches()

# PHASE 2, REP-MAJOR: repetition k of every arm runs adjacent in time. Until 2026-09-13 all repetitions of
# one arm ran before the next arm started, so pairing repetition k of two arms shared nothing but an index
# and controlled no common nuisance factor: an unpaired comparison wearing a paired label. Interleaving is
# what earns the paired interval, by putting the arms under the same drift and machine load.
reps_walls = {n_: [] for n_ in ARMS}; reps_cold = {n_: [] for n_ in ARMS}; last = {}
for _k in range(a.reps):
    for name, cfg in ARMS.items():
        if name in failed:
            continue
        try:
            r = run_screened(X, y, lam, a.alpha, tol=a.tol, **cfg)
        except Exception as e:
            failed[name] = f"{type(e).__name__}: {str(e)[:300]}"
            out["arms"][name] = {"error": failed[name]}; continue
        reps_walls[name].append(float(r["wall_s"])); reps_cold[name].append(float(r["wall_incl_compile_s"]))
        last[name] = r

for name, cfg in ARMS.items():
    if name in failed or not reps_walls[name]:
        continue
    rs = [last[name]]
    ok = last[name]["converged"]
    walls = reps_walls[name]
    cold = reps_cold[name]
    out["arms"][name] = {"config": {k: (str(v) if k == "dtype" else v) for k, v in cfg.items()},
                         "converged": ok, "walls": walls, "walls_incl_compile": cold,
                         "median_wall_s": statistics.median(walls),
                         "median_wall_incl_compile_s": statistics.median(cold),
                         "compile_s": float(rs[0]["compile_s"]),
                         "first_solve_total_s": float(first_total.get(name, float("nan"))),
                         "first_solve_compile_s": float(first_compile.get(name, float("nan"))),
                         "interleaved": True, "evals": int(rs[0]["evals"]),
                         "gap": float(rs[0]["gap"]), "nnz": int(rs[0]["nnz"]),
                         "L_values": [float(v) for v in rs[0]["L_values"]],
                         "replans": rs[0].get("replans"), "active_sizes": rs[0].get("active_sizes")}
    print(f"{name:20s} {statistics.median(walls):8.4f}s warm  {statistics.median(cold):8.4f}s repeat  "
          f"{first_total.get(name, float('nan')):8.4f}s first  "
          f"{rs[0]['evals']:5d} evals  gap {rs[0]['gap']:.1e}  L {[round(v,5) for v in rs[0]['L_values']][:3]}"
          f"{'' if ok else '  NOT CERTIFIED'}", flush=True)
jax.clear_caches()

# ---- the two contrasts the novelty argument rests on, as PAIRED differences ----
out["contrasts"] = {}
def contrast(label, fast, slow):
    A, B = out["arms"].get(fast, {}), out["arms"].get(slow, {})
    if not (A.get("walls") and B.get("walls") and A.get("converged") and B.get("converged")):
        return
    r = paired_ratio(A["walls"], B["walls"])
    d = paired_interval(A["walls"], B["walls"])
    out["contrasts"][label] = {"faster": fast, "slower": slow, **r, "mean_diff_s": d["mean_diff"],
                               "evals_fast": A["evals"], "evals_slow": B["evals"]}
    print(f"  {label:46s} {r['geo_ratio']:6.3f}x  [{r['lo']:.3f}, {r['hi']:.3f}]  {r['verdict']}"
          f"   evals {B['evals']} -> {A['evals']}", flush=True)

print("\nPAIRED CONTRASTS (geometric ratio, 95% interval on the paired log differences):", flush=True)
contrast("reduced-set L vs full L (momentum reset)", "ws_reducedL_reset", "ws_fullL_reset")
contrast("reduced-set L vs full L (momentum kept)", "ws_reducedL_keep", "ws_fullL_keep")
contrast("momentum kept vs reset (full L)", "ws_fullL_keep", "ws_fullL_reset")
contrast("momentum kept vs reset (reduced-set L)", "ws_reducedL_keep", "ws_reducedL_reset")
contrast("both ingredients vs neither", "ws_reducedL_keep", "ws_fullL_reset")
contrast("both ingredients vs plain FISTA", "ws_reducedL_keep", "plain_fista")

# Write the record BEFORE choosing a best arm: when every arm fails (e.g. the GPU cannot hold X), the
# old code raised ValueError on min() over an empty generator and died WITHOUT writing anything, which is
# how six epsilon_train failures left no diagnostic record at all.
json.dump(out, open(a.out, "w"), indent=1)
if not any(v.get("converged") for v in out["arms"].values()):
    print("no arm certified on this instance; errors recorded in " + a.out, flush=True)
    for k, v in out["arms"].items():
        if v.get("error"):
            print(f"  {k:20s} {v['error'][:150]}", flush=True)
    print("WS_FISTA_ABLATION_DONE", flush=True)
    raise SystemExit(0)

if a.baseline_json and os.path.exists(a.baseline_json):
    b = json.load(open(a.baseline_json))
    cu = b.get("cuml") or {}
    if cu.get("seconds"):
        out["cuml"] = {"seconds": cu["seconds"], "passed": bool(cu.get("passed")), "n_iter": cu.get("n_iter"),
                       "lam_matches": abs(b.get("lam", lam) - lam) <= 1e-9 * max(lam, 1e-30)}
        best = min((k for k, v in out["arms"].items() if v.get("converged")),
                   key=lambda k: out["arms"][k]["median_wall_s"])
        out["cuml"]["vs_best_warm"] = cu["seconds"] / out["arms"][best]["median_wall_s"]
        out["cuml"]["vs_best_cold"] = cu["seconds"] / out["arms"][best]["median_wall_incl_compile_s"]
        print(f"\ncuML {cu['seconds']:.4f}s (passed {cu.get('passed')}) vs best arm {best}: "
              f"{out['cuml']['vs_best_warm']:.2f}x warm, {out['cuml']['vs_best_cold']:.2f}x cold", flush=True)

json.dump(out, open(a.out, "w"), indent=1)
print("WS_FISTA_ABLATION_DONE", flush=True)
