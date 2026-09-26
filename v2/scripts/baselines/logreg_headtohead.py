"""Elastic-net logistic regression: our certified solvers against author code.

Every solver's w is scored by ONE certificate, the relative duality gap
(atlas.certify.residuals.logreg_enet_rel_gap), computed in float64 on the
full data. A baseline failing the target gap is rerun with its tolerance
tightened tenfold, up to four times, and timed at the setting that passes.
Baselines run as subprocesses in their own environments (enetenv: skglm,
scikit-learn, adelie; cumlenv: cuML on the GPU).
"""
import argparse, json, os, subprocess, sys, tempfile, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from atlas.bench import logreg_cell as lc
from atlas.certify.residuals import logreg_enet_rel_gap
from atlas.genome import Backend
from atlas.schemes.base_schemes import Budget, FBSParams, FISTAParams, build_scheme, run_built

HOME = os.path.expanduser("~")
_bw = os.path.exists(os.path.join(HOME, "bw_cumlenv"))     # the Blackwell box names its envs bw_*
_enet = os.path.join(HOME, "bw_enetenv" if _bw else "enetenv", "bin", "python")
_cuml = os.path.join(HOME, "bw_cumlenv" if _bw else "cumlenv", "bin", "python")
ENVS = {"skglm": _enet, "sklearn_saga": _enet, "adelie": _enet, "cuml": _cuml}
WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logreg_worker.py")
ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True); ap.add_argument("--alpha", type=float, default=0.5)
ap.add_argument("--lam-ratio", type=float, default=0.1); ap.add_argument("--target", type=float, default=1e-6)
ap.add_argument("--baselines", default="skglm,sklearn_saga,cuml"); ap.add_argument("--ours-device", default=None)
ap.add_argument("--ours-cap", type=int, default=200000); ap.add_argument("--out", required=True)
ap.add_argument("--ours", default="fbs,fista", help="our schemes to run")
ap.add_argument("--baseline-precision", default="float32", choices=("float32", "float64"), help="cuML fit precision")
ap.add_argument("--tol-start", type=float, default=1e-4, help="loosest baseline tolerance tried; tightened 10x until our certificate passes")
ap.add_argument("--ours-precision", default="f64", help="f64 | f32_cert64 (float32 iterates, float64 certificate)")
a = ap.parse_args()
X = np.load(a.data + "/X.npy").astype(np.float64); y = np.load(a.data + "/y.npy").astype(np.float64)
lam = a.lam_ratio * lc.lam_max(X, y, a.alpha)
prob = lc.composite(X, y, lam=lam, alpha=a.alpha)
# X and y are arguments, not closures: a closed-over X is compiled in as a constant (6.4 GB on the
# epsilon training set, a 40 GB process and a very long CPU compile)
from types import SimpleNamespace
_gap = jax.jit(lambda w, X_, y_: logreg_enet_rel_gap(w, SimpleNamespace(X=X_, y=y_), prob.g))
gap = lambda w: _gap(w, prob.f.X, prob.f.y)
rec = {"data": a.data, "n": X.shape[0], "p": X.shape[1], "alpha": a.alpha, "lam": lam, "target": a.target}
print(f"n={X.shape[0]} p={X.shape[1]} alpha={a.alpha} lam={lam:.3e} (lam_max x {a.lam_ratio}) target gap {a.target:g}", flush=True)
# ours: catalog schemes, certified by the duality gap
be = (Backend(precision=a.ours_precision, K=64, device=a.ours_device) if a.ours_device
      else Backend(precision=a.ours_precision, K=64))
step = 1.0 / prob.f.props.lipschitz
for scheme in [x for x in a.ours.split(",") if x]:
    params = FBSParams(step=step) if scheme == "fbs" else FISTAParams(step=step)
    built = build_scheme(prob, scheme, params)
    r = run_built(built, built.cert, params, Budget(max_iters=a.ours_cap, tol=a.target, sample_every=1000), backend=be)
    g = float(gap(jnp.asarray(r.x)))
    rec["ours_" + scheme + ("" if a.ours_precision == "f64" else "_" + a.ours_precision)] = {"precision": a.ours_precision, "seconds": float(r.wall_time), "iters": int(r.iters), "gap": g, "passed": g <= a.target,
                             "nnz": int((np.abs(r.x) > 0).sum()), "device": a.ours_device or "default"}
    print(f"  ours {scheme:6s} {a.ours_precision:10s} {r.wall_time:8.3f}s iters {r.iters:7d} gap {g:.2e} nnz {int((np.abs(r.x) > 0).sum())}", flush=True)
for b in [s for s in a.baselines.split(",") if s]:
    tries = []
    for k in range(int(round(np.log10(a.tol_start / 1e-8))) + 1):
        tol = a.tol_start / 10 ** k
        with tempfile.TemporaryDirectory() as td:
            wpath = os.path.join(td, "w.npy")
            cmd = [ENVS[b], WORKER, "--solver", b, "--data", a.data, "--lam", repr(lam), "--alpha", repr(a.alpha),
                   "--tol", repr(tol), "--max-iter", "100000", "--w-out", wpath] + (["--precision", a.baseline_precision] if b == "cuml" else [])
            pr = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, "OMP_NUM_THREADS": "1"})
            line = [l for l in pr.stdout.splitlines() if l.startswith("{")]
            if pr.returncode or not line:
                tries.append({"tol": tol, "error": (pr.stderr or pr.stdout)[-300:]}); break
            info = json.loads(line[-1]); w = np.load(wpath)
        gb = float(gap(jnp.asarray(w)))
        tries.append({"tol": tol, "seconds": info["seconds"], "n_iter": info["n_iter"], "gap": gb, "nnz": int((np.abs(w) > 0).sum())})
        if gb <= a.target:
            break
    best = tries[-1]
    rec[b] = {"passed": "gap" in best and best["gap"] <= a.target, **best, "tries": tries,
              "precision": a.baseline_precision if b == "cuml" else "float64"}
    print(f"  {b:13s} " + (f"{best['seconds']:8.3f}s iters {best['n_iter']:7d} gap {best['gap']:.2e} nnz {best['nnz']} (tol {best['tol']:g})"
                           if "gap" in best else f"ERROR {best.get('error', '')[-160:]}"), flush=True)
json.dump(rec, open(a.out, "w"), indent=1)
