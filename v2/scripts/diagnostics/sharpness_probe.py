"""Empirical sharpness: predict iteration count from the gap decay we already log.

The linear rate of restarted PDHG is governed by a sharpness constant, not by
the condition number; the global Hoffman constant that bounds it is famously
conservative and expensive. But precision.py already records gap_history, the
float64 certified gap at every certificate evaluation. Fitting log(gap) against
iteration over the FIRST few checks gives an empirical local sharpness rate at
zero extra cost, available long before the solve finishes.

This probe asks whether that early rate predicts the final iteration count.
"""
import argparse, json, math
from pathlib import Path
import numpy as np, jax, jax.numpy as jnp
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from atlas.backend.hardware import prepare_base
from atlas.bench.lp_cell import build_instances
from atlas.bench.kermany import load_patches
from atlas.ir.spec import tv_denoise_problem
from atlas.genome import Genome, GenomeParams
import atlas.backend.precision as P
jax.config.update("jax_enable_x64", True)

ap = argparse.ArgumentParser()
ap.add_argument("--K", type=int, default=64)
ap.add_argument("--tol", type=float, default=1e-4)
ap.add_argument("--max-iters", type=int, default=20000)
ap.add_argument("--n-probe", type=int, default=3, help="certificate checks used for the fit")
ap.add_argument("--out", default="")
a = ap.parse_args()

GEN = Genome("condat_vu", GenomeParams(tau=.05, sigma=1., rho=1.9))

def solve_with_history(prob, gen):
    built, cert, params, _ = prepare_base(prob, gen)
    rho = float(gen.params.rho)
    dyn = jax.tree_util.tree_map(jnp.asarray, built.dyn)
    mj = jax.jit(built.metric)
    def step(state, k, d):
        if built.relaxed_T is not None:
            return built.relaxed_T(state, k, d, rho)
        Tx = built.T(state, k, d)
        return Tx if rho == 1.0 else jax.tree_util.tree_map(
            lambda x, y: (1.-rho)*x + rho*y, state, Tx)
    st0 = jax.tree_util.tree_map(lambda x: jnp.asarray(x, jnp.float64), built.state0)
    gaps, gi = [], []
    st, k, conv, _, _, _ = P._run_phase(step, st0, 0, a.K, a.max_iters,
                                        lambda s: mj(s, dyn), a.tol, gaps, gi,
                                        dyn=dyn, static_key=None)
    return np.asarray(gaps, float), np.asarray(gi, float), int(k), bool(conv)

def sharpness(gaps, iters, n_probe):
    """Least-squares slope of log(gap) vs iteration over the first n_probe checks.

    Returns tau, the iterations per e-fold. Smaller tau means a sharper problem
    and a faster solve.
    """
    m = min(n_probe, len(gaps))
    g, it = gaps[:m], iters[:m]
    ok = np.isfinite(g) & (g > 0)
    if ok.sum() < 2: return float('nan')
    sl = np.polyfit(it[ok], np.log(g[ok]), 1)[0]
    return float('inf') if sl >= 0 else -1.0 / sl

rows = []
# LP family: netflow grid and regular, several sizes/degrees
for fam, size, deg in [("grid",12,4),("grid",16,4),("grid",24,4),("grid",32,4),
                       ("regular",256,3),("regular",256,4),("regular",256,6),("regular",512,4)]:
    try:
        tr, _ = build_instances(2, 0, fam, size, seed=0, degree=deg)
    except Exception as e:
        print(f"  skip {fam}/{size}/d{deg}: {str(e)[:70]}"); continue
    for i, prob in enumerate(tr):
        try:
            g, it, N, conv = solve_with_history(prob, GEN)
        except Exception as e:
            print(f"  fail {fam}/{size}: {str(e)[:70]}"); continue
        tau = sharpness(g, it, a.n_probe)
        rows.append({"family": f"{fam}{size}d{deg}", "i": i, "N": N, "conv": conv,
                     "tau": tau, "n_checks": len(g),
                     "gaps": g.tolist(), "gap_iters": it.tolist()})
        print(f"{fam:8s} size={size:4d} d={deg} i={i}  tau={tau:10.1f}  N={N:6d} conv={conv}", flush=True)

# TV family for contrast
for sz in (128, 256):
    patches, _ = load_patches(2, size=sz, split="train", seed=0, return_meta=True)
    for i in range(2):
        prob = tv_denoise_problem(np.asarray(patches[i]), lam=0.1)
        g, it, N, conv = solve_with_history(prob, GEN)
        tau = sharpness(g, it, a.n_probe)
        rows.append({"family": f"tv{sz}", "i": i, "N": N, "conv": conv,
                     "tau": tau, "n_checks": len(g),
                     "gaps": g.tolist(), "gap_iters": it.tolist()})
        print(f"tv       size={sz:4d}      i={i}  tau={tau:10.1f}  N={N:6d} conv={conv}", flush=True)

good = [r for r in rows if math.isfinite(r["tau"]) and r["conv"]]
if len(good) >= 3:
    x = np.log([r["tau"] for r in good]); y = np.log([r["N"] for r in good])
    r = float(np.corrcoef(x, y)[0, 1])
    print(f"\n  correlation of log(early sharpness tau) with log(final N): "
          f"r = {r:.3f} over {len(good)} converged instances")
    print(f"  (tau uses only the first {a.n_probe} certificate checks, "
          f"i.e. {a.n_probe * a.K} iterations of a {int(np.median([g['N'] for g in good]))}-iteration median solve)")
if a.out:
    Path(a.out).write_text(json.dumps({"args": vars(a), "rows": rows}, indent=1))
    print("wrote", a.out)
