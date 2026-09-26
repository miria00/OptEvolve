"""Sweep omega0 = ||c||/||b|| and ask whether the optimal step ratio tracks it.

c -> alpha*c is IDENTICALLY (tau, sigma) -> (alpha*tau, sigma/alpha): the gate
product tau*sigma*||L||^2 is invariant, so the admissible region does not move
at all, while the ratio r = tau/sigma moves by alpha^2. omega0 has been pinned
to 7.35-7.92 on every instance ever measured here, so every "structure does not
predict the recipe" negative was tested against a variable that never varied.

At fixed gate product theta, tau = sqrt(theta*r)/||L||, sigma = sqrt(theta/r)/||L||.
"""
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from atlas.backend.hardware import prepare_base
from atlas.bench.lp_cell import build_instances
from atlas.bench.ot import transport_composite
try:
    from atlas.bench.mcf import mcf_composite
except Exception:
    mcf_composite = None
from atlas.genome import Genome, GenomeParams
import atlas.backend.precision as P

ap = argparse.ArgumentParser()
ap.add_argument("--cells", default="grid:16,grid:24,regular:256:4,regular:512:4")
ap.add_argument("--alphas", default="0.01,0.03,0.1,0.3,1,3,10,30,100")
ap.add_argument("--ratios", default="0.01,0.03,0.1,0.3,1,3,10,30,100")
ap.add_argument("--theta", type=float, default=0.99)
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--K", type=int, default=64)
ap.add_argument("--tol", type=float, default=1e-4)
ap.add_argument("--max-iters", type=int, default=60000)
ap.add_argument("--out", required=True)
a = ap.parse_args()

def solve(prob, tau, sigma, rho=1.0):
    gen = Genome("pdhg", GenomeParams(tau=float(tau), sigma=float(sigma), rho=rho))
    built, cert, params, _ = prepare_base(prob, gen)          # raises if gate rejects
    dyn = jax.tree_util.tree_map(jnp.asarray, built.dyn)
    mj = jax.jit(built.metric)
    def step(st, k, d):
        Tx = built.T(st, k, d)
        return Tx if rho == 1.0 else jax.tree_util.tree_map(
            lambda x, y: (1.-rho)*x + rho*y, st, Tx)
    st0 = jax.tree_util.tree_map(lambda x: jnp.asarray(x, jnp.float64), built.state0)
    gaps, gi = [], []
    t0 = time.perf_counter()
    st, k, conv, _, _, e_s = P._run_phase(step, st0, 0, a.K, a.max_iters, lambda s: mj(s, dyn),
                                          a.tol, gaps, gi, dyn=dyn, static_key=None)
    return int(k), bool(conv), float(e_s), float(gaps[-1] if gaps else np.inf)

rows = []
for spec in a.cells.split(","):
    parts = spec.split(":")
    fam = parts[0]
    for seed in range(a.seeds):
        try:
            if fam == "mcf":
                if mcf_composite is None:
                    raise RuntimeError("mcf_composite unavailable")
                sz, K = int(parts[1]), int(parts[2])
                base = mcf_composite(topology="grid", size=sz, n_commodities=K, seed=seed)
            elif fam == "ot":
                m, nn = int(parts[1]), int(parts[2])
                base = transport_composite(m, nn, seed=seed)
            else:
                size = int(parts[1])
                deg = int(parts[2]) if len(parts) > 2 else 4
                tr, _ = build_instances(1, 0, fam, size, seed=seed, degree=deg)
                base = tr[0]
        except Exception as e:
            print(f"skip {spec} seed{seed}: {str(e)[:60]}", flush=True); continue
        c0 = np.asarray(base.data["c"], dtype=np.float64)
        b0 = np.asarray(base.data["b"], dtype=np.float64)
        Lnorm = float(base.L.norm_bound)
        for alpha in (float(v) for v in a.alphas.split(",")):
            import dataclasses
            cdata = dict(base.data); cdata["c"] = jnp.asarray(c0 * alpha)
            prob = dataclasses.replace(base, f=type(base.f)(c=jnp.asarray(c0 * alpha)),
                                       data=cdata)
            omega0 = float(np.linalg.norm(c0 * alpha) / max(np.linalg.norm(b0), 1e-300))
            for r in (float(v) for v in a.ratios.split(",")):
                tau = np.sqrt(a.theta * r) / Lnorm
                sig = np.sqrt(a.theta / r) / Lnorm
                try:
                    N, conv, e_s, gap = solve(prob, tau, sig)
                except Exception as e:
                    rows.append({"cell": spec, "seed": seed, "alpha": alpha,
                                 "omega0": omega0, "r": r, "error": str(e)[:80]}); continue
                rows.append({"cell": spec, "seed": seed, "alpha": alpha, "omega0": omega0,
                             "r": r, "tau": float(tau), "sigma": float(sig),
                             "Lnorm": Lnorm, "N": N, "conv": conv, "exec_s": e_s, "gap": gap})
                print(f"{spec} s{seed} a={alpha:<7g} w0={omega0:10.3f} r={r:<7g} "
                      f"N={N:7d} conv={conv}", flush=True)
        Path(a.out).write_text(json.dumps({"args": vars(a), "rows": rows}, indent=1))
Path(a.out).write_text(json.dumps({"args": vars(a), "rows": rows}, indent=1))
print("wrote", a.out)
