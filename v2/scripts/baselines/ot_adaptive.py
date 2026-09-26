"""DOTmark on the GPU: fixed vs PDLP-adaptive primal weight, judged against POT emd.

omega0 = ||c||/||b|| (PDLP's initial primal weight). Arms: fixed at omega0,
adaptive from omega0, fixed at omega0/30 (the ratio calibrated on
CauchyDensity 1009-1010). Plain PDHG, reduction operator, K = 64, fill 0.98.
Objective error is relative to the exact emd cost; the certificate alone is
not trusted for OT (its 1 + ||b|| normalisation is nearly absolute).
"""
import argparse, json, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, jax
jax.config.update("jax_enable_x64", True)
import ot
from atlas.backend.adaptive import run_adaptive
from atlas.bench import dotmark
from atlas.schemes.base_schemes import PDHGParams, build_scheme

ap = argparse.ArgumentParser()
ap.add_argument("--res", type=int, default=32); ap.add_argument("--classes", default=",".join(dotmark.CLASSES))
ap.add_argument("--pairs", default="1001-1002,1003-1004,1005-1006"); ap.add_argument("--extra", default="CauchyDensity:1009-1010")
ap.add_argument("--tol", type=float, default=1e-6); ap.add_argument("--cap", type=int, default=100000)
ap.add_argument("--out", required=True)
ap.add_argument("--arms", default="fixed:1,pdlp:1,fixed:30",
                help="rule:kappa[:max_updates], rule in fixed|pdlp; kappa scales omega0 down")
ap.add_argument("--epoch-chunks", type=int, default=8)
a = ap.parse_args()
jobs = [tuple(x.split(":")) for x in a.extra.split(",") if x] + [(c, p) for c in a.classes.split(",") for p in a.pairs.split(",") if p]
rows = []
for cls, pr in jobs:
    i1, i2 = map(int, pr.split("-"))
    prob = dotmark.composite(cls, a.res, i1, i2, operator="reduction")
    A, B, C = dotmark.pair(cls, a.res, i1, i2)
    t0 = time.perf_counter(); ref = float(ot.emd2(A, B, C, numItermax=10**9)); t_emd = time.perf_counter() - t0
    W0 = float(np.linalg.norm(np.asarray(prob.data["c"])) / np.linalg.norm(np.asarray(prob.data["b"])))
    nb = float(prob.L.norm_bound)
    rec = {"cls": cls, "pair": pr, "res": a.res, "emd_s": t_emd, "emd_cost": ref, "omega0": W0}
    for spec in a.arms.split(","):
        parts = spec.split(":"); rule, kappa = parts[0], float(parts[1])
        mu = int(parts[2]) if len(parts) > 2 else None
        name = f"{rule}_k{parts[1]}" + (f"_u{mu}" if mu is not None else "")
        w = W0 / kappa
        params = PDHGParams(t=0.99 / (nb * w), s=0.99 * w / nb)
        built = build_scheme(prob, "pdhg", params)
        r = run_adaptive(built, built.cert, scheme="pdhg", tol=a.tol, max_iters=a.cap, K=64, rule=rule, omega0=w,
                         epoch_chunks=a.epoch_chunks, max_updates=mu)
        oe = abs(float(np.asarray(prob.data["c"]) @ r["x"]) - ref) / abs(ref)
        rec[name] = {"iters": r["iters"], "wall_s": r["wall_s"], "converged": r["converged"], "metric": r["metric"],
                     "obj_rel_err": oe, "n_updates": r["n_updates"], "eta_spent": r["eta_spent"],
                     "omega_final_over_omega0": r["omega_history"][-1][1] / W0}
    rows.append(rec)
    f = lambda v: (f"{v['wall_s']:6.2f}s it={v['iters']:6d} err={v['obj_rel_err']:.0e}{'' if v['converged'] else ' CAP'}"
                   f" w/w0={v['omega_final_over_omega0']:.3g} upd={v['n_updates']}")
    print(f"{cls:16s} {pr} emd {t_emd:6.2f}s | " + " | ".join(f"{k} {f(v)}" for k, v in rec.items() if isinstance(v, dict)), flush=True)
    json.dump({"args": vars(a), "rows": rows}, open(a.out, "w"), indent=1)
print("wrote", a.out)
