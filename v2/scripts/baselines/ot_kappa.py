"""DOTmark: plain PDHG with r = kappa/omega0 on the GPU against POT emd (CPU, exact).

omega0 = ||c||/||b|| (PDLP's initial primal weight); r = sqrt(t/s), t*s*||A||^2 = 0.9801.
Reports iterations, GPU exec time, the shared certificate, and the objective error
against the exact network-simplex cost, at each tolerance.
"""
import argparse, json, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import jax, numpy as np
jax.config.update("jax_enable_x64", True)
import ot
from atlas.bench import dotmark
from atlas.certify.residuals import lp_rel_gap
from atlas.genome import Backend
from atlas.schemes.base_schemes import Budget, PDHGParams, build_scheme, run_built

ap = argparse.ArgumentParser()
ap.add_argument("--cls", default="CauchyDensity"); ap.add_argument("--res", type=int, default=32)
ap.add_argument("--pair", default="1009-1010"); ap.add_argument("--kappas", default="30,100,300,1000,3000,10000")
ap.add_argument("--tols", default="1e-4,1e-6"); ap.add_argument("--cap", type=int, default=50000)
ap.add_argument("--out", required=True)
a = ap.parse_args()
i1, i2 = map(int, a.pair.split("-"))
prob = dotmark.composite(a.cls, a.res, i1, i2, operator="reduction")
A, B, C = dotmark.pair(a.cls, a.res, i1, i2)
t0 = time.perf_counter(); ref = float(ot.emd2(A, B, C, numItermax=10**9)); t_emd = time.perf_counter() - t0
W0 = float(np.linalg.norm(np.asarray(prob.data["c"])) / np.linalg.norm(np.asarray(prob.data["b"])))
nb = float(prob.L.norm_bound)
print(f"{a.cls} {a.pair} res {a.res}: POT emd {t_emd:.3f}s cost {ref:.6g}; omega0 {W0:.4g}", flush=True)
rows = []
for tol in map(float, a.tols.split(",")):
    for k in map(float, a.kappas.split(",")):
        r = k / W0; t, s = 0.99 * r / nb, 0.99 / (r * nb)
        params = PDHGParams(t=t, s=s)
        built = build_scheme(prob, "pdhg", params)
        res = run_built(built, built.cert, params, Budget(max_iters=a.cap, tol=tol, sample_every=1000),
                        backend=Backend(precision="f64", K=64, device="gpu"))
        cv = float(lp_rel_gap(jax.numpy.asarray(res.x), jax.numpy.asarray(res.u), prob.data["c"], prob.L, prob.data["b"]))
        oe = abs(float(np.asarray(prob.data["c"]) @ np.asarray(res.x)) - ref) / abs(ref)
        row = {"cls": a.cls, "pair": a.pair, "res": a.res, "tol": tol, "kappa": k, "iters": int(res.iters),
               "t_exec_s": float(res.wall_time), "cert": cv, "passed": cv <= tol and res.iters < a.cap,
               "obj_rel_err": oe, "emd_s": t_emd}
        rows.append(row)
        print(f"  tol {tol:.0e} kappa {k:<7g} iters {row['iters']:6d} gpu {row['t_exec_s']:7.2f}s cert {cv:.2e} "
              f"obj err {oe:.1e} {'PASS' if row['passed'] else 'FAIL'}  (emd {t_emd:.2f}s)", flush=True)
        json.dump({"emd_cost": ref, "emd_s": t_emd, "omega0": W0, "rows": rows}, open(a.out, "w"), indent=1)
