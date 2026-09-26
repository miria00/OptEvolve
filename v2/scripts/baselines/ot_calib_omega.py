"""Omega0-scaled GPU calibration (ratios r = kappa/omega0). Derived from the wide-grid on DOTmark (CauchyDensity 1009-1010, res 32).

The first grid (r in [0.03, 30]) put the best Euclidean ratio at its upper edge
and the best entropic ratio at its lower edge: censored. This widens both and
adds the genome's reflected Halpern arm with fixed-period anchor restarts
(rho = 2, halpern, restart_k), the restarted-Halpern-PDHG recipe expressible in
our genes. N is device-invariant, so the GPU is used for speed only.
"""
import json, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import jax, numpy as np
jax.config.update("jax_enable_x64", True)
import ot
from atlas.bench import dotmark
from atlas.genome import Backend
from atlas.schemes.base_schemes import Budget, PDHGParams, _wrap_cert, build_scheme, run_built

prob = dotmark.composite("CauchyDensity", 32, 1009, 1010, operator="reduction")
A, B, C = dotmark.pair("CauchyDensity", 32, 1009, 1010)
ref = float(ot.emd2(A, B, C, numItermax=10**8))
nbE = float(prob.L.norm_bound); nbB = 2 ** 0.5
W0 = float(np.linalg.norm(np.asarray(prob.data["c"])) / np.linalg.norm(np.asarray(prob.data["b"])))
print(f"omega0 = ||c||/||b|| = {W0:.4g}", flush=True)
# r = sqrt(t/s); PDLP's primal weight omega sets t/s = 1/omega^2, i.e. r = 1/omega0 (kappa = 1)
arms = ([("pdhg", k / W0, None, None) for k in (1e-2, 1e-1, 1, 10, 100)]
        + [("pdhg_entropic", r, None, None) for r in (1e-6, 1e-5)]
        + [("pdhg", k / W0, 2.0, rk) for k in (0.1, 1, 10) for rk in (256, 1024, 4096)])
rows = []
for scheme, r, rho, rk in arms:
    nb = nbE if scheme == "pdhg" else nbB
    t, s = 0.99 * r / nb, 0.99 / (r * nb)
    params = PDHGParams(t=t, s=s, rho=rho, halpern=rk is not None, restart_k=rk)
    built = build_scheme(prob, scheme, params)
    cert = _wrap_cert(built.cert, rho, rk is not None, rk)
    res = run_built(built, cert, params, Budget(max_iters=50000, tol=1e-4, sample_every=1000),
                    backend=Backend(precision="f64", K=64, device="gpu"))
    x, u = np.asarray(res.x), np.asarray(res.u)
    from atlas.certify.residuals import lp_rel_gap
    cv = float(lp_rel_gap(jax.numpy.asarray(x), jax.numpy.asarray(u), prob.data["c"], prob.L, prob.data["b"]))
    row = {"scheme": scheme, "ratio": r, "rho": rho, "restart_k": rk, "iters": int(res.iters),
           "t_exec_s": float(res.wall_time), "cert": cv, "passed": cv <= 1e-4 and res.iters < 50000,
           "obj_rel_err": abs(float(np.asarray(prob.data["c"]) @ x) - ref) / max(1.0, abs(ref))}
    rows.append(row)
    arm = scheme + ("" if rk is None else f"+halpern(rho=2,k={rk})")
    print(f"{arm:32s} r={r:<7g} iters {row['iters']:6d} exec {row['t_exec_s']:6.2f}s cert {cv:.2e} "
          f"obj_err {row['obj_rel_err']:.1e} {'PASS' if row['passed'] else 'FAIL'}", flush=True)
    json.dump({"emd_cost": ref, "rows": rows}, open("results/ot_dotmark_20260910/calib_gpu_omega_res32.json", "w"), indent=1)
