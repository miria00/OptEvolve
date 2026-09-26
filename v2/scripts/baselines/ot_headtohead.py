"""DOTmark optimal transport: our PDHG variants against POT's network simplex.

Data: real DOTmark 1.0 images (data/dotmark, MANIFEST.json). Instance = two
images of one class at one resolution, normalised to mass 1, squared Euclidean
grid cost (the DOTmark definition). Every solver's (plan, dual) is scored by
the same certificate, atlas.certify.residuals.lp_rel_gap, on the same
matrix-free composite; POT's potentials (f, g) enter as u = [-f; g].

Baseline: ot.emd (POT; network simplex of Bonneel et al. 2011), author code,
exact. Ours: Euclidean PDHG (gate t*s*||A||_2^2 < 1, ||A||_2 = sqrt(m+n)) and
entropic PDHG (gate t*s*||A||_{1->2}^2 < 1, ||A||_{1->2} = sqrt(2)), f64 on the
CPU, single thread, with t*s*norm^2 = 0.9801 and the primal/dual ratio r the
only free choice. Mode "calibrate" picks r per arm on ONE pair outside the test
set; mode "eval" freezes it and runs the test pairs.
"""
import argparse, json, math, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import ot
from atlas.bench import dotmark
from atlas.certify.residuals import lp_rel_gap
from atlas.genome import Backend
from atlas.schemes.base_schemes import Budget, PDHGParams, build_scheme, run_built

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=("calibrate", "eval"), required=True)
ap.add_argument("--res", type=int, default=32)
ap.add_argument("--calib", default="CauchyDensity:1009-1010")
ap.add_argument("--ratios", default="0.03,0.1,0.3,1,3,10,30")
ap.add_argument("--schemes", default="pdhg,pdhg_entropic")
ap.add_argument("--operator", default="incidence", choices=("incidence", "reduction"))
ap.add_argument("--device", default=None, choices=("cpu", "gpu"), help="our arms; POT is always CPU")
ap.add_argument("--classes", default=",".join(dotmark.CLASSES))
ap.add_argument("--pairs", default="1001-1002,1003-1004,1005-1006")
ap.add_argument("--ratio-pdhg", type=float); ap.add_argument("--ratio-entropic", type=float)
ap.add_argument("--kappa", type=float, default=None,
                help="eval: set the ratio per instance by our rule r = omega0 / kappa, omega0 = ||c||/||b||")
ap.add_argument("--kappas", default=None,
                help="calibrate: comma list of kappa values swept as r = omega0 / kappa")
ap.add_argument("--tol", type=float, default=1e-4)
ap.add_argument("--cap", type=int, default=20000)
ap.add_argument("--K", type=int, default=64)
ap.add_argument("--sinkhorn-eps", default="", help="comma list of entropic regularisations; each is run by "
                "POT's Sinkhorn and scored by the same certificate (the standard baseline once the exact "
                "network simplex is out of reach)")
ap.add_argument("--out", required=True)
a = ap.parse_args()
NORM = {"pdhg": None, "pdhg_entropic": math.sqrt(2.0)}

def steps(prob, scheme, r):
    nb = float(prob.L.norm_bound) if scheme == "pdhg" else NORM[scheme]
    return 0.99 * r / nb, 0.99 / (r * nb)

def omega0(prob):
    """The primal weight of the instance: ||c|| / ||b|| (the rule calibrated at resolution 32)."""
    return float(np.linalg.norm(np.asarray(prob.data["c"])) / np.linalg.norm(np.asarray(prob.data["b"])))


def ratio_for(prob, kappa):
    """This harness's r is the RECIPROCAL of the primal weight: steps() sets t = fill r / ||A|| and
    s = fill / (r ||A||), so s / t = 1 / r^2, while atlas.backend.adaptive.boundary_steps defines
    omega = sqrt(s / t). Our rule is omega = omega0 / kappa, hence r = kappa / omega0. Passing omega
    directly inverts the steps (t = 1e5, s = 1e-10 at resolution 64) and nothing converges."""
    return kappa / omega0(prob)


def ours(prob, scheme, r, ref_cost):
    t, s = steps(prob, scheme, r)
    params = PDHGParams(t=t, s=s)
    t0 = time.perf_counter(); built = build_scheme(prob, scheme, params); t_prep = time.perf_counter() - t0
    res = run_built(built, built.cert, params, Budget(max_iters=a.cap, tol=a.tol, sample_every=500),
                    backend=Backend(precision="f64", K=a.K, device=a.device))
    x, u = jnp.asarray(res.x), jnp.asarray(res.u)
    cert = float(lp_rel_gap(x, u, prob.data["c"], prob.L, prob.data["b"]))
    obj = float(np.asarray(prob.data["c"]) @ res.x)
    return {"scheme": scheme, "ratio": r, "t": t, "s": s, "iters": int(res.iters), "t_exec_s": float(res.wall_time),
            "t_prepare_s": t_prep, "cert": cert, "passed": cert <= a.tol and res.iters < a.cap,
            "obj": obj, "obj_rel_err": abs(obj - ref_cost) / max(1.0, abs(ref_cost))}

def sinkhorn(cls, i1, i2, prob, eps):
    """POT's entropic solver at one regularisation, scored by OUR certificate: the potentials give the dual
    point, so an entropic plan is judged on the true LP gap, not on its own regularised objective."""
    A, B, C = dotmark.pair(cls, a.res, i1, i2)
    scale = float(C.max())
    t0 = time.perf_counter()
    G, log = ot.sinkhorn(A, B, C / scale, eps, numItermax=20000, log=True, method="sinkhorn_log")
    t = time.perf_counter() - t0
    # sinkhorn_log returns the potentials already in log space: f = scale eps log_u, g = scale eps log_v.
    # Taking a further log of log_u gave NaN certificates on the first attempt.
    lu = np.asarray(log["log_u"] if "log_u" in log else np.log(np.maximum(log["u"], 1e-300))).ravel()
    lv = np.asarray(log["log_v"] if "log_v" in log else np.log(np.maximum(log["v"], 1e-300))).ravel()
    f, g = scale * eps * lu, scale * eps * lv
    u = jnp.asarray(np.concatenate([-f, g]))
    cert = float(lp_rel_gap(jnp.asarray(G.ravel()), u, prob.data["c"], prob.L, prob.data["b"]))
    obj = float(np.asarray(prob.data["c"]) @ G.ravel())
    return {"eps": eps, "t_s": t, "cost": obj, "cert": cert}


def emd(cls, i1, i2, prob):
    A, B, C = dotmark.pair(cls, a.res, i1, i2)
    t0 = time.perf_counter(); G, log = ot.emd(A, B, C, numItermax=10**8, log=True); t = time.perf_counter() - t0
    u = jnp.asarray(np.concatenate([-log["u"], log["v"]]))
    cert = float(lp_rel_gap(jnp.asarray(G.ravel()), u, prob.data["c"], prob.L, prob.data["b"]))
    return {"t_s": t, "cost": float(log["cost"]), "cert": cert, "warning": log.get("warning")}

out = {"args": vars(a), "rows": []}
if a.mode == "calibrate":
    cls, ids = a.calib.split(":"); i1, i2 = map(int, ids.split("-"))
    prob = dotmark.composite(cls, a.res, i1, i2, operator=a.operator); ref = emd(cls, i1, i2, prob)
    print(f"calibration pair {cls} {i1}-{i2} res {a.res}: emd {ref['t_s']:.3f}s cert {ref['cert']:.1e}", flush=True)
    for scheme in a.schemes.split(","):
        sweep = ([(k, ratio_for(prob, k)) for k in map(float, a.kappas.split(","))] if a.kappas
                 else [(None, r) for r in map(float, a.ratios.split(","))])
        for kap, r in sweep:
            row = ours(prob, scheme, r, ref["cost"]); row["kappa"] = kap; row["omega0"] = omega0(prob)
            row["omega"] = (omega0(prob) / kap) if kap else None
            out["rows"].append(row)
            print(f"  {scheme:14s} kappa={kap if kap else float('nan'):<6g} r={r:<10.4g} iters {row['iters']:6d} exec {row['t_exec_s']:8.2f}s cert {row['cert']:.1e} "
                  f"obj err {row['obj_rel_err']:.1e} {'PASS' if row['passed'] else 'FAIL'}", flush=True)
            json.dump(out, open(a.out, "w"), indent=1)
else:
    for cls in a.classes.split(","):
        for pr in a.pairs.split(","):
            i1, i2 = map(int, pr.split("-"))
            prob = dotmark.composite(cls, a.res, i1, i2, operator=a.operator); ref = emd(cls, i1, i2, prob)
            rec = {"class": cls, "ids": [i1, i2], "res": a.res, "emd": ref,
                   "sinkhorn": [sinkhorn(cls, i1, i2, prob, float(e)) for e in a.sinkhorn_eps.split(",") if e],
                   "pdhg": ours(prob, "pdhg", ratio_for(prob, a.kappa) if a.kappa else a.ratio_pdhg, ref["cost"]),
                   "pdhg_entropic": ours(prob, "pdhg_entropic", ratio_for(prob, a.kappa) if a.kappa else a.ratio_entropic,
                                         ref["cost"])}
            out["rows"].append(rec)
            f = lambda v: f"{v['t_exec_s']:7.2f}s it={v['iters']:5d}" if v["passed"] else f"   FAIL it={v['iters']:5d} cert={v['cert']:.0e}"
            print(f"{cls:16s} {i1}-{i2} emd {ref['t_s']:6.3f}s | pdhg {f(rec['pdhg'])} | entropic {f(rec['pdhg_entropic'])}", flush=True)
            json.dump(out, open(a.out, "w"), indent=1)
print("wrote", a.out)
