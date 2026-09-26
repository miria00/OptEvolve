"""Quasi-Newton primal metric vs scalar Condat-Vu on Maros-Meszaros (same runner).

All arms run through run_built's policy path (f64, K = 64, tol 1e-4 on
qp_rel_kkt, cap 5e4, CPU single thread), so only the primal step differs.
  scalar      Condat-Vu with t = 1/L_f, s on the gate boundary (fill 0.99)
  krylov:m    L-BFGS metric from m Krylov directions of P with exact secant
              images (P + mu I) s, mu = 1e-3 L_f
  iterates:m  L-BFGS metric from the last m steps of a 512-iteration scalar
              warm-up (the qN-AP3 pattern of Bonnans et al. 1995), then a warm
              start; reported iterations include the warm-up
Every metric is scaled onto the gate boundary lambda_max(T P)/2 + s||A T A^T|| = 0.99.
"""
import argparse, json, sys, time
from dataclasses import replace
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from pathlib import Path
from atlas.bench import qp_cell
from atlas.certify.residuals import qp_rel_kkt
from atlas.genome import Backend
from atlas.operators.primal_metric import krylov_pairs, lbfgs_metric, scale_for_gate
from atlas.schemes.base_schemes import (Budget, CondatVuMetricParams, CondatVuParams, build_scheme, run_built)

ap = argparse.ArgumentParser()
ap.add_argument("--shard", default="0/1"); ap.add_argument("--out", required=True)
ap.add_argument("--arms", default="scalar,krylov:10,iterates:10")
ap.add_argument("--names", default="")
a = ap.parse_args()
DATA = Path("/home/miria/OptEvolve/v2/data/maros_meszaros")
man = sorted([e for e in json.load(open(DATA / "MANIFEST.json"))["instances"] if e["nnzA"] + e["nnzP"] <= 1e6],
             key=lambda e: e["nnzA"] + e["nnzP"])
if a.names:
    man = [e for e in man if e["name"] in set(a.names.split(","))]
i, N = map(int, a.shard.split("/")); pick = man[i::N]
BK, CAP, TOL, W = Backend(precision="f64", K=64), 50000, 1e-4, 512

def score(prob, r):
    return float(qp_rel_kkt(jnp.asarray(r.x), jnp.asarray(r.u), prob.f.P, prob.f.q, prob.L, prob.h.lo, prob.h.hi))

rows = []
for e in pick:
    prob = qp_cell.load(str(DATA / e["file"]))
    n = int(prob.shape[0]); m = int(prob.L.forward(jnp.zeros(prob.shape)).shape[0])
    Lf, nb = float(prob.f.props.lipschitz), float(prob.L.norm_bound)
    rec = {"name": e["name"], "n": n, "m": m, "nnz": e["nnzA"] + e["nnzP"]}
    p0 = CondatVuParams(t=1.0 / Lf, s=0.49 * Lf / nb ** 2)
    for arm in a.arms.split(","):
        try:
            t0 = time.perf_counter(); warm = 0
            if arm == "scalar":
                built, params = build_scheme(prob, "condat_vu", p0), p0
            else:
                kind, k = arm.split(":"); k = int(k)
                if kind == "krylov":
                    S, Y = krylov_pairs(prob.f.P.forward, n, k, mu=1e-3 * Lf)
                    start = None
                else:
                    b0 = build_scheme(prob, "condat_vu", p0)
                    step = jax.jit(lambda z: b0.T(z, 0, b0.dyn))
                    z, xs = b0.state0, []
                    for it in range(W):
                        z = step(z)
                        if it >= W - k - 1:
                            xs.append(np.asarray(z[0]))
                    S = np.diff(np.array(xs), axis=0)
                    Y = np.stack([np.asarray(prob.f.P.forward(jnp.asarray(s_))) + 1e-3 * Lf * s_ for s_ in S])
                    start, warm = z, W
                met, s, lamP, nAT = scale_for_gate(lbfgs_metric(S, Y), prob.f.P, prob.L, n, m)
                params = CondatVuMetricParams(s=s, metric=met)
                built = build_scheme(prob, "condat_vu_metric", params)
                if start is not None:
                    built = replace(built, state0=start)
            t_prep = time.perf_counter() - t0
            r = run_built(built, built.cert, params, Budget(max_iters=CAP, tol=TOL, sample_every=1000), backend=BK)
            kkt = score(prob, r)
            rec[arm] = {"iters": int(r.iters) + warm, "t_exec_s": float(r.wall_time), "t_prepare_s": t_prep,
                        "cert": kkt, "passed": kkt <= TOL and r.iters < CAP}
            if arm != "scalar":
                rec[arm].update({"lamP": lamP, "theta": s * nAT, "pairs_kept": int(met.S.shape[0])})
        except Exception as ex:
            rec[arm] = {"passed": False, "error": f"{type(ex).__name__}: {str(ex)[:160]}"}
    rows.append(rec)
    print(e["name"], " | ".join(f"{k} {'PASS' if v.get('passed') else 'FAIL'} it={v.get('iters')}"
                                for k, v in rec.items() if isinstance(v, dict)), flush=True)
    json.dump({"args": vars(a), "rows": rows}, open(a.out, "w"), indent=1)
print("wrote", a.out)
