"""Adaptive (PDLP primal-weight) vs fixed step ratio on Maros-Meszaros, same runner.

Condat-Vu, f64, CPU single thread, K = 64, tol 1e-4 on qp_rel_kkt, cap 5e4,
fill 0.99 (the head-to-head's setting). Arms: plain and Ruiz(10), each with the
controller off ("fixed") and on ("pdlp"); the variable metric budget is
declared and reported. Baselines were timed in results/qp_20260910; join on name.
"""
import argparse, json, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from pathlib import Path
from atlas.backend.adaptive import run_adaptive
from atlas.backend.hardware import prepare_base
from atlas.bench import qp_cell
from atlas.certify.residuals import qp_rel_kkt
from atlas.genome import Genome, GenomeParams, Metric
from atlas.schemes.rescale import rescale_qp_problem

ap = argparse.ArgumentParser()
ap.add_argument("--shard", default="0/1"); ap.add_argument("--out", required=True)
ap.add_argument("--arms", default="ruiz:10+fixed,ruiz:10+pdlp,none+pdlp")
ap.add_argument("--budget", type=float, default=1e4)
a = ap.parse_args()
DATA = Path("/home/miria/OptEvolve/v2/data/maros_meszaros")
man = sorted([e for e in json.load(open(DATA / "MANIFEST.json"))["instances"] if e["nnzA"] + e["nnzP"] <= 1e6],
             key=lambda e: e["nnzA"] + e["nnzP"])
i, N = map(int, a.shard.split("/")); pick = man[i::N]
rows = []
for e in pick:
    prob = qp_cell.load(str(DATA / e["file"])); rec = {"name": e["name"], "n": e["n"], "nnz": e["nnzA"] + e["nnzP"]}
    for arm in a.arms.split(","):
        met, rule = arm.split("+")
        metric = None if met == "none" else Metric(kind="ruiz", iters=int(met.split(":")[1]))
        run_on = prob if metric is None else rescale_qp_problem(prob, "ruiz", iters=metric.iters)[0]
        Lf, nb = float(run_on.f.props.lipschitz), float(run_on.L.norm_bound)
        t = 1.0 / Lf; s = 0.49 / (t * nb ** 2)
        try:
            built, cert, _, _ = prepare_base(prob, Genome("condat_vu", GenomeParams(tau=t, sigma=s), metric=metric))
            r = run_adaptive(built, cert, scheme="condat_vu", tol=1e-4, max_iters=50000, K=64, rule=rule,
                             eta_budget=a.budget, fill=0.99)
            kkt = float(qp_rel_kkt(jnp.asarray(r["x"]), jnp.asarray(r["u"]), prob.f.P, prob.f.q, prob.L, prob.h.lo, prob.h.hi))
            w = r["omega_history"]
            rec[arm] = {"iters": r["iters"], "wall_s": r["wall_s"], "compile_s": r["compile_s"], "cert": kkt,
                        "passed": kkt <= 1e-4 and r["iters"] < 50000, "n_updates": r["n_updates"],
                        "eta_spent": r["eta_spent"], "frozen": r["frozen"], "omega_ratio": w[-1][1] / w[0][1]}
        except Exception as ex:
            rec[arm] = {"passed": False, "error": f"{type(ex).__name__}: {str(ex)[:160]}"}
    rows.append(rec)
    print(e["name"], " | ".join(f"{k} {'PASS' if v.get('passed') else 'FAIL'} it={v.get('iters')} upd={v.get('n_updates')}"
                                for k, v in rec.items() if isinstance(v, dict)), flush=True)
    json.dump({"args": vars(a), "rows": rows}, open(a.out, "w"), indent=1)
print("wrote", a.out)
