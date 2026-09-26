"""Audit of QP passes independent of the certificate's internals: rerun an arm on every
instance it passed, keep (x, u), and measure absolute bound violation, iterate size, and the
objective against the baselines' objectives (OSQP/SCS/Clarabel at matched accuracy)."""
import argparse, glob, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from atlas.bench import qp_cell
from atlas.certify.residuals import qp_rel_kkt
from atlas.evolve.kb_composer import Recipe, run_recipe

ap = argparse.ArgumentParser(); ap.add_argument("--shard", default="0/1"); ap.add_argument("--out", required=True)
a = ap.parse_args()
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
comp = {r["name"]: r for f in glob.glob(os.path.join(ROOT, "results/qp_composed_20260911/shard*.json")) for r in json.load(open(f))["rows"]}
base = {r["name"]: r for f in glob.glob(os.path.join(ROOT, "results/qp_20260910/full_shard*.json")) for r in json.load(open(f))["rows"]}
ARMS = {"cv": Recipe("condat_vu", 0.98), "cv+aa5": Recipe("condat_vu", 0.98, anderson=5)}
todo = sorted((n, k) for n, r in comp.items() for k in ARMS if r.get(k, {}).get("passed"))
i, N = map(int, a.shard.split("/"))
rows = []
for name, arm in todo[i::N]:
    d = qp_cell.read_mm(os.path.join(ROOT, "data/maros_meszaros", name + ".mat")); prob = qp_cell.to_composite(d)
    out = run_recipe(prob, ARMS[arm], tol=1e-4, max_evals=50000, K=64)
    x, u = out["x"], out["u"]; Ax = d["A"] @ x
    viol = float(np.max(np.maximum(d["l"] - Ax, 0) + np.maximum(Ax - d["u"], 0)))
    fin = np.concatenate([np.abs(d["l"][np.isfinite(d["l"])]), np.abs(d["u"][np.isfinite(d["u"])]), [0.0]])
    obj = float(0.5 * x @ (d["P"] @ x) + d["q"] @ x + d["r"])
    refs = [base[name][k]["obj"] for k in ("clarabel", "osqp", "scs") if base.get(name, {}).get(k, {}).get("passed")]
    ref = refs[0] if refs else None
    rows.append({"name": name, "arm": arm, "evals": out["evals"], "kkt": float(qp_rel_kkt(jnp.asarray(x), jnp.asarray(u), prob.f.P, prob.f.q, prob.L, prob.h.lo, prob.h.hi)),
                 "abs_violation": viol, "data_bound_scale": float(np.max(fin)), "x_inf": float(np.max(np.abs(x))),
                 "obj": obj, "ref_obj": ref, "obj_rel_diff": None if ref is None else abs(obj - ref) / max(1.0, abs(ref))})
    jax.clear_caches()
    r = rows[-1]
    print(f"{name:12s} {arm:7s} viol {r['abs_violation']:.1e} |x| {r['x_inf']:.1e} obj {obj:.6g} ref {ref} diff {r['obj_rel_diff']}", flush=True)
    json.dump({"rows": rows}, open(a.out, "w"), indent=1)
print("wrote", a.out)
