"""Family-wide test of the recipe the KB loop found on AUG3DCQP: Condat-Vu + safeguarded
Anderson(5), against plain Condat-Vu, on every Maros-Meszaros instance at the head-to-head's
certificate (qp_rel_kkt <= 1e-4, cap 5e4 evaluations, K = 64, CPU single thread). Baselines
(OSQP, SCS, Clarabel, matched accuracy) come from results/qp_20260910; join on name."""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import jax
jax.config.update("jax_enable_x64", True)
from atlas.bench import qp_cell
from atlas.evolve.kb_composer import Recipe, run_recipe

ap = argparse.ArgumentParser(); ap.add_argument("--shard", default="0/1"); ap.add_argument("--out", required=True)
a = ap.parse_args()
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
man = sorted([e for e in json.load(open(os.path.join(ROOT, "data/maros_meszaros/MANIFEST.json")))["instances"]
              if e["nnzA"] + e["nnzP"] <= 1e6], key=lambda e: e["nnzA"] + e["nnzP"])
i, N = map(int, a.shard.split("/"))
ARMS = {"cv": Recipe("condat_vu", 0.98), "cv+aa5": Recipe("condat_vu", 0.98, anderson=5)}
rows = []
for e in man[i::N]:
    rec = {"name": e["name"], "n": e["n"], "nnz": e["nnzA"] + e["nnzP"]}
    try:
        prob = qp_cell.load(os.path.join(ROOT, "data/maros_meszaros", e["file"]))
        for k, r in ARMS.items():
            out = run_recipe(prob, r, tol=1e-4, max_evals=50000, K=64)
            rec[k] = {"evals": out["evals"], "wall_s": out["wall_s"], "gap": out["gap"], "passed": out["converged"],
                      "aa_accepted": out["aa_accepted"]}
            jax.clear_caches()
    except Exception as ex:
        rec["error"] = f"{type(ex).__name__}: {str(ex)[:160]}"
    rows.append(rec)
    print(e["name"], {k: (v["passed"], v["evals"], round(v["wall_s"], 3)) for k, v in rec.items() if isinstance(v, dict)},
          rec.get("error", ""), flush=True)
    json.dump({"rows": rows}, open(a.out, "w"), indent=1)
print("wrote", a.out)
