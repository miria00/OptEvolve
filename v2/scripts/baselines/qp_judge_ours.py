"""Apply qp_accuracy_judge (per-row feasibility + objective against a trusted high-accuracy
reference) to every pass of our composed Condat-Vu arms."""
import argparse, glob, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, jax
jax.config.update("jax_enable_x64", True)
from atlas.bench import qp_cell
from atlas.certify.residuals import qp_accuracy_judge
from atlas.evolve.kb_composer import Recipe, run_recipe

ap = argparse.ArgumentParser(); ap.add_argument("--shard", default="0/1"); ap.add_argument("--out", required=True)
ap.add_argument("--feas-tol", type=float, default=1e-4, help="per-row feasibility tolerance; match the run tolerance")
ap.add_argument("--composed-dir", default="results/qp_composed_20260911", help="qp_composed.py shard outputs to judge")
a = ap.parse_args()
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
comp = {r["name"]: r for f in glob.glob(os.path.join(ROOT, a.composed_dir, "shard*.json")) for r in json.load(open(f))["rows"]}
base = {r["name"]: r for f in glob.glob(os.path.join(ROOT, "results/qp_20260910/full_shard*.json")) for r in json.load(open(f))["rows"]}
def trusted(n):
    ref = base.get(n, {}).get("ref")
    st = str(ref.get("status", "")).lower() if isinstance(ref, dict) else ""
    return ref["obj"] if isinstance(ref, dict) and "obj" in ref and "solved" in st and "inaccurate" not in st else None
ARMS = {"cv": Recipe("condat_vu", 0.98), "cv+aa5": Recipe("condat_vu", 0.98, anderson=5)}
todo = sorted((n, k) for n, r in comp.items() for k in ARMS if r.get(k, {}).get("passed"))
i, N = map(int, a.shard.split("/"))
rows = []
for name, arm in todo[i::N]:
    d = qp_cell.read_mm(os.path.join(ROOT, "data/maros_meszaros", name + ".mat")); prob = qp_cell.to_composite(d)
    out = run_recipe(prob, ARMS[arm], tol=1e-4, max_evals=50000, K=64)
    j = qp_accuracy_judge(out["x"], d["P"], d["q"], d["r"], d["A"], d["l"], d["u"], ref_obj=trusted(name), feas_tol=a.feas_tol)
    rows.append({"name": name, "arm": arm, **j}); jax.clear_caches()
    print(name, arm, j["verdict"], f"{j['row_violation']:.1e}", j["obj_err"], flush=True)
    json.dump({"rows": rows}, open(a.out, "w"), indent=1)
print("wrote", a.out)
