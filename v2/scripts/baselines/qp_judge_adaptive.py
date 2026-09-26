"""Apply qp_accuracy_judge to every pass of the adaptive-ratio QP arms (results/adaptive_20260911):
rerun each arm exactly as qp_adaptive.py ran it, keep x, judge per-row feasibility (1e-4) and the
objective against the trusted Clarabel-1e-9 reference."""
import argparse, glob, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, jax
jax.config.update("jax_enable_x64", True)
from atlas.backend.adaptive import run_adaptive
from atlas.backend.hardware import prepare_base
from atlas.bench import qp_cell
from atlas.certify.residuals import qp_accuracy_judge
from atlas.genome import Genome, GenomeParams, Metric
from atlas.schemes.rescale import rescale_qp_problem

ap = argparse.ArgumentParser(); ap.add_argument("--shard", default="0/1"); ap.add_argument("--out", required=True)
a = ap.parse_args()
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ad = {r["name"]: r for f in glob.glob(os.path.join(ROOT, "results/adaptive_20260911/qp_shard*.json")) for r in json.load(open(f))["rows"]}
base = {r["name"]: r for f in glob.glob(os.path.join(ROOT, "results/qp_20260910/full_shard*.json")) for r in json.load(open(f))["rows"]}
files = {e["name"]: e["file"] for e in json.load(open(os.path.join(ROOT, "data/maros_meszaros/MANIFEST.json")))["instances"]}
def trusted(n):
    ref = base.get(n, {}).get("ref"); st = str(ref.get("status", "")).lower() if isinstance(ref, dict) else ""
    return ref["obj"] if isinstance(ref, dict) and "obj" in ref and "solved" in st and "inaccurate" not in st else None
ARMS = ("ruiz:10+fixed", "ruiz:10+pdlp", "none+pdlp")
todo = sorted((n, k) for n, r in ad.items() for k in ARMS if r.get(k, {}).get("passed"))
i, N = map(int, a.shard.split("/"))
rows = []
for name, arm in todo[i::N]:
    d = qp_cell.read_mm(os.path.join(ROOT, "data/maros_meszaros", files[name])); prob = qp_cell.to_composite(d)
    met, rule = arm.split("+")
    metric = None if met == "none" else Metric(kind="ruiz", iters=int(met.split(":")[1]))
    run_on = prob if metric is None else rescale_qp_problem(prob, "ruiz", iters=metric.iters)[0]
    Lf, nb = float(run_on.f.props.lipschitz), float(run_on.L.norm_bound)
    t = 1.0 / Lf; s = 0.49 / (t * nb ** 2)
    built, cert, _, _ = prepare_base(prob, Genome("condat_vu", GenomeParams(tau=t, sigma=s), metric=metric))
    r = run_adaptive(built, cert, scheme="condat_vu", tol=1e-4, max_iters=50000, K=64, rule=rule, eta_budget=1e4, fill=0.99)
    j = qp_accuracy_judge(r["x"], d["P"], d["q"], d["r"], d["A"], d["l"], d["u"], ref_obj=trusted(name), feas_tol=1e-4)
    rows.append({"name": name, "arm": arm, **j}); jax.clear_caches()
    print(name, arm, j["verdict"], flush=True)
    json.dump({"rows": rows}, open(a.out, "w"), indent=1)
print("wrote", a.out)
