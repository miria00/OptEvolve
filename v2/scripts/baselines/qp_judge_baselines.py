"""Apply qp_accuracy_judge to OSQP and Clarabel passes: rerun each at the exact tolerance it needed
in the head-to-head (same settings), keep x, judge per-row feasibility and objective against the
trusted reference. Instances where the solver originally took over --max-seconds are skipped and listed."""
import argparse, glob, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, scipy.sparse as sp
from atlas.bench import qp_cell
from atlas.certify.residuals import qp_accuracy_judge

ap = argparse.ArgumentParser(); ap.add_argument("--shard", default="0/1"); ap.add_argument("--out", required=True)
ap.add_argument("--feas-tol", type=float, default=1e-4, help="per-row feasibility tolerance; match the run tolerance")
ap.add_argument("--max-seconds", type=float, default=60.0)
a = ap.parse_args()
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
base = {r["name"]: r for f in glob.glob(os.path.join(ROOT, "results/qp_20260910/full_shard*.json")) for r in json.load(open(f))["rows"]}
def trusted(n):
    ref = base.get(n, {}).get("ref")
    st = str(ref.get("status", "")).lower() if isinstance(ref, dict) else ""
    return ref["obj"] if isinstance(ref, dict) and "obj" in ref and "solved" in st and "inaccurate" not in st else None

def to_conic(d):
    A, l, u = d["A"].tocsr(), d["l"], d["u"]
    eq = np.isfinite(l) & np.isfinite(u) & (l == u); up = np.isfinite(u) & ~eq; lo = np.isfinite(l) & ~eq
    return sp.vstack([A[eq], A[up], -A[lo]]).tocsc(), np.concatenate([u[eq], u[up], -l[lo]]), int(eq.sum()), int(up.sum() + lo.sum())

def solve(s, d, eps):
    if s == "osqp":
        import osqp
        m = osqp.OSQP(); m.setup(P=sp.triu(d["P"]).tocsc(), q=d["q"], A=d["A"], l=d["l"], u=d["u"], verbose=False,
                                 eps_abs=eps, eps_rel=eps, max_iter=200000, polishing=False)
        return m.solve().x
    import clarabel
    Ac, bc, nz, nl = to_conic(d)
    cones = ([clarabel.ZeroConeT(nz)] if nz else []) + ([clarabel.NonnegativeConeT(nl)] if nl else [])
    st = clarabel.DefaultSettings(); st.verbose = False; st.tol_gap_abs = st.tol_gap_rel = st.tol_feas = eps
    return np.asarray(clarabel.DefaultSolver(sp.triu(d["P"]).tocsc(), d["q"], Ac, bc, cones, st).solve().x)

todo, skipped = [], []
for n, r in sorted(base.items()):
    for s in ("osqp", "clarabel"):
        v = r.get(s, {})
        if v.get("passed"):
            (todo if v["t_s"] <= a.max_seconds else skipped).append((n, s, v["eps"]))
i, N = map(int, a.shard.split("/"))
rows = []
for n, s, eps in todo[i::N]:
    d = qp_cell.read_mm(os.path.join(ROOT, "data/maros_meszaros", base[n]["name"] + ".mat")) if False else \
        qp_cell.read_mm(next(os.path.join(ROOT, "data/maros_meszaros", e["file"]) for e in
                             json.load(open(os.path.join(ROOT, "data/maros_meszaros/MANIFEST.json")))["instances"] if e["name"] == n))
    try:
        x = solve(s, d, eps)
        j = qp_accuracy_judge(x, d["P"], d["q"], d["r"], d["A"], d["l"], d["u"], ref_obj=trusted(n), feas_tol=a.feas_tol)
    except Exception as ex:
        j = {"verdict": "error", "error": str(ex)[:120]}
    rows.append({"name": n, "solver": s, "eps": eps, **j})
    print(n, s, j["verdict"], flush=True)
    json.dump({"rows": rows, "skipped_over_max_seconds": skipped if i == 0 else []}, open(a.out, "w"), indent=1)
print("wrote", a.out, "| skipped (slow):", len(skipped))
