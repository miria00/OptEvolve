"""Convex QP head to head on Maros-Meszaros: our Condat-Vu against author code.

Every solver's returned (x, y) is scored by ONE certificate,
atlas.certify.residuals.qp_rel_kkt (OSQP's primal and dual residuals plus a
duality gap), at the same tolerance. A baseline whose own stopping rule
returns a point failing the shared certificate is re-run with its tolerance
tightened tenfold, up to three times, and timed at the setting that passes, so
times are at matched accuracy rather than at nominal tolerance.

All solvers run single-threaded on the CPU, where these baselines live.
OSQP and SCS are first-order (ADMM) and so in our algorithm class; Clarabel is
an interior-point method, reported for context only.
"""
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, scipy.sparse as sp
import atlas  # noqa: F401
from atlas.bench import qp_cell
from atlas.bench.qp_cell import to_conic, y_from_conic
from atlas.certify.residuals import qp_rel_kkt
from atlas.backend.hardware import prepare_base
from atlas.genome import Backend, Genome, GenomeParams, Metric, Wrapper
from atlas.schemes.rescale import rescale_qp_problem
from atlas.schemes.base_schemes import Budget, run_built

ap = argparse.ArgumentParser()
ap.add_argument("--names", default="", help="comma list; default = all under --max-nnz")
ap.add_argument("--max-nnz", type=int, default=200000)
ap.add_argument("--tol", type=float, default=1e-4)
ap.add_argument("--cap", type=int, default=50000, help="our iteration cap")
ap.add_argument("--K", type=int, default=64)
ap.add_argument("--arms", default="none,ruiz:10", help="our arms, comma separated; each is tokens joined by +: none | ruiz:<iters> | haug")
ap.add_argument("--skip-baselines", action="store_true", help="only run our arms (baselines measured elsewhere)")
ap.add_argument("--out", required=True)
ap.add_argument("--shard", default="", help="i/N round-robin over size-sorted instances")
a = ap.parse_args()
DATA = Path("/home/miria/OptEvolve/v2/data/maros_meszaros")
man = json.load(open(DATA / "MANIFEST.json"))["instances"]
if a.names:
    want = set(a.names.split(",")); pick = [e for e in man if e["name"] in want]
else:
    pick = [e for e in man if e["nnzA"] + e["nnzP"] <= a.max_nnz]
pick.sort(key=lambda e: e["nnzA"] + e["nnzP"])
if a.shard:
    i, N = map(int, a.shard.split("/")); pick = pick[i::N]

def cert(prob, x, y):
    return float(qp_rel_kkt(np.asarray(x, float), np.asarray(y, float),
                            prob.f.P, prob.f.q, prob.L, prob.h.lo, prob.h.hi))

def objective(d, x):
    x = np.asarray(x, float)
    return float(0.5 * x @ (d["P"] @ x) + d["q"] @ x + d["r"])

def matched(run_at, prob, tol):
    """Run a baseline at eps = tol, tightening until the shared certificate passes."""
    tries = []
    for k in range(4):
        eps = tol / 10 ** k
        try:
            t, x, y, status = run_at(eps)
        except Exception as e:
            tries.append({"eps": eps, "error": f"{type(e).__name__}: {str(e)[:120]}"}); break
        c = cert(prob, x, y) if x is not None else float("inf")
        tries.append({"eps": eps, "t_s": t, "cert": c, "status": str(status)})
        if c <= tol:
            return {"passed": True, "obj": objective(prob.data, x), "t_s": t, "eps": eps, "cert": c, "status": str(status), "tries": tries}
    return {"passed": False, "tries": tries}

def run_osqp(d):
    import osqp
    def at(eps):
        t0 = time.perf_counter(); s = osqp.OSQP()
        s.setup(P=sp.triu(d["P"]).tocsc(), q=d["q"], A=d["A"], l=d["l"], u=d["u"], verbose=False,
                eps_abs=eps, eps_rel=eps, max_iter=200000, polishing=False)
        r = s.solve(); return time.perf_counter() - t0, r.x, r.y, r.info.status
    return at

def run_scs(d):
    import scs
    Ac, bc, nz, nl, idx = to_conic(d)
    def at(eps):
        t0 = time.perf_counter()
        s = scs.SCS({"P": sp.triu(d["P"]).tocsc(), "A": Ac, "b": bc, "c": d["q"]}, {"z": nz, "l": nl},
                    eps_abs=eps, eps_rel=eps, max_iters=200000, verbose=False)
        r = s.solve(); t = time.perf_counter() - t0
        return t, r["x"], y_from_conic(r["y"], idx, d["A"].shape[0]), r["info"]["status"]
    return at

def run_clarabel(d):
    import clarabel
    Ac, bc, nz, nl, idx = to_conic(d)
    cones = ([clarabel.ZeroConeT(nz)] if nz else []) + ([clarabel.NonnegativeConeT(nl)] if nl else [])
    def at(eps):
        st = clarabel.DefaultSettings(); st.verbose = False
        st.tol_gap_abs = st.tol_gap_rel = st.tol_feas = eps
        t0 = time.perf_counter()
        s = clarabel.DefaultSolver(sp.triu(d["P"]).tocsc(), d["q"], Ac, bc, cones, st)
        r = s.solve(); t = time.perf_counter() - t0
        return t, np.asarray(r.x), y_from_conic(np.asarray(r.z), idx, d["A"].shape[0]), r.status
    return at

def run_ours(prob, tol, arm):
    toks = arm.split("+")
    rz = [t for t in toks if t.startswith("ruiz:")]
    metric = Metric(kind="ruiz", iters=int(rz[0].split(":")[1])) if rz else None
    wrapper = Wrapper(kind="haugazeau") if "haug" in toks else Wrapper()
    run_on = prob if metric is None else rescale_qp_problem(prob, "ruiz", iters=metric.iters)[0]
    Lf, nA = float(run_on.f.props.lipschitz), float(run_on.L.norm_bound)
    tau = 1.0 / Lf; sigma = 0.49 / (tau * nA ** 2)      # t*L_f/2 + t*s*||A||^2 = 0.99
    g = Genome("condat_vu", GenomeParams(tau=tau, sigma=sigma), metric=metric, wrapper=wrapper,
               backend=Backend(precision="f64", K=a.K))
    t0 = time.perf_counter()
    built, c, params, _ = prepare_base(prob, g); t_prep = time.perf_counter() - t0
    t0 = time.perf_counter()
    r = run_built(built, c, params, Budget(max_iters=a.cap, tol=tol, sample_every=250), backend=g.backend)
    t_call = time.perf_counter() - t0
    x = np.asarray(r.x); y = np.asarray(r.u) if getattr(r, "u", None) is not None else None
    cv = cert(prob, x, y) if y is not None else float("inf")
    return {"passed": cv <= tol and r.iters < a.cap, "obj": objective(prob.data, x), "t_exec_s": float(getattr(r, "wall_time", np.nan)),
            "t_call_s": t_call, "t_prepare_s": t_prep, "iters": int(r.iters), "cert": cv,
            "tau": tau, "sigma": sigma}

rows = []
print(f"{len(pick)} instances, tol={a.tol}, our cap={a.cap} iterations, K={a.K}", flush=True)
for e in pick:
    rec = {"name": e["name"], "n": e["n"], "m": e["m"], "nnz": e["nnzA"] + e["nnzP"]}
    try:
        d = qp_cell.read_mm(str(DATA / e["file"])); prob = qp_cell.to_composite(d)
    except Exception as ex:
        rec["load_error"] = str(ex)[:120]; rows.append(rec); continue
    try:   # high-accuracy reference objective (interior point, 1e-9)
        if a.skip_baselines:
            raise RuntimeError("skipped")
        t_, x_, y_, st_ = run_clarabel(d)(1e-9)
        rec["ref"] = {"obj": objective(d, x_), "cert": cert(prob, x_, y_), "status": str(st_)}
    except Exception as ex:
        rec["ref"] = {"error": str(ex)[:120]}
    for name, fn in (() if a.skip_baselines else (("osqp", run_osqp), ("scs", run_scs), ("clarabel", run_clarabel))):
        rec[name] = matched(fn(d), prob, a.tol)
    for arm in a.arms.split(","):
        try:
            rec["ours_" + arm] = run_ours(prob, a.tol, arm)
        except Exception as ex:
            rec["ours_" + arm] = {"passed": False, "error": f"{type(ex).__name__}: {str(ex)[:160]}"}
    rows.append(rec)
    fmt = lambda v: (f"{v.get('t_s', v.get('t_exec_s', float('nan'))):8.3f}s" if v.get("passed") else "    FAIL ")
    ours = " ".join(f"{arm} {fmt(rec['ours_' + arm])} it={rec['ours_' + arm].get('iters', '-')}"
                    for arm in a.arms.split(","))
    base = ("" if a.skip_baselines else f"osqp {fmt(rec['osqp'])} scs {fmt(rec['scs'])} clarabel {fmt(rec['clarabel'])} ")
    print(f"{e['name']:12s} n={e['n']:6d} nnz={rec['nnz']:8d} | {base}| ours(exec) {ours}", flush=True)
    json.dump({"args": vars(a), "rows": rows}, open(a.out, "w"), indent=1)
print("wrote", a.out)
