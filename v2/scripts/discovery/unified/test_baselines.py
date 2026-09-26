"""SMOKE TEST for the virtual-best baseline field: every baseline solver in atlas/bench/baseline_proc.py, each in its
own isolated interpreter, on three small Maros-Meszaros QPs and one small LP, judged by OUR checker.

Instances: QAFIRO (sparse non-diagonal P, equality rows), QPCBLEND (diagonal P, so OR-Tools PDLP can take it),
CVXQP1_S (non-diagonal P, n=100) and the LP AFIRO, which is QAFIRO with P = 0 (Maros-Meszaros built QAFIRO by adding a
quadratic term to the netlib LP, and the OSQP-format file carries the variable bounds as rows, so the LP is bounded).

Every solver runs at every ladder rung eps, and every returned point is judged at EVERY rung with uni.accept against a
Clarabel 1e-9 reference: the solver's own status never decides acceptance. A solver without an isolated venv is
reported as NO-VENV and not run, because falling back to our interpreter would break the isolation rule silently.

    OPTEVOLVE_BASELINE_ENV_ROOT=/home/miria/baseline_envs python test_baselines.py [--jsonl out.jsonl]

The optional JSONL has one record per (instance, solver, eps, judged tol) in the virtual_best.py input format.
BASELINE_THREADS, if set, is passed through to every solver (baseline_proc reads it).
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# the parent never needs the GPU (the 4090 may be serving something else); cupdlpx's child is not a JAX process
os.environ.setdefault("JAX_PLATFORMS", "cpu")
for _root in ("/home/miria/baseline_envs", "/workspace" + os.pathsep + "/root/baseline_envs"):
    if all(os.path.isdir(p) for p in _root.split(os.pathsep)):
        os.environ.setdefault("OPTEVOLVE_BASELINE_ENV_ROOT", _root)
        break
import numpy as np
import scipy.sparse as sp
from realdata import load_mm
from uni import accept, reference
try:
    from uni import LADDER
except ImportError:                     # a uni.py synced before the ladder was added (the H100 copy on 2026-09-15)
    LADDER = (1e-2, 1e-3, 1e-4, 1e-6)
from atlas.bench.baseline_proc import ALL_SOLVERS, _env_python, run_baseline_proc

ap = argparse.ArgumentParser()
ap.add_argument("--solvers", default=",".join(ALL_SOLVERS))
ap.add_argument("--instances", default="QAFIRO,QPCBLEND,CVXQP1_S,LP:AFIRO")
ap.add_argument("--timeout", type=float, default=60.0)
ap.add_argument("--jsonl", default="")
a = ap.parse_args()


def load(name):
    if name.startswith("LP:"):
        d = load_mm("QAFIRO")
        d.update(P=sp.csc_matrix(d["P"].shape), name="LP/AFIRO(QAFIRO,P=0)")
        return d
    return load_mm(name)


def judge(x, d, obj_ref, tol):
    if x is None or not np.all(np.isfinite(x)):
        return False, float("inf"), float("inf")
    return accept(np.asarray(x, float), d, obj_ref, tol)


solvers = [s for s in a.solvers.split(",") if s]
print(f"env root {os.environ.get('OPTEVOLVE_BASELINE_ENV_ROOT')}  BASELINE_THREADS={os.environ.get('BASELINE_THREADS')}")
for s in solvers:
    py = _env_python(s)
    print(f"  {s:9s} {py}{'   NO-VENV (would fall back to this interpreter)' if py == sys.executable else ''}")
out = open(a.jsonl, "a") if a.jsonl else None
summary = {}
for name in a.instances.split(","):
    d = load(name)
    obj_ref, ref_t = reference(d, timeout_s=300)
    if obj_ref is None:
        print(f"== {d['name']}: reference failed: {ref_t}")
        continue
    n, m = d["P"].shape[0], d["A"].shape[0]
    print(f"\n== {d['name']}  n={n} m={m} P.nnz={d['P'].nnz} obj_ref={obj_ref:.10g}")
    print(f"  {'solver':9s} {'eps':>6s} {'status':30s} {'iters':>6s} {'t_s':>9s} {'rowviol':>8s} {'objgap':>8s} "
          f"{'dres':>8s}  accepted@" + "/".join(f"{t:g}" for t in LADDER))
    for s in solvers:
        if _env_python(s) == sys.executable:
            print(f"  {s:9s} NO-VENV")
            continue
        for eps in LADDER:
            rb = run_baseline_proc(d, s, eps, timeout_s=a.timeout, scratch=os.environ.get("OPTEVOLVE_SCRATCH"))
            if "error" in rb:
                msg = rb["error"].splitlines()[-1] if rb["error"] else "?"
                print(f"  {s:9s} {eps:6.0e} ERROR {msg[:110]}")
                summary.setdefault((d["name"], s), {})[eps] = None
                if rb.get("unsupported"):
                    break                                 # the class does not change with eps
                continue
            x, y = rb["x"], rb["y"]
            acc = {t: judge(x, d, obj_ref, t) for t in LADDER}
            _, rv, gap = acc[LADDER[0]]
            # stationarity residual on the original rows; catches a dual returned with the wrong sign convention
            dres = (float(np.abs(d["P"] @ x + d["q"] + d["A"].T @ y).max() / (1 + np.abs(d["q"]).max()))
                    if np.all(np.isfinite(y)) and np.all(np.isfinite(x)) else float("nan"))
            marks = "".join("Y" if acc[t][0] else "." for t in LADDER)
            print(f"  {s:9s} {eps:6.0e} {str(rb['status'])[:30]:30s} {rb['iters']:6d} {rb['t_s']:9.2e} {rv:8.1e} "
                  f"{gap:8.1e} {dres:8.1e}  {marks}")
            summary.setdefault((d["name"], s), {})[eps] = (bool(acc[eps][0]), rb["t_s"])
            if out:
                for t in LADDER:
                    out.write(json.dumps({"instance": d["name"], "tol": t, "solver": s, "eps": eps, "t_s": rb["t_s"],
                                          "accepted": bool(acc[t][0]), "rv": acc[t][1], "gap": acc[t][2],
                                          "status": str(rb["status"]), "iters": rb["iters"],
                                          "threads": rb.get("threads")}) + "\n")
if out:
    out.close()

# Matched rung: the run at eps = tol judged at tol, the cell a virtual best is built from.
print("\nMATCHED RUNG (run at eps = tol, judged at tol): time in ms, '-' = rejected by our checker, 'n/a' = error")
for inst in dict.fromkeys(k[0] for k in summary):
    print(f"  {inst}\n    {'solver':9s} " + " ".join(f"{t:>8g}" for t in LADDER))
    for s in solvers:
        if (inst, s) not in summary:
            continue
        cells = [summary[(inst, s)].get(t) for t in LADDER]
        print(f"    {s:9s} " + " ".join(f"{'n/a' if v is None else (f'{1e3 * v[1]:.2f}' if v[0] else '-'):>8s}"
                                       for v in cells))
