"""Head to head against AUTHOR CODE on REAL downloaded instances.

Comparing against our own implementation is not a result. This runs our recipe
against solvers written by their own authors, on instances downloaded from the
public benchmark sets with recorded URLs and sha256, never on data we generated.

Baselines:
  HiGHS   via highspy / scipy, the HiGHS team's own code. Simplex or IPM, so it
          is a different algorithm class; reported for context, not as a
          like-for-like first-order comparison.
  PDLP    via ortools.pdlp, written by the PDLP authors. This is the correct
          like-for-like baseline: restarted primal-dual hybrid gradient, the
          same algorithm family as ours.

Data: data/lp_suites, fetched from plato.asu.edu (Mittelmann) and MIPLIB 2017,
manifest with per-instance URL, sha256, rows, cols, nnz.
"""
import argparse, json, os, sys, time
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--category", default="fctp", help="fctp | network | miplib2017-benchmark")
ap.add_argument("--max-nnz", type=int, default=20000)
ap.add_argument("--limit", type=int, default=8)
ap.add_argument("--tol", type=float, default=1e-4)
ap.add_argument("--time-limit", type=float, default=120.0)
ap.add_argument("--out", default="")
a = ap.parse_args()

from atlas.bench.mps_cell import (read_mps, to_standard_form, to_composite,
                                  solve_standard_highs, MANIFEST_PATH)

MAN = json.load(open(MANIFEST_PATH))
ROOT = Path(MANIFEST_PATH).parent

def category_of(inst):
    u = inst.get("url", "")
    if "lptestset/" in u:
        return u.split("lptestset/")[1].split("/")[0]
    return inst.get("source", "?")

picks = [i for i in MAN["instances"]
         if i.get("usable") and (i.get("nnz") or 0) <= a.max_nnz
         and (a.category == "any" or category_of(i) == a.category)]
picks.sort(key=lambda i: i.get("nnz") or 0)
picks = picks[:a.limit]
print(f"category={a.category}  instances={len(picks)}  (real, downloaded, sha256 in manifest)\n")

def run_highs(std):
    t0 = time.perf_counter()
    r = solve_standard_highs(std)
    return time.perf_counter() - t0, r

def run_pdlp(std, tol, tl):
    """PDLP, OR-Tools, the PDLP authors' own code.

    Runs in a separate interpreter: the ortools wheel bundles its own HiGHS and
    its libortools.so fails to load alongside the installed highspy with an
    undefined-symbol error. The subprocess boundary also keeps PDLP's threads
    out of our timing.
    """
    import subprocess, tempfile
    import scipy.sparse as sp
    A = sp.csr_matrix(std.A)
    f = tempfile.mktemp(suffix=".npz")
    np.savez(f, c=np.asarray(std.c, np.float64), b=np.asarray(std.b, np.float64),
             Ad=A.data, Ai=A.indices, Ap=A.indptr, Ashape=np.array(A.shape))
    try:
        r = subprocess.run(["/home/miria/pdlpenv/bin/python",
                            "/home/miria/OptEvolve/v2/scripts/baselines/pdlp_worker.py",
                            f, str(tol), str(tl)],
                           capture_output=True, text=True, timeout=tl + 60)
        if r.returncode != 0:
            return float("nan"), {"error": (r.stderr or "")[-160:]}
        d = json.loads(r.stdout.strip().splitlines()[-1])
        return d["t_s"], d
    finally:
        try: os.unlink(f)
        except OSError: pass

rows = []
for inst in picks:
    path = str(ROOT / inst["file"])
    try:
        raw = read_mps(path); std = to_standard_form(raw)
    except Exception as e:
        print(f"  {inst['name']:<20s} LOAD FAIL {str(e)[:60]}"); continue
    rec = {"name": inst["name"], "url": inst.get("url"), "sha256": inst.get("sha256"),
           "rows": inst.get("rows"), "cols": inst.get("cols"), "nnz": inst.get("nnz"),
           "std_n": int(std.c.shape[0]), "std_m": int(std.b.shape[0])}
    try:
        dt, r = run_highs(std)
        rec["highs"] = {"t_s": dt, "objective": float(r.get("objective", np.nan)),
                        "status": str(r.get("status"))}
    except Exception as e:
        rec["highs"] = {"error": str(e)[:100]}
    try:
        dt, r = run_pdlp(std, a.tol, a.time_limit)
        rec["pdlp"] = {"t_s": dt, **r}
    except Exception as e:
        rec["pdlp"] = {"error": str(e)[:160]}
    rows.append(rec)
    h = rec.get("highs", {}); p = rec.get("pdlp", {})
    print(f"  {inst['name']:<20s} n={rec['std_n']:<7d} m={rec['std_m']:<7d} "
          f"HiGHS {h.get('t_s', float('nan')):7.3f}s  "
          f"PDLP {p.get('t_s', float('nan')):7.3f}s "
          f"{('iters=' + str(p['iters'])) if 'iters' in p else p.get('error','')[:50]}", flush=True)

if a.out:
    Path(a.out).write_text(json.dumps({"args": vars(a), "rows": rows}, indent=1))
    print("\nwrote", a.out)
