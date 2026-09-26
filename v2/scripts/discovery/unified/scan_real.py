"""STATIC SCAN of real benchmarks (no solves): does the structure the formulation rule acts on occur in practice?

Per instance: size; bound rows; the diagnosis (top singular direction's row mass and singular gap after Ruiz); the
plan (absorbable block kind, or why not); and a census independent of dominance: rows with >= 2 nonzeros whose whole
support carries bound rows with a finite side (each is a box-hyperplane or box-slab set a projection can absorb), and a
greedy disjoint family of them (a product of such sets is also exactly projectable, block by block).
"""
import argparse, json, os, sys, time, traceback
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import scipy.sparse as sp

ap = argparse.ArgumentParser()
ap.add_argument("--set", choices=("mm", "qplib"), required=True)
ap.add_argument("--qplib-dir", default="")
ap.add_argument("--out", required=True)
ap.add_argument("--procs", type=int, default=24)
a = ap.parse_args()


def census(d):
    A = sp.csr_matrix(d["A"]); l, u = d["l"], d["u"]
    nnz = np.diff(A.indptr)
    n = A.shape[1]
    lo, hi, has = np.full(n, -np.inf), np.full(n, np.inf), np.zeros(n, bool)
    for r in np.flatnonzero(nnz == 1):
        c, v = A.indices[A.indptr[r]], A.data[A.indptr[r]]
        if v == 0:
            continue
        rl, ru = (l[r] / v, u[r] / v) if v > 0 else (u[r] / v, l[r] / v)
        lo[c], hi[c] = max(lo[c], rl), min(hi[c], ru)
        has[c] = True
    fin = has & (np.isfinite(lo) | np.isfinite(hi))
    rows = [r for r in np.flatnonzero(nnz >= 2) if fin[A.indices[A.indptr[r]:A.indptr[r + 1]]].all()]
    used, disjoint, cols = np.zeros(n, bool), 0, 0
    for r in sorted(rows, key=lambda r: -nnz[r]):
        s = A.indices[A.indptr[r]:A.indptr[r + 1]]
        if not used[s].any():
            used[s] = True; disjoint += 1; cols += s.size
    eq = int(np.sum(np.isclose(l, u) & (nnz >= 2)))
    return {"bound_rows": int((nnz == 1).sum()), "general_rows": int((nnz >= 2).sum()), "eq_rows": eq,
            "absorbable_rows": len(rows), "disjoint_absorbable": disjoint, "disjoint_cols": cols,
            "max_row_nnz": int(nnz.max()) if nnz.size else 0}


def one(key):
    from realdata import load_mm, load_qplib
    from uni import diagnose, plan_block
    t0 = time.time()
    rec = {"instance": key}
    try:
        d = load_mm(key) if a.set == "mm" else load_qplib(os.path.join(a.qplib_dir, f"QPLIB_{key}.qplib"))
        rec.update({"n": int(d["P"].shape[0]), "m": int(d["A"].shape[0]), "nnzA": int(d["A"].nnz),
                    "nnzP": int(d["P"].nnz)})
        rec.update(census(d))
        try:
            diag = diagnose(d)
            rec["diag"] = diag
            plan, why = plan_block(d, diag)
            rec["plan"] = plan["kind"] if plan else None
            rec["plan_why"] = why
            if plan:
                rec["plan_n_x"] = plan["n_x"]
        except Exception as e:
            rec["diag_error"] = f"{type(e).__name__}: {str(e)[:200]}"
    except Exception:
        rec["error"] = traceback.format_exc()[-400:]
    rec["scan_s"] = time.time() - t0
    return rec


if __name__ == "__main__":
    if a.set == "mm":
        from realdata import mm_names
        keys = mm_names()
    else:
        keys = sorted(f.split("_")[1].split(".")[0] for f in os.listdir(a.qplib_dir) if f.endswith(".qplib"))
    with Pool(a.procs) as pool, open(a.out, "w") as out:
        for rec in pool.imap_unordered(one, keys):
            out.write(json.dumps(rec, default=float) + "\n"); out.flush()
            print(rec["instance"], rec.get("n"), rec.get("plan"), (rec.get("diag") or {}).get("fires"),
                  rec.get("absorbable_rows"), rec.get("error", "")[:80], flush=True)
    print("SCAN_DONE")
