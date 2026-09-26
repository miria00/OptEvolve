"""Structure census over the MIPLIB 2017 benchmark LP relaxations: what the tool's DETECTORS see, with no solve.

For each instance this records the converted size, the two structure counts the relocation transformations depend on
(single-nonzero bound rows, and rows whose whole support carries a bound: the "absorbable" rows), the greedy disjoint
family plan_product actually picks, whether the dominant-row diagnosis fires, and what plan_block returns (a plan or
the reason it refused). Nothing here solves anything, so it is pure CPU and parallel across instances.

Also computes the CEILING the greedy rule is measured against: the union of the supports of ALL absorbable rows, i.e.
the coverage a selection rule could reach if disjointness were free. The greedy rule takes rows in decreasing nonzero
count and skips any row sharing a column with one already taken, so the gap between its cover and this ceiling is the
planner's blind spot expressed in columns.

Usage
  python miplib_census.py --out census.jsonl [--workers 48] [--timeout 900] [--names a,b,c]
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class Timeout(Exception):
    pass


def _alarm(seconds):
    """Hard per-stage wall guard. Set inside the worker so one pathological instance cannot stall the census."""
    def handler(signum, frame):
        raise Timeout(f"stage exceeded {seconds}s")
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(int(seconds))


def _stage(fn, seconds, rec, key):
    """Run fn under a wall guard, timing it; failures and timeouts land in rec as strings, never as a crash."""
    t0 = time.perf_counter()
    try:
        _alarm(seconds)
        out = fn()
        return out
    except Exception as e:
        rec[key + "_error"] = f"{type(e).__name__}: {str(e)[:200]}"
        return None
    finally:
        signal.alarm(0)
        rec[key + "_s"] = round(time.perf_counter() - t0, 3)


def bound_structure(d):
    """The raw counts the relocation detectors read off (A, l, u), independent of any selection rule.

    A bound row is a row with exactly one nonzero: it pins one column to an interval, so relocation can absorb that
    column's box into the proximal step. A row of two or more nonzeros is ABSORBABLE when every column in its support
    carries such a bound row with at least one finite side, which is plan_product's admission test verbatim.

    n_bound_rows counts every single-nonzero row; the bound folding below skips the ones whose coefficient is an
    explicitly stored zero, which HiGHS does not produce, so the two agree on this data.

    ceiling_cover is the union of the supports of ALL absorbable rows: an UPPER BOUND on what any selection could
    absorb, not a reachable target, because a product set needs its blocks disjoint. It is the right yardstick for
    "how much structure is present" and the wrong one for "how much the greedy rule left behind"; miplib_planner
    measures the second by actually running other selection rules.
    """
    A = sp.csr_matrix(d["A"])
    l = np.asarray(d["l"], dtype=float)
    u = np.asarray(d["u"], dtype=float)
    m, n = A.shape
    nnz = np.diff(A.indptr)
    lo, hi = np.full(n, -np.inf), np.full(n, np.inf)
    has = np.zeros(n, dtype=bool)
    single = np.flatnonzero(nnz == 1)
    if single.size:
        cols = A.indices[A.indptr[single]]
        vals = A.data[A.indptr[single]]
        ok = vals != 0.0
        cols, vals, rows = cols[ok], vals[ok], single[ok]
        pos = vals > 0
        rl = np.where(pos, l[rows] / vals, u[rows] / vals)
        ru = np.where(pos, u[rows] / vals, l[rows] / vals)
        np.maximum.at(lo, cols, rl)
        np.minimum.at(hi, cols, ru)
        has[cols] = True
    fin = has & (np.isfinite(lo) | np.isfinite(hi))
    multi = np.flatnonzero(nnz >= 2)
    absorbable, cover_union = [], np.zeros(n, dtype=bool)
    for r in multi:
        c = A.indices[A.indptr[r]:A.indptr[r + 1]]
        v = A.data[A.indptr[r]:A.indptr[r + 1]]
        if fin[c].all() and np.all(v != 0.0):
            absorbable.append(int(r))
            cover_union[c] = True
    eq = np.isclose(l, u)
    return {"m": int(m), "n": int(n), "nnz": int(A.nnz), "n_bound_rows": int(single.size),
            "n_bounded_cols": int(has.sum()), "n_finitely_bounded_cols": int(fin.sum()),
            "n_multi_rows": int(multi.size), "n_absorbable_rows": len(absorbable),
            "absorbable_frac_of_multi": float(len(absorbable) / max(1, multi.size)),
            "ceiling_cover": float(cover_union.sum() / max(1, n)),
            "n_eq_rows": int(eq.sum()), "n_absorbable_eq_rows": int(sum(bool(eq[r]) for r in absorbable)),
            "max_row_nnz": int(nnz.max()) if m else 0, "mean_row_nnz": float(nnz.mean()) if m else 0.0,
            "absorbable_row_nnz_max": int(max((nnz[r] for r in absorbable), default=0)),
            "absorbable_rows": absorbable}


def census_one(name, timeout=900.0, keep_rows=False):
    import miplib
    import uni

    rec = {"name": name}
    t0 = time.perf_counter()
    d = _stage(lambda: miplib.load_miplib(name), timeout, rec, "load")
    if d is None:
        rec["total_s"] = round(time.perf_counter() - t0, 3)
        return rec
    rec["meta"] = d["meta"]
    bs = _stage(lambda: bound_structure(d), timeout, rec, "bounds")
    if bs is not None:
        rows = bs.pop("absorbable_rows")
        rec.update(bs)
        if keep_rows:
            rec["absorbable_rows"] = rows

    pp = _stage(lambda: uni.plan_product(d), timeout, rec, "plan_product")
    if pp is not None:
        pplan, reason = pp
        rec["plan_product"] = {"reason": reason}
        if pplan is not None:
            rec["plan_product"].update(K=int(pplan["K"]), n_x=int(pplan["n_x"]), cover=float(pplan["cover"]),
                                       fully_absorbed=bool(pplan["fully_absorbed"]),
                                       keep_rows=int(len(pplan["keep"])),
                                       block_sizes={"min": int(np.bincount(pplan["bid"]).min()),
                                                    "max": int(np.bincount(pplan["bid"]).max()),
                                                    "mean": float(np.bincount(pplan["bid"]).mean())})

    diag = _stage(lambda: uni.diagnose(d), timeout, rec, "diagnose")
    if diag is not None:
        rec["diagnose"] = {k: (float(v) if isinstance(v, (int, float, np.floating)) and k != "top_row"
                               else (int(v) if k == "top_row" else bool(v))) for k, v in diag.items()}
        pb = _stage(lambda: uni.plan_block(d, diag), timeout, rec, "plan_block")
        if pb is not None:
            bplan, reason = pb
            rec["plan_block"] = {"reason": reason}
            if bplan is not None:
                rec["plan_block"].update(kind=bplan["kind"], n_x=int(bplan["n_x"]),
                                         cover=float(bplan["n_x"] / max(1, rec.get("n", 1))),
                                         dense_row=int(bplan["dense_row"]),
                                         fully_absorbed=bool(bplan["fully_absorbed"]),
                                         keep_rows=int(len(bplan["keep"])))
    rec["total_s"] = round(time.perf_counter() - t0, 3)
    return rec


def _worker(args):
    name, timeout = args
    try:
        return census_one(name, timeout=timeout)
    except Exception as e:                            # a worker must return a record, never die
        return {"name": name, "fatal_error": f"{type(e).__name__}: {str(e)[:300]}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--names", default="")
    args = ap.parse_args()

    import miplib
    from multiprocessing import get_context

    # all 236, not list_miplib(usable_only=True): the two the manifest marks unusable do read (miplib._read_mps_highs)
    entries = miplib.list_miplib()
    names = [x for x in args.names.split(",") if x] or [e["name"] for e in entries]
    # biggest first: with a fixed pool that is the shortest makespan
    order = {e["name"]: (e["nnz"] or 0) for e in miplib.list_miplib()}
    names = sorted(names, key=lambda x: -order.get(x, 0))
    todo = [(x, args.timeout) for x in names]
    t0 = time.perf_counter()
    done = 0
    with open(args.out, "w") as fh, get_context("spawn").Pool(args.workers, maxtasksperchild=4) as pool:
        for rec in pool.imap_unordered(_worker, todo):
            fh.write(json.dumps(rec, default=str) + "\n")
            fh.flush()
            done += 1
            if done % 10 == 0:
                print(f"{done}/{len(todo)} {time.perf_counter() - t0:.0f}s", flush=True)
    print(f"census done {done} in {time.perf_counter() - t0:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
