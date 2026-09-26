"""MIPLIB 2017 benchmark instances as LP RELAXATIONS in the solver's (P, q, r, A, l, u) form.

SOURCE. The 236 .mps.gz files fetched under data/lp_suites/miplib (MIPLIB 2017 benchmark set,
https://miplib.zib.de/WebData/instances/<name>.mps.gz), indexed by data/lp_suites/MANIFEST.json with the sha256 the
fetcher recorded. Two of the 236 are marked unusable in that manifest ("highspy read failed: HighsStatus.kWarning");
both are in fact readable, see _read_mps_highs, so all 236 load here.

RELAXATION. atlas.bench.mps_cell.read_mps parses the file through HiGHS, which reads integrality markers and then
hands back col_lower / col_upper unchanged; nothing here consults raw.n_integer except to record it. Dropping the
integrality requirement IS the continuous relaxation, so the dict below is the LP relaxation of the MIP by
construction, with the same bounds the MIP declares (binaries keep [0, 1]).

CONVERSION (load_miplib) follows netlib.load_netlib exactly, so a MIPLIB case and a Netlib case reach the router in
the same encoding:
  free rows (both row bounds infinite) are dropped: they constrain nothing;
  empty rows are dropped when 0 lies in [rl, ru] and raise otherwise (an all-zero row only adds a spurious zero
      singular value and a row the relative-violation metric can never move);
  every column with a finite lower or upper bound gets one identity row (fixed columns get l = u);
  maximization is negated into minimization; the objective constant goes to r.
Values at or beyond 1e20 are infinite (the HiGHS default infinite_bound).

PUBLISHED VALUES. MIPLIB 2017 does NOT publish LP relaxation values, and there is no table to cite for the benchmark
set as a whole (checked 2026-09-15): the instance pages and the benchmark table's "Objective" column are the MIP
optimum (documented there as "Optimal or best-known objective value", and equal to the =opt= line of
miplib2017.solu); miplib2017.solu carries only MIP bounds; the supplementary raw_data.zip features tables are 147
purely structural columns with no LP-derived field; and the MIPLIB 2017 paper computed root LP values only for its
14,571-instance NEOS pre-selection pool and states the performance data "is not meant for publication". The LP
benchmark papers that run on MIPLIB relaxations (PDLP, cuPDLP.jl, cuPDLP-C) and Mittelmann's pages report run times,
not objectives.

PUBLISHED_LP below is what does exist: the MIPLIB 3.0 catalog's "LP SOLUTION - optimal solution to the linear
relaxation of the problem" column, for the four of its instances that are still in the MIPLIB 2017 BENCHMARK set.
Everything else is certified here by AGREEMENT OF INDEPENDENT ROUTES: HiGHS on the converted dict, HiGHS on the
ORIGINAL model with integrality cleared (mps_cell.oracle_mps, a code path that never touches this conversion), and
Clarabel at 1e-9 in an isolated interpreter (uni.reference's own call). A mismatch that HiGHS shares is a conversion
question; one only Clarabel shows is a reference-solver question.

VALIDATION (2026-09-15, all 236 loaded; records in the study's validation.jsonl)
  Published:  HiGHS on the converted dict matches the MIPLIB 3.0 LP SOLUTION on all four instances that have one:
      air05 25877.60926785 vs 25877.609 (1.0e-8, the catalog's own three decimals), mas76 38893.90364052 vs
      38893.903641 (1.2e-11), markshare2 and pk1 both exactly 0.0.
  Conversion: HiGHS on the converted dict vs mps_cell.oracle_mps on the ORIGINAL model agrees to 3.5e-15 or better on
      every instance checked, so the row dropping, the bound rows and the sense negation are faithful.
  Reference: Clarabel at 1e-9 agrees with HiGHS to 1.1e-10 or better on nine of ten, and is off by 8.3e-6 on mas76.
      As on Netlib, uni.reference's Clarabel call is the weakest of the three; a MIPLIB study should take HiGHS as
      the acceptance reference, which is what miplib_gap.reference_obj does.

CACHE. The census and the gap runs need the converted dict in an interpreter that has jax but not highspy (the H100's
/workspace/venv), while the only interpreter there WITH highspy (the isolated highs env) has no jax. cache_dict()
writes the converted dict as a compressed .npz keyed by name, and load_miplib() reads it when present, so the parse
happens once in whichever interpreter can do it. read_mps_raw falls back to a highspy-only reader for the same
reason: atlas.bench.mps_cell imports jax at module scope.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import scipy.sparse as sp

_HERE = os.path.dirname(os.path.abspath(__file__))
_V2 = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
if _V2 not in sys.path:
    sys.path.insert(0, _V2)

INF = 1e20
SOURCE = "miplib2017-benchmark"
# data lives locally under v2/data/lp_suites; on the H100 it is staged in /dev/shm (only ~3 GB free on /workspace)
_ROOTS = (os.environ.get("OPTEVOLVE_MIPLIB_ROOT"), os.path.join(_V2, "data", "lp_suites"), "/dev/shm/data")
MIPLIB_CACHE = os.environ.get("OPTEVOLVE_MIPLIB_CACHE", "/dev/shm/miplib_npz")

# LP relaxation optima from the MIPLIB 3.0 catalog's "LP SOLUTION" column
# (https://miplib2010.zib.de/miplib3/miplib3_cat.txt; Bixby, Ceria, McZeal, Savelsbergh, "An Updated Mixed Integer
# Programming Library: MIPLIB 3.0", Optima 58, 1998), restricted to instances still in the MIPLIB 2017 BENCHMARK set.
# These are the only published LP relaxation values that apply to any instance of this suite. The catalog's companion
# INT SOLUTION column matches miplib2017.solu exactly for air05, markshare2 and pk1, which is the evidence that the
# instance files did not change between the two libraries; mas76's catalog INT SOLUTION drops a digit (4005.1 for the
# true 40005.05399) while its LP value is consistent, so mas76 is kept and noswot, whose catalog INT SOLUTION is the
# historical wrong -43, is not (it is outside the 2017 benchmark set anyway).
PUBLISHED_LP = {"air05": 25877.609, "markshare2": 0.0, "mas76": 38893.903641, "pk1": 0.0}
PUBLISHED_LP_SOURCE = "MIPLIB 3.0 catalog (Bixby et al., Optima 58, 1998), LP SOLUTION column"


def data_root(root=None):
    """Directory holding MANIFEST.json and the miplib/ subdirectory of .mps.gz files."""
    for r in ((root,) + _ROOTS):
        if r and os.path.exists(os.path.join(r, "MANIFEST.json")):
            return r
        if r and os.path.exists(os.path.join(r, "miplib", "MANIFEST.json")):
            return os.path.join(r, "miplib")
    raise FileNotFoundError(f"no MANIFEST.json under any of {_ROOTS}")


def manifest(root=None):
    with open(os.path.join(data_root(root), "MANIFEST.json")) as fh:
        doc = json.load(fh)
    return doc["instances"] if isinstance(doc, dict) else doc


def list_miplib(root=None, usable_only=False):
    """[{name, file, rows, cols, nnz, n_integer, maximize, usable, skip_reason, holdout_tier, bytes}] for the MIPLIB
    entries of the suite manifest, in increasing ORIGINAL-model nnz (unreadable entries, whose sizes are null, last).

    rows/cols/nnz are the ORIGINAL model's, as the fetcher recorded them; the converted dict has more rows (one per
    bounded column) and the same number of columns.
    """
    out = []
    for e in manifest(root):
        if e.get("source") != SOURCE:
            continue
        if usable_only and not e.get("usable", False):
            continue
        out.append({k: e.get(k) for k in ("name", "file", "rows", "cols", "nnz", "n_integer", "maximize", "usable",
                                          "skip_reason", "holdout_tier", "bytes", "sha256")})
    return sorted(out, key=lambda e: (e["nnz"] is None, e["nnz"] or 0, e["name"]))


def instance_path(name, root=None):
    r = data_root(root)
    for cand in (os.path.join(r, "miplib", name + ".mps.gz"), os.path.join(r, name + ".mps.gz")):
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError(f"{name}: no .mps.gz under {r}")


# ----------------------------------------------------------------------------------------------------------------
# Reading
# ----------------------------------------------------------------------------------------------------------------
class _RawLP:
    """The fields of mps_cell.RawLP the conversion below uses, for the highspy-only fallback reader."""

    def __init__(self, c, col_lower, col_upper, row_lower, row_upper, M, offset, maximize, name, n_integer):
        self.c, self.col_lower, self.col_upper = c, col_lower, col_upper
        self.row_lower, self.row_upper, self.M = row_lower, row_upper, M
        self.offset, self.maximize, self.name, self.n_integer = offset, maximize, name, n_integer


def _read_mps_highs(path):
    """mps_cell.read_mps with no jax import: the same highspy calls, the same fields, used when importing
    atlas.bench.mps_cell fails because the interpreter has highspy but not jax, and when that reader REFUSES a file
    HiGHS actually parsed.

    kWarning is accepted here, unlike in mps_cell.read_mps, which treats any status other than kOk as a read failure.
    That costs two of the 236 benchmark instances: momentum1 and var-smallemery-m6j6 are marked unusable in the suite
    manifest with "highspy read failed: HighsStatus.kWarning", and the warning HiGHS actually emits is
    "LP matrix packed vector contains 7 |value| in [2.51e-10, 1e-09] less than or equal to 1e-09: ignored". The model
    is fully read; HiGHS has dropped a handful of coefficients below its own 1e-9 small-matrix-value threshold, which
    is a rounding decision inside HiGHS and not a parse error. Only kError loses the model.
    """
    import highspy

    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    status = h.readModel(path)
    if status == highspy.HighsStatus.kError:
        raise ValueError(f"HiGHS could not read {path!r}: {status}")
    lp = h.getLp()
    m0, n0 = int(lp.num_row_), int(lp.num_col_)
    am = lp.a_matrix_
    start = np.asarray(am.start_, dtype=np.int64)
    index = np.asarray(am.index_, dtype=np.int64)
    value = np.asarray(am.value_, dtype=np.float64)
    if am.format_ == highspy.MatrixFormat.kColwise:
        M = sp.csc_matrix((value, index, start), shape=(m0, n0))
    else:
        M = sp.csr_matrix((value, index, start), shape=(m0, n0)).tocsc()
    integrality = np.asarray(lp.integrality_, dtype=np.int64)
    return _RawLP(c=np.asarray(lp.col_cost_, dtype=np.float64),
                  col_lower=np.asarray(lp.col_lower_, dtype=np.float64),
                  col_upper=np.asarray(lp.col_upper_, dtype=np.float64),
                  row_lower=np.asarray(lp.row_lower_, dtype=np.float64),
                  row_upper=np.asarray(lp.row_upper_, dtype=np.float64),
                  M=M, offset=float(lp.offset_), maximize=(lp.sense_ == highspy.ObjSense.kMaximize),
                  name=str(lp.model_name_) or os.path.basename(path),
                  n_integer=int(np.sum(integrality != 0)) if integrality.size else 0)


def read_mps_raw(name, root=None):
    """RawLP of one instance as HiGHS reads it (integrality parsed and then ignored: the LP relaxation).

    The shared reader is tried first; the highspy-only one covers both interpreters without jax and the kWarning
    files the shared reader rejects (see _read_mps_highs).
    """
    path = instance_path(name, root)
    try:
        from atlas.bench.mps_cell import read_mps
    except ImportError:
        return _read_mps_highs(path)
    try:
        return read_mps(path)
    except ValueError:
        return _read_mps_highs(path)


# ----------------------------------------------------------------------------------------------------------------
# Conversion
# ----------------------------------------------------------------------------------------------------------------
def convert(raw, name):
    """RawLP -> OSQP-form dict (P zero, q, r, A = [general rows; identity bound rows], l, u). See module docstring."""
    M = sp.csr_matrix(raw.M)
    m0, n = M.shape
    rl, ru = raw.row_lower.astype(float).copy(), raw.row_upper.astype(float).copy()
    xl, xu = raw.col_lower.astype(float).copy(), raw.col_upper.astype(float).copy()
    for v in (rl, xl):
        v[v <= -INF] = -np.inf
    for v in (ru, xu):
        v[v >= INF] = np.inf
    if np.any(rl > ru) or np.any(xl > xu):
        raise ValueError(f"{name}: inconsistent bounds (lower > upper)")
    free = np.isinf(rl) & np.isinf(ru)
    empty = np.diff(M.indptr) == 0
    bad_empty = empty & ~free & ((rl > 0) | (ru < 0))
    if bad_empty.any():
        raise ValueError(f"{name}: empty row with 0 outside its bounds (infeasible)")
    keep = ~free & ~empty
    s = -1.0 if raw.maximize else 1.0
    bounded = np.flatnonzero(np.isfinite(xl) | np.isfinite(xu))
    B = sp.csr_matrix((np.ones(bounded.size), (np.arange(bounded.size), bounded)), shape=(bounded.size, n))
    A = sp.vstack([M[keep], B], format="csc")
    return {"P": sp.csc_matrix((n, n)), "q": s * raw.c, "r": s * raw.offset, "A": A,
            "l": np.concatenate([rl[keep], xl[bounded]]), "u": np.concatenate([ru[keep], xu[bounded]]),
            "name": f"MIPLIB/{name}",
            "meta": {"rows_mps": int(m0), "cols": int(n), "free_rows_dropped": int(free.sum()),
                     "empty_rows_dropped": int((empty & ~free).sum()), "general_rows": int(keep.sum()),
                     "bound_rows": int(bounded.size), "fixed_cols": int(np.sum(xl == xu)),
                     "free_cols": int(np.sum(np.isinf(xl) & np.isinf(xu))), "maximize": bool(raw.maximize),
                     "sense_sign": s, "n_integer_relaxed": int(raw.n_integer)}}


def cache_path(name, cache=None):
    return os.path.join(cache or MIPLIB_CACHE, name + ".npz")


def cache_dict(name, root=None, cache=None, overwrite=False):
    """Parse one instance and store the converted dict as a compressed .npz. Returns the path."""
    path = cache_path(name, cache)
    if os.path.exists(path) and not overwrite:
        return path
    d = convert(read_mps_raw(name, root), name)
    A = sp.csc_matrix(d["A"])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + f".{os.getpid()}.tmp.npz"          # atomic publish: parallel builders must not read a partial file
    np.savez_compressed(tmp, data=A.data, indices=A.indices, indptr=A.indptr, shape=np.asarray(A.shape),
                        q=d["q"], l=d["l"], u=d["u"], r=np.asarray([d["r"]]), meta=np.asarray([json.dumps(d["meta"])]))
    os.replace(tmp, path)
    return path


def load_cached(name, cache=None):
    z = np.load(cache_path(name, cache), allow_pickle=False)
    m, n = (int(v) for v in z["shape"])
    A = sp.csc_matrix((z["data"], z["indices"], z["indptr"]), shape=(m, n))
    return {"P": sp.csc_matrix((n, n)), "q": z["q"], "r": float(z["r"][0]), "A": A, "l": z["l"], "u": z["u"],
            "name": f"MIPLIB/{name}", "meta": json.loads(str(z["meta"][0]))}


def load_miplib(name, root=None, cache=None, use_cache=True):
    """OSQP-form dict (P zero, q, r, A, l, u) for the LP RELAXATION of one MIPLIB 2017 benchmark instance."""
    if use_cache and os.path.exists(cache_path(name, cache)):
        return load_cached(name, cache)
    return convert(read_mps_raw(name, root), name)


def objective(d, x):
    """Original objective q'x + r (P is zero for an LP relaxation but kept general)."""
    return float(0.5 * x @ (sp.csc_matrix(d["P"]) @ x) + np.asarray(d["q"]) @ x + d["r"])


# ----------------------------------------------------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------------------------------------------------
def validate(name, root=None, cache=None, highs_timeout=1800.0, highs_solver="choose", clarabel_timeout=1800.0,
             oracle=True, published=None):
    """Cross-check one relaxation's optimal value across independent routes (module docstring).

    highs:    netlib.highs_objective on the CONVERTED dict, in process.
    oracle:   mps_cell.oracle_mps on the ORIGINAL .mps.gz with integrality cleared, which shares no code with the
              conversion above; its P_star already includes the model's own offset and sense.
    clarabel: run_baseline_proc at 1e-9 in an isolated interpreter, the call uni.reference makes.
    published: {name: value} of published LP relaxation values, defaulting to PUBLISHED_LP. The published column is
              the original model's objective, so it is compared with q'x + r, not with the q'x that uni.reference and
              uni.accept use.
    Relative errors use the standard |a - b| / max(1, |b|).
    """
    import time

    rec = {"name": name}
    t0 = time.perf_counter()
    try:
        d = load_miplib(name, root=root, cache=cache)
    except Exception as e:
        rec["load_error"] = f"{type(e).__name__}: {e}"
        return rec
    rec["load_s"] = time.perf_counter() - t0
    A = sp.csc_matrix(d["A"])
    rec.update(n=int(A.shape[1]), m=int(A.shape[0]), nnz=int(A.nnz), r=float(d["r"]), meta=d["meta"])

    def rel(v, ref):
        return None if (v is None or ref is None) else abs(v - ref) / max(1.0, abs(ref))

    from netlib import highs_objective

    t0 = time.perf_counter()
    try:
        st, obj = highs_objective(d, time_limit=highs_timeout, solver=highs_solver)
        rec["highs"] = {"status": st, "obj": obj, "s": time.perf_counter() - t0}
    except Exception as e:
        rec["highs"] = {"error": f"{type(e).__name__}: {e}"}
    if oracle:
        t0 = time.perf_counter()
        try:
            from atlas.bench.mps_cell import oracle_mps
            o = oracle_mps(instance_path(name, root))
            rec["oracle"] = {"status": o["status"], "obj": o["P_star"], "cache_hit": o["cache_hit"],
                             "s": time.perf_counter() - t0}
        except Exception as e:
            rec["oracle"] = {"error": f"{type(e).__name__}: {e}"}
    t0 = time.perf_counter()
    from atlas.bench.baseline_proc import run_baseline_proc
    rb = run_baseline_proc(d, "clarabel", 1e-9, timeout_s=clarabel_timeout, scratch=os.environ.get("OPTEVOLVE_SCRATCH"))
    if "error" in rb:
        rec["clarabel"] = {"error": str(rb["error"])[:300]}
    else:
        from atlas.certify.residuals import qp_accuracy_judge
        x = np.asarray(rb["x"])
        j = qp_accuracy_judge(x, d["P"], d["q"], d["r"], d["A"], d["l"], d["u"], ref_obj=None, feas_tol=1e-6)
        rec["clarabel"] = {"obj": objective(d, x), "row_violation": float(j["row_violation"]),
                           "t_s": rb.get("t_s"), "s": time.perf_counter() - t0}
    hv = (rec.get("highs") or {}).get("obj")
    ov = (rec.get("oracle") or {}).get("obj")
    cv = (rec.get("clarabel") or {}).get("obj")
    pv = (PUBLISHED_LP if published is None else published).get(name)
    rec["agree"] = {"highs_vs_oracle": rel(hv, ov), "clarabel_vs_highs": rel(cv, hv),
                    "highs_vs_published": rel(hv, pv), "published": pv,
                    "published_source": PUBLISHED_LP_SOURCE if pv is not None else None}
    return rec
