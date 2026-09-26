"""Netlib LP (the 98 problems of the lp/data PROBLEM SUMMARY TABLE) in the solver's (P, q, r, A, l, u) form.

SOURCES (recorded per file in data/netlib/MANIFEST.json by fetch_netlib)
  92 problems   https://www.netlib.org/lp/data/<name>     netlib compressed format, expanded with netlib's own emps.c
                                                           (https://www.netlib.org/lp/data/emps.c), the canonical route
  STOCFOR3,     https://www.zib.de/userpage/koch/perplex/data/netlib/mps/<name>.mps.gz
  TRUSS                                                    netlib ships only Fortran generator bundles for these two;
                                                           the ZIB copies are the MPS files Koch verified in exact
                                                           rational arithmetic ("The final NETLIB-LP results", Oper.
                                                           Res. Lett. 32(2):138-142, 2004)
  QAP8, QAP12,  https://www.netlib.org/lp/generators/qap/  netlib ships Terri Johnson's generator (newlp.f) and data files;
  QAP15                                                    _qap_mps below is a line-for-line port that writes the same
                                                           fixed-format MPS (validated only through the optimal value)

PUBLISHED OPTIMA
  "minos"  PROBLEM SUMMARY TABLE of https://www.netlib.org/lp/data/readme (MINOS 5.3, VAX double precision), with the
           readme's CPLEX(Sparc) and MINOS(MIPS) alternatives kept alongside where it lists them
  "koch"   the exact rational objective of the perPlex verification (https://www.zib.de/userpage/koch/perplex/data/
           netlib/txt/<name>.txt.gz, header line "Objvalue"), available for 93 problems (no QAP, no STANDGUB)
STANDGUB has no published value: its GUB marker rows 'EGROUP' and 'ENDX' are not LP rows, and the readme says that with
the lines naming them removed it IS STANDATA. The loader removes them, so its reference is STANDATA's value.
Both published columns EXCLUDE the objective-row RHS constant. E226 is the only problem with one (RHS -7.113 on the
objective row, which MPS semantics and HiGHS read as r = +7.113): q'x matches both tables to 1e-11, q'x + r is off by
exactly 7.113. Published values are therefore compared with q'x, which is also what uni.reference returns.

CONVERSION (load_netlib). HiGHS parses the MPS (mps_cell.read_mps: RANGES, all BOUNDS kinds, objective RHS as offset,
OBJSENSE). Then, in OSQP form with bounds as identity rows (the realdata.load_qplib convention):
  free rows (N rows other than the objective, or both row bounds infinite) are dropped: they constrain nothing;
  empty rows are dropped when 0 lies in [rl, ru] and raise otherwise, since an all-zero row only adds a spurious
      zero singular value and a row the relative-violation metric can never move;
  every column with a finite lower or upper bound gets one identity row (fixed columns get l = u);
  maximization is negated into minimization; the objective constant goes to r.
Values at or beyond 1e20 are infinite (the HiGHS default infinite_bound).

VALIDATION (2026-09-15, all 98 loaded; records in results/lp_families_20260915/netlib_validation.jsonl)
  HiGHS on the converted dict: within 1e-10 of Koch's exact value on all 95 problems that have one, and within 4e-11
      of the MINOS value on QAP8/12/15, so the conversion (and the QAP generator port) reproduces the published LPs.
  The MINOS table itself is off by more than 1e-6 on ten problems (80BAU3B, GANGES, GREENBEA 1.3e-3, GREENBEB, NESM,
      PILOT 1.5e-4, PILOT.WE, PILOT87, SCRS8, STOCFOR3); Koch's value and the readme's CPLEX column agree with HiGHS.
  Clarabel at 1e-9 (uni.reference): within 1e-6 of the best published value on 93 of 98. It is WRONG on DFL001
      (status DualInfeasible after one iteration, row violation 0.99) and GREENBEA (status Solved, row violation
      1.5e-5, objective off 1.3e-3), and off by 1.2e-6 to 1.8e-5 on PILOT, PILOT.JA and PILOT87. Do not use
      uni.reference as the acceptance reference on those five without a cross-check.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from fractions import Fraction

import numpy as np
import scipy.sparse as sp

_HERE = os.path.dirname(os.path.abspath(__file__))
_V2 = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
if _V2 not in sys.path:
    sys.path.insert(0, _V2)

NETLIB_DIR = os.path.join(_V2, "data", "netlib")
NETLIB_BASE = "https://www.netlib.org/lp/data"
QAP_BASE = "https://www.netlib.org/lp/generators/qap"
ZIB_BASE = "https://www.zib.de/userpage/koch/perplex/data/netlib"
INF = 1e20
ZIB_MPS = ("stocfor3", "truss")
QAP = {"qap8": 8, "qap12": 12, "qap15": 15}


# ----------------------------------------------------------------------------------------------------------------
# Published values
# ----------------------------------------------------------------------------------------------------------------
def parse_readme(text):
    """{lower name: {"rows", "cols", "nnz", "minos", "cplex_sparc", "minos_mips"}} from the netlib lp/data readme.

    Names are lowercased as netlib's file names are (PILOT.JA -> pilot.ja). "minos" is None where the table says
    "(see NOTES)" (STANDGUB)."""
    out = {}
    lines = text.splitlines()
    i0 = next(i for i, s in enumerate(lines) if s.strip() == "PROBLEM SUMMARY TABLE")   # the title, not the prose
    i1 = next(i for i, s in enumerate(lines) if s.strip() == "BOUND-TYPE TABLE")
    for s in lines[i0 + 1:i1]:
        t = s.split()
        if len(t) < 5 or not re.match(r"^[0-9A-Z][0-9A-Z.\-]*$", t[0]) or not t[1].isdigit():
            continue
        # the optimum is the only token in exponent notation; byte counts are plain integers, STANDGUB has none
        vals = re.findall(r"-?\d\.\d+E[+-]\d+", s)
        out[t[0].lower()] = {"rows": int(t[1]), "cols": int(t[2]), "nnz": int(t[3]),
                             "minos": float(vals[-1]) if vals else None, "cplex_sparc": None, "minos_mips": None}
    # the table of values that differ under CPLEX (Sparc) and MINOS (MIPS); column position decides which is which
    j0 = next(i for i, s in enumerate(lines) if "CPLEX(Sparc)" in s and "MINOS(MIPS)" in s)
    hdr = lines[j0]
    split_col = (hdr.index("CPLEX(Sparc)") + hdr.index("MINOS(MIPS)")) // 2     # values sit left of their headers
    for s in lines[j0 + 1:]:
        if s.strip().startswith("The above CPLEX"):
            break
        t = s.split()
        if not t or t[0].lower() not in out:
            continue
        name = t[0].lower()
        for m in re.finditer(r"-?\d\.\d+E[+-]\d+", s):
            key = "cplex_sparc" if m.start() < split_col else "minos_mips"
            out[name][key] = float(m.group(0))
    return out


def _koch_name(name):
    return name.replace(".", "-")


def _koch_objvalue(name, timeout=60):
    """Exact optimum from the perPlex solution header, as (float, "p/q" string); None when ZIB has no file."""
    url = f"{ZIB_BASE}/txt/{_koch_name(name)}.txt.gz"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            with gzip.GzipFile(fileobj=r) as g:
                for _ in range(40):
                    line = g.readline().decode("latin-1")
                    if line.startswith("* Objvalue"):
                        q = line.split(":", 1)[1].strip()
                        # PILOT87 and MAROS-R7 have numerators of 5000+ digits, past Python's default int-parse limit
                        lim = sys.get_int_max_str_digits()
                        sys.set_int_max_str_digits(0)
                        try:
                            val = float(Fraction(q))
                        finally:
                            sys.set_int_max_str_digits(lim)
                        return val, (q if len(q) <= 200 else f"<{len(q)} characters, see koch_url>"), url
    except Exception:
        return None
    return None


# ----------------------------------------------------------------------------------------------------------------
# QAP generator: port of netlib lp/generators/qap/newlp.f (Terri Johnson)
# ----------------------------------------------------------------------------------------------------------------
def _qap_mps(data_text):
    """MPS text for the linearized QAP of one netlib data file. Follows newlp.f statement by statement (1-based
    indices kept), including its fixed-column layout, so the output is the file the Fortran program writes."""
    toks = [int(x) for x in re.split(r"[,\s]+", data_text.strip()) if x]
    M = toks[0]
    IA = np.array(toks[1:1 + M * M]).reshape(M, M)
    IDIS = np.zeros((M + 1, M + 1), dtype=np.int64)
    IFLOW = np.zeros((M + 1, M + 1), dtype=np.int64)
    for i in range(1, M + 1):
        for j in range(1, M + 1):
            if j > i:
                IDIS[i, j] = IDIS[j, i] = IA[i - 1, j - 1]
            elif j < i:
                IFLOW[i, j] = IFLOW[j, i] = IA[i - 1, j - 1]
    ICOST = {}
    for L in range(1, M + 1):
        for K in range(2, M + 1):
            for J in range(1, M + 1):
                if J == L:
                    continue
                for I in range(1, K):
                    it = int(IFLOW[I, K] * IDIS[J, L])
                    ICOST[(I, J, K, L)] = it
                    ICOST[(K, L, I, J)] = it
    MM, MTM, MTWO = M * M, 2 * (M - 1), 2 * M
    out = [f"NAME     NEWLP{M:02d}", "ROWS", "  N NOBJ"]
    out += [f"  E NX{i:02d}" for i in range(1, MTWO + 1)]
    out += [f"  E C{i:02d}A{j:03d}" for j in range(1, MM + 1) for i in range(1, MTM + 1)]
    out.append("COLUMNS")
    col = lambda name, row, val: f"    {name:<10}{row:<10}{val}"
    k = 0
    for i in range(1, M + 1):
        for j in range(1, M + 1):
            k += 1
            out.append(col(f"X{k:03d}", f"NX{i:02d}", "1.0"))
            out.append(col(f"X{k:03d}", f"NX{M + j:02d}", "1.0"))
            out += [col(f"X{k:03d}", f"C{l:02d}A{k:03d}", "-1.0") for l in range(1, MTM + 1)]
    for K in range(1, M + 1):
        IRA = K
        for L in range(1, M + 1):
            KL = (K - 1) * M + L
            ICA = L
            for I in range(K + 1, M + 1):
                IR = I
                IC = 0
                for J in range(1, M + 1):
                    if J == L:
                        continue
                    ocoef = float(ICOST.get((I, J, K, L), 0) + ICOST.get((K, L, I, J), 0))
                    IC += 1
                    IT = IC + M - 1
                    if J < L:
                        IJ, ITA = (I - 2) * (M - 1) + J, ICA + M - 2
                    else:
                        IJ, ITA = (I - 2) * (M - 1) + J - 1, ICA + M - 1
                    KLA = (I - 1) * M + J
                    y = f"Y{IJ:03d}A{KL:03d}"
                    out.append(col(y, "NOBJ", f"{ocoef:10.1f}"))
                    out.append(col(y, f"C{IR - 1:02d}A{KL:03d}", "1.0"))
                    out.append(col(y, f"C{IT:02d}A{KL:03d}", "1.0"))
                    out.append(col(y, f"C{IRA:02d}A{KLA:03d}", "1.0"))
                    out.append(col(y, f"C{ITA:02d}A{KLA:03d}", "1.0"))
    out.append("RHS")
    out += [col("RHS", f"NX{i:02d}", "1.0") for i in range(1, MTWO + 1)]
    out.append("ENDATA")
    return "\n".join(out) + "\n"


# ----------------------------------------------------------------------------------------------------------------
# Fetch
# ----------------------------------------------------------------------------------------------------------------
def _get(url, timeout=120):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read()


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_netlib(dest=NETLIB_DIR, names=None, koch=True, verbose=True):
    """Download (or generate) every problem of the readme table into dest/<name>.mps and write dest/MANIFEST.json.

    Existing .mps files are kept (their sha256 is recorded again), so a rerun only fills gaps."""
    os.makedirs(dest, exist_ok=True)
    readme_path = os.path.join(dest, "readme")
    if not os.path.exists(readme_path):
        with open(readme_path, "wb") as f:
            f.write(_get(f"{NETLIB_BASE}/readme"))
    table = parse_readme(open(readme_path, encoding="latin-1").read())
    names = list(table) if names is None else [n.lower() for n in names]
    tmp = tempfile.mkdtemp(prefix="netlib_emps_")
    emps = os.path.join(tmp, "emps")
    try:
        src = os.path.join(tmp, "emps.c")
        with open(src, "wb") as f:
            f.write(_get(f"{NETLIB_BASE}/emps.c"))
        subprocess.run(["gcc", "-O2", "-w", src, "-o", emps], check=True)
        manifest_path = os.path.join(dest, "MANIFEST.json")
        old = json.load(open(manifest_path))["problems"] if os.path.exists(manifest_path) else {}
        recs = dict(old)
        for name in names:
            path = os.path.join(dest, name + ".mps")
            rec = dict(old.get(name, {}))
            rec.update({"file": name + ".mps", **table[name]})
            try:
                if name in QAP:
                    url = f"{QAP_BASE}/data.{QAP[name]}"
                    rec.update(method="generated: python port of newlp.f", url=url, generator=f"{QAP_BASE}/newlp.f")
                    if not os.path.exists(path):
                        with open(path, "w") as f:
                            f.write(_qap_mps(_get(url).decode()))
                elif name in ZIB_MPS:
                    url = f"{ZIB_BASE}/mps/{_koch_name(name)}.mps.gz"
                    rec.update(method="zib perplex mps.gz, gunzipped", url=url)
                    if not os.path.exists(path):
                        with open(path, "wb") as f:
                            f.write(gzip.decompress(_get(url)))
                else:
                    url = f"{NETLIB_BASE}/{name}"
                    rec.update(method="netlib compressed, expanded with emps.c", url=url)
                    if not os.path.exists(path):
                        raw = os.path.join(tmp, name + ".z")
                        with open(raw, "wb") as f:
                            f.write(_get(url))
                        with open(path, "wb") as f:
                            subprocess.run([emps, raw], stdout=f, check=True)
                rec.update(bytes=os.path.getsize(path), sha256=_sha256(path), error=None)
            except Exception as e:                                          # keep going: report, do not hide
                rec.update(error=f"{type(e).__name__}: {e}")
                if os.path.exists(path) and os.path.getsize(path) == 0:
                    os.unlink(path)
            if koch and name not in QAP and rec.get("koch") is None:
                kv = _koch_objvalue("standata" if name == "standgub" else name)
                if kv is not None:
                    rec.update(koch=kv[0], koch_exact=kv[1], koch_url=kv[2])
            recs[name] = rec
            if verbose:
                print(f"{name:10s} {rec.get('error') or 'ok'} {rec.get('bytes')}", flush=True)
        manifest = {"readme": f"{NETLIB_BASE}/readme", "emps": f"{NETLIB_BASE}/emps.c",
                    "koch": f"{ZIB_BASE}/txt/", "problems": recs}
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=1)
        return manifest
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ----------------------------------------------------------------------------------------------------------------
# Load
# ----------------------------------------------------------------------------------------------------------------
def manifest(root=None):
    return json.load(open(os.path.join(root or NETLIB_DIR, "MANIFEST.json")))["problems"]


def netlib_names(root=None):
    d = root or NETLIB_DIR
    return sorted(f[:-4] for f in os.listdir(d) if f.endswith(".mps"))


def published_optimum(name, root=None, prefer="koch"):
    """(value, source) for the loaded problem, comparable with q'x (not q'x + r); STANDGUB maps to STANDATA.

    prefer="koch" takes the exact perPlex value where ZIB has one and the readme's MINOS 5.3 value otherwise (the QAPs).
    The MINOS table is off by more than 1e-6 relative on ten problems (GREENBEA by 1.3e-3, PILOT by 1.5e-4), where
    Koch's value, the readme's own CPLEX column and HiGHS on the converted dict agree to 1e-11; prefer="minos" gives
    the table value regardless."""
    m = manifest(root)
    key = "standata" if name == "standgub" else name
    via = " via STANDATA" if key != name else ""
    if prefer == "koch" and m[key].get("koch") is not None:
        return m[key]["koch"], "Koch 2004 perPlex exact value" + via
    return m[key]["minos"], "netlib lp/data readme (MINOS 5.3)" + via


def _strip_gub_markers(path):
    """STANDGUB's GUB marker rows are not LP rows (readme NOTES); drop every line naming them into a temp copy."""
    with open(path, encoding="latin-1") as f:
        text = f.read()
    if "'EGROUP'" not in text:
        return path, None
    fd, tmp = tempfile.mkstemp(suffix=".mps")
    with os.fdopen(fd, "w") as f:
        f.writelines(s for s in text.splitlines(True) if "'EGROUP'" not in s and "'ENDX'" not in s)
    return tmp, tmp


def read_mps_raw(name, root=None):
    """mps_cell.RawLP of one problem as HiGHS reads it, GUB markers removed."""
    from atlas.bench.mps_cell import read_mps

    rpath, tmp = _strip_gub_markers(os.path.join(root or NETLIB_DIR, name.lower() + ".mps"))
    try:
        return read_mps(rpath)
    finally:
        if tmp is not None:
            os.unlink(tmp)


def load_netlib(name, root=None):
    """OSQP-form dict (P zero, q, r, A = [general rows; identity bound rows], l, u) for one Netlib problem."""
    name = name.lower()
    raw = read_mps_raw(name, root)
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
            "name": f"NETLIB/{name}",
            "meta": {"rows_mps": int(m0), "cols": int(n), "free_rows_dropped": int(free.sum()),
                     "empty_rows_dropped": int((empty & ~free).sum()), "general_rows": int(keep.sum()),
                     "bound_rows": int(bounded.size), "fixed_cols": int(np.sum(xl == xu)),
                     "free_cols": int(np.sum(np.isinf(xl) & np.isinf(xu))), "maximize": bool(raw.maximize),
                     "sense_sign": s}}


def objective(d, x):
    """Original objective q'x + r (P is zero for an LP but kept general)."""
    return float(0.5 * x @ (sp.csc_matrix(d["P"]) @ x) + np.asarray(d["q"]) @ x + d["r"])


def highs_objective(d, time_limit=600.0, solver="choose"):
    """Solve the CONVERTED dict with HiGHS (independent of the MPS reader's own model), returning (status, q'x + r).

    This checks the conversion itself: a mismatch with the published value that HiGHS shares is a data or reader
    question, one that only Clarabel shows is a reference-solver question."""
    import highspy

    A = sp.csc_matrix(d["A"])
    m, n = A.shape
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("time_limit", float(time_limit))
    h.setOptionValue("solver", solver)
    lp = highspy.HighsLp()
    lp.num_col_, lp.num_row_ = n, m
    lp.col_cost_ = np.asarray(d["q"], dtype=float)
    lp.offset_ = float(d["r"])
    lp.col_lower_ = np.full(n, -highspy.kHighsInf)
    lp.col_upper_ = np.full(n, highspy.kHighsInf)
    lp.row_lower_ = np.where(np.isinf(d["l"]), -highspy.kHighsInf, d["l"])
    lp.row_upper_ = np.where(np.isinf(d["u"]), highspy.kHighsInf, d["u"])
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = A.indptr
    lp.a_matrix_.index_ = A.indices
    lp.a_matrix_.value_ = A.data
    lp.a_matrix_.num_col_, lp.a_matrix_.num_row_ = n, m
    h.passModel(lp)
    h.run()
    return h.modelStatusToString(h.getModelStatus()), float(h.getInfo().objective_function_value)


def validate(name, clarabel_timeout=1800.0, highs_timeout=1800.0, highs_solver="choose", root=None):
    """Load one problem and compare against the published optima.

    clarabel: the call uni.reference makes (run_baseline_proc, clarabel, eps 1e-9, isolated interpreter), repeated here
              only so the status and the point's own row violation are kept; its objective q'x + r is uni.reference's
              value plus r.
    highs:    the converted dict solved in process (highs_objective), which separates a conversion error from a
              reference-solver error. highs_solver="ipm" is for QAP15, where dual simplex runs past half an hour.
    Relative errors are |q'x - published| / max(1, |published|): the published tables leave out r (module docstring)."""
    import time
    from atlas.bench.baseline_proc import run_baseline_proc
    from atlas.certify.residuals import qp_accuracy_judge

    rec = {"name": name}
    m = manifest(root)
    key = "standata" if name == "standgub" else name
    pub = {"minos": m[key].get("minos"), "koch": m[key].get("koch"), "cplex_sparc": m[key].get("cplex_sparc"),
           "minos_mips": m[key].get("minos_mips")}
    rec["published"] = pub
    t0 = time.perf_counter()
    try:
        d = load_netlib(name, root)
    except Exception as e:
        rec["load_error"] = f"{type(e).__name__}: {e}"
        return rec
    rec["load_s"] = time.perf_counter() - t0
    A = sp.csc_matrix(d["A"])
    rec.update(n=int(A.shape[1]), m=int(A.shape[0]), nnz=int(A.nnz), r=float(d["r"]), meta=d["meta"])

    def rel(v, ref):
        return None if (v is None or ref is None) else abs(v - ref) / max(1.0, abs(ref))

    t0 = time.perf_counter()
    try:
        st, obj = highs_objective(d, time_limit=highs_timeout, solver=highs_solver)
        rec["highs"] = {"status": st, "obj": obj, "s": time.perf_counter() - t0, "solver": highs_solver,
                        **{f"rel_{k}": rel(obj - d["r"], v) for k, v in pub.items()}}
    except Exception as e:
        rec["highs"] = {"error": f"{type(e).__name__}: {e}"}
    rb = run_baseline_proc(d, "clarabel", 1e-9, timeout_s=clarabel_timeout, scratch=os.environ.get("OPTEVOLVE_SCRATCH"))
    if "error" in rb:
        rec["clarabel"] = {"error": rb["error"][:300]}
    else:
        x = np.asarray(rb["x"])
        obj = objective(d, x)
        j = qp_accuracy_judge(x, d["P"], d["q"], d["r"], A, d["l"], d["u"], feas_tol=1e-6)
        rec["clarabel"] = {"status": rb.get("status"), "obj": obj, "s": rb.get("t_s"), "iters": rb.get("iters"),
                           "row_violation": j["row_violation"],
                           **{f"rel_{k}": rel(obj - d["r"], v) for k, v in pub.items()}}
    return rec


if __name__ == "__main__":
    # python netlib.py                      fetch everything into data/netlib
    # python netlib.py validate [--ipm] NAME [...]  one JSON line per problem on stdout
    if len(sys.argv) > 1 and sys.argv[1] == "validate":
        args = sys.argv[2:]
        ipm = "--ipm" in args
        for nm in (a for a in args if a != "--ipm"):
            print(json.dumps(validate(nm, highs_solver="ipm" if ipm else "choose")), flush=True)
    else:
        fetch_netlib()
