"""CBLIB (the Conic Benchmark Library, https://cblib.zib.de) continuous second-order cone programs in the pipeline's
conic form: the QP linear fields (P, q, r, A, l, u with +-inf for absent bounds) plus "soc", a list of cone constraints
{F (csc m_k x n), g (m_k), f (n), h (float)} meaning ||F x + g||_2 <= f'x + h.

CBF files state  min/max c'x + c0  s.t.  x in K_var (cones over consecutive variables),  Ax + b in K_con (cones over
consecutive rows). Accepted cones are F, L+, L-, L= (linear) and Q, QR (second order); anything else (PSD variables or
constraints, EXP, POW) or any INT section makes the instance unsupported. P is always zero: CBF objectives are linear.
Maximization is negated into minimization, as realdata.load_qplib does; "obj_sign" maps our objective back.

Of the 3346 CBLIB files (scanned 2026-09-15), 72 qualify; the 28 with at most ~400k nonzeros are in data/cblib with
manifest.json. Reproduce with the subcommands  scan -> count -> fetch --names ... -> validate  (see _main).
"""
import csv
import gzip
import io
import json
import os
import numpy as np
import scipy.sparse as sp

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "data", "cblib")
URL = "https://cblib.zib.de/download/all/"
# Published primal/dual objectives (MOSEK, from the CBLIB 2014 paper). Instances added to CBLIB after 2014 have no
# published objective anywhere we could find, so for those only feasibility and cross-solver agreement are checkable.
STAT_URL = "https://raw.githubusercontent.com/HFriberg/cblib-base/master/cblib/instances/stat.set-cblib2014.csv"
LINEAR = ("F", "L+", "L-", "L=")
SECOND_ORDER = ("Q", "QR")
DATA_KEYWORDS = ("OBJFCOORD", "OBJACOORD", "OBJBCOORD", "FCOORD", "ACOORD", "BCOORD", "HCOORD", "DCOORD", "CHANGE")
# One dense f per cone costs 8 n bytes, so instances with many small cones over many variables blow up
# (chainsing-10000-1 would need 29 GiB). Refuse before allocating rather than let the OOM killer take the process.
MAX_DENSE_F_BYTES = 2 * 1024 ** 3


def _open(path):
    return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8") if str(path).endswith(".gz") else open(path)


def _records(f):
    """Token lists of the non-comment, non-empty lines. The format only allows comments between blocks, so skipping
    them everywhere is safe."""
    for raw in f:
        s = raw.strip()
        if s and not s.startswith("#"):
            yield s.split()


def _block(it, cnt, width):
    """`cnt` coordinate lines of `width` tokens as a (cnt, width) float array, parsed in one numpy call because
    per-line float() dominates load time on the 100k-nonzero instances."""
    if cnt == 0:
        return np.zeros((0, width))
    toks = []
    for _ in range(cnt):
        toks.extend(next(it)[:width])
    return np.array(toks, dtype=float).reshape(cnt, width)


def read_cbf(f, structure_only=False, skip_data=False):
    """Parse a CBF stream into a raw record (sizes, cone lists, coordinate arrays). With structure_only the scan stops
    at the first problem-data keyword, which is enough to classify a remote file without downloading its data; with
    skip_data the coordinate blocks are counted (rec["counts"]) but not parsed.
    Only the first instance of an embedded hotstart sequence is read: CHANGE is treated as end of file, as the format
    permits."""
    rec = {"ver": None, "sense": "MIN", "n": 0, "var": [], "int": 0, "psdvar": [], "m": 0, "con": [], "psdcon": [],
           "powcones": 0, "psd_data": 0, "obja": np.zeros((0, 2)), "objb": 0.0, "acoord": np.zeros((0, 3)),
           "bcoord": np.zeros((0, 2)), "counts": {}}
    it = _records(f)
    for tok in it:
        kw = tok[0]
        if kw == "CHANGE" or (structure_only and kw in DATA_KEYWORDS):
            break
        if kw == "VER":
            rec["ver"] = int(next(it)[0])
        elif kw == "OBJSENSE":
            rec["sense"] = next(it)[0].upper()
        elif kw in ("POWCONES", "POW*CONES"):
            k = int(next(it)[0])
            rec["powcones"] += k
            for _ in range(k):
                for _ in range(int(next(it)[0])):
                    next(it)
        elif kw in ("PSDVAR", "PSDCON"):
            rec[kw.lower()] = [int(next(it)[0]) for _ in range(int(next(it)[0]))]
        elif kw in ("VAR", "CON"):
            size, k = (int(t) for t in next(it)[:2])
            cones = [(t[0], int(t[1])) for t in (next(it) for _ in range(k))]
            if sum(d for _, d in cones) != size:
                raise ValueError(f"{kw} cone dimensions sum to {sum(d for _, d in cones)}, header says {size}")
            rec["n" if kw == "VAR" else "m"], rec[kw.lower()] = size, cones
        elif kw == "INT":
            rec["int"] = int(next(it)[0])
            for _ in range(rec["int"]):
                next(it)
        elif kw == "OBJBCOORD":
            rec["objb"] = float(next(it)[0])
        elif kw in ("OBJACOORD", "ACOORD", "BCOORD"):
            cnt = int(next(it)[0])
            rec["counts"][kw] = cnt
            key, width = {"OBJACOORD": ("obja", 2), "ACOORD": ("acoord", 3), "BCOORD": ("bcoord", 2)}[kw]
            if skip_data:
                for _ in range(cnt):
                    next(it)
            else:
                rec[key] = _block(it, cnt, width)
        elif kw in ("OBJFCOORD", "FCOORD", "HCOORD", "DCOORD"):
            # PSD data only exists alongside PSDVAR/PSDCON, which makes the instance unsupported; skip without parsing.
            cnt = int(next(it)[0])
            rec["psd_data"] += cnt
            for _ in range(cnt):
                next(it)
        else:
            raise ValueError(f"unknown CBF keyword {kw!r}")
    return rec


def unsupported_reason(rec):
    """None when the record is a continuous problem over linear and second-order cones only, else why not."""
    if rec["int"]:
        return f"{rec['int']} integer variables"
    if rec["psdvar"] or rec["psdcon"] or rec["psd_data"]:
        return "PSD variables or constraints"
    bad = sorted({c for c, _ in rec["var"] + rec["con"] if c not in LINEAR + SECOND_ORDER})
    return f"unsupported cones {bad}" if bad else None


def _n_soc(rec):
    return sum(1 for c, k in rec["var"] + rec["con"] if c in SECOND_ORDER and not (c == "Q" and k == 1))


def _f_vec(M, rows, weight, n, dense):
    """weight * (sum of rows `rows` of M) as a cone's f: a dense n-array, or a 1-D csr_array when dense is False, so
    f @ x is a scalar in both cases."""
    R = M[rows].tocoo()
    if dense:
        out = np.zeros(n)
        np.add.at(out, R.col, weight * R.data)
        return out
    return sp.csr_array(sp.coo_array((weight * R.data, (R.col,)), shape=(n,)))


def _cone_blocks(M, b, offset, cones, lin, soc, dense_f=True):
    """Map cones over consecutive rows of the affine map M x + b (M csr) into linear rows and SOC constraints.

    Q  on rows (r0, r1..): ||(Mx+b)[r1:]|| <= (Mx+b)[r0].
    QR on rows (r0, r1, r2..): 2 t1 t2 >= ||w||^2, t1, t2 >= 0 with t = (Mx+b)[r0], (Mx+b)[r1]. The orthogonal rotation
       u = (t1+t2)/sqrt2, v = (t1-t2)/sqrt2 gives u^2 - v^2 = 2 t1 t2, and u >= |v| forces t1, t2 >= 0, so the cone is
       exactly ||(v, w)|| <= u. Using the orthogonal rotation (not ||(t1-t2, sqrt2 w)|| <= t1+t2) keeps w's scale.
    Q of dimension 1 is t >= 0 and becomes a linear row, since an SOC with an empty F is a degenerate special case
    downstream solvers need not accept.
    """
    s2 = np.sqrt(0.5)
    n = M.shape[1]
    for kind, k in cones:
        r0 = offset
        offset += k
        if kind == "F":
            continue                              # free rows constrain nothing; dropping them avoids all-inf rows
        if kind in ("L+", "L-", "L="):
            lin.append((slice(r0, r0 + k), kind))
        elif kind == "Q" and k == 1:
            lin.append((slice(r0, r0 + 1), "L+"))
        elif kind == "Q":
            # contiguous slices, not index arrays: CSR slicing is several times faster over tens of thousands of cones
            soc.append({"F": sp.csc_matrix(M[r0 + 1:r0 + k]), "g": b[r0 + 1:r0 + k].copy(),
                        "f": _f_vec(M, slice(r0, r0 + 1), 1.0, n, dense_f), "h": float(b[r0])})
        elif kind == "QR":
            t1, t2 = M[r0], M[r0 + 1]
            F = sp.vstack([s2 * (t1 - t2), M[r0 + 2:r0 + k]], format="csc")
            g = np.concatenate([[s2 * (b[r0] - b[r0 + 1])], b[r0 + 2:r0 + k]])
            soc.append({"F": F, "g": g, "f": _f_vec(M, slice(r0, r0 + 2), s2, n, dense_f),
                        "h": float(s2 * (b[r0] + b[r0 + 1]))})
        else:
            raise ValueError(f"unsupported cone {kind!r}")


def load_cbf(path, dense_f=True, max_dense_f_bytes=MAX_DENSE_F_BYTES):
    """Load a continuous linear/second-order CBF instance (.cbf or .cbf.gz) into the conic dict.

    dense_f=True (the socp.py contract) makes every f a dense n-array and refuses instances whose f's would exceed
    max_dense_f_bytes; dense_f=False gives 1-D scipy csr_array f's instead, which is what makes the many-small-cone
    instances (chainsing, nql, qssp, strain) loadable at all.

    Rows of A are, in order: linear constraint rows (L+, L-, L= cones of CON, file order), then variable bound rows
    (identity rows for L+, L-, L= cones of VAR). Variable Q/QR cones become SOC constraints on identity rows, placed
    after the constraint cones. Constants of affine expressions move into l/u (linear) and g/h (SOC)."""
    with _open(path) as f:
        rec = read_cbf(f)
    why = unsupported_reason(rec)
    if why:
        raise ValueError(f"{os.path.basename(str(path))}: {why}")
    n, m = rec["n"], rec["m"]
    n_soc = _n_soc(rec)
    if dense_f and 8 * n * n_soc > max_dense_f_bytes:
        raise MemoryError(f"dense f needs {8 * n * n_soc / 1024 ** 3:.1f} GiB ({n_soc} cones x {n} variables)")
    c = np.zeros(n)
    oa = rec["obja"]
    np.add.at(c, oa[:, 0].astype(int), oa[:, 1])
    ac = rec["acoord"]
    A = sp.csr_matrix((ac[:, 2], (ac[:, 0].astype(int), ac[:, 1].astype(int))), shape=(m, n))
    b = np.zeros(m)
    bc = rec["bcoord"]
    np.add.at(b, bc[:, 0].astype(int), bc[:, 1])
    lin_con, lin_var, soc_con, soc_var = [], [], [], []
    _cone_blocks(A, b, 0, rec["con"], lin_con, soc_con, dense_f)
    _cone_blocks(sp.identity(n, format="csr"), np.zeros(n), 0, rec["var"], lin_var, soc_var, dense_f)
    blocks, lo, up = [], [], []
    for M, bb, lin in ((A, b, lin_con), (sp.identity(n, format="csr"), np.zeros(n), lin_var)):
        for rows, kind in lin:
            blocks.append(M[rows])
            k = rows.stop - rows.start
            lo.append(-bb[rows] if kind in ("L+", "L=") else np.full(k, -np.inf))
            up.append(-bb[rows] if kind in ("L-", "L=") else np.full(k, np.inf))
    Alin = sp.vstack(blocks, format="csc") if blocks else sp.csc_matrix((0, n))
    sign = -1.0 if rec["sense"] == "MAX" else 1.0
    cones = {}
    for kind, k in rec["var"] + rec["con"]:
        cones[kind] = cones.get(kind, 0) + 1
    name = os.path.basename(str(path)).replace(".gz", "").replace(".cbf", "")
    return {"P": sp.csc_matrix((n, n)), "q": sign * c, "r": sign * rec["objb"], "A": Alin,
            "l": np.concatenate(lo) if lo else np.zeros(0), "u": np.concatenate(up) if up else np.zeros(0),
            "soc": soc_con + soc_var, "name": "CBLIB/" + name, "sense": rec["sense"], "obj_sign": sign,
            "cbf_cones": cones, "cbf_nnz": int(rec["counts"].get("ACOORD", 0))}


def objective(d, x):
    """Objective of `x` in the file's own sense (undoes the MAX negation), for comparison with published values."""
    x = np.asarray(x, dtype=float)
    return d["obj_sign"] * float(0.5 * x @ (d["P"] @ x) + d["q"] @ x + d["r"])


def violation(d, x, relative=False):
    """(linear, soc) maximum violations of `x`: max(l - Ax, Ax - u, 0) and max(||F x + g|| - (f'x + h), 0). With
    relative, each row is divided by 1 + |its bound| and each cone by 1 + |f'x + h|, the scaling interior point
    tolerances are stated in, so badly scaled instances (sched_*_orig) are not misread as infeasible."""
    x = np.asarray(x, dtype=float)
    Ax = d["A"] @ x
    lin = np.maximum(d["l"] - Ax, Ax - d["u"])
    if relative:
        lin = lin / (1.0 + np.where(Ax < d["l"], np.abs(d["l"]), np.abs(d["u"])))
    worst_lin = float(np.max(lin, initial=0.0))
    worst_soc = 0.0
    for c in d["soc"]:
        t = float(c["f"] @ x + c["h"])
        v = float(np.linalg.norm(c["F"] @ x + c["g"])) - t
        worst_soc = max(worst_soc, v / (1.0 + abs(t)) if relative else v)
    return max(worst_lin, 0.0), worst_soc


def known_objectives(root=None):
    """{name: (primal objective, dual objective)} published for the CBLIB 2014 set, from the local copy of
    stat.set-cblib2014.csv (columns PACK;NAME;SENSE;certificate;error;OBJECTIVE;certificate;error;OBJECTIVE). Only rows
    with both certificates parse, which is exactly the 41 continuous instances; mixed-integer rows have no dual."""
    path = os.path.join(root or ROOT, "stat.set-cblib2014.csv")
    if not os.path.exists(path):
        return {}
    out = {}
    with open(path) as fh:
        # positional, not DictReader: the header repeats ERROR and OBJECTIVE, so a dict silently keeps only the dual
        for row in list(csv.reader(fh, delimiter=";"))[1:]:
            try:
                out[row[1]] = (float(row[5]), float(row[8]))
            except (IndexError, ValueError):
                pass
    return out


def list_instances(root=None):
    """Downloaded instances as recorded in manifest.json (name, url, sizes, cone counts, published objective), sorted
    by nonzeros. Falls back to the bare file names when there is no manifest."""
    d = root or ROOT
    man = os.path.join(d, "manifest.json")
    if os.path.exists(man):
        with open(man) as fh:
            return sorted(json.load(fh), key=lambda e: e.get("nnz", 0))
    return [{"name": f.split(".cbf")[0], "path": os.path.join(d, f)} for f in sorted(os.listdir(d)) if ".cbf" in f]


def instance_path(name, root=None):
    d = root or ROOT
    for ext in (".cbf.gz", ".cbf"):
        if os.path.exists(os.path.join(d, name + ext)):
            return os.path.join(d, name + ext)
    raise FileNotFoundError(name)


def scan_remote(name, url=URL, timeout=120, full=False):
    """Structure of one remote instance. By default the gzip stream is read only up to the first data keyword, so
    classifying all ~3.3k CBLIB files costs a few MB instead of the 7 GB full download; full=True streams the whole
    file to count nonzeros (nothing is kept on disk)."""
    import urllib.request
    with urllib.request.urlopen(url + name + ".cbf.gz", timeout=timeout) as resp:
        with io.TextIOWrapper(gzip.GzipFile(fileobj=resp), encoding="utf-8") as f:
            rec = read_cbf(f, structure_only=not full, skip_data=True)
    out = {"name": name, "n": rec["n"], "m": rec["m"], "int": rec["int"], "psd": bool(rec["psdvar"] or rec["psdcon"]),
           "var_cones": sorted({c for c, _ in rec["var"]}), "con_cones": sorted({c for c, _ in rec["con"]}),
           "n_soc": _n_soc(rec), "reason": unsupported_reason(rec)}
    if full:
        out["nnz"] = rec["counts"].get("ACOORD", 0)
        out["obj_nnz"] = rec["counts"].get("OBJACOORD", 0)
    return out


def _listing(url=URL, timeout=120):
    """(name, compressed size) of every instance in the CBLIB directory listing."""
    import re
    import urllib.request
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        txt = resp.read().decode("utf-8", "replace")
    return re.findall(r'<a href="([^"]+)\.cbf\.gz">[^<]*</a>\s+\S+ \S+\s+(\S+)', txt)


def _parallel(fn, items, workers, out_path):
    """Apply fn to items with a few threads, appending one JSON line per result so an interrupted scan resumes."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    done = set()
    if os.path.exists(out_path):
        with open(out_path) as fh:
            done = {json.loads(line)["name"] for line in fh}
    todo = [i for i in items if i not in done]
    with open(out_path, "a") as fh, ThreadPoolExecutor(workers) as ex:
        for fut in as_completed([ex.submit(fn, i) for i in todo]):
            fh.write(json.dumps(fut.result()) + "\n")
            fh.flush()


def _retry(fn, name, tries=3):
    err = None
    for _ in range(tries):
        try:
            return fn(name)
        except Exception as e:                    # network hiccups on a 3.3k-file scan; record rather than abort
            err = repr(e)
    return {"name": name, "error": err}


def _main():
    import argparse
    import time
    import urllib.request
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s1 = sub.add_parser("scan", help="classify every CBLIB instance from its streamed header (few MB total)")
    s1.add_argument("--out", default=os.path.join(ROOT, "scan_all.jsonl"))
    s1.add_argument("--workers", type=int, default=8)
    s2 = sub.add_parser("count", help="stream the qualifying instances fully to count nonzeros")
    s2.add_argument("--scan", default=os.path.join(ROOT, "scan_all.jsonl"))
    s2.add_argument("--out", default=os.path.join(ROOT, "catalog_socp.jsonl"))
    s2.add_argument("--workers", type=int, default=4)
    s2.add_argument("--max-gz-mb", type=float, default=10.0, help="skip files whose compressed size exceeds this")
    s3 = sub.add_parser("fetch", help="download named instances and write manifest.json")
    s3.add_argument("--names", required=True, help="comma-separated")
    s3.add_argument("--catalog", default=os.path.join(ROOT, "catalog_socp.jsonl"))
    s4 = sub.add_parser("validate", help="solve every manifest instance with Clarabel (isolated env) and judge it")
    s4.add_argument("--out", default=os.path.join(ROOT, "validation_clarabel.json"))
    # 1e-10, not 1e-8: on the strain instances (nql, qssp) Clarabel at 1e-8 stops ~1e-6 (relative) above the optimum
    # that Clarabel at 1e-12 and SCS at 1e-10 agree on, which is the same size as the discrepancy being measured
    s4.add_argument("--tol", type=float, default=1e-10)
    s4.add_argument("--timeout", type=float, default=900.0)
    s4.add_argument("--scratch", default=os.environ.get("OPTEVOLVE_SCRATCH") or None)
    a = ap.parse_args()
    os.makedirs(ROOT, exist_ok=True)

    if a.cmd == "scan":
        sizes = dict(_listing())
        _parallel(lambda nm: dict(_retry(scan_remote, nm), gz_size=sizes[nm]), list(sizes), a.workers, a.out)
    elif a.cmd == "count":
        with open(a.scan) as fh:
            recs = [json.loads(line) for line in fh]
        mb = lambda t: float(t[:-1]) * {"K": 1e-3, "M": 1.0, "G": 1e3}[t[-1]] if t[-1] in "KMG" else float(t) * 1e-6
        names = [r["name"] for r in recs
                 if "error" not in r and r["reason"] is None and mb(r["gz_size"]) <= a.max_gz_mb]
        _parallel(lambda nm: _retry(lambda x: scan_remote(x, full=True), nm), names, a.workers, a.out)
    elif a.cmd == "fetch":
        with open(a.catalog) as fh:
            cat = {r["name"]: r for r in map(json.loads, fh)}
        stat = os.path.join(ROOT, os.path.basename(STAT_URL))
        if not os.path.exists(stat):
            urllib.request.urlretrieve(STAT_URL, stat)
        known = known_objectives()
        man_path = os.path.join(ROOT, "manifest.json")
        man = {e["name"]: e for e in (json.load(open(man_path)) if os.path.exists(man_path) else [])}
        for nm in a.names.split(","):
            url = URL + nm + ".cbf.gz"
            dest = os.path.join(ROOT, nm + ".cbf.gz")
            if not os.path.exists(dest):
                urllib.request.urlretrieve(url, dest + ".part")
                os.replace(dest + ".part", dest)
            c = cat.get(nm, {})
            primal, dual = known.get(nm, (None, None))
            man[nm] = {"name": nm, "url": url, "file": os.path.basename(dest), "n": c.get("n"), "m": c.get("m"),
                       "nnz": c.get("nnz"), "n_soc": c.get("n_soc"), "known_obj": primal, "known_dual_obj": dual,
                       "cones": sorted(set(c.get("var_cones", [])) | set(c.get("con_cones", [])))}
        with open(man_path, "w") as fh:
            json.dump(sorted(man.values(), key=lambda e: e["nnz"] or 0), fh, indent=1)
    elif a.cmd == "validate":
        from cblib_clarabel_runner import run_clarabel, dual_certificate
        rows = []
        for e in list_instances():
            rec = {"name": e["name"], "nnz": e.get("nnz"), "known_obj": e.get("known_obj"),
                   "known_dual_obj": e.get("known_dual_obj")}
            try:
                t0 = time.perf_counter()
                path = instance_path(e["name"])
                try:
                    d, rec["f_format"] = load_cbf(path), "dense"
                except MemoryError:
                    d, rec["f_format"] = load_cbf(path, dense_f=False), "sparse"
                rec.update(load_s=time.perf_counter() - t0, n=d["q"].size, m_lin=d["A"].shape[0], n_soc=len(d["soc"]),
                           soc_rows=int(sum(1 + c["F"].shape[0] for c in d["soc"])))
                if rec["f_format"] == "dense":
                    # both representations must describe the same cones, or the sparse path is unvalidated
                    ds = load_cbf(path, dense_f=False)
                    rec["sparse_f_equal"] = all(np.array_equal(a["f"], b["f"].toarray()) and a["h"] == b["h"]
                                                for a, b in zip(d["soc"], ds["soc"]))
                    del ds
            except Exception as ex:
                rec["parse_error"] = repr(ex)
                rows.append(rec)
                print(json.dumps(rec), flush=True)
                continue
            r = run_clarabel(d, tol=a.tol, timeout_s=a.timeout, scratch=a.scratch)
            if "error" in r:
                rec["solver_error"] = r["error"]
            else:
                x = r.pop("x")
                cert = dual_certificate(d, r.pop("y"))
                # dual_obj is in the minimized sense; obj_sign maps it back like the primal objective
                rec.update(dual_obj=d["obj_sign"] * cert["dual_obj"], dual_res=cert["dual_res"])
                obj = objective(d, x)
                lin, soc = violation(d, x)
                lin_rel, soc_rel = violation(d, x, relative=True)
                rec.update({k: v for k, v in r.items()}, obj=obj, lin_viol=lin, soc_viol=soc, lin_viol_rel=lin_rel,
                           soc_viol_rel=soc_rel,
                           x_inf=float(np.max(np.abs(x), initial=0.0)))
                if rec["known_obj"] is not None:
                    # the published pair is MOSEK's primal and dual objective, each with its own certificate error,
                    # so the comparison is against both rather than a single "optimal value"
                    rec["rel_diff"] = abs(obj - rec["known_obj"]) / max(1.0, abs(rec["known_obj"]))
                    rec["rel_diff_dual"] = abs(obj - rec["known_dual_obj"]) / max(1.0, abs(rec["known_dual_obj"]))
            rows.append(rec)
            print(json.dumps(rec), flush=True)
        with open(a.out, "w") as fh:
            json.dump(rows, fh, indent=1)


if __name__ == "__main__":
    _main()
