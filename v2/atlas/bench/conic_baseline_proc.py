"""Conic baselines (Clarabel, SCS) in their own interpreter: the second-order cone sibling of baseline_proc.py.

baseline_proc.py only speaks the QP form l <= Ax <= u. A second-order cone program needs the conic standard form

    min 1/2 x'Px + q'x   s.t.   A x + s = b,   s in {0}^nz x R_+^nl x Q^{d_1} x ... x Q^{d_K},
    Q^d = {(t, z) in R x R^{d-1} : ||z||_2 <= t},

which both Clarabel and SCS accept with the same cone order and the same SOC convention (head first). The instance
is converted HERE, in the parent, so the child needs only numpy, scipy and the solver: the child never imports
atlas or jax, which is what lets it run from an isolated baseline environment. The interpreter is located by the
same rule as baseline_proc.py (OPTEVOLVE_BASELINE_ENV_<SOLVER>, then OPTEVOLVE_BASELINE_ENV_ROOT/venv_<solver>,
then the current interpreter), reused from there rather than copied so the two runners cannot drift apart.

Conic instance dict (see scripts/discovery/unified/socp.py): the QP fields P, q, r, A, l, u for LINEAR rows plus
"cones", a list of {F (sparse, m_k x n), g (m_k,), f (n,) dense or sparse 1 x n, h (scalar)} meaning
||F x + g||_2 <= f'x + h. A sparse f matters at scale: facility location has one cone per point and a dense f per
cone is O(K n) memory (80 GB at 100,000 points).

Timing: baseline_proc.solve builds its conic form inside the child's clock. Here the conversion runs in the parent,
so its time is measured there and added: t_s = t_convert_s + t_solver_s, the same charge as baseline_proc.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np
import scipy.sparse as sp

SOLVERS = ("clarabel", "scs", "ecos")


def _is_csr64(M, shape):
    return sp.issparse(M) and M.format == "csr" and M.dtype == np.float64 and M.shape == shape


def frow(f, n: int):
    """A cone's f as a 1 x n CSR row, from a dense vector or any sparse shape holding n entries (no copy if already)."""
    if _is_csr64(f, (1, n)):
        return f
    if sp.issparse(f):
        return sp.csr_matrix(sp.csr_matrix(f, dtype=np.float64).reshape(1, n))
    return sp.csr_matrix(np.asarray(f, dtype=np.float64).reshape(1, n))


def stack_cones(cones, n: int):
    """Rows [f_1'; F_1; ...; f_K'; F_K] as one CSR matrix, offsets [h_1; g_1; ...] and the cone sizes 1 + m_k.

    The cost is per cone, not per nonzero: every sparse-format conversion is a Python-level object. Measured on
    facility_location at 100,000 cones, converting f and F for each cone took 5.2 s whether the blocks were then
    joined by scipy.sparse.vstack or by concatenating CSR arrays. Using blocks already in CSR float64 as they are and
    converting an F object shared by many cones (MEB, facility location) once brought it to 0.43 s. This runs on
    both sides of a comparison (our formulation build and the baseline's conversion), so it is charged to both.
    """
    data, indices, counts, off, dims = [], [], [], [], []
    shared = {}
    for c in cones:
        f = frow(c["f"], n)
        F = c["F"]
        if not _is_csr64(F, (F.shape[0], n)):
            key = id(F)
            if key not in shared:
                shared[key] = (F, sp.csr_matrix(F, dtype=np.float64))   # keep F alive so its id is not reused
            F = shared[key][1]
        if F.shape[1] != n:
            raise ValueError("cone F has the wrong number of columns")
        data += [f.data[:f.indptr[-1]], F.data[:F.indptr[-1]]]         # storage may be longer than nnz
        indices += [f.indices[:f.indptr[-1]], F.indices[:F.indptr[-1]]]
        counts += [np.array([f.nnz]), np.diff(F.indptr)]
        off += [np.array([float(c["h"])]), np.asarray(c["g"], dtype=np.float64).ravel()]
        dims.append(1 + F.shape[0])
    if not cones:
        return sp.csr_matrix((0, n)), np.zeros(0), []
    counts = np.concatenate(counts)
    indptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    M = sp.csr_matrix((np.concatenate(data), np.concatenate(indices), indptr), shape=(counts.size, n))
    return M, np.concatenate(off), dims


def to_clarabel(d: dict) -> dict:
    """Conic standard form of a conic instance dict. Returns P (upper triangle), q, r, A, b and the cone sizes.

    Linear rows follow baseline_proc.py: equalities l = u as A x + s = u with s in the zero cone, finite upper bounds
    as A x + s = u and finite lower bounds as -A x + s = -l with s >= 0 (a two-sided row appears in both). A cone
    ||F x + g|| <= f'x + h is s = (f'x + h, F x + g) in Q^{1+m}, i.e. rows -[f'; F] and right-hand side [h; g].
    """
    A = sp.csr_matrix(d["A"]).astype(np.float64)
    n = sp.csc_matrix(d["P"]).shape[0]
    l = np.asarray(d["l"], dtype=np.float64)
    u = np.asarray(d["u"], dtype=np.float64)
    eq = np.isfinite(l) & np.isfinite(u) & (l == u)
    up = np.isfinite(u) & ~eq
    lo = np.isfinite(l) & ~eq
    M, off, soc_dims = stack_cones(d.get("cones", []), n)
    Ac = sp.vstack([A[eq], A[up], -A[lo], -M], format="csc")
    return {"P": sp.triu(sp.csc_matrix(d["P"]).astype(np.float64), format="csc"), "q": np.asarray(d["q"], float),
            "r": float(d.get("r", 0.0)), "A": sp.csc_matrix(Ac), "b": np.concatenate([u[eq], u[up], -l[lo], off]),
            "n_zero": int(eq.sum()), "n_nonneg": int(up.sum() + lo.sum()), "soc_dims": np.asarray(soc_dims, int)}


def dump_conic(c: dict, path: str) -> None:
    P, A = sp.csc_matrix(c["P"]), sp.csc_matrix(c["A"])
    np.savez(path, Pd=P.data, Pi=P.indices, Pp=P.indptr, Ad=A.data, Ai=A.indices, Ap=A.indptr,
             q=np.asarray(c["q"]), b=np.asarray(c["b"]), soc_dims=np.asarray(c["soc_dims"], dtype=np.int64),
             sizes=np.array([P.shape[0], A.shape[0], c["n_zero"], c["n_nonneg"]], dtype=np.int64))


def load_conic(path: str) -> dict:
    z = np.load(path)
    n, m, nz, nl = (int(v) for v in z["sizes"])
    return {"P": sp.csc_matrix((z["Pd"], z["Pi"], z["Pp"]), shape=(n, n)),
            "A": sp.csc_matrix((z["Ad"], z["Ai"], z["Ap"]), shape=(m, n)),
            "q": z["q"], "b": z["b"], "n_zero": nz, "n_nonneg": nl, "soc_dims": [int(k) for k in z["soc_dims"]]}


def solve(c: dict, solver: str, eps: float, timeout_s: float):
    """Solve the standard form in THIS process. Import before the clock, clock before setup (baseline_proc contract)."""
    import importlib
    importlib.import_module(solver)
    t0 = time.perf_counter()
    if solver == "clarabel":
        import clarabel
        cones = ([clarabel.ZeroConeT(c["n_zero"])] if c["n_zero"] else []) \
            + ([clarabel.NonnegativeConeT(c["n_nonneg"])] if c["n_nonneg"] else []) \
            + [clarabel.SecondOrderConeT(k) for k in c["soc_dims"]]
        cfg = clarabel.DefaultSettings()
        cfg.verbose = False
        cfg.tol_gap_abs = cfg.tol_gap_rel = cfg.tol_feas = eps
        cfg.time_limit = timeout_s
        sol = clarabel.DefaultSolver(c["P"], c["q"], c["A"], c["b"], cones, cfg).solve()
        x, z, st, it = np.array(sol.x), np.array(sol.z), str(sol.status), int(sol.iterations)
    elif solver == "scs":
        import scs
        sol = scs.SCS({"P": c["P"], "A": c["A"], "b": c["b"], "c": c["q"]},
                      {"z": c["n_zero"], "l": c["n_nonneg"], "q": list(c["soc_dims"])},
                      eps_abs=eps, eps_rel=eps, time_limit_secs=timeout_s, verbose=False).solve()
        x, z, st, it = np.asarray(sol["x"]), np.asarray(sol["y"]), str(sol["info"]["status"]), int(sol["info"]["iter"])
    elif solver == "ecos":
        # ECOS speaks  min c'x  s.t.  Ax = b,  Gx + s = h,  s in R_+^l x Q^{q_1} x ...  and has no quadratic term.
        # Our standard form stacks rows as [zero cone; nonneg; SOC], so the split is a slice, not a rebuild: the
        # leading n_zero rows are the equalities and the rest are already in ECOS's cone order.
        import ecos
        nz, nl = c["n_zero"], c["n_nonneg"]
        A = sp.csc_matrix(c["A"])
        G = sp.csc_matrix(A[nz:])
        h = np.asarray(c["b"], float)[nz:]
        dims = {"l": int(nl), "q": [int(k) for k in c["soc_dims"]]}
        kw = dict(verbose=False, feastol=eps, abstol=eps, reltol=eps)
        if nz:
            sol = ecos.solve(np.asarray(c["q"], float), G, h, dims, sp.csc_matrix(A[:nz]),
                             np.asarray(c["b"], float)[:nz], **kw)
        else:
            sol = ecos.solve(np.asarray(c["q"], float), G, h, dims, **kw)
        x = np.asarray(sol["x"], float)
        z = np.concatenate([np.asarray(sol.get("y", np.zeros(nz)), float).ravel(),
                            np.asarray(sol["z"], float).ravel()])
        st, it = str(sol["info"]["exitFlag"]), int(sol["info"]["iter"])
    else:
        raise ValueError(f"unknown conic baseline solver {solver!r}")
    return {"t_s": time.perf_counter() - t0, "x": x, "z": z, "status": st, "iters": it}


def run_conic_baseline_proc(d: dict, solver: str, eps: float, timeout_s: float = 600.0,
                            scratch: str | None = None):
    """Solve conic instance `d` with `solver` in its isolated interpreter. Errors come back as records, not raises."""
    from atlas.bench.baseline_proc import _env_python
    py = _env_python(solver)
    tmpdir = scratch or tempfile.gettempdir()
    with tempfile.NamedTemporaryFile(suffix=".npz", dir=tmpdir, delete=False) as fh:
        inst = fh.name
    out = inst + ".out.npz"
    try:
        if solver == "ecos" and sp.csc_matrix(d["P"]).nnz:
            # Not a failure to report as a loss: ECOS has no quadratic term, so this class is outside it.
            return {"unsupported": True, "error": "ecos accepts no quadratic objective", "solver_python": py}
        c0 = time.perf_counter()
        conic = to_clarabel(d)
        t_convert = time.perf_counter() - c0
        dump_conic(conic, inst)
        cmd = [py, os.path.abspath(__file__), inst, solver, repr(float(eps)), repr(float(timeout_s)), out]
        env = dict(os.environ, JAX_PLATFORMS="cpu")
        pr = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s + float(os.environ.get("OPTEVOLVE_BASELINE_HARD_MARGIN", "120")), env=env)
        if pr.returncode != 0:
            return {"error": (pr.stderr or pr.stdout).strip()[:400], "solver_python": py}
        rec = json.loads(pr.stdout.strip().splitlines()[-1])
        rec["t_solver_s"], rec["t_convert_s"] = rec["t_s"], t_convert
        rec["t_s"] = rec["t_solver_s"] + t_convert
        z = np.load(out)
        rec["x"], rec["z"] = z["x"], z["z"]
        rec["solver_python"] = py
        return rec
    except subprocess.TimeoutExpired:
        return {"error": f"timeout after {timeout_s}s", "solver_python": py}
    finally:
        for p in (inst, out):
            try:
                os.unlink(p)
            except OSError:
                pass


if __name__ == "__main__":
    inst, solver, eps, tmo, out = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
    r = solve(load_conic(inst), solver, eps, tmo)
    np.savez(out, x=np.asarray(r.pop("x")), z=np.asarray(r.pop("z")))
    print(json.dumps(r))
