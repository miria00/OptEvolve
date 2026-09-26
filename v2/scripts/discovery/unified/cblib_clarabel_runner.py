"""Clarabel on a conic (linear + second-order cone) instance, run in the baseline's isolated interpreter.

Same split as atlas/bench/baseline_proc.py, which only handles linear cones: the parent dumps the conic dict to an
.npz and launches this file under the interpreter baseline_proc._env_python("clarabel") selects; the child needs only
numpy, scipy and clarabel, so it never imports the pipeline. The child reports Clarabel's point and status; the parent
judges that point against the LOADED constraints, so a parse error cannot hide behind Clarabel agreeing with itself.
"""
import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np
import scipy.sparse as sp


def _csc_parts(prefix, M):
    M = sp.csc_matrix(M)
    return {prefix + "d": M.data, prefix + "i": M.indices, prefix + "p": M.indptr,
            prefix + "shape": np.array(M.shape)}


def _csc_load(z, prefix):
    return sp.csc_matrix((z[prefix + "d"], z[prefix + "i"], z[prefix + "p"]), shape=tuple(z[prefix + "shape"]))


def stack_soc(d):
    """Every SOC as rows [f'; F] with constants [h; g], so s = [h; g] - (-[f'; F]) x lives in one matrix instead of
    thousands of small ones. Assembled from triplets because vstack over tens of thousands of tiny matrices is the slow
    part on the chainsing/nql instances. f may be dense or 1-D sparse. Returns (G csc, hg, dims)."""
    n = d["q"].size
    I, J, V, consts, dims = [], [], [], [], []
    row = 0
    for c in d["soc"]:
        f = c["f"]
        if sp.issparse(f):
            fc = sp.coo_array(f)
            fj, fv = fc.coords[0], fc.data
        else:
            fj = np.flatnonzero(f)
            fv = np.asarray(f)[fj]
        Fc = sp.coo_matrix(c["F"])
        I += [np.full(fj.size, row), Fc.row + row + 1]
        J += [fj, Fc.col]
        V += [fv, Fc.data]
        consts += [[c["h"]], np.asarray(c["g"], dtype=float)]
        dims.append(1 + Fc.shape[0])
        row += 1 + Fc.shape[0]
    cat = lambda xs, dt: np.concatenate(xs).astype(dt) if xs else np.zeros(0, dt)
    G = sp.csc_matrix((cat(V, float), (cat(I, np.int64), cat(J, np.int64))), shape=(row, n))
    return G, cat(consts, float), np.array(dims, dtype=np.int64)


def standard_form(A, l, u, G, hg):
    """Clarabel's A x + s = b, s in K: equality rows (zero cone), then u-rows and l-rows (nonnegative cone), then the
    stacked SOCs. Shared by the child (to build the solve) and the parent (to check the dual certificate)."""
    eqm = np.isfinite(l) & np.isfinite(u) & (l == u)
    up, lo = np.isfinite(u) & ~eqm, np.isfinite(l) & ~eqm
    Ac = sp.vstack([A[eqm], A[up], -A[lo], -G], format="csc")
    bc = np.concatenate([u[eqm], u[up], -l[lo], hg])
    return Ac, bc, int(eqm.sum()), int(up.sum() + lo.sum())


def dump_conic(d, path):
    G, hg, dims = stack_soc(d)
    np.savez(path, q=d["q"], r=np.array(d["r"]), l=d["l"], u=d["u"], hg=hg, dims=dims,
             **_csc_parts("P", d["P"]), **_csc_parts("A", d["A"]), **_csc_parts("G", G))


def solve(path, tol, timeout_s):
    import clarabel
    z = np.load(path)
    P, A, G = _csc_load(z, "P"), _csc_load(z, "A"), _csc_load(z, "G")
    Ac, bc, nz, nl = standard_form(A, z["l"], z["u"], G, z["hg"])
    cones = ([clarabel.ZeroConeT(nz)] if nz else []) + ([clarabel.NonnegativeConeT(nl)] if nl else [])
    cones += [clarabel.SecondOrderConeT(int(k)) for k in z["dims"]]
    cfg = clarabel.DefaultSettings()
    cfg.verbose = False
    cfg.tol_gap_abs = cfg.tol_gap_rel = cfg.tol_feas = tol
    cfg.time_limit = timeout_s
    t0 = time.perf_counter()
    sol = clarabel.DefaultSolver(sp.triu(P, format="csc"), z["q"], Ac, bc, cones, cfg).solve()
    return {"t_s": time.perf_counter() - t0, "status": str(sol.status), "iters": int(sol.iterations),
            "obj_clarabel": float(sol.obj_val) + float(z["r"])}, np.array(sol.x), np.array(sol.z)


def dual_certificate(d, y):
    """Independent check of Clarabel's multiplier y for  min q'x + r  s.t.  Ac x + s = bc, s in K  (P is zero for CBF
    instances): the dual objective -bc'y + r and the residual ||q + Ac'y||_inf after projecting y onto K (K is
    self-dual). When the residual is tiny, -bc'y + r is a lower bound on every feasible objective, so it certifies (or
    refutes) a published optimal value without trusting any solver's own status."""
    G, hg, dims = stack_soc(d)
    Ac, bc, nz, nl = standard_form(d["A"], d["l"], d["u"], G, hg)
    y = np.array(y, dtype=float)
    y[nz:nz + nl] = np.maximum(y[nz:nz + nl], 0.0)
    off = nz + nl
    for k in dims:
        t, w = y[off], y[off + 1:off + k]
        nw = np.linalg.norm(w)
        if nw > t:                                 # SOC projection: onto the boundary ray, or to zero if below -||w||
            a = max((t + nw) / 2.0, 0.0)
            y[off], y[off + 1:off + k] = a, w * (a / nw)
        off += k
    res = d["q"] + Ac.T @ y
    return {"dual_obj": float(-bc @ y + d["r"]), "dual_res": float(np.max(np.abs(res), initial=0.0))}


def run_clarabel(d, tol=1e-10, timeout_s=600.0, scratch=None):
    """Solve the conic dict `d` with Clarabel in its isolated interpreter. Returns {status, iters, t_s, obj_clarabel,
    x, y (multipliers in standard_form order), solver_python} or {error, solver_python}."""
    v2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
    sys.path.insert(0, os.path.abspath(v2))
    from atlas.bench.baseline_proc import _env_python
    py = _env_python("clarabel")
    with tempfile.NamedTemporaryFile(suffix=".npz", dir=scratch or tempfile.gettempdir(), delete=False) as fh:
        inst = fh.name
    out = inst + ".out.npz"
    try:
        dump_conic(d, inst)
        cmd = [py, os.path.abspath(__file__), inst, repr(float(tol)), repr(float(timeout_s)), out]
        pr = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s + 300,
                            env=dict(os.environ, JAX_PLATFORMS="cpu"))
        if pr.returncode != 0:
            return {"error": (pr.stderr or pr.stdout).strip()[-400:], "solver_python": py}
        rec = json.loads(pr.stdout.strip().splitlines()[-1])
        zz = np.load(out)
        rec["x"], rec["y"] = zz["x"], zz["y"]
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
    inst, tol, tmo, out = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
    rec, x, y = solve(inst, tol, tmo)
    np.savez(out, x=x, y=y)
    print(json.dumps(rec))
