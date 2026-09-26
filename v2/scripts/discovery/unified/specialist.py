"""Recognize a QP as the encoding of a named statistical problem, and route it to a solver for THAT problem.

Why this exists. Our lasso family is a QP in $[x; y; t]$ with $y = Ax - b$ and $-t \\le x \\le t$. Every solver the
router could previously reach (Clarabel, PIQP, HiGHS, SCS, OSQP, and our own operator) treats that QP as an opaque
convex program with $2n + m$ variables. A coordinate-descent lasso solver instead works on the $n$-dimensional
statistical problem and touches only the active support, which on these instances is 91 of 10,000 columns. Measured
on the identical instance (2026-09-15, 4090): scikit-learn 0.048 s at $n = 100{,}000$ against our 17.6 s, because the
support is 2,807 columns. No amount of operator synthesis closes a gap that comes from solving a larger problem than
the one that was asked.

So the fix is not a better operator, it is recognition. The detector below reads the block structure of $(P, q, A, l,
u)$ and decides whether the QP IS a lasso encoding. It never looks at the instance name or the generator that made
it, because the claim we want to support is that the vault recognizes formulations, not benchmarks.

Two rules keep this honest:

  * the specialist's answer is lifted back to the QP variable $z = [x;\\ Ax - b;\\ |x|]$ and judged by the SAME
    `problem_adapters.accept` check as every other candidate, on the original problem, at the same tolerance. A
    specialist gets no credit for reporting convergence in its own objective.
  * detection is conservative. A structure we cannot verify returns None and the router proceeds exactly as before,
    so a false negative costs a routing opportunity and never a wrong answer.
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

# Interpreter holding celer / skglm / scikit-learn. Isolated from ours for the same reason the baselines are: the
# solver environment must never acquire a competitor's package.
SPECIALIST_ENV = os.environ.get("OPTEVOLVE_SPECIALIST_ENV", "/home/miria/enetenv/bin/python")
SPECIALISTS = ("celer", "skglm", "sklearn")


def detect_lasso(d, atol=1e-9):
    """Return {A, b, lam, n, m, ...} if this QP IS a lasso encoding under ANY permutation, else None.

    Permutation invariance is required, not a refinement: the router permutes the rows and columns of every instance
    it builds (`uni.permute_instance`) precisely so that nothing can recognize a family by index position. A detector
    that keyed on block order would fire on the generator's output and never once on a routed case, which would make
    the whole result an artifact of how the benchmark is written down.

    So the test is on structure alone. Variables are classified by the pair (diag(P), q): the residual block y is
    where P is 1 and q is 0, the penalty block t is where P is 0 and q is a single positive constant, the free block
    x is where both are 0. Rows are classified by their bounds: m equalities, n rows bounded above by 0, n bounded
    below by 0. Then each row must have exactly the incidence pattern the encoding forces (an equality row touches
    exactly one y with coefficient -1 and no t; a one-sided row touches exactly one x with +1 and exactly one t with
    -1 or +1 and no y), and the x-to-t pairing read off the upper rows must agree with the one read off the lower
    rows. Any mismatch returns None, which costs a routing opportunity and never an answer.

    All of it is vectorized over nonzeros: the check is one pass over A, so a detector that fires on nothing still
    costs about what one sparse matrix-vector product costs.
    """
    P, q, A = d["P"], np.asarray(d["q"], float), sp.csr_matrix(d["A"])
    l, u = np.asarray(d["l"], float), np.asarray(d["u"], float)
    N = P.shape[0]
    P = sp.csr_matrix(P)
    dg = np.asarray(P.diagonal(), float)
    if P.nnz != int((dg != 0).sum()):
        return None                                        # P must be diagonal: an off-diagonal is a different model
    S = np.flatnonzero(dg != 0)
    m = S.size
    if m == 0 or not np.allclose(dg[S], 1.0, atol=atol) or (N - m) % 2:
        return None
    n = (N - m) // 2
    if n == 0 or A.shape[0] != m + 2 * n:
        return None
    if not np.allclose(q[S], 0.0, atol=atol):
        return None
    T = np.flatnonzero(q > 0)
    if T.size != n:
        return None
    lam = float(q[T[0]])
    if lam <= 0 or not np.allclose(q[T], lam, atol=1e-9 * max(1.0, lam)):
        return None                                        # one penalty weight shared by every coordinate
    cls = np.zeros(N, dtype=np.int8)                       # 0 = x (free), 1 = y (residual), 2 = t (penalty)
    cls[S] = 1
    cls[T] = 2
    X = np.flatnonzero(cls == 0)
    if X.size != n or not np.allclose(q[X], 0.0, atol=atol) or np.any(dg[X] != 0) or np.any(dg[T] != 0):
        return None

    fin = np.isfinite(l) & np.isfinite(u)
    E = np.flatnonzero(fin & (np.abs(u - l) <= atol))
    B1 = np.flatnonzero(np.isneginf(l) & np.isfinite(u) & (np.abs(u) <= atol))     # x - t <= 0
    B2 = np.flatnonzero(np.isposinf(u) & np.isfinite(l) & (np.abs(l) <= atol))     # x + t >= 0
    if E.size != m or B1.size != n or B2.size != n or E.size + B1.size + B2.size != A.shape[0]:
        return None

    nr = A.shape[0]
    rows = np.repeat(np.arange(nr), np.diff(A.indptr))
    ccls = cls[A.indices]
    # bincount, not np.add.at: the unbuffered add is a Python-level scatter and cost 0.18 s on a 1.5M-nonzero lasso,
    # more than celer then took to solve it, which would have made recognition the dominant term in its own win.
    cnt = np.bincount(rows * 3 + ccls, minlength=3 * nr).reshape(nr, 3)
    if not (np.all(cnt[E, 1] == 1) and np.all(cnt[E, 2] == 0)
            and np.all(cnt[B1, 0] == 1) and np.all(cnt[B1, 1] == 0) and np.all(cnt[B1, 2] == 1)
            and np.all(cnt[B2, 0] == 1) and np.all(cnt[B2, 1] == 0) and np.all(cnt[B2, 2] == 1)):
        return None
    rowset = np.zeros(nr, np.int8)                         # membership as a lookup, not a repeated np.isin search
    rowset[E], rowset[B1], rowset[B2] = 1, 2, 3

    def single(mask_cls, want_tag, k, sign):
        """For rows tagged `want_tag`, the unique column of class `mask_cls`, checking its coefficient is `sign`.
        Rows come out in increasing row index, so the result is indexed by rank within the sorted tag set."""
        sel = (ccls == mask_cls) & (rowset[rows] == want_tag)
        c, v = A.indices[sel], A.data[sel]
        if c.size != k or not np.allclose(v, sign, atol=atol):
            return None
        return c                                           # CSR order is already row-major, so already sorted by row

    Es = np.sort(E)
    y_of_row = single(1, 1, m, -1.0)
    x_of_up, t_of_up = single(0, 2, n, 1.0), single(2, 2, n, -1.0)
    x_of_lo, t_of_lo = single(0, 3, n, 1.0), single(2, 3, n, 1.0)
    if any(v is None for v in (y_of_row, x_of_up, t_of_up, x_of_lo, t_of_lo)):
        return None
    if (np.unique(y_of_row).size != m or np.unique(x_of_up).size != n or np.unique(t_of_up).size != n
            or np.unique(x_of_lo).size != n):
        return None
    pair_up, pair_lo = np.full(N, -1, np.int64), np.full(N, -1, np.int64)
    pair_up[x_of_up] = t_of_up
    pair_lo[x_of_lo] = t_of_lo
    if not np.array_equal(pair_up[X], pair_lo[X]) or np.any(pair_up[X] < 0):
        return None                                        # the two sides must pair the same x with the same t

    cols = np.sort(X)
    Asub = sp.csc_matrix(A[Es][:, cols])
    b = u[Es].copy()
    return {"form": "lasso", "A": Asub, "b": b, "lam": lam, "n": n, "m": m,
            "cols": cols, "rows": Es, "y_of_row": y_of_row, "t_of_col": pair_up[cols], "N": N}


def lift(x, det):
    """Lift a statistical solution back to the QP variable, into the positions the permutation put the blocks in.

    Exact by construction: $y = Ax - b$ satisfies every equality row and $t = |x|$ satisfies both one-sided rows, so
    the lift contributes no feasibility error of its own and whatever `accept` then reports is the specialist's own
    optimality gap on the original problem.
    """
    x = np.asarray(x, float).ravel()
    z = np.zeros(det["N"], float)
    z[det["cols"]] = x
    z[det["y_of_row"]] = det["A"] @ x - det["b"]
    z[det["t_of_col"]] = np.abs(x)
    return z


def run_specialist(d, name, obj_ref, tol, timeout_s, det=None):
    """Run `name` on the recognized statistical problem out of process, lift, and judge on the original QP.

    Returns the record shape `problem_adapters.run_engine` returns, so the router scores a specialist against a
    generic solver with no special case: `wall_s` is what a first visit pays, `t_solver_s` the routed cost.
    """
    import problem_adapters as PA
    t0 = time.perf_counter()
    det = det or detect_lasso(d)
    if det is None:
        return {"accepted": False, "error": "structure not recognized", "wall_s": time.perf_counter() - t0}
    tmp = tempfile.mkdtemp(prefix="spec_")
    try:
        sp.save_npz(os.path.join(tmp, "A.npz"), det["A"])
        np.savez(os.path.join(tmp, "meta.npz"), b=det["b"], lam=det["lam"])
        cmd = [SPECIALIST_ENV, os.path.join(os.path.dirname(os.path.abspath(__file__)), "specialist_proc.py"),
               "--dir", tmp, "--solver", name, "--tol", repr(float(tol))]
        env = dict(os.environ)
        env.pop("LD_LIBRARY_PATH", None)                  # the local CUDA path makes numpy pick a wrong libgomp
        pr = subprocess.run(cmd, capture_output=True, text=True, timeout=max(1.0, timeout_s), env=env)
        if pr.returncode != 0:
            return {"accepted": False, "error": (pr.stderr or pr.stdout)[-200:], "wall_s": time.perf_counter() - t0}
        rb = json.loads(pr.stdout.strip().splitlines()[-1])
        x = np.load(os.path.join(tmp, "x.npy"))
        ok, rv, gap = PA.accept(lift(x, det), d, obj_ref, tol)
        return {"accepted": bool(ok), "t_solver_s": rb["t_s"], "eps": rb["tol"], "rv": rv, "gap": gap,
                "support": rb.get("nnz_x"), "wall_s": time.perf_counter() - t0}
    except subprocess.TimeoutExpired:
        return {"accepted": False, "error": "timeout", "wall_s": time.perf_counter() - t0}
    except Exception as e:
        return {"accepted": False, "error": repr(e)[:200], "wall_s": time.perf_counter() - t0}
    finally:
        for f in ("A.npz", "meta.npz", "x.npy"):
            try:
                os.remove(os.path.join(tmp, f))
            except OSError:
                pass
        try:
            os.rmdir(tmp)
        except OSError:
            pass


if __name__ == "__main__":
    # Detector self-check. Three things have to hold before this is allowed to change any routing decision:
    #   1. it fires on lasso and on no other family (a false positive would send a different problem to a solver that
    #      cannot represent it);
    #   2. it fires on the PERMUTED encoding, which is the only form the router ever sees;
    #   3. the recovered (A, b, lam) and the lift reproduce the original QP objective exactly, so a specialist's
    #      answer is judged on the same number every other candidate is judged on.
    sys.path.insert(0, "/home/miria/OptEvolve/v2")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from atlas.bench.qp_gen import FAMILIES
    from uni import permute_instance
    for fam in ("lasso", "huber", "svm", "portfolio", "control"):
        try:
            dd = FAMILIES[fam](2000, seed=0)
        except Exception as e:
            print(f"{fam:10s} skipped ({e})")
            continue
        raw, perm = detect_lasso(dd), detect_lasso(permute_instance(dd, 7))
        line = f"{fam:10s} canonical={raw is not None} permuted={perm is not None}"
        if perm is not None:
            rng = np.random.default_rng(3)
            xs = rng.standard_normal(perm["n"]) * (rng.random(perm["n"]) < 0.05)
            dp = permute_instance(dd, 7)
            z = lift(xs, perm)
            qp = 0.5 * z @ (dp["P"] @ z) + np.asarray(dp["q"]) @ z
            nat = 0.5 * float(np.sum((perm["A"] @ xs - perm["b"]) ** 2)) + perm["lam"] * float(np.abs(xs).sum())
            rowv = float(np.max(np.maximum(sp.csc_matrix(dp["A"]) @ z - dp["u"],
                                           dp["l"] - sp.csc_matrix(dp["A"]) @ z)))
            line += (f" n={perm['n']} m={perm['m']} lam={perm['lam']:.5g}"
                     f" objective_match={abs(qp - nat) <= 1e-8 * (1 + abs(nat))} lift_violation={rowv:.2e}")
        print(line)
