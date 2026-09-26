"""Convex QP cell: the Maros-Meszaros test set as composite problems.

    min (1/2) x'Px + q'x + r   s.t.   l <= Ax <= u

Data: the qpsolvers/maros_meszaros_qpbenchmark .mat files (keys n, m, P, q, r,
A, l, u), which fold variable bounds into A; this is OSQP's native form. As a
composite: f = QuadraticForm(P, q, r), g = 0, h = indicator of [l, u], L = A.
Bounds with magnitude >= 1e20 are the set's encoding of infinity.
"""
from __future__ import annotations

import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import jax.numpy as jnp

from atlas.bench.mps_cell import SparseLinOp
from atlas.ir.spec import CompositeProblem
from atlas.operators.prox import Box, PropSet, QuadraticForm, Zero

INF = 1e20


def read_mm(path: str) -> dict:
    m = sio.loadmat(path)
    P = sp.csc_matrix(m["P"]).astype(np.float64)
    # Some distributions store only one triangle; symmetrise without doubling the diagonal.
    if abs(P - P.T).max() > 1e-12 * max(1.0, abs(P).max()):
        tri = sp.triu(P) if sp.tril(P, -1).nnz == 0 else sp.tril(P)
        P = (tri + tri.T - sp.diags(tri.diagonal())).tocsc()
    l = np.asarray(m["l"], dtype=np.float64).ravel().copy()
    u = np.asarray(m["u"], dtype=np.float64).ravel().copy()
    l[l <= -INF] = -np.inf
    u[u >= INF] = np.inf
    return {"P": P, "q": np.asarray(m["q"], dtype=np.float64).ravel(),
            "r": float(np.asarray(m.get("r", 0.0)).ravel()[0]) if np.asarray(m.get("r", 0.0)).size else 0.0,
            "A": sp.csc_matrix(m["A"]).astype(np.float64), "l": l, "u": u}


def to_composite(d: dict, name: str = "qp_standard",
                 power_iters: int | None = None,
                 gpu_threshold: int | None = None) -> CompositeProblem:
    # power_iters threads down to the norm-bound power iteration, which is 74% of the rescale
    # cost on lasso n_var 50,000. None keeps the historical 500.
    P_op = (SparseLinOp.from_scipy(d["P"], power_iters=power_iters, gpu_threshold=gpu_threshold)
            if d["P"].nnz else None)
    A_op = SparseLinOp.from_scipy(d["A"], power_iters=power_iters, gpu_threshold=gpu_threshold)
    Lf = float(P_op.norm_bound) if P_op is not None else 0.0
    if P_op is None:
        raise ValueError("P is empty: this instance is an LP; use lp_cell")
    f = QuadraticForm(P=P_op, q=jnp.asarray(d["q"]), r=d["r"],
                      props=PropSet(lipschitz=Lf, cocoercivity=1.0 / Lf,
                                    prox_friendly=False, strong_convexity=0.0))
    return CompositeProblem(
        f=f, g=Zero(), h=Box(lo=jnp.asarray(d["l"]), hi=jnp.asarray(d["u"])), L=A_op,
        data={"P": d["P"], "q": d["q"], "r": d["r"], "A": d["A"], "l": d["l"], "u": d["u"]},
        shape=(d["P"].shape[0],), name=name)


def load(path: str) -> CompositeProblem:
    return to_composite(read_mm(path))


def to_conic(d: dict):
    """l <= A x <= u as the conic form A_c x + s = b_c, s in {0}^eq x R_+^ineq (Clarabel, SCS)."""
    A, l, u = d["A"].tocsr(), d["l"], d["u"]
    eq = np.isfinite(l) & np.isfinite(u) & (l == u)
    up = np.isfinite(u) & ~eq
    lo = np.isfinite(l) & ~eq
    Ac = sp.vstack([A[eq], A[up], -A[lo]]).tocsc()
    bc = np.concatenate([u[eq], u[up], -l[lo]])
    return Ac, bc, int(eq.sum()), int(up.sum() + lo.sum()), (eq, up, lo)


def y_from_conic(yc, idx, m: int):
    """Conic duals back to one multiplier per row of A (the sign convention of qp_rel_kkt)."""
    eq, up, lo = idx; ne, nu = int(eq.sum()), int(up.sum())
    y = np.zeros(m); y[eq] = yc[:ne]; y[up] += yc[ne:ne + nu]; y[lo] -= yc[ne + nu:]
    return y


def clarabel_reference(d: dict, eps: float = 1e-9):
    """High-accuracy interior-point reference: (x, y, status string, objective)."""
    import clarabel
    Ac, bc, nz, nl, idx = to_conic(d)
    cones = ([clarabel.ZeroConeT(nz)] if nz else []) + ([clarabel.NonnegativeConeT(nl)] if nl else [])
    st = clarabel.DefaultSettings(); st.verbose = False
    st.tol_gap_abs = st.tol_gap_rel = st.tol_feas = eps
    r = clarabel.DefaultSolver(sp.triu(d["P"]).tocsc(), d["q"], Ac, bc, cones, st).solve()
    x = np.asarray(r.x)
    return x, y_from_conic(np.asarray(r.z), idx, d["A"].shape[0]), str(r.status), float(0.5 * x @ (d["P"] @ x) + d["q"] @ x + d["r"])
