"""Fully constrained least squares over a spectral library, batched.

    min_X  1/2 || D X - Y ||_F^2    s.t.  X >= 0,  1^T x_i = 1  for every pixel i

Separable across pixels, and every pixel shares the same D, so the Hessian
D^T D is formed once and the iteration is one GEMM.  The constraint set is the
unit simplex, which is a box intersected with a hyperplane: the exact
projection is the operator this project already certified.

Acceptance is NOT a solver's own stopping rule.  For each pixel we recompute
the Frank-Wolfe gap

    g(x) = grad^T x - min_j grad_j  >=  f(x) - f*,

which is a rigorous upper bound on suboptimality for a linear-minimization
oracle over the simplex, and we score the WORST pixel, relative to 1 + |f|.
Feasibility is checked separately.  Any solver's X can be graded by this.
"""
from __future__ import annotations
import numpy as np


def project_simplex_np(V):
    """Exact Euclidean projection of each COLUMN of V onto the unit simplex."""
    n, m = V.shape
    U = np.sort(V, axis=0)[::-1]
    css = np.cumsum(U, axis=0) - 1.0
    ind = np.arange(1, n + 1).reshape(-1, 1)
    cond = U - css / ind > 0
    rho = n - 1 - np.argmax(cond[::-1], axis=0)
    theta = css[rho, np.arange(m)] / (rho + 1.0)
    return np.maximum(V - theta, 0.0)


def fw_gap_and_feas(D, Y, X, DtD=None, DtY=None):
    """Worst-pixel relative Frank-Wolfe gap and worst feasibility violation."""
    if DtD is None:
        DtD = D.T @ D
    if DtY is None:
        DtY = D.T @ Y
    G = DtD @ X - DtY                      # gradient of f, per pixel
    R = D @ X - Y
    f = 0.5 * np.einsum('ij,ij->j', R, R)  # per-pixel objective
    gap = np.einsum('ij,ij->j', G, X) - G.min(axis=0)
    rel = gap / (1.0 + np.abs(f))
    feas = max(float(np.maximum(0.0, -X).max()),
               float(np.abs(X.sum(axis=0) - 1.0).max()))
    return float(rel.max()), float(np.median(rel)), feas, float(f.sum())
