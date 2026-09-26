"""REFERENCE-FREE certificate: how good is a point when no solver can produce a reference solution?

For min 1/2 x'Px + q'x subject to l <= Ax <= u (the OSQP form used throughout, variable bounds written as identity
rows), a certificate needs a dual point. Solvers return one; our relocated formulations keep multipliers for the rows
left in the linear operator and hide the absorbed ones inside a projection, so this module RECOVERS a dual from the
primal point instead, and then reports quantities that are valid for ANY dual:

  primal_rel   max_i violation_i / (1 + |bound_i|)                     (per-row, the project's acceptance measure)
  dual_rel     ||P x + q + A'y||_inf / (1 + max(||P x||_inf, ||q||_inf, ||A'y||_inf))
  gap_rel      |primal_obj - dual_obj| / (1 + |primal_obj| + |dual_obj|)
  dual_obj     the Lagrangian value at the recovered y, a LOWER BOUND on the optimum when dual_rel is zero; the
               reported `bound_slack` states how much the residual could move it, so a claim can be made honestly

The recovery solves a bound-constrained least squares for y with the sign pattern implied by complementary slackness
(y_i <= 0 on rows at their lower bound, y_i >= 0 at their upper bound, y_i = 0 on inactive rows, free on equalities).
It is not a proof of optimality: it is the same kind of test as the operator certificate, and it is reported with its
own residuals so a reader can see what it does and does not establish.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.optimize import lsq_linear


def _active_sets(Ax, l, u, tol):
    at_l = np.isfinite(l) & (Ax <= l + tol * (1 + np.abs(np.where(np.isfinite(l), l, 0.0))))
    at_u = np.isfinite(u) & (Ax >= u - tol * (1 + np.abs(np.where(np.isfinite(u), u, 0.0))))
    eq = np.isfinite(l) & np.isfinite(u) & np.isclose(l, u)
    return at_l & ~eq, at_u & ~eq, eq


def recover_dual(d, x, tol=1e-6, max_iter=200):
    """Least-squares dual with complementary-slackness signs. Rows that are inactive get y = 0 and are dropped, which
    keeps the system small: only active rows can carry a multiplier."""
    P, A = sp.csc_matrix(d["P"]), sp.csc_matrix(d["A"])
    q, l, u = np.asarray(d["q"], float), np.asarray(d["l"], float), np.asarray(d["u"], float)
    Ax = A @ x
    at_l, at_u, eq = _active_sets(Ax, l, u, tol)
    keep = at_l | at_u | eq
    idx = np.flatnonzero(keep)
    y = np.zeros(A.shape[0])
    if idx.size == 0:
        return y
    B = A[idx].T.tocsc()                                   # solve B y_act = -(P x + q)
    rhs = -(P @ x + q)
    lo = np.where(at_u[idx], 0.0, -np.inf)
    hi = np.where(at_l[idx], 0.0, np.inf)
    lo = np.where(eq[idx], -np.inf, lo)
    hi = np.where(eq[idx], np.inf, hi)
    sol = lsq_linear(B, rhs, bounds=(lo, hi), max_iter=max_iter, tol=1e-12, lsq_solver="lsmr")
    y[idx] = sol.x
    return y


def certificate(d, x, y=None, active_tol=None, tol_sweep=(1e-9, 1e-7, 1e-5, 1e-3, 1e-2)):
    """When no dual is supplied, the active set is not known exactly: a row that is nearly active can carry a
    multiplier, and labelling it inactive makes the recovery fail (measured on MM DUAL4 and LOTSCHD). Sweep the
    activity tolerance and keep the dual that certifies best; report which tolerance was used."""
    if y is None and active_tol is None:
        best = None
        for t in tol_sweep:
            c = _certificate_at(d, x, recover_dual(d, x, tol=t), active_tol=t)
            c["active_tol"] = t
            if best is None or max(c["dual_rel"], c["gap_rel"]) < max(best["dual_rel"], best["gap_rel"]):
                best = c
        return best
    return _certificate_at(d, x, y, active_tol or 1e-6)


def _certificate_at(d, x, y=None, active_tol=1e-6):
    P, A = sp.csc_matrix(d["P"]), sp.csc_matrix(d["A"])
    q, l, u = np.asarray(d["q"], float), np.asarray(d["l"], float), np.asarray(d["u"], float)
    r = float(d.get("r", 0.0))
    x = np.asarray(x, float)
    Ax = A @ x
    viol_l = np.where(np.isfinite(l), np.maximum(l - Ax, 0) / (1 + np.abs(np.where(np.isfinite(l), l, 0.0))), 0.0)
    viol_u = np.where(np.isfinite(u), np.maximum(Ax - u, 0) / (1 + np.abs(np.where(np.isfinite(u), u, 0.0))), 0.0)
    primal_rel = float(np.max(np.maximum(viol_l, viol_u))) if Ax.size else 0.0
    if y is None:
        y = recover_dual(d, x, tol=active_tol)
    y = np.asarray(y, float)
    Px = P @ x
    Aty = A.T @ y
    res = Px + q + Aty
    scale = 1.0 + max(float(np.max(np.abs(Px))) if Px.size else 0.0, float(np.max(np.abs(q))) if q.size else 0.0,
                      float(np.max(np.abs(Aty))) if Aty.size else 0.0)
    dual_rel = float(np.max(np.abs(res))) / scale if res.size else 0.0
    primal_obj = float(0.5 * x @ (P @ x) + q @ x + r)
    # Lagrangian at y: inf_x 1/2 x'Px + q'x + y'(Ax) - (support of the box [l, u] at y)
    yp, yn = np.maximum(y, 0.0), np.minimum(y, 0.0)
    support = float(np.sum(np.where(np.isfinite(u), u, 0.0) * yp) + np.sum(np.where(np.isfinite(l), l, 0.0) * yn))
    # with x eliminated by stationarity (exact when res = 0): dual_obj = -1/2 x'Px - support
    dual_obj = float(-0.5 * x @ (P @ x) - support + r)
    gap_rel = abs(primal_obj - dual_obj) / (1 + abs(primal_obj) + abs(dual_obj))
    bound_slack = float(np.max(np.abs(res)) * np.max(np.abs(x))) if res.size and x.size else 0.0
    return {"primal_rel": primal_rel, "dual_rel": dual_rel, "gap_rel": gap_rel, "primal_obj": primal_obj,
            "dual_obj": dual_obj, "bound_slack_abs": bound_slack,
            "verdict": "certified" if max(primal_rel, dual_rel, gap_rel) <= 1e-6 else "not certified at 1e-6"}
