"""Level-3 runtime certificates: fixed-point residual and TV duality gap.

Rigorous TV duality gap. With P(x) = (1/2)||x-y||^2 + lam||Dx||_1 and
lam||z||_1 = max_{||u||_inf <= lam} <z, u>, the saddle problem is

    min_x max_{||u||_inf <= lam} (1/2)||x-y||^2 + <Dx, u>.

The inner minimization over x is attained at x = y - D^T u, giving the
dual value

    G(u) = <y, D^T u> - (1/2)||D^T u||^2,   feasible iff ||u||_inf <= lam.

For any x and any FEASIBLE u:  gap(x, u) = P(x) - G(u) >= 0, and = 0 only
at a primal-dual optimum. Termination metric: rel_gap = gap/(|P(x)|+1).
"""
from __future__ import annotations

import json
import platform
import time

import jax
import jax.numpy as jnp


def fp_residual(state_new, state_old) -> jnp.ndarray:
    """||x_{k+1} - x_k|| over the full (pytree) iterate."""
    leaves_new = jax.tree_util.tree_leaves(state_new)
    leaves_old = jax.tree_util.tree_leaves(state_old)
    sq = sum(jnp.sum((a - b) ** 2) for a, b in zip(leaves_new, leaves_old))
    return jnp.sqrt(sq)


def tv_primal(x, y, lam, D) -> jnp.ndarray:
    """P(x) = (1/2)||x - y||^2 + lam * ||D x||_1."""
    return 0.5 * jnp.sum((x - y) ** 2) + lam * jnp.sum(jnp.abs(D.forward(x)))


def tv_dual(u, y, lam, D) -> jnp.ndarray:
    """G(u) = <y, D^T u> - (1/2)||D^T u||^2 after projecting u to feasibility.

    u is clipped to ||u||_inf <= lam so the returned value is always a
    valid (feasible) dual bound and the gap is rigorous.
    """
    u = jnp.clip(u, -lam, lam)
    Dtu = D.adjoint(u)
    return jnp.sum(y * Dtu) - 0.5 * jnp.sum(Dtu**2)


def tv_gap(x, u, y, lam, D) -> jnp.ndarray:
    """gap(x, u) = P(x) - G(clip(u)) >= 0; -> 0 at the optimum."""
    return tv_primal(x, y, lam, D) - tv_dual(u, y, lam, D)


def rel_gap(x, u, y, lam, D) -> jnp.ndarray:
    """Relative duality gap: gap / (|P(x)| + 1)."""
    P = tv_primal(x, y, lam, D)
    G = tv_dual(u, y, lam, D)
    return (P - G) / (jnp.abs(P) + 1.0)


# ---------------------------------------------------------------------------
# LP relative duality gap (PDLP-style, jit-safe). Lives here (not in
# atlas.bench.lp_cell) so atlas.schemes can terminate LP solves on it
# without importing the bench package.
# ---------------------------------------------------------------------------


def _lp_fwd(A, x):
    return A.forward(x) if hasattr(A, "forward") else jnp.asarray(A) @ x


def _lp_adj(A, u):
    return A.adjoint(u) if hasattr(A, "adjoint") else jnp.asarray(A).T @ u


def lp_rel_gap(x, u, c, A, b):
    """Scalar termination certificate for the LP cell (jnp, jit-safe).

    Convention: u is the PDHG dual iterate of the saddle
    L(x, u) = c'x + <u, Ax - b> over x >= 0, so the standard LP dual
    variable is y = -u (dual: max b'y s.t. A'y <= c; dual value -b'u).

    Returns max(rel_primal_feas, rel_dual_feas, rel_gap) with
        rel_primal_feas = ||A x^ - b|| / (1 + ||b||),      x^ = max(x, 0)
        rel_dual_feas   = ||max(-(c + A'u), 0)|| / (1 + ||c||)
        rel_gap         = |c'x^ + b'u| / (1 + |c'x^| + |b'u|).
    On exactly feasible (x, y=-u) the raw gap c'x - b'y = c'x + b'u >= 0
    by weak duality (tested); A may be a LinOp or a dense array.
    """
    x = jnp.asarray(x)
    u = jnp.asarray(u)
    c = jnp.asarray(c)
    b = jnp.asarray(b)
    xp = jnp.maximum(x, 0.0)
    P = jnp.vdot(c, xp)
    D = -jnp.vdot(b, u)
    pres = jnp.linalg.norm(_lp_fwd(A, xp) - b) / (1.0 + jnp.linalg.norm(b))
    slack = c + _lp_adj(A, u)
    dres = jnp.linalg.norm(jnp.maximum(-slack, 0.0)) / (1.0 + jnp.linalg.norm(c))
    gap = jnp.abs(P - D) / (1.0 + jnp.abs(P) + jnp.abs(D))
    return jnp.maximum(jnp.maximum(pres, dres), gap)


def _support_box(y, lo, hi):
    """sigma_C(y) = sup_{z in [lo, hi]} <y, z>, finite parts only."""
    hi_f = jnp.where(jnp.isfinite(hi), hi, 0.0)
    lo_f = jnp.where(jnp.isfinite(lo), lo, 0.0)
    return jnp.sum(jnp.where(y > 0, y * hi_f, 0.0) + jnp.where(y < 0, y * lo_f, 0.0))


def qp_rel_kkt(x, y, P, q, A, lo, hi):
    """Relative KKT certificate for  min (1/2)x'Px + q'x  s.t.  lo <= Ax <= hi.

    y is the dual of the constraint block, with the Condat-Vu saddle
    f(x) + <Ax, y> - h*(y), h the indicator of [lo, hi]; stationarity is then
    Px + q + A'y = 0, the same sign convention as OSQP.

    Returns max(rel_primal, rel_dual, rel_gap):
      rel_primal = ||Ax - clip(Ax, lo, hi)||_inf / (1 + max(||Ax||_inf, ||clip||_inf))
      rel_dual   = (||Px + q + A'y||_inf + cone) / (1 + max(||Px||_inf, ||A'y||_inf, ||q||_inf))
      rel_gap    = |x'Px + q'x + sigma_C(y)| / (1 + |(1/2)x'Px + q'x| + |sigma_C(y)|)
    The first two are OSQP's termination residuals with eps_abs = eps_rel = tol,
    so this function scores OSQP's returned (x, y) and ours identically. `cone`
    charges any dual component pointing at an infinite bound (y_i > 0 where
    hi_i = inf, or y_i < 0 where lo_i = -inf), where sigma_C would be +inf;
    without it the support term would silently read those entries as zero.
    """
    x = jnp.asarray(x); y = jnp.asarray(y)
    Ax = _lp_fwd(A, x)
    z = jnp.clip(Ax, lo, hi)
    Px = _lp_fwd(P, x)
    ATy = _lp_adj(A, y)
    inf = lambda v: jnp.max(jnp.abs(v))
    rel_p = inf(Ax - z) / (1.0 + jnp.maximum(inf(Ax), inf(z)))
    cone = (jnp.max(jnp.where(jnp.isinf(hi), jnp.maximum(y, 0.0), 0.0))
            + jnp.max(jnp.where(jnp.isinf(lo), jnp.maximum(-y, 0.0), 0.0)))
    rel_d = (inf(Px + q + ATy) + cone) / (1.0 + jnp.maximum(jnp.maximum(inf(Px), inf(ATy)), inf(q)))
    quad = jnp.vdot(x, Px)
    sup = _support_box(y, lo, hi)
    gap = jnp.abs(quad + jnp.vdot(q, x) + sup) / (1.0 + jnp.abs(0.5 * quad + jnp.vdot(q, x)) + jnp.abs(sup))
    return jnp.maximum(jnp.maximum(rel_p, rel_d), gap)


# ---------------------------------------------------------------------------
# Level-4 empirical certificate: benchmark card
# ---------------------------------------------------------------------------


def benchmark_card(result, problem, tol: float, extra: dict | None = None) -> dict:
    """Assemble a JSON-serializable benchmark card for one solve.

    Fields: instance, scheme + cert, time-to-tol, iterations, final
    residuals/gaps, hardware note.
    """
    card = {
        "instance": {
            "name": problem.name,
            "shape": list(problem.shape),
            "lam": problem.data.get("lam"),
        },
        "cert": result.cert.to_dict() if result.cert is not None else None,
        "tolerance_rel_gap": tol,
        "iterations": int(result.iters),
        "wall_time_s": float(result.wall_time),
        "final_fp_residual": float(result.fp_residual_history[-1]),
        "final_rel_gap": float(result.rel_gap_history[-1]),
        "hardware": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "jax_backend": jax.default_backend(),
            "jax_version": jax.__version__,
            "x64": bool(jax.config.jax_enable_x64),
            "note": "CPU-only (JAX_PLATFORMS=cpu); GPU reserved for vLLM server",
        },
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if extra:
        card.update(extra)
    json.dumps(card)  # fail fast if not serializable
    return card


def logreg_enet_gap_parts(w, f, g):
    """(P, D, v, p) for elastic-net logistic regression: primal P(w), dual D(u) at the natural dual
    point u = grad F(Xw) (rescaled into dual feasibility when alpha = 1), v = -X^T u and the
    probabilities p_i in [0, 1] that define u. See logreg_enet_rel_gap for the formulas."""
    X, y, lam, alpha = f.X, f.y, g.lam, g.alpha
    n = X.shape[0]
    z = y * (X @ w)
    P = jnp.mean(jnp.logaddexp(0.0, -z)) + g.value(w)
    p = jax.nn.sigmoid(-z)
    v = (X.T @ (y * p)) / n
    scale = jnp.where(alpha >= 1.0, jnp.minimum(1.0, lam / jnp.maximum(jnp.max(jnp.abs(v)), 1e-300)), 1.0)
    p, v = p * scale, v * scale
    ent = jnp.where(p > 0, p * jnp.log(jnp.maximum(p, 1e-300)), 0.0) + \
          jnp.where(p < 1, (1 - p) * jnp.log(jnp.maximum(1 - p, 1e-300)), 0.0)
    excess = jnp.maximum(jnp.abs(v) - lam * alpha, 0.0)
    Gstar = jnp.where(alpha >= 1.0, 0.0, jnp.sum(excess ** 2) / (2.0 * lam * jnp.maximum(1.0 - alpha, 1e-300)))
    D = -jnp.mean(ent) - Gstar
    return P, D, v, p


def logreg_enet_gap_safe_screen(w, f, g, margin: float = 1e-9, colnorm=None):
    """Gap Safe screening for elastic-net logistic regression (Ndiaye, Fercoq, Gramfort, Salmon,
    JMLR 2017). The dual is 4n-strongly concave (the binary-entropy term has curvature >= 4), so
    the dual optimum lies within r = sqrt((P - D) / (2 n)) of the dual point u of the certificate.
    At the optimum w*_j = 0 whenever |x_j^T u*| < lam alpha, hence feature j is SAFELY zero when
        |x_j^T u| + r ||x_j|| < lam alpha.
    The strict inequality is applied with a relative margin, lam alpha (1 - margin): with
    alpha = 1 every active feature sits exactly on |x_j^T u*| = lam, and after the rescaling into
    dual feasibility rounding can leave it one ulp below lam, which the bare strict test would
    wrongly screen (caught by tests/test_screening.py). The margin only screens less.
    Pass the column norms ||x_j|| as `colnorm` to avoid recomputing them (a full pass over X).
    Returns (mask of provably zero features, r, absolute gap)."""
    P, D, v, _ = logreg_enet_gap_parts(w, f, g)
    gap = jnp.maximum(P - D, 0.0)
    r = jnp.sqrt(gap / (2.0 * f.X.shape[0]))
    colnorm = jnp.linalg.norm(f.X, axis=0) if colnorm is None else colnorm
    return jnp.abs(v) + r * colnorm < g.lam * g.alpha * (1.0 - margin), r, gap


def logreg_enet_rel_gap(w, f, g):
    """Relative duality gap of elastic-net logistic regression (atlas.bench.logreg_cell).

    Primal P(w) = F(Xw) + G(w). Fenchel dual at the natural point u = grad F(Xw):
    with p_i = sigmoid(-y_i x_i^T w) in (0, 1),
        F*(u) = (1/n) sum_i [p_i log p_i + (1 - p_i) log(1 - p_i)],
        G*(v) = sum_j (|v_j| - lam alpha)_+^2 / (2 lam (1 - alpha))   (alpha < 1),
        D = -F*(u) - G*(-X^T u),  X^T u = -(1/n) X^T (y p).
    For alpha = 1, G* is the indicator of ||v||_inf <= lam and u is scaled by
    min(1, lam / ||X^T u||_inf), which keeps p in [0, 1]. Returns (P - D) / max(|P|, 1e-300).
    """
    P, D, _, _ = logreg_enet_gap_parts(w, f, g)
    return (P - D) / jnp.maximum(jnp.abs(P), 1e-300)



def qp_accuracy_judge(x, P, q, r, A, lo, hi, ref_obj=None, feas_tol=1e-4, obj_tol=1e-3):
    """Judge whether a returned QP point is actually a solution, for claims and comparisons.

    The KKT certificates (qp_rel_kkt, following OSQP) normalise every residual by ONE global
    scale (norms of the iterate or of the data), which on badly scaled instances hides
    violations: on BOYD2 a point missing 167,970 rows by more than 1e-6 of their own bounds
    passed at 7e-5 because the largest bound is 2.2e11; on PRIMALC2 and PRIMALC8
    Anderson-extrapolated iterates of size 3e8 to 5e8 passed as well. This judge applies
      per-row feasibility:  max_i  violation_i / (1 + |bound_i|)  <=  feas_tol,
      objective accuracy:   |obj - ref_obj| / max(1, |ref_obj|)  <=  obj_tol  (when a trusted
                            high-accuracy reference exists; otherwise the verdict is
                            "no reference" and no accuracy claim may be made).
    feas_tol must match the tolerance the solver was run to (a 1e-4 run cannot be expected to be
    feasible to 1e-6 row by row). Returns a dict with both numbers and a verdict. NumPy inputs (P, A scipy sparse)."""
    import numpy as np
    x = np.asarray(x, dtype=np.float64)
    Ax = A @ x
    lo, hi = np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)
    viol_lo = np.where(np.isfinite(lo), np.maximum(lo - Ax, 0.0) / (1.0 + np.abs(np.where(np.isfinite(lo), lo, 0.0))), 0.0)
    viol_hi = np.where(np.isfinite(hi), np.maximum(Ax - hi, 0.0) / (1.0 + np.abs(np.where(np.isfinite(hi), hi, 0.0))), 0.0)
    row_viol = float(np.max(np.maximum(viol_lo, viol_hi))) if Ax.size else 0.0
    obj = float(0.5 * x @ (P @ x) + np.asarray(q) @ x + r)
    obj_err = None if ref_obj is None else abs(obj - ref_obj) / max(1.0, abs(ref_obj))
    feasible = row_viol <= feas_tol
    if not feasible:
        verdict = "infeasible"
    elif obj_err is None:
        verdict = "feasible, no reference"
    else:
        verdict = "solution" if obj_err <= obj_tol else "feasible, objective off"
    return {"row_violation": row_viol, "obj": obj, "obj_err": obj_err, "verdict": verdict}
