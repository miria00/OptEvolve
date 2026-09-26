"""Fixed external acceptance; generated code cannot select its own metric."""
from __future__ import annotations
import numpy as np

def certify(D, Y, X, tol=1e-4, feasibility_tol=1e-9):
    X = np.asarray(X, dtype=np.float64)
    if X.shape != (D.shape[1], Y.shape[1]):
        return {'accepted': False, 'error': 'wrong output shape', 'worst_gap': 1e300,
                'feasibility': 1e300, 'fraction_certified': 0.0}
    if not np.isfinite(X).all():
        return {'accepted': False, 'error': 'nonfinite output', 'worst_gap': 1e300,
                'feasibility': 1e300, 'fraction_certified': 0.0}
    # Direct residual, not the cancellation-prone expanded Gram objective.
    R = D @ X - Y
    G = D.T @ R
    f = 0.5 * np.sum(R * R, axis=0)
    gap = (np.sum(G * X, axis=0) - np.min(G, axis=0)) / (1 + np.abs(f))
    feas = np.maximum(np.max(np.maximum(-X, 0), axis=0), np.abs(X.sum(axis=0) - 1))
    # A substantially negative gap is invalid, not an excellent reward.
    good = np.isfinite(gap) & (gap >= -1e-9) & (gap <= tol) & (feas <= feasibility_tol)
    return {'accepted': bool(np.all(good)), 'worst_gap': float(np.max(gap)),
            'min_gap': float(np.min(gap)), 'feasibility': float(np.max(feas)),
            'fraction_certified': float(np.mean(good)), 'objective': float(np.sum(f))}

def aggregate(rows, cap_s):
    """All failures cost the full cap. Progress is diagnostic, never the objective."""
    times = [min(r['elapsed_s'], cap_s) if r.get('accepted') else cap_s for r in rows]
    return {'solved': sum(bool(r.get('accepted')) for r in rows), 'total': len(rows),
            'penalized_geomean_s': float(np.exp(np.mean(np.log(np.maximum(times, 1e-9))))),
            'times': times}
