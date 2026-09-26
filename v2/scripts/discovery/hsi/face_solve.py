"""Exact face solve, imported from the lasso engine's derivation.

The lasso path solver does not iterate to the answer.  Once the active set is
known it solves the face exactly, because on the face defined by the support
the solution satisfies a small linear system.  The same is true here: on the
face where the support is S, the fully constrained least squares problem

    min 1/2||D x - y||^2   s.t.  x >= 0,  1^T x = 1

has, for the interior of that face, the KKT system

    [ D_S^T D_S   1 ] [x_S]   [ D_S^T y ]
    [ 1^T         0 ] [ nu ] = [    1    ]

which is (|S|+1) square.  Supports here are 5 to 30 of 498, so this is tiny.

The lasso engine's lesson about rank-deficient faces carries over: flooring
one diagonal entry at a time is not the regularization a fresh factorization
would pick, so a face that does not solve cleanly is REFUSED and handed back
to the iterative method rather than patched.  A solved face is accepted only
if it is feasible and its recomputed gap is an improvement, which is the
re-admission pass that makes the heuristic support exact.
"""
from __future__ import annotations
import numpy as np


def _unused_face_polish(D, Y, Z, DtD=None, DtY=None, supp_tol=1e-10, kmax=64):
    """Return a polished Z and a per-pixel flag saying whether the face solve
    was accepted.  Never returns a point worse than the one it was given."""
    n, m = Z.shape
    if DtD is None:
        DtD = D.T @ D
    if DtY is None:
        DtY = D.T @ Y
    Zp = Z.copy()
    accepted = np.zeros(m, dtype=bool)
    refused_rank = 0
    refused_size = 0
    refused_worse = 0

    def relgap(x, j):
        g = DtD @ x - DtY[:, j]
        r = D @ x - Y[:, j]
        f = 0.5 * float(r @ r)
        return (float(g @ x) - float(g.min())) / (1.0 + abs(f))

    for j in range(m):
        S = np.flatnonzero(Z[:, j] > supp_tol)
        if S.size == 0 or S.size > kmax:
            refused_size += 1
            continue
        k = S.size
        K = np.zeros((k + 1, k + 1))
        K[:k, :k] = DtD[np.ix_(S, S)]
        K[:k, k] = 1.0
        K[k, :k] = 1.0
        rhs = np.empty(k + 1)
        rhs[:k] = DtY[S, j]
        rhs[k] = 1.0
        try:
            sol = np.linalg.solve(K, rhs)
        except np.linalg.LinAlgError:
            refused_rank += 1
            continue
        xs = sol[:k]
        if not np.all(np.isfinite(xs)) or xs.min() < -1e-12:
            refused_rank += 1
            continue
        cand = np.zeros(n)
        cand[S] = np.maximum(xs, 0.0)
        s = cand.sum()
        if not (abs(s - 1.0) <= 1e-9):
            refused_rank += 1
            continue
        if relgap(cand, j) <= relgap(Z[:, j], j):
            Zp[:, j] = cand
            accepted[j] = True
        else:
            refused_worse += 1
    return Zp, accepted, {"refused_rank": refused_rank, "refused_size": refused_size,
                          "refused_worse": refused_worse}


def active_set_solve(D, Y, Z0, DtD=None, DtY=None, supp_tol=1e-10,
                     kmax=96, max_rounds=60, tol=1e-12):
    """Exact active-set solve seeded from an iterative point.

    Pairs the face solve with the re-admission pass that makes it exact, which
    is the pairing the lasso engine uses: the face solve alone is a heuristic
    about the support and refuses most pixels, because a loose iterate's
    support carries columns that belong at zero.  Dropping the negatives and
    re-admitting any column whose reduced gradient violates its KKT condition
    turns it into a finitely terminating method.

    KKT for min 1/2||Dx-y||^2 s.t. x>=0, 1^T x = 1:
        g = D^T D x - D^T y,   g_j = nu      for j in support
                               g_j >= nu     for j outside
    """
    n, m = Z0.shape
    if DtD is None:
        DtD = D.T @ D
    if DtY is None:
        DtY = D.T @ Y
    Z = Z0.copy()
    rounds = np.zeros(m, dtype=np.int32)
    status = np.zeros(m, dtype=np.int8)      # 1 solved, 0 gave up
    for j in range(m):
        S = np.flatnonzero(Z0[:, j] > supp_tol)
        if S.size == 0:
            S = np.array([int(np.argmax(DtY[:, j]))])
        b = DtY[:, j]
        for r in range(max_rounds):
            rounds[j] = r + 1
            if S.size > kmax:
                break
            k = S.size
            K = np.zeros((k + 1, k + 1))
            K[:k, :k] = DtD[np.ix_(S, S)]
            K[:k, k] = 1.0
            K[k, :k] = 1.0
            rhs = np.concatenate([b[S], [1.0]])
            try:
                sol = np.linalg.lstsq(K, rhs, rcond=None)[0]
            except np.linalg.LinAlgError:
                break
            xs, nu = sol[:k], sol[k]
            if xs.min() < -tol:                      # infeasible face: shrink
                S = S[xs > tol]
                if S.size == 0:
                    break
                continue
            x = np.zeros(n)
            x[S] = np.maximum(xs, 0.0)
            g = DtD @ x - b
            viol = g - nu
            viol[S] = np.inf
            jmin = int(np.argmin(viol))
            if viol[jmin] < -1e-11:                  # re-admission pass
                S = np.append(S, jmin)
                continue
            Z[:, j] = x
            status[j] = 1
            break
    return Z, status, rounds
