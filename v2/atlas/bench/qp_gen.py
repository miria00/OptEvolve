"""Generated convex QPs with a size knob, shared by the scale sweep and the creation loop.

Two standard families from the OSQP benchmark set, written here so no download is needed; the seed and the
density are recorded with every instance. Both return the {P, q, r, A, l, u} dict that
atlas.bench.qp_cell.to_composite consumes, so the same instance can be handed to OSQP, SCS, Clarabel and to
our own schemes.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def _sprandn(m, n, density, seed):
    """Sparse standard-normal matrix, sampled in O(nnz) memory.

    scipy.sparse.random with an integer random_state routes to RandomState.choice(m*n, k,
    replace=False), which PERMUTES THE WHOLE INDEX POPULATION: at m=50,000 n=100,000 that is a
    37.3 GiB int64 allocation, and it is why every sweep above n_var 75,000 was OOM-killed
    (2026-09-13). Rejection sampling needs O(k), about 40 MB at the same shape.

    NOTE: this draws a DIFFERENT matrix than the old path for the same seed, so instances
    generated before 2026-09-13 are not reproduced here. A scaling study must regenerate every
    size with this function rather than mixing the two.
    """
    rng = np.random.default_rng(seed)
    mn = int(m) * int(n)
    k = int(round(density * mn))
    if k <= 0:
        return sp.csc_matrix((m, n))
    idx = np.unique(rng.integers(0, mn, size=k, dtype=np.int64))
    while idx.size < k:                       # top up the collisions, still O(k)
        extra = rng.integers(0, mn, size=2 * (k - idx.size) + 8, dtype=np.int64)
        idx = np.unique(np.concatenate([idx, extra]))
    idx = idx[:k]
    rows, cols = np.divmod(idx, n)
    return sp.csc_matrix((rng.standard_normal(k), (rows, cols)), shape=(m, n))


def lasso(n, m=None, seed=0, density=1e-3):
    """OSQP benchmark lasso: min 0.5 y^T y + lam 1^T t, y = A x - b, -t <= x <= t."""
    rng = np.random.default_rng(seed)
    m = m or n // 2
    A = _sprandn(m, n, density, seed)
    v = rng.standard_normal(n) * (rng.random(n) < 0.1)
    b = A @ v + rng.standard_normal(m) * 0.1
    lam = 0.2 * float(np.max(np.abs(A.T @ b)))
    P = sp.block_diag([sp.csc_matrix((n, n)), sp.identity(m, format="csc"), sp.csc_matrix((n, n))], format="csc")
    q = np.concatenate([np.zeros(n), np.zeros(m), lam * np.ones(n)])
    Aeq = sp.hstack([A, -sp.identity(m, format="csc"), sp.csc_matrix((m, n))], format="csc")
    Abox1 = sp.hstack([sp.identity(n, format="csc"), sp.csc_matrix((n, m)), -sp.identity(n, format="csc")], format="csc")
    Abox2 = sp.hstack([sp.identity(n, format="csc"), sp.csc_matrix((n, m)), sp.identity(n, format="csc")], format="csc")
    Ac = sp.vstack([Aeq, Abox1, Abox2], format="csc")
    l = np.concatenate([b, -np.inf * np.ones(n), np.zeros(n)])
    u = np.concatenate([b, np.zeros(n), np.inf * np.ones(n)])
    return {"P": P, "q": q, "r": 0.0, "A": Ac, "l": l, "u": u}


def svm(nsamp, seed=0, density=1e-3):
    """OSQP benchmark SVM: min 0.5 w^T w + C 1^T xi, xi >= 1 - d_i (a_i^T w), xi >= 0."""
    rng = np.random.default_rng(seed)
    nfeat = max(10, nsamp // 10)
    d = np.sign(rng.standard_normal(nsamp))
    A = _sprandn(nsamp, nfeat, density, seed)
    A = sp.csc_matrix(A.multiply((d / 4.0)[:, None]) + _sprandn(nsamp, nfeat, density, seed + 1))
    C = 1.0
    P = sp.block_diag([sp.identity(nfeat, format="csc"), sp.csc_matrix((nsamp, nsamp))], format="csc")
    q = np.concatenate([np.zeros(nfeat), C * np.ones(nsamp)])
    Amarg = sp.hstack([-sp.diags(d) @ A, -sp.identity(nsamp, format="csc")], format="csc")
    Axi = sp.hstack([sp.csc_matrix((nsamp, nfeat)), sp.identity(nsamp, format="csc")], format="csc")
    Ac = sp.vstack([Amarg, Axi], format="csc")
    l = np.concatenate([-np.inf * np.ones(nsamp), np.zeros(nsamp)])
    u = np.concatenate([-np.ones(nsamp), np.inf * np.ones(nsamp)])
    return {"P": P, "q": q, "r": 0.0, "A": Ac, "l": l, "u": u}



def meb_dual(npts, seed=0, density=1e-3, dim=None):
    """Dual of the minimum enclosing ball (Chebyshev centre) of npts points, a classical problem.

        min_lambda  lambda' G G' lambda - c'lambda   s.t.  lambda >= 0,  1'lambda = 1,
        c_i = ||a_i||^2,  G the point matrix, centre = G'lambda.

    A THIRD real family for the relocation rule, and one with different provenance from portfolio
    (computational geometry rather than finance) and from svm_dual (a box-hyperplane rather than a
    simplex). The feasible set is exactly the probability simplex, so the detector should fire and
    the derived transform should be the SIMPLEX branch, not the box-hyperplane one.

    G G' is npts x npts and dense, so it is never formed; the lift z = G'lambda keeps the coupling
    sparse, exactly as in portfolio and svm_dual. Variables are [lambda (npts), z (dim)].
    Note P has a ZERO block on lambda, so mu/L = 0 as in svm_dual, while portfolio's D block is
    strictly positive; the three families therefore span both curvature regimes.
    """
    rng = np.random.default_rng(seed)
    dim = dim or max(10, npts // 10)
    G = _sprandn(npts, dim, density, seed)
    c = np.asarray(G.multiply(G).sum(axis=1)).ravel()          # ||a_i||^2, row norms of G
    P = sp.block_diag([sp.csc_matrix((npts, npts)), 2.0 * sp.identity(dim, format="csc")],
                      format="csc")
    q = np.concatenate([-c, np.zeros(dim)])
    Afac = sp.hstack([G.T.tocsc(), -sp.identity(dim, format="csc")], format="csc")   # G'l - z = 0
    Asum = sp.hstack([sp.csc_matrix(np.ones((1, npts))), sp.csc_matrix((1, dim))], format="csc")
    Apos = sp.hstack([sp.identity(npts, format="csc"), sp.csc_matrix((npts, dim))], format="csc")
    Ac = sp.vstack([Afac, Asum, Apos], format="csc")
    l = np.concatenate([np.zeros(dim), [1.0], np.zeros(npts)])
    u = np.concatenate([np.zeros(dim), [1.0], np.inf * np.ones(npts)])
    return {"P": P, "q": q, "r": 0.0, "A": Ac, "l": l, "u": u}


def svm_dual(nsamp, seed=0, density=1e-3, C=1.0):
    """Dual of the standard soft-margin SVM WITH INTERCEPT, which is what libsvm/SMO actually solve.

    max_a  1'a - 0.5 a' Q a   s.t.  0 <= a <= C,  d'a = 0,   Q = diag(d) A A' diag(d)

    Q is nsamp x nsamp and dense, so it is never formed. Instead keep the factor G = diag(d) A and
    lift z = G'a, which makes the objective 0.5 z'z - 1'a with a SPARSE nfeat-row coupling. This is
    the same lifted shape as portfolio, and it is the point: the equality d'a = 0 is a DENSE row, so
    the relocation detector should fire, while the residual left in L is only nfeat = nsamp/10 rows.

    Note the family differs from svm() in this module, which is the PRIMAL and has no intercept and
    therefore no equality row at all.

    Variables u = [a (nsamp), z (nfeat)].
    """
    rng = np.random.default_rng(seed)
    nfeat = max(10, nsamp // 10)
    d = np.sign(rng.standard_normal(nsamp))
    A = _sprandn(nsamp, nfeat, density, seed)
    A = sp.csc_matrix(A.multiply((d / 4.0)[:, None]) + _sprandn(nsamp, nfeat, density, seed + 1))
    G = sp.csc_matrix(sp.diags(d) @ A)                      # nsamp x nfeat
    P = sp.block_diag([sp.csc_matrix((nsamp, nsamp)), sp.identity(nfeat, format="csc")],
                      format="csc")
    q = np.concatenate([-np.ones(nsamp), np.zeros(nfeat)])
    Afac = sp.hstack([G.T.tocsc(), -sp.identity(nfeat, format="csc")], format="csc")   # G'a - z = 0
    Abox = sp.hstack([sp.identity(nsamp, format="csc"), sp.csc_matrix((nsamp, nfeat))], format="csc")
    Aeq = sp.hstack([sp.csc_matrix(d.reshape(1, -1)), sp.csc_matrix((1, nfeat))], format="csc")
    Ac = sp.vstack([Afac, Abox, Aeq], format="csc")
    l = np.concatenate([np.zeros(nfeat), np.zeros(nsamp), [0.0]])
    u = np.concatenate([np.zeros(nfeat), C * np.ones(nsamp), [0.0]])
    return {"P": P, "q": q, "r": 0.0, "A": Ac, "l": l, "u": u}


def portfolio(n, k=None, seed=0, density=1e-3):
    """OSQP benchmark portfolio: min x^T D x + y^T y - mu^T x / gamma, y = F^T x, 1^T x = 1, x >= 0.

    The factor model is what makes this family distinct from lasso: the Hessian is diagonal plus a
    rank-k term, lifted here so P stays diagonal and the coupling lives in A. k = sqrt(n) by default.
    """
    rng = np.random.default_rng(seed)
    k = k or max(2, int(np.sqrt(n)))
    F = _sprandn(n, k, density, seed)
    D = rng.random(n) * np.sqrt(k)
    mu = rng.standard_normal(n)
    gamma = 1.0
    P = sp.block_diag([2.0 * sp.diags(D), 2.0 * sp.identity(k, format="csc")], format="csc")
    q = np.concatenate([-mu / gamma, np.zeros(k)])
    Afac = sp.hstack([F.T.tocsc(), -sp.identity(k, format="csc")], format="csc")
    Abud = sp.hstack([sp.csc_matrix(np.ones((1, n))), sp.csc_matrix((1, k))], format="csc")
    Apos = sp.hstack([sp.identity(n, format="csc"), sp.csc_matrix((n, k))], format="csc")
    Ac = sp.vstack([Afac, Abud, Apos], format="csc")
    l = np.concatenate([np.zeros(k), [1.0], np.zeros(n)])
    u = np.concatenate([np.zeros(k), [1.0], np.inf * np.ones(n)])
    return {"P": P, "q": q, "r": 0.0, "A": Ac, "l": l, "u": u}


def control(nx, nu=None, horizon=10, seed=0, density=None):
    """OSQP benchmark control: linear MPC over a horizon, min sum x_t^T Q x_t + u_t^T R u_t.

    Banded block-tridiagonal structure, unlike the random sparsity of lasso/svm/portfolio, which is
    the point of including it: axis 2 is about structure, not only size. `density` is accepted and
    ignored because the dynamics blocks are dense and the global sparsity follows from the horizon.
    """
    rng = np.random.default_rng(seed)
    nu = nu or max(1, nx // 4)
    T = int(horizon)
    N = T + 1                                     # states x_0 .. x_T
    Ad = rng.standard_normal((nx, nx)) / np.sqrt(nx)
    Ad *= 0.95 / max(1e-12, np.max(np.abs(np.linalg.eigvals(Ad))))   # stable, so the horizon is meaningful
    Bd = rng.standard_normal((nx, nu)) / np.sqrt(nx)
    Q = sp.identity(nx, format="csc")
    QT = 10.0 * sp.identity(nx, format="csc")     # terminal cost
    R = 0.1 * sp.identity(nu, format="csc")
    P = sp.block_diag([sp.kron(sp.identity(T), Q), QT, sp.kron(sp.identity(T), R)], format="csc")
    q = np.zeros(nx * N + nu * T)
    # x_{t+1} - Ad x_t - Bd u_t = 0
    Adyn = sp.hstack([sp.kron(sp.eye(T, N, k=1), sp.identity(nx)) - sp.kron(sp.eye(T, N, k=0), sp.csc_matrix(Ad)),
                      -sp.kron(sp.identity(T), sp.csc_matrix(Bd))], format="csc")
    Ax0 = sp.hstack([sp.identity(nx, format="csc"), sp.csc_matrix((nx, nx * (N - 1))),
                     sp.csc_matrix((nx, nu * T))], format="csc")
    Abx = sp.hstack([sp.identity(nx * N, format="csc"), sp.csc_matrix((nx * N, nu * T))], format="csc")
    Abu = sp.hstack([sp.csc_matrix((nu * T, nx * N)), sp.identity(nu * T, format="csc")], format="csc")
    Ac = sp.vstack([Ax0, Adyn, Abx, Abu], format="csc")
    x0 = rng.standard_normal(nx) * 0.5
    xmax, umax = 5.0, 0.5
    l = np.concatenate([x0, np.zeros(nx * T), -xmax * np.ones(nx * N), -umax * np.ones(nu * T)])
    u = np.concatenate([x0, np.zeros(nx * T), xmax * np.ones(nx * N), umax * np.ones(nu * T)])
    return {"P": P, "q": q, "r": 0.0, "A": Ac, "l": l, "u": u}


def huber(m, n=None, seed=0, density=1e-3):
    """OSQP benchmark Huber fitting: min sum phi_hub(a_i^T x - b_i), lifted to a QP.

    min u^T u + 2M 1^T (r + s) s.t. A x - b - u = r - s, r >= 0, s >= 0. The lift is why this is a
    separate family rather than a reweighted lasso: the quadratic block sits on the residual, so P is
    singular with a very different null space than lasso's.
    """
    rng = np.random.default_rng(seed)
    n = n or max(10, m // 2)
    A = _sprandn(m, n, density, seed)
    v = rng.standard_normal(n)
    b = A @ v + rng.standard_normal(m) * 0.1
    b += (rng.random(m) < 0.05) * rng.standard_normal(m) * 10.0      # outliers, which is the point of Huber
    M = 1.0
    Z_n = sp.csc_matrix((n, n))
    P = sp.block_diag([Z_n, 2.0 * sp.identity(m, format="csc"),
                       sp.csc_matrix((m, m)), sp.csc_matrix((m, m))], format="csc")
    q = np.concatenate([np.zeros(n), np.zeros(m), 2.0 * M * np.ones(m), 2.0 * M * np.ones(m)])
    Aeq = sp.hstack([A, -sp.identity(m, format="csc"), -sp.identity(m, format="csc"),
                     sp.identity(m, format="csc")], format="csc")
    Ar = sp.hstack([sp.csc_matrix((m, n + m)), sp.identity(m, format="csc"), sp.csc_matrix((m, m))], format="csc")
    As = sp.hstack([sp.csc_matrix((m, n + 2 * m)), sp.identity(m, format="csc")], format="csc")
    Ac = sp.vstack([Aeq, Ar, As], format="csc")
    l = np.concatenate([b, np.zeros(2 * m)])
    u = np.concatenate([b, np.inf * np.ones(2 * m)])
    return {"P": P, "q": q, "r": 0.0, "A": Ac, "l": l, "u": u}


def control_h(horizon, nx=40, seed=0, density=None):
    """control scaled by HORIZON rather than state dimension.

    The size knob in FAMILIES maps to the first positional argument, and for `control` that is nx.
    Scaling nx makes the problem denser per variable (measured nnz/n_var 38.9 -> 75.9 -> 150.0 as nx
    doubles, because nnz ~ 10*nx^2 while n_var ~ 13.5*nx), so reaching n_var 50,000 that way needs
    ~137M nonzeros. Scaling the horizon holds nnz/n_var flat at ~41 (measured 41.0 / 41.6 / 41.8 at
    horizons 40 / 160 / 640) and keeps the block-tridiagonal structure intact.

    DO NOT USE THIS AS A SCALE TEST. Measured at nx=40: the stabilizing dynamics (spectral radius 0.95)
    drive the state to zero within about 40 stages regardless of horizon, so at horizon 160 only 41 of
    161 stages are active (||x_t|| falls from 3.56 to 6.12e-08 by t=39 and 6.64e-27 by t=159, inputs
    nonzero in 52 of 160 stages). 75% of the instance is padding, and the optimal objective is identical
    to eleven digits at horizon 40 and 160. A speedup measured here would largely reflect how a solver
    handles near-zero blocks, not control-family difficulty.

    To test control at scale, grow nx instead (n_var ~ 13.5*nx, nnz ~ 10*nx^2): nx=1200 gives n_var
    ~16,200 with ~14M nonzeros, which clears the ~16,000 crossover and is runnable.

    Kept only for structural experiments where long, mostly-inactive horizons are the point.
    """
    return control(nx, horizon=int(horizon), seed=seed)


def portfolio_hr(n, k=2500, seed=0, density=1e-3):
    """portfolio with a HIGH-RANK factor term (k fixed, default 2500) rather than k = sqrt(n).

    Exists because the default low-rank regime is the one family we lose, and a single one-off run
    suggested the loss is regime-specific: at n=5000, k=2500 we measured 1.18x (ours 0.686 s vs
    clarabel 0.813 s) where the default k=70 loses 0.02x. That claim now sits in the skill's
    auto-loaded description on ONE run with no interval, which is weaker evidence than every other
    number beside it. Registering it lets the reps harness give it paired repetitions and seeds.

    NOTE the two dead predictors: fill RATIO and factorization work per variable BOTH fail to explain
    which k wins (the latter predicted wins at k=250 and k=1000 that measured as 50x and 2x losses).
    """
    return portfolio(n, k=int(k), seed=seed, density=density)


# One lookup so a new family plugs into the sweep and the search without an if-chain.
FAMILIES = {"lasso": lasso, "svm": svm, "portfolio": portfolio, "control": control,
            "control_h": control_h, "huber": huber,
            "portfolio_hr": portfolio_hr, "svm_dual": svm_dual, "meb_dual": meb_dual}
