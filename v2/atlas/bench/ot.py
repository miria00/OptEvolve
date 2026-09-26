"""Third LP family: discrete optimal transport as bipartite min-cost flow.

Unregularised discrete OT

    min_P  <C, P>   s.t.   P 1_n = a,  P^T 1_m = b,  P >= 0

IS a min-cost network-flow LP on the complete bipartite digraph K_{m,n}; it
is not an analogy.  Vectorising the plan row-major, x = vec(P) in R^{mn} with
x_{i*n+j} = P_ij, and taking

    nodes   0 .. m-1     sources          (n_nodes = m + n)
    nodes   m .. m+n-1   sinks
    arc e = i*n + j      tail(e) = i, head(e) = m + j       (n_arcs = m*n)

the node-arc incidence matrix A (A[tail(e), e] = +1, A[head(e), e] = -1) acts as

    (A x)_i     = sum_j P_ij           for i < m
    (A x)_{m+j} = -sum_i P_ij          for j < n

so  A x = b_vec  with  b_vec = [a ; -b]  is EXACTLY the pair of marginal
constraints.  With c = vec(C) the OT LP is verbatim the standard-form LP that
atlas.bench.lp_cell already maps onto the `lp_standard` composite

    f(x) = <c, x>   (affine, L_f = 0)     g(x) = I_{x >= 0}
    h(z) = I_{z = b_vec}                  L    = A

so Condat-Vu collapses to PDHG and the construction gate stays
t*s*||L||^2 < 1.  NO new atoms, NO new certificate, NO new scheme.

WHAT IS STRUCTURALLY NEW (this is why OT is worth a third family)
-----------------------------------------------------------------
netflow's grid_arcs / random_regular_arcs emit every undirected edge in BOTH
directions; the bipartite arcs here are oriented ONE way (source -> sink).
That single difference changes A A^T from 2*Laplacian to exactly the
Laplacian of K_{m,n}, whose spectrum is {0, m^(n-1), n^(m-1), m+n}.  Hence

    ||A||_2 = sqrt(m + n)      EXACTLY, in closed form,

with full singular spectrum {0, sqrt(min(m,n)), sqrt(max(m,n)), sqrt(m+n)}
and rank m+n-1.  Compare:

    netflow grid:    ||A|| = sqrt(8(1 - cos(pi(k-1)/k))) -> 4      BOUNDED in size
    netflow regular: ||A|| <= sqrt(4d)                            BOUNDED in size
    OT bipartite:    ||A|| = sqrt(m + n)                          UNBOUNDED, ~ sqrt(N)

So the admissible region of the method axis SHRINKS AS 1/N on OT:

    tau * sigma  <  1 / (safety^2 (m + n))  =  0.4805 / (m + n)   at safety = 1.02.

The binary "legal / illegal" gate of the netflow families becomes a
SIZE-PARAMETERISED law here, and it is decided symbolically, in O(1), with no
power iteration at all (`transport_opnorm`).

CONSTRUCTION GUARANTEES (both inherited from netflow, one strictly stronger)
---------------------------------------------------------------------------
FEASIBLE: `atlas.bench.netflow._finish_instance` is reused VERBATIM.  It draws
    x_feas ~ U(0, flow_scale)^{mn} first and sets b := A x_feas, so x_feas is
    primal feasible exactly.  On a bipartite graph this also forces
    sum(a) = sum(b) identically (both equal sum(x_feas)), which is the OT
    mass-balance condition -- it comes free, it is never imposed.

BOUNDED: STRONGER than netflow's.  netflow argues boundedness from c > 0
    componentwise.  Here the m source rows of A are exactly kron(I_m, 1_n^T),
    so d >= 0 with A d = 0 gives sum_j d_ij = 0 for every i, and with d >= 0
    that forces d = 0.  The recession cone of {A x = b, x >= 0} is {0}: the
    transportation polytope is COMPACT (contained in 0 <= x_ij <= min(a_i,b_j)).
    Boundedness therefore holds for ANY cost vector, so the geometric costs
    below -- which may contain zeros and are not integers -- are safe.
    (`recession_cone_is_trivial` checks this numerically.)

DETERMINISTIC in (m, n, seed, cost options): all randomness flows through one
    np.random.default_rng(seed), same convention as netflow.

THE CONDITIONING KNOB
---------------------
`gen_transport(..., cost="geometric", spread=s, dim=d, p=p)` replaces the
integer cost vector by a squared/`p`-th-power Euclidean point-cloud cost.  See
`point_cloud_costs` for the exact definition of what `spread` changes and the
measured effect on the iteration count.  b and x_feas are NOT touched by the
cost knob (dataclasses.replace on a frozen NetflowLP), so feasibility is
preserved bit-for-bit while conditioning is swept.

MEASURED (RTX 4090, local, 2026-09-09; reproduce with `python -m atlas.bench.ot`)
--------------------------------------------------------------------------------
L.norm_bound, i.e. IncidenceLinOp.from_arcs' 500-step power iteration, against
the closed form 1.02*sqrt(m+n) and against the exact dense SVD:

    (m, n)     dense svd_max   sqrt(m+n)      L.norm_bound   1.02*sqrt(m+n)  rel
    (4, 4)      2.828427125    2.828427125     2.884996       2.884996      2.9e-14
    (8, 8)      4.000000000    4.000000000     4.080000       4.080000      1.4e-14
    (8, 12)     4.472135955    4.472135955     4.561579       4.561579      4.1e-14
    (16, 16)    5.656854249    5.656854249     5.769991       5.769991      9.5e-15
    (20, 30)    7.071067812    7.071067812     7.212489       7.212489      2.6e-14
    (32, 32)    8.000000000    8.000000000     8.160000       8.160000      2.6e-14
    (64, 64)   11.313708499   11.313708499    11.539983      11.539983      3.2e-14

SCALING: norm_bound = 1.02*sqrt(m+n) to <= 4.1e-14 relative at every size, and
the dense top singular value equals sqrt(m+n) to machine precision.  Doubling N
multiplies ||L|| by sqrt(2) (4.080 -> 5.770 -> 8.160 -> 11.540 at N = 8, 16, 32,
64), so power iteration can be skipped entirely.  The mapping check
|A x - (P 1, -P^T 1)|_inf ran 8.9e-16 .. 2.8e-14 over the same sizes.

THE GATE for the two standard presets on m = n = N (lhs = t*s*||L||^2, PDHG;
prepare_base agreed with every verdict below, raising TypeCheckError on REJECT):
    banked   (t=.05, s=1.0, rho=1.9):
        N=4 0.416160 ADMIT | N=8 0.832320 ADMIT | N=16 1.664640 REJECT
        | N=32 3.329280 REJECT | N=64 6.658560 REJECT
    balanced (t=.25, s=.25, rho=1.5):
        N=4 0.520200 ADMIT | N=8 1.040400 REJECT | N=16 2.080800 REJECT
        | N=32 4.161600 REJECT | N=64 8.323200 REJECT
Both presets die by N=16; balanced already dies at N=8.  Any OT genome must
scale its steps by 1/sqrt(m+n) (see `scaled_steps`) or use a PC metric gene.

SOLVES that certify (prepare_base -> precision._run_phase, chunk K=64, tol 1e-4).
Iteration counts and gaps are bit-reproducible across runs; exec times are the
cold/warm range over two runs of the driver:
    N=4  banked   lhs=0.416160  6848 iters, gap 9.499e-05, 0.27-1.03s exec
    N=4  balanced lhs=0.520200  1280 iters, gap 7.892e-05, 0.09-0.22s exec
    N=8  banked   lhs=0.832320  4992 iters, gap 8.024e-05, 0.20-0.55s exec
ORACLE CROSS-CHECK at N=8, banked, tol 1e-8: 19520 iters to gap 8.825e-09,
PDHG objective 189.5814943553 vs HiGHS P* = 189.5814933276, relative error
5.39e-09; recovered plan satisfies |P 1 - a|_inf = 1.88e-07,
|P^T 1 - b|_inf = 1.63e-07, and sum(a) - sum(b) = 0.0 EXACTLY.

CONDITIONING SWEEP at m = n = 16 with admissible scaled steps
(tau = sigma = 0.171577, lhs = 0.980100), PDHG rho = 1, tol 1e-4:
    cost=integer                 contrast 1.4754 ->   1536 iters
    cost=geometric spread=1.00   contrast 2.3321 ->  53760 iters
    cost=geometric spread=0.30   contrast 0.9285 ->  81408 iters
    cost=geometric spread=0.10   contrast 0.3146 -> 169728 iters
    cost=geometric spread=0.03   contrast 0.0941 -> 179456 iters
So the knob spans 3.34x WITHIN the geometric family, monotone in `spread`, and
the integer/geometric mode switch is a separate and much larger 35x jump.
Note contrast is comparable only within a cost model: the integer costs have
LOWER contrast yet converge 35x faster, because contrast measures the cost
matrix's column dynamic range, not the vertex geometry of the optimal face.
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np

from atlas.bench.netflow import NetflowLP, _finish_instance
from atlas.bench import lp_cell

__all__ = [
    "bipartite_arcs",
    "transport_opnorm",
    "point_cloud_costs",
    "cost_conditioning",
    "gen_transport",
    "transport_instance",
    "transport_composite",
    "build_transport_instances",
    "construction_gate",
    "recession_cone_is_trivial",
    "COST_MODES",
]

COST_MODES = ("integer", "geometric")

#: Default ceiling (bytes) on the dense incidence matrix materialised by
#: lp_cell.from_netflow.  A is (m+n) x (m*n) float64 = 8*(m+n)*m*n bytes, i.e.
#: 16*N^3 bytes for m = n = N: 33.6 MB at N=128, 268 MB at N=256, 2.1 GB at
#: N=512.  transport_composite refuses to exceed this rather than OOM.
DENSE_LIMIT_BYTES = 512 * 1024 * 1024


# ---------------------------------------------------------------------------
# Topology
# ---------------------------------------------------------------------------


def bipartite_arcs(m: int, n: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Complete bipartite digraph K_{m,n}, arcs oriented source -> sink ONLY.

    Returns (tail, head, n_nodes) in netflow's arc-list convention, with
    n_arcs = m*n and n_nodes = m+n.  Arc e = i*n + j carries the row-major
    plan entry P_ij, so x.reshape(m, n) is the transport plan.

    Deterministic.  Unlike grid_arcs / random_regular_arcs this emits each
    edge ONCE (single orientation), which is what makes ||A|| = sqrt(m+n)
    rather than sqrt(2) times a Laplacian norm.
    """
    if m < 1 or n < 1:
        raise ValueError("optimal transport needs m >= 1 and n >= 1")
    i = np.repeat(np.arange(m), n)
    j = np.tile(np.arange(n), m)
    return i.astype(np.int64), (m + j).astype(np.int64), int(m + n)


def transport_opnorm(m: int, n: int, safety: float = 1.02) -> float:
    """Closed-form CERTIFIED bound on ||A||_2 for K_{m,n}: safety*sqrt(m+n).

    Exact because A A^T is the Laplacian of K_{m,n} (single-orientation arcs),
    whose spectrum is {0, m with multiplicity n-1, n with multiplicity m-1,
    m+n}.  ||A||_2 = sqrt(m+n) to machine precision at every (m, n) tested;
    `safety` reproduces the 1.02 inflation that lp_cell.power_iteration_norm
    applies, so this value is interchangeable with L.norm_bound and can be
    passed straight to validate(..., lp_opnorm=...).  O(1): no power
    iteration, no matrix.
    """
    return float(safety) * math.sqrt(int(m) + int(n))


# ---------------------------------------------------------------------------
# The conditioning knob
# ---------------------------------------------------------------------------


def point_cloud_costs(
    m: int,
    n: int,
    rng: np.random.Generator,
    dim: int = 2,
    spread: float = 1.0,
    p: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Geometric OT costs C_ij = ||u_i - v_j||_2^p from two point clouds.

    EXACTLY what the knobs change
    -----------------------------
    sources  u_i ~ U([0, 1]^dim)                                 (never scaled)
    sinks    v_j = 0.5*1_dim + spread * (w_j - 0.5*1_dim),  w_j ~ U([0,1]^dim)

    so `spread` is the linear shrink factor of the SINK cloud about the centre
    of the unit cube:

        spread = 1   both clouds fill the cube: generic geometric OT.
        spread -> 0  every sink collapses onto the centre, so
                     C_ij -> ||u_i - 0.5*1||^p, which is INDEPENDENT of j.
                     Every column of C becomes the same vector, hence every
                     plan with the prescribed marginals attains the same
                     objective: the optimal face of the transportation
                     polytope grows to the whole polytope in the limit.

    The measurable proxy is the column contrast reported by
    `cost_conditioning`,

        contrast(C) = mean_i (max_j C_ij - min_j C_ij) / mean(C),

    which decays linearly in `spread` (measured at m=n=16: spread 1.0 / 0.3 /
    0.1 / 0.03 -> contrast 2.3321 / 0.9285 / 0.3146 / 0.0941, and PDHG
    iterations to gap 1e-4 rise 53760 / 81408 / 169728 / 179456).  It is a
    WITHIN-COST-MODEL proxy only: the integer costs have contrast 1.4754 yet
    converge in 1536 iterations, because contrast reads the cost matrix's
    column dynamic range, not the vertex geometry of the optimal face.
    Low contrast = near-degenerate LP = slow first-order convergence, so
    `spread` sweeps CONDITIONING while
    leaving the constraint matrix, the marginals and the feasible point
    bit-for-bit unchanged (only c moves).  This is the point of the knob: it
    separates the conditioning axis from the geometry/gate axis, which depends
    only on (m, n).

    Secondary knobs, both documented as such:
      `dim`  ambient dimension.  Pairwise distances in [0,1]^dim concentrate
             around sqrt(dim/6) with O(1) fluctuation, so LARGE dim also
             lowers contrast (a second, weaker route to ill-conditioning).
      `p`    cost exponent; p = 2 is the squared-Euclidean / Wasserstein-2
             ground cost, p = 1 the Euclidean / Wasserstein-1 one.  Larger p
             stretches the cost dynamic range and RAISES contrast.

    Returns (C, u, v) with C of shape (m, n).
    """
    if not (spread > 0.0):
        raise ValueError("spread must be > 0 (spread -> 0 is the degenerate limit)")
    if dim < 1:
        raise ValueError("dim must be >= 1")
    u = rng.uniform(0.0, 1.0, size=(m, dim))
    w = rng.uniform(0.0, 1.0, size=(n, dim))
    v = 0.5 + float(spread) * (w - 0.5)
    d2 = ((u[:, None, :] - v[None, :, :]) ** 2).sum(-1)
    if p == 2.0:
        C = d2
    else:
        C = np.sqrt(np.maximum(d2, 0.0)) ** float(p)
    return C.astype(np.float64), u, v


def cost_conditioning(nf: NetflowLP) -> dict:
    """Conditioning diagnostics of a transport instance's cost matrix.

    contrast     mean_i (max_j C_ij - min_j C_ij) / mean(C)   <- the knob's target
    row_contrast same, transposed (over i for fixed j)
    dyn_range    C.max() / max(C.min(), tiny)
    cv           std(C) / mean(C)
    """
    meta = nf.meta
    m, n = int(meta["m"]), int(meta["n"])
    C = np.asarray(nf.c, dtype=np.float64).reshape(m, n)
    mean = float(C.mean())
    return {
        "contrast": float((C.max(axis=1) - C.min(axis=1)).mean() / mean),
        "row_contrast": float((C.max(axis=0) - C.min(axis=0)).mean() / mean),
        "dyn_range": float(C.max() / max(C.min(), 1e-300)),
        "cv": float(C.std() / mean),
        "c_min": float(C.min()),
        "c_max": float(C.max()),
        "c_mean": mean,
    }


# ---------------------------------------------------------------------------
# Instance generator
# ---------------------------------------------------------------------------


def gen_transport(
    m: int,
    n: int,
    seed: int,
    cost_max: int = 10,
    flow_scale: float = 2.0,
    cost: str = "integer",
    dim: int = 2,
    spread: float = 1.0,
    p: float = 2.0,
) -> NetflowLP:
    """Discrete OT instance on K_{m,n} as a NetflowLP (n_arcs = m*n).

    cost="integer"    netflow's recipe VERBATIM: c ~ U{1..cost_max} integers.
                      Both the netflow boundedness argument (c > 0) and the
                      stronger compactness argument apply.
    cost="geometric"  c = vec(C) with C from `point_cloud_costs(m, n, rng,
                      dim, spread, p)`.  Only c changes: b and x_feas come
                      from the same _finish_instance draw, so the instance is
                      still primal feasible EXACTLY (dataclasses.replace on a
                      frozen NetflowLP).  Bounded because the transportation
                      polytope is compact -- no c > 0 assumption is needed.

    The transport plan is x.reshape(m, n); the marginals are
    a = b_vec[:m] and b = -b_vec[m:], and sum(a) = sum(b) exactly.
    """
    if cost not in COST_MODES:
        raise ValueError(f"unknown cost mode {cost!r}; choose from {COST_MODES}")
    rng = np.random.default_rng(seed)
    tail, head, n_nodes = bipartite_arcs(m, n)
    meta = {
        "family": "ot_bipartite",
        "m": int(m),
        "n": int(n),
        "seed": int(seed),
        "n_nodes": int(n_nodes),
        "n_arcs": int(m * n),
        "cost": cost,
        "opnorm_exact": math.sqrt(m + n),
    }
    nf = _finish_instance(tail, head, n_nodes, rng, cost_max, flow_scale, meta)
    if cost == "integer":
        nf.meta["cost_max"] = int(cost_max)
        return nf
    C, u, v = point_cloud_costs(m, n, rng, dim=dim, spread=spread, p=p)
    gmeta = dict(meta)
    gmeta.update({"dim": int(dim), "spread": float(spread), "p": float(p),
                  "points_source": u, "points_sink": v})
    return dataclasses.replace(nf, c=C.ravel(), meta=gmeta)


def transport_marginals(nf: NetflowLP) -> tuple[np.ndarray, np.ndarray]:
    """(a, b) marginals of a transport instance, from b_vec = [a ; -b]."""
    m = int(nf.meta["m"])
    return np.asarray(nf.b[:m]), -np.asarray(nf.b[m:])


# ---------------------------------------------------------------------------
# Composite adapter (reuses lp_cell end to end)
# ---------------------------------------------------------------------------


def transport_instance(
    m: int, n: int, seed: int, dense_limit_bytes: int = DENSE_LIMIT_BYTES, **kw
):
    """LPInstance via lp_cell.from_netflow (dense A materialised: see guard)."""
    need = 8 * (m + n) * m * n
    if need > dense_limit_bytes:
        raise MemoryError(
            f"lp_cell.from_netflow materialises a dense ({m + n} x {m * n}) "
            f"incidence matrix = {need / 2**20:.0f} MB > "
            f"{dense_limit_bytes / 2**20:.0f} MB limit. The solve path never "
            "needs A_dense (only data['c'], data['b'] and problem.L); build "
            "the CompositeProblem directly with an IncidenceLinOp for large m,n."
        )
    return lp_cell.from_netflow(gen_transport(m, n, seed, **kw))


def transport_composite(
    m: int, n: int, seed: int, dense_limit_bytes: int = DENSE_LIMIT_BYTES, **kw
):
    """`lp_standard` CompositeProblem for discrete OT on K_{m,n}.

    Exactly lp_cell.from_netflow -> lp_cell.to_composite, so the problem is
    indistinguishable from a netflow instance to base_schemes (gap_kind
    "lp_gap", certificate lp_rel_gap), to prepare_base, and to lp_cell.oracle.
    problem.L is an IncidenceLinOp whose norm_bound equals
    transport_opnorm(m, n) to machine precision.

    kw is forwarded to gen_transport (cost / dim / spread / p / cost_max /
    flow_scale).
    """
    return lp_cell.to_composite(
        transport_instance(m, n, seed, dense_limit_bytes=dense_limit_bytes, **kw)
    )


def build_transport_instances(
    n_train: int, n_holdout: int, size: int, seed: int, aspect: float = 1.0, **kw
) -> tuple[list, list]:
    """(train, holdout) lists of lp_standard composites; mirrors
    lp_cell.build_instances: instance i uses derived seed seed*10007 + i,
    train = first n_train, holdout = the next n_holdout (disjoint seeds).

    size is m; n = max(1, round(size*aspect)), so rectangular OT (m != n) is
    reachable and the closed-form norm sqrt(m+n) still covers it.
    """
    m = int(size)
    n = max(1, int(round(size * aspect)))
    probs = [
        transport_composite(m, n, seed * 10007 + i, **kw)
        for i in range(n_train + n_holdout)
    ]
    return probs[:n_train], probs[n_train:]


# ---------------------------------------------------------------------------
# The construction gate, symbolically, before any execution
# ---------------------------------------------------------------------------


def construction_gate(
    m: int,
    n: int,
    tau: float,
    sigma: float,
    L_smooth: float = 0.0,
    scheme: str = "pdhg",
    safety: float = 1.02,
) -> dict:
    """Decide the PDHG / Condat-Vu side condition for OT WITHOUT building it.

    PDHG:      tau*sigma*||L||^2 < 1
    CondatVu:  tau*L_f/2 + tau*sigma*||L||^2 < 1   (L_f = 0 for the LP cost,
               so the two coincide on `lp_standard`).

    Uses the closed form ||L|| <= safety*sqrt(m+n), so the verdict is O(1) and
    exact.  Returns the lhs, the verdict, and the largest admissible step
    product at this size.
    """
    nb = transport_opnorm(m, n, safety)
    lhs = tau * L_smooth / 2.0 + tau * sigma * nb * nb
    return {
        "m": int(m),
        "n": int(n),
        "scheme": scheme,
        "tau": float(tau),
        "sigma": float(sigma),
        "opnorm": nb,
        "opnorm_exact": math.sqrt(m + n),
        "lhs": float(lhs),
        "admits": bool(tau > 0 and sigma > 0 and lhs < 1.0),
        "max_tau_sigma": float(1.0 / (nb * nb)),
    }


def scaled_steps(m: int, n: int, ratio: float = 1.0, fill: float = 0.99,
                 safety: float = 1.02) -> tuple[float, float]:
    """An always-admissible (tau, sigma) at size (m, n): the gate lhs is
    fill^2 < 1 for every m, n.  `ratio` = tau/sigma is the free gene at fixed
    product tau*sigma = fill^2/(safety^2 (m+n))."""
    nb = transport_opnorm(m, n, safety)
    r = math.sqrt(float(ratio))
    return fill * r / nb, fill / (r * nb)


def recession_cone_is_trivial(m: int, n: int, tol: float = 1e-9) -> bool:
    """Numerically confirm the boundedness proof: max sum(d) s.t. A d = 0,
    0 <= d <= 1 equals 0, i.e. the recession cone of the transportation
    polytope is {0} and the LP is bounded for ANY cost vector."""
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix

    tail, head, n_nodes = bipartite_arcs(m, n)
    e = np.arange(m * n)
    A = coo_matrix(
        (np.concatenate([np.ones(m * n), -np.ones(m * n)]),
         (np.concatenate([tail, head]), np.concatenate([e, e]))),
        shape=(n_nodes, m * n),
    ).tocsr()
    res = linprog(-np.ones(m * n), A_eq=A, b_eq=np.zeros(n_nodes),
                  bounds=(0, 1), method="highs")
    return bool(res.status == 0 and abs(res.fun) <= tol)


# ---------------------------------------------------------------------------
# Self-test / measurement driver (python -m atlas.bench.ot)
# ---------------------------------------------------------------------------


def _self_test() -> None:  # pragma: no cover - measurement driver
    import time
    import jax
    import jax.numpy as jnp
    from atlas.backend.hardware import prepare_base
    from atlas.backend import precision as P
    from atlas.genome import Genome, GenomeParams

    print("=" * 74)
    print("A) mapping A x == concat(P 1, -P^T 1) and analytic ||A||")
    print("=" * 74)
    for (m, n) in [(4, 4), (8, 8), (8, 12), (16, 16), (20, 30), (32, 32), (64, 64)]:
        nf = gen_transport(m, n, seed=0)
        inst = lp_cell.from_netflow(nf)
        A = inst.A_dense
        Pm = nf.x_feas.reshape(m, n)
        err = float(np.max(np.abs(A @ nf.x_feas -
                                  np.concatenate([Pm.sum(1), -Pm.sum(0)]))))
        sv = np.linalg.svd(A, compute_uv=False)
        nb, ana = float(inst.L.norm_bound), transport_opnorm(m, n)
        print(f"  m={m:3d} n={n:3d} |Ax-(P1,-P'1)|={err:.2e} "
              f"svd_max={sv[0]:.9f} sqrt(m+n)={math.sqrt(m+n):.9f} "
              f"norm_bound={nb:.6f} analytic={ana:.6f} rel={abs(nb-ana)/ana:.1e}")

    print("bounded (recession cone {0}) at (8,8):",
          recession_cone_is_trivial(8, 8))

    print("=" * 74)
    print("B) construction gate, banked (.05,1.,1.9) vs balanced (.25,.25,1.5)")
    print("=" * 74)
    for N in (4, 8, 16, 32, 64):
        gb = construction_gate(N, N, 0.05, 1.0)
        gl = construction_gate(N, N, 0.25, 0.25)
        print(f"  N={N:3d} ||L||={gb['opnorm']:.6f}  "
              f"banked lhs={gb['lhs']:.6f} {'ADMIT' if gb['admits'] else 'REJECT'}"
              f"   balanced lhs={gl['lhs']:.6f} "
              f"{'ADMIT' if gl['admits'] else 'REJECT'}")

    print("=" * 74)
    print("C) prepare_base + _run_phase on the 4090")
    print("=" * 74)
    presets = {
        "banked  (t=.05,s=1.0,rho=1.9)": GenomeParams(tau=.05, sigma=1., rho=1.9),
        "balanced(t=.25,s=.25,rho=1.5)": GenomeParams(tau=.25, sigma=.25, rho=1.5),
    }
    for N in (4, 8, 16):
        prob = transport_composite(N, N, seed=0)
        for label, gp in presets.items():
            gen = Genome("pdhg", gp)
            g = construction_gate(N, N, float(gp.tau), float(gp.sigma))
            try:
                built, cert, params, _ = prepare_base(prob, gen)
            except Exception as exc:
                print(f"  N={N:3d} {label}: lhs={g['lhs']:.6f} "
                      f"REJECTED by prepare_base -> {type(exc).__name__}: "
                      f"{str(exc).splitlines()[0][:90]}")
                continue
            dyn64 = jax.tree_util.tree_map(jnp.asarray, built.dyn)
            mj = jax.jit(built.metric)
            gap64 = lambda st: mj(st, dyn64)
            rho = float(gp.rho)

            def step(state, k, d):
                if built.relaxed_T is not None:
                    return built.relaxed_T(state, k, d, rho)
                Tx = built.T(state, k, d)
                return jax.tree_util.tree_map(
                    lambda a, b: (1. - rho) * a + rho * b, state, Tx)

            st0 = jax.tree_util.tree_map(
                lambda a: jnp.asarray(a, jnp.float64), built.state0)
            gaps, gi = [], []
            t0 = time.perf_counter()
            st, k, conv, _, _, e_s = P._run_phase(
                step, st0, 0, 64, 200_000, gap64, 1e-4, gaps, gi,
                dyn=dyn64, static_key=None)
            wall = time.perf_counter() - t0
            print(f"  N={N:3d} {label}: lhs={g['lhs']:.6f} ADMIT "
                  f"cert={cert.condition_str} satisfied={cert.satisfied} "
                  f"-> iters={k} gap={gaps[-1]:.3e} conv={conv} "
                  f"exec={e_s:.3f}s wall={wall:.3f}s")

    print("=" * 74)
    print("D) oracle cross-check: N=8 banked to tol 1e-8 vs HiGHS")
    print("=" * 74)
    prob = transport_composite(8, 8, seed=0)
    built, cert, params, _ = prepare_base(
        prob, Genome("pdhg", GenomeParams(tau=.05, sigma=1., rho=1.9)))
    dyn64 = jax.tree_util.tree_map(jnp.asarray, built.dyn)
    mj = jax.jit(built.metric)

    def step19(s, k, d):
        if built.relaxed_T is not None:
            return built.relaxed_T(s, k, d, 1.9)
        Tx = built.T(s, k, d)
        return jax.tree_util.tree_map(lambda a, b: -0.9 * a + 1.9 * b, s, Tx)

    st0 = jax.tree_util.tree_map(
        lambda a: jnp.asarray(a, jnp.float64), built.state0)
    gaps, gi = [], []
    st, k, conv, _, _, _ = P._run_phase(
        step19, st0, 0, 64, 500_000, lambda s: mj(s, dyn64), 1e-8,
        gaps, gi, dyn=dyn64, static_key=None)
    xp = np.maximum(np.asarray(st[0]), 0.0)
    orc = lp_cell.oracle(prob)
    obj = float(np.asarray(prob.data["c"]) @ xp)
    bvec = np.asarray(prob.data["b"])
    a_, b_ = bvec[:8], -bvec[8:]
    Pm = xp.reshape(8, 8)
    print(f"  iters={k} gap={gaps[-1]:.3e} conv={conv}")
    print(f"  PDHG obj={obj:.10f}  HiGHS P*={orc['P_star']:.10f}  "
          f"relerr={abs(obj - orc['P_star']) / (1 + abs(orc['P_star'])):.3e}")
    print(f"  |P1-a|={np.max(np.abs(Pm.sum(1) - a_)):.3e} "
          f"|P'1-b|={np.max(np.abs(Pm.sum(0) - b_)):.3e} "
          f"sum(a)-sum(b)={a_.sum() - b_.sum():.3e}")

    print("=" * 74)
    print("E) conditioning knob: spread sweep at m=n=16, admissible steps")
    print("=" * 74)
    tau, sigma = scaled_steps(16, 16)
    gen = Genome("pdhg", GenomeParams(tau=tau, sigma=sigma, rho=1.0))
    print(f"  scaled steps tau={tau:.6f} sigma={sigma:.6f} "
          f"lhs={construction_gate(16,16,tau,sigma)['lhs']:.6f}")
    for label, kw in [("integer   ", {}),
                      ("geo s=1.00", {"cost": "geometric", "spread": 1.0}),
                      ("geo s=0.30", {"cost": "geometric", "spread": 0.3}),
                      ("geo s=0.10", {"cost": "geometric", "spread": 0.1}),
                      ("geo s=0.03", {"cost": "geometric", "spread": 0.03})]:
        nf = gen_transport(16, 16, seed=0, **kw)
        cond = cost_conditioning(nf)
        prob = lp_cell.to_composite(lp_cell.from_netflow(nf))
        built, cert, params, _ = prepare_base(prob, gen)
        dyn64 = jax.tree_util.tree_map(jnp.asarray, built.dyn)
        mj = jax.jit(built.metric)
        st0 = jax.tree_util.tree_map(
            lambda a: jnp.asarray(a, jnp.float64), built.state0)
        gaps, gi = [], []
        st, k, conv, _, _, e_s = P._run_phase(
            built.T, st0, 0, 256, 400_000, lambda s: mj(s, dyn64), 1e-4,
            gaps, gi, dyn=dyn64, static_key=None)
        print(f"  {label} contrast={cond['contrast']:.4f} "
              f"cv={cond['cv']:.4f} -> iters={k} gap={gaps[-1]:.3e} conv={conv} "
              f"exec={e_s:.3f}s")


if __name__ == "__main__":  # pragma: no cover
    _self_test()
