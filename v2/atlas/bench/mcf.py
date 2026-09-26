"""Multi-commodity min-cost flow (MCF) as a standard-form LP.

This is the *structure-knob* benchmark cell: K, the number of commodities
sharing one graph, is a continuous, closed-form, PREDICTABLE knob on the
construction gate  t*s*||L||^2 < 1.  Everything else (topology, costs,
tolerance, genome) can be held fixed while K moves the admissible region
of the method axis.

------------------------------------------------------------------------
1. THE LP
------------------------------------------------------------------------
K commodities route over a shared directed graph G = (V, E) with
N = |V| nodes, E = |E| arcs, node-arc incidence A (N x E, +1 at tail,
-1 at head; the same arc lists as atlas.bench.netflow).

Primal variable, stacked commodity-major with one shared slack block:

    x = [ vec(F) ; s ] in R^n,   F in R^{K x E} (row k = commodity k),
                                 s in R^E       (capacity slacks)
    n = (K + 1) * E

Rows: K conservation blocks, then E capacity rows:

    m = K * N + E

    min  <c, x>
    s.t. A F_k^T = b_k            for k = 1..K   (flow conservation)
         sum_k F_k(e) + s_e = u_e for e in E     (shared arc capacity)
         x >= 0

which is exactly  min c'x s.t. L x = b, x >= 0  with

    L = [ I_K (x) A       |  0   ]      (K*N rows)
        [ 1_K^T (x) I_E   |  I_E ]      (E rows)

    c = [ vec(C) ; 0_E ],   C in R^{K x E} integer arc costs >= 1
    b = [ vec(B) ; u    ],  B in R^{K x N} supplies, u in R^E capacities

nnz(L) = 3*K*E + E  (asserted at build time).

The composite mapping is the *same* one as atlas.bench.lp_cell /
atlas.bench.mps_cell: f = LinearCost (L_f = 0), g = NonnegOrthant,
h = AffineEqualitySet, L = mps_cell.SparseLinOp.  name == "lp_standard",
so atlas.schemes.base_schemes dispatches gap_kind == "lp_gap" and
atlas.certify.residuals.lp_rel_gap certifies it with no new code.

The capacity rows are what make this multi-commodity: drop them and L is
block diagonal, i.e. K independent netflow instances.

------------------------------------------------------------------------
2. FEASIBILITY AND BOUNDEDNESS BY CONSTRUCTION
------------------------------------------------------------------------
FEASIBILITY (capacity="reference", the default).  Mirrors
netflow._finish_instance and never calls a solver:

  (a) draw a nonnegative reference flow  X in R^{K x E}, X_k >= 0;
  (b) set the supplies from it:  b_k := A X_k  (so X_k is exactly
      conservation-feasible for commodity k);
  (c) set the capacities generously:  u := slack_factor * sum_k X_k
      with slack_factor > 1;
  (d) the slack block is then  s_feas := u - sum_k X_k
                                       = (slack_factor - 1) * sum_k X_k >= 0,
      strictly positive on every arc carrying reference flow.

  x_feas = [vec(X) ; s_feas] therefore satisfies L x_feas = b exactly
  and x_feas >= 0.  MEASURED (grid k=8/12, K=1..4): ||L x_feas - b|| = 0.0
  exactly and min(x_feas) ~ 1.5e-4 > 0.

  CAVEAT, measured, report it with any use of this mode: generous
  capacities make the coupling INACTIVE AT THE OPTIMUM.  A dense random
  reference flow leaves slack exactly where a min-cost optimum wants to
  route, so the coupled optimum equals the sum of the K independent
  min-cost flows: grid k=8, K=4, slack_factor=1.25 gives excess 0.019%
  with 2/224 binding rows and 0 contended arcs (coupling_report()).  The
  instance is still coupled in the OPERATOR sense -- L is not block
  diagonal and ||L||/||A|| = 1.129 -- which is what the construction gate
  sees, but it is NOT a hard multi-commodity instance.  Use
  capacity="congestion" when the coupling must bind.

FEASIBILITY (capacity="congestion").  For instances whose coupling is
ACTIVE at the optimum, capacities come from a min-congestion (concurrent
flow) presolve: one HiGHS LP of the same size returns lam* and a routing
X with sum_k X_k <= lam* * w elementwise; then u := tightness * lam* * w
with tightness >= 1 and s_feas = u - sum_k X_k >= (tightness-1)*lam**w >= 0.
The presolve routing IS the feasibility certificate.  This route costs a
solver call at generation time (measured 0.03 s at grid k=8/K=4; seconds
at k=32/K=16, hence the on-disk cache), so it is not the default.
MEASURED (grid k=8, tightness=1.02): the coupling binds hard -- K=2
excess 106.2% with 22/224 binding rows; K=4 excess 84.8% with 38/224
binding and 7 contended arcs.

BOUNDEDNESS.  Stronger than netflow's, and independent of c: the feasible
set is COMPACT, so the LP value is finite for ANY cost vector (in
particular the slack columns may carry cost 0).  Proof: a recession
direction d >= 0 with L d = 0 has A d_k = 0 for each k and
sum_k d_k + d_s = 0 with every term nonnegative, hence d_k = 0 for all k
and d_s = 0.  The recession cone is {0}.

DETERMINISM.  All randomness flows through one np.random.default_rng(seed),
same contract as netflow.py.

------------------------------------------------------------------------
3. HOW ||L|| AND THE DIMENSIONS SCALE WITH K  --  CLOSED FORM
------------------------------------------------------------------------
Let sigma = ||A||_2 (the single-commodity incidence norm).  Then

    ||L||_2^2 = ( sigma^2 + K + 1
                  + sqrt( (sigma^2 - K - 1)^2 + 4*K*sigma^2 ) ) / 2

DERIVATION.  L L^T = [[ I_K (x) A A^T , 1_K (x) A ],
                      [ 1_K^T (x) A^T , (K+1) I_E ]].
Off the symmetric subspace {w_k all equal} the coupling annihilates and
L L^T acts as I (x) A A^T, eigenvalues <= sigma^2.  On the symmetric
subspace the SVD of A block-diagonalises L L^T into 2x2 blocks
[[sigma_i^2, sigma_i], [K sigma_i, K+1]] with characteristic polynomial
lam^2 - (sigma_i^2 + K + 1) lam + sigma_i^2, whose larger root increases
in sigma_i; the max is at sigma_i = sigma.  (opnorm_sq_closed_form below;
checked against dense SVD and against the power-iteration bound in the
self-test.)

CONSEQUENCES (this is the point of the cell):

    ||L||^2 = K + 1 + sigma^2 + O(1/K)      so   ||L|| = Theta(sqrt K)
    max legal step product   t*s < 1/||L||^2 ~ 1/(K + 1 + sigma^2)
                                             = Theta(1/K)

    n   = (K+1) * E        Theta(K)
    m   = K * N + E        Theta(K)
    nnz = 3*K*E + E        Theta(K)

So the *dimensions* grow linearly in K while the *admissible parameter
region* shrinks like 1/K.  A genome with a fixed (t, s) is admissible iff

    K <= K*(t, s) = max{ K : t*s * ||L||^2(sigma, K) < 1 }

which k_star() computes SYMBOLICALLY, before any execution -- the same
quantity prepare_base checks.  MEASURED on grid k=8 (see self-test):
the banked set (t=.05, s=1) is ADMITTED at K=1 and K=2 and REJECTED at
K=4; the balanced set (t=s=.25) is REJECTED at every K including K=1.

CAVEAT that must be reported alongside any K-scaling claim: the frozen
Ruiz+Pock-Chambolle substrate rescales L to unit norm, which removes the
K-dependence of the *gate* (it collapses to t*s < 1 for every K) while
leaving the K-dependence of the *iteration count*.  The K-scaling of the
gate above is a statement about UNSCALED coordinates.

------------------------------------------------------------------------
4. WHY NOT lp_cell.LPInstance
------------------------------------------------------------------------
LPInstance requires a DENSE (m, n) matrix and content-hashes it with
A.tobytes(); at grid k=32 / K=16 that is 11 GB.  This cell therefore goes
through mps_cell.StandardFormLP + mps_cell.to_composite (sparse CSR ->
BCOO SparseLinOp), which also gives data["A_csr"] for
hardware.operator_profile and solve_standard_highs() as the oracle.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from typing import Any, Optional

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog

from atlas.bench.lp_cell import power_iteration_norm
from atlas.bench.mps_cell import (
    SparseLinOp,
    StandardFormLP,
    solve_standard_highs,
)
from atlas.bench.mps_cell import to_composite as _std_to_composite
from atlas.bench.netflow import grid_arcs, random_regular_arcs
from atlas.ir.spec import CompositeProblem

TOPOLOGIES = ("grid", "regular")
CAPACITY_MODES = ("reference", "congestion")

_V2_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
CONGESTION_CACHE_DIR = os.path.join(_V2_ROOT, "data", "mcf_congestion_cache")


# ---------------------------------------------------------------------------
# Closed-form operator norm and the symbolic gate
# ---------------------------------------------------------------------------


def opnorm_sq_closed_form(sigma: float, K: int) -> float:
    """||L||_2^2 in closed form from sigma = ||A||_2 and K (docstring SS3)."""
    if K < 1:
        raise ValueError("K must be >= 1")
    s2 = float(sigma) ** 2
    kk = float(K)
    disc = (s2 - kk - 1.0) ** 2 + 4.0 * kk * s2
    return float((s2 + kk + 1.0 + np.sqrt(disc)) / 2.0)


def opnorm_closed_form(sigma: float, K: int, safety: float = 1.0) -> float:
    """||L||_2 (optionally inflated by `safety`, matching power_iteration_norm)."""
    return float(safety * np.sqrt(opnorm_sq_closed_form(sigma, K)))


def max_legal_ts(sigma: float, K: int, safety: float = 1.0) -> float:
    """sup of the admissible step product: t*s < 1 / ||L||^2."""
    return 1.0 / (safety ** 2 * opnorm_sq_closed_form(sigma, K))


def k_star(sigma: float, t: float, s: float, safety: float = 1.0,
           k_max: int = 1 << 20) -> int:
    """Largest K with t*s*||L||^2 < 1, i.e. the largest number of commodities
    at which a genome with these steps still passes the construction gate.
    Returns 0 if the genome is inadmissible even at K = 1.  Purely symbolic:
    no matrix is built, nothing is executed."""
    ts = float(t) * float(s)
    if ts <= 0.0:
        raise ValueError("t and s must be positive")
    if ts * safety ** 2 * opnorm_sq_closed_form(sigma, 1) >= 1.0:
        return 0
    lo, hi = 1, 2
    while hi <= k_max and ts * safety ** 2 * opnorm_sq_closed_form(sigma, hi) < 1.0:
        lo, hi = hi, hi * 2
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if ts * safety ** 2 * opnorm_sq_closed_form(sigma, mid) < 1.0:
            lo = mid
        else:
            hi = mid
    return int(lo)


def gate_prediction(sigma: float, K: int, t: float, s: float,
                    safety: float = 1.0, L_smooth: float = 0.0) -> dict:
    """The construction gate, predicted symbolically from (sigma, K, t, s).

    Mirrors typecheck_pdhg / typecheck_condat_vu exactly (with L_f = 0 for
    the LP linear cost the two conditions coincide)."""
    nrm2 = safety ** 2 * opnorm_sq_closed_form(sigma, K)
    lhs = t * L_smooth / 2.0 + t * s * nrm2
    return {
        "K": int(K),
        "t": float(t),
        "s": float(s),
        "opnorm": float(np.sqrt(nrm2)),
        "opnorm_sq": float(nrm2),
        "ts": float(t * s),
        "ts_opnorm_sq": float(lhs),
        "admissible": bool(t > 0 and s > 0 and lhs < 1.0),
        "max_legal_ts": float(1.0 / nrm2),
        "k_star": k_star(sigma, t, s, safety=safety),
    }


def dimensions(n_nodes: int, n_arcs: int, K: int) -> dict:
    """(n, m, nnz) of the standard form as a function of K."""
    N, E = int(n_nodes), int(n_arcs)
    return {"n": (K + 1) * E, "m": K * N + E, "nnz": 3 * K * E + E,
            "n_nodes": N, "n_arcs": E, "K": int(K)}


# ---------------------------------------------------------------------------
# Instance container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MCFlowLP:
    """A multi-commodity min-cost-flow LP in arc-list form (numpy only)."""

    tail: np.ndarray        # (E,) int64
    head: np.ndarray        # (E,) int64
    n_nodes: int
    K: int
    C: np.ndarray           # (K, E) float64, integer costs >= 1
    B: np.ndarray           # (K, N) float64 supplies; B_k = A X_k
    u: np.ndarray           # (E,) float64 capacities
    x_feas: np.ndarray      # ((K+1)*E,) float64 >= 0, L x_feas = b exactly
    sigma: float            # ||A||_2 of the single-commodity incidence (raw)
    meta: dict = field(default_factory=dict)

    # -- shapes -------------------------------------------------------------
    @property
    def n_arcs(self) -> int:
        return int(self.tail.shape[0])

    @property
    def n(self) -> int:
        return (self.K + 1) * self.n_arcs

    @property
    def m(self) -> int:
        return self.K * int(self.n_nodes) + self.n_arcs

    # -- LP data ------------------------------------------------------------
    @property
    def c(self) -> np.ndarray:
        """(n,) cost: per-commodity arc costs then zero on the slack block."""
        return np.concatenate(
            [np.asarray(self.C, dtype=np.float64).ravel(),
             np.zeros(self.n_arcs, dtype=np.float64)]
        )

    @property
    def b(self) -> np.ndarray:
        """(m,) rhs: stacked supplies then capacities."""
        return np.concatenate(
            [np.asarray(self.B, dtype=np.float64).ravel(),
             np.asarray(self.u, dtype=np.float64)]
        )

    def incidence(self) -> sp.csr_matrix:
        """(N, E) single-commodity node-arc incidence, sparse."""
        return incidence_matrix(self.tail, self.head, self.n_nodes)

    def constraint_matrix(self) -> sp.csr_matrix:
        """(m, n) constraint matrix L."""
        return mcf_constraint_matrix(self.tail, self.head, self.n_nodes, self.K)

    # -- gate ---------------------------------------------------------------
    def opnorm_closed_form(self, safety: float = 1.0) -> float:
        return opnorm_closed_form(self.sigma, self.K, safety=safety)

    def gate(self, t: float, s: float, safety: float = 1.02,
             L_smooth: float = 0.0) -> dict:
        """Symbolic construction-gate prediction for this instance."""
        return gate_prediction(self.sigma, self.K, t, s, safety=safety,
                               L_smooth=L_smooth)

    def primal_residual(self) -> float:
        """||L x_feas - b||_2  (should be ~1e-14; the feasibility check)."""
        return float(np.linalg.norm(
            self.constraint_matrix() @ self.x_feas - self.b))


# ---------------------------------------------------------------------------
# Matrix assembly
# ---------------------------------------------------------------------------


def incidence_matrix(tail: np.ndarray, head: np.ndarray,
                     n_nodes: int) -> sp.csr_matrix:
    E = int(np.asarray(tail).shape[0])
    ar = np.arange(E)
    rows = np.concatenate([np.asarray(tail), np.asarray(head)])
    cols = np.concatenate([ar, ar])
    vals = np.concatenate([np.ones(E), -np.ones(E)])
    return sp.coo_matrix((vals, (rows, cols)),
                         shape=(int(n_nodes), E)).tocsr()


def mcf_constraint_matrix(tail: np.ndarray, head: np.ndarray, n_nodes: int,
                          K: int) -> sp.csr_matrix:
    """L = [[I_K (x) A, 0], [1_K^T (x) I_E, I_E]] built directly in COO.

    Asserts nnz == 3*K*E + E, which also rules out silent duplicate-index
    cancellation (it would fire on a self-loop, tail(e) == head(e))."""
    tail = np.asarray(tail, dtype=np.int64)
    head = np.asarray(head, dtype=np.int64)
    if np.any(tail == head):
        raise ValueError("self-loop arc: incidence column would vanish")
    N, E, K = int(n_nodes), int(tail.shape[0]), int(K)
    if K < 1:
        raise ValueError("K must be >= 1")
    ar = np.arange(E, dtype=np.int64)
    kk = np.arange(K, dtype=np.int64)

    col_flow = (kk[:, None] * E + ar[None, :]).ravel()          # (K*E,)
    row_tail = (kk[:, None] * N + tail[None, :]).ravel()
    row_head = (kk[:, None] * N + head[None, :]).ravel()
    row_cap = np.tile(K * N + ar, K)                            # (K*E,)

    rows = np.concatenate([row_tail, row_head, row_cap, K * N + ar])
    cols = np.concatenate([col_flow, col_flow, col_flow, K * E + ar])
    vals = np.concatenate([np.ones(K * E), -np.ones(K * E),
                           np.ones(K * E), np.ones(E)])
    L = sp.coo_matrix((vals, (rows, cols)),
                      shape=(K * N + E, (K + 1) * E)).tocsr()
    L.sum_duplicates()
    L.eliminate_zeros()
    expected = 3 * K * E + E
    if L.nnz != expected:
        raise AssertionError(f"nnz(L) = {L.nnz}, expected {expected}")
    return L


# ---------------------------------------------------------------------------
# Topology
# ---------------------------------------------------------------------------


def mcf_arcs(topology: str, size: int, rng: np.random.Generator,
             degree: int = 4) -> tuple[np.ndarray, np.ndarray, int]:
    """Arc list for a topology, reusing atlas.bench.netflow verbatim.

    "grid":    size = grid side k, N = k^2, E = 4*k*(k-1).
    "regular": size = n_nodes, random `degree`-regular, E = n_nodes*degree.
    """
    if topology == "grid":
        return grid_arcs(int(size))
    if topology == "regular":
        return random_regular_arcs(int(size), int(degree), rng)
    raise ValueError(f"unknown topology {topology!r}; choose from {TOPOLOGIES}")


# ---------------------------------------------------------------------------
# Capacity construction
# ---------------------------------------------------------------------------


def _reference_capacities(
    A: sp.csr_matrix, E: int, K: int, rng: np.random.Generator,
    flow_scale: float, slack_factor: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Draw the reference flow, derive supplies and generous capacities.

    Returns (X (K,E), B (K,N), u (E,)).  See docstring SS2: b := A X_k and
    u := slack_factor * sum_k X_k make x_feas = [vec(X); u - sum_k X_k]
    exactly feasible with a strictly positive slack block wherever any
    reference flow is carried."""
    if slack_factor <= 1.0:
        raise ValueError("slack_factor must be > 1 for strict slack feasibility")
    X = rng.uniform(0.0, float(flow_scale), size=(K, E))
    B = (A @ X.T).T                      # (K, N), B_k = A X_k
    u = float(slack_factor) * X.sum(axis=0)
    return X, np.ascontiguousarray(B), u


def _congestion_key(topology: str, size: int, K: int, seed: int, degree: int,
                    demand_max: int, B=None, w=None) -> str:
    """Cache key for the congestion presolve.

    The generator arguments alone are NOT sufficient. B is drawn from an RNG
    stream that upstream draws (the cost matrix, sized by cost_max) have
    already advanced, so two calls agreeing on every listed argument can still
    present different supplies. Keying on the arguments alone therefore
    returned a routing for the wrong B, and the resulting instance was
    silently infeasible: primal residual 1.93e+01 with a warm cache against
    0.0 cold, with no assertion firing. Hash the LP data itself.
    """
    h = hashlib.sha256()
    h.update(repr((topology, int(size), int(K), int(seed), int(degree),
                   int(demand_max), "mcf-congestion-v2")).encode())
    if B is not None:
        h.update(np.ascontiguousarray(B, dtype=np.float64).tobytes())
    if w is not None:
        h.update(np.ascontiguousarray(w, dtype=np.float64).tobytes())
    return h.hexdigest()[:32]


def _min_congestion_routing(
    A: sp.csr_matrix, E: int, K: int, B: np.ndarray, w: np.ndarray,
    cache_key: Optional[str] = None, cache_dir: str = CONGESTION_CACHE_DIR,
) -> tuple[float, np.ndarray]:
    """Concurrent-flow presolve:  min lam s.t. A x_k = b_k, sum_k x_k <= lam*w,
    x >= 0.  Returns (lam*, X (K,E)).  One HiGHS LP of the same size as the
    instance; cached on disk by `cache_key` because it costs up to seconds."""
    if cache_key is not None:
        os.makedirs(cache_dir, exist_ok=True)
        path = os.path.join(cache_dir, f"cong_{cache_key}.npz")
        if os.path.exists(path):
            z = np.load(path)
            lam_c, X_c = float(z["lam"]), np.asarray(z["X"])
            # Never trust a cache hit: verify the routing satisfies THIS B.
            resid = float(np.abs(A @ X_c.T - B.T).max()) if X_c.size else np.inf
            if not np.isfinite(resid) or resid > 1e-7:
                raise AssertionError(
                    f"congestion cache hit is infeasible for this instance "
                    f"(max |A x_k - b_k| = {resid:.3e}); stale entry at {path}")
            return lam_c, X_c
    N = A.shape[0]
    blk = sp.kron(sp.identity(K, format="csr"), A, format="csr")   # (K*N, K*E)
    A_eq = sp.hstack([blk, sp.csr_matrix((K * N, 1))], format="csr")
    ones_blk = sp.hstack([sp.identity(E, format="csr")] * K, format="csr")
    A_ub = sp.hstack([ones_blk, -np.asarray(w).reshape(E, 1)], format="csr")
    obj = np.concatenate([np.zeros(K * E), [1.0]])
    res = linprog(obj, A_ub=A_ub, b_ub=np.zeros(E),
                  A_eq=A_eq, b_eq=B.ravel(), bounds=(0, None), method="highs")
    if res.status != 0:
        raise RuntimeError(
            f"min-congestion presolve failed (status {res.status}): {res.message}"
        )
    lam = float(res.x[-1])
    X = np.asarray(res.x[:K * E]).reshape(K, E)
    X = np.maximum(X, 0.0)
    if cache_key is not None:
        np.savez_compressed(path, lam=lam, X=X)
    return lam, X


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


def gen_mcf(
    topology: str = "grid",
    size: int = 8,
    n_commodities: int = 2,
    seed: int = 0,
    *,
    cost_max: int = 10,
    flow_scale: float = 2.0,
    slack_factor: float = 1.25,
    capacity: str = "reference",
    tightness: float = 1.02,
    demand_max: int = 10,
    degree: int = 4,
    congestion_cache_dir: str = CONGESTION_CACHE_DIR,
) -> MCFlowLP:
    """A feasible, bounded multi-commodity min-cost-flow LP.

    topology       "grid" (size = side k) or "regular" (size = n_nodes).
    n_commodities  K, the structure knob (docstring SS3).
    capacity       "reference" (default, no solver: u = slack_factor *
                   sum_k X_k over a random reference flow) or "congestion"
                   (min-congestion presolve; coupling active at the optimum).
    slack_factor   > 1; how generous the capacities are in "reference" mode.
    tightness      >= 1; capacity multiplier in "congestion" mode.

    Deterministic in (topology, size, n_commodities, seed, degree).
    """
    if capacity not in CAPACITY_MODES:
        raise ValueError(f"capacity must be one of {CAPACITY_MODES}")
    K = int(n_commodities)
    if K < 1:
        raise ValueError("n_commodities must be >= 1")
    rng = np.random.default_rng(seed)
    tail, head, n_nodes = mcf_arcs(topology, size, rng, degree=degree)
    E, N = int(tail.shape[0]), int(n_nodes)
    A = incidence_matrix(tail, head, N)

    C = rng.integers(1, int(cost_max) + 1, size=(K, E)).astype(np.float64)

    lam_star = None
    if capacity == "reference":
        X, B, u = _reference_capacities(A, E, K, rng, flow_scale, slack_factor)
    else:
        if tightness < 1.0:
            raise ValueError("tightness < 1 does not guarantee feasibility")
        # single-source / single-sink integer demands
        B = np.zeros((K, N), dtype=np.float64)
        for k in range(K):
            src, dst = rng.choice(N, size=2, replace=False)
            d = float(rng.integers(1, int(demand_max) + 1))
            B[k, src] += d
            B[k, dst] -= d
        w = np.ones(E, dtype=np.float64)
        key = _congestion_key(topology, size, K, seed, degree, demand_max, B, w)
        lam_star, X = _min_congestion_routing(
            A, E, K, B, w, cache_key=key, cache_dir=congestion_cache_dir)
        u = float(tightness) * lam_star * w

    load = X.sum(axis=0)
    s_feas = u - load
    if np.any(s_feas < -1e-12):
        raise AssertionError("slack block negative: capacities too tight")
    s_feas = np.maximum(s_feas, 0.0)
    x_feas = np.concatenate([X.ravel(), s_feas])

    # sigma = ||A||_2 (raw, no safety factor): the closed form's input.
    A_csc = A.tocsc()
    sigma = power_iteration_norm(lambda v: A @ v, lambda y: A_csc.T @ y,
                                 E, safety=1.0)

    dims = dimensions(N, E, K)
    meta = {
        "family": f"mcf_{topology}",
        "topology": topology,
        "size": int(size),
        "K": K,
        "seed": int(seed),
        "degree": int(degree) if topology == "regular" else None,
        "capacity_mode": capacity,
        "slack_factor": float(slack_factor),
        "tightness": float(tightness) if capacity == "congestion" else None,
        "lam_star": lam_star,
        "cost_max": int(cost_max),
        "n_nodes": N,
        "n_arcs": E,
        "opnorm_A": float(sigma),
        "opnorm_closed_form": opnorm_closed_form(sigma, K),
        "opnorm_sq_closed_form": opnorm_sq_closed_form(sigma, K),
        "max_legal_ts": max_legal_ts(sigma, K),
        **dims,
    }
    return MCFlowLP(tail=tail, head=head, n_nodes=N, K=K, C=C, B=B, u=u,
                    x_feas=x_feas, sigma=float(sigma), meta=meta)


# ---------------------------------------------------------------------------
# Standard form + composite
# ---------------------------------------------------------------------------


def to_standard_form(inst: MCFlowLP) -> StandardFormLP:
    """MCFlowLP -> mps_cell.StandardFormLP (sparse, min c'x s.t. Lx=b, x>=0)."""
    L = inst.constraint_matrix()
    meta = dict(inst.meta)
    meta.update(rows=inst.m, cols=inst.n, nnz=int(L.nnz))
    return StandardFormLP(
        c=inst.c, b=inst.b, A=L,
        shift_const=0.0, sense_sign=1.0, offset0=0.0,
        name=str(inst.meta.get("family", "mcf")), meta=meta,
    )


def mcf_composite(
    inst: Optional[MCFlowLP] = None,
    *,
    norm_mode: str = "power",
    safety: float = 1.02,
    **gen_kwargs: Any,
) -> CompositeProblem:
    """An 'lp_standard' CompositeProblem for a multi-commodity flow LP.

    Pass an MCFlowLP, or generator kwargs (topology=, size=,
    n_commodities=, seed=, ...) to build one on the fly.

    norm_mode "power"       -> SparseLinOp.from_scipy power iteration
                               (x`safety`), the mps_cell default.
              "closed_form" -> the analytic ||L|| of docstring SS3, x`safety`;
                               skips power iteration entirely.

    The result is directly usable by atlas.backend.hardware.prepare_base:
    f = LinearCost (L_f = 0), g = NonnegOrthant, h = AffineEqualitySet,
    gap_kind == "lp_gap", data["A_csr"] present for operator_profile.
    """
    if inst is None:
        inst = gen_mcf(**gen_kwargs)
    elif gen_kwargs:
        raise TypeError("pass either an MCFlowLP or generator kwargs, not both")
    std = to_standard_form(inst)
    prob = _std_to_composite(std)
    if norm_mode == "closed_form":
        from jax.experimental import sparse as jsparse

        L_csr = std.A
        L_op = SparseLinOp(
            A=jsparse.BCOO.from_scipy_sparse(L_csr),
            AT=jsparse.BCOO.from_scipy_sparse(L_csr.T.tocsr()),
            norm_bound=opnorm_closed_form(inst.sigma, inst.K, safety=safety),
            shape=tuple(L_csr.shape),
        )
        prob = CompositeProblem(
            f=prob.f, g=prob.g, h=prob.h, L=L_op, data=prob.data,
            shape=prob.shape, name=prob.name,
        )
    elif norm_mode != "power":
        raise ValueError("norm_mode must be 'power' or 'closed_form'")
    prob.data["meta"].update(
        norm_mode=norm_mode,
        norm_bound=float(prob.L.norm_bound),
        opnorm_A=float(inst.sigma),
        opnorm_closed_form_safe=opnorm_closed_form(inst.sigma, inst.K,
                                                   safety=safety),
    )
    prob.data["instance"] = inst
    return prob


def build_instances(
    n_train: int, n_holdout: int, topology: str, size: int,
    n_commodities: int, seed: int, **gen_kwargs: Any,
) -> tuple[list, list]:
    """(train, holdout) lists of lp_standard composites; mirrors
    lp_cell.build_instances (instance i uses derived seed seed*10007 + i,
    train = the first n_train, holdout = the next n_holdout)."""
    probs = [
        mcf_composite(gen_mcf(topology=topology, size=size,
                              n_commodities=n_commodities,
                              seed=seed * 10007 + i, **gen_kwargs))
        for i in range(n_train + n_holdout)
    ]
    return probs[:n_train], probs[n_train:]


# ---------------------------------------------------------------------------
# Oracle + coupling diagnostic
# ---------------------------------------------------------------------------


def solve_mcf_highs(inst: MCFlowLP) -> dict:
    """HiGHS reference solve of the coupled standard form."""
    return solve_standard_highs(to_standard_form(inst))


def decoupled_optimum(inst: MCFlowLP) -> dict:
    """Sum of the K INDEPENDENT min-cost flows (capacity rows dropped).

    If the coupled optimum equals this, the capacity coupling is inactive
    at the optimum and the instance is 'K independent netflow problems' in
    the optimization sense (it is still coupled in the OPERATOR sense: L is
    not block diagonal and ||L|| > ||A||)."""
    A = inst.incidence()
    total, per_k = 0.0, []
    for k in range(inst.K):
        res = linprog(inst.C[k], A_eq=A, b_eq=inst.B[k], bounds=(0, None),
                      method="highs")
        if res.status != 0:
            raise RuntimeError(f"independent flow k={k} failed: {res.message}")
        per_k.append(float(res.fun))
        total += float(res.fun)
    return {"sum_independent": total, "per_commodity": per_k}


def coupling_report(inst: MCFlowLP, tol: float = 1e-7) -> dict:
    """How much the shared capacity actually binds.

    excess       = (P_coupled - sum_independent) / sum_independent
    n_binding    = arcs with u_e - sum_k x_k(e) < tol at the coupled optimum
    n_contended  = binding arcs carrying flow from >= 2 commodities
    opnorm_ratio = ||L|| / ||A||  (the OPERATOR-sense coupling, nonzero even
                   when excess == 0)
    """
    cpl = solve_mcf_highs(inst)
    ind = decoupled_optimum(inst)
    E, K = inst.n_arcs, inst.K
    x = np.asarray(cpl["x"])
    F = x[:K * E].reshape(K, E)
    load = F.sum(axis=0)
    slack = inst.u - load
    binding = slack < tol
    contended = binding & ((F > tol).sum(axis=0) >= 2)
    denom = ind["sum_independent"] if ind["sum_independent"] != 0 else 1.0
    return {
        "P_coupled": cpl["std_fun"],
        "sum_independent": ind["sum_independent"],
        "excess": (cpl["std_fun"] - ind["sum_independent"]) / denom,
        "n_binding": int(binding.sum()),
        "n_contended": int(contended.sum()),
        "n_arcs": E,
        "opnorm_A": inst.sigma,
        "opnorm_L": inst.opnorm_closed_form(),
        "opnorm_ratio": inst.opnorm_closed_form() / inst.sigma,
    }


def gate_table(sigma: float, Ks, genomes: dict, safety: float = 1.02) -> list:
    """Symbolic admissibility table: one row per (genome, K).

    genomes: {name: (t, s)}.  Nothing is built and nothing is executed --
    this is the prediction that prepare_base then confirms."""
    rows = []
    for name, (t, s) in genomes.items():
        ks = k_star(sigma, t, s, safety=safety)
        for K in Ks:
            g = gate_prediction(sigma, K, t, s, safety=safety)
            g["genome"] = name
            g["k_star"] = ks
            rows.append(g)
    return rows
