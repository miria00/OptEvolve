"""Min-cost network-flow LP generators (pure numpy, no jax).

Uncapacitated min-cost flow on a directed graph G = (V, E):

    min_x  c'x   s.t.   A x = b,  x >= 0

where A is the (n_nodes x n_arcs) node-arc incidence matrix
(A[tail(e), e] = +1, A[head(e), e] = -1), b the node supply vector,
c strictly positive integer arc costs.

Construction guarantees:
- FEASIBLE by construction: a nonnegative flow x_feas is drawn first and
  b := A x_feas, so x_feas is primal feasible exactly.
- BOUNDED: costs are integers in {1, ..., cost_max}, hence c > 0
  componentwise; any recession direction d >= 0, A d = 0, d != 0 has
  c'd > 0, so the LP value is finite.
- DETERMINISTIC in (family, size, seed): all randomness flows through one
  np.random.default_rng(seed).

Note A is rank-deficient (rows sum to 0); b = A x_feas is in the range
space by construction and HiGHS handles the redundancy in presolve.

Families:
- grid: k x k grid, arcs in BOTH directions between 4-neighbors
  (n_nodes = k^2, n_arcs = 4*k*(k-1)).
- regular: random d-regular undirected graph (configuration model with
  rejection of self-loops/multi-edges), each edge oriented both ways
  (n_arcs = n_nodes * d).

The jax-side LinOp wrapper and the composite mapping live in
atlas.bench.lp_cell (this module stays importable without jax).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class NetflowLP:
    """A min-cost-flow LP in arc-list form (numpy only)."""

    tail: np.ndarray  # (n_arcs,) int64, tail node of each arc (+1 row)
    head: np.ndarray  # (n_arcs,) int64, head node of each arc (-1 row)
    n_nodes: int
    c: np.ndarray  # (n_arcs,) float64, integer-valued costs >= 1
    b: np.ndarray  # (n_nodes,) float64, supplies; b = A @ x_feas
    x_feas: np.ndarray  # (n_arcs,) float64 >= 0, feasible by construction
    meta: dict = field(default_factory=dict)

    @property
    def n_arcs(self) -> int:
        return int(self.tail.shape[0])

    def incidence_dense(self) -> np.ndarray:
        """Dense (n_nodes, n_arcs) incidence matrix (oracle / small checks)."""
        A = np.zeros((self.n_nodes, self.n_arcs), dtype=np.float64)
        cols = np.arange(self.n_arcs)
        np.add.at(A, (self.tail, cols), 1.0)
        np.add.at(A, (self.head, cols), -1.0)
        return A


# ---------------------------------------------------------------------------
# Graph topologies (arc lists)
# ---------------------------------------------------------------------------


def grid_arcs(k: int) -> tuple[np.ndarray, np.ndarray, int]:
    """k x k grid, both directions between 4-neighbors. Deterministic."""
    if k < 2:
        raise ValueError("grid needs k >= 2")
    idx = np.arange(k * k).reshape(k, k)
    tails, heads = [], []
    # horizontal neighbors
    a, bb = idx[:, :-1].ravel(), idx[:, 1:].ravel()
    tails += [a, bb]
    heads += [bb, a]
    # vertical neighbors
    a, bb = idx[:-1, :].ravel(), idx[1:, :].ravel()
    tails += [a, bb]
    heads += [bb, a]
    return (
        np.concatenate(tails).astype(np.int64),
        np.concatenate(heads).astype(np.int64),
        k * k,
    )


def random_regular_arcs(
    n_nodes: int, degree: int, rng: np.random.Generator, max_tries: int = 500
) -> tuple[np.ndarray, np.ndarray, int]:
    """Random d-regular simple undirected graph, each edge oriented both ways.

    Configuration (pairing) model with rejection: shuffle the degree*n stubs,
    pair them consecutively, reject pairings with self-loops or repeated
    edges. Deterministic given rng state.
    """
    if n_nodes * degree % 2 != 0:
        raise ValueError("n_nodes * degree must be even")
    if degree >= n_nodes:
        raise ValueError("degree must be < n_nodes")
    stubs = np.repeat(np.arange(n_nodes), degree)
    for _ in range(max_tries):
        perm = rng.permutation(stubs)
        u, v = perm[0::2], perm[1::2]
        if np.any(u == v):
            continue
        lo, hi = np.minimum(u, v), np.maximum(u, v)
        keys = lo * n_nodes + hi
        if np.unique(keys).size != keys.size:
            continue
        tail = np.concatenate([u, v]).astype(np.int64)
        head = np.concatenate([v, u]).astype(np.int64)
        return tail, head, n_nodes
    raise RuntimeError(
        f"configuration model failed after {max_tries} tries "
        f"(n_nodes={n_nodes}, degree={degree})"
    )


# ---------------------------------------------------------------------------
# Instance generators
# ---------------------------------------------------------------------------


def _finish_instance(
    tail: np.ndarray,
    head: np.ndarray,
    n_nodes: int,
    rng: np.random.Generator,
    cost_max: int,
    flow_scale: float,
    meta: dict,
) -> NetflowLP:
    n_arcs = tail.shape[0]
    c = rng.integers(1, cost_max + 1, size=n_arcs).astype(np.float64)
    x_feas = rng.uniform(0.0, flow_scale, size=n_arcs)
    b = np.zeros(n_nodes, dtype=np.float64)
    np.add.at(b, tail, x_feas)
    np.add.at(b, head, -x_feas)
    return NetflowLP(
        tail=tail, head=head, n_nodes=n_nodes, c=c, b=b, x_feas=x_feas, meta=meta
    )


def gen_grid_flow(
    k: int, seed: int, cost_max: int = 10, flow_scale: float = 2.0
) -> NetflowLP:
    """Min-cost flow on a k x k grid (n_arcs = 4*k*(k-1))."""
    rng = np.random.default_rng(seed)
    tail, head, n_nodes = grid_arcs(k)
    meta = {"family": "netflow_grid", "k": int(k), "seed": int(seed),
            "n_nodes": int(n_nodes), "n_arcs": int(tail.shape[0])}
    return _finish_instance(tail, head, n_nodes, rng, cost_max, flow_scale, meta)


def gen_regular_flow(
    n_nodes: int,
    seed: int,
    degree: int = 4,
    cost_max: int = 10,
    flow_scale: float = 2.0,
) -> NetflowLP:
    """Min-cost flow on a random d-regular graph (n_arcs = n_nodes*degree)."""
    rng = np.random.default_rng(seed)
    tail, head, n_nodes = random_regular_arcs(n_nodes, degree, rng)
    meta = {"family": "netflow_regular", "n_nodes": int(n_nodes),
            "degree": int(degree), "seed": int(seed),
            "n_arcs": int(tail.shape[0])}
    return _finish_instance(tail, head, n_nodes, rng, cost_max, flow_scale, meta)
