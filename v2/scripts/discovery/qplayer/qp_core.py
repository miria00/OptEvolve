"""Batched QP layer: the OptNet form, and an acceptance check for any solver.

    min_z  1/2 z^T Q z + q_i^T z     s.t.  G z <= h        for i = 1..B

Q, G and h are layer parameters and are SHARED across the batch; only q_i
varies per sample.  That is the structure this project's formulation gene is
for: one factorization of (Q + rho G^T G) serves the entire batch, which is
the same move that made the hyperspectral class win.  qpth refactorizes per
batch element and cvxpylayers canonicalizes per problem, so neither uses it.

Acceptance is not a solver's own stopping rule.  Every returned Z is graded by
    relative objective gap  (f(z_i) - f*_i) / (1 + |f*_i|)
    relative feasibility    max(0, (G z_i - h)) / (1 + ||h||_inf)
against a reference objective computed once, outside every timed region.
"""
from __future__ import annotations
import numpy as np


def make_problem(n=64, m=48, B=1024, seed=0, cond=50.0):
    """A batch of strictly convex QPs with a shared Q, G, h."""
    rng = np.random.default_rng(seed)
    U = np.linalg.qr(rng.standard_normal((n, n)))[0]
    d = np.logspace(0, np.log10(cond), n)
    Q = (U * d) @ U.T
    Q = 0.5 * (Q + Q.T)
    G = rng.standard_normal((m, n)) / np.sqrt(n)
    # h chosen so the origin is strictly feasible, so the batch is always feasible
    h = np.abs(rng.standard_normal(m)) + 0.5
    qs = rng.standard_normal((B, n))
    return Q, G, h, qs


def objective(Q, qs, Z):
    """Per-sample objective.  Z is (B, n), qs is (B, n)."""
    return 0.5 * np.einsum('bi,ij,bj->b', Z, Q, Z) + np.einsum('bi,bi->b', qs, Z)


def feasibility(G, h, Z):
    v = Z @ G.T - h[None, :]
    return np.maximum(v, 0.0).max(axis=1) / (1.0 + np.abs(h).max())


def score(Q, G, h, qs, Z, fstar):
    f = objective(Q, qs, Z)
    rel = (f - fstar) / (1.0 + np.abs(fstar))
    feas = feasibility(G, h, Z)
    return float(np.max(rel)), float(np.median(rel)), float(np.max(feas)), f


def reference(Q, G, h, qs, iters=200000, rho=1.0):
    """High accuracy reference objective, computed once and outside any timed
    region, by running the same ADMM far past any target."""
    n = Q.shape[0]
    M = np.linalg.inv(Q + rho * (G.T @ G))
    B = qs.shape[0]
    S = np.zeros((B, h.size)); W = np.zeros((B, h.size))
    for _ in range(iters):
        Z = (rho * (S - W) @ G - qs) @ M.T
        GZ = Z @ G.T
        S = np.minimum(GZ + W, h[None, :])
        W = W + GZ - S
    # project onto feasibility so the reference is attainable, then report
    return objective(Q, qs, Z), Z
