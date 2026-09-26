"""Child process for specialist_struct.py: runs ONE specialist in ITS isolated interpreter and writes the answer.

Contract, the same one baseline_proc.py follows: every import happens BEFORE the clock starts, and the clock covers
the solver's own setup (graph construction, integer scaling, design-matrix handoff). What is reported as t_s is
therefore the whole cost of using the specialist on data already in memory, not a kernel time.

Reads --dir/in.npz, writes --dir/out.npz and one JSON line on stdout.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True)
ap.add_argument("--family", required=True, choices=("flow", "glasso", "meb"))
ap.add_argument("--solver", required=True)
ap.add_argument("--tol", type=float, default=1e-3)
a = ap.parse_args()
IN = os.path.join(a.dir, "in.npz")
OUT = os.path.join(a.dir, "out.npz")


def emit(rec):
    print(json.dumps(rec, default=float))
    sys.exit(0)


# ------------------------------------------------------------------------------------------------------------------
# min-cost network flow
# ------------------------------------------------------------------------------------------------------------------
def scale_to_int(supply, cap, cost, scale_flow, scale_cost):
    """Integer min-cost-flow data whose optimum is within O(1/scale) of the continuous one.

    Three things have to survive rounding. Supplies must still sum to zero, or the integer problem is infeasible:
    the residual is spread ONE UNIT AT A TIME over as many nodes as it takes, so no single right-hand side moves by
    more than 1.5/scale_flow rather than dumping the whole residual on one node. Capacities are rounded UP, which
    relaxes rather than tightens, so a feasible continuous flow still has an integer counterpart. Costs are rounded
    to nearest, which perturbs the objective by O(1/scale_cost) relative.
    """
    s = np.rint(np.asarray(supply, float) * scale_flow).astype(np.int64)
    resid = int(s.sum())
    if resid:
        step = -1 if resid > 0 else 1
        idx = np.argsort(-np.abs(s))[:abs(resid)]
        if idx.size < abs(resid):                          # fewer nodes than units: fall back to one bulk fix
            s[idx[0] if idx.size else 0] -= resid
        else:
            s[idx] += step
    c = np.asarray(cap, float)
    big = np.sum(np.abs(s)) + 1                            # an uncapacitated arc cannot carry more than total supply
    ci = np.where(np.isfinite(c), np.ceil(np.asarray(np.where(np.isfinite(c), c, 0.0)) * scale_flow), big)
    ci = np.minimum(ci, big).astype(np.int64)
    w = np.rint(np.asarray(cost, float) * scale_cost).astype(np.int64)
    return s, ci, w


def run_flow():
    z = np.load(IN)
    tail, head = z["tail"].astype(np.int64), z["head"].astype(np.int64)
    cost, cap, supply = z["cost"].astype(float), z["cap"].astype(float), z["supply"].astype(float)
    n_nodes, n_arcs = int(z["n_nodes"]), tail.size
    SF, SC = 1e6, 1e4                                      # flow and cost scales: see scale_to_int

    if a.solver == "pot":
        import ot                                          # noqa: F401  import before the clock
        t0 = time.perf_counter()
        # An uncapacitated transportation problem: sources are the positive supplies, sinks the negative ones, and
        # the arc set must be the complete bipartite graph for ot.emd's dense cost matrix to represent it.
        src_nodes = np.flatnonzero(supply > 0)
        snk_nodes = np.flatnonzero(supply < 0)
        ns, nk = src_nodes.size, snk_nodes.size
        if ns * nk != n_arcs or ns == 0 or nk == 0:
            emit({"unsupported": True, "error": "arc set is not the complete bipartite graph"})
        pos = np.full(n_nodes, -1, np.int64)
        pos[src_nodes] = np.arange(ns)
        pos[snk_nodes] = np.arange(nk)
        i, j = pos[tail], pos[head]
        if np.any(i < 0) or np.any(j < 0):
            emit({"unsupported": True, "error": "an arc endpoint has zero supply"})
        M = np.full((ns, nk), np.nan)
        M[i, j] = cost
        if np.isnan(M).any():
            emit({"unsupported": True, "error": "arc set is not the complete bipartite graph"})
        av, bv = supply[src_nodes], -supply[snk_nodes]
        tot = 0.5 * (av.sum() + bv.sum())
        av = av * (tot / av.sum())                         # ot.emd needs the marginals to match to its own epsilon
        bv = bv * (tot / bv.sum())
        G = ot.emd(np.ascontiguousarray(av), np.ascontiguousarray(bv), np.ascontiguousarray(M), numItermax=1000000)
        x = G[i, j]
        t = time.perf_counter() - t0
        return x, t, {"engine": "POT network simplex", "shape": [int(ns), int(nk)]}

    if a.solver == "ortools":
        from ortools.graph.python import min_cost_flow      # import before the clock
        t0 = time.perf_counter()
        s, ci, w = scale_to_int(supply, cap, cost, SF, SC)
        mcf = min_cost_flow.SimpleMinCostFlow()
        mcf.add_arcs_with_capacity_and_unit_cost(tail, head, ci, w)
        mcf.set_nodes_supplies(np.arange(n_nodes, dtype=np.int64), s)
        st = mcf.solve()
        if st != mcf.OPTIMAL:
            emit({"unsupported": True, "error": f"ortools status {st}"})
        x = np.asarray(mcf.flows(np.arange(n_arcs, dtype=np.int64)), float) / SF
        t = time.perf_counter() - t0
        return x, t, {"engine": "OR-Tools cost-scaling min-cost flow", "status": int(st)}

    if a.solver == "networkx":
        import networkx as nx                              # import before the clock
        t0 = time.perf_counter()
        s, ci, w = scale_to_int(supply, cap, cost, SF, SC)
        G = nx.DiGraph()
        G.add_nodes_from((int(v), {"demand": int(-s[v])}) for v in range(n_nodes))
        for e in range(n_arcs):
            G.add_edge(int(tail[e]), int(head[e]), capacity=int(ci[e]), weight=int(w[e]), arc=e)
        _, flow = nx.network_simplex(G)
        x = np.zeros(n_arcs)
        for uu, dd in flow.items():
            for vv, val in dd.items():
                x[G[uu][vv]["arc"]] = val / SF
        t = time.perf_counter() - t0
        return x, t, {"engine": "networkx network simplex"}
    emit({"unsupported": True, "error": f"unknown flow solver {a.solver!r}"})


# ------------------------------------------------------------------------------------------------------------------
# group lasso
# ------------------------------------------------------------------------------------------------------------------
def run_glasso():
    import scipy.sparse as sp
    from skglm import GroupLasso                           # imports before the clock
    from skglm.utils.data import grp_converter            # noqa: F401
    z = np.load(IN)
    A = sp.load_npz(os.path.join(a.dir, "A.npz")).tocsc()
    b, lam = z["b"].astype(float), float(z["lam"])
    gi, gs = z["group_index"].astype(np.int64), z["group_sizes"].astype(np.int64)
    m = A.shape[0]
    sizes = np.unique(gs)
    # skglm's descent kernels are numba, so a fresh process pays code generation on the first fit. That compile is
    # LEFT INSIDE the clock deliberately. Warming it up first was tried and rejected: it moved 3 s out of the clocked
    # time but added 6 s of wall time (t_solver 8.06 s / wall 9.23 s became 5.19 s / 15.01 s), so reporting the
    # warmed number would have been a smaller number for a more expensive run. What a first visit actually pays is
    # the compile plus the solve; a persistent worker would pay the 1.97 s solve alone.
    t0 = time.perf_counter()
    if sizes.size == 1:
        # Contiguous equal-size groups: skglm takes the size directly, but our columns are in the ORIGINAL
        # (permuted) order, so the design matrix is reordered into group order and the answer is scattered back.
        Ag = A[:, gi]
        est = GroupLasso(groups=int(sizes[0]), alpha=lam / m, tol=a.tol * 0.1, fit_intercept=False, max_iter=200)
        est.fit(Ag, b)
        w = np.zeros(A.shape[1])
        w[gi] = np.asarray(est.coef_, float).ravel()
    else:
        emit({"unsupported": True, "error": "skglm's GroupLasso wants equal-size groups"})
    t = time.perf_counter() - t0
    return w, t, {"engine": "skglm block coordinate descent", "nnz": int(np.count_nonzero(w))}


# ------------------------------------------------------------------------------------------------------------------
# minimum enclosing ball
# ------------------------------------------------------------------------------------------------------------------
def run_meb():
    """Frank-Wolfe with away steps on the MEB dual (Yildirim 2008, Algorithm 3.1 / Badoiu-Clarkson core sets).

    The dual is  max_{u in simplex}  sum_i u_i ||p_i||^2 - ||P' u||^2,  whose value is r*^2 and whose maximizer gives
    the center c = P' u. Every iterate yields a FEASIBLE primal point for free: take that c and set the radius to
    max_i ||c - p_i||, so the answer never violates a cone and the only error left is in the objective. The stopping
    rule is on exactly that error: with delta = max_i ||c - p_i||^2 / f(u) - 1 and f(u) <= r*^2, the returned radius
    overshoots by at most sqrt(1 + delta) - 1 in relative terms.

    Cost per iteration is one (n x dim) matrix-vector product, so it is linear in the sample where an interior-point
    method on the cone program factorizes a system that grows with it.
    """
    z = np.load(IN)
    P = np.ascontiguousarray(z["points"].astype(np.float64))
    n, dim = P.shape
    t0 = time.perf_counter()
    sq = np.einsum("ij,ij->i", P, P)
    u = np.zeros(n)
    if n == 1:
        return P[0], 0.0, time.perf_counter() - t0, {"iters": 0}
    i0 = 0
    d2 = sq - 2.0 * (P @ P[i0]) + sq[i0]
    ia = int(np.argmax(d2))
    d2 = sq - 2.0 * (P @ P[ia]) + sq[ia]
    ib = int(np.argmax(d2))
    u[ia] = u[ib] = 0.5
    target = a.tol * 0.5                                   # relative slack on the radius; delta ~ 2 * that
    eps = max(1e-12, 2.0 * target)
    active = {ia, ib}
    c = P.T @ u
    it = 0
    for it in range(1, 200001):
        d2 = sq - 2.0 * (P @ c) + float(c @ c)
        kp = int(np.argmax(d2))
        gmax = float(d2[kp])
        f = float(u @ sq) - float(c @ c)                   # f(u) = r_u^2, a lower bound on r*^2
        if f <= 0:
            break
        dp = gmax / f - 1.0
        act = np.fromiter(active, np.int64, len(active))
        km = int(act[np.argmin(d2[act])])
        dm = 1.0 - float(d2[km]) / f
        if max(dp, dm) <= eps:
            break
        if dp >= dm:
            lam = dp / (2.0 * (1.0 + dp))
            u *= (1.0 - lam)
            u[kp] += lam
            c = (1.0 - lam) * c + lam * P[kp]
            active = {i for i in active if u[i] > 0}
            active.add(kp)
        else:
            lam = min(dm / (2.0 * (1.0 - dm)), u[km] / max(1e-300, 1.0 - u[km]))
            u *= (1.0 + lam)
            u[km] -= lam
            c = (1.0 + lam) * c - lam * P[km]
            if u[km] <= 0:
                u[km] = 0.0
                active.discard(km)
            active = {i for i in active if u[i] > 0}
    d2 = sq - 2.0 * (P @ c) + float(c @ c)
    radius = float(np.sqrt(max(0.0, float(np.max(d2)))))   # feasible by construction: it IS the covering radius
    t = time.perf_counter() - t0
    return c, radius, t, {"iters": it, "core_set": len(active)}


if __name__ == "__main__":
    if a.family == "flow":
        x, t, info = run_flow()
        np.savez(OUT, x=np.asarray(x, float))
    elif a.family == "glasso":
        x, t, info = run_glasso()
        np.savez(OUT, x=np.asarray(x, float))
    else:
        c, r, t, info = run_meb()
        np.savez(OUT, center=np.asarray(c, float), radius=float(r))
    print(json.dumps({"t_s": t, "eps": a.tol, "info": info}, default=float))
