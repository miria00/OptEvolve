"""Five structured LP classes in the solver's (P, q, r, A, l, u) form, plus the adapters the unified pipeline needs
to run an LP (2026-09-15).

Every generator draws a feasible point FIRST and sets right-hand sides and capacities from it, so feasibility never
depends on a solver, and every feasible set is COMPACT (each variable is bounded by its own bound rows or by a
nonnegative row sum), so the LP is bounded for ANY cost vector. The returned dict carries that point as "x_feas".
Variable bounds are identity rows of A (the OSQP convention of realdata.load_qplib), P is the zero matrix.

    make_lp(family, size, seed=0, **difficulty)

STRUCTURE REGISTRY: which constraint blocks are absorbable into the proximal step as a product of box-hyperplane or
box-slab sets {lo <= x_T <= hi, a'x_T in [b_lo, b_hi]}. uni.plan_product absorbs a DISJOINT family of rows whose whole
support carries finite bound rows, taken greedily by decreasing row length (ties by row index), so which family it
picks is decided by nnz per row, not by what the family means.

  transport     (a) discrete OT on K_{m,n} (reuses atlas.bench.ot.gen_transport, n_var = m n)
                rows:  m source rows  sum_j x_ij = a_i   (equality, nnz n)
                       n sink rows    sum_i x_ij = b_j   (equality, nnz m)
                       x >= 0 identity rows
                absorbable: EITHER marginal family is a product of scaled simplices covering every variable (cover 1);
                the two families overlap, so exactly one can move. The greedy takes the LONGER rows (sources when
                n > m); for square OT permutation order decides. What stays in L is the other marginal.
  netflow_cap   (b) min-cost flow with arc capacities (reuses atlas.bench.netflow grid/regular arc lists)
                rows:  node conservation  sum_out x - sum_in x = b_v  (equality, nnz = degree, coefficients +-1)
                       0 <= x_e <= cap_e identity rows
                absorbable: node rows are box-hyperplane sets, but every arc lies in two node rows, so a disjoint
                family is an independent set of nodes. The grid is bipartite, so one color class of the
                checkerboard covers every arc; the greedy does not find it and stops at cover 0.80-0.91 (measured,
                grid 8/16/32 on permuted encodings), leaving the uncovered arcs' node rows in L.
  mcf_joint     (c) multi-commodity flow with joint capacities (reuses atlas.bench.mcf.gen_mcf, n_var = K E)
                rows:  K N conservation rows (equality, nnz = degree)
                       E joint capacity rows  sum_k F_k(e) <= cap_e  (box-slab, nnz K)
                       F >= 0 identity rows
                absorbable: the capacity rows are a disjoint box-slab family covering every variable (cover 1);
                the conservation rows of all commodities are a second, partial family. The greedy takes capacity
                rows only when K exceeds the row length of a node (4, 6, 8 on a grid); at the default K = 4 it
                takes an independent set of node rows instead (measured cover 1.0 at grid 4, 0.80-0.82 at 8 and
                12), so the joint capacity rows, the coupling that makes the problem multi-commodity, stay in L.
  gub_packing   (d) packing / assignment with GUB rows, n_var = groups * group_size
                rows:  one GUB row per group  sum_{j in G} x_j <= 1  (or = 1 with gub="eq"; nnz group_size)
                       n_knap knapsack rows  w_i'x <= cap_i  (box-slab, nnz about density * n_var)
                       0 <= x <= 1 identity rows
                absorbable: the GUB rows are a disjoint box-slab family covering every variable (cover 1), and each
                knapsack row is a box-slab set on its support. The greedy takes ONE knapsack row first (the longest
                row) and then only the GUB rows off its support, so on dense knapsacks it absorbs a single dense row
                and leaves every GUB row in L.
  production    (e) multi-period production planning with inventory, n_var = 2 P T
                rows:  balance  s_{i,t-1} + p_it - s_it = d_it   (equality, nnz 3; 2 at t = 1): the time coupling
                       resource capacity  sum_i a_ki p_it <= C_kt  (box-slab, nnz P)
                       0 <= p <= pmax, 0 <= s <= smax identity rows
                absorbable: for ONE resource k the capacity rows over t are disjoint and cover all production
                variables (cover 1/2); balance rows are box-hyperplane sets but consecutive periods share s_it, so
                only a staggered subset is disjoint. The greedy takes resource 1's capacity rows when P > 3.

  Existing atlas.bench families, for reference: ot / netflow / mcf are the uncapacitated or slack-form standard
  LPs of (a)-(c) (x >= 0 is their only bound, so in standard form the equality rows are box-hyperplane sets with
  hi = +inf); lp_cell's dense random LP has no absorbable row (every row touches every column, so a disjoint
  family has one row); mps_cell instances are whatever their rows are, which plan_product reads directly.

LP ADAPTERS. The pipeline was written for QPs and two of its entry points refuse an LP (no shared file is edited here):
  uni.diagnose and uni.build_formulation("std_ruiz") go through qp_cell.to_composite, which raises on P.nnz == 0.
      lp_diagnose and lp_formulation rebuild the same objects (the same kkt_ruiz_equilibrate call, the same
      scaled A) without that check.
  uni.measure_ladder sets the primal step t = 1/L_f and the dual step s = 0.49/(t ||L||^2). For an LP L_f = 0, and
      uni.build_formulation's relocated path substitutes 1e-12, which gives t = 1e12 and s ~ 1e-12: a primal weight
      of 1e12, not a step rule. Any nonnegative number is a valid Lipschitz constant of a constant gradient, so
      lp_formulation DECLARES L_f = omega * ||L|| for the formulation's own ||L||. That makes t = 1/(omega ||L||) and
      s = 0.49 omega / ||L||: the balanced PDHG step at omega = 1, with t s ||L||^2 = 0.49 where an LP would admit up
      to 1 (the gate still reads 0.5 + 0.49 < 1). The same rule applies to every formulation, so the factor is not
      a between-formulation advantage.
"""
from __future__ import annotations

import dataclasses
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np
import scipy.sparse as sp

import uni
import jax.numpy as jnp
from atlas.bench.mcf import gen_mcf, incidence_matrix
from atlas.bench.mps_cell import SparseLinOp
from atlas.bench.netflow import gen_grid_flow, gen_regular_flow
from atlas.bench.ot import gen_transport
from atlas.ir.spec import CompositeProblem
from atlas.operators.base import PropSet
from atlas.operators.prox import Box, QuadraticForm, Zero
from atlas.schemes.rescale import kkt_ruiz_equilibrate

FAMILIES = ("transport", "netflow_cap", "mcf_joint", "gub_packing", "production")


def _osqp(G, gl, gu, lo, hi, q, x_feas, name, meta):
    """General rows G (gl <= Gx <= gu) plus one identity row per variable with a finite bound."""
    G = sp.csr_matrix(G)
    n = G.shape[1]
    lo, hi = np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
    b = np.flatnonzero(np.isfinite(lo) | np.isfinite(hi))
    B = sp.csr_matrix((np.ones(b.size), (np.arange(b.size), b)), shape=(b.size, n))
    meta = dict(meta, general_rows=int(G.shape[0]), bound_rows=int(b.size))
    return {"P": sp.csc_matrix((n, n)), "q": np.asarray(q, dtype=float), "r": 0.0,
            "A": sp.vstack([G, B], format="csc"), "l": np.concatenate([gl, lo[b]]), "u": np.concatenate([gu, hi[b]]),
            "x_feas": np.asarray(x_feas, dtype=float), "name": name, "meta": meta}


# ----------------------------------------------------------------------------------------------------------------
# (a) transport
# ----------------------------------------------------------------------------------------------------------------
def transport(size, seed=0, aspect=1.0, spread=1.0, p=2.0, dim=2, cost="geometric", flow_scale=2.0):
    """Discrete OT with m = size sources and n = round(aspect * size) sinks.

    SIZE: size (n_var = m n). DIFFICULTY: spread, the sink-cloud shrink of ot.point_cloud_costs; spread -> 0 makes
    every column of the cost matrix identical, so the optimal face grows toward the whole polytope (degenerate, slow
    for first-order methods). p is a secondary cost-range knob. Feasible: the marginals are the row and column sums
    of ot's reference plan x_feas >= 0. Bounded: 0 <= x_ij <= min(a_i, b_j)."""
    m, n = int(size), max(1, int(round(size * aspect)))
    nf = gen_transport(m, n, seed, flow_scale=flow_scale, cost=cost, dim=dim, spread=spread, p=p)
    N = m * n
    arcs = np.arange(N)
    # ot's incidence carries -1 on sink rows; positive coefficients state the same marginal and read naturally
    R = sp.csr_matrix((np.ones(N), (nf.tail, arcs)), shape=(m, N))
    C = sp.csr_matrix((np.ones(N), (nf.head - m, arcs)), shape=(n, N))
    a, b = R @ nf.x_feas, C @ nf.x_feas
    rhs = np.concatenate([a, b])
    return _osqp(sp.vstack([R, C]), rhs, rhs, np.zeros(N), np.full(N, np.inf), nf.c, nf.x_feas,
                 f"transport_{m}x{n}", {"family": "transport", "size": int(size), "seed": int(seed), "m": m, "n": n,
                                        "spread": float(spread), "p": float(p), "cost": cost})


# ----------------------------------------------------------------------------------------------------------------
# (b) capacitated min-cost network flow
# ----------------------------------------------------------------------------------------------------------------
def netflow_cap(size, seed=0, topology="grid", tightness=0.5, cost_range=None, cost_max=10, flow_scale=2.0, degree=4):
    """Min-cost flow on netflow's grid (size = side k) or random regular graph (size = n_nodes) with capacities.

    SIZE: size. DIFFICULTY: tightness in [0, 1], cap_e = x_feas_e + (1 - tightness) flow_scale w_e with
    w_e ~ U(0.5, 1.5): at 1 every arc's capacity equals the reference flow, so many capacities bind; at 0 they rarely
    do. cost_range R > 1 replaces the integer costs by log-uniform costs on [1, R] (cost conditioning); small
    cost_max gives many equal-cost paths (degenerate). Feasible: supplies b = A x_feas and 0 <= x_feas <= cap.
    Bounded: box."""
    if not 0.0 <= tightness <= 1.0:
        raise ValueError("tightness must lie in [0, 1]")
    if topology == "grid":
        nf = gen_grid_flow(int(size), seed, cost_max=cost_max, flow_scale=flow_scale)
    elif topology == "regular":
        nf = gen_regular_flow(int(size), seed, degree=degree, cost_max=cost_max, flow_scale=flow_scale)
    else:
        raise ValueError(topology)
    E = nf.n_arcs
    rng = np.random.default_rng([seed, 2])                          # own stream: netflow's draws stay bit-identical
    cap = nf.x_feas + (1.0 - tightness) * flow_scale * rng.uniform(0.5, 1.5, E)
    c = nf.c if cost_range is None else np.exp(rng.uniform(0.0, np.log(float(cost_range)), E))
    Inc = incidence_matrix(nf.tail, nf.head, nf.n_nodes)
    return _osqp(Inc, nf.b, nf.b, np.zeros(E), cap, c, nf.x_feas, f"netflow_cap_{topology}{size}",
                 {"family": "netflow_cap", "size": int(size), "seed": int(seed), "topology": topology,
                  "n_nodes": int(nf.n_nodes), "n_arcs": int(E), "tightness": float(tightness),
                  "cost_range": cost_range, "cost_max": int(cost_max)})


# ----------------------------------------------------------------------------------------------------------------
# (c) multi-commodity flow with joint capacities
# ----------------------------------------------------------------------------------------------------------------
def mcf_joint(size, seed=0, n_commodities=4, topology="grid", capacity="congestion", tightness=1.05,
              slack_factor=1.25, cost_max=10, demand_max=10, degree=4):
    """K commodities on one graph, each with its own conservation rows, sharing arc capacities.

    SIZE: size (grid side or n_nodes) and n_commodities. DIFFICULTY: capacity="congestion" sets cap = tightness *
    lambda* from mcf's min-congestion presolve, so tightness -> 1 makes the joint rows bind (mcf measured 38 binding
    rows of 224 at grid 8, K = 4, tightness 1.02); capacity="reference" with slack_factor > 1 is the loose regime
    whose coupling is inactive at the optimum. Feasible: the presolve routing (or reference flow) x_feas. Bounded:
    F >= 0 and sum_k F_k(e) <= cap_e bound every variable. The slack block of mcf's standard form is dropped: the
    capacity rows are written as inequalities, which is the box-slab form the planner recognizes."""
    inst = gen_mcf(topology, int(size), int(n_commodities), seed, cost_max=cost_max, capacity=capacity,
                   tightness=tightness, slack_factor=slack_factor, demand_max=demand_max, degree=degree)
    K, E, N = inst.K, inst.n_arcs, int(inst.n_nodes)
    Inc = incidence_matrix(inst.tail, inst.head, N)
    cons = sp.kron(sp.identity(K, format="csr"), Inc, format="csr")         # row block k, columns k*E + e
    capr = sp.hstack([sp.identity(E, format="csr")] * K, format="csr")      # sum_k F[k, e]
    nv = K * E
    gl = np.concatenate([inst.B.ravel(), np.full(E, -np.inf)])
    gu = np.concatenate([inst.B.ravel(), inst.u])
    return _osqp(sp.vstack([cons, capr]), gl, gu, np.zeros(nv), np.full(nv, np.inf), inst.C.ravel(),
                 inst.x_feas[:nv], f"mcf_joint_{topology}{size}_K{K}",
                 {"family": "mcf_joint", "size": int(size), "seed": int(seed), "K": K, "n_nodes": N, "n_arcs": E,
                  "capacity": capacity, "tightness": float(tightness), "lam_star": inst.meta.get("lam_star")})


# ----------------------------------------------------------------------------------------------------------------
# (d) packing / assignment with GUB rows
# ----------------------------------------------------------------------------------------------------------------
def gub_packing(size, seed=0, group_size=5, n_knap=4, knap_density=1.0, tightness=0.5, correlation=0.0, gub="le"):
    """max v'x over GUB groups and coupling knapsacks, as min -v'x.

    SIZE: size = number of groups (n_var = size * group_size). DIFFICULTY: tightness in [0, 1] places each capacity
    between the weight of the reference point (1, binding) and the largest weight any GUB-feasible point can carry
    (0, never binding); correlation in [0, 1] mixes the values toward the aggregate weight, so at 1 the value/weight
    ratios tie and the LP is degenerate (the strongly correlated knapsack). gub="eq" turns the groups into
    assignment rows. Feasible: x_feas = 1/group_size on every item meets every GUB row with equality and every
    capacity by construction. Bounded: box [0, 1]."""
    if gub not in ("le", "eq"):
        raise ValueError("gub must be 'le' or 'eq'")
    if not (0.0 <= tightness <= 1.0 and 0.0 <= correlation <= 1.0):
        raise ValueError("tightness and correlation must lie in [0, 1]")
    rng = np.random.default_rng(seed)
    G, g, r = int(size), int(group_size), int(n_knap)
    n = G * g
    group = np.repeat(np.arange(G), g)
    W = rng.uniform(0.1, 1.0, (r, n)) * (rng.random((r, n)) < knap_density)
    for i in range(r):                                               # an empty knapsack row would constrain nothing
        if not W[i].any():
            W[i, rng.integers(n)] = rng.uniform(0.1, 1.0)
    x_feas = np.full(n, 1.0 / g)
    use = W @ x_feas
    wmax = np.array([np.max(W[i].reshape(G, g), axis=1).sum() for i in range(r)])
    cap = use + (1.0 - tightness) * (wmax - use)
    wbar = W.mean(axis=0) / max(W.mean(), 1e-300)
    v = correlation * wbar + (1.0 - correlation) * rng.uniform(0.1, 1.0, n)
    Gm = sp.csr_matrix((np.ones(n), (group, np.arange(n))), shape=(G, n))
    gl = np.concatenate([np.ones(G) if gub == "eq" else np.full(G, -np.inf), np.full(r, -np.inf)])
    gu = np.concatenate([np.ones(G), cap])
    return _osqp(sp.vstack([Gm, sp.csr_matrix(W)]), gl, gu, np.zeros(n), np.ones(n), -v, x_feas,
                 f"gub_packing_{G}x{g}", {"family": "gub_packing", "size": G, "seed": int(seed), "group_size": g,
                                          "n_knap": r, "knap_density": float(knap_density),
                                          "tightness": float(tightness), "correlation": float(correlation),
                                          "gub": gub})


# ----------------------------------------------------------------------------------------------------------------
# (e) production planning with inventory
# ----------------------------------------------------------------------------------------------------------------
def production(size, seed=0, n_products=4, n_resources=2, tightness=0.5, seasonality=0.5, hold_ratio=0.02,
               pmax_mult=2.0):
    """Multi-period, multi-product production planning.

    min sum c_it p_it + h_i s_it  s.t.  s_{i,t-1} + p_it - s_it = d_it,  sum_i a_ki p_it <= C_kt,
                                        0 <= p_it <= pmax_i,  0 <= s_it <= smax_i,  s_{i,0} = 0.
    SIZE: size = T periods (n_var = 2 n_products T). DIFFICULTY: tightness in [0, 1] places C_kt between the
    resource use of the level plan (1: capacity binds every period, so demand peaks must be built ahead) and the use
    at pmax (0: never binds); seasonality in [0, 1] is the demand amplitude, which is what forces inventory across
    periods; hold_ratio scales holding against production cost (small makes long inventory chains cheap, many
    near-ties between periods). Feasible: the level plan p_i = max_t cumulative demand / t, whose inventory
    t p_i - cumulative demand is nonnegative; demand is recomputed from the plan so the balance rows hold to
    rounding. Bounded: box."""
    rng = np.random.default_rng(seed)
    T, P, R = int(size), int(n_products), int(n_resources)
    t = np.arange(1, T + 1)
    period = min(T, 12)
    base = rng.uniform(5.0, 15.0, P)
    phase = rng.uniform(0.0, 2 * np.pi, P)
    d = base[:, None] * (1.0 + seasonality * np.sin(2 * np.pi * t[None, :] / period + phase[:, None])) \
        * rng.uniform(0.9, 1.1, (P, T))
    p_level = np.max(np.cumsum(d, axis=1) / t[None, :], axis=1)
    s_feas = np.maximum(t[None, :] * p_level[:, None] - np.cumsum(d, axis=1), 0.0)
    s_prev = np.concatenate([np.zeros((P, 1)), s_feas[:, :-1]], axis=1)
    d = s_prev + p_level[:, None] - s_feas                                  # exact balance for the level plan
    pmax = pmax_mult * p_level
    smax = 1.5 * s_feas.max(axis=1) + 0.1 * base
    a = rng.uniform(0.5, 1.5, (R, P))
    lev, top = a @ p_level, a @ pmax
    C = lev[:, None] + (1.0 - tightness) * (top - lev)[:, None] * rng.uniform(0.8, 1.0, (R, T))
    c_prod = rng.uniform(1.0, 2.0, P)[:, None] * (1.0 + 0.3 * np.sin(2 * np.pi * t[None, :] / period
                                                                     + rng.uniform(0, 2 * np.pi, P)[:, None]))
    hold = hold_ratio * c_prod.mean(axis=1)
    pi = lambda i, k: i * T + k                                             # p_ik, then s_ik offset by P T
    I, J, V = [], [], []
    row = 0
    for i in range(P):
        for k in range(T):
            I += [row, row]; J += [pi(i, k), P * T + pi(i, k)]; V += [1.0, -1.0]
            if k > 0:
                I.append(row); J.append(P * T + pi(i, k - 1)); V.append(1.0)
            row += 1
    for r_ in range(R):
        for k in range(T):
            I += [row] * P; J += [pi(i, k) for i in range(P)]; V += a[r_].tolist()
            row += 1
    Gm = sp.csr_matrix((V, (I, J)), shape=(row, 2 * P * T))
    gl = np.concatenate([d.ravel(), np.full(R * T, -np.inf)])
    gu = np.concatenate([d.ravel(), C.ravel()])
    lo = np.zeros(2 * P * T)
    hi = np.concatenate([np.repeat(pmax, T), np.repeat(smax, T)])
    q = np.concatenate([c_prod.ravel(), np.repeat(hold, T)])
    x_feas = np.concatenate([np.repeat(p_level, T), s_feas.ravel()])
    return _osqp(Gm, gl, gu, lo, hi, q, x_feas, f"production_T{T}_P{P}",
                 {"family": "production", "size": T, "seed": int(seed), "n_products": P, "n_resources": R,
                  "tightness": float(tightness), "seasonality": float(seasonality), "hold_ratio": float(hold_ratio)})


_GEN = {"transport": transport, "netflow_cap": netflow_cap, "mcf_joint": mcf_joint, "gub_packing": gub_packing,
        "production": production}


def make_lp(family, size, seed=0, **difficulty):
    """OSQP-form LP dict of one family; extra keys: x_feas (the construction's feasible point), name, meta."""
    if family not in _GEN:
        raise ValueError(f"unknown LP family {family!r}; have {FAMILIES}")
    return _GEN[family](size, seed=seed, **difficulty)


def row_violation(d, x):
    """Per-row relative violation max_i viol_i / (1 + |bound_i|), the acceptance metric of uni.accept."""
    Ax = sp.csc_matrix(d["A"]) @ x
    l, u = np.asarray(d["l"]), np.asarray(d["u"])
    vl = np.where(np.isfinite(l), np.maximum(l - Ax, 0) / (1 + np.abs(np.where(np.isfinite(l), l, 0))), 0)
    vu = np.where(np.isfinite(u), np.maximum(Ax - u, 0) / (1 + np.abs(np.where(np.isfinite(u), u, 0))), 0)
    return float(np.max(np.maximum(vl, vu))) if Ax.size else 0.0


# ----------------------------------------------------------------------------------------------------------------
# LP adapters for the pipeline (see module docstring)
# ----------------------------------------------------------------------------------------------------------------
def lp_diagnose(d):
    """uni.diagnose without qp_cell.to_composite: the same 10-pass KKT Ruiz and the same firing rule."""
    d1, d2 = kkt_ruiz_equilibrate(sp.csc_matrix(d["P"]), sp.csc_matrix(d["A"]), 10)
    As = (sp.diags(d1) @ sp.csc_matrix(d["A"]) @ sp.diags(d2)).tocsc()
    S, U = uni.top_spectrum(As)
    mass = U[:, 0] ** 2
    j = int(np.argmax(mass))
    gap = float(S[0] / max(S[1], 1e-30))
    return {"top_row": j, "row_mass": float(mass[j]), "singular_gap": gap,
            "fires": bool(mass[j] > 0.5 and gap >= 3.0)}


def _declared_props(normL, omega):
    Ld = float(omega) * max(float(normL), 1e-12)
    return PropSet(lipschitz=Ld, cocoercivity=1.0 / Ld, prox_friendly=False, strong_convexity=0.0)


def lp_formulation(d, cand, plan=None, impl="bisect", omega=1.0, **kw):
    """uni.build_formulation for an LP, with L_f declared as omega * ||L|| (module docstring). QPs pass through."""
    if sp.csc_matrix(d["P"]).nnz:
        return uni._build_formulation(d, cand, plan=plan, impl=impl, **kw)
    n = d["P"].shape[0]
    if cand in ("std_raw", "std_ruiz"):
        t0 = time.perf_counter()
        A = sp.csc_matrix(d["A"])
        q, l, u = (np.asarray(d[k], dtype=float) for k in ("q", "l", "u"))
        if cand == "std_ruiz":
            d1, d2 = kkt_ruiz_equilibrate(sp.csc_matrix(d["P"]), A, 10)        # rescale_qp_problem's call
            A = (sp.diags(d1) @ A @ sp.diags(d2)).tocsc()
            q, l, u = d2 * q, d1 * l, d1 * u
        else:
            d2 = np.ones(n)
        L = SparseLinOp.from_scipy(A, power_iters=uni.POWER_ITERS)
        P0 = sp.csc_matrix((n, n))
        f = QuadraticForm(P=SparseLinOp.from_scipy(P0, power_iters=uni.POWER_ITERS), q=jnp.asarray(q), r=d["r"],
                          props=_declared_props(L.norm_bound, omega))
        prob = CompositeProblem(f=f, g=Zero(), h=Box(lo=jnp.asarray(l), hi=jnp.asarray(u)), L=L,
                                data={"P": P0, "q": q, "r": d["r"], "A": A, "l": l, "u": u},
                                shape=(n,), name="qp_standard")
        return uni.Formulation(cand, prob, lambda z, d2=d2: d2 * z, A, np.arange(n), time.perf_counter() - t0, 0)
    form = uni._build_formulation(d, cand, plan=plan, impl=impl, **kw)
    f = dataclasses.replace(form.prob.f, props=_declared_props(form.prob.L.norm_bound, omega))
    form.prob = dataclasses.replace(form.prob, f=f)
    return form


# ----------------------------------------------------------------------------------------------------------------
# Detection and iteration study
# ----------------------------------------------------------------------------------------------------------------
SIZES = {"transport": (10, 20, 40), "netflow_cap": (8, 16, 32), "mcf_joint": (4, 8, 12),
         "gub_packing": (20, 80, 320), "production": (12, 48, 192)}


def study(family, size, seed=0, cap=20000, permute=True, omega=1.0, cands=("std_ruiz", "reloc_rows"), **difficulty):
    """One instance through the pipeline: reference, plan_product, lp_diagnose + plan_block, and measure_ladder for
    std_ruiz and (when a product plan exists) reloc_rows with the bisection product projection.

    The encoding is permuted by default, as everywhere in the unified experiments, so no plan can lean on the
    generator's row order."""
    d0 = make_lp(family, size, seed, **difficulty)
    d = uni.permute_instance(d0, seed + 1000) if permute else d0
    A = sp.csc_matrix(d["A"])
    rec = {"family": family, "size": int(size), "seed": int(seed), "difficulty": difficulty, "name": d0["name"],
           "n_var": int(A.shape[1]), "rows": int(A.shape[0]), "nnz": int(A.nnz),
           "x_feas_row_violation": row_violation(d0, d0["x_feas"])}
    t0 = time.perf_counter()
    obj_ref, info = uni.reference(d)
    rec["ref_s"] = time.perf_counter() - t0
    if obj_ref is None:
        rec["reference_error"] = str(info)[:300]
        return rec
    rec["obj_ref"] = obj_ref
    rec["obj_x_feas"] = float(np.asarray(d0["q"]) @ d0["x_feas"])
    pplan, pmsg = uni.plan_product(d)
    rec["plan_product"] = {"status": pmsg} if pplan is None else {
        "status": pmsg, "K": int(pplan["K"]), "cover": float(pplan["cover"]), "rows_kept": int(pplan["keep"].size),
        "block_nnz": sorted(set(np.bincount(pplan["bid"]).tolist()))[:6], "fully_absorbed": pplan["fully_absorbed"]}
    diag = lp_diagnose(d)
    bplan, bmsg = uni.plan_block(d, diag)
    rec["diagnose"] = diag
    rec["plan_block"] = {"status": bmsg} if bplan is None else {"status": bmsg, "kind": bplan["kind"],
                                                                "n_x": bplan["n_x"]}
    for cand in cands:
        if cand.startswith("reloc") and pplan is None:
            continue
        form = lp_formulation(d, cand, plan=pplan, impl="bisect", omega=omega)
        t0 = time.perf_counter()
        r = uni.measure_ladder(form, d, obj_ref, cap=cap)
        rec[cand] = {"N_at": r["N_at"], "iters_run": r["iters_run"], "row_violation": r["row_violation"],
                     "obj_gap": r["obj_gap"], "normL": r["normL"], "t_iter": r["t_iter"],
                     "wall_s": time.perf_counter() - t0}
    return rec


if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser()
    ap.add_argument("--families", nargs="*", default=list(FAMILIES))
    ap.add_argument("--cap", type=int, default=20000)
    ap.add_argument("--omega", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rows = []
    for fam in args.families:
        for size in SIZES[fam]:
            rec = study(fam, size, seed=args.seed, cap=args.cap, omega=args.omega)
            rows.append(rec)
            print(json.dumps(rec, default=float), flush=True)
            if args.out:
                with open(args.out, "w") as f:
                    json.dump(rows, f, indent=1, default=float)
