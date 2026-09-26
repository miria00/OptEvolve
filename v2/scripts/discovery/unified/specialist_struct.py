"""Recognize an LP or SOCP as the encoding of a named combinatorial or statistical problem, and route it to a solver
for THAT problem. The structural sibling of specialist.py, which does the same job for the lasso QP encoding.

Why this exists. A field of general convex solvers can be complete in the sense that every instance is inside every
solver's problem class, and still be INCOMPLETE in the sense that matters: the fastest known method for an instance
is a method for its structure, not for its class. Three gaps we can name on this workload:

  * a min-cost network flow LP is solved by the network simplex in a running time that has nothing to do with the
    size of its constraint matrix. Measured earlier (2026-09-15): POT's network simplex solved a 1,000,000-variable
    transport problem in 4.8 s where HiGHS needed 118 s and Clarabel 256 s.
  * a group lasso SOCP is an n-dimensional statistical problem with a small active support, written as a cone
    program with one cone per group. A block-coordinate-descent solver touches only the active groups.
  * a minimum enclosing ball SOCP has one cone per POINT, so an interior-point method factorizes a system that grows
    with the sample. The core-set algorithms for MEB are linear in the sample and touch a handful of points.

Same two rules as specialist.py keep this honest:

  * the specialist's answer is lifted back to the ORIGINAL variable and judged by the same `problem_adapters.accept`
    check as every other candidate, at the same tolerance. A specialist gets no credit for reporting convergence in
    its own objective.
  * detection is conservative and reads structure only, never a name or a generator. Every instance the router sees
    has had its rows and columns randomly permuted (`uni.permute_instance`, `socp.permute_socp`), so a detector that
    keyed on block order would fire on the generator's output and never once on a routed case. A structure we cannot
    verify returns None and the router proceeds exactly as before: a false negative costs a routing opportunity and
    never a wrong answer.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time

import numpy as np
import scipy.sparse as sp

# Interpreters holding the specialist packages. Isolated from ours for the same reason the baselines are: the solver
# environment must never acquire a competitor's package.
NETFLOW_ENV = os.environ.get("OPTEVOLVE_NETFLOW_ENV", "/home/miria/baseline_envs/venv_netsimplex/bin/python")
POT_ENV = os.environ.get("OPTEVOLVE_POT_ENV", "/home/miria/baseline_envs/venv_pot/bin/python")
ENET_ENV = os.environ.get("OPTEVOLVE_SPECIALIST_ENV", "/home/miria/enetenv/bin/python")

HERE = os.path.dirname(os.path.abspath(__file__))
PROC = os.path.join(HERE, "specialist_struct_proc.py")

FLOW_SPECIALISTS = ("pot", "ortools", "networkx")
GLASSO_SPECIALISTS = ("skglm",)
MEB_SPECIALISTS = ("coreset",)


# ------------------------------------------------------------------------------------------------------------------
# (1) min-cost network flow
# ------------------------------------------------------------------------------------------------------------------
def detect_network_flow(d, atol=1e-9):
    """Return a min-cost-flow description if this LP IS a network flow encoding under ANY permutation, else None.

    The encoding the LP families produce (`lp_families._osqp`) is rows [G; B]: the structural rows G and one identity
    row per bounded variable. So a network flow instance has EXACTLY three nonzeros per column, one of them in a
    singleton row. That count is the cheap gate, and everything after it is one pass over the nonzeros.

    Two coefficient patterns are accepted in the structural rows, and they are the same problem written twice:
      * one +1 and one -1 per column: a node-arc incidence matrix, tail positive (atlas.bench.mcf.incidence_matrix).
      * two +1 per column: the transportation form, where the sink marginals are stated with positive coefficients
        (lp_families.transport). Two-coloring the row graph recovers the two sides, and negating one side turns it
        into the incidence form, so both reach the same solvers.
    Returns {tail, head, cost, cap, supply, n_nodes, n_arcs, cols, bipartite, ...} in ARC order = column order.
    """
    if sp.csc_matrix(d["P"]).nnz:
        return None                                        # a quadratic objective is not a network flow
    A = sp.csc_matrix(d["A"])
    A.sort_indices()
    n, m = A.shape[1], A.shape[0]
    if n == 0 or m <= n:
        return None
    if not np.all(np.diff(A.indptr) == 3):
        return None                                        # cheap gate: two structural entries plus one bound row
    l, u = np.asarray(d["l"], float), np.asarray(d["u"], float)
    Ar = A.tocsr()
    Ar.sort_indices()
    rowcnt = np.diff(Ar.indptr)
    bnd = np.flatnonzero(rowcnt == 1)
    if bnd.size != n:
        return None
    bcol = Ar.indices[Ar.indptr[bnd]]
    bval = Ar.data[Ar.indptr[bnd]]
    if not np.allclose(bval, 1.0, atol=atol) or np.unique(bcol).size != n:
        return None
    lo = np.empty(n)
    hi = np.empty(n)
    lo[bcol] = l[bnd]
    hi[bcol] = u[bnd]
    if not np.allclose(lo, 0.0, atol=atol) or np.any(hi < -atol):
        return None                                        # flows must be nonnegative

    keep = np.ones(m, bool)
    keep[bnd] = False
    stru = np.flatnonzero(keep)
    if stru.size + n != m:
        return None
    ls, us = l[stru], u[stru]
    if not (np.all(np.isfinite(ls)) and np.all(np.isfinite(us)) and np.all(np.abs(us - ls) <= atol)):
        return None                                        # conservation rows are equalities
    S = sp.csc_matrix(Ar[stru])
    S.sort_indices()
    if not np.all(np.diff(S.indptr) == 2):
        return None
    r0, r1 = S.indices[0::2], S.indices[1::2]
    v0, v1 = S.data[0::2], S.data[1::2]
    ns = stru.size
    supply = ls.copy()

    pos0, neg0 = np.isclose(v0, 1.0, atol=atol), np.isclose(v0, -1.0, atol=atol)
    pos1, neg1 = np.isclose(v1, 1.0, atol=atol), np.isclose(v1, -1.0, atol=atol)
    bipartite = False
    if np.all((pos0 & neg1) | (neg0 & pos1)):
        tail = np.where(pos0, r0, r1)
        head = np.where(pos0, r1, r0)
    elif np.all(pos0 & pos1):
        side = _two_color(r0, r1, ns)                      # transportation form: recover the two marginal sides
        if side is None:
            return None
        bipartite = True
        src = side[r0] == 0
        tail = np.where(src, r0, r1)
        head = np.where(src, r1, r0)
        supply = np.where(side == 0, supply, -supply)      # negate the sink rows: +1 becomes the incidence -1
    else:
        return None
    if np.any(tail == head):
        return None
    return {"form": "netflow", "tail": tail, "head": head, "cost": np.asarray(d["q"], float).copy(),
            "cap": hi, "supply": supply, "n_nodes": int(ns), "n_arcs": int(n), "bipartite": bool(bipartite),
            "rows": stru, "N": n}


def _two_color(r0, r1, n_nodes):
    """Two-color the graph whose edges are (r0[e], r1[e]); None if it is not bipartite.

    A breadth-first pass per connected component, using scipy's C implementation of both steps: the coloring is the
    parity of the BFS depth, and the answer is VERIFIED by checking every edge joins two colors, so a wrong labeling
    cannot slip through as a recognized structure.
    """
    from scipy.sparse.csgraph import breadth_first_order, connected_components
    E = r0.size
    G = sp.csr_matrix((np.ones(2 * E), (np.concatenate([r0, r1]), np.concatenate([r1, r0]))),
                      shape=(n_nodes, n_nodes))
    ncomp, labels = connected_components(G, directed=False)
    color = np.full(n_nodes, -1, np.int8)
    for comp in range(ncomp):
        start = int(np.flatnonzero(labels == comp)[0])
        order, pred = breadth_first_order(G, start, directed=False, return_predecessors=True)
        color[start] = 0
        for v in order[1:]:                                # BFS order, so a node's predecessor is already colored
            color[v] = 1 - color[pred[v]]
    if np.any(color < 0) or np.any(color[r0] == color[r1]):
        return None                                        # an odd cycle: not a bipartite marginal structure
    return color


def lift_flow(x_arc, det):
    """Arc flows are already in column order, so the lift is the identity. Kept explicit for symmetry."""
    z = np.zeros(det["N"], float)
    z[:] = np.asarray(x_arc, float).ravel()
    return z


# ------------------------------------------------------------------------------------------------------------------
# (2) group lasso
# ------------------------------------------------------------------------------------------------------------------
def detect_group_lasso(d, atol=1e-9):
    """Return {A, b, lam, groups, ...} if this SOCP IS a group lasso encoding under ANY permutation, else None.

    The lifted encoding (socp.group_lasso) is variables [x (n), t (G), y (m)] with P = diag(0, 0, I), q = (0, lam 1,
    0), linear rows A x - y = b, and one cone ||x_g|| <= t_g per group. Variables are classified by (diag(P), q)
    exactly as the lasso detector does, and the groups are read off the cones, so nothing depends on block order.
    """
    cones = d.get("cones") or []
    if not cones:
        return None
    P = sp.csr_matrix(d["P"])
    q = np.asarray(d["q"], float)
    N = P.shape[0]
    dg = np.asarray(P.diagonal(), float)
    if P.nnz != int((dg != 0).sum()):
        return None                                        # P must be diagonal
    S = np.flatnonzero(dg != 0)
    m = S.size
    if m == 0 or not np.allclose(dg[S], 1.0, atol=atol) or not np.allclose(q[S], 0.0, atol=atol):
        return None
    T = np.flatnonzero(q > 0)
    G = len(cones)
    if T.size != G:
        return None
    lam = float(q[T[0]])
    if lam <= 0 or not np.allclose(q[T], lam, atol=1e-9 * max(1.0, lam)):
        return None                                        # one penalty weight shared by every group
    n = N - m - G
    if n <= 0:
        return None
    cls = np.zeros(N, np.int8)                             # 0 = x, 1 = y (residual), 2 = t (penalty)
    cls[S] = 1
    cls[T] = 2
    X = np.flatnonzero(cls == 0)
    if X.size != n:
        return None

    groups = []
    seen = np.zeros(N, bool)
    for c in cones:
        if abs(float(c["h"])) > atol or not np.allclose(np.asarray(c["g"], float), 0.0, atol=atol):
            return None
        f = sp.csr_matrix(c["f"]).reshape(1, N)
        if f.nnz != 1 or not np.isclose(f.data[0], 1.0, atol=atol) or cls[f.indices[0]] != 2:
            return None                                    # the cone head must be one t variable
        F = sp.csr_matrix(c["F"])
        if F.shape[1] != N or F.nnz != F.shape[0]:
            return None
        if not np.allclose(F.data, 1.0, atol=atol):
            return None                                    # F must be a plain selector of x coordinates
        cols = np.sort(F.indices)
        if np.any(cls[cols] != 0) or np.any(seen[cols]):
            return None
        seen[cols] = True
        groups.append(cols)
    if int(seen.sum()) != n:
        return None                                        # the groups must partition x

    A = sp.csr_matrix(d["A"])
    l, u = np.asarray(d["l"], float), np.asarray(d["u"], float)
    if A.shape[0] != m or not np.allclose(u - l, 0.0, atol=atol):
        return None
    Ay = sp.csc_matrix(A[:, S])
    if Ay.nnz != m or not np.allclose(Ay.data, -1.0, atol=atol):
        return None                                        # exactly one -1 per residual variable
    if sp.csc_matrix(A[:, T]).nnz:
        return None
    rows_of_y = np.zeros(m, np.int64)                      # residual variable S[j] appears in row rows_of_y[j]
    rows_of_y[np.arange(m)] = Ay.indices
    order = np.argsort(rows_of_y)                          # put the rows in row order, carrying y with them
    Asub = sp.csc_matrix(A[:, X][rows_of_y[order]])
    b = u[rows_of_y[order]].copy()
    cols = np.asarray(X)
    pos = np.full(N, -1, np.int64)
    pos[cols] = np.arange(n)
    return {"form": "group_lasso", "A": Asub, "b": b, "lam": lam, "n": n, "m": m, "G": G,
            "cols": cols, "group_index": np.concatenate([pos[g] for g in groups]),
            "group_sizes": np.array([g.size for g in groups], np.int64),
            "group_cols": groups, "t_of_group": np.array([sp.csr_matrix(c["f"]).indices[0] for c in cones], np.int64),
            "y_of_row": S[order], "N": N}


def lift_group_lasso(x, det):
    """Lift x to the SOCP variable [x, t = ||x_g||, y = A x - b], exact on every row and every cone."""
    x = np.asarray(x, float).ravel()
    z = np.zeros(det["N"], float)
    z[det["cols"]] = x
    z[det["y_of_row"]] = det["A"] @ x - det["b"]
    for g, col in zip(det["group_cols"], det["t_of_group"]):
        z[col] = float(np.linalg.norm(z[g]))
    return z


# ------------------------------------------------------------------------------------------------------------------
# (3) minimum enclosing ball
# ------------------------------------------------------------------------------------------------------------------
def detect_meb(d, atol=1e-9):
    """Return {points, r_index, c_cols, ...} if this SOCP IS a minimum enclosing ball, else None.

    socp.min_enclosing_ball is variables [r, c (dim)], no linear rows, objective r, and one cone
    ||c - p_i|| <= r per point, written {F = selector of the c columns, g = -p_i, f = e_r, h = 0}. The permutation
    shuffles the columns, the cone order AND the rows WITHIN each cone independently, so a point is read back by
    pairing each cone row with the column its selector picks, never by row position.
    """
    cones = d.get("cones") or []
    if not cones or sp.csc_matrix(d["P"]).nnz:
        return None
    if sp.csc_matrix(d["A"]).shape[0] != 0:
        return None                                        # MEB has no linear rows
    q = np.asarray(d["q"], float)
    N = q.size
    nz = np.flatnonzero(q != 0.0)
    if nz.size != 1 or not np.isclose(q[nz[0]], 1.0, atol=atol):
        return None                                        # the objective is exactly the radius
    ir = int(nz[0])
    dim = None
    c_cols = None
    pts = None
    for k, c in enumerate(cones):
        if abs(float(c["h"])) > atol:
            return None
        f = sp.csr_matrix(c["f"]).reshape(1, N)
        if f.nnz != 1 or not np.isclose(f.data[0], 1.0, atol=atol) or int(f.indices[0]) != ir:
            return None                                    # every cone head is the same radius variable
        F = sp.csr_matrix(c["F"])
        if F.shape[1] != N or F.nnz != F.shape[0] or not np.allclose(F.data, 1.0, atol=atol):
            return None
        if np.any(np.diff(sp.csc_matrix(F).indptr) > 1):
            return None                                    # a plain selector: one 1 per row and per column
        # Sorting by COLUMN is what makes this permutation invariant: the cone's rows were shuffled independently of
        # every other cone's, so a point is read back through the column each row selects, never by row position.
        Fcoo = F.tocoo()
        order = np.argsort(Fcoo.col)
        cols = np.asarray(Fcoo.col)[order]
        rows_for_cols = np.asarray(Fcoo.row)[order]
        if dim is None:
            dim = F.shape[0]
            if dim == 0 or ir in set(cols.tolist()):
                return None
            c_cols = cols
            pts = np.empty((len(cones), dim), float)
        elif F.shape[0] != dim or not np.array_equal(cols, c_cols):
            return None                                    # every cone must use the same c block
        g = np.asarray(c["g"], float).ravel()
        if g.size != dim:
            return None
        pts[k] = -g[rows_for_cols]                         # g = -p, reindexed from cone-row order to column order
    return {"form": "meb", "points": pts, "r_index": ir, "c_cols": np.asarray(c_cols), "dim": int(dim),
            "n_points": len(cones), "N": N}


def lift_meb(center, radius, det):
    z = np.zeros(det["N"], float)
    z[det["r_index"]] = float(radius)
    z[det["c_cols"]] = np.asarray(center, float).ravel()
    return z


# ------------------------------------------------------------------------------------------------------------------
# dispatch
# ------------------------------------------------------------------------------------------------------------------
_ENVS = {"pot": POT_ENV, "ortools": NETFLOW_ENV, "networkx": NETFLOW_ENV, "skglm": ENET_ENV,
         "coreset": NETFLOW_ENV}


def run_struct_specialist(d, engine, obj_ref, tol, timeout_s, det=None):
    """Run `engine` ("flow:pot", "glasso:skglm", "meb:coreset", ...) out of process, lift, judge on the original.

    Returns the record shape `problem_adapters.run_engine` returns, so the router scores a specialist against a
    generic solver with no special case: `wall_s` is what a first visit pays, `t_solver_s` the routed cost.
    """
    import problem_adapters as PA
    t0 = time.perf_counter()
    family, _, name = engine.partition(":")
    detect = {"flow": detect_network_flow, "glasso": detect_group_lasso, "meb": detect_meb}.get(family)
    if detect is None:
        return {"accepted": False, "error": f"unknown specialist family {family!r}",
                "wall_s": time.perf_counter() - t0}
    det = det if det is not None else detect(d)
    if det is None:
        return {"accepted": False, "error": "structure not recognized", "wall_s": time.perf_counter() - t0}
    tmp = tempfile.mkdtemp(prefix="sspec_")
    try:
        if family == "flow":
            if name == "pot" and (det["bipartite"] is False or np.any(np.isfinite(det["cap"]))):
                return {"accepted": False, "unsupported": True,
                        "error": "POT's emd needs an uncapacitated transportation problem",
                        "wall_s": time.perf_counter() - t0}
            np.savez(os.path.join(tmp, "in.npz"), tail=det["tail"], head=det["head"], cost=det["cost"],
                     cap=det["cap"], supply=det["supply"], n_nodes=det["n_nodes"])
        elif family == "glasso":
            sp.save_npz(os.path.join(tmp, "A.npz"), sp.csc_matrix(det["A"]))
            np.savez(os.path.join(tmp, "in.npz"), b=det["b"], lam=det["lam"],
                     group_index=det["group_index"], group_sizes=det["group_sizes"])
        else:
            np.savez(os.path.join(tmp, "in.npz"), points=det["points"])
        cmd = [_ENVS[name], PROC, "--dir", tmp, "--family", family, "--solver", name, "--tol", repr(float(tol))]
        env = dict(os.environ)
        env.pop("LD_LIBRARY_PATH", None)                   # the local CUDA path makes numpy pick a wrong libgomp
        env["JAX_PLATFORMS"] = "cpu"
        pr = subprocess.run(cmd, capture_output=True, text=True, timeout=max(1.0, timeout_s), env=env)
        if pr.returncode != 0:
            return {"accepted": False, "error": (pr.stderr or pr.stdout)[-300:],
                    "wall_s": time.perf_counter() - t0}
        rb = json.loads(pr.stdout.strip().splitlines()[-1])
        if rb.get("unsupported"):
            return {"accepted": False, "unsupported": True, "error": rb.get("error"),
                    "wall_s": time.perf_counter() - t0}
        z = np.load(os.path.join(tmp, "out.npz"))
        if family == "flow":
            x = lift_flow(z["x"], det)
        elif family == "glasso":
            x = lift_group_lasso(z["x"], det)
        else:
            x = lift_meb(z["center"], float(z["radius"]), det)
        ok, rv, gap = PA.accept(x, d, obj_ref, tol)
        return {"accepted": bool(ok), "t_solver_s": rb["t_s"], "eps": rb.get("eps", tol), "rv": rv, "gap": gap,
                "info": rb.get("info"), "wall_s": time.perf_counter() - t0}
    except subprocess.TimeoutExpired:
        return {"accepted": False, "error": "timeout", "wall_s": time.perf_counter() - t0}
    except Exception as e:
        return {"accepted": False, "error": repr(e)[:300], "wall_s": time.perf_counter() - t0}
    finally:
        for f in ("in.npz", "A.npz", "out.npz"):
            try:
                os.remove(os.path.join(tmp, f))
            except OSError:
                pass
        try:
            os.rmdir(tmp)
        except OSError:
            pass


def structural_engines(d):
    """Engine names unlocked by recognizing this instance's structure. Cheap enough to run inside a solve budget."""
    import problem_adapters as PA
    k = PA.kind(d)
    if k == "lp":
        det = detect_network_flow(d)
        if det is None:
            return ()
        if det["bipartite"] and not np.any(np.isfinite(det["cap"])):
            return ("flow:pot", "flow:ortools", "flow:networkx")
        return ("flow:ortools", "flow:networkx")
    if k == "socp":
        if detect_meb(d) is not None:
            return ("meb:coreset",)
        if detect_group_lasso(d) is not None:
            return ("glasso:skglm",)
    return ()


if __name__ == "__main__":
    # Detector self-check. The same three things specialist.py checks have to hold before any of this is allowed to
    # change a routing decision:
    #   1. each detector fires on ITS family and on no other (a false positive would hand a problem to a solver that
    #      cannot represent it);
    #   2. it fires on the PERMUTED encoding, which is the only form the router ever sees;
    #   3. the lift reproduces the original problem exactly, so a specialist's answer is judged on the same numbers
    #      every other candidate is judged on.
    import sys as _sys
    _sys.path.insert(0, "/home/miria/OptEvolve/v2")
    _sys.path.insert(0, HERE)
    import socp as _socp
    from lp_families import make_lp
    from uni import permute_instance

    print("LP families (detect_network_flow): expect transport and netflow_cap only")
    for fam, size in (("transport", 30), ("netflow_cap", 12), ("mcf_joint", 6), ("gub_packing", 200),
                      ("production", 200)):
        d0 = make_lp(fam, size, seed=3)
        d0 = {k: d0[k] for k in ("P", "q", "r", "A", "l", "u")}
        raw, perm = detect_network_flow(d0), detect_network_flow(permute_instance(d0, 1003))
        line = f"  {fam:14s} canonical={raw is not None} permuted={perm is not None}"
        if perm is not None:
            # conservation and the objective must be reproduced by the recovered graph, not just by the solver
            import scipy.sparse as _sp
            dp = permute_instance(d0, 1003)
            A = _sp.csc_matrix(dp["A"])
            x = np.zeros(A.shape[1])
            x[:] = 0.0
            inc = _sp.csr_matrix((np.concatenate([np.ones(perm["n_arcs"]), -np.ones(perm["n_arcs"])]),
                                  (np.concatenate([perm["tail"], perm["head"]]),
                                   np.concatenate([np.arange(perm["n_arcs"])] * 2))),
                                 shape=(perm["n_nodes"], perm["n_arcs"]))
            line += (f" nodes={perm['n_nodes']} arcs={perm['n_arcs']} bipartite={perm['bipartite']}"
                     f" capacitated={bool(np.any(np.isfinite(perm['cap'])))}"
                     f" incidence_cols_sum_to_zero={bool(np.allclose(np.asarray(inc.sum(axis=0)).ravel(), 0.0))}")
        print(line)

    print("SOCP families (detect_meb / detect_group_lasso): expect exactly one hit each")
    for fam, size in (("min_enclosing_ball", 400), ("group_lasso", 400), ("facility_location", 400),
                      ("robust_lp", 200), ("portfolio_risk", 400)):
        d0 = _socp.make_socp(fam, size, seed=3)
        dp, _ = _socp.permute_socp(d0, 1003)
        m_raw, m_perm = detect_meb(d0) is not None, detect_meb(dp) is not None
        g_raw, g_perm = detect_group_lasso(d0) is not None, detect_group_lasso(dp) is not None
        extra = ""
        if g_perm:
            det = detect_group_lasso(dp)
            rng = np.random.default_rng(3)
            xs = rng.standard_normal(det["n"]) * (rng.random(det["n"]) < 0.1)
            z = lift_group_lasso(xs, det)
            lin, cone = _socp.socp_violation(z, dp)
            groups = det["group_index"].reshape(det["G"], -1)
            nat = 0.5 * float(np.sum((det["A"] @ xs - det["b"]) ** 2)) \
                + det["lam"] * float(np.sum(np.linalg.norm(xs[groups], axis=1)))
            obj = _socp.socp_objective(z, dp)
            extra = (f" lift_lin_viol={lin:.2e} lift_cone_viol={cone:.2e}"
                     f" objective_match={abs(obj - nat) <= 1e-8 * (1 + abs(nat))}")
        if m_perm:
            det = detect_meb(dp)
            rng = np.random.default_rng(3)
            c = rng.standard_normal(det["dim"]) * 0.1
            r = float(np.max(np.linalg.norm(det["points"] - c, axis=1)))
            z = lift_meb(c, r, det)
            lin, cone = _socp.socp_violation(z, dp)
            extra = (f" dim={det['dim']} points={det['n_points']} lift_lin_viol={lin:.2e}"
                     f" lift_cone_viol={cone:.2e} objective_match="
                     f"{abs(_socp.socp_objective(z, dp) - r) <= 1e-9 * (1 + abs(r))}")
        print(f"  {fam:20s} meb={m_raw}/{m_perm} glasso={g_raw}/{g_perm}{extra}")
