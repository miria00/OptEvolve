"""Real benchmark instances in the solver's (P, q, r, A, l, u) form: Maros-Meszaros (138 convex QPs, the OSQP-format
.mat files of qpsolvers/maros_meszaros_qpbenchmark) and the convex continuous linearly constrained QPs of QPLIB
(types CCL and DCL; the set RLQP evaluates on). Infinite bounds are +-inf. Nothing here is family-specific.
"""
import os
import numpy as np
import scipy.io as sio
import scipy.sparse as sp

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "data")
BIG = 1e19


def _clean_bounds(l, u):
    l = np.asarray(l, dtype=float).ravel().copy()
    u = np.asarray(u, dtype=float).ravel().copy()
    l[l <= -BIG] = -np.inf
    u[u >= BIG] = np.inf
    return l, u


def load_mm(name, root=None):
    m = sio.loadmat(os.path.join(root or os.path.join(ROOT, "maros_meszaros"), name + ".mat"))
    P = sp.csc_matrix(m["P"]).astype(float)
    if sp.linalg.norm(P - P.T) > 1e-9 * max(sp.linalg.norm(P), 1.0):          # stored as one triangle
        P = sp.csc_matrix(P + P.T - sp.diags(P.diagonal()))
    l, u = _clean_bounds(m["l"], m["u"])
    return {"P": P, "q": np.asarray(m["q"], dtype=float).ravel(), "r": float(np.asarray(m["r"]).ravel()[0]),
            "A": sp.csc_matrix(m["A"]).astype(float), "l": l, "u": u, "name": "MM/" + name}


def mm_names(root=None):
    d = root or os.path.join(ROOT, "maros_meszaros")
    return sorted(f[:-4] for f in os.listdir(d) if f.endswith(".mat"))


def load_qplib(path):
    """Parse a .qplib file of type (L|D|C)(C)(L|B|N): objective 1/2 x'Q0x + b0'x + q0 over the stored lower triangle,
    1-based indices, linear constraints cl <= Ax <= cu, bounds xl <= x <= xu. Bounds become identity rows of A, the
    OSQP convention the rest of the pipeline uses. Maximization is negated into minimization."""
    lines = []
    for raw in open(path):
        s = raw.split("#", 1)[0].strip()
        if s:
            lines.append(s)
    it = iter(lines)
    nxt = lambda: next(it).split()
    name = nxt()[0]
    ptype = nxt()[0].upper()
    sense = nxt()[0].lower()
    n = int(nxt()[0])
    has_cons = ptype[2] not in ("N", "B")
    m = int(nxt()[0]) if has_cons else 0
    I, J, V = [], [], []
    if ptype[0] != "L":
        for _ in range(int(nxt()[0])):
            i, j, v = nxt()[:3]
            i, j, v = int(i) - 1, int(j) - 1, float(v)
            if i == j:
                I.append(i); J.append(j); V.append(v)
            else:
                # the objective is 1/2 x'Qx over the LOWER TRIANGLE as stored, so an off-diagonal entry v contributes
                # v/2 x_i x_j; the symmetric P carries v/2 in both (i,j) and (j,i). Mirroring the full v doubled every
                # coupling term: the objective at the published solution was off by 12-47% on all 5 CCL instances,
                # and matches to 1e-15 with this convention (DCL instances have no off-diagonals and matched before).
                I += [i, j]; J += [j, i]; V += [0.5 * v, 0.5 * v]
    P = sp.csc_matrix((V, (I, J)), shape=(n, n))
    b0 = np.full(n, float(nxt()[0]))
    for _ in range(int(nxt()[0])):
        i, v = nxt()[:2]
        b0[int(i) - 1] = float(v)
    r = float(nxt()[0])
    if ptype[2] in ("Q", "C"):
        raise ValueError("quadratic constraints not supported")
    A = sp.csc_matrix((m, n))
    if has_cons:
        I, J, V = [], [], []
        for _ in range(int(nxt()[0])):
            k, i, v = nxt()[:3]
            I.append(int(k) - 1); J.append(int(i) - 1); V.append(float(v))
        A = sp.csc_matrix((V, (I, J)), shape=(m, n))
    inf = float(nxt()[0])

    def vec(size):
        x = np.full(size, float(nxt()[0]))
        for _ in range(int(nxt()[0])):
            i, v = nxt()[:2]
            x[int(i) - 1] = float(v)
        return x

    cl, cu = (vec(m), vec(m)) if has_cons else (np.zeros(0), np.zeros(0))
    xl, xu = vec(n), vec(n)
    cl[cl <= -inf] = -np.inf; cu[cu >= inf] = np.inf
    xl[xl <= -inf] = -np.inf; xu[xu >= inf] = np.inf
    if sense.startswith("max"):
        P, b0, r = -P, -b0, -r
    bounded = np.flatnonzero(np.isfinite(xl) | np.isfinite(xu))
    B = sp.csc_matrix((np.ones(bounded.size), (np.arange(bounded.size), bounded)), shape=(bounded.size, n))
    return {"P": sp.csc_matrix(P), "q": b0, "r": r, "A": sp.csc_matrix(sp.vstack([A, B])),
            "l": np.concatenate([cl, xl[bounded]]), "u": np.concatenate([cu, xu[bounded]]),
            "name": f"QPLIB/{name}", "ptype": ptype}


def load_qplib_solution(path, n):
    """QPLIB .sol files use GAMS names: x1 is the objective variable, so x(k) is variable k-1 (1-based); only nonzeros
    are listed. Returns (x, objective)."""
    import re
    x, obj = np.zeros(n), None
    for raw in open(path):
        t = raw.split()
        if not t:
            continue
        if t[0] == "objvar":
            obj = float(t[1])
        elif re.match(r"^x\d+$", t[0]):
            x[int(t[0][1:]) - 2] = float(t[1])
    return x, obj
