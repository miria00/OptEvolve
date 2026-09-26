"""Second-order cone programs in the unified formulation-selection machinery (2026-09-15).

Instance form. The QP dict of uni.py (P, q, r, A, l, u; A holds the LINEAR rows, bounds as identity rows, +-inf
allowed) plus "cones": a list of {F (sparse, m_k x n), g (m_k,), f (n,) dense or sparse 1 x n, h (scalar)}, each
meaning
    ||F x + g||_2 <= f'x + h.
The generators emit sparse f: with one cone per data point a dense f is O(K n) memory.
The objective is 1/2 x'Px + q'x + r and r IS counted (unlike uni.accept), so a reference and an accepted iterate are
compared on the true objective value.

Formulations (build_socp_formulation), all Condat-Vu composites min f(x) + g(x) + h(Lx):
  std_raw / std_ruiz   every constraint in L = [A; f_1'; F_1; ...; f_K'; F_K]; h is Box over the linear rows times
                       a product of SHIFTED second-order cones over the conic rows. The affine offsets enter the
                       projection, not the operator: Lx + c in K has projection P(y) = P_K(y + c) - c, exact.
                       Ruiz ties the row scale inside each cone block, because a positive multiple of a cone is
                       the same cone while a per-row scaling of it is an ellipsoidal cone with no closed form.
  reloc_cone(_raw)     cones of the form ||a s_S x_S + g|| <= a s_t x_t + h on original variables (one nonzero per
                       row of F, all of modulus a, in distinct columns S; f a single entry of modulus a outside S,
                       or f = 0, a ball) leave L and become an exact projection in g. The map
                       x_C -> (a s_t x_t + h, a s_S x_S + g) is a similarity, and Euclidean projection commutes with
                       similarities, so the projection is the cone's closed form conjugated by it. Ruiz here also
                       ties the COLUMN scale inside each absorbed cone, which keeps that map a similarity.

h atoms have only prox (a projection). Condat-Vu takes prox_{s h*} through the Moreau identity in
atlas.schemes.base_schemes._prox_conj, prox_{s h*}(v) = v - s P(v/s), which is exact for any closed convex set, so
no conjugate needs to be written by hand and the offsets are inside P where they belong.

Linear objectives. Four of the five families have P = 0, and uni's step rule t = 1/L_f divides by L_f. A constant
gradient is L-Lipschitz for EVERY L >= 0, so f declares L_f = ||L|| / w, which is a valid constant, keeps the Level-1
gate satisfied, and turns uni's rule into PDHG steps t = w/||L||, s = 0.49/(w ||L||). The balance w is NOT neutral:
on portfolio_risk n=200 (after Ruiz) w = 1 needs 50,100 iterations to 1e-6 and w = 0.1 needs 4,650. The default
w = ||rhs|| / ||c|| on the data the iteration sees (rhs = finite row bounds and cone offsets left in L) is PDLP's
initial primal weight (Applegate et al. 2021) written for the primal step; it is a data rule, not a tuned constant,
and a float overrides it. On a quadratic objective w is unused and uni's rule applies unchanged.

Acceptance (accept_socp) extends uni.accept: per-row relative violation on linear rows, per-cone relative violation
max(0, ||Fx+g|| - (f'x+h)) / (1 + |f'x+h|), and relative objective error, all <= tol against a Clarabel 1e-9
reference (1e-8 when 1e-9 ends AlmostSolved) solved out of process (atlas.bench.conic_baseline_proc).
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import scipy.sparse as sp

import uni                                                     # puts v2 on sys.path and enables x64
import jax
import jax.numpy as jnp

from atlas.bench.conic_baseline_proc import frow, run_conic_baseline_proc, stack_cones
from atlas.bench.conic_baseline_proc import to_clarabel  # noqa: F401  (re-exported: part of this module's interface)
from atlas.bench.lp_cell import LinearCost
from atlas.bench.mps_cell import SparseLinOp
from atlas.bench.qp_gen import _sprandn
from atlas.ir.spec import CompositeProblem
from atlas.operators.base import PropSet
from atlas.operators.prox import Box, QuadraticForm, Zero
from atlas.schemes.base_schemes import CondatVuParams
from atlas.schemes.compile import build_scheme
from atlas.schemes.rescale import QP_MAX_SCALING, QP_MIN_SCALING
from uni import POWER_ITERS, Formulation


# ----------------------------------------------------------------------------------------------------------------
# Instance form
# ----------------------------------------------------------------------------------------------------------------
def validate_socp(d):
    n = sp.csc_matrix(d["P"]).shape[0]
    A = sp.csc_matrix(d["A"])
    if A.shape[1] != n or np.asarray(d["q"]).shape != (n,):
        raise ValueError("P, q, A disagree on the number of variables")
    if np.asarray(d["l"]).shape != (A.shape[0],) or np.asarray(d["u"]).shape != (A.shape[0],):
        raise ValueError("l, u must have one entry per linear row")
    for k, c in enumerate(d.get("cones", [])):
        F, f = c["F"], c["f"]
        fshape_ok = f.shape in ((1, n), (n, 1)) if sp.issparse(f) else np.shape(f) == (n,)
        if F.shape[1] != n or np.shape(c["g"]) != (F.shape[0],) or not fshape_ok or np.ndim(c["h"]) != 0:
            raise ValueError(f"cone {k}: F (m x n), g (m,), f (n,) or sparse 1 x n, h scalar required")
    return n


def socp_objective(x, d):
    return float(0.5 * x @ (sp.csc_matrix(d["P"]) @ x) + np.asarray(d["q"]) @ x + float(d.get("r", 0.0)))


def reference(d, eps=1e-9, fallback_eps=(1e-8,), timeout_s=1800):
    """Clarabel in its isolated interpreter. Returns (objective, x), or (None, error string).

    Only status Solved is trusted. At 1e-9 Clarabel sometimes stops at AlmostSolved, which certifies only its reduced
    tolerances (about 1e-4 to 5e-5) and so cannot referee a 1e-6 verdict; seen on group_lasso with fewer samples than
    features. Those instances are re-solved at the fallback tolerances, which are still 100x tighter than the
    tightest rung of the ladder.
    """
    err = None
    for e in (eps,) + tuple(fallback_eps):
        rb = run_conic_baseline_proc(d, "clarabel", e, timeout_s=timeout_s, scratch=os.environ.get("OPTEVOLVE_SCRATCH"))
        if "error" in rb:
            return None, rb["error"]
        if rb["status"] == "Solved":
            x = np.asarray(rb["x"], dtype=float)
            return socp_objective(x, d), x
        err = f"clarabel status {rb['status']} at eps {e:g}"
    return None, err


def permute_socp(d, seed):
    """Random column, linear-row, cone-order and within-cone tail permutation. Same problem, different encoding.

    Returns (d', pc) with x_original[pc] = x_permuted. Head and tail stay distinguishable (f vs F), so this is every
    relabeling that leaves each cone the same set.
    """
    rng = np.random.default_rng(seed)
    n, m = d["P"].shape[0], d["A"].shape[0]
    pc, pr = rng.permutation(n), rng.permutation(m)
    out = {"P": sp.csc_matrix(sp.csc_matrix(d["P"])[pc][:, pc]), "q": np.asarray(d["q"])[pc], "r": d.get("r", 0.0),
           "A": sp.csc_matrix(sp.csc_matrix(d["A"])[pr][:, pc]), "l": np.asarray(d["l"])[pr],
           "u": np.asarray(d["u"])[pr], "cones": []}
    cones = d.get("cones", [])
    for k in rng.permutation(len(cones)):
        c = cones[k]
        pt = rng.permutation(sp.csc_matrix(c["F"]).shape[0])
        out["cones"].append({"F": sp.csc_matrix(sp.csr_matrix(c["F"])[pt][:, pc]), "g": np.asarray(c["g"])[pt],
                             "f": frow(c["f"], n)[:, pc], "h": float(c["h"])})
    if "name" in d:
        out["name"] = d["name"] + "/perm"
    return out, pc


def conic_rows(cones, n):
    """Stack cones as rows [f_k'; F_k] with offsets [h_k; g_k]. Returns (M csr, offset, bid, head mask, dims)."""
    M, off, dims = stack_cones(cones, n)
    dims_a = np.asarray(dims, dtype=np.int64)
    bid = np.repeat(np.arange(dims_a.size, dtype=np.int64), dims_a)
    head = np.zeros(bid.size, dtype=bool)
    head[np.concatenate([[0], np.cumsum(dims_a)[:-1]]).astype(np.int64) if dims_a.size else []] = True
    return M, off, bid, head, dims


# ----------------------------------------------------------------------------------------------------------------
# Exact projections. Closed forms (Bauschke and Combettes, Convex Analysis and Monotone Operator Theory, Ex. 29.11 for
# the Lorentz cone; Parikh and Boyd, Proximal Algorithms, 6.3.2): for (t, z) with ||z|| <= t the point itself, for
# ||z|| <= -t the origin, otherwise ((t + ||z||)/2) (1, z/||z||). The third branch only fires when ||z|| > |t| >= 0,
# so its division is safe; the guard is there so the untaken branch never produces a NaN under jnp.where.
# ----------------------------------------------------------------------------------------------------------------
def project_soc(t, z):
    nz = jnp.linalg.norm(z)
    inside, polar = nz <= t, nz <= -t
    alpha = 0.5 * (t + nz)
    t_new = jnp.where(inside, t, jnp.where(polar, 0.0, alpha))
    z_new = jnp.where(inside, z, jnp.where(polar, 0.0, (alpha / jnp.where(nz > 0, nz, 1.0)) * z))
    return t_new, z_new


def project_soc_product(w, bid, head, K):
    """Projection onto Q^{d_1} x ... x Q^{d_K} with cones of DIFFERENT sizes, without a loop over cones.

    bid[i] is the cone of entry i and head[i] marks its t (exactly one per cone, at any position). Per-cone norms are
    segment sums, so the cost is O(total size) and the trace does not depend on the cone sizes. A cone of size 1 has
    an empty tail and projects onto t >= 0.
    """
    tail_sq = jax.ops.segment_sum(jnp.where(head, 0.0, w * w), bid, num_segments=K)
    t = jax.ops.segment_sum(jnp.where(head, w, 0.0), bid, num_segments=K)
    nz = jnp.sqrt(tail_sq)
    inside, polar = nz <= t, nz <= -t
    alpha = 0.5 * (t + nz)
    t_new = jnp.where(inside, t, jnp.where(polar, 0.0, alpha))
    scale = jnp.where(inside, 1.0, jnp.where(polar, 0.0, alpha / jnp.where(nz > 0, nz, 1.0)))
    return jnp.where(head, t_new[bid], scale[bid] * w)


def project_l2_ball(z, center, radius):
    dz = z - center
    nd = jnp.linalg.norm(dz)
    return jnp.where(nd <= radius, z, center + (radius / jnp.where(nd > 0, nd, 1.0)) * dz)


def project_ball_product(w, bid, center, radius, K):
    """Projection onto a product of l2 balls {||w_k - center_k|| <= radius_k} of different sizes (segment ops)."""
    dw = w - center
    nd = jnp.sqrt(jax.ops.segment_sum(dw * dw, bid, num_segments=K))
    inside = nd <= radius
    scale = radius / jnp.where(nd > 0, nd, 1.0)
    return jnp.where(inside[bid], w, center + scale[bid] * dw)


@dataclass(frozen=True)
class SOCProj:
    """Indicator of {(t, z) : ||z||_2 <= t} on the whole vector, t = v[0]."""

    props: PropSet = field(default=PropSet(averagedness=0.5, prox_friendly=True))

    def prox(self, v, step=None):
        t, z = project_soc(v[0], v[1:])
        return jnp.concatenate([t[None], z])


@dataclass(frozen=True)
class SOCProductProj:
    """Indicator of {v : sign * v + offset in Q^{d_1} x ... x Q^{d_K}}, sign in {-1, +1} entrywise.

    As h over stacked conic rows sign = 1 and offset = [h_k; g_k]. As g on relocated variables sign and offset carry
    the similarity normalized by its modulus, so the projection is P(v) = sign * (P_K(sign * v + offset) - offset);
    multiplying by +-1 is exact, which is why the modulus is divided into the offset instead of kept as a scale.
    """

    bid: object
    head: object
    sign: object
    offset: object
    K: int
    props: PropSet = field(default=PropSet(averagedness=0.5, prox_friendly=True))

    def prox(self, v, step=None):
        y = project_soc_product(self.sign * v + self.offset, self.bid, self.head, self.K)
        return self.sign * (y - self.offset)


@dataclass(frozen=True)
class L2BallProj:
    """Indicator of {z : ||z - center||_2 <= radius}."""

    center: object
    radius: object
    props: PropSet = field(default=PropSet(averagedness=0.5, prox_friendly=True))

    def prox(self, v, step=None):
        return project_l2_ball(v, self.center, self.radius)


@dataclass(frozen=True)
class BallProductProj:
    """Indicator of a product of l2 balls over segments bid (centers per entry, radii per ball)."""

    bid: object
    center: object
    radius: object
    K: int
    props: PropSet = field(default=PropSet(averagedness=0.5, prox_friendly=True))

    def prox(self, v, step=None):
        return project_ball_product(v, self.bid, self.center, self.radius, self.K)


@dataclass(frozen=True)
class SliceProduct:
    """Separable sum over consecutive slices of the argument: its prox is the concatenation of the parts' proxes.

    This is how Box over linear rows and cones over conic rows form ONE h, and how relocated cones share g with the
    free variables (Zero). The loop is over parts (at most three), never over cones.
    """

    parts: tuple
    sizes: tuple
    props: PropSet = field(default=PropSet(averagedness=0.5, prox_friendly=True))

    def prox(self, v, step=None):
        out, o = [], 0
        for p, s in zip(self.parts, self.sizes):
            out.append(p.prox(v[o:o + s], step))
            o += s
        return jnp.concatenate(out)


jax.tree_util.register_dataclass(SOCProj, data_fields=[], meta_fields=["props"])
jax.tree_util.register_dataclass(SOCProductProj, data_fields=["bid", "head", "sign", "offset"],
                                 meta_fields=["K", "props"])
jax.tree_util.register_dataclass(L2BallProj, data_fields=["center", "radius"], meta_fields=["props"])
jax.tree_util.register_dataclass(BallProductProj, data_fields=["bid", "center", "radius"], meta_fields=["K", "props"])
jax.tree_util.register_dataclass(SliceProduct, data_fields=["parts"], meta_fields=["sizes", "props"])


def _slices(parts):
    """SliceProduct over the nonempty (atom, size) pairs; a single part is returned bare."""
    parts = [(a, s) for a, s in parts if s > 0]
    if len(parts) == 1:
        return parts[0][0]
    return SliceProduct(parts=tuple(a for a, _ in parts), sizes=tuple(int(s) for _, s in parts))


# ----------------------------------------------------------------------------------------------------------------
# Cone-aware Ruiz
# ----------------------------------------------------------------------------------------------------------------
def _group_max(vals, group):
    g = np.zeros(int(group.max()) + 1 if group.size else 0)
    np.maximum.at(g, group, vals)
    return g[group]


def cone_ruiz(P, M, row_group, col_group, iters=10):
    """kkt_ruiz_equilibrate (atlas.schemes.rescale) with scales TIED inside groups; returns (d1 rows, d2 columns).

    The group maximum is taken on the raw norms, before the small-norm replacement: a cone whose head row is zero
    (f = 0) would otherwise pull its whole block to the replacement value 1. With singleton groups every step is
    the original function's, so a cone-free instance gets exactly the uni std_ruiz scaling.
    """
    Ps, Ms = sp.csc_matrix(P, dtype=np.float64), sp.csc_matrix(M, dtype=np.float64)
    d1, d2 = np.ones(Ms.shape[0]), np.ones(Ms.shape[1])
    for _ in range(int(iters)):
        col = np.maximum(abs(Ps).max(axis=0).toarray().ravel(),
                         abs(Ms).max(axis=0).toarray().ravel() if Ms.shape[0] else 0.0)
        row = abs(Ms).max(axis=1).toarray().ravel() if Ms.shape[0] else np.zeros(0)
        col, row = _group_max(col, col_group), _group_max(row, row_group)
        col = np.where(col < QP_MIN_SCALING, 1.0, col)
        row = np.where(row < QP_MIN_SCALING, 1.0, row)
        dk = 1.0 / np.sqrt(np.clip(col, QP_MIN_SCALING, QP_MAX_SCALING))
        ek = 1.0 / np.sqrt(np.clip(row, QP_MIN_SCALING, QP_MAX_SCALING))
        Ps = (sp.diags(dk) @ Ps @ sp.diags(dk)).tocsc()
        Ms = (sp.diags(ek) @ Ms @ sp.diags(dk)).tocsc()
        d1 *= ek
        d2 *= dk
    return d1, d2


# ----------------------------------------------------------------------------------------------------------------
# Relocation plan: which cones are an exact projection on original variables
# ----------------------------------------------------------------------------------------------------------------
def plan_cone(d):
    """Greedy disjoint family of relocatable cones, read off the data as column SETS (permutation-robust).

    Relocatable: every row of F has exactly one nonzero, in distinct columns S, all of the same modulus a (exact
    equality, so the projection stays exact), and either f = 0 with h >= 0 (a ball) or f has one nonzero, at a column
    outside S, of modulus a. Larger cones are taken first. None means nothing is relocatable.
    """
    n = validate_socp(d)
    cands = []
    for k, c in enumerate(d.get("cones", [])):
        F = sp.csr_matrix(c["F"]).astype(np.float64)
        F.eliminate_zeros()
        mk = F.shape[0]
        if mk == 0 or not np.all(np.diff(F.indptr) == 1):
            continue
        cols, vals = F.indices.copy(), F.data.copy()
        if np.unique(cols).size != mk:
            continue
        a = abs(vals[0])
        if not np.all(np.abs(vals) == a):
            continue
        fr = frow(c["f"], n).copy()                              # frow may return the caller's own matrix
        fr.eliminate_zeros()
        fn, fv = fr.indices, fr.data
        g = np.asarray(c["g"], dtype=np.float64)
        h = float(c["h"])
        if fn.size == 0:
            if h < 0:
                continue                                         # empty set: leave it in L for the solver to report
            cands.append({"kind": "ball", "k": k, "S": cols, "sgn": np.sign(vals), "a": a, "g": g, "h": h,
                          "cols": cols})
        elif fn.size == 1 and fn[0] not in set(cols.tolist()) and abs(fv[0]) == a:
            j = int(fn[0])
            cands.append({"kind": "soc", "k": k, "head": j, "sh": float(np.sign(fv[0])), "S": cols,
                          "sgn": np.sign(vals), "a": a, "g": g, "h": h, "cols": np.concatenate([[j], cols])})
    used = np.zeros(n, dtype=bool)
    chosen = []
    for c in sorted(cands, key=lambda c: (-c["cols"].size, c["k"])):
        if not used[c["cols"]].any():
            used[c["cols"]] = True
            chosen.append(c)
    if not chosen:
        return None, "no relocatable cone"
    socs = sorted([c for c in chosen if c["kind"] == "soc"], key=lambda c: c["k"])
    balls = sorted([c for c in chosen if c["kind"] == "ball"], key=lambda c: c["k"])
    T = np.concatenate([c["cols"] for c in socs + balls]).astype(np.int64)
    rest = np.array(sorted(set(range(n)) - set(T.tolist())), dtype=np.int64)
    absorbed = {c["k"] for c in chosen}
    keep = [k for k in range(len(d.get("cones", []))) if k not in absorbed]
    return {"socs": socs, "balls": balls, "T": T, "perm": np.concatenate([T, rest]), "keep_cones": keep,
            "n_soc": int(sum(c["cols"].size for c in socs)), "n_ball": int(sum(c["cols"].size for c in balls)),
            "n_x": int(T.size), "K": len(chosen)}, "RELOCATABLE_CONE"


# ----------------------------------------------------------------------------------------------------------------
# Formulations
# ----------------------------------------------------------------------------------------------------------------
def _objective_atom(P, q, nb, primal_weight, l, u, off):
    """(f atom, primal weight used or None). See the module docstring for the linear-objective rule."""
    P = sp.csc_matrix(P)
    if P.nnz:
        P_op = SparseLinOp.from_scipy(P, power_iters=POWER_ITERS)
        Lf = float(P_op.norm_bound)
        if Lf > 0:
            return QuadraticForm(P=P_op, q=jnp.asarray(q), r=0.0,
                                 props=PropSet(lipschitz=Lf, cocoercivity=1.0 / Lf, prox_friendly=False,
                                               strong_convexity=0.0)), None
    w = primal_weight
    if w == "auto":
        rhs = np.linalg.norm(np.concatenate([l[np.isfinite(l)], u[np.isfinite(u)], off]))
        cn = np.linalg.norm(q)
        w = rhs / cn if rhs > 0 and cn > 0 else 1.0
    Lf = max(float(nb), 1e-12) / float(w)
    return LinearCost(c=jnp.asarray(q), props=PropSet(lipschitz=Lf, cocoercivity=1.0 / Lf, averagedness=0.5,
                                                      prox_friendly=True)), float(w)


def _h_atom(l, u, off, bid, head, K):
    parts = [(Box(lo=jnp.asarray(l), hi=jnp.asarray(u)), len(l))]
    if K:
        parts.append((SOCProductProj(bid=jnp.asarray(bid), head=jnp.asarray(head), sign=jnp.ones(len(off)),
                                     offset=jnp.asarray(off), K=K), len(off)))
    return _slices(parts)


def build_socp_formulation(d, cand, plan=None, ruiz_iters=10, primal_weight="auto"):
    """cand in {std_raw, std_ruiz, reloc_cone, reloc_cone_raw}. Returns a uni.Formulation (decode maps to ORIGINAL x).

    reloc_cone applies cone-aware Ruiz (columns tied inside each absorbed cone) and is the counterpart of std_ruiz;
    reloc_cone_raw applies none and is the counterpart of std_raw.
    """
    t0 = time.perf_counter()
    n = validate_socp(d)
    P = sp.csc_matrix(d["P"]).astype(np.float64)
    q = np.asarray(d["q"], dtype=np.float64)
    A = sp.csr_matrix(d["A"]).astype(np.float64)
    l, u = np.asarray(d["l"], dtype=np.float64), np.asarray(d["u"], dtype=np.float64)
    cones = d.get("cones", [])
    if cand in ("std_raw", "std_ruiz"):
        Mc, off, bid, head, dims = conic_rows(cones, n)
        M = sp.csr_matrix(sp.vstack([A, Mc]))
        m_lin, K = A.shape[0], len(dims)
        if M.shape[0] == 0:
            raise ValueError("no constraints: nothing for L")
        d1, d2 = np.ones(M.shape[0]), np.ones(n)
        if cand == "std_ruiz":
            row_group = np.concatenate([np.arange(m_lin), m_lin + bid]).astype(np.int64)
            d1, d2 = cone_ruiz(P, M, row_group, np.arange(n), iters=ruiz_iters)
        Ms = sp.csc_matrix(sp.diags(d1) @ M @ sp.diags(d2))
        Ps = sp.csc_matrix(sp.diags(d2) @ P @ sp.diags(d2))
        qs = d2 * q
        ls, us = d1[:m_lin] * l, d1[:m_lin] * u
        offs = d1[m_lin:] * off                  # d1 is constant on each cone block, so delta K = K and c -> delta c
        L = SparseLinOp.from_scipy(Ms, power_iters=POWER_ITERS)
        f, w = _objective_atom(Ps, qs, L.norm_bound, primal_weight, ls, us, offs)
        h = _h_atom(ls, us, offs, bid, head, K)
        prob = CompositeProblem(f=f, g=Zero(), h=h, L=L,
                                data={"P": Ps, "q": qs, "r": d.get("r", 0.0), "L": Ms, "l": ls, "u": us,
                                      "offset": offs, "cone_dims": dims, "primal_weight": w},
                                shape=(n,), name=f"socp_{cand}")
        return Formulation(cand, prob, lambda z, d2=d2: d2 * np.asarray(z), Ms, np.arange(n),
                           time.perf_counter() - t0, 0)
    if cand not in ("reloc_cone", "reloc_cone_raw"):
        raise ValueError(cand)
    if plan is None:
        plan, why = plan_cone(d)
        if plan is None:
            raise ValueError(f"reloc_cone not applicable: {why}")
    perm, n_x = plan["perm"], plan["n_x"]
    kept = [cones[k] for k in plan["keep_cones"]]
    if A.shape[0] == 0 and not kept:
        # every constraint is absorbed, but the iteration still needs a linear operator: add ONE row the absorbed set
        # already implies (f'x + h >= 0 for a cone, |a s_0 x_0 + g_0| <= h for a ball), so the problem is unchanged
        c0 = (plan["socs"] + plan["balls"])[0]
        if c0["kind"] == "soc":
            j, coef, lo_, hi_ = c0["head"], c0["sh"] * c0["a"], -c0["h"], np.inf
        else:
            j, coef, lo_, hi_ = int(c0["S"][0]), c0["sgn"][0] * c0["a"], -c0["h"] - c0["g"][0], c0["h"] - c0["g"][0]
        A = sp.csr_matrix((np.array([coef]), (np.array([0]), np.array([j]))), shape=(1, n))
        l, u = np.array([lo_]), np.array([hi_])
    Mc, off, bid, head, dims = conic_rows(kept, n)
    M = sp.csr_matrix(sp.vstack([A, Mc]))[:, perm]
    Pp = sp.csc_matrix(P[perm][:, perm])
    qp_ = q[perm]
    m_lin, K = A.shape[0], len(dims)
    # column groups in own coordinates: one per absorbed cone (leading, contiguous), singletons for the rest
    sizes = [c["cols"].size for c in plan["socs"] + plan["balls"]]
    col_group = np.concatenate([np.repeat(np.arange(len(sizes)), sizes),
                                len(sizes) + np.arange(n - n_x)]).astype(np.int64)
    d1, d2 = np.ones(M.shape[0]), np.ones(n)
    if cand == "reloc_cone":
        row_group = np.concatenate([np.arange(m_lin), m_lin + bid]).astype(np.int64)
        d1, d2 = cone_ruiz(Pp, M, row_group, col_group, iters=ruiz_iters)
    Ms = sp.csc_matrix(sp.diags(d1) @ M @ sp.diags(d2))
    Ps = sp.csc_matrix(sp.diags(d2) @ Pp @ sp.diags(d2))
    qs = d2 * qp_
    ls, us, offs = d1[:m_lin] * l, d1[:m_lin] * u, d1[m_lin:] * off
    # g: in own coordinates x_orig[perm] = d2 * z with d2 = delta on a cone, the cone reads
    # (s_t z_t + h/(a delta), s_S z_S + g/(a delta)) in Q, and a ball ||z_S + s g/(a delta)|| <= h/(a delta)
    parts, o = [], 0
    if plan["socs"]:
        sgn, offv, bidv, headv = [], [], [], []
        for i, c in enumerate(plan["socs"]):
            ad = c["a"] * d2[o]
            sgn += [[c["sh"]], c["sgn"]]
            offv += [[c["h"] / ad], c["g"] / ad]
            bidv.append(np.full(c["cols"].size, i))
            headv.append(np.arange(c["cols"].size) == 0)
            o += c["cols"].size
        parts.append((SOCProductProj(bid=jnp.asarray(np.concatenate(bidv)), head=jnp.asarray(np.concatenate(headv)),
                                     sign=jnp.asarray(np.concatenate(sgn)), offset=jnp.asarray(np.concatenate(offv)),
                                     K=len(plan["socs"])), plan["n_soc"]))
    if plan["balls"]:
        cen, rad, bidv = [], [], []
        for i, c in enumerate(plan["balls"]):
            ad = c["a"] * d2[o]
            cen.append(-c["sgn"] * c["g"] / ad)
            rad.append(c["h"] / ad)
            bidv.append(np.full(c["cols"].size, i))
            o += c["cols"].size
        parts.append((BallProductProj(bid=jnp.asarray(np.concatenate(bidv)), center=jnp.asarray(np.concatenate(cen)),
                                      radius=jnp.asarray(np.array(rad)), K=len(plan["balls"])), plan["n_ball"]))
    parts.append((Zero(), n - n_x))
    g = _slices(parts)
    L = SparseLinOp.from_scipy(Ms, power_iters=POWER_ITERS)
    f, w = _objective_atom(Ps, qs, L.norm_bound, primal_weight, ls, us, offs)
    h = _h_atom(ls, us, offs, bid, head, K)
    prob = CompositeProblem(f=f, g=g, h=h, L=L,
                            data={"P": Ps, "q": qs, "r": d.get("r", 0.0), "L": Ms, "l": ls, "u": us, "offset": offs,
                                  "cone_dims": dims, "primal_weight": w},
                            shape=(n,), name=f"socp_{cand}")

    def decode(z, perm=perm, d2=d2, n=n):
        x = np.empty(n)
        x[perm] = d2 * np.asarray(z)
        return x

    return Formulation(f"{cand}:{plan['K']}cones", prob, decode, Ms, perm, time.perf_counter() - t0, n_x)


# ----------------------------------------------------------------------------------------------------------------
# Acceptance and measurement
# ----------------------------------------------------------------------------------------------------------------
def judge_data(d):
    """Precomputed arrays for socp_violation, so a measurement loop does not restack the cones at every check."""
    n = validate_socp(d)
    M, off, bid, head, dims = conic_rows(d.get("cones", []), n)
    return {"P": sp.csc_matrix(d["P"]), "A": sp.csr_matrix(d["A"]), "l": np.asarray(d["l"], float),
            "u": np.asarray(d["u"], float), "M": M, "off": off, "bid": bid, "head": head, "K": len(dims)}


def socp_violation(x, d, jd=None):
    """(linear row violation, cone violation), both relative: viol_i / (1 + |bound_i|) and
    max(0, ||F x + g|| - (f'x + h)) / (1 + |f'x + h|), maximized over rows and cones."""
    jd = jd or judge_data(d)
    x = np.asarray(x, dtype=np.float64)
    lin = 0.0
    if jd["A"].shape[0]:
        Ax = jd["A"] @ x
        l, u = jd["l"], jd["u"]
        vl = np.where(np.isfinite(l), np.maximum(l - Ax, 0.0) / (1.0 + np.abs(np.where(np.isfinite(l), l, 0.0))), 0.0)
        vu = np.where(np.isfinite(u), np.maximum(Ax - u, 0.0) / (1.0 + np.abs(np.where(np.isfinite(u), u, 0.0))), 0.0)
        lin = float(np.max(np.maximum(vl, vu)))
    cone = 0.0
    if jd["K"]:
        y = jd["M"] @ x + jd["off"]
        t = y[jd["head"]]
        nz = np.sqrt(np.bincount(jd["bid"][~jd["head"]], weights=y[~jd["head"]] ** 2, minlength=jd["K"]))
        cone = float(np.max(np.maximum(nz - t, 0.0) / (1.0 + np.abs(t))))
    return lin, cone


def accept_socp(x, d, obj_ref, tol, jd=None):
    """Same verdict shape as uni.accept: (ok, max relative feasibility violation, relative objective error)."""
    lin, cone = socp_violation(x, d, jd)
    gap = abs(socp_objective(x, d) - obj_ref) / max(abs(obj_ref), 1e-300)
    viol = max(lin, cone)
    return (viol <= tol and gap <= tol), viol, float(gap)


def measure_ladder_socp(form, d, obj_ref, tols=uni.LADDER, cap=100000, K=50, deadline=None):
    """uni.measure_ladder with accept_socp: iterations to acceptance at every rung, resolution K, one run.

    Same step rule (t = 1/L_f, s = 0.49/(t ||L||^2)), same chunked jit loop, same stop rule. Also returns the first
    accepted original-coordinate iterate per rung (x_at), which the relocation equivalence check compares.
    """
    prob = form.prob
    Lf, nb = float(prob.f.props.lipschitz), float(prob.L.norm_bound)
    t = 1.0 / Lf
    s = 0.49 / (t * nb ** 2)
    t0 = time.perf_counter()
    built = build_scheme(prob, "condat_vu", CondatVuParams(t=t, s=s), check=True)
    T, dyn = built.T, jax.tree_util.tree_map(jnp.asarray, built.dyn)
    v = jax.tree_util.tree_map(jnp.asarray, built.state0)
    # same bounded-overshoot rule as uni.measure_ladder: one chunk of a heavy conic iteration ran a 20 s budget to
    # 82 s, so the first chunk is small and K adapts towards a fifth of a second
    K_req, K = K, (min(K, 5) if deadline is not None else K)
    make_exe = lambda k: jax.jit(lambda state, dyn_: jax.lax.fori_loop(0, k, lambda i, z: T(z, i, dyn_), state))
    exe = make_exe(K)
    chunk_s, recompiles = [None], [0]
    v = exe(v, dyn)
    jax.block_until_ready(v)
    setup_s = time.perf_counter() - t0 + form.build_s
    jd = judge_data(d)
    it, iter_time = K, 0.0
    hit = {tol: None for tol in tols}
    x_at = {}
    rv = gap = None
    x = None
    while True:
        x = form.decode(np.asarray(v[0]))
        if not np.all(np.isfinite(x)):
            break
        _, rv, gap = accept_socp(x, d, obj_ref, max(tols), jd)
        for tol in tols:
            if hit[tol] is None and rv <= tol and gap <= tol:
                hit[tol] = it
                x_at[f"{tol:g}"] = x.copy()
        if hit[min(tols)] is not None or it >= cap or (deadline is not None and time.perf_counter() >= deadline):
            break
        if deadline is not None and chunk_s[0] is not None and recompiles[0] < 3:
            left = max(deadline - time.perf_counter(), 1e-9)
            K_new = int(np.clip(K * min(0.2, 0.25 * left) / max(chunk_s[0], 1e-9), 1, K_req))
            if K_new != K and (K_new < K * 0.6 or K_new > K * 2):
                K, recompiles[0] = K_new, recompiles[0] + 1
                exe = make_exe(K)
                chunk_s[0] = None
        c0 = time.perf_counter()
        v = exe(v, dyn)
        jax.block_until_ready(v)
        chunk_s[0] = time.perf_counter() - c0
        iter_time += chunk_s[0]
        it += K
    ran = max(it - K, 1)
    lin, cone = socp_violation(x, d, jd) if x is not None and np.all(np.isfinite(x)) else (None, None)
    return {"N_at": {f"{tol:g}": hit[tol] for tol in tols}, "iters_run": it, "setup_s": setup_s,
            "t_iter": iter_time / ran, "row_violation": rv, "lin_violation": lin, "cone_violation": cone,
            "obj_gap": gap, "normL": nb, "Lf": Lf, "x": x, "x_at": x_at}


# ----------------------------------------------------------------------------------------------------------------
# Generators. Each is feasible (a strictly feasible point is built in) and bounded by construction, seedable, and
# returns the conic dict. Sizes are the first argument; the other knobs set difficulty.
# ----------------------------------------------------------------------------------------------------------------
def _eye_rows(cols, n):
    """Rows selecting the given columns: e_{cols[i]}' in row i. CSR, so stack_cones uses it without a conversion."""
    cols = np.asarray(cols, dtype=np.int64)
    return sp.csr_matrix((np.ones(cols.size), (np.arange(cols.size), cols)), shape=(cols.size, n))


def _unit_row(j, n):
    """e_j' as a sparse 1 x n row (a dense f per cone would be O(K n) memory)."""
    return sp.csr_matrix((np.ones(1), (np.zeros(1, dtype=np.int64), np.array([j]))), shape=(1, n))


def portfolio_risk(n, k=None, seed=0, density=None, risk_mult=1.5, lift=False):
    """max mu'x  s.t.  ||Sigma^{1/2} x|| <= sigma, 1'x = 1, x >= 0, with Sigma = F F' + diag(D).

    Sigma^{1/2} x is written as G x with G = [F'; diag(sqrt(D))], since ||G x||^2 = x' Sigma x exactly and G stays
    sparse. sigma = risk_mult * ||G x_u|| at the uniform portfolio, so x_u is strictly feasible for risk_mult > 1 and
    the budget simplex keeps the problem bounded; the all-in-one-asset optimum of the unconstrained LP has risk
    O(1) against O(1/sqrt(n)) for x_u, so the cone is active unless risk_mult is very large. risk_mult near 1
    tightens it. lift=True adds y = G x as variables (the CVXPY-style lifted form): the cone becomes the ball
    ||y|| <= sigma on original variables, which reloc_cone can absorb.
    """
    rng = np.random.default_rng(seed)
    k = k or max(2, int(np.sqrt(n)))
    density = density if density is not None else min(1.0, 5.0 / k)
    F = _sprandn(n, k, density, seed)
    D = rng.uniform(0.5, 1.5, n)
    mu = 1.0 + 0.5 * rng.standard_normal(n)
    G = sp.csc_matrix(sp.vstack([F.T, sp.diags(np.sqrt(D))]))
    sigma = float(risk_mult * np.linalg.norm(G @ (np.ones(n) / n)))
    mG = G.shape[0]
    if not lift:
        A = sp.vstack([sp.csc_matrix(np.ones((1, n))), sp.identity(n, format="csc")], format="csc")
        cones = [{"F": G, "g": np.zeros(mG), "f": sp.csr_matrix((1, n)), "h": sigma}]
        nv = n
    else:
        nv = n + mG
        A = sp.vstack([sp.hstack([G, -sp.identity(mG, format="csc")]),
                       sp.hstack([sp.csc_matrix(np.ones((1, n))), sp.csc_matrix((1, mG))]),
                       sp.hstack([sp.identity(n, format="csc"), sp.csc_matrix((n, mG))])], format="csc")
        cones = [{"F": _eye_rows(n + np.arange(mG), nv), "g": np.zeros(mG), "f": sp.csr_matrix((1, nv)), "h": sigma}]
    l = np.concatenate([np.zeros(mG) if lift else [], [1.0], np.zeros(n)])
    u = np.concatenate([np.zeros(mG) if lift else [], [1.0], np.full(n, np.inf)])
    return {"P": sp.csc_matrix((nv, nv)), "q": np.concatenate([-mu, np.zeros(nv - n)]), "r": 0.0, "A": A,
            "l": l, "u": u, "cones": cones, "name": f"portfolio_risk(n={n},k={k},risk={risk_mult:g},lift={lift})"}


def group_lasso(n, group_size=5, m=None, seed=0, lam_frac=0.3, density=None, active_frac=0.2, noise=0.1,
                lift=True):
    """min 1/2 ||A x - b||^2 + lam sum_g t_g  s.t.  ||x_g|| <= t_g, groups of group_size contiguous features.

    lam = lam_frac * lam_max with lam_max = max_g ||A_g' b||, the smallest lam at which x = 0 is optimal, so
    lam_frac in (0, 1) gives a nonzero group-sparse solution and smaller values activate more groups. Always
    feasible (t large); bounded because the objective is >= 0. lift=True uses the OSQP lasso form (residual
    y = A x - b as variables, P = diag(0, 0, I)), which keeps P sparse; lift=False puts A'A in P.
    Variables: [x (n), t (n/group_size)] and y (m) when lifted.
    """
    rng = np.random.default_rng(seed)
    G = max(1, n // group_size)
    n = G * group_size
    m = m or n
    density = density if density is not None else min(1.0, 10.0 / m)
    Ad = _sprandn(m, n, density, seed)
    groups = np.arange(n).reshape(G, group_size)
    active = rng.random(G) < active_frac
    active[rng.integers(G)] = True
    xt = np.zeros(n)
    xt[groups[active].ravel()] = rng.standard_normal(int(active.sum()) * group_size)
    b = Ad @ xt + noise * rng.standard_normal(m)
    Atb = Ad.T @ b
    lam = float(lam_frac * np.max(np.linalg.norm(Atb[groups], axis=1)))
    nv = n + G + (m if lift else 0)
    cones = [{"F": _eye_rows(groups[i], nv), "g": np.zeros(group_size), "f": _unit_row(n + i, nv), "h": 0.0}
             for i in range(G)]
    if lift:
        P = sp.block_diag([sp.csc_matrix((n + G, n + G)), sp.identity(m, format="csc")], format="csc")
        q = np.concatenate([np.zeros(n), lam * np.ones(G), np.zeros(m)])
        A = sp.hstack([Ad, sp.csc_matrix((m, G)), -sp.identity(m, format="csc")], format="csc")
        l = u = b.copy()
        r = 0.0
    else:
        P = sp.block_diag([sp.csc_matrix(Ad.T @ Ad), sp.csc_matrix((G, G))], format="csc")
        q = np.concatenate([-Atb, lam * np.ones(G)])
        A, l, u = sp.csc_matrix((0, nv)), np.zeros(0), np.zeros(0)
        r = 0.5 * float(b @ b)
    return {"P": P, "q": q, "r": r, "A": A, "l": l, "u": u.copy(), "cones": cones,
            "name": f"group_lasso(n={n},gs={group_size},m={m},lam={lam_frac:g},lift={lift})"}


def min_enclosing_ball(n_points, dim=None, seed=0, anisotropy=1.0, outlier_frac=0.0):
    """min r  s.t.  ||c - p_i|| <= r. Variables [r, c (dim)].

    Feasible (c = 0, r = max ||p_i||, strictly for any larger r) and bounded (r >= ||c - p_1|| >= 0); r* > 0 unless
    all points coincide. anisotropy >= 1 spreads the axis scales from 1 down to 1/anisotropy, which makes the
    conic rows badly scaled against each other; outlier_frac moves a fraction of points 5x further out, so few
    cones are active at the solution (a degenerate, support-set-of-few-points optimum).
    """
    rng = np.random.default_rng(seed)
    dim = dim or max(2, int(np.sqrt(n_points)))
    scales = anisotropy ** (-np.linspace(0.0, 1.0, dim))
    pts = rng.standard_normal((n_points, dim)) * scales
    out = rng.random(n_points) < outlier_frac
    pts[out] *= 5.0
    nv = 1 + dim
    Fc = _eye_rows(1 + np.arange(dim), nv)
    e0 = _unit_row(0, nv)
    cones = [{"F": Fc, "g": -pts[i], "f": e0, "h": 0.0} for i in range(n_points)]
    return {"P": sp.csc_matrix((nv, nv)), "q": np.eye(1, nv, 0).ravel(), "r": 0.0, "A": sp.csc_matrix((0, nv)),
            "l": np.zeros(0), "u": np.zeros(0), "cones": cones,
            "name": f"meb(N={n_points},dim={dim},aniso={anisotropy:g},out={outlier_frac:g})"}


def robust_lp(n, m=None, seed=0, unc_rank=None, rho=0.1, density=None, box=1.0):
    """min c'x  s.t.  a_i'x + ||E_i x|| <= b_i (ellipsoidal uncertainty {a_i + E_i' v : ||v|| <= 1}), -box <= x <= box.

    b_i = slack_i > 0 makes x = 0 strictly feasible, so c'x* <= -eps ||c||^2 < 0 (bounded away from zero, which the
    relative objective test needs), and the box bounds it. The nominal rows have about 5 nonzeros per unit of box
    width over the slack, so the box optimum -box sign(c) violates them and the cones are active. rho scales the
    uncertainty (larger is more conservative and more curved); unc_rank is the rank of each E_i.
    """
    rng = np.random.default_rng(seed)
    m = m or max(1, n // 2)
    unc_rank = unc_rank or max(1, min(5, n // 4))
    density = density if density is not None else min(1.0, 5.0 / n)
    An = sp.csr_matrix(_sprandn(m, n, density, seed))
    c = rng.standard_normal(n)
    slack = rng.uniform(0.5, 1.5, m)
    cones = []
    for i in range(m):
        E = rho * _sprandn(unc_rank, n, max(density, min(1.0, 3.0 / n)), seed + 1 + i)
        cones.append({"F": sp.csc_matrix(E), "g": np.zeros(unc_rank), "f": -An[i], "h": float(slack[i])})
    return {"P": sp.csc_matrix((n, n)), "q": c, "r": 0.0, "A": sp.identity(n, format="csc"),
            "l": -box * np.ones(n), "u": box * np.ones(n), "cones": cones,
            "name": f"robust_lp(n={n},m={m},rank={unc_rank},rho={rho:g})"}


def facility_location(n_points, dim=2, seed=0, weight_skew=0.0, box_offset=0.0, box_halfwidth=3.0):
    """min sum_i w_i t_i  s.t.  ||x - p_i|| <= t_i,  lo <= x <= hi (Weber problem in a box). Variables [x (dim), t].

    Feasible and bounded (w > 0, t >= 0); the value is > 0 unless every point coincides. weight_skew draws
    w = exp(skew * N(0,1)): a dominant weight puts the optimum AT a data point, the apex of that cone, where the
    problem is nonsmooth. The box is centered box_offset standard deviations away from the centroid along a random
    direction with half-width box_halfwidth standard deviations, so box_offset > box_halfwidth makes it active.
    """
    rng = np.random.default_rng(seed)
    pts = rng.standard_normal((n_points, dim))
    w = np.exp(weight_skew * rng.standard_normal(n_points))
    w = w * (n_points / w.sum())
    std = float(pts.std()) or 1.0
    direction = rng.standard_normal(dim)
    direction /= np.linalg.norm(direction)
    center = pts.mean(axis=0) + box_offset * std * direction
    nv = dim + n_points
    Fx = _eye_rows(np.arange(dim), nv)
    cones = [{"F": Fx, "g": -pts[i], "f": _unit_row(dim + i, nv), "h": 0.0} for i in range(n_points)]
    return {"P": sp.csc_matrix((nv, nv)), "q": np.concatenate([np.zeros(dim), w]), "r": 0.0,
            "A": _eye_rows(np.arange(dim), nv), "l": center - box_halfwidth * std,
            "u": center + box_halfwidth * std, "cones": cones,
            "name": f"facility(N={n_points},dim={dim},skew={weight_skew:g},off={box_offset:g})"}


FAMILIES = {"portfolio_risk": portfolio_risk, "group_lasso": group_lasso, "min_enclosing_ball": min_enclosing_ball,
            "robust_lp": robust_lp, "facility_location": facility_location}


def make_socp(family, size, seed=0, **kw):
    return FAMILIES[family](size, seed=seed, **kw)
