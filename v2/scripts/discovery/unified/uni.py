"""Shared machinery for the unified formulation-selection experiments (2026-09-15).

The thesis under test: for a matrix-free first-order method the formulation (which constraint block is absorbed
into the proximal step, which stays in the linear operator) sets the iteration count N, while HOW the projection
is implemented sets the per-iteration cost t and, only if inexact, perturbs N. One cost model
    T(P, S) = T_setup(S) + N(P, S) * t(S, hardware)
is then both the formulation selector and the specification generator for operator synthesis.

Everything here measures against the ORIGINAL problem: an iterate is accepted when its row violation on the
original (P, q, A, l, u) is at most tol AND its objective is within tol of a Clarabel reference solved at 1e-9.
The base iteration is plain Condat-Vu with gate-derived fixed steps, deliberately NOT the adaptive driver, so the
iteration count reflects the formulation and not a controller.

Encodings are randomized by default (row and column permutation). The earlier relocation rule assumed the absorbed
variables formed a leading prefix and refused on every shuffled instance; plan_block() here reads the absorbed
block as a SET and canonicalizes internally, so a decision cannot depend on the generator's column order.
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import numpy as np
import scipy.sparse as sp
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from scipy.sparse.linalg import svds

from atlas.bench import qp_cell
from atlas.bench.baseline_proc import run_baseline_proc
from atlas.bench.mps_cell import SparseLinOp
from atlas.bench.qp_gen import _sprandn, meb_dual, portfolio, svm_dual
from atlas.bench.qp_gen import FAMILIES as QP_FAMILIES
from atlas.certify.residuals import qp_accuracy_judge
from atlas.ir.spec import CompositeProblem
from atlas.operators.base import PropSet
from atlas.operators.prox import Box, QuadraticForm, Zero, project_box_hyperplane, project_simplex
from atlas.schemes.base_schemes import CondatVuParams
from atlas.schemes.compile import build_scheme
from atlas.schemes.rescale import rescale_qp_problem

POWER_ITERS = 100


# ----------------------------------------------------------------------------------------------------------------
# Families with an absorbable block. portfolio / svm_dual / meb_dual come from qp_gen; the two below are new.
# ----------------------------------------------------------------------------------------------------------------
def knapsack_ridge(n, rho=0.0, seed=0, density=1e-3):
    """min rho/2 |x|^2 + 1/2 |y|^2  s.t.  F x - y = g, 0 <= x <= 1, a'x = b. Box-hyperplane block, vector RHS."""
    rng = np.random.default_rng(seed)
    m = n // 2
    F = _sprandn(m, n, density, seed)
    xt = rng.random(n)
    g = F @ xt + rng.standard_normal(m) * 0.01
    a = np.abs(rng.standard_normal(n)) + 0.1
    b = float(a @ xt)
    Pxx = sp.identity(n, format="csc") * rho if rho > 0 else sp.csc_matrix((n, n))
    P = sp.block_diag([Pxx, sp.identity(m, format="csc")], format="csc")
    A = sp.vstack([sp.hstack([F, -sp.identity(m, format="csc")], format="csc"),
                   sp.hstack([sp.identity(n, format="csc"), sp.csc_matrix((n, m))], format="csc"),
                   sp.hstack([sp.csc_matrix(a.reshape(1, -1)), sp.csc_matrix((1, m))], format="csc")], format="csc")
    l = np.concatenate([g, np.zeros(n), [b]])
    u = np.concatenate([g, np.ones(n), [b]])
    return {"P": P, "q": np.zeros(n + m), "r": 0.0, "A": A, "l": l, "u": u}


def budget_band(n, k=None, seed=0, density=1e-3, band=(0.9, 1.0), cap_mult=20.0):
    """Portfolio with a budget BAND and per-asset position limits (Moehle, Gindi, Boyd, Kochenderfer 2023 style).

    min x'Dx + y'y - mu'x  s.t.  y = F'x,  eta_lo <= 1'x <= eta_hi,  0 <= x <= cap.
    The dense row is an INEQUALITY, so the absorbable set is a box intersected with a slab, which is NOT in the
    original operator library: before this module the only options were to keep the row in L or to use a generic
    alternating projection.
    """
    rng = np.random.default_rng(seed)
    k = k or max(2, int(np.sqrt(n)))
    F = _sprandn(n, k, density, seed)
    D = rng.random(n) * np.sqrt(k)
    mu = rng.standard_normal(n)
    cap = min(1.0, cap_mult / n)
    P = sp.block_diag([2.0 * sp.diags(D), 2.0 * sp.identity(k, format="csc")], format="csc")
    q = np.concatenate([-mu, np.zeros(k)])
    A = sp.vstack([sp.hstack([F.T.tocsc(), -sp.identity(k, format="csc")], format="csc"),
                   sp.hstack([sp.csc_matrix(np.ones((1, n))), sp.csc_matrix((1, k))], format="csc"),
                   sp.hstack([sp.identity(n, format="csc"), sp.csc_matrix((n, k))], format="csc")], format="csc")
    l = np.concatenate([np.zeros(k), [band[0]], np.zeros(n)])
    u = np.concatenate([np.zeros(k), [band[1]], cap * np.ones(n)])
    return {"P": P, "q": q, "r": 0.0, "A": A, "l": l, "u": u}


def make_instance(family, size, seed=0, **kw):
    if family == "portfolio":
        return portfolio(size, k=kw.get("k"), seed=seed)
    if family == "svm_dual":
        return svm_dual(size, seed=seed)
    if family == "meb_dual":
        return meb_dual(size, seed=seed)
    if family == "knapsack_ridge":
        return knapsack_ridge(size, rho=kw.get("rho", 0.0), seed=seed)
    if family == "budget_band":
        return budget_band(size, k=kw.get("k"), seed=seed)
    return QP_FAMILIES[family](size, seed=seed)


def permute_instance(d, seed):
    """Random row AND column permutation. Same problem, different encoding."""
    rng = np.random.default_rng(seed)
    n, m = d["P"].shape[0], d["A"].shape[0]
    pc, pr = rng.permutation(n), rng.permutation(m)
    P = sp.csc_matrix(d["P"])[pc][:, pc]
    A = sp.csc_matrix(d["A"])[pr][:, pc]
    return {"P": sp.csc_matrix(P), "q": np.asarray(d["q"])[pc], "r": d["r"], "A": sp.csc_matrix(A),
            "l": np.asarray(d["l"])[pr], "u": np.asarray(d["u"])[pr]}


# ----------------------------------------------------------------------------------------------------------------
# Diagnosis and a permutation-robust plan
# ----------------------------------------------------------------------------------------------------------------
def top_spectrum(A, k=2, oversample=6, power_iters=7, seed=0):
    """Top-k singular values and left vectors, at BOUNDED cost.

    ARPACK (svds) is not bounded: on an LP with 12k variables and clustered singular values it took 55 s, and this is
    a heuristic diagnosis paid inside a solve budget. A randomized range finder with a fixed number of power
    iterations costs a fixed number of sparse products and is accurate enough to rank one row against the rest.
    """
    A = sp.csc_matrix(A).astype(np.float64)
    if min(A.shape) <= k + 1:                       # tiny instances (HS21, TAME) go dense
        U, S, _ = np.linalg.svd(A.toarray(), full_matrices=False)
        S = np.concatenate([S, np.zeros(max(0, k - S.size))])[:k]
        U = np.concatenate([U, np.zeros((U.shape[0], max(0, k - U.shape[1])))], axis=1)[:, :k]
        return S, U
    ell = min(k + oversample, min(A.shape) - 1)
    rng = np.random.default_rng(seed)
    Y = A @ rng.standard_normal((A.shape[1], ell))
    Q, _ = np.linalg.qr(Y)
    for _ in range(power_iters):
        Q, _ = np.linalg.qr(A @ (A.T @ Q))
    B = Q.T @ A
    Ub, S, _ = np.linalg.svd(B, full_matrices=False)
    U = Q @ Ub
    order = np.argsort(S)[::-1][:k]
    return S[order], U[:, order]


def _top_spectrum_arpack(A, k=2):
    U, S, _ = svds(A, k=k)
    order = np.argsort(S)[::-1]
    return S[order], U[:, order]


def diagnose(d):
    if sp.csc_matrix(d["P"]).nnz == 0:
        import lp_families
        return lp_families.lp_diagnose(d)
    return _diagnose(d)


def _diagnose(d):
    prob = qp_cell.to_composite(d, power_iters=POWER_ITERS)
    resc, _, _, _ = rescale_qp_problem(prob, "ruiz", iters=10, power_iters=POWER_ITERS)
    S, U = top_spectrum(resc.data["A"])
    mass = U[:, 0] ** 2
    j = int(np.argmax(mass))
    gap = float(S[0] / max(S[1], 1e-30))
    return {"top_row": j, "row_mass": float(mass[j]), "singular_gap": gap,
            "fires": bool(mass[j] > 0.5 and gap >= 3.0)}


def plan_block(d, diag):
    """Read the absorbable block off (A, l, u) as a SET of columns. None means not relocatable by this rule."""
    A = sp.csr_matrix(d["A"])
    l = np.asarray(d["l"], dtype=float)
    u = np.asarray(d["u"], dtype=float)
    j = int(diag["top_row"])
    s_, e_ = A.indptr[j], A.indptr[j + 1]
    T = A.indices[s_:e_].copy()
    a = A.data[s_:e_].copy()
    if T.size == 0:
        return None, "empty dense row"
    order = np.argsort(T)
    T, a = T[order], a[order]
    pos = {int(c): i for i, c in enumerate(T)}
    lo = np.full(T.size, -np.inf)
    hi = np.full(T.size, np.inf)
    bounded = np.zeros(T.size, dtype=bool)
    bound_rows = []
    nnz_per_row = np.diff(A.indptr)
    for r in np.flatnonzero(nnz_per_row == 1):
        if r == j:
            continue
        c = int(A.indices[A.indptr[r]])
        if c not in pos:
            continue
        v = float(A.data[A.indptr[r]])
        if v == 0.0:
            continue
        rl, ru = (l[r] / v, u[r] / v) if v > 0 else (u[r] / v, l[r] / v)
        i = pos[c]
        lo[i], hi[i] = max(lo[i], rl), min(hi[i], ru)
        bounded[i] = True
        bound_rows.append(int(r))
    if not bounded.all():
        return None, f"only {int(bounded.sum())} of {T.size} absorbed vars carry bound rows"
    keep = np.array(sorted(set(range(A.shape[0])) - set(bound_rows) - {j}), dtype=int)
    fully_absorbed = keep.size == 0
    if fully_absorbed:
        # every constraint is in the absorbed set, so the method is projected gradient on f. The iteration still needs a
        # linear operator: keep ONE bound row of an absorbed variable in L. It is redundant (the prox enforces the same
        # bound exactly), so the problem is unchanged, and it has norm |coefficient|.
        keep = np.array([bound_rows[0]], dtype=int)
    if np.isclose(l[j], u[j]):
        is_simplex = (np.allclose(a, 1.0) and np.isclose(l[j], 1.0) and np.allclose(lo, 0.0) and np.isinf(hi).all())
        kind = "simplex" if is_simplex else "box_hyperplane"
    else:
        kind = "box_slab"
    n = d["P"].shape[0]
    rest = np.array(sorted(set(range(n)) - set(T.tolist())), dtype=int)
    perm = np.concatenate([T, rest])                                     # canonical order: absorbed block first
    return {"kind": kind, "T": T, "a": a, "lo": lo, "hi": hi, "b_lo": float(l[j]), "b_hi": float(u[j]),
            "keep": keep, "perm": perm, "n_x": int(T.size), "dense_row": j,
            "fully_absorbed": bool(fully_absorbed)}, "RELOCATABLE"


# ----------------------------------------------------------------------------------------------------------------
# Projection implementations. Exact ones must return the same point; Dykstra-k is a generic inexact alternative.
# ----------------------------------------------------------------------------------------------------------------
def _p_affine_band(x, a, b_lo, b_hi, a_sq):
    s = jnp.vdot(a, x)
    return x - a * (s - jnp.clip(s, b_lo, b_hi)) / a_sq


def bp_project(v, lo, hi, a, b):
    """Exact breakpoint-sort projection onto {lo <= x <= hi, a'x = b}, valid for coefficients of ANY sign.

    The sort-based sweep assumes a > 0. A first version did not handle negative coefficients, passed the operator
    certificate (whose tests only used a > 0, the stated contract), and then failed to converge on svm_dual, whose dense
    row d'alpha = 0 has entries of both signs. Acceptance on the original problem caught it. The fix is an exact change
    of variables y = sign(a) x: the set becomes {lo' <= y <= hi', |a|'y = b} with the bounds swapped and negated where
    a < 0, and ||x - v|| = ||y - sign(a) v||, so projecting sign(a) v in y and mapping back is the exact projection.
    """
    sgn = jnp.where(a < 0, -1.0, 1.0)
    v_, a_ = sgn * v, jnp.abs(a)
    lo_, hi_ = jnp.where(sgn > 0, lo, -hi), jnp.where(sgn > 0, hi, -lo)
    return sgn * _bp_project_pos(v_, lo_, hi_, a_, b)


def _bp_project_pos(v, lo, hi, a, b):
    """Breakpoint sweep for a > 0 only. Use bp_project, which reduces any sign pattern to this case."""
    big = 1e300
    hi_fin = jnp.isfinite(hi)
    C0 = jnp.sum(jnp.where(hi_fin, a * jnp.where(hi_fin, hi, 0.0), a * v))
    S0 = jnp.sum(jnp.where(hi_fin, 0.0, -a * a))
    bp_hi = jnp.where(hi_fin, (v - jnp.where(hi_fin, hi, 0.0)) / a, -big)
    dC_hi = jnp.where(hi_fin, a * v - a * jnp.where(hi_fin, hi, 0.0), 0.0)
    dS_hi = jnp.where(hi_fin, -a * a, 0.0)
    bp = jnp.concatenate([bp_hi, (v - lo) / a])
    dC = jnp.concatenate([dC_hi, a * lo - a * v])
    dS = jnp.concatenate([dS_hi, a * a])
    order = jnp.argsort(bp)
    bp, dC, dS = bp[order], dC[order], dS[order]
    Cpre = C0 + jnp.concatenate([jnp.zeros(1), jnp.cumsum(dC)])
    Spre = S0 + jnp.concatenate([jnp.zeros(1), jnp.cumsum(dS)])
    phi_at = Cpre[:-1] + Spre[:-1] * bp
    cnt = jnp.sum(phi_at >= b)
    Cs, Ss = Cpre[cnt], Spre[cnt]
    mu_lin = (b - Cs) / jnp.where(Ss == 0.0, -1.0, Ss)
    mu_flat = jnp.where(cnt > 0, bp[jnp.maximum(cnt - 1, 0)], bp[0])
    mu = jnp.where(Ss == 0.0, mu_flat, mu_lin)
    return jnp.clip(v - mu * a, lo, hi)


def newton_project(v, lo, hi, a, b, steps=12):
    """Exact safeguarded semismooth Newton with an exact breakpoint bracket; copy of human_newton.project."""
    big = 1e300
    a2 = a * a
    hi_fin = jnp.isfinite(hi)
    bp_lo = (v - lo) / a
    bp_hi = jnp.where(hi_fin, (v - jnp.where(hi_fin, hi, 0.0)) / a, big)
    h0 = jnp.max(bp_lo)
    mu_min = jnp.minimum(jnp.min(bp_hi), jnp.min(bp_lo))
    C0 = jnp.sum(jnp.where(hi_fin, a * jnp.where(hi_fin, hi, 0.0), a * v))
    S0 = -jnp.sum(jnp.where(hi_fin, 0.0, a2))
    mu_lin = jnp.where(S0 < 0, (b - C0) / jnp.where(S0 < 0, S0, -1.0), mu_min)
    l0 = jnp.minimum(mu_min, mu_lin)

    def body(_, st):
        l_, h_, mu = st
        x = jnp.clip(v - mu * a, lo, hi)
        phi = jnp.dot(a, x) - b
        l_ = jnp.where(phi > 0, mu, l_)
        h_ = jnp.where(phi > 0, h_, mu)
        free = (x > lo) & (x < hi)
        slope = jnp.sum(jnp.where(free, a2, 0.0))
        mu_n = mu + phi / jnp.where(slope > 0, slope, 1.0)
        ok = (slope > 0) & (mu_n >= l_) & (mu_n <= h_)
        return l_, h_, jnp.where(phi == 0.0, mu, jnp.where(ok, mu_n, 0.5 * (l_ + h_)))

    _, _, mu = jax.lax.fori_loop(0, steps, body, (l0, h0, 0.5 * (l0 + h0)))
    return jnp.clip(v - mu * a, lo, hi)


_OPERATOR_FILES = {}


def load_operator(path):
    """project(v, lo, hi, a, b) from a candidate file (a synthesized operator); cached so jit traces reuse it."""
    if path not in _OPERATOR_FILES:
        import importlib.util
        spec = importlib.util.spec_from_file_location(f"op_{len(_OPERATOR_FILES)}", path)
        mod = importlib.util.module_from_spec(spec)
        mod.__dict__.update({"jax": jax, "jnp": jnp, "np": np})     # same prelude as opcheck.py, for both arms
        spec.loader.exec_module(mod)
        _OPERATOR_FILES[path] = mod.project
    return _OPERATOR_FILES[path]


def project_block(x, kind, impl, lo, hi, a, b_lo, b_hi, iters, k):
    if impl == "null":
        # TIMING BASELINE ONLY, not a projection: the cheapest elementwise map on the block, so the relocated iteration
        # timed with it measures the base cost of everything except the projection's own work.
        return jnp.clip(x, lo, hi)
    if impl.startswith("file:"):
        b_eff = b_lo
        if kind == "box_slab":
            b_eff = jnp.clip(jnp.vdot(a, jnp.clip(x, lo, hi)), b_lo, b_hi)
        return load_operator(impl[5:])(x, lo, hi, a, b_eff)
    if impl == "sort":
        return project_simplex(x)                                        # only valid for kind == simplex
    if impl == "bisect":
        if kind == "box_slab":
            # target the nearest point of the band: if a'clip(x) is inside it, mu = 0 reproduces clip(x)
            target = jnp.clip(jnp.vdot(a, jnp.clip(x, lo, hi)), b_lo, b_hi)
            return project_box_hyperplane(x, lo, hi, a, target, iters=iters)
        return project_box_hyperplane(x, lo, hi, a, b_lo, iters=iters)
    if impl in ("breakpoint", "newton"):
        b_eff = b_lo
        if kind == "box_slab":
            b_eff = jnp.clip(jnp.vdot(a, jnp.clip(x, lo, hi)), b_lo, b_hi)
        if impl == "breakpoint":
            return bp_project(x, lo, hi, a, b_eff)
        return newton_project(x, lo, hi, a, b_eff, steps=iters)
    if impl == "dykstra":
        a_sq = jnp.maximum(jnp.vdot(a, a), 1e-300)

        def body(_, st):
            xx, p, qq = st
            y = jnp.clip(xx + p, lo, hi)
            p = xx + p - y
            xn = _p_affine_band(y + qq, a, b_lo, b_hi, a_sq)
            qq = y + qq - xn
            return xn, p, qq

        z0 = jnp.zeros_like(x)
        xk, _, _ = jax.lax.fori_loop(0, k, body, (x, z0, z0))
        return xk
    raise ValueError(impl)


@dataclass(frozen=True)
class BlockProj:
    """Indicator of the absorbed set on the leading (canonicalized) block; the rest passes through."""

    lo: object
    hi: object
    a: object
    b_lo: object
    b_hi: object
    n_x: int
    kind: str
    impl: str
    iters: int = 60
    k: int = 10
    pad: int = 0                                  # extra idempotent re-applications: pure cost, same output
    props: PropSet = field(default=PropSet(averagedness=0.5, prox_friendly=True))

    def prox(self, v, step=None):
        x, y = v[: self.n_x], v[self.n_x:]
        p = project_block(x, self.kind, self.impl, self.lo, self.hi, self.a, self.b_lo, self.b_hi,
                          self.iters, self.k)
        for _ in range(self.pad):
            p = project_block(p, self.kind, self.impl, self.lo, self.hi, self.a, self.b_lo, self.b_hi,
                              self.iters, self.k)
        return jnp.concatenate([p, y])


jax.tree_util.register_dataclass(
    BlockProj, data_fields=["lo", "hi", "a", "b_lo", "b_hi"],
    meta_fields=["n_x", "kind", "impl", "iters", "k", "pad", "props"])


# ----------------------------------------------------------------------------------------------------------------
# PRODUCT relocation: absorb a DISJOINT FAMILY of bounded rows, not one dominant row.
# On real benchmarks the dominant-row diagnosis rarely fires (Maros-Meszaros 7/137, QPLIB 0/10) while rows whose whole
# support is bounded are common (MM 99/137; on 6 of 10 QPLIB instances EVERY general row is, with 3-17 nonzeros).
# A product of box-hyperplane / box-slab sets is projected exactly block by block, so any disjoint family can move
# from L into the proximal step. Which family pays is a cost-model question (N and t), not a norm heuristic.
# ----------------------------------------------------------------------------------------------------------------
def product_row_scores(rule, info):
    """Priority of each absorbable row for the greedy disjoint family (higher is taken first).

    `info` has one entry per candidate row: nnz, l2 (norm of its coefficients), amax, is_eq, and the row index. The
    shipped rule takes the WIDEST rows, which on real MIPLIB relaxations leaves big-M rows (2 nonzeros, norm 2e4) in
    the linear operator while absorbing wide rows of norm 11; selecting by norm converged 10x faster on several
    instances. Neither rule dominates, so the rule is a CANDIDATE the cost model compares, not a global switch.
    """
    if callable(rule):
        return np.asarray(rule(info), dtype=float)
    if rule in (None, "width"):
        return info["nnz"].astype(float)
    if rule == "narrow":
        # MANY SMALL ROWS FIRST. On RouterBench the widest absorbable row is the dense budget row (nnz = n), so taking
        # it first blocks the whole simplex family: measured 23,200 iterations against 150 for the simplex family on
        # the capacitated variant, a 155x difference between two formulations whose operator norms differ by 0.06%.
        return -info["nnz"].astype(float)
    if rule == "norm":
        return info["l2"]
    if rule == "norm_per_col":
        return info["l2"] / np.maximum(info["nnz"], 1)
    if rule == "eq_first":
        return info["is_eq"] * 1e12 + info["nnz"]
    raise ValueError(f"unknown product row rule {rule!r}")


def plan_product(d, min_nnz=2, rule=None, max_nnz=None):
    A = sp.csr_matrix(d["A"])
    l = np.asarray(d["l"], dtype=float)
    u = np.asarray(d["u"], dtype=float)
    m, n = A.shape
    nnz = np.diff(A.indptr)
    # VECTORIZED BOUND-ROW SCAN. The Python loop ran once per single-nonzero row: 1.23 million iterations on a large
    # transport instance, inside a solve budget.
    lo, hi = np.full(n, -np.inf), np.full(n, np.inf)
    has = np.zeros(n, dtype=bool)
    bnd_cols = np.zeros(0, dtype=np.int64)          # bound-row index as two aligned arrays, column-sorted, NOT a dict:
    bnd_rows = np.zeros(0, dtype=np.int64)          # the dict build was one Python iteration per distinct column, which
    br = np.flatnonzero(nnz == 1)                   # is 16.8M iterations on a 64x64 transport instance
    if br.size:
        cols = A.indices[A.indptr[br]]
        vals = A.data[A.indptr[br]]
        keep = vals != 0.0
        br, cols, vals = br[keep], cols[keep], vals[keep]
        pos = vals > 0
        rl = np.where(pos, l[br] / vals, u[br] / vals)
        ru = np.where(pos, u[br] / vals, l[br] / vals)
        np.maximum.at(lo, cols, rl)
        np.minimum.at(hi, cols, ru)
        has[cols] = True
        order = np.argsort(cols, kind="stable")
        bnd_cols, bnd_rows = cols[order].astype(np.int64), br[order].astype(np.int64)
    fin = has & (np.isfinite(lo) | np.isfinite(hi))
    # VECTORIZED CANDIDATE SCAN. The Python loop over rows cost 58 s on an LP with 12k variables, and structure
    # detection is part of a first visit's budget: per row, count support columns that are not bounded and entries
    # that are zero, with one reduceat over the CSR data.
    starts = A.indptr[:-1]
    has_rows = np.flatnonzero(nnz > 0)
    bad_col = (~fin)[A.indices].astype(np.int64)
    zero_val = (A.data == 0.0).astype(np.int64)
    bad_per_row = np.zeros(m, dtype=np.int64)
    zero_per_row = np.zeros(m, dtype=np.int64)
    if has_rows.size:
        bad_per_row[has_rows] = np.add.reduceat(bad_col, starts[has_rows])
        zero_per_row[has_rows] = np.add.reduceat(zero_val, starts[has_rows])
    ok_rows = (nnz >= min_nnz) & (bad_per_row == 0) & (zero_per_row == 0)
    if max_nnz is not None:
        ok_rows &= nnz <= max_nnz            # exclude dense coupling rows so a wide row cannot block a whole family
    cands = np.flatnonzero(ok_rows).tolist()
    used = np.zeros(n, dtype=bool)
    rows = []
    if cands:
        ci = np.asarray(cands)
        l2 = np.sqrt(np.add.reduceat((A.data ** 2), A.indptr[ci])) if ci.size else np.zeros(0)
        amax = np.array([np.max(np.abs(A.data[A.indptr[r]:A.indptr[r + 1]])) for r in cands])
        info = {"row": ci, "nnz": nnz[ci], "l2": l2, "amax": amax,
                "is_eq": np.isclose(l[ci], u[ci]).astype(float)}
        pri = product_row_scores(rule, info)
        order = [int(ci[i]) for i in np.argsort(-pri, kind="stable")]
    else:
        order = []
    for r in order:
        cols = A.indices[A.indptr[r]:A.indptr[r + 1]]
        if not used[cols].any():
            used[cols] = True
            rows.append(r)
    if not rows:
        return None, "no absorbable rows"
    T, bid, a = [], [], []
    for k, r in enumerate(rows):
        cols = A.indices[A.indptr[r]:A.indptr[r + 1]]
        order = np.argsort(cols)
        T.append(cols[order]); a.append(A.data[A.indptr[r]:A.indptr[r + 1]][order]); bid.append(np.full(cols.size, k))
    T, a, bid = np.concatenate(T), np.concatenate(a), np.concatenate(bid)
    # Set arithmetic over range(m) and range(n) built Python sets of 16.8M elements per call and, with the dict above,
    # was the whole of a 413 s structure detection that consumed a 120 s solve budget before a single formulation was
    # tried. Boolean masks are the same computation: absorbed rows, plus the bound rows of every absorbed column.
    in_T = np.zeros(n, dtype=bool)
    in_T[T] = True
    removed = np.zeros(m, dtype=bool)
    removed[np.asarray(rows, dtype=np.int64)] = True
    if bnd_cols.size:
        removed[bnd_rows[in_T[bnd_cols]]] = True
    keep = np.flatnonzero(~removed)
    fully_absorbed = keep.size == 0
    if fully_absorbed:
        first = np.searchsorted(bnd_cols, int(T[0]))                     # one redundant bound row, as in plan_block
        keep = np.array([bnd_rows[first]], dtype=int)
    rest = np.flatnonzero(~in_T)
    return {"kind": "product", "rule": rule or "width", "T": T, "bid": bid, "a": a, "lo": lo[T], "hi": hi[T],
            "b_lo": l[np.array(rows)], "b_hi": u[np.array(rows)], "K": len(rows), "rows": np.array(rows),
            "keep": keep, "perm": np.concatenate([T, rest]), "n_x": int(T.size), "cover": float(T.size / n),
            "fully_absorbed": bool(fully_absorbed)}, "RELOCATABLE_PRODUCT"


def product_project(v, lo, hi, a, bid, b_lo, b_hi, K, iters=60, expand=40, polish=2):
    """Exact projection onto the product over blocks k of {lo <= x <= hi, a_k'x in [b_lo_k, b_hi_k]}, all blocks at once.

    Per block, phi_k(mu) = a_k'clip(v - mu a_k, lo, hi) is nonincreasing in mu for coefficients of ANY sign (its slope
    is -sum over free coordinates of a^2). The target is the band point nearest a_k'clip(v), so mu = 0 is the answer
    when that is already inside the band. Bracket: the finite breakpoints, widened by doubling until the signs are
    right (unbounded sides make phi linear beyond the last breakpoint); then bisection; then safeguarded Newton steps
    that land exactly on the root's linear piece.
    """
    seg = lambda z: jax.ops.segment_sum(z, bid, num_segments=K)
    target = jnp.clip(seg(a * jnp.clip(v, lo, hi)), b_lo, b_hi)

    def phi(mu):
        return seg(a * jnp.clip(v - mu[bid] * a, lo, hi)) - target

    bp1 = jnp.where(jnp.isfinite(lo), (v - jnp.where(jnp.isfinite(lo), lo, 0.0)) / a, jnp.nan)
    bp2 = jnp.where(jnp.isfinite(hi), (v - jnp.where(jnp.isfinite(hi), hi, 0.0)) / a, jnp.nan)
    mins = jnp.minimum(jnp.where(jnp.isnan(bp1), jnp.inf, bp1), jnp.where(jnp.isnan(bp2), jnp.inf, bp2))
    maxs = jnp.maximum(jnp.where(jnp.isnan(bp1), -jnp.inf, bp1), jnp.where(jnp.isnan(bp2), -jnp.inf, bp2))
    mlo = jax.ops.segment_min(mins, bid, num_segments=K)
    mhi = jax.ops.segment_max(maxs, bid, num_segments=K)
    width = 1.0 + (mhi - mlo)
    l0, h0 = mlo - width, mhi + width

    def widen(_, st):
        l_, h_ = st
        w = h_ - l_
        return jnp.where(phi(l_) < 0, l_ - w, l_), jnp.where(phi(h_) > 0, h_ + w, h_)

    l_, h_ = jax.lax.fori_loop(0, expand, widen, (l0, h0))

    def bisect(_, st):
        l_, h_ = st
        mid = 0.5 * (l_ + h_)
        pos = phi(mid) > 0
        return jnp.where(pos, mid, l_), jnp.where(pos, h_, mid)

    l_, h_ = jax.lax.fori_loop(0, iters, bisect, (l_, h_))
    mu = 0.5 * (l_ + h_)
    a2 = a * a

    def newton(_, mu):
        x = jnp.clip(v - mu[bid] * a, lo, hi)
        r = seg(a * x) - target
        slope = seg(jnp.where((x > lo) & (x < hi), a2, 0.0))
        mu_n = mu + r / jnp.where(slope > 0, slope, 1.0)
        ok = (slope > 0) & (mu_n >= l_) & (mu_n <= h_)
        return jnp.where(ok, mu_n, mu)

    mu = jax.lax.fori_loop(0, polish, newton, mu)
    return jnp.clip(v - mu[bid] * a, lo, hi)


@dataclass(frozen=True)
class ProductProj:
    """Indicator of a product of box-hyperplane/box-slab sets on the leading (canonicalized) block."""

    lo: object
    hi: object
    a: object
    bid: object
    b_lo: object
    b_hi: object
    n_x: int
    K: int
    impl: str = "bisect"
    iters: int = 60
    props: PropSet = field(default=PropSet(averagedness=0.5, prox_friendly=True))

    def prox(self, v, step=None):
        x, y = v[: self.n_x], v[self.n_x:]
        if self.impl.startswith("file:"):
            p = load_operator(self.impl[5:])(x, self.lo, self.hi, self.a, self.bid, self.b_lo, self.b_hi, self.K)
        else:
            p = product_project(x, self.lo, self.hi, self.a, self.bid, self.b_lo, self.b_hi, self.K, iters=self.iters)
        return jnp.concatenate([p, y])


jax.tree_util.register_dataclass(
    ProductProj, data_fields=["lo", "hi", "a", "bid", "b_lo", "b_hi"], meta_fields=["n_x", "K", "impl", "iters", "props"])


# ----------------------------------------------------------------------------------------------------------------
# Formulations
# ----------------------------------------------------------------------------------------------------------------
@dataclass
class Formulation:
    name: str
    prob: object
    decode: object          # z (numpy) -> x in ORIGINAL coordinates
    L_csc: object           # the linear operator the iteration applies, in its own coordinates
    col_map: object         # own-coordinate column index -> original column index
    build_s: float
    absorbed: int


def build_formulation(d, cand, plan=None, impl="bisect", iters=60, k=10, pad=0):
    """LPs (P = 0) go through lp_families.lp_formulation: qp_cell.to_composite rejects an empty P, and a zero
    Lipschitz constant would make the primal step t = 1/L_f unbounded; the LP path declares L_f = omega * ||L||."""
    if sp.csc_matrix(d["P"]).nnz == 0:
        import lp_families
        return lp_families.lp_formulation(d, cand, plan=plan, impl=impl, iters=iters, k=k, pad=pad)
    return _build_formulation(d, cand, plan=plan, impl=impl, iters=iters, k=k, pad=pad)


def _build_formulation(d, cand, plan=None, impl="bisect", iters=60, k=10, pad=0):
    t0 = time.perf_counter()
    n = d["P"].shape[0]
    if cand == "std_raw":
        prob = qp_cell.to_composite(d, power_iters=POWER_ITERS)
        return Formulation(cand, prob, lambda z: z, sp.csc_matrix(d["A"]), np.arange(n),
                           time.perf_counter() - t0, 0)
    if cand == "std_ruiz":
        base = qp_cell.to_composite(d, power_iters=POWER_ITERS)
        resc, d1, d2, _ = rescale_qp_problem(base, "ruiz", iters=10, power_iters=POWER_ITERS)
        d2 = np.asarray(d2)
        return Formulation(cand, resc, lambda z, d2=d2: d2 * z, sp.csc_matrix(resc.data["A"]), np.arange(n),
                           time.perf_counter() - t0, 0)
    if cand not in ("reloc", "reloc_rows"):
        raise ValueError(cand)
    perm = plan["perm"]
    P = sp.csc_matrix(d["P"])[perm][:, perm]
    q = np.asarray(d["q"])[perm]
    A_keep = sp.csc_matrix(sp.csc_matrix(d["A"])[plan["keep"]][:, perm])
    lk, uk = np.asarray(d["l"])[plan["keep"]], np.asarray(d["u"])[plan["keep"]]
    if cand == "reloc_rows":
        # ROW-ONLY equilibration of what stays in L: scaling a row and its bounds by r > 0 leaves the feasible set, the
        # x-coordinates and the absorbed projection's geometry unchanged (column scaling would make it ellipsoidal).
        r = np.ones(A_keep.shape[0])
        M = sp.csr_matrix(A_keep)
        for _ in range(10):
            rn = np.sqrt(np.maximum(np.asarray(abs(sp.diags(r) @ M).max(axis=1).todense()).ravel(), 1e-300))
            r = r / rn
        A_keep = sp.csc_matrix(sp.diags(r) @ M)
        lk, uk = lk * r, uk * r
    L = SparseLinOp.from_scipy(A_keep, power_iters=POWER_ITERS)
    P_op = SparseLinOp.from_scipy(P, power_iters=POWER_ITERS)
    Lf = max(float(P_op.norm_bound), 1e-12)
    f = QuadraticForm(P=P_op, q=jnp.asarray(q), r=d["r"],
                      props=PropSet(lipschitz=Lf, cocoercivity=1.0 / Lf, prox_friendly=False,
                                    strong_convexity=0.0))
    kind = plan["kind"]
    if impl == "sort" and kind != "simplex":
        raise ValueError("sort impl is simplex-only")
    if kind == "product":
        g = ProductProj(lo=jnp.asarray(plan["lo"]), hi=jnp.asarray(plan["hi"]), a=jnp.asarray(plan["a"]),
                        bid=jnp.asarray(plan["bid"]), b_lo=jnp.asarray(plan["b_lo"]), b_hi=jnp.asarray(plan["b_hi"]),
                        n_x=plan["n_x"], K=plan["K"], impl=impl, iters=iters)
    else:
        g = BlockProj(lo=jnp.asarray(plan["lo"]), hi=jnp.asarray(plan["hi"]), a=jnp.asarray(plan["a"]),
                      b_lo=jnp.asarray(plan["b_lo"]), b_hi=jnp.asarray(plan["b_hi"]), n_x=plan["n_x"],
                      kind=kind, impl=impl, iters=iters, k=k, pad=pad)
    h = Box(lo=jnp.asarray(lk), hi=jnp.asarray(uk))
    prob = CompositeProblem(f=f, g=g, h=h, L=L,
                            data={"P": P, "q": q, "r": d["r"], "A": A_keep, "l": lk, "u": uk},
                            shape=(n,), name=f"reloc_{kind}_{impl}")

    def decode(z, perm=perm, n=n):
        x = np.empty(n)
        x[perm] = z
        return x

    return Formulation(("reloc_rows" if cand == "reloc_rows" else "reloc") + ("_prod" if kind == "product" else "")
                       + f":{impl}" + (f"{k}" if impl == "dykstra" else "") + (f"+pad{pad}" if pad else ""),
                       prob, decode, A_keep, perm, time.perf_counter() - t0, plan["n_x"])


# ----------------------------------------------------------------------------------------------------------------
# Reference, acceptance, measurement
# ----------------------------------------------------------------------------------------------------------------
def reference(d, timeout_s=1800):
    rb = run_baseline_proc(d, "clarabel", 1e-9, timeout_s=timeout_s, scratch=os.environ.get("OPTEVOLVE_SCRATCH"))
    if "error" in rb:
        return None, rb["error"]
    x = rb["x"]
    return float(0.5 * x @ (sp.csc_matrix(d["P"]) @ x) + np.asarray(d["q"]) @ x), rb.get("t_s")


def accept(x, d, obj_ref, tol):
    j = qp_accuracy_judge(x, d["P"], d["q"], d["r"], d["A"], d["l"], d["u"], ref_obj=None, feas_tol=tol)
    obj = float(0.5 * x @ (sp.csc_matrix(d["P"]) @ x) + np.asarray(d["q"]) @ x)
    # relative objective error with the standard 1 + |ref| denominator: dividing by |ref| alone made acceptance
    # arbitrarily harsh near a zero optimum (QPCBLEND, obj_ref 0.0078, rejected every baseline solver)
    gap = abs(obj - obj_ref) / (1.0 + abs(obj_ref))
    return (j["row_violation"] <= tol and gap <= tol), float(j["row_violation"]), float(gap)


def measure(form, d, obj_ref, tol=1e-6, cap=100000, K=50, probe_iters=200):
    """Iterations to original-problem acceptance, plus a probe snapshot for features. N is resolved to K."""
    prob = form.prob
    Lf, nb = float(prob.f.props.lipschitz), float(prob.L.norm_bound)
    t = 1.0 / Lf
    s = 0.49 / (t * nb ** 2)
    t0 = time.perf_counter()
    built = build_scheme(prob, "condat_vu", CondatVuParams(t=t, s=s), check=True)
    T, dyn = built.T, jax.tree_util.tree_map(jnp.asarray, built.dyn)
    v = jax.tree_util.tree_map(jnp.asarray, built.state0)

    def chunk(state, dyn_):
        return jax.lax.fori_loop(0, K, lambda i, z: T(z, i, dyn_), state)

    exe = jax.jit(chunk)
    v = exe(v, dyn)                                              # compile + first chunk
    jax.block_until_ready(v)
    setup_s = time.perf_counter() - t0 + form.build_s
    it = K
    iter_time = 0.0
    hist = []                                                    # (iter, fixed-point residual of the primal)
    probe = None
    prev = np.asarray(v[0])
    N, last_rv, last_gap = None, None, None
    while True:
        x = form.decode(np.asarray(v[0]))
        if not np.all(np.isfinite(x)):
            break
        ok, last_rv, last_gap = accept(x, d, obj_ref, tol)
        if probe is None and it >= probe_iters:
            probe = {"iter": it, "z": np.asarray(v[0]).copy(), "x": x.copy()}
        if ok:
            N = it
            break
        if it >= cap:
            break
        c0 = time.perf_counter()
        v = exe(v, dyn)
        jax.block_until_ready(v)
        iter_time += time.perf_counter() - c0
        it += K
        cur = np.asarray(v[0])
        hist.append((it, float(np.linalg.norm(cur - prev))))
        prev = cur
    if probe is None:
        probe = {"iter": it, "z": np.asarray(v[0]).copy(), "x": form.decode(np.asarray(v[0]))}
    ran = max(it - K, 1)
    return {"N": N, "censored": N is None, "iters_run": it, "setup_s": setup_s,
            "t_iter": iter_time / ran if ran > 0 else None, "row_violation": last_rv, "obj_gap": last_gap,
            "normL": nb, "Lf": Lf, "hist": hist, "probe": probe}


LADDER = (1e-2, 1e-3, 1e-4, 1e-6)


def measure_ladder(form, d, obj_ref, tols=LADDER, cap=100000, K=50, deadline=None):
    """Iterations to acceptance at EVERY rung of an accuracy ladder in one run (resolution K).

    Acceptance is unchanged in kind: per-row relative feasibility max_i viol_i/(1+|bound_i|) and relative objective
    error, both <= tol. Globally normalized KKT residuals are NOT used: they hide violations on badly scaled
    instances (BOYD2, PRIMALC2/8 in this project's ledger). The run stops when the tightest rung is met or at cap.
    """
    prob = form.prob
    Lf, nb = float(prob.f.props.lipschitz), float(prob.L.norm_bound)
    t = 1.0 / Lf
    s = 0.49 / (t * nb ** 2)
    t0 = time.perf_counter()
    built = build_scheme(prob, "condat_vu", CondatVuParams(t=t, s=s), check=True)
    T, dyn = built.T, jax.tree_util.tree_map(jnp.asarray, built.dyn)
    v = jax.tree_util.tree_map(jnp.asarray, built.state0)
    # CHUNK SIZE. The deadline can only be tested between chunks, so the first chunk is deliberately small: on a heavy
    # instance (thousands of cone blocks) one chunk of 50 iterations ran tens of seconds past a 20 s budget. K then
    # adapts towards a chunk of about a fifth of a second, recompiling at most a few times.
    K_req, K = K, (min(K, 5) if deadline is not None else K)
    make_exe = lambda k: jax.jit(lambda state, dyn_: jax.lax.fori_loop(0, k, lambda i, z: T(z, i, dyn_), state))
    exe = make_exe(K)
    v = exe(v, dyn)
    jax.block_until_ready(v)
    setup_s = time.perf_counter() - t0 + form.build_s
    it, iter_time = K, 0.0
    recompiles = 0
    hit = {tol: None for tol in tols}
    rv = gap = None
    chunk_s = None
    while True:
        # BOUND THE OVERSHOOT. The deadline is only testable between chunks, so on a heavy instance one chunk of K
        # iterations can run far past it (a 20 s budget reported 82 s). After the first chunk, shrink K so that a
        # chunk costs at most a quarter of the remaining time; the recompile is paid once and only when it matters.
        if deadline is not None and chunk_s is not None and recompiles < 3:
            left = max(deadline - time.perf_counter(), 1e-9)
            target = min(0.2, 0.25 * left)
            K_new = int(np.clip(K * target / max(chunk_s, 1e-9), 1, K_req))
            if K_new != K and (K_new < K * 0.6 or K_new > K * 2):
                K, recompiles = K_new, recompiles + 1
                exe = make_exe(K)
                chunk_s = None
        x = form.decode(np.asarray(v[0]))
        if not np.all(np.isfinite(x)):
            break
        _, rv, gap = accept(x, d, obj_ref, max(tols))
        for tol in tols:
            if hit[tol] is None and rv <= tol and gap <= tol:
                hit[tol] = it
        if hit[min(tols)] is not None or it >= cap or (deadline is not None and time.perf_counter() >= deadline):
            break
        c0 = time.perf_counter()
        v = exe(v, dyn)
        jax.block_until_ready(v)
        chunk_s = time.perf_counter() - c0
        iter_time += chunk_s
        it += K
    ran = max(it - K, 1)
    return {"N_at": {f"{tol:g}": hit[tol] for tol in tols}, "iters_run": it, "setup_s": setup_s,
            "t_iter": iter_time / ran, "row_violation": rv, "obj_gap": gap, "normL": nb, "Lf": Lf}


def time_clean(form, iters, K=50, pad_label="", blocks=15, max_s=0.35, min_block_s=2e-3):
    """Per-iteration cost of `form` with no host checks: the MEDIAN of `blocks` short timed blocks.

    One timed block is not a trustworthy measurement on a machine that shares its GPU with anything else. The same
    32-variable QP read 28.8 us and 1484 us per iteration in the same regime, a spread of 50x, because a co-resident
    inference server took the device for part of one block; the GPU sits at its clocked P-state throughout, so this
    is burst contention and not a clock ramp. A median needs more than half the blocks to be hit before it moves,
    so it survives that where a single block does not.

    The cost is bounded on both axes and is charged to the caller exactly as the single block was. A sizing chunk
    fixes the block: enough chunks that a block lasts at least min_block_s, so the one host synchronization a block
    costs is a negligible share of it, and few enough that all `blocks` blocks together fit in max_s. When a single
    chunk of K already exceeds max_s / blocks the chunk itself is shrunk ONCE (one extra compile); that only happens
    when one iteration is expensive enough for the per-launch overhead to be negligible anyway, and in that regime
    the whole calibration is CHEAPER than the old single block of `iters` iterations, not dearer.

    Returns per_iter_s (the median) together with the spread it survived, so the noise stays visible in the record.
    """
    prob = form.prob
    Lf, nb = float(prob.f.props.lipschitz), float(prob.L.norm_bound)
    t = 1.0 / Lf
    s = 0.49 / (t * nb ** 2)
    c0 = time.perf_counter()
    built = build_scheme(prob, "condat_vu", CondatVuParams(t=t, s=s), check=True)
    T, dyn = built.T, jax.tree_util.tree_map(jnp.asarray, built.dyn)
    v = jax.tree_util.tree_map(jnp.asarray, built.state0)
    mk = lambda k: jax.jit(lambda state, dyn_: jax.lax.fori_loop(0, k, lambda i, z: T(z, i, dyn_), state))
    exe = mk(K)
    v = exe(v, dyn)
    jax.block_until_ready(v)
    setup_s = time.perf_counter() - c0 + form.build_s      # build + trace + compile + first chunk, as before
    blocks = max(15, int(blocks))
    c0 = time.perf_counter()                               # SIZING CHUNK: one chunk, to fix the block
    v = exe(v, dyn)
    jax.block_until_ready(v)
    one = max(time.perf_counter() - c0, 1e-9)
    K_cal, cpb = K, 1
    if blocks * one > max_s:
        K_cal = int(np.clip(int(K * max_s / (blocks * one)), 4, K))
        if K_cal < K:
            exe = mk(K_cal)
            v = exe(v, dyn)                                # compile the shrunk chunk; not timed as a block
            jax.block_until_ready(v)
    else:
        room = max(1, int(max_s / (blocks * one)))
        cpb = int(np.clip(int(np.ceil(min_block_s / one)), 1, min(room, 400)))
    rates = []
    for _ in range(blocks):
        c0 = time.perf_counter()
        for _ in range(cpb):
            v = exe(v, dyn)
        jax.block_until_ready(v)
        rates.append((time.perf_counter() - c0) / (cpb * K_cal))
    per_iter = float(np.median(rates))
    return {"setup_s": setup_s, "per_iter_s": per_iter, "total_s": setup_s + per_iter * iters,
            "per_iter_min_s": float(min(rates)), "per_iter_max_s": float(max(rates)),
            "per_iter_first_s": rates[0], "cal_blocks": len(rates), "cal_K": K_cal, "cal_chunks_per_block": cpb,
            "cal_iters": len(rates) * cpb * K_cal,
            "cal_spread": float(max(rates) / max(min(rates), 1e-30))}


# ----------------------------------------------------------------------------------------------------------------
# Features: static (no solve) and probe (charged: probe_iters iterations of the candidate itself)
# ----------------------------------------------------------------------------------------------------------------
def _spec_norm(M):
    M = sp.csc_matrix(M)
    if M.shape[0] == 0 or M.shape[1] == 0 or M.nnz == 0:
        return 0.0
    if min(M.shape) <= 2:
        return float(np.linalg.norm(M.toarray(), 2))
    return float(svds(M.astype(np.float64), k=1, return_singular_vectors=False)[0])


def features(form, d, plan, meas, probe_iters=200):
    L = sp.csc_matrix(form.L_csc)
    n = d["P"].shape[0]
    Pdiag = sp.csc_matrix(d["P"]).diagonal()
    Lp = np.max(np.abs(Pdiag)) if Pdiag.size else 1.0
    feat = {"n_var": n, "log_n": float(np.log(n)), "rows_L": L.shape[0], "rank_ratio": L.shape[0] / n,
            "nnz_ratio": L.nnz / n, "absorbed_frac": form.absorbed / n, "log_normL": float(np.log(meas["normL"])),
            "muL_P": float(np.min(Pdiag) / max(Lp, 1e-300))}
    try:
        S, U = top_spectrum(L)
        feat["log_gap_L"] = float(np.log(S[0] / max(S[1], 1e-30)))
        feat["rowmass_L"] = float(np.max(U[:, 0] ** 2))
    except Exception:
        feat["log_gap_L"], feat["rowmass_L"] = 0.0, 0.0
    # probe features, from the candidate's own iterate after probe_iters iterations
    x = meas["probe"]["x"]
    T = plan["T"]
    xT = x[T]
    scale = max(1.0, float(np.max(np.abs(xT))))
    tolb = 1e-7 * scale
    free_T = (xT > plan["lo"] + tolb) & (xT < plan["hi"] - tolb)
    feat["free_frac_T"] = float(free_T.mean())
    rest = np.array(sorted(set(range(n)) - set(T.tolist())), dtype=int)
    free_orig = np.concatenate([T[free_T], rest])
    cd = Pdiag[free_orig] if free_orig.size else np.array([0.0])
    feat["curv_free_min"] = float(np.min(cd) / max(Lp, 1e-300))
    feat["curv_free_mean"] = float(np.mean(cd) / max(Lp, 1e-300))
    inv = np.empty(n, dtype=int)
    inv[form.col_map] = np.arange(n)
    own_cols = inv[free_orig]
    try:
        feat["log_coupling_free"] = float(np.log(max(_spec_norm(L[:, own_cols]), 1e-30)))
    except Exception:
        feat["log_coupling_free"] = float(np.log(max(meas["normL"], 1e-30)))
    h = [r for (i, r) in meas["hist"] if i <= max(probe_iters, 2 * 50) and r > 0]
    if len(h) >= 3:
        k = np.arange(len(h))
        feat["decay_rate"] = float(np.polyfit(k, np.log(h), 1)[0])
    else:
        feat["decay_rate"] = 0.0
    feat["log_free_dim"] = float(np.log(max(free_orig.size, 1)))
    return feat
