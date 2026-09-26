"""Structure probe: problem structure vs recipe admissibility and cost.

WHAT THIS ANSWERS
-----------------
The construction gate for PDHG is  t*s*||L||^2 < 1  (typecheck_pdhg), and for
Condat-Vu  t*L_f/2 + t*s*||L||^2 < 1  (typecheck_condat_vu).  Both constrain
only the PRODUCT  theta := t*s*||L||^2 .  The RATIO  r := t/s  is completely
unconstrained by the gate.  So the gate can rank a recipe as legal/illegal but
it cannot, by itself, rank two legal recipes.

This harness sweeps a grid of (t, theta) pairs.  For a fixed theta the step
sizes are set as

    s = theta / (t * ||L||^2)        ==>   t*s*||L||^2 = theta  EXACTLY

so every column of the grid has the identical gate value and the identical
gate verdict, while t (equivalently r = t/s = t^2 ||L||^2 / theta) sweeps
across two decades.  If the product alone predicted performance, N would be
constant along a column.  It is not (measured: up to 21x on netflow grid).

Alongside that it measures, per instance:
  * ||L||_2 as the solver sees it:  L.norm_bound  (power iteration x 1.02
    safety for the netflow/dense LinOps; the analytic sqrt(8) for Grad2D).
  * the TRUE extreme singular values of the constraint operator on its
    NONZERO spectrum, and

        kappa_plus := sigma_max(L) / sigma_min_plus(L)

    where sigma_min_plus is the smallest singular value on the range, i.e.
    the rank-th largest.  The rank is taken STRUCTURALLY, never by a
    numerical cutoff:
      - node-arc incidence A (netflow grid / regular):  A A^T = 2 L_graph and
        null(A^T) = span{component indicators}, so rank(A) = n_nodes - #comp
        with #comp from scipy.sparse.csgraph.connected_components.
      - Grad2D (TV):  null(D) = constants, so rank(D) = H*W - 1.
      - dense random A (m < n):  generically full row rank, rank = m.
    A numerical rank at a relative cutoff is ALSO reported, purely as a
    cross-check on the structural one.
  * cheap "free" features: equilibration headroom (row/col/entry infinity-norm
    dispersion), the PDLP primal weight ||c||/||b||, degree/nnz statistics.

Everything is recorded to one JSON per family under
    results/structure_20260909/<family>_<device>.json

KAPPA COMPUTATION LADDER (first feasible wins; the rest are still recorded
when they are cheap enough, together with their pairwise agreement):
  1. dense_svd        numpy.linalg.svd on the materialized (n_out, n_in) L.
  2. gram_eigvalsh    numpy.linalg.eigvalsh on the smaller Gram (exact, same
                      answer, O(min-side^3) instead of O(n_out * n_in^2)).
  3. shifted_eigsh    scipy.sparse.linalg.eigsh on the MATRIX-FREE operator
                      M = A A^T + c N N^T, c = ||A||_1 ||A||_inf >= lambda_max
                      by Holder, with N an orthonormal basis of the known null
                      space.  The shift moves the null directions to the TOP of
                      the spectrum, so lambda_min(M) = sigma_min_plus^2 exactly
                      and no deflation/reorthogonalisation is needed.
                      (Projecting only after each matvec is NOT sufficient --
                      the null direction leaks back and sigma_min_plus reads 0.)
  4. closed_form      grid: kappa_plus = (2*sqrt(2)/pi)*k;
                      TV Neumann forward difference: the exact eigenvalues of
                      D^T D are 4 sin^2(pi i/(2H)) + 4 sin^2(pi j/(2W)).
  5. "skipped"        recorded verbatim when every route is over budget.

WHAT IS NOT DONE HERE: nothing in this file feeds a certificate or a
typecheck.  L.norm_bound (an upper bound) remains the gate input; the measured
sigma_max is prediction-only.  Termination stays with
atlas.backend.precision's float64 original-coordinate certificate.

USAGE
-----
  python scripts/structure_probe.py --family grid    --sizes 16,32
  python scripts/structure_probe.py --family regular --sizes 1024 --degrees 3,4
  python scripts/structure_probe.py --family tv      --sizes 32,64
  python scripts/structure_probe.py --report results/structure_20260909/grid_4090.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from pathlib import Path

REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as csgraph
import scipy.sparse.linalg as spla

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

import atlas.backend.precision as P
from atlas.backend.hardware import prepare_base
from atlas.bench.lp_cell import IncidenceLinOp, MatrixLinOp, build_instances
from atlas.genome import Genome, GenomeParams
from atlas.operators.linear import Grad2D
from atlas.typing_ import TypeCheckError

DEFAULT_OUT_DIR = os.path.join(REPO_ROOT, "results", "structure_20260909")
LP_FAMILIES = ("grid", "regular", "dense")
ALL_FAMILIES = LP_FAMILIES + ("tv",)


# ---------------------------------------------------------------------------
# Operator materialization (sparse / dense), all optional and budget-capped
# ---------------------------------------------------------------------------


def sparse_operator(problem):
    """scipy CSR of L, or None when L has no cheap sparse form.

    Never touches problem.data["A_dense"]: the incidence matrix is rebuilt
    from the arc list, which is the matrix-free path the features are
    supposed to live on.
    """
    L = problem.L
    if isinstance(L, IncidenceLinOp):
        tail = np.asarray(L.tail, dtype=np.int64)
        head = np.asarray(L.head, dtype=np.int64)
        n_arcs = tail.shape[0]
        cols = np.arange(n_arcs, dtype=np.int64)
        rows = np.concatenate([tail, head])
        cols2 = np.concatenate([cols, cols])
        vals = np.concatenate([np.ones(n_arcs), -np.ones(n_arcs)])
        A = sp.coo_matrix((vals, (rows, cols2)), shape=(L.n_nodes, n_arcs))
        return A.tocsr()
    if isinstance(L, MatrixLinOp):
        return sp.csr_matrix(np.asarray(L.A, dtype=np.float64))
    if isinstance(L, Grad2D):
        H, W = L.shape
        n = H * W
        idx = np.arange(n).reshape(H, W)
        rows, cols, vals = [], [], []
        # channel 0: u[0,i,j] = x[i+1,j] - x[i,j] for i < H-1
        r0 = idx[:-1, :].ravel()
        rows += [r0, r0]
        cols += [idx[1:, :].ravel(), idx[:-1, :].ravel()]
        vals += [np.ones(r0.size), -np.ones(r0.size)]
        # channel 1 lives in output rows n .. 2n-1
        r1 = n + idx[:, :-1].ravel()
        rows += [r1, r1]
        cols += [idx[:, 1:].ravel(), idx[:, :-1].ravel()]
        vals += [np.ones(r1.size), -np.ones(r1.size)]
        A = sp.coo_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(2 * n, n),
        )
        return A.tocsr()
    return None


def structural_rank(problem):
    """(rank, null_dim_of_the_smaller_Gram, provenance) from STRUCTURE only."""
    L = problem.L
    if isinstance(L, IncidenceLinOp):
        tail = np.asarray(L.tail, dtype=np.int64)
        head = np.asarray(L.head, dtype=np.int64)
        n = int(L.n_nodes)
        g = sp.coo_matrix(
            (np.ones(tail.size), (tail, head)), shape=(n, n)
        ).tocsr()
        n_comp, _ = csgraph.connected_components(g, directed=False)
        return n - n_comp, n_comp, f"incidence: rank = n_nodes - #components = {n} - {n_comp}"
    if isinstance(L, Grad2D):
        H, W = L.shape
        return H * W - 1, 1, "Grad2D Neumann: null(D) = constants, rank = H*W - 1"
    if isinstance(L, MatrixLinOp):
        m, n = np.asarray(L.A).shape
        return min(m, n), 0, f"dense gaussian A ({m}x{n}): generically full rank"
    return None, None, "unknown operator type"


# ---------------------------------------------------------------------------
# Spectrum routes
# ---------------------------------------------------------------------------


def _svals_from_gram(G, rank, tol_rel=1e-10):
    """(sigma_max, sigma_min_plus, numerical_rank) from a dense Gram matrix."""
    lam = np.linalg.eigvalsh(np.asarray(G, dtype=np.float64))
    lam = np.clip(lam, 0.0, None)          # symmetric PSD; kill -1e-16 noise
    lam_sorted = np.sort(lam)              # ascending
    d = lam_sorted.size
    sigma_max = float(math.sqrt(lam_sorted[-1]))
    num_rank = int(np.sum(lam_sorted > tol_rel * lam_sorted[-1]))
    if rank is None or rank <= 0 or rank > d:
        return sigma_max, None, num_rank
    sigma_min_plus = float(math.sqrt(lam_sorted[d - rank]))
    return sigma_max, sigma_min_plus, num_rank


def kappa_dense_svd(A_sparse, rank, max_elems):
    """numpy.linalg.svd on the materialized dense operator."""
    if A_sparse is None:
        return None
    m, n = A_sparse.shape
    if m * n > max_elems:
        return None
    t0 = time.perf_counter()
    sv = np.linalg.svd(np.asarray(A_sparse.toarray(), dtype=np.float64),
                       compute_uv=False)
    sv = np.sort(sv)[::-1]
    smax = float(sv[0])
    num_rank = int(np.sum(sv > 1e-10 * smax))
    smin = float(sv[rank - 1]) if (rank and 0 < rank <= sv.size) else None
    return {
        "method": "dense_svd",
        "sigma_max": smax,
        "sigma_min_plus": smin,
        "kappa_plus": (smax / smin) if smin else None,
        "numerical_rank": num_rank,
        "seconds": time.perf_counter() - t0,
        "dense_elems": int(m * n),
    }


def kappa_gram_eigvalsh(A_sparse, rank, max_dim):
    """Exact ends via eigvalsh on the SMALLER Gram (A A^T or A^T A)."""
    if A_sparse is None:
        return None
    m, n = A_sparse.shape
    d = min(m, n)
    if d > max_dim:
        return None
    t0 = time.perf_counter()
    if m <= n:
        G = (A_sparse @ A_sparse.T).toarray()
        gram_null = d - (rank if rank else d)
    else:
        G = (A_sparse.T @ A_sparse).toarray()
        gram_null = d - (rank if rank else d)
    smax, smin, num_rank = _svals_from_gram(G, rank)
    return {
        "method": "gram_eigvalsh",
        "gram_dim": int(d),
        "gram_null_dim": int(gram_null),
        "sigma_max": smax,
        "sigma_min_plus": smin,
        "kappa_plus": (smax / smin) if smin else None,
        "numerical_rank": num_rank,
        "seconds": time.perf_counter() - t0,
    }


def _null_basis(problem, d, side):
    """Orthonormal basis of null(G) for the smaller Gram G, or None.

    side == "rows"  -> G = A A^T  (netflow, dense LP)
    side == "cols"  -> G = A^T A  (Grad2D)
    """
    L = problem.L
    if side == "rows" and isinstance(L, IncidenceLinOp):
        tail = np.asarray(L.tail, dtype=np.int64)
        head = np.asarray(L.head, dtype=np.int64)
        n = int(L.n_nodes)
        g = sp.coo_matrix((np.ones(tail.size), (tail, head)), shape=(n, n)).tocsr()
        n_comp, labels = csgraph.connected_components(g, directed=False)
        N = np.zeros((n, n_comp))
        N[np.arange(n), labels] = 1.0
        N /= np.linalg.norm(N, axis=0, keepdims=True)
        return N
    if side == "cols" and isinstance(L, Grad2D):
        N = np.ones((d, 1)) / math.sqrt(d)
        return N
    if side == "rows" and isinstance(L, MatrixLinOp):
        return np.zeros((d, 0))
    return None


def kappa_shifted_eigsh(problem, A_sparse, rank, max_nnz, k_extra=1,
                        tol=0.0, maxiter=None):
    """Matrix-free ends via the STRUCTURAL SHIFT M = G + c N N^T.

    c = ||A||_1 * ||A||_inf >= lambda_max(G) (Holder), so every null direction
    is pushed to the top of the spectrum and eigsh(which="SA") returns
    lambda_min(M) = sigma_min_plus(A)^2 with no deflation and no
    reorthogonalisation bookkeeping.
    """
    if A_sparse is None or A_sparse.nnz > max_nnz:
        return None
    m, n = A_sparse.shape
    side = "rows" if m <= n else "cols"
    d = min(m, n)
    N = _null_basis(problem, d, side)
    if N is None:
        return None
    absA = abs(A_sparse)
    c = float(absA.sum(axis=0).max()) * float(absA.sum(axis=1).max())
    t0 = time.perf_counter()
    if side == "rows":
        def mv(y):
            return A_sparse @ (A_sparse.T @ y)
    else:
        def mv(y):
            return A_sparse.T @ (A_sparse @ y)

    if N.shape[1] == 0:
        def Mv(y):
            return mv(y)
    else:
        def Mv(y):
            return mv(y) + c * (N @ (N.T @ y))

    op = spla.LinearOperator((d, d), matvec=Mv, rmatvec=Mv, dtype=np.float64)
    nev = min(k_extra, d - 1)
    lam_hi = spla.eigsh(op, k=nev, which="LA", return_eigenvectors=False,
                        tol=tol, maxiter=maxiter)
    lam_lo = spla.eigsh(op, k=nev, which="SA", return_eigenvectors=False,
                        tol=tol, maxiter=maxiter)
    top = float(np.max(lam_hi))
    # the shifted null directions sit at ~c + 0; the true lambda_max is the
    # largest eigenvalue that is NOT one of them.  With c >= lambda_max the
    # shifted block is >= c, so recover lambda_max from the un-shifted matvec
    # via a second solve restricted by deflation-free power on G itself.
    lam_max_G = float(np.max(spla.eigsh(
        spla.LinearOperator((d, d), matvec=mv, rmatvec=mv, dtype=np.float64),
        k=nev, which="LA", return_eigenvectors=False, tol=tol, maxiter=maxiter)))
    smax = math.sqrt(max(lam_max_G, 0.0))
    smin = math.sqrt(max(float(np.min(lam_lo)), 0.0))
    return {
        "method": "shifted_eigsh",
        "shift_c": c,
        "shifted_top": top,
        "null_dim": int(N.shape[1]),
        "sigma_max": smax,
        "sigma_min_plus": smin if smin > 0 else None,
        "kappa_plus": (smax / smin) if smin > 0 else None,
        "numerical_rank": None,
        "seconds": time.perf_counter() - t0,
    }


def kappa_closed_form(problem):
    """Exact/known closed forms, O(1)."""
    L = problem.L
    meta = problem.data.get("meta", {}) if isinstance(problem.data, dict) else {}
    fam = meta.get("family", "")
    if isinstance(L, Grad2D):
        H, W = L.shape
        i = np.arange(H)
        j = np.arange(W)
        e = (4 * np.sin(np.pi * i / (2 * H)) ** 2)[:, None] + \
            (4 * np.sin(np.pi * j / (2 * W)) ** 2)[None, :]
        e = np.sort(e.ravel())
        smax = float(math.sqrt(e[-1]))
        smin = float(math.sqrt(e[1]))
        return {"method": "closed_form",
                "expr": "eig(D^T D) = 4sin^2(pi i/2H) + 4sin^2(pi j/2W)",
                "sigma_max": smax, "sigma_min_plus": smin,
                "kappa_plus": smax / smin, "seconds": 0.0}
    if fam == "netflow_grid":
        k = int(meta["k"])
        return {"method": "closed_form",
                "expr": "kappa_plus = (2*sqrt(2)/pi) * k  (asymptotic)",
                "sigma_max": None, "sigma_min_plus": None,
                "kappa_plus": (2.0 * math.sqrt(2.0) / math.pi) * k,
                "seconds": 0.0}
    if fam == "netflow_regular":
        d = int(meta["degree"])
        if d >= 3:
            denom = d - 2.0 * math.sqrt(d - 1.0)
            if denom > 0:
                return {"method": "closed_form",
                        "expr": "kappa_plus <= sqrt(2d/(d-2sqrt(d-1)))  (Alon-Boppana bound)",
                        "sigma_max": None, "sigma_min_plus": None,
                        "kappa_plus_upper_bound": math.sqrt(2.0 * d / denom),
                        "kappa_plus": None, "seconds": 0.0}
    return None


# ---------------------------------------------------------------------------
# Tier-0 free features
# ---------------------------------------------------------------------------


def equilibration_headroom(A_sparse):
    """rho_row / rho_col / rho_entry: what Ruiz / Pock-Chambolle can equilibrate."""
    if A_sparse is None:
        return None
    A = A_sparse.tocsr()
    absA = abs(A)
    row_inf = np.asarray(absA.max(axis=1).todense()).ravel()
    col_inf = np.asarray(absA.max(axis=0).todense()).ravel()
    row_inf = row_inf[row_inf > 0]
    col_inf = col_inf[col_inf > 0]
    vals = np.abs(A.data)
    vals = vals[vals > 0]
    row_nnz = np.diff(A.indptr)
    return {
        "rho_row": float(row_inf.max() / row_inf.min()),
        "rho_col": float(col_inf.max() / col_inf.min()),
        "rho_entry": float(vals.max() / vals.min()),
        "nnz": int(A.nnz),
        "rows": int(A.shape[0]),
        "cols": int(A.shape[1]),
        "row_nnz_min": int(row_nnz.min()),
        "row_nnz_max": int(row_nnz.max()),
        "row_nnz_mean": float(row_nnz.mean()),
        "row_nnz_quantiles": np.quantile(row_nnz, [0.0, 0.5, 0.9, 1.0]).tolist(),
    }


def primal_weight(problem):
    """PDLP primal weight omega0 = ||c||_2 / ||b||_2 (LP only)."""
    d = problem.data
    if "c" not in d or "b" not in d:
        return None
    c = np.asarray(d["c"], dtype=np.float64)
    b = np.asarray(d["b"], dtype=np.float64)
    nb = float(np.linalg.norm(b))
    return {
        "c_norm2": float(np.linalg.norm(c)),
        "b_norm2": nb,
        "omega0": float(np.linalg.norm(c) / nb) if nb > 0 else None,
        "c_inf": float(np.abs(c).max()),
        "b_inf": float(np.abs(b).max()),
    }


def structure_features(problem, *, max_dense_elems, max_gram_dim, max_nnz,
                       iterative=True):
    """Every Tier-0/Tier-1 structure feature for one CompositeProblem."""
    t_all = time.perf_counter()
    feats = {
        "problem_name": problem.name,
        "shape": list(problem.shape),
        "meta": {k: (int(v) if isinstance(v, (int, np.integer)) else v)
                 for k, v in (problem.data.get("meta", {}) or {}).items()
                 if not isinstance(v, np.ndarray)},
        "norm_bound": float(problem.L.norm_bound),
        "norm_bound_source": (
            "power_iteration x1.02 safety"
            if isinstance(problem.L, (IncidenceLinOp, MatrixLinOp))
            else "analytic sqrt(8)" if isinstance(problem.L, Grad2D) else "unknown"
        ),
    }
    A_sparse = sparse_operator(problem)
    rank, gram_null, rank_src = structural_rank(problem)
    feats["structural_rank"] = rank
    feats["structural_gram_null_dim"] = gram_null
    feats["structural_rank_source"] = rank_src
    feats["equilibration"] = equilibration_headroom(A_sparse)
    feats["primal_weight"] = primal_weight(problem)

    routes = {}
    r = kappa_dense_svd(A_sparse, rank, max_dense_elems)
    if r:
        routes["dense_svd"] = r
    r = kappa_gram_eigvalsh(A_sparse, rank, max_gram_dim)
    if r:
        routes["gram_eigvalsh"] = r
    if iterative and not routes:
        try:
            r = kappa_shifted_eigsh(problem, A_sparse, rank, max_nnz)
            if r:
                routes["shifted_eigsh"] = r
        except Exception as e:                                  # noqa: BLE001
            routes["shifted_eigsh"] = {"method": "shifted_eigsh",
                                       "error": str(e)[:200]}
    r = kappa_closed_form(problem)
    if r:
        routes["closed_form"] = r
    feats["kappa_routes"] = routes

    order = ("dense_svd", "gram_eigvalsh", "shifted_eigsh", "closed_form")
    primary = None
    for name in order:
        cand = routes.get(name)
        if cand and cand.get("kappa_plus"):
            primary = name
            break
    if primary is None:
        feats["kappa_plus"] = None
        feats["sigma_max"] = None
        feats["sigma_min_plus"] = None
        feats["kappa_method"] = "skipped"
        feats["kappa_skip_reason"] = (
            f"dense elems > {max_dense_elems:g}, gram dim > {max_gram_dim}, "
            f"nnz > {max_nnz:g} or no closed form"
        )
    else:
        src = routes[primary]
        feats["kappa_plus"] = src["kappa_plus"]
        feats["sigma_max"] = src.get("sigma_max")
        feats["sigma_min_plus"] = src.get("sigma_min_plus")
        feats["kappa_method"] = primary
        feats["kappa_skip_reason"] = None
        # cross-route agreement (a silent deflation bug shows up here)
        agree = {}
        for name, other in routes.items():
            if name == primary or not other.get("kappa_plus"):
                continue
            agree[name] = abs(other["kappa_plus"] / src["kappa_plus"] - 1.0)
        feats["kappa_cross_route_rel_diff"] = agree
    # how much the 1.02-inflated gate norm overstates the true sigma_max
    if feats.get("sigma_max"):
        feats["norm_bound_over_sigma_max"] = (
            feats["norm_bound"] / feats["sigma_max"])
    feats["t_feature_s"] = time.perf_counter() - t_all
    return feats


# ---------------------------------------------------------------------------
# Instance construction
# ---------------------------------------------------------------------------


def build_cell(family, size, degree, n_instances, seed, tv_lam, tv_split,
               tv_crop):
    """List of CompositeProblems for one (family, size, degree) cell."""
    if family in LP_FAMILIES:
        train, _ = build_instances(n_instances, 0, family, size, seed=seed,
                                   degree=degree)
        return train
    if family == "tv":
        from atlas.bench.kermany import load_patches
        from atlas.ir.spec import tv_denoise_problem
        patches = load_patches(n_instances, size=size, split=tv_split,
                               seed=seed, crop=tv_crop)
        out = []
        for i, y in enumerate(patches):
            p = tv_denoise_problem(y, tv_lam)
            p.data["meta"] = {"family": "tv_denoise", "size": int(size),
                              "lam": float(tv_lam), "patch": int(i),
                              "seed": int(seed), "split": tv_split}
            out.append(p)
        return out
    raise ValueError(f"unknown family {family!r}; choose from {ALL_FAMILIES}")


def smooth_lipschitz(problem):
    f = problem.f
    props = getattr(f, "props", None)
    v = getattr(props, "lipschitz", None) if props is not None else None
    return 0.0 if v is None else float(v)


# ---------------------------------------------------------------------------
# One solve
# ---------------------------------------------------------------------------


def make_step(built, rho):
    def step(state, k, d):
        if built.relaxed_T is not None:
            return built.relaxed_T(state, k, d, rho)
        Tx = built.T(state, k, d)
        if rho == 1.0:
            return Tx
        return jax.tree_util.tree_map(lambda a, b: (1.0 - rho) * a + rho * b,
                                      state, Tx)
    return step


def gate_lhs(scheme, t, s, opnorm, L_f):
    """The gate's own left-hand side, computed independently of prepare_base."""
    prod = t * s * opnorm ** 2
    if scheme == "condat_vu":
        return t * L_f / 2.0 + prod, prod
    return prod, prod


def run_setting(problem, scheme, t, s, rho, *, K, max_iters, tol, repeats,
                loop):
    """Gate + (if admitted) solve.  Returns one flat record."""
    opnorm = float(problem.L.norm_bound)
    L_f = smooth_lipschitz(problem)
    lhs, prod = gate_lhs(scheme, t, s, opnorm, L_f)
    rec = {
        "scheme": scheme, "t": float(t), "s": float(s),
        "rho": float(rho), "ratio_t_over_s": float(t / s),
        "opnorm_gate": opnorm, "L_smooth": L_f,
        "theta_gate": float(prod), "gate_lhs": float(lhs),
    }
    genome = Genome(scheme, GenomeParams(tau=float(t), sigma=float(s),
                                         rho=(None if rho == 1.0 else float(rho))))
    try:
        built, cert, params, _ = prepare_base(problem, genome)
    except TypeCheckError as e:
        rec.update(gate_admit=False, gate_error=str(e)[:300],
                   gate_condition=None, N=None, converged=None,
                   final_gap=None, exec_s=None)
        return rec
    rec["gate_admit"] = True
    rec["gate_error"] = None
    rec["gate_condition"] = getattr(cert, "condition_str", None)
    rec["gate_averagedness"] = getattr(cert, "averagedness", None)

    dyn64 = jax.tree_util.tree_map(jnp.asarray, built.dyn)
    metric_jit = jax.jit(built.metric)           # an UNJITTED metric dominates timing

    def gap64(state):
        return metric_jit(state, dyn64)

    state0 = jax.tree_util.tree_map(
        lambda a: jnp.asarray(a, jnp.float64), built.state0)
    step = make_step(built, rho)

    if loop in ("static", "both"):
        times, N, conv, gap, comp = [], None, None, None, 0.0
        for _ in range(repeats):
            gaps, gap_iters = [], []
            st, k, converged, extra, compile_s, exec_s = P._run_phase(
                step, state0, 0, K, max_iters, gap64, tol, gaps, gap_iters,
                dyn=dyn64, static_key=built.static_key)
            times.append(exec_s)
            N, conv, gap, comp = k, converged, float(gaps[-1]), compile_s
        timed = times[1:] if len(times) > 1 else times
        rec.update(N=int(N), converged=bool(conv), final_gap=gap,
                   nonfinite_abort=bool(extra),
                   exec_s=float(np.median(timed)), exec_s_all=times,
                   compile_s=float(comp), loop="static")
    if loop in ("traced", "both"):
        times, pr = [], None
        for _ in range(repeats):
            pr = P.run_device_policy(
                step, state0, dyn64, lambda s64, gd: metric_jit(s64, gd),
                dyn64, K=K, max_iters=max_iters, tol=tol,
                static_key=built.static_key)
            times.append(pr.exec_time_s)
        timed = times[1:] if len(times) > 1 else times
        traced = {"N": int(pr.iters), "converged": bool(pr.converged),
                  "final_gap": float(pr.certified_gap),
                  "exec_s": float(np.median(timed)), "exec_s_all": times,
                  "compile_s": float(pr.compile_time_s)}
        if loop == "traced":
            rec.update(N=traced["N"], converged=traced["converged"],
                       final_gap=traced["final_gap"], exec_s=traced["exec_s"],
                       exec_s_all=traced["exec_s_all"],
                       compile_s=traced["compile_s"], loop="traced")
        else:
            rec["traced"] = traced
    return rec


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------


def device_tag(explicit=None):
    if explicit:
        return explicit
    kind = jax.devices()[0].device_kind
    m = re.search(r"(\d{4})", kind)
    if m:
        return m.group(1)
    return re.sub(r"[^a-z0-9]+", "_", kind.lower()).strip("_")


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


def sweep(args):
    taus = [v.strip() for v in args.taus.split(",")]
    thetas = [float(v) for v in args.thetas.split(",")]
    sizes = [int(v) for v in args.sizes.split(",")]
    degrees = [int(v) for v in args.degrees.split(",")]
    if args.family != "regular":
        degrees = degrees[:1]

    out = {
        "version": 1,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "args": vars(args),
        "device": {
            "kind": jax.devices()[0].device_kind,
            "platform": jax.devices()[0].platform,
            "jax_version": jax.__version__,
            "tag": device_tag(args.device),
        },
        "step_parameterisation": (
            "s = theta / (t * ||L||^2) with ||L|| = L.norm_bound, so "
            "t*s*||L||^2 == theta EXACTLY for every t in the sweep; the gate "
            "verdict is therefore a function of theta alone and t sweeps the "
            "gate-free ratio r = t/s = t^2||L||^2/theta"
        ),
        "instances": [],
        "runs": [],
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.family}_{device_tag(args.device)}.json"

    def flush():
        out_path.write_text(json.dumps(out, indent=1, default=_json_default))

    for size in sizes:
        for degree in degrees:
            cell_id = (f"{args.family}_s{size}"
                       + (f"_d{degree}" if args.family == "regular" else ""))
            t0 = time.perf_counter()
            problems = build_cell(args.family, size, degree, args.n_instances,
                                  args.seed, args.tv_lam, args.tv_split,
                                  args.tv_crop)
            build_s = time.perf_counter() - t0
            print(f"\n=== cell {cell_id}: {len(problems)} instances "
                  f"(build {build_s:.2f} s) ===", flush=True)
            for i, prob in enumerate(problems):
                feats = structure_features(
                    prob,
                    max_dense_elems=args.kappa_max_dense_elems,
                    max_gram_dim=args.kappa_max_gram_dim,
                    max_nnz=args.kappa_max_nnz,
                    iterative=not args.no_iterative_kappa,
                )
                feats.update(cell_id=cell_id, family=args.family, size=size,
                             degree=degree, instance=i,
                             build_seconds=build_s / max(len(problems), 1))
                out["instances"].append(feats)
                kp = feats["kappa_plus"]
                print(f"  [{cell_id} #{i}] n={prob.shape[0]} "
                      f"m={len(prob.data['b']) if 'b' in prob.data else '-'} "
                      f"||L||_gate={feats['norm_bound']:.6f} "
                      f"sigma_max={feats['sigma_max'] if feats['sigma_max'] is None else round(feats['sigma_max'], 6)} "
                      f"kappa+={'skipped' if kp is None else round(kp, 4)} "
                      f"({feats['kappa_method']}, {feats['t_feature_s']:.3f} s)",
                      flush=True)
                flush()

                nb = feats["norm_bound"]
                for theta in thetas:
                    for tok in taus:
                        if tok == "bal":
                            # the catalog "balanced" split: t == s exactly,
                            # i.e. r = t/s = 1, still at the same product theta
                            t = math.sqrt(theta) / nb
                        else:
                            t = float(tok)
                        s = theta / (t * nb * nb)
                        rec = run_setting(
                            prob, args.scheme, t, s, args.rho,
                            K=args.K, max_iters=args.max_iters, tol=args.tol,
                            repeats=args.repeats, loop=args.loop)
                        rec.update(t_label=tok,
                                   cell_id=cell_id, family=args.family,
                                   size=size, degree=degree, instance=i,
                                   theta_nominal=theta,
                                   kappa_plus=kp,
                                   sigma_max=feats["sigma_max"],
                                   theta_true=(
                                       None if feats["sigma_max"] is None
                                       else t * s * feats["sigma_max"] ** 2))
                        out["runs"].append(rec)
                        if rec["gate_admit"]:
                            print(f"    theta={theta:<5g} t={t:<8.5g}[{tok:<5s}] s={s:<10.5f} "
                                  f"r=t/s={t/s:<10.4g} ADMIT  N={rec['N']:>7d} "
                                  f"conv={str(rec['converged']):<5s} "
                                  f"exec={rec['exec_s']:.4f}s "
                                  f"gap={rec['final_gap']:.2e}", flush=True)
                        else:
                            print(f"    theta={theta:<5g} t={t:<8.5g}[{tok:<5s}] s={s:<10.5f} "
                                  f"r=t/s={t/s:<10.4g} REJECT ({rec['gate_error'][:70]})",
                                  flush=True)
                        flush()
    flush()
    print(f"\nwrote {out_path}")
    return out, str(out_path)


# ---------------------------------------------------------------------------
# Summary: does the product alone predict the best setting?
# ---------------------------------------------------------------------------


def _geomean(xs):
    xs = [x for x in xs if x and x > 0]
    return float(math.exp(sum(math.log(x) for x in xs) / len(xs))) if xs else None


def _spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.size < 3:
        return None
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    den = math.sqrt(float((ra ** 2).sum() * (rb ** 2).sum()))
    return float((ra * rb).sum() / den) if den > 0 else None


def summarize(out, stream=sys.stdout):
    def w(*a):
        print(*a, file=stream, flush=True)

    runs = out["runs"]
    insts = out["instances"]
    cells = []
    for r in insts:
        if r["cell_id"] not in [c["cell_id"] for c in cells]:
            cells.append({"cell_id": r["cell_id"], "kappa_plus": r["kappa_plus"],
                          "norm_bound": r["norm_bound"],
                          "sigma_max": r["sigma_max"],
                          "kappa_method": r["kappa_method"],
                          "omega0": (r["primal_weight"] or {}).get("omega0"),
                          "rho_row": (r["equilibration"] or {}).get("rho_row"),
                          "rho_col": (r["equilibration"] or {}).get("rho_col"),
                          "rho_entry": (r["equilibration"] or {}).get("rho_entry"),
                          "n": int(np.prod(r["shape"])) if r["shape"] else None})
    thetas = sorted({r["theta_nominal"] for r in runs})
    for r in runs:
        r.setdefault("t_label", f"{r['t']:g}")
    tau_t = {}
    for r in runs:
        tau_t.setdefault(r["t_label"], []).append(r["t"])
    taus = sorted(tau_t, key=lambda k: float(np.median(tau_t[k])))

    w("\n" + "=" * 100)
    w("STRUCTURE FEATURES PER CELL  (geometric mean over instances where they vary)")
    w("=" * 100)
    w(f"{'cell':<20}{'n':>7}{'||L||_gate':>12}{'sigma_max':>12}"
      f"{'ratio':>8}{'kappa_plus':>12}{'method':>16}{'omega0':>9}"
      f"{'rho_row':>9}{'rho_col':>9}{'rho_ent':>9}")
    for c in cells:
        ratio = (c["norm_bound"] / c["sigma_max"]) if c["sigma_max"] else float("nan")
        w(f"{c['cell_id']:<20}{c['n'] or 0:>7}{c['norm_bound']:>12.6f}"
          f"{(c['sigma_max'] or float('nan')):>12.6f}{ratio:>8.4f}"
          f"{(c['kappa_plus'] or float('nan')):>12.4f}{c['kappa_method']:>16}"
          f"{(c['omega0'] or float('nan')):>9.3f}"
          f"{(c['rho_row'] or float('nan')):>9.3f}"
          f"{(c['rho_col'] or float('nan')):>9.3f}"
          f"{(c['rho_entry'] or float('nan')):>9.3f}")
    nbs = [c["norm_bound"] for c in cells]
    kps = [c["kappa_plus"] for c in cells if c["kappa_plus"]]
    if nbs:
        w(f"\n  ||L||_gate spread over cells : {max(nbs)/min(nbs):.4f}x")
    if len(kps) > 1:
        w(f"  kappa_plus spread over cells : {max(kps)/min(kps):.4f}x")

    # ---- gate verdicts: a function of theta alone, by construction
    w("\n" + "=" * 100)
    w("GATE VERDICT vs theta = t*s*||L||^2   (identical for every t in a column)")
    w("=" * 100)
    for th in thetas:
        sub = [r for r in runs if r["theta_nominal"] == th]
        adm = sum(1 for r in sub if r["gate_admit"])
        tt = sorted({r["theta_true"] for r in sub if r.get("theta_true")})
        tt_s = f"  t*s*sigma_max^2 in [{min(tt):.4f}, {max(tt):.4f}]" if tt else ""
        w(f"  theta={th:<6g}  admitted {adm:>4d}/{len(sub):<4d} runs{tt_s}")
    w("  NOTE the gate uses ||L||_gate = power-iteration bound x 1.02 safety, so a")
    w("  nominal theta slightly above 1 can still have a TRUE product below 1.")

    # ---- the decisive table: N along a column of constant theta
    w("\n" + "=" * 100)
    w("ITERATIONS N vs (theta, t) -- geometric mean over instances; * = hit the cap")
    w("=" * 100)
    table = {}
    for c in cells:
        w(f"\n  cell {c['cell_id']}  (kappa_plus="
          f"{'skipped' if not c['kappa_plus'] else round(c['kappa_plus'], 3)},"
          f" ||L||_gate={c['norm_bound']:.4f})")
        w("    " + f"{'theta \\ t':<12}" + "".join(f"{t:>12s}" for t in taus)
          + f"{'  spread':>10}")
        for th in thetas:
            cells_row, vals = [], []
            for t in taus:
                sub = [r for r in runs
                       if r["cell_id"] == c["cell_id"]
                       and r["theta_nominal"] == th and r["t_label"] == t]
                if not sub:
                    cells_row.append(f"{'-':>12}")
                    continue
                if not sub[0]["gate_admit"]:
                    cells_row.append(f"{'REJECT':>12}")
                    continue
                g = _geomean([r["N"] for r in sub])
                capped = any(not r["converged"] for r in sub)
                table[(c["cell_id"], th, t)] = (
                    g, capped, float(np.median([r["t"] for r in sub])),
                    float(np.median([r["ratio_t_over_s"] for r in sub])),
                    float(np.median([r["N"] for r in sub])),
                    float(min(r["N"] for r in sub)),
                    float(max(r["N"] for r in sub)))
                vals.append(g)
                cells_row.append(f"{int(round(g)):>11d}" + ("*" if capped else " "))
            sp = f"{max(vals)/min(vals):>9.2f}x" if len(vals) > 1 else f"{'-':>10}"
            w("    " + f"{th:<12g}" + "".join(cells_row) + sp)

    # ---- Q1: is the product alone enough?
    w("\n" + "=" * 100)
    w("Q1  DOES theta = t*s*||L||^2 ALONE PREDICT THE BEST SETTING?")
    w("=" * 100)
    w("    If it did, N would be constant along a row of fixed theta (spread 1.00x).")
    worst = 0.0
    for c in cells:
        for th in thetas:
            vals = [table[k][0] for k in table
                    if k[0] == c["cell_id"] and k[1] == th]
            if len(vals) > 1:
                worst = max(worst, max(vals) / min(vals))
    w(f"    MEASURED worst within-theta spread over all cells: {worst:.2f}x")
    w("    => theta fixes the gate verdict but NOT the cost; the free ratio r = t/s")
    w("       carries the performance, and the gate is blind to it by construction.")

    # ---- Q2: best setting per cell, and is one global setting enough?
    w("\n" + "=" * 100)
    w("Q2  BEST SETTING PER CELL, AND THE COST OF ONE GLOBAL SETTING")
    w("=" * 100)
    best = {}
    for c in cells:
        keys = [k for k in table if k[0] == c["cell_id"]]
        if not keys:
            continue
        bk = min(keys, key=lambda k: table[k][0])
        best[c["cell_id"]] = (bk, table[bk][0])
        e = table[bk]
        w(f"    {c['cell_id']:<20} best (theta={bk[1]:g}, t={e[2]:.5g} "
          f"[{bk[2]}], r=t/s={e[3]:.4g})  N={int(round(e[0]))}"
          f"  per-seed N in [{int(e[5])}, {int(e[6])}]"
          + ("  [CAPPED]" if e[1] else ""))
    settings = sorted({(k[1], k[2]) for k in table}, key=lambda z: (z[0], str(z[1])))
    scored = []
    for th, t in settings:
        regrets = []
        for cid, (_, bn) in best.items():
            v = table.get((cid, th, t))
            if v is None:
                regrets = None
                break
            regrets.append(v[0] / bn)
        if regrets:
            scored.append((max(regrets), _geomean(regrets), th, t))
    if scored:
        scored.sort()
        w("\n    best SINGLE GLOBAL (theta, t), ranked by worst-case regret:")
        for mx, gm, th, t in scored[:8]:
            w(f"      theta={th:<6g} t={str(t):<6s}  max regret {mx:.3f}x   "
              f"geomean regret {gm:.3f}x")
        mx, gm, th, t = scored[0]
        w(f"    => one global recipe (theta={th:g}, t={t}) costs at most "
          f"{mx:.3f}x the per-cell oracle here.")
        bal = [z for z in scored if z[3] == "bal"]
        if bal:
            mxb, gmb, thb, _ = min(bal)
            w(f"    the CATALOG BALANCED split t == s (r = 1), best theta={thb:g}: "
              f"max regret {mxb:.3f}x, geomean {gmb:.3f}x -- this is what the "
              f"gate alone leaves you with.")

    # ---- Q3: does the condition number carry the extra information?
    w("\n" + "=" * 100)
    w("Q3  IS THE TRUE CONDITION NUMBER NEEDED ON TOP OF THE PRODUCT?")
    w("=" * 100)
    xs, ys, zs, names = [], [], [], []
    for c in cells:
        if not c["kappa_plus"] or c["cell_id"] not in best:
            continue
        bk, bn = best[c["cell_id"]]
        r_best = table[bk][3]
        xs.append(c["kappa_plus"])
        ys.append(math.log2(r_best))
        zs.append(bn)
        names.append(c["cell_id"])
        w(f"    {c['cell_id']:<20} kappa_plus={c['kappa_plus']:>9.4f}  "
          f"||L||_gate={c['norm_bound']:.4f}  best t={table[bk][2]:<8.5g}  "
          f"log2(r_best)={math.log2(r_best):>7.3f}  N_best={int(round(bn))}")
    if len(xs) >= 3:
        w(f"\n    Spearman(kappa_plus, log2 r_best) = {_spearman(xs, ys)}")
        w(f"    Spearman(kappa_plus, N_best)       = {_spearman(xs, zs)}")
        nb_list = [c["norm_bound"] for c in cells if c["kappa_plus"]]
        w(f"    Spearman(||L||_gate, N_best)       = {_spearman(nb_list, zs)}")
    else:
        w("    (fewer than 3 cells with a kappa_plus: correlations not computed;")
        w("     read the per-cell rows above directly)")
    w("=" * 100)


# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", default="grid", choices=list(ALL_FAMILIES))
    ap.add_argument("--sizes", default="16,32",
                    help="comma list; grid side k / n_nodes / n vars / TV patch side")
    ap.add_argument("--degrees", default="4",
                    help="comma list of degrees (family 'regular' only)")
    ap.add_argument("--n-instances", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scheme", default="pdhg", choices=("pdhg", "condat_vu"))
    ap.add_argument("--taus", default="0.005,0.01,0.02,0.05,0.1,0.25,bal,0.5",
                    help="comma list of t values; the token 'bal' means the "
                         "catalog balanced split t == s at the same product")
    ap.add_argument("--thetas", default="0.3,0.6,0.9,0.99,1.02")
    ap.add_argument("--rho", type=float, default=1.0)
    ap.add_argument("--K", type=int, default=64, help="certificate cadence")
    ap.add_argument("--max-iters", type=int, default=150000)
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--repeats", type=int, default=2,
                    help="solve repeats; the first is a warm-up, median of the rest")
    ap.add_argument("--loop", default="static",
                    choices=("static", "traced", "both"))
    ap.add_argument("--kappa-max-dense-elems", type=float, default=8e6)
    ap.add_argument("--kappa-max-gram-dim", type=int, default=4096)
    ap.add_argument("--kappa-max-nnz", type=float, default=4e7)
    ap.add_argument("--no-iterative-kappa", action="store_true")
    ap.add_argument("--tv-lam", type=float, default=0.1)
    ap.add_argument("--tv-split", default="test")
    ap.add_argument("--tv-crop", default="center")
    ap.add_argument("--device", default=None, help="tag for the output filename")
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--report", default=None,
                    help="comma list of existing JSONs: re-print the (merged) "
                         "summary and exit.  Merging several families is how "
                         "the cross-family question is answered.")
    args = ap.parse_args()

    if args.report:
        paths = [q.strip() for q in args.report.split(",") if q.strip()]
        merged = {"instances": [], "runs": [], "sources": paths}
        for q in paths:
            d = json.loads(Path(q).read_text())
            merged["instances"] += d["instances"]
            merged["runs"] += d["runs"]
            merged.setdefault("device", d.get("device"))
            merged.setdefault("args", d.get("args"))
        summarize(merged)
        return

    t0 = time.perf_counter()
    out, path = sweep(args)
    out["wall_seconds"] = time.perf_counter() - t0
    Path(path).write_text(json.dumps(out, indent=1, default=_json_default))
    summarize(out)
    print(f"\ntotal wall {out['wall_seconds']:.1f} s -> {path}")


if __name__ == "__main__":
    main()
