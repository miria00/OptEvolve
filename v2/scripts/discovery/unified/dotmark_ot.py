"""DOTmark discrete optimal transport as OSQP-form LPs, plus a SIDE-SELECTABLE product plan (2026-09-15).

DOTmark (Schrieber, Schuhmacher, Gottschlich 2017) ships ten image classes at resolutions 32, 64, 128, 256, 512,
ten images each. A transport instance is a PAIR of images from one class, normalized to probability measures on the
same pixel grid, with the squared-Euclidean ground cost between pixel centers. That is the real-data version of
lp_families.transport, and it lands in the same OSQP dict (P = 0, general rows then identity bound rows), so
uni.plan_product, uni.plan_block, lp_families.lp_formulation and router.solve all read it unchanged.

WHY THIS INSTANCE CLASS IS THE PRODUCT-OF-SIMPLICES STRESS TEST
--------------------------------------------------------------
With x = vec(Pi) in R^{mn} the constraint rows are

    source rows   sum_j x_ij = a_i        m rows, nnz n each      kron(I_m, 1_n^T)
    sink rows     sum_i x_ij = b_j        n rows, nnz m each      kron(1_m^T, I_n)
    bound rows    x_ij >= 0               mn rows, nnz 1 each     I

EITHER marginal family is a disjoint set of rows whose whole support carries a finite bound row, so EITHER is a
product of mn/K scaled simplices that uni.product_project handles exactly, and each covers every variable (cover 1).
The two families overlap in every single variable, so exactly ONE can move into the proximal step; the other stays
in the linear operator. On a square instance (m = n, which is every same-resolution DOTmark pair) the two families
are INDISTINGUISHABLE to uni.plan_product's greedy: it sorts absorbable rows by (-nnz, row index) and every general
row has nnz = m = n, so the sort degenerates to row index and the family of the LOWEST-INDEXED general row wins.
Under the pipeline's randomized encoding that is a coin flip. `plan_side` below builds the same plan for the family
the caller names, which is what makes the rows-versus-columns iteration comparison possible.

MEMORY. The ground cost is a vector, not a matrix: c = vec(C) of length mn is the LP's q, so nothing is gained by
keeping C implicit (`cost_vector` never forms an mn x mn object; it forms the m x n cost and ravels it). The LINEAR
OPERATOR is what grows, and the pipeline needs it explicitly (plan_product reads A's rows, SparseLinOp builds the
jax operator from a scipy matrix, uni.accept multiplies by A). A is (m + n + mn) x mn with mn + mn + mn = 3mn
nonzeros, all +1:

    resolution   mn = n_var     rows        A nnz      A in CSC (measured: data + indices + indptr)
    16x16          65,536       66,048      196,608     2.6 MB
    32x32       1,048,576    1,050,624    3,145,728    41.9 MB
    64x64      16,777,216   16,785,408   50,331,648   671.1 MB

so the explicit operator is affordable through 64x64 on a host (the jax side then holds it twice, as BCOO A and
BCOO A^T). The matrix-free form of the SAME operator is a pair of segment sums over a reshaped x plus the identity
block, which is where a synthesized operator would get its win; nothing in this file needs it, and the rest of the
pipeline reads A as a scipy matrix, so A is built explicitly and its cost is reported rather than hidden.

CONVENTIONS
  pixel centers  ((i + 0.5) / R, (j + 0.5) / R) in [0, 1]^2, so the ground cost is in [0, 2).
  marginals      image / image.sum(); mass balance is not imposed but CONSTRUCTED: the right-hand side is the pair
                 of marginals OF THE PRODUCT COUPLING a b^T, read off by the same sparse matvec the acceptance check
                 uses, so x_feas is feasible to machine precision and sum(a) = sum(b) identically (the trick of
                 atlas.bench.netflow._finish_instance, which lp_families.transport also inherits).
  resolution 16  DOTmark has no 16x16 file; it is a 2x2 block SUM of the 32x32 image, which is the pushforward of
                 the measure under the coarsening map, so it is the same measure on a coarser grid, not a resample.
"""
from __future__ import annotations

import itertools
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(_HERE))))

import numpy as np
import scipy.sparse as sp

CLASSES = ("CauchyDensity", "ClassicImages", "GRFmoderate", "GRFrough", "GRFsmooth", "LogGRF", "LogitGRF",
           "MicroscopyImages", "Shapes", "WhiteNoise")
NATIVE = (32, 64, 128, 256, 512)
IMAGE_IDS = tuple(range(1001, 1011))
PAIRS = tuple(itertools.combinations(range(10), 2))          # the 45 DOTmark pairs of a class, in DOTmark's order

_ROOTS = ("/dev/shm/data/dotmark", "/home/miria/OptEvolve/v2/data/dotmark",
          os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(_HERE))), "data", "dotmark"))


def root() -> str:
    r = os.environ.get("OPTEVOLVE_DOTMARK_ROOT")
    if r:
        return r
    for c in _ROOTS:
        if os.path.isdir(os.path.join(c, "Data")):
            return c
    raise FileNotFoundError(f"DOTmark not found; set OPTEVOLVE_DOTMARK_ROOT (looked in {_ROOTS})")


def listing(resolution: int | None = None) -> dict:
    """What is on disk: classes, the native resolutions present, and the pair indices a caller may ask for."""
    base = os.path.join(root(), "Data")
    out = {}
    for cls in sorted(os.listdir(base)):
        d = os.path.join(base, cls)
        if not os.path.isdir(d):
            continue
        res = sorted({int(f[4:f.index("_")]) for f in os.listdir(d) if f.startswith("data") and f.endswith(".csv")})
        ids = sorted({int(f[f.index("_") + 1:-4]) for f in os.listdir(d) if f.startswith("data")})
        out[cls] = {"native_resolutions": res, "image_ids": ids, "n_pairs": len(PAIRS)}
    if resolution is not None:
        out = {k: v for k, v in out.items() if _source_resolution(resolution) in v["native_resolutions"]}
    return out


def _source_resolution(resolution: int) -> int:
    """The native file a requested resolution is read from (16 comes from the 32x32 file by block summation)."""
    if resolution in NATIVE:
        return resolution
    for r in NATIVE:
        if r % resolution == 0:
            return r
    raise ValueError(f"resolution {resolution} is not a divisor of any DOTmark resolution {NATIVE}")


def load_image(class_name: str, image_id: int, resolution: int) -> np.ndarray:
    """One DOTmark image as a nonnegative float array of shape (resolution, resolution)."""
    src = _source_resolution(resolution)
    path = os.path.join(root(), "Data", class_name, f"data{src}_{image_id}.csv")
    img = np.loadtxt(path, delimiter=",", dtype=np.float64)
    if img.shape != (src, src):
        raise ValueError(f"{path}: expected {src}x{src}, got {img.shape}")
    if src != resolution:
        f = src // resolution
        img = img.reshape(resolution, f, resolution, f).sum(axis=(1, 3))     # pushforward of the measure
    return img


def cost_vector(resolution: int, p: float = 2.0) -> np.ndarray:
    """vec(C) for C[i, j] = |z_i - z_j|^p on pixel centers of the unit square. Length resolution^4.

    Built from the m x n cost, never from an (mn x mn) object. p = 2 is the DOTmark / Wasserstein-2 convention.
    """
    R = int(resolution)
    k = np.arange(R, dtype=np.float64)
    ctr = (k + 0.5) / R
    dr = ctr[:, None] - ctr[None, :]
    d2 = dr ** 2
    # C[(r1,c1),(r2,c2)] = (ctr[r1]-ctr[r2])^2 + (ctr[c1]-ctr[c2])^2, so C = d2 (x) 1 + 1 (x) d2 on the grid index
    C = d2[:, None, :, None] + d2[None, :, None, :]
    if p != 2.0:
        C = C ** (p / 2.0)
    return C.reshape(R * R, R * R).ravel()


def make_ot(class_name: str, pair_index: int, resolution: int, *, p: float = 2.0, permute: bool = True,
            seed: int = 0, with_feas: bool = True) -> dict:
    """A DOTmark transport LP in the pipeline's OSQP dict.

    Returns P, q, r, A, l, u plus x_feas (the product coupling, feasible by construction), name and meta. meta
    carries `row_family` ALIGNED WITH THE RETURNED ROW ORDER (0 = source/row marginal, 1 = sink/column marginal,
    2 = bound row), which is what plan_side needs after the encoding is permuted.
    """
    if not 0 <= pair_index < len(PAIRS):
        raise ValueError(f"pair_index must be in [0, {len(PAIRS)})")
    i1, i2 = PAIRS[pair_index]
    id1, id2 = IMAGE_IDS[i1], IMAGE_IDS[i2]
    im1, im2 = load_image(class_name, id1, resolution), load_image(class_name, id2, resolution)
    R = int(resolution)
    M = R * R
    N = M * M
    a0 = im1.ravel() / im1.sum()
    b0 = im2.ravel() / im2.sum()

    Rm = sp.csr_matrix((np.ones(N), (np.repeat(np.arange(M), M), np.arange(N))), shape=(M, N))
    Cm = sp.csr_matrix((np.ones(N), (np.tile(np.arange(M), M), np.arange(N))), shape=(M, N))
    x_feas = (a0[:, None] * b0[None, :]).ravel()          # product coupling: the cheapest exactly feasible point
    a = Rm @ x_feas
    b = Cm @ x_feas                                       # marginals READ OFF x_feas, so mass balance is exact
    q = cost_vector(R, p)

    G = sp.vstack([Rm, Cm], format="csr")
    rhs = np.concatenate([a, b])
    B = sp.identity(N, format="csr", dtype=np.float64)
    A = sp.vstack([G, B], format="csc")
    l = np.concatenate([rhs, np.zeros(N)])
    u = np.concatenate([rhs, np.full(N, np.inf)])
    fam = np.concatenate([np.zeros(M, dtype=np.int8), np.ones(M, dtype=np.int8), np.full(N, 2, dtype=np.int8)])
    meta = {"dataset": "DOTmark_1.0", "class": class_name, "pair_index": int(pair_index),
            "image_ids": [int(id1), int(id2)], "resolution": R, "m": M, "n": M, "p": float(p),
            "n_var": int(N), "rows": int(A.shape[0]), "nnz": int(A.nnz),
            "zero_mass_sources": int((a0 == 0).sum()), "zero_mass_sinks": int((b0 == 0).sum()),
            "mass_a": float(a.sum()), "mass_b": float(b.sum()), "cost_max": float(q.max()),
            "row_family": fam}
    d = {"P": sp.csc_matrix((N, N)), "q": q, "r": 0.0, "A": A, "l": l, "u": u,
         "name": f"dotmark_{class_name}_{id1}v{id2}_{R}x{R}", "meta": meta}
    if with_feas:
        d["x_feas"] = x_feas
    return permute_ot(d, 1000 + seed) if permute else d


def permute_ot(d: dict, seed: int) -> dict:
    """uni.permute_instance, but it also carries row_family / x_feas through the permutation.

    uni.permute_instance drops every key it does not know, so the row labels would be lost and plan_side could not
    name a family on a permuted encoding. Same permutation semantics: the problem is unchanged, the encoding is not.
    """
    rng = np.random.default_rng(seed)
    n, m = d["P"].shape[0], d["A"].shape[0]
    pc, pr = rng.permutation(n), rng.permutation(m)
    A = sp.csc_matrix(sp.csc_matrix(d["A"])[pr][:, pc])
    out = {"P": sp.csc_matrix((n, n)), "q": np.asarray(d["q"])[pc], "r": d["r"], "A": A,
           "l": np.asarray(d["l"])[pr], "u": np.asarray(d["u"])[pr], "name": d.get("name")}
    meta = dict(d.get("meta") or {})
    fam = meta.get("row_family")
    if fam is None:
        raise ValueError("permute_ot needs meta['row_family']")
    meta["row_family"] = np.asarray(fam)[pr]
    meta["col_perm"], meta["row_perm"], meta["permute_seed"] = pc, pr, int(seed)
    out["meta"] = meta
    if "x_feas" in d:
        out["x_feas"] = np.asarray(d["x_feas"])[pc]
    return out


def solver_dict(d: dict) -> dict:
    """The six keys the pipeline's entry points accept (uni.reference / router.solve tolerate extras, but a
    baseline dump should not carry a 134 MB x_feas)."""
    return {k: d[k] for k in ("P", "q", "r", "A", "l", "u")}


def feasibility(d: dict) -> dict:
    """Mass balance and the acceptance metric's row violation at x_feas, on the ORIGINAL (possibly permuted) dict."""
    import lp_families
    meta = d["meta"]
    fam = np.asarray(meta["row_family"])
    l = np.asarray(d["l"])
    ma, mb = float(l[fam == 0].sum()), float(l[fam == 1].sum())
    out = {"mass_source": ma, "mass_sink": mb, "mass_imbalance": abs(ma - mb),
           "mass_imbalance_rel": abs(ma - mb) / max(abs(ma), 1e-300)}
    if "x_feas" in d:
        out["x_feas_row_violation"] = lp_families.row_violation(d, d["x_feas"])
        out["obj_x_feas"] = float(np.asarray(d["q"]) @ d["x_feas"])
    return out


# ----------------------------------------------------------------------------------------------------------------
# THE PLANNER QUESTION: the same product plan, for the family the caller names.
# ----------------------------------------------------------------------------------------------------------------
def _plan_from_rows(d: dict, rows: np.ndarray) -> dict:
    """uni.plan_product's construction, for a caller-chosen disjoint family of rows.

    Identical in kind to plan_product's return (same keys, same canonicalization: absorbed columns first, in the
    order the blocks are listed), so uni.build_formulation's `kind == "product"` branch takes it unchanged. The
    ONLY thing this replaces is the greedy's choice of which rows to take.
    """
    A = sp.csr_matrix(d["A"])
    l, u = np.asarray(d["l"], dtype=float), np.asarray(d["u"], dtype=float)
    m, n = A.shape
    nnz = np.diff(A.indptr)
    lo, hi = np.full(n, -np.inf), np.full(n, np.inf)
    # vectorized where uni.plan_product loops: at 64x64 there are 16.8M single-nonzero bound rows and a Python loop
    # over them costs minutes, which would dominate every structure report on this family
    br = np.flatnonzero(nnz == 1)
    bc = A.indices[A.indptr[br]]
    bv = A.data[A.indptr[br]]
    ok = bv != 0.0
    br, bc, bv = br[ok], bc[ok], bv[ok]
    pos = bv > 0
    rl = np.where(pos, l[br] / bv, u[br] / bv)
    ru = np.where(pos, u[br] / bv, l[br] / bv)
    np.maximum.at(lo, bc, rl)
    np.minimum.at(hi, bc, ru)
    rows = np.asarray(sorted(int(r) for r in rows), dtype=int)
    T, bid, a = [], [], []
    seen = np.zeros(n, dtype=bool)
    for k, r in enumerate(rows):
        cols = A.indices[A.indptr[r]:A.indptr[r + 1]]
        if seen[cols].any():
            raise ValueError(f"row {r} overlaps an earlier row: the family is not disjoint")
        seen[cols] = True
        order = np.argsort(cols)
        T.append(cols[order])
        a.append(A.data[A.indptr[r]:A.indptr[r + 1]][order])
        bid.append(np.full(cols.size, k))
    T, a, bid = np.concatenate(T), np.concatenate(a), np.concatenate(bid)
    if not (np.isfinite(lo[T]) | np.isfinite(hi[T])).all():
        raise ValueError("some absorbed column carries no finite bound row")
    drop = np.zeros(m, dtype=bool)
    drop[rows] = True
    drop[br[seen[bc]]] = True                              # every bound row of an absorbed column leaves L too
    keep = np.flatnonzero(~drop)
    fully_absorbed = keep.size == 0
    if fully_absorbed:
        keep = br[bc == int(T[0])][:1]
    rest = np.flatnonzero(~seen)
    return {"kind": "product", "T": T, "bid": bid, "a": a, "lo": lo[T], "hi": hi[T],
            "b_lo": l[rows], "b_hi": u[rows], "K": int(rows.size), "rows": rows, "keep": keep,
            "perm": np.concatenate([T, rest]), "n_x": int(T.size), "cover": float(T.size / n),
            "fully_absorbed": bool(fully_absorbed)}


def plan_side(d: dict, side: str) -> dict:
    """The product plan that absorbs the named marginal family. side in {"source", "sink", "rows", "cols"}.

    "source"/"rows" absorbs sum_j x_ij = a_i (the ROW marginal of the plan, one simplex per source pixel) and leaves
    the column marginal in L; "sink"/"cols" is the mirror image. Both are exact product projections and both cover
    every variable, so the pair is a controlled experiment on the planner's choice alone.
    """
    key = {"source": 0, "rows": 0, "row": 0, "sink": 1, "cols": 1, "col": 1, "column": 1}
    if side not in key:
        raise ValueError(f"side must be one of {sorted(key)}")
    fam = np.asarray(d["meta"]["row_family"])
    return _plan_from_rows(d, np.flatnonzero(fam == key[side]))


def plan_side_of(d: dict, plan: dict) -> str:
    """Which marginal family a plan's rows belong to ("source", "sink" or "mixed")."""
    fam = np.asarray(d["meta"]["row_family"])[np.asarray(plan["rows"])]
    vals = set(fam.tolist())
    return {0: "source", 1: "sink"}.get(vals.pop(), "mixed") if len(vals) == 1 else "mixed"


def plans_agree(p: dict, q: dict) -> bool:
    """Whether two plan dicts describe the same absorbed set (used to check plan_side against uni.plan_product)."""
    if p is None or q is None:
        return False
    return (set(np.asarray(p["rows"]).tolist()) == set(np.asarray(q["rows"]).tolist())
            and int(p["K"]) == int(q["K"]) and np.array_equal(np.asarray(p["T"]), np.asarray(q["T"]))
            and np.array_equal(np.asarray(p["bid"]), np.asarray(q["bid"])))
