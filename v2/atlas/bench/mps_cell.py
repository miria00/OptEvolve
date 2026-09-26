"""MPS benchmark cell: real-world LPs (Mittelmann / MIPLIB relaxations)
loaded through HiGHS and converted to the standard form the LP composite
mapping expects:

    min_x  c'x   s.t.   A x = b,  x >= 0.

PRIMARY ROUTE (highspy): Highs().readModel(path) parses the MPS file
(HiGHS handles fixed and free MPS, gzip, OBJSENSE, RANGES, all BOUNDS
kinds; integrality markers are read and then IGNORED here, which is
exactly the LP relaxation). getLp() hands back the general form

    min/max  q'x + offset0   s.t.   rl <= M x <= ru,   l <= x <= u

as dense cost/bound vectors plus a column-wise sparse M. The v1
hand-rolled parser at OptimizationAtlas/atlas/mps.py was consulted for
the corner cases (inline '*' in row names, missing RHS set names, E rows
with RANGES, MI/BV/FR bounds); all of those are handled inside HiGHS
itself, which is the point of the highspy route.

STANDARD-FORM CONVERSION (to_standard_form). Each transform below is
numbered, applied in this order, and covered by tests
(tests/test_mps_cell.py verifies the converted optimum against HiGHS on
the ORIGINAL model to 1e-6 on real instances):

  T1  sense: work cost wc = q for minimize, wc = -q for maximize. The
      original objective is recovered as
      original = s * (std_opt + shift_const) + offset0, s = +-1.
  T2  free rows (rl = -inf, ru = +inf): dropped (no constraint).
  T3  equality rows (rl == ru finite): kept as A_i x = rl_i.
  T4  inequality / ranged rows: slack variable r with bounds
      [rl_i, ru_i] and row  M_i x - r = 0  (b_i = 0). One-sided rows are
      the special case with an infinite bound on one side; ranged rows
      (RANGES section) give r a finite box, handled by T5c below.
  After T2-T4 the system is  A z = b  with box bounds lo <= z <= hi on
  the extended variable vector z (originals + row slacks). Per variable:
  T5a lower-bounded (lo finite, hi = +inf): shift z = lo + z',
      z' >= 0;  b -= A_col * lo;  shift_const += wc_j * lo.
  T5b upper-bounded only (lo = -inf, hi finite): reflect z = hi - z',
      z' >= 0; column and cost negated; b -= A_col * hi;
      shift_const += wc_j * hi.
  T5c boxed (both finite, lo < hi): shift by lo as in T5a, then add the
      box row  z' + w = hi - lo  with a fresh slack w >= 0 (cost 0).
  T5d fixed (lo == hi): eliminate the column;  b -= A_col * lo;
      shift_const += wc_j * lo.
  T5e free (both infinite): split z = z+ - z-, z+, z- >= 0 (a negated
      copy of the column with negated cost is appended).

  The converted problem's variables do not map back to the original x
  one-to-one (shifts, reflections, splits, eliminations); only the
  OBJECTIVE mapping (T1) is preserved and tested, which is what the
  bench cell needs.

COMPOSITE MAPPING: same atoms as atlas.bench.lp_cell (LinearCost /
NonnegOrthant / AffineEqualitySet), with L a SparseLinOp: a BCOO sparse
matvec (jit-safe) with a power-iteration norm bound inflated by 1.02
(atlas.bench.lp_cell.power_iteration_norm on scipy CSR/CSC callables).
The lp_gap certificate (atlas.certify.residuals.lp_rel_gap) and the
generalized schemes' name == "lp_standard" dispatch are reused verbatim.

ORACLE: HiGHS solve of the ORIGINAL MPS model (not the conversion),
cached by the sha256 content hash of the file bytes under
V2/data/oracle_cache_mps/. solve_standard_highs() independently solves
the CONVERTED form (scipy linprog, sparse HiGHS) for conversion tests.

BATCHING NOTE (same-shape batching for the batched backend): group
instances by (rows, cols) of the CONVERTED standard form; use
group_by_shape(). Real MPS suites are heterogeneous, so most groups are
singletons; same-shape batches come from the synthetic families
(atlas.bench.lp_cell / netflow), and the manifest carries (rows, cols)
so a planner can pre-group without loading matrices.

GATE NOTE: as with the non-default netflow families, the Stage-0
constant atlas.genome.OPNORM_LP does NOT cover MPS instances; gate
genomes with validate(..., lp_opnorm=gate_opnorm(instances)) using
atlas.bench.lp_cell.gate_opnorm (SparseLinOp carries norm_bound, so
gate_opnorm accepts these composites unchanged).
"""
from __future__ import annotations

import bz2
import gzip
import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional, Sequence

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
from jax.experimental import sparse as jsparse
from scipy.optimize import linprog

from atlas.bench.lp_cell import (
    AffineEqualitySet,
    LinearCost,
    NonnegOrthant,
    power_iteration_norm,
)
from atlas.ir.spec import CompositeProblem

_V2_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
DATA_ROOT = os.path.join(_V2_ROOT, "data", "lp_suites")
MANIFEST_PATH = os.path.join(DATA_ROOT, "MANIFEST.json")
ORACLE_CACHE_MPS_DIR = os.path.join(_V2_ROOT, "data", "oracle_cache_mps")

_INF_THRESHOLD = 1e29  # HiGHS uses kHighsInf (= inf in highspy wheels);
# treat any magnitude above this as infinite, matching HiGHS semantics.


# ---------------------------------------------------------------------------
# Reading: highspy front end
# ---------------------------------------------------------------------------


def _is_inf(v: np.ndarray) -> np.ndarray:
    return np.abs(v) >= _INF_THRESHOLD


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _highs_readable_path(path: str) -> tuple[str, Optional[str]]:
    """HiGHS reads .mps and .mps.gz natively; .bz2 is decompressed to a
    temp .mps first. Returns (readable_path, tempfile_or_None)."""
    if path.endswith(".bz2"):
        fd, tmp = tempfile.mkstemp(suffix=".mps")
        with os.fdopen(fd, "wb") as out, bz2.open(path, "rb") as src:
            shutil.copyfileobj(src, out)
        return tmp, tmp
    return path, None


@dataclass(frozen=True)
class RawLP:
    """General-form LP as HiGHS hands it back (see module docstring)."""

    c: np.ndarray  # (n0,) objective coefficients q
    col_lower: np.ndarray  # (n0,) l
    col_upper: np.ndarray  # (n0,) u
    row_lower: np.ndarray  # (m0,) rl
    row_upper: np.ndarray  # (m0,) ru
    M: sp.csc_matrix  # (m0, n0)
    offset: float  # offset0
    maximize: bool
    name: str
    n_integer: int  # count of integrality-marked columns (relaxed away)

    @property
    def nnz(self) -> int:
        return int(self.M.nnz)


def read_mps(path: str) -> RawLP:
    """Parse an MPS file (.mps / .mps.gz / .mps.bz2) via highspy."""
    import highspy

    rpath, tmp = _highs_readable_path(path)
    try:
        h = highspy.Highs()
        h.setOptionValue("output_flag", False)
        status = h.readModel(rpath)
        if status != highspy.HighsStatus.kOk:
            raise ValueError(f"HiGHS could not read {path!r}: {status}")
        lp = h.getLp()
        m0, n0 = int(lp.num_row_), int(lp.num_col_)
        am = lp.a_matrix_
        start = np.asarray(am.start_, dtype=np.int64)
        index = np.asarray(am.index_, dtype=np.int64)
        value = np.asarray(am.value_, dtype=np.float64)
        if am.format_ == highspy.MatrixFormat.kColwise:
            M = sp.csc_matrix((value, index, start), shape=(m0, n0))
        else:  # kRowwise
            M = sp.csr_matrix((value, index, start), shape=(m0, n0)).tocsc()
        integrality = np.asarray(lp.integrality_, dtype=np.int64)
        n_int = int(np.sum(integrality != 0)) if integrality.size else 0
        return RawLP(
            c=np.asarray(lp.col_cost_, dtype=np.float64),
            col_lower=np.asarray(lp.col_lower_, dtype=np.float64),
            col_upper=np.asarray(lp.col_upper_, dtype=np.float64),
            row_lower=np.asarray(lp.row_lower_, dtype=np.float64),
            row_upper=np.asarray(lp.row_upper_, dtype=np.float64),
            M=M,
            offset=float(lp.offset_),
            maximize=(lp.sense_ == highspy.ObjSense.kMaximize),
            name=str(lp.model_name_) or os.path.basename(path),
            n_integer=n_int,
        )
    finally:
        if tmp is not None:
            os.unlink(tmp)


# ---------------------------------------------------------------------------
# Standard-form conversion (transforms T1-T5, module docstring)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StandardFormLP:
    """min c'x s.t. Ax = b, x >= 0 plus the objective-recovery record."""

    c: np.ndarray  # (n,)
    b: np.ndarray  # (m,)
    A: sp.csr_matrix  # (m, n)
    shift_const: float  # T5 accumulated wc-weighted shifts
    sense_sign: float  # +1 minimize, -1 maximize (T1)
    offset0: float  # HiGHS objective offset
    name: str
    meta: dict = field(default_factory=dict)

    @property
    def m(self) -> int:
        return int(self.b.shape[0])

    @property
    def n(self) -> int:
        return int(self.c.shape[0])

    def original_objective(self, std_val: float) -> float:
        """Map a converted-form objective value back to the original
        model's objective (T1): s * (val + shift_const) + offset0."""
        return float(self.sense_sign * (std_val + self.shift_const) + self.offset0)


def to_standard_form(raw: RawLP) -> StandardFormLP:
    """Apply T1-T5. Raises ValueError on inconsistent bounds."""
    m0, n0 = raw.M.shape
    rl, ru = raw.row_lower.copy(), raw.row_upper.copy()
    lo0, hi0 = raw.col_lower.copy(), raw.col_upper.copy()

    if np.any(rl > ru):
        raise ValueError(f"{raw.name}: row with rl > ru (infeasible)")
    if np.any(lo0 > hi0):
        raise ValueError(f"{raw.name}: variable with lo > hi (infeasible)")

    # T1: sense
    s = -1.0 if raw.maximize else 1.0
    wc = s * raw.c

    rl_inf, ru_inf = _is_inf(rl), _is_inf(ru)
    free_rows = rl_inf & ru_inf  # T2
    eq_rows = (~rl_inf) & (~ru_inf) & (rl == ru)  # T3
    ineq_rows = ~(free_rows | eq_rows)  # T4

    # T4: one slack column per inequality/ranged row, entry -1.
    ineq_idx = np.flatnonzero(ineq_rows)
    k = ineq_idx.size
    S = sp.csc_matrix(
        (-np.ones(k), (ineq_idx, np.arange(k))), shape=(m0, k)
    )
    A1 = sp.hstack([raw.M, S], format="csr")
    b0 = np.where(eq_rows, np.where(rl_inf, 0.0, rl), 0.0)
    lo = np.concatenate([lo0, rl[ineq_idx]])
    hi = np.concatenate([hi0, ru[ineq_idx]])
    wc1 = np.concatenate([wc, np.zeros(k)])

    # T2: drop free rows.
    keep_rows = ~free_rows
    if not np.all(keep_rows):
        A1 = A1[keep_rows]
        b0 = b0[keep_rows]

    # T5: variable bound patterns.
    lo_inf, hi_inf = _is_inf(lo), _is_inf(hi)
    fixed = (~lo_inf) & (~hi_inf) & (lo == hi)  # T5d
    boxed = (~lo_inf) & (~hi_inf) & ~fixed  # T5c
    upper_only = lo_inf & (~hi_inf)  # T5b
    free_vars = lo_inf & hi_inf  # T5e
    # lower-bounded (T5a) = the rest; shift target:
    t = np.where(~lo_inf, lo, 0.0)  # lower/boxed/fixed shift by lo
    t = np.where(upper_only, hi, t)  # reflected vars shift by hi

    b = b0 - A1 @ t
    shift_const = float(wc1 @ t)

    sign = np.where(upper_only, -1.0, 1.0)
    A2 = A1 @ sp.diags(sign)
    c2 = wc1 * sign

    keep_cols = ~fixed  # T5d eliminates fixed columns
    kept_idx = np.flatnonzero(keep_cols)
    A_keep = A2[:, kept_idx].tocsr()
    c_keep = c2[kept_idx]

    # T5e: negated copies of free columns.
    free_idx = np.flatnonzero(free_vars)
    n_free = free_idx.size
    blocks_top = [A_keep]
    c_parts = [c_keep]
    if n_free:
        blocks_top.append((-A2[:, free_idx]).tocsr())
        c_parts.append(-c2[free_idx])

    # T5c: box rows z' + w = hi - lo over kept columns.
    pos_in_kept = -np.ones(lo.size, dtype=np.int64)
    pos_in_kept[kept_idx] = np.arange(kept_idx.size)
    boxed_idx = np.flatnonzero(boxed)
    nb = boxed_idx.size
    n_top = kept_idx.size + n_free
    if nb:
        blocks_top.append(sp.csr_matrix((A_keep.shape[0], nb)))
        c_parts.append(np.zeros(nb))
        E = sp.csr_matrix(
            (np.ones(nb), (np.arange(nb), pos_in_kept[boxed_idx])),
            shape=(nb, n_top),
        )
        box_block = sp.hstack([E, sp.identity(nb, format="csr")], format="csr")
        A_std = sp.vstack(
            [sp.hstack(blocks_top, format="csr"), box_block], format="csr"
        )
        b_std = np.concatenate([b, (hi - lo)[boxed_idx]])
    else:
        A_std = sp.hstack(blocks_top, format="csr") if len(blocks_top) > 1 else A_keep
        b_std = b
    c_std = np.concatenate(c_parts)

    meta = {
        "orig_rows": int(m0),
        "orig_cols": int(n0),
        "orig_nnz": raw.nnz,
        "n_eq_rows": int(np.sum(eq_rows)),
        "n_ineq_rows": int(k),
        "n_free_rows": int(np.sum(free_rows)),
        "n_fixed_vars": int(np.sum(fixed)),
        "n_boxed_vars": int(nb),
        "n_free_vars": int(n_free),
        "n_reflected_vars": int(np.sum(upper_only)),
        "n_integer_relaxed": raw.n_integer,
        "maximize": bool(raw.maximize),
    }
    return StandardFormLP(
        c=c_std,
        b=b_std,
        A=A_std.tocsr(),
        shift_const=shift_const,
        sense_sign=s,
        offset0=raw.offset,
        name=raw.name,
        meta=meta,
    )


def load_standard_form(path: str) -> StandardFormLP:
    """read_mps + to_standard_form in one call."""
    return to_standard_form(read_mps(path))


# ---------------------------------------------------------------------------
# SparseLinOp: jit-safe BCOO matvec with a power-iteration norm bound
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class SparseLinOp:
    """Sparse (m, n) matrix as a LinOp: BCOO matvec forward, pre-built
    transposed BCOO for the adjoint, power-iteration norm bound x1.02.

    eq=False keeps identity hashing so instances work in the kernel-cache
    static_key (same convention as lp_cell.MatrixLinOp)."""

    A: Any  # BCOO (m, n)
    AT: Any  # BCOO (n, m)
    norm_bound: float
    shape: tuple

    @staticmethod
    def from_scipy(A_sp: sp.spmatrix, safety: float = 1.02,
                   power_iters: int | None = None,
                   gpu_threshold: int | None = None) -> "SparseLinOp":
        """power_iters: cap on the power iteration. None keeps lp_cell's default of 500.

        Measured on lasso n_var 50,000 (scripts/discovery/qp_setup_profile.py): 500 iterations cost
        0.327 s and 100 cost 0.067 s, while the resulting ||A|| differs by 5.0e-4 relative. The bound
        is inflated by `safety` = 1.02 anyway, so 100 sits forty times inside the margin already
        applied. 20 is NOT safe (3.5e-2 exceeds the inflation, so the "bound" could fall below the true
        norm). Default is unchanged so no existing caller moves.
        """
        A_csr = A_sp.tocsr().astype(np.float64)
        A_csc = A_csr.tocsc()
        kw = {} if power_iters is None else {"iters": int(power_iters)}
        # gpu_threshold: nnz at or above which the power iteration runs on the GPU. None = never, the
        # historical path. MUST be gated: measured 0.72x (SLOWER) at 290k nonzeros, 12.49x at 5.45M.
        if gpu_threshold is not None and A_csr.nnz >= int(gpu_threshold):
            try:
                import jax as _jax
                if _jax.devices()[0].platform != "cpu":
                    return SparseLinOp(
                        A=jsparse.BCOO.from_scipy_sparse(A_csr),
                        AT=jsparse.BCOO.from_scipy_sparse(A_csr.T.tocsr()),
                        norm_bound=_power_iteration_norm_gpu(
                            A_csr, iters=int(power_iters) if power_iters else 500, safety=safety),
                        shape=tuple(A_csr.shape),
                    )
            except Exception:
                pass                      # no GPU backend: fall through to the numpy path
        nb = power_iteration_norm(
            lambda x: A_csr @ x,
            lambda u: A_csc.T @ u,
            A_csr.shape[1],
            safety=safety,
            **kw,
        )
        return SparseLinOp(
            A=jsparse.BCOO.from_scipy_sparse(A_csr),
            AT=jsparse.BCOO.from_scipy_sparse(A_csr.T.tocsr()),
            norm_bound=float(nb),
            shape=tuple(A_csr.shape),
        )

    def forward(self, x):
        return self._typed(self.A, x.dtype) @ x

    def adjoint(self, u):
        return self._typed(self.AT, u.dtype) @ u

    @staticmethod
    def _typed(matrix, dtype):
        # BCOO.astype goes through sparsify and can introduce expensive
        # duplicate-index sorting into the compiled iteration. Only the
        # values change precision; indices and their invariants are identical.
        if matrix.dtype == dtype:
            return matrix
        return jsparse.BCOO((matrix.data.astype(dtype), matrix.indices),
                           shape=matrix.shape,
                           indices_sorted=matrix.indices_sorted,
                           unique_indices=matrix.unique_indices)


# ---------------------------------------------------------------------------
# Composite adapter (reuses the lp_cell atoms; name == "lp_standard")
# ---------------------------------------------------------------------------


def to_composite(std: StandardFormLP) -> CompositeProblem:
    """StandardFormLP -> lp_standard composite: g = orthant (prox = clip),
    prox_{s h*}(v) = v - s b (AffineEqualitySet), L = sparse matvec with
    the power-iteration norm bound. Dispatches through the same certified
    scheme path as the synthetic LP cell (gap_kind == "lp_gap")."""
    c = jnp.asarray(std.c, dtype=jnp.float64)
    b = jnp.asarray(std.b, dtype=jnp.float64)
    L = SparseLinOp.from_scipy(std.A)
    meta = dict(std.meta)
    meta.update(
        name=std.name,
        rows=std.m,
        cols=std.n,
        nnz=int(std.A.nnz),
        shift_const=std.shift_const,
        sense_sign=std.sense_sign,
        offset0=std.offset0,
    )
    return CompositeProblem(
        f=LinearCost(c=c),
        g=NonnegOrthant(),
        h=AffineEqualitySet(b=b),
        L=L,
        data={"c": c, "b": b, "A_csr": std.A, "meta": meta},
        shape=(std.n,),
        name="lp_standard",
    )


def original_objective_from_meta(meta: dict, std_val: float) -> float:
    """Objective mapping T1 from a composite's data['meta'] record."""
    return float(
        meta["sense_sign"] * (std_val + meta["shift_const"]) + meta["offset0"]
    )


# ---------------------------------------------------------------------------
# Oracles: HiGHS on the original model (cached) + on the converted form
# ---------------------------------------------------------------------------


def oracle_mps(path: str, cache_dir: str = ORACLE_CACHE_MPS_DIR) -> dict:
    """HiGHS solve of the ORIGINAL MPS model, cached by file sha256.

    Returns {"P_star", "status", "rows", "cols", "nnz", "cache_hit",
    "key"}. P_star includes the model's objective offset and sense (it is
    the objective value HiGHS reports for the original model)."""
    import highspy

    key = file_sha256(path)
    os.makedirs(cache_dir, exist_ok=True)
    cpath = os.path.join(cache_dir, f"mps_{key[:32]}.json")
    if os.path.exists(cpath):
        with open(cpath) as f:
            rec = json.load(f)
        rec["cache_hit"] = True
        return rec

    rpath, tmp = _highs_readable_path(path)
    try:
        h = highspy.Highs()
        h.setOptionValue("output_flag", False)
        if h.readModel(rpath) != highspy.HighsStatus.kOk:
            raise ValueError(f"HiGHS could not read {path!r}")
        lp = h.getLp()
        rows, cols, nnz = int(lp.num_row_), int(lp.num_col_), len(lp.a_matrix_.value_)
        # LP relaxation: clear integrality before solving.
        h.changeColsIntegrality(
            cols,
            np.arange(cols, dtype=np.int32),
            np.full(cols, highspy.HighsVarType.kContinuous),
        )
        h.run()
        status = h.getModelStatus()
        rec = {
            "P_star": float(h.getObjectiveValue()),
            "status": h.modelStatusToString(status),
            "optimal": status == highspy.HighsModelStatus.kOptimal,
            "rows": rows,
            "cols": cols,
            "nnz": int(nnz),
            "key": key,
            "cache_hit": False,
        }
    finally:
        if tmp is not None:
            os.unlink(tmp)
    with open(cpath, "w") as f:
        json.dump(rec, f)
    return rec


def solve_standard_highs(std: StandardFormLP) -> dict:
    """HiGHS (scipy linprog, sparse) solve of the CONVERTED standard form.
    Independent verification of T1-T5: original_objective(fun) must match
    oracle_mps on the source file (tested to 1e-6 relative)."""
    res = linprog(
        std.c,
        A_eq=std.A,
        b_eq=std.b,
        bounds=(0, None),
        method="highs",
    )
    if res.status != 0:
        raise RuntimeError(
            f"HiGHS failed on converted {std.name!r} "
            f"(status {res.status}): {res.message}"
        )
    return {
        "std_fun": float(res.fun),
        "P_star_original_scale": std.original_objective(res.fun),
        "x": res.x,
    }


# ---------------------------------------------------------------------------
# Manifest-driven instance building
# ---------------------------------------------------------------------------


def load_manifest(manifest_path: str = MANIFEST_PATH) -> list[dict]:
    with open(manifest_path) as f:
        doc = json.load(f)
    return doc["instances"] if isinstance(doc, dict) else doc


def build_instances(
    manifest_filter: Optional[Callable[[dict], bool] | Sequence[str]] = None,
    *,
    manifest_path: str = MANIFEST_PATH,
    data_root: str = DATA_ROOT,
    max_nnz: Optional[int] = None,
    max_instances: Optional[int] = None,
) -> list[CompositeProblem]:
    """lp_standard composites from the fetched suites' MANIFEST.json.

    manifest_filter: a callable(entry) -> bool over manifest entries, or a
    sequence of instance names, or None (all usable entries). max_nnz caps
    by ORIGINAL-model nonzeros (manifest field); entries marked unusable
    are always skipped. Entries are processed in manifest order.

    Each composite's data["meta"] carries the manifest entry (source,
    sha256, holdout_tier) plus the conversion record, so
    original_objective_from_meta() maps solver objectives back to the
    original model's scale for oracle comparison.
    """
    entries = load_manifest(manifest_path)
    if manifest_filter is None:
        pred = lambda e: True  # noqa: E731
    elif callable(manifest_filter):
        pred = manifest_filter
    else:
        wanted = set(manifest_filter)
        pred = lambda e: e["name"] in wanted  # noqa: E731

    out: list[CompositeProblem] = []
    for e in entries:
        if not e.get("usable", False):
            continue
        if max_nnz is not None and e["nnz"] > max_nnz:
            continue
        if not pred(e):
            continue
        path = os.path.join(data_root, e["file"])
        std = load_standard_form(path)
        prob = to_composite(std)
        prob.data["meta"].update(
            source=e.get("source"),
            sha256=e.get("sha256"),
            holdout_tier=e.get("holdout_tier"),
            file=e["file"],
        )
        out.append(prob)
        if max_instances is not None and len(out) >= max_instances:
            break
    return out


def group_by_shape(problems: Iterable[CompositeProblem]) -> dict[tuple, list]:
    """Group lp_standard composites by CONVERTED (rows, cols) for the
    same-shape batched backend (vmap needs identical shapes). Real MPS
    instances rarely coincide, so expect mostly singleton groups; the
    synthetic families are the same-shape batching workhorse."""
    groups: dict[tuple, list] = {}
    for p in problems:
        m = int(p.data["b"].shape[0])
        n = int(p.shape[0])
        groups.setdefault((m, n), []).append(p)
    return groups


def _power_iteration_norm_gpu(A_csr, iters: int, tol: float = 1e-13, seed: int = 0,
                              safety: float = 1.02) -> float:
    """||A||_2 by power iteration on the GPU, mirroring lp_cell.power_iteration_norm exactly.

    Same seed, same recurrence, same tolerance test, so the result matches the numpy path (measured
    bit-identical, relative difference 0.0, on lasso at n_var 50,000 and 250,000).

    Defined at module level AFTER the class on purpose: an earlier attempt spliced it into the class
    body, which dedented out of `class SparseLinOp` and silently removed `from_scipy` from the class.
    `from_scipy` resolves this name at call time, so definition order does not matter.
    """
    import jax.numpy as jnp

    Aj = jsparse.BCOO.from_scipy_sparse(A_csr)
    ATj = jsparse.BCOO.from_scipy_sparse(A_csr.T.tocsr())
    rng = np.random.default_rng(seed)
    v = jnp.asarray(rng.standard_normal(A_csr.shape[1]))
    nv = float(jnp.linalg.norm(v))
    if nv == 0.0:
        raise RuntimeError("power iteration: degenerate start vector")
    v = v / nv
    sigma = 0.0
    for _ in range(int(iters)):
        w = Aj @ v
        s_new = float(jnp.linalg.norm(w))
        if s_new == 0.0:
            return 0.0
        z = ATj @ (w / s_new)
        nz = float(jnp.linalg.norm(z))
        if nz == 0.0:
            break
        v = z / nz
        if abs(s_new - sigma) <= tol * max(1.0, s_new):
            sigma = s_new
            break
        sigma = s_new
    return float(safety * sigma)
