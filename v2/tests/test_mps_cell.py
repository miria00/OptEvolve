"""Tests for atlas.bench.mps_cell: MPS -> standard-form conversion
(each transform T1-T5 on handcrafted files + objective equivalence vs
HiGHS on >= 10 real instances), the SparseLinOp, the composite adapter,
and the PDHG sanity solve on small Mittelmann instances.

Real-instance tests skip cleanly when the suites have not been fetched
(scripts/fetch_lp_suites.sh); the handcrafted-MPS tests always run.
"""
from __future__ import annotations

import gzip
import os
import textwrap

import numpy as np
import pytest
import scipy.sparse as sp

from atlas.bench import mps_cell as mc
from atlas.bench.lp_cell import gate_opnorm, lp_gap_terms
from atlas.genome import Genome, GenomeParams
from atlas.schemes.base_schemes import _gap_kind
from atlas.schemes.compile import solve_genome
from atlas.wrappers.km import Budget

HAVE_SUITES = os.path.exists(mc.MANIFEST_PATH)
needs_suites = pytest.mark.skipif(
    not HAVE_SUITES,
    reason="LP suites not fetched (run scripts/fetch_lp_suites.sh)",
)


def _write_mps(tmp_path, text, name="prob.mps", gz=False):
    body = textwrap.dedent(text).lstrip("\n")
    if gz:
        p = tmp_path / (name + ".gz")
        with gzip.open(p, "wt") as f:
            f.write(body)
    else:
        p = tmp_path / name
        p.write_text(body)
    return str(p)


def _roundtrip(path, atol=1e-9):
    """Converted-form optimum mapped back == HiGHS on the original."""
    std = mc.load_standard_form(path)
    ref = mc.solve_standard_highs(std)
    orc = mc.oracle_mps(path, cache_dir=os.path.join(os.path.dirname(path), "cache"))
    assert orc["optimal"], orc["status"]
    got, want = ref["P_star_original_scale"], orc["P_star"]
    assert got == pytest.approx(want, rel=1e-6, abs=atol), (
        f"{os.path.basename(path)}: converted {got} vs original {want}"
    )
    return std


# ---------------------------------------------------------------------------
# Handcrafted MPS: one file per transform family
# ---------------------------------------------------------------------------

MPS_BASIC = """
NAME          BASIC
ROWS
 N  COST
 L  LIM1
 G  LIM2
 E  EQ1
COLUMNS
    X1        COST         1.0   LIM1         1.0
    X1        LIM2         1.0
    X2        COST         2.0   LIM1         1.0
    X2        EQ1          1.0
    X3        COST        -1.0   LIM2         1.0
    X3        EQ1          1.0
RHS
    RHS       LIM1         4.0   LIM2         1.0
    RHS       EQ1          7.0
BOUNDS
 UP BND       X3           4.0
ENDATA
"""

MPS_RANGES = """
NAME          RANGED
ROWS
 N  COST
 L  R1
 G  R2
 E  R3
COLUMNS
    X1        COST         1.0   R1           1.0
    X1        R2           1.0   R3           1.0
    X2        COST         1.5   R1           2.0
    X2        R2          -1.0   R3           1.0
RHS
    RHS       R1          10.0   R2           1.0
    RHS       R3           5.0
RANGES
    RNG       R1           4.0   R2           3.0
    RNG       R3           2.0
ENDATA
"""

MPS_BOUNDS = """
NAME          BNDS
ROWS
 N  COST
 E  E1
 L  L1
COLUMNS
    XFREE     COST         1.0   E1           1.0
    XNEG      COST         0.5   E1           1.0
    XNEG      L1           1.0
    XBOX      COST        -1.0   E1           1.0
    XBOX      L1           2.0
    XFIX      COST         5.0   E1           1.0
RHS
    RHS       E1           3.0   L1           8.0
BOUNDS
 FR BND       XFREE
 MI BND       XNEG
 UP BND       XNEG         2.0
 LO BND       XBOX        -1.0
 UP BND       XBOX         3.0
 FX BND       XFIX         0.5
ENDATA
"""

MPS_MAX = """
NAME          MAXPROB
OBJSENSE
    MAX
ROWS
 N  PROFIT
 L  CAP1
 L  CAP2
COLUMNS
    X1        PROFIT       3.0   CAP1         1.0
    X1        CAP2         2.0
    X2        PROFIT       5.0   CAP1         2.0
    X2        CAP2         1.0
RHS
    RHS       CAP1        14.0   CAP2        10.0
ENDATA
"""

MPS_INT = """
NAME          RELAX
ROWS
 N  COST
 G  C1
COLUMNS
    MARKER                 'MARKER'                 'INTORG'
    X1        COST         2.0   C1           3.0
    MARKER                 'MARKER'                 'INTEND'
    X2        COST         3.0   C1           2.0
RHS
    RHS       C1           7.0
ENDATA
"""


class TestHandcraftedConversion:
    def test_basic_l_g_e_rows_and_ub(self, tmp_path):
        std = _roundtrip(_write_mps(tmp_path, MPS_BASIC))
        # 1 E row + 2 slacked ineq rows + 1 box row (X3's UP bound... the
        # L/G slacks are one-sided, X3 boxed [0,4]).
        assert std.meta["n_eq_rows"] == 1
        assert std.meta["n_ineq_rows"] == 2
        assert std.meta["n_boxed_vars"] >= 1

    def test_ranges_on_l_g_e_rows(self, tmp_path):
        std = _roundtrip(_write_mps(tmp_path, MPS_RANGES))
        # every ranged row's slack is boxed (finite on both sides)
        assert std.meta["n_ineq_rows"] == 3
        assert std.meta["n_boxed_vars"] == 3

    def test_bound_kinds_fr_mi_box_fx(self, tmp_path):
        std = _roundtrip(_write_mps(tmp_path, MPS_BOUNDS))
        assert std.meta["n_free_vars"] == 1  # FR -> split (T5e)
        assert std.meta["n_fixed_vars"] == 1  # FX -> eliminated (T5d)
        assert std.meta["n_reflected_vars"] >= 1  # MI+UP -> reflect (T5b)
        assert std.meta["n_boxed_vars"] >= 1  # LO+UP -> box row (T5c)

    def test_maximization_sense(self, tmp_path):
        std = _roundtrip(_write_mps(tmp_path, MPS_MAX))
        assert std.sense_sign == -1.0
        # max 3x1+5x2 s.t. x1+2x2<=14, 2x1+x2<=10 -> optimum at (2,6) = 36
        ref = mc.solve_standard_highs(std)
        assert ref["P_star_original_scale"] == pytest.approx(36.0, abs=1e-9)

    def test_integer_markers_are_relaxed(self, tmp_path):
        path = _write_mps(tmp_path, MPS_INT)
        raw = mc.read_mps(path)
        assert raw.n_integer == 1
        std = _roundtrip(path)  # oracle also clears integrality
        # MPS convention (HiGHS follows it): INTORG columns without
        # explicit bounds default to [0, 1]. LP relaxation: min 2x1+3x2,
        # 3x1+2x2 >= 7, x1 <= 1 -> x1=1, x2=2, objective 8.
        ref = mc.solve_standard_highs(std)
        assert ref["P_star_original_scale"] == pytest.approx(8.0, abs=1e-9)

    def test_gzip_input(self, tmp_path):
        _roundtrip(_write_mps(tmp_path, MPS_BASIC, gz=True))

    def test_standard_form_is_standard(self, tmp_path):
        """The converted form really is min c'x, Ax=b, x>=0: solving it
        with explicit nonnegativity-only bounds is what
        solve_standard_highs does; here we check feasibility of its
        solution under the equality constraints."""
        std = mc.load_standard_form(_write_mps(tmp_path, MPS_RANGES))
        ref = mc.solve_standard_highs(std)
        x = ref["x"]
        assert np.all(x >= -1e-9)
        assert np.linalg.norm(std.A @ x - std.b) < 1e-7


# ---------------------------------------------------------------------------
# SparseLinOp
# ---------------------------------------------------------------------------


class TestSparseLinOp:
    def test_forward_adjoint_and_norm_bound(self):
        rng = np.random.default_rng(0)
        A = sp.random(23, 41, density=0.2, random_state=rng, format="csr")
        op = mc.SparseLinOp.from_scipy(A)
        x = rng.standard_normal(41)
        u = rng.standard_normal(23)
        np.testing.assert_allclose(np.asarray(op.forward(x)), A @ x, atol=1e-12)
        np.testing.assert_allclose(np.asarray(op.adjoint(u)), A.T @ u, atol=1e-12)
        # adjoint identity <Ax, u> == <x, A'u>
        assert float(np.dot(np.asarray(op.forward(x)), u)) == pytest.approx(
            float(np.dot(x, np.asarray(op.adjoint(u)))), rel=1e-12
        )
        true_norm = np.linalg.svd(A.toarray(), compute_uv=False)[0]
        assert op.norm_bound >= true_norm  # x1.02 safety upper-bounds
        assert op.norm_bound <= 1.05 * true_norm


# ---------------------------------------------------------------------------
# Composite adapter
# ---------------------------------------------------------------------------


class TestComposite:
    def test_lp_gap_dispatch_and_certificate_at_optimum(self, tmp_path):
        std = mc.load_standard_form(_write_mps(tmp_path, MPS_BASIC))
        prob = mc.to_composite(std)
        assert prob.name == "lp_standard"
        assert _gap_kind(prob) == "lp_gap"
        assert gate_opnorm([prob]) == pytest.approx(prob.L.norm_bound)
        # at the HiGHS primal-dual optimum the lp gap certificate ~ 0
        from scipy.optimize import linprog

        res = linprog(std.c, A_eq=std.A, b_eq=std.b, bounds=(0, None), method="highs")
        u = -np.asarray(res.eqlin.marginals)
        terms = lp_gap_terms(res.x, u, std.c, prob.L, std.b)
        if terms["cert"] > 1e-7:  # scipy dual sign has flipped across versions
            terms = lp_gap_terms(res.x, -u, std.c, prob.L, std.b)
        assert terms["cert"] < 1e-7

    def test_objective_mapping_from_meta(self, tmp_path):
        std = mc.load_standard_form(_write_mps(tmp_path, MPS_MAX))
        prob = mc.to_composite(std)
        v = 123.456
        assert mc.original_objective_from_meta(
            prob.data["meta"], v
        ) == pytest.approx(std.original_objective(v))

    def test_group_by_shape(self, tmp_path):
        std = mc.load_standard_form(_write_mps(tmp_path, MPS_BASIC))
        p1 = mc.to_composite(std)
        p2 = mc.to_composite(std)
        std2 = mc.load_standard_form(_write_mps(tmp_path, MPS_RANGES, name="r.mps"))
        p3 = mc.to_composite(std2)
        groups = mc.group_by_shape([p1, p2, p3])
        assert len(groups) == 2
        assert sorted(len(v) for v in groups.values()) == [1, 2]


# ---------------------------------------------------------------------------
# Real suites (skip when not fetched)
# ---------------------------------------------------------------------------


def _usable(entries, source=None, tier=None, max_nnz=None):
    out = [e for e in entries if e.get("usable")]
    if source:
        out = [e for e in out if source in e["source"]]
    if tier:
        out = [e for e in out if e.get("holdout_tier") == tier]
    if max_nnz:
        out = [e for e in out if e["nnz"] <= max_nnz]
    return out


@needs_suites
class TestRealSuites:
    def test_manifest_counts(self):
        entries = mc.load_manifest()
        mitt = _usable(entries, source="mittelmann")
        mip = _usable(entries, source="miplib")
        assert len(mitt) >= 15, f"only {len(mitt)} usable Mittelmann instances"
        assert len(mip) >= 40, f"only {len(mip)} usable MIPLIB instances"
        for e in mitt + mip:
            for k in ("name", "file", "rows", "cols", "nnz", "sha256", "holdout_tier"):
                assert e.get(k) is not None, (e["name"], k)

    def test_conversion_matches_highs_on_10_real_instances(self):
        """The core equivalence check: converted standard form optimum
        (mapped by T1) == HiGHS on the original file, 1e-6 relative, on
        >= 10 real instances spanning both suites."""
        entries = mc.load_manifest()
        picks = (
            _usable(entries, source="mittelmann", max_nnz=200_000)[:5]
            + _usable(entries, source="miplib", max_nnz=200_000)[:7]
        )
        assert len(picks) >= 10
        checked = 0
        for e in picks:
            path = os.path.join(mc.DATA_ROOT, e["file"])
            std = mc.load_standard_form(path)
            orc = mc.oracle_mps(path)
            if not orc["optimal"]:  # e.g. unbounded relaxation: no optimum
                continue
            ref = mc.solve_standard_highs(std)
            got, want = ref["P_star_original_scale"], orc["P_star"]
            assert got == pytest.approx(want, rel=1e-6, abs=1e-6), e["name"]
            checked += 1
        assert checked >= 10

    def test_build_instances_filter_and_meta(self):
        probs = mc.build_instances(max_nnz=50_000, max_instances=3)
        assert 1 <= len(probs) <= 3
        for p in probs:
            assert p.name == "lp_standard"
            assert _gap_kind(p) == "lp_gap"
            meta = p.data["meta"]
            assert meta["holdout_tier"] in ("small", "medium", "large")
            assert p.L.norm_bound > 0
        names = [p.data["meta"]["name"] for p in probs]
        sub = mc.build_instances(manifest_filter=names[:1], max_nnz=50_000)
        assert len(sub) == 1

    def test_pdhg_sanity_three_small_mittelmann(self):
        """Deliverable 4: solve 3 small Mittelmann instances with the
        existing certified PDHG (solve_genome) at rel gap 1e-4 and
        compare the objective to HiGHS on the original model."""
        entries = mc.load_manifest()
        small = _usable(entries, source="mittelmann", max_nnz=5_000)
        small.sort(key=lambda e: e["nnz"])
        assert len(small) >= 3, "need 3 small Mittelmann instances"
        solved = 0
        for e in small:
            if solved == 3:
                break
            path = os.path.join(mc.DATA_ROOT, e["file"])
            orc = mc.oracle_mps(path)
            if not orc["optimal"]:
                continue
            std = mc.load_standard_form(path)
            prob = mc.to_composite(std)
            nb = prob.L.norm_bound
            g = Genome(
                scheme="pdhg",
                params=GenomeParams(tau=0.9 / nb, sigma=0.9 / nb),
            )
            res = solve_genome(
                prob, g, Budget(max_iters=400_000, tol=1e-4, sample_every=200)
            )
            gaps = np.asarray(res.rel_gap_history)
            final_gap = float(gaps[np.isfinite(gaps)][-1])
            assert final_gap <= 1e-4, (e["name"], final_gap)
            x = np.maximum(np.asarray(res.x), 0.0)
            obj = mc.original_objective_from_meta(
                prob.data["meta"], float(std.c @ x)
            )
            want = orc["P_star"]
            # rel-gap 1e-4 certificate bounds the objective error at the
            # same order; allow a small multiple for feasibility rounding.
            assert obj == pytest.approx(want, rel=5e-3, abs=5e-3), (
                e["name"], obj, want, final_gap,
            )
            solved += 1
        assert solved == 3
