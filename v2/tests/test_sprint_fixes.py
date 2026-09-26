"""Regression tests for the 2026-08-26 sprint adversarial-verification fixes.

Covers:
1. avg blends of drs/dys parents with different tau are REJECTED (their
   fixed-point encoding depends on the resolvent step, so different-tau
   parents share no common fixed point; the old gate certified a genome
   that provably could not certify at runtime).
2. The precision-policy execution path caches its compiled chunk runner
   across solves (km-path parity) and reports wall_time EXCLUDING compile.
3. time_to_tol honors the precision path's gap_iters cadence instead of
   assuming sample_every (which understated time by K/sample_every).
4. The lp_netflow Stage-0 gate constant OPNORM_LP is sound only for
   degree-2 regular netflow; validate(..., lp_opnorm=gate_opnorm(instances))
   is the sound gate for other families.
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pytest

from atlas.bench.lp_cell import build_instances, gate_opnorm
from atlas.bench.metrics import time_to_tol
from atlas.genome import OPNORM_LP, genome_from_dict, validate
from atlas.ir.spec import lasso_box_problem, tv_denoise_problem
from atlas.schemes.compile import solve_genome
from atlas.typing_ import TypeCheckError
from atlas.wrappers.km import Budget


# ---------------------------------------------------------------------------
# 1. drs/dys avg blends need identical tau
# ---------------------------------------------------------------------------


def _avg(scheme, tau_a, tau_b, lam=0.5):
    return genome_from_dict(
        {
            "scheme": "mix",
            "mix": {
                "kind": "avg",
                "lambda": lam,
                "scheme_a": {"scheme": scheme, "params": {"tau": tau_a}},
                "scheme_b": {"scheme": scheme, "params": {"tau": tau_b}},
            },
        }
    )


def test_gate_rejects_mixed_tau_dys_avg():
    v = validate(_avg("dys", 0.1, 1.8), problem_kind="three_composite")
    assert not v.ok
    assert "IDENTICAL tau" in v.error


def test_gate_rejects_mixed_tau_drs_avg():
    v = validate(_avg("drs", 0.05, 2.0), problem_kind="tv_denoise")
    assert not v.ok
    assert "IDENTICAL tau" in v.error


def test_gate_accepts_same_tau_drs_avg_and_free_step_fbs_avg():
    assert validate(_avg("drs", 0.7, 0.7), problem_kind="tv_denoise").ok
    # fbs fixed points are step-independent: stepsizes stay free
    assert validate(_avg("fbs", 0.05, 0.2), problem_kind="tv_denoise").ok


def test_solve_path_rejects_mixed_tau_drs_avg():
    y = np.random.default_rng(0).uniform(0.0, 1.0, (16, 16))
    problem = tv_denoise_problem(y, lam=0.1)
    with pytest.raises(TypeCheckError, match="identical tau"):
        solve_genome(problem, _avg("drs", 0.05, 2.0), Budget(max_iters=200))


def test_same_tau_dys_avg_still_certifies_at_runtime():
    y = np.random.default_rng(1).uniform(0.0, 1.0, (10, 10))
    problem = lasso_box_problem(y, lam=0.15, lo=0.0, hi=1.0)
    g = _avg("dys", 0.9, 0.9)
    assert validate(g, problem_kind="three_composite").ok
    res = solve_genome(problem, g, Budget(max_iters=20_000, tol=1e-6))
    assert float(res.rel_gap_history[-1]) <= 1e-6


# ---------------------------------------------------------------------------
# 2. precision-policy chunk cache + wall excludes compile
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tv16():
    y = np.random.default_rng(2).uniform(0.0, 1.0, (16, 16))
    return tv_denoise_problem(y, lam=0.1)


def _cv(backend):
    return genome_from_dict(
        {
            "scheme": "condat_vu",
            "params": {"tau": 0.5, "sigma": 0.186},
            "backend": backend,
        }
    )


def test_policy_path_caches_chunk_compile_across_solves(tv16):
    from atlas.backend import precision

    budget = Budget(max_iters=4000, tol=1e-4, sample_every=100)
    g = _cv({"precision": "f64", "K": 100})
    precision.clear_chunk_cache()
    r1 = solve_genome(tv16, g, budget)
    n_after_first = precision.chunk_compile_count()
    assert n_after_first >= 1
    assert r1.extra["wall_includes_compile"] is False
    assert r1.extra["compile_time_s"] > 0.0
    r2 = solve_genome(tv16, g, budget)
    assert precision.chunk_compile_count() == n_after_first  # cache hit
    assert r2.extra["compile_time_s"] == 0.0
    assert float(r2.rel_gap_history[-1]) <= 1e-4
    # a DIFFERENT stepsize genome must also hit the cache (dyn is dynamic)
    g3 = _cv({"precision": "f64", "K": 100})
    g3 = genome_from_dict(
        {
            "scheme": "condat_vu",
            "params": {"tau": 0.3, "sigma": 0.2},
            "backend": {"precision": "f64", "K": 100},
        }
    )
    r3 = solve_genome(tv16, g3, budget)
    assert precision.chunk_compile_count() == n_after_first
    assert float(r3.rel_gap_history[-1]) <= 1e-4


def test_policy_path_different_stepsizes_give_different_solves(tv16):
    # cache-correctness guard: dynamic dyn must actually change the math
    budget = Budget(max_iters=2000, tol=1e-12, sample_every=100)
    from atlas.backend import precision

    precision.clear_chunk_cache()
    r_a = solve_genome(tv16, _cv({"precision": "f64", "K": 500}), budget)
    r_b = solve_genome(
        tv16,
        genome_from_dict(
            {
                "scheme": "condat_vu",
                "params": {"tau": 0.1, "sigma": 0.5},
                "backend": {"precision": "f64", "K": 500},
            }
        ),
        budget,
    )
    # same executable (one compile), different trajectories
    assert precision.chunk_compile_count() == 1
    assert not np.allclose(r_a.x, r_b.x, atol=0) or not np.array_equal(
        r_a.rel_gap_history, r_b.rel_gap_history
    )


def test_policy_and_km_paths_agree_on_solution(tv16):
    budget = Budget(max_iters=8000, tol=1e-6, sample_every=100)
    g_km = genome_from_dict(
        {"scheme": "condat_vu", "params": {"tau": 0.5, "sigma": 0.186}}
    )
    r_km = solve_genome(tv16, g_km, budget)
    r_pol = solve_genome(tv16, _cv({"precision": "f64", "K": 100}), budget)
    # both certify the same 1e-6 duality gap; iterates may differ slightly
    # because the two paths stop on different iteration grids
    assert float(r_km.rel_gap_history[-1]) <= 1e-6
    assert float(r_pol.rel_gap_history[-1]) <= 1e-6
    assert np.allclose(r_km.x, r_pol.x, atol=1e-3)


# ---------------------------------------------------------------------------
# 3. time_to_tol honors the policy path's gap_iters cadence
# ---------------------------------------------------------------------------


class _FakeResult:
    def __init__(self, history, iters, wall, gap_iters=None):
        self.rel_gap_history = np.asarray(history, dtype=float)
        self.iters = iters
        self.wall_time = wall
        self.extra = (
            {"backend": {"gap_iters": list(gap_iters)}} if gap_iters else {}
        )


def test_time_to_tol_uses_backend_gap_iters_cadence():
    # policy genome, K=500, certifies at its first (and only) check
    r = _FakeResult(history=[5e-5], iters=500, wall=0.061, gap_iters=[500])
    t, iter_at = time_to_tol(r, tol=1e-4, sample_every=100)
    assert iter_at == 500  # not 100
    assert t == pytest.approx(0.061)  # not 0.0122 (the old 5x underestimate)


def test_time_to_tol_km_cadence_unchanged():
    r = _FakeResult(history=[1e-2, 1e-3, 5e-5], iters=300, wall=0.03)
    t, iter_at = time_to_tol(r, tol=1e-4, sample_every=100)
    assert iter_at == 300
    assert t == pytest.approx(0.03)


def test_time_to_tol_policy_solve_end_to_end(tv16):
    budget = Budget(max_iters=4000, tol=1e-4, sample_every=100)
    r = solve_genome(tv16, _cv({"precision": "f64", "K": 500}), budget)
    assert float(r.rel_gap_history[-1]) <= 1e-4
    t, iter_at = time_to_tol(r, tol=1e-4, sample_every=100)
    # crossing iteration must be a multiple of K=500, never of the
    # sample_every=100 grid alone
    assert iter_at % 500 == 0


# ---------------------------------------------------------------------------
# 4. lp gate constant scope + gate_opnorm override
# ---------------------------------------------------------------------------


def _near_max_pdhg():
    ts = 0.99 / (OPNORM_LP**2)  # t*s*OPNORM_LP^2 = 0.99 < 1
    return genome_from_dict(
        {
            "scheme": "pdhg",
            "params": {"tau": ts**0.5, "sigma": ts**0.5},
        }
    )


def test_degree4_norm_exceeds_gate_constant_and_override_rejects():
    train, holdout = build_instances(
        n_train=1, n_holdout=1, family="regular", size=40, seed=7, degree=4
    )
    opn = gate_opnorm(train + holdout)
    assert opn > OPNORM_LP  # the footgun: default degree-4 breaks the constant
    g = _near_max_pdhg()
    assert validate(g, problem_kind="lp_netflow").ok  # old gate: passes
    v = validate(g, problem_kind="lp_netflow", lp_opnorm=opn)
    assert not v.ok  # sound gate: rejected before solve time


def test_degree2_family_still_covered_by_gate_constant():
    train, holdout = build_instances(
        n_train=2, n_holdout=1, family="regular", size=40, seed=7, degree=2
    )
    assert gate_opnorm(train + holdout) <= OPNORM_LP + 1e-12
