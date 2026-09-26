"""Numerical contract tests for the hardware-aware execution path."""
from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest

from atlas.backend.hardware import context_block, operator_profile
from atlas.backend.precision import F64Policy, SchedulePolicy
from atlas.certify.residuals import lp_rel_gap, rel_gap
from atlas.genome import Backend, Genome, GenomeParams, Wrapper, Metric
from atlas.ir.spec import tv_denoise_problem
from atlas.schemes.compile import solve_genome
from atlas.wrappers.km import Budget


def tv(seed=19):
    return tv_denoise_problem(np.random.default_rng(seed).random((12, 12)), .1)


@pytest.mark.parametrize("halpern", [False, True])
def test_cadence_preserves_fixed_iteration_trajectory_and_partial_cap(halpern):
    p = tv()
    g = Genome("pdhg", GenomeParams(tau=.2, sigma=.4, rho=1.5),
               Wrapper(halpern=halpern, restart="fixed_k" if halpern else "none",
                       restart_k=7 if halpern else None))
    b = Budget(max_iters=37, tol=0, sample_every=1)
    baseline = solve_genome(p, g, b)
    for k in (1, 8, 100):
        r = solve_genome(p, replace(g, backend=Backend(K=k)), b)
        assert r.iters == 37
        np.testing.assert_allclose(r.x, baseline.x, atol=2e-14, rtol=2e-14)
        np.testing.assert_allclose(r.u, baseline.u, atol=2e-14, rtol=2e-14)
        expected = list(range(k, 37, k)) + [37]
        assert r.extra["backend"]["gap_iters"] == expected
        assert r.extra["backend"]["certificate_checks"] == len(expected)


def test_cache_does_not_reuse_original_certificate_data_from_previous_instance():
    # TV problems of the same shape intentionally share a compiled executable.
    g = Genome("condat_vu", GenomeParams(tau=.05, sigma=1., rho=1.9),
               backend=Backend(K=17))
    for p in (tv(1), tv(2)):
        r = solve_genome(p, g, Budget(max_iters=3000, tol=1e-6))
        independent = float(rel_gap(jnp.asarray(r.x), jnp.asarray(r.u),
                                    p.f.y, .1, p.L))
        assert independent == pytest.approx(r.rel_gap_history[-1], abs=2e-14)
        assert independent <= 1e-6


@pytest.mark.parametrize("precision", ["f64", "f32_cert64", "schedule"])
def test_metric_policy_checks_original_coordinates(precision):
    from atlas.bench.lp_cell import gen_lp_dense, to_composite
    p = to_composite(gen_lp_dense(12, 5, 7))
    g = Genome("pdhg", GenomeParams(tau=.7, sigma=.7, rho=1.5),
               Wrapper(halpern=True, restart="fixed_k", restart_k=7),
               backend=Backend(precision=precision, K=8,
                               theta=.01 if precision == "schedule" else None),
               metric=Metric(kind="ruiz_pc", iters=10, alpha=1.))
    r = solve_genome(p, g, Budget(max_iters=37, tol=1e-12))
    expected = float(lp_rel_gap(jnp.asarray(r.x), jnp.asarray(r.u),
                                p.data["c"], p.L, p.data["b"]))
    assert r.iters <= 37
    assert r.rel_gap_history[-1] == pytest.approx(expected, abs=2e-12)
    if precision == "schedule":
        assert r.extra["backend"]["switch_iter"] is not None


def test_legacy_phase_does_not_overrun_budget():
    def step(dtype):
        return lambda s, k: s + jnp.asarray(1., dtype)
    init = lambda dtype: jnp.asarray(0., dtype)
    gap = lambda s: jnp.asarray(1., jnp.float64)
    r = F64Policy(K=8).run(step, init, gap, 0., 19)
    assert r.iters == 19
    assert float(r.state) == 19
    assert r.gap_iters.tolist() == [8, 16, 19]


def test_hardware_profile_and_context_conditions(tmp_path):
    p = tv()
    g = Genome("pdhg", GenomeParams(tau=.2, sigma=.4))
    profile = operator_profile(p, g, repeats=2, block_iters=3, hlo_dir=tmp_path)
    assert profile["operators"]["certificate_f64"]["median_us"] > 0
    assert profile["operators"]["certificate_f64"]["gpu_launch_count"] is None
    assert (tmp_path / "certificate_f64.hlo.txt").exists()
    assert context_block(profile, "none") is None
    assert "operator_costs_us" not in context_block(profile, "identity")
    assert "operator_costs_us" in context_block(profile, "measured")


def test_llm_fallback_stream_stays_decoupled_after_reseed():
    from atlas.evolve.llm_backend import LLMBackend
    from atlas.evolve.random_backend import RandomBackend
    from atlas.genome import content_hash
    be = LLMBackend(seed=42, http_post=lambda _: "invalid")
    be.reseed(42)
    random_control = RandomBackend(seed=43)
    assert [content_hash(g) for g in be.ask(4)] != [
        content_hash(g) for g in random_control.ask(4)]
