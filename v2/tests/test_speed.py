"""Speed levers: dynamic-parameter kernels (ONE compile per scheme per
shape), process-parallel fitness evaluation, persistent compilation cache
configuration."""
import os

import numpy as np
import pytest

from atlas import (
    Budget,
    CondatVuParams,
    FBSParams,
    PDHGParams,
    solve_condat_vu,
    solve_fbs,
    solve_pdhg,
    tv_denoise_problem,
)
from atlas.wrappers.km import clear_kernel_cache, kernel_compile_count


def _problem(tv_instance_32):
    y, lam = tv_instance_32
    return tv_denoise_problem(y, lam)


# ---------------------------------------------------------------------------
# Lever 1: dynamic stepsize params -> one compile per scheme per shape
# ---------------------------------------------------------------------------


class TestKernelCache:
    def test_one_compile_per_scheme_per_shape(self, tv_instance_32):
        p = _problem(tv_instance_32)
        b = Budget(max_iters=2_000, tol=1e-4, sample_every=100)
        clear_kernel_cache()
        # 5 different genomes of the same scheme on the same shape
        for t, s in [(0.5, 0.99 / 4), (0.1, 0.9), (0.05, 1.0), (0.2, 0.5), (0.8, 0.12)]:
            solve_pdhg(p, PDHGParams(t=t, s=s), b)
        assert kernel_compile_count() == 1
        # over-relaxation rides the SAME kernel (rho is a dynamic argument)
        solve_pdhg(p, PDHGParams(t=0.5, s=0.99 / 4, rho=1.5), b)
        assert kernel_compile_count() == 1
        # halpern is a structurally different loop: exactly one more compile
        solve_pdhg(p, PDHGParams(t=0.5, s=0.99 / 4, halpern=True), b)
        solve_pdhg(p, PDHGParams(t=0.1, s=0.9, halpern=True), b)
        solve_pdhg(p, PDHGParams(t=0.1, s=0.9, halpern=True, restart_k=50), b)
        assert kernel_compile_count() == 2
        # a second problem instance with the SAME shape reuses everything
        rng = np.random.default_rng(9)
        p2 = tv_denoise_problem(rng.standard_normal((32, 32)), 0.07)
        solve_pdhg(p2, PDHGParams(t=0.3, s=0.3), b)
        assert kernel_compile_count() == 2
        # another scheme compiles its own kernel exactly once
        solve_condat_vu(p, CondatVuParams(t=0.5, s=0.05), b)
        solve_condat_vu(p, CondatVuParams(t=0.1, s=0.5), b)
        assert kernel_compile_count() == 3

    def test_dynamic_params_change_results_correctly(self, tv_instance_32):
        """Guard against stale-constant bugs: two different stepsizes through
        the SAME cached kernel must give genuinely different, correct runs."""
        p = _problem(tv_instance_32)
        b = Budget(max_iters=100_000, tol=1e-8, sample_every=100)
        clear_kernel_cache()
        r1 = solve_fbs(p, FBSParams(step=0.24), b)
        r2 = solve_fbs(p, FBSParams(step=0.01), b)
        assert kernel_compile_count() == 1
        assert r1.iters != r2.iters  # the stepsize really flowed through
        # both converged to the same optimum (1-strongly-convex data term)
        assert np.max(np.abs(r1.x - r2.x)) < 1e-3
        assert r1.rel_gap_history[-1] <= 1e-8
        assert r2.rel_gap_history[-1] <= 1e-8

    def test_problem_data_is_dynamic_not_baked(self, tv_instance_32):
        """Two instances (same shape, different y/lam) through one kernel
        must produce instance-correct answers (data flows via pytree
        leaves, never baked into the compiled code)."""
        y, lam = tv_instance_32
        rng = np.random.default_rng(21)
        y2 = 0.5 + 0.1 * rng.standard_normal(y.shape)
        p1 = tv_denoise_problem(y, lam)
        p2 = tv_denoise_problem(y2, 0.03)
        b = Budget(max_iters=100_000, tol=1e-8, sample_every=100)
        clear_kernel_cache()
        r1 = solve_pdhg(p1, PDHGParams(t=0.5, s=0.99 / 4), b)
        r2 = solve_pdhg(p2, PDHGParams(t=0.5, s=0.99 / 4), b)
        assert kernel_compile_count() == 1
        assert not np.allclose(r1.x, r2.x)
        # each x is the denoised version of ITS OWN y
        from atlas.certify.residuals import rel_gap
        import jax.numpy as jnp

        assert float(rel_gap(jnp.asarray(r2.x), jnp.asarray(r2.u), jnp.asarray(y2), 0.03, p2.L)) <= 1e-8


# ---------------------------------------------------------------------------
# Lever 2: process-parallel fitness evaluation
# ---------------------------------------------------------------------------


class TestParallelEvaluation:
    def test_pool_map_matches_serial(self):
        from atlas.evolve import RandomBackend, evolve, make_pool
        from atlas.evolve.parallel import demo_fitness

        hist_serial = evolve(
            RandomBackend(seed=11), demo_fitness, generations=3, pop_size=6, seed=11
        )
        with make_pool(2, ctx="fork") as pool:  # pure-python fitness: fork ok
            hist_par = evolve(
                RandomBackend(seed=11), demo_fitness, generations=3, pop_size=6,
                seed=11, pool=pool,
            )
        assert hist_par["best_genome"] == hist_serial["best_genome"]
        assert hist_par["best_fitness"] == hist_serial["best_fitness"]
        assert [g["fitnesses"] for g in hist_par["generations"]] == [
            g["fitnesses"] for g in hist_serial["generations"]
        ]
        assert (
            hist_par["n_unique_evaluated"] == hist_serial["n_unique_evaluated"]
        )

    def test_genomes_pickle(self):
        import pickle

        from atlas.evolve import RandomBackend

        for g in RandomBackend(seed=5, p_mix=0.5, p_switch=0.4).ask(20):
            g2 = pickle.loads(pickle.dumps(g))
            assert g2 == g


# ---------------------------------------------------------------------------
# Lever 3: persistent compilation cache
# ---------------------------------------------------------------------------


def test_persistent_compilation_cache_configured():
    import jax

    d = jax.config.jax_compilation_cache_dir
    assert d and "atlas_jax" in d
    assert os.path.isdir(d)
