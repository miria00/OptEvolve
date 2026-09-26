"""Generated update admission and multi-step original-coordinate regressions."""
from dataclasses import replace
import jax
import numpy as np
import pytest

from atlas.backend.cuda_lp import VARIANTS, _csr, generated_source
from atlas.genome import (Backend, Genome, GenomeParams, Metric, Wrapper,
                          genome_to_dict, genome_from_dict, content_hash, validate)
from atlas.schemes.compile import solve_genome
from atlas.wrappers.km import Budget


def test_kernel_genes_are_hashed_and_old_default_serialization_is_unchanged():
    g = Genome("pdhg", GenomeParams(tau=.7, sigma=.7), backend=Backend())
    assert "kernel" not in genome_to_dict(g)["backend"]
    for variant in VARIANTS:
        v = replace(g, backend=replace(g.backend, kernel=variant))
        assert genome_from_dict(genome_to_dict(v)) == v
        assert content_hash(v) != content_hash(g)
    bad = replace(g, backend=Backend(precision="f32_cert64", kernel="csr_warp128"))
    assert not validate(bad, "lp_netflow").ok
    assert "__shfl_down_sync" in generated_source()


def test_unsupported_kernel_problem_is_rejected_before_cuda_loading():
    from atlas.ir.spec import tv_denoise_problem
    from atlas.typing_ import TypeCheckError
    p = tv_denoise_problem(np.zeros((5, 5)), .1)
    g = Genome("pdhg", GenomeParams(tau=.2, sigma=.4),
               backend=Backend(kernel="csr_thread128"))
    with pytest.raises(TypeCheckError, match="standard LP"):
        solve_genome(p, g, Budget(max_iters=5))


@pytest.mark.skipif(jax.devices()[0].platform != "gpu", reason="CUDA FFI integration")
@pytest.mark.parametrize("variant", list(VARIANTS))
@pytest.mark.parametrize("anchored", [False, True])
def test_generated_map_preserves_wrapped_metric_trajectory(variant, anchored):
    from atlas.bench.lp_cell import gen_lp_dense, to_composite
    p = to_composite(gen_lp_dense(19, 8, 701))
    wrapper = Wrapper(halpern=anchored, restart="fixed_k" if anchored else "none",
                      restart_k=11 if anchored else None)
    g = Genome("pdhg", GenomeParams(tau=.7, sigma=.7, rho=1.5), wrapper,
               backend=Backend(K=13), metric=Metric(kind="ruiz_pc", iters=10, alpha=1.))
    # Non-multiple final chunk, fixed trajectory, and two same-shape instances.
    for seed in (701, 702):
        p = to_composite(gen_lp_dense(19, 8, seed))
        budget = Budget(max_iters=107, tol=0.)
        ref = solve_genome(p, g, budget)
        got = solve_genome(p, replace(g, backend=replace(g.backend, kernel=variant)), budget)
        assert got.iters == ref.iters == 107
        np.testing.assert_allclose(got.x, ref.x, rtol=2e-9, atol=2e-9)
        np.testing.assert_allclose(got.u, ref.u, rtol=2e-9, atol=2e-9)
        np.testing.assert_allclose(got.rel_gap_history, ref.rel_gap_history, rtol=2e-9, atol=2e-9)
        assert got.extra["backend"]["kernel_evidence"]["admission_states"] == 4


@pytest.mark.skipif(jax.devices()[0].platform != "gpu", reason="CUDA FFI integration")
@pytest.mark.parametrize("variant", list(VARIANTS))
def test_irregular_csr_with_empty_rows_and_columns(variant):
    from atlas.backend.cuda_lp import load_kernels
    import jax.numpy as jnp
    import scipy.sparse as sp
    rng = np.random.default_rng(730)
    A = sp.random(37, 53, density=.19, random_state=rng).toarray()
    A[3] = 0
    A[:, 8] = 0
    v, old, base, cost = [rng.normal(size=n) for n in (53, 53, 37, 37)]
    target = load_kernels()[1][variant]
    for dual in (0, 1):
        call = jax.jit(lambda v, old, base, cost: jax.ffi.ffi_call(
            target, jax.ShapeDtypeStruct(base.shape, base.dtype))(
                *_csr(A), v, old, base, cost, jnp.asarray(.3), dual=np.int64(dual)))
        got = call(*map(jnp.asarray, (v, old, base, cost)))
        expected = base + .3 * (A @ (2*v-old)) - .3*cost if dual else np.maximum(
            base - .3*(A@v) - .3*cost, 0)
        np.testing.assert_allclose(got, expected, rtol=2e-13, atol=2e-13)
