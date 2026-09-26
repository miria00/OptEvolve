"""Boundary admission, runtime cache separation and fused-wrapper equivalence."""
from dataclasses import replace
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from atlas.backend.cuda_tv import VARIANTS, call_update, load_kernels
from atlas.backend.unit_tests import ref_tv_update
from atlas.genome import (Backend, Genome, GenomeParams, Wrapper, Metric,
                          content_hash, genome_to_dict, genome_from_dict, validate)
from atlas.ir.spec import tv_denoise_problem
from atlas.operators.prox import Box
from atlas.schemes.compile import solve_genome
from atlas.typing_ import TypeCheckError
from atlas.wrappers.km import Budget

GPU = pytest.mark.skipif(jax.devices()[0].platform != "gpu", reason="CUDA FFI integration")


def test_tv_gene_roundtrip_and_scope():
    base = Genome("condat_vu", GenomeParams(tau=.05, sigma=1., rho=1.9), backend=Backend(K=64))
    for name in VARIANTS:
        g = replace(base, backend=replace(base.backend, kernel=name))
        assert genome_from_dict(genome_to_dict(g)) == g
        assert content_hash(g) != content_hash(base)
        assert validate(g, "tv_denoise").ok
        assert not validate(replace(g, backend=replace(g.backend, precision="schedule", theta=.01)), "tv_denoise").ok
    p = tv_denoise_problem(np.zeros((5, 7)), .1)
    g = replace(base, backend=Backend(kernel="tv_fused128"))
    for bad in (replace(p, g=Box(0.,1.)), replace(p, name="fake")):
        with pytest.raises(TypeCheckError, match="canonical"):
            solve_genome(bad, g, Budget(max_iters=5))
    with pytest.raises(TypeCheckError, match="canonical"):
        solve_genome(p, replace(g, metric=Metric(kind="ruiz_pc", iters=10, alpha=1.)), Budget(max_iters=5))


def test_math_gate_precedes_native_loading(monkeypatch):
    import atlas.backend.cuda_tv as module
    monkeypatch.setattr(module, "load_kernels", lambda: pytest.fail("native load preceded construction rejection"))
    p = tv_denoise_problem(np.zeros((5, 7)), .1)
    g = Genome("pdhg", GenomeParams(tau=2., sigma=2.), backend=Backend(kernel="tv_fused128"))
    with pytest.raises(TypeCheckError):
        solve_genome(p, g, Budget(max_iters=5))


@GPU
@pytest.mark.parametrize("name", list(VARIANTS))
@pytest.mark.parametrize("shape", [(1,1), (1,7), (9,1), (9,7), (17,33)])
def test_native_boundaries_match_independent_matrix_reference(name, shape):
    rng = np.random.default_rng(942)
    x, y = (rng.normal(size=shape) for _ in range(2))
    u = rng.normal(size=(2,*shape))
    # Inactive last-row/last-column components must not enter the adjoint.
    u[0,-1,:] = 1000.; u[1,:,-1] = -1000.
    target = load_kernels()[1][name]
    for proximal in (False, True):
        for t,s,lam,rho in ((.05,1.,.1,1.), (.25,.25,.3,1.5), (.01,.9,.001,1.9)):
            got = call_update(target, *map(jnp.asarray,(x,u,y)), t,s,lam,proximal,rho)
            packed = np.concatenate([np.asarray(v).ravel() for v in got])
            np.testing.assert_allclose(packed, ref_tv_update(x,u,y,t,s,lam,proximal,rho), rtol=2e-12, atol=2e-12)


@GPU
@pytest.mark.parametrize("name", list(VARIANTS))
@pytest.mark.parametrize("scheme", ["pdhg", "condat_vu"])
@pytest.mark.parametrize("anchored", [False, True])
def test_fused_solver_trajectory_and_same_shape_data_are_correct(name, scheme, anchored):
    wrapper = Wrapper(halpern=anchored, restart="fixed_k" if anchored else "none", restart_k=11 if anchored else None)
    g = Genome(scheme, GenomeParams(tau=.05, sigma=1., rho=1.9), wrapper, backend=Backend(K=13))
    for seed in (41,42):
        p = tv_denoise_problem(np.random.default_rng(seed).random((17,31)), .1)
        budget = Budget(max_iters=137,tol=0.)
        ref = solve_genome(p,g,budget)
        got = solve_genome(p,replace(g,backend=replace(g.backend,kernel=name)),budget)
        assert got.iters == ref.iters == 137
        np.testing.assert_allclose(got.x, ref.x, rtol=2e-11, atol=2e-11)
        np.testing.assert_allclose(got.u, ref.u, rtol=2e-11, atol=2e-11)
        np.testing.assert_allclose(got.rel_gap_history, ref.rel_gap_history, rtol=2e-11, atol=2e-11)
        assert got.extra["backend"]["kernel_evidence"]["registry_check"]["passed"]
