"""The placement gene: hash-preserving when absent, pinned when set."""
import dataclasses

import jax
import numpy as np
import pytest

import atlas  # noqa: F401
from atlas.backend.hardware import prepare_base
from atlas.genome import (Backend, Genome, GenomeParams, canonical_dict,
                          content_hash, genome_from_dict, genome_to_dict, validate)
from atlas.ir.spec import tv_denoise_problem
from atlas.schemes.base_schemes import Budget, run_built


def _g(**backend):
    return Genome("condat_vu", GenomeParams(tau=.05, sigma=1., rho=1.9),
                  backend=Backend(precision="f64", K=16, **backend))


def _has(platform):
    try:
        jax.devices(platform); return True
    except RuntimeError:
        return False


def test_absent_device_leaves_serialisation_unchanged():
    g = _g()
    assert "device" not in genome_to_dict(g)["backend"]
    assert "device" not in canonical_dict(g)["backend"]


def test_device_roundtrips_and_changes_the_hash():
    for d in ("cpu", "gpu"):
        g = _g(device=d)
        g2 = genome_from_dict(genome_to_dict(g))
        assert g2.backend == g.backend and content_hash(g2) == content_hash(g)
    assert len({content_hash(_g()), content_hash(_g(device="cpu")), content_hash(_g(device="gpu"))}) == 3


def test_validate_rejects_unknown_device_and_cpu_cuda_kernel():
    assert not validate(_g(device="tpu")).ok
    bad = Genome("condat_vu", GenomeParams(tau=.05, sigma=1., rho=1.9),
                 backend=Backend(precision="f64", K=16, kernel="tv_fused128", device="cpu"))
    assert not validate(bad).ok


def _solve(g):
    y = np.random.default_rng(0).random((16, 16))
    built, cert, params, _ = prepare_base(tv_denoise_problem(y, 0.1), g)
    return run_built(built, cert, params, Budget(max_iters=400, tol=1e-4, sample_every=250), backend=g.backend)


@pytest.mark.skipif(not _has("cpu"), reason="needs a cpu backend")
def test_pinned_cpu_matches_default_on_a_cpu_process():
    if _has("cuda"):
        pytest.skip("default device is the GPU in this process")
    r0, r1 = _solve(_g()), _solve(_g(device="cpu"))
    assert r0.iters == r1.iters
    np.testing.assert_allclose(r0.x, r1.x, atol=1e-12)


@pytest.mark.skipif(_has("cuda"), reason="only meaningful without a GPU backend")
def test_missing_backend_is_an_error_not_a_fallback():
    with pytest.raises(RuntimeError, match="no cuda backend"):
        _solve(_g(device="gpu"))


@pytest.mark.skipif(not (_has("cpu") and _has("cuda")), reason="needs cpu and cuda backends in one process")
def test_same_genome_solves_on_both_devices():
    rc, rg = _solve(_g(device="cpu")), _solve(_g(device="gpu"))
    np.testing.assert_allclose(rc.x, rg.x, atol=1e-8)
