"""Hawkeye per-atom kernel checks: production atoms vs independent numpy
references, via the atlas.backend.unit_tests registry."""
import numpy as np
import pytest

from atlas.backend import unit_tests as ut


@pytest.mark.parametrize("name", sorted(ut.REGISTRY))
def test_atom_matches_reference(name):
    report = ut.run_check(name)
    assert report.passed, str(report)


def test_run_all_covers_all_five_atoms():
    reports = ut.run_all()
    assert sorted(r.name for r in reports) == sorted(ut.REGISTRY)
    assert all(r.passed for r in reports), [str(r) for r in reports]


def test_harness_catches_a_wrong_kernel():
    # A deliberately broken prox_l1 (identity) must FAIL the check: the
    # harness is only trustworthy if it can reject.
    def broken_prox_l1(v, step, lam=1.0):
        return v

    spec = ut.REGISTRY["prox_l1"]
    report = ut.check(broken_prox_l1, spec.reference, spec.make_inputs,
                      tolerance=spec.tol, name="broken_prox_l1")
    assert not report.passed


def test_harness_rejects_nonfinite_kernel_outputs():
    for value in (np.nan, np.inf, -np.inf):
        report = ut.run_check("prox_l1", lambda v, step, lam: np.full_like(v, value))
        assert not report.passed


def test_harness_catches_a_wrong_adjoint():
    # An adjoint with a flipped sign convention must fail the exact-transpose
    # and inner-product checks (the v1-style silent-adjoint-bug class).
    impls = ut.default_impls()
    fwd, adj = impls["grad2d"]

    def bad_adjoint(u):
        return -np.asarray(adj(u))

    report = ut.check_linop(fwd, bad_adjoint, tolerance=1e-12,
                            name="grad2d_bad_adjoint")
    assert not report.passed


def test_f32_kernel_variant_passes_at_declared_tolerance():
    # The registry gate for future kernel variants: an f32 prox_l1 kernel
    # fails at the f64 tolerance but passes at its declared f32 tolerance.
    import jax.numpy as jnp

    def prox_l1_f32(v, step, lam=1.0):
        v32 = jnp.asarray(v, jnp.float32)
        t = jnp.float32(step * lam)
        return jnp.sign(v32) * jnp.maximum(jnp.abs(v32) - t, 0.0)

    strict = ut.run_check("prox_l1", prox_l1_f32,
                          tolerance=ut.DEFAULT_TOL_F64)
    declared = ut.run_check("prox_l1", prox_l1_f32,
                            tolerance=ut.DEFAULT_TOL_F32)
    assert not strict.passed
    assert declared.passed
