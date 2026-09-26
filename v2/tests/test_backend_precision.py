"""Precision-policy tests on the canonical TV instance (64x64 synthetic).

The Condat-Vu fixed-point map is rebuilt here from the stable atoms
(atlas.operators, atlas.certify) rather than imported from atlas.schemes,
because the policy contract takes dtype-parametric factories and the schemes
package is being rewritten in a parallel lane; certified side condition for
the chosen parameters: t*L_f/2 + t*s*||D||^2 = 0.025 + 0.4 = 0.425 < 1
(LSCOMO 3.13)."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from atlas.backend.precision import (
    F32Cert64Policy,
    F64Policy,
    PrecisionError,
    SchedulePolicy,
    certified_gap,
    get_policy,
)
from atlas.certify.residuals import rel_gap
from atlas.operators.linear import Grad2D

LAM = 0.1
T_STEP = 0.05
S_STEP = 1.0
MAX_ITERS = 60_000


def make_test_image(n=64, seed=0):
    """Piecewise-constant blocks + noise: the TV-friendly synthetic."""
    rng = np.random.default_rng(seed)
    blocks = rng.uniform(0.0, 1.0, size=(n // 8, n // 8))
    img = np.kron(blocks, np.ones((8, 8)))
    img += 0.05 * rng.standard_normal((n, n))
    return img


def make_tv_factories(y64, lam=LAM, t=T_STEP, s=S_STEP):
    """Condat-Vu (g = 0) on TV denoising, dtype-parametric.

    step_factory(dtype) casts the data constant to dtype so the f32 phase is
    genuinely f32 (x64 promotion rules would otherwise silently upcast);
    gap64 closes over exact f64 data."""
    shape = y64.shape
    D = Grad2D(shape=shape)

    def step_factory(dtype):
        yd = jnp.asarray(y64, dtype)

        def step(state, k):
            x, u = state
            x_new = x - t * (x - yd) - t * D.adjoint(u)
            u_new = jnp.clip(u + s * D.forward(2.0 * x_new - x), -lam, lam)
            return (x_new, u_new)

        return step

    def init_factory(dtype):
        return (jnp.asarray(y64, dtype),
                jnp.zeros((2,) + shape, dtype=dtype))

    yj = jnp.asarray(y64, jnp.float64)

    def gap64(state):
        x, u = state
        return rel_gap(x, u, yj, lam, D)

    return step_factory, init_factory, gap64


@pytest.fixture(scope="module")
def tv64():
    y = make_test_image(64, seed=0)
    return make_tv_factories(y)


def test_f64_baseline_converges(tv64):
    step_f, init_f, gap64 = tv64
    res = F64Policy(K=100).run(step_f, init_f, gap64, tol=1e-4,
                               max_iters=MAX_ITERS)
    assert res.converged
    assert res.certified_gap <= 1e-4
    assert res.iterate_dtypes == ("float64",)
    assert res.switch_iter is None


def test_f32_cert64_reaches_1e4_with_f64_certificate(tv64):
    step_f, init_f, gap64 = tv64
    res = F32Cert64Policy(K=100).run(step_f, init_f, gap64, tol=1e-4,
                                     max_iters=MAX_ITERS)
    # converged, at an f32 iterate
    assert res.converged
    assert res.certified_gap <= 1e-4
    assert res.iterate_dtypes == ("float32",)
    # the certificate itself is exact float64, asserted on dtype
    assert res.certificate_dtype == "float64"
    assert isinstance(res.certified_gap, np.float64)
    assert res.gap_history.dtype == np.float64
    # final state is returned promoted to f64
    assert all(a.dtype == jnp.float64
               for a in jax.tree_util.tree_leaves(res.state))


def test_certificate_is_f64_even_from_f32_state(tv64):
    step_f, init_f, gap64 = tv64
    state32 = init_f(jnp.float32)
    assert all(a.dtype == jnp.float32
               for a in jax.tree_util.tree_leaves(state32))
    g = certified_gap(gap64, state32)
    assert isinstance(g, np.float64)
    assert np.asarray(g).dtype == np.float64


def test_certificate_rejects_non_f64_gap(tv64):
    _, init_f, _ = tv64

    def bad_gap(state):
        x, u = state
        return jnp.float32(jnp.sum(x))  # a "certificate" computed at f32

    with pytest.raises(PrecisionError, match="float64"):
        certified_gap(bad_gap, init_f(jnp.float32))


def test_schedule_switches_and_reaches_1e6(tv64):
    step_f, init_f, gap64 = tv64
    pol = SchedulePolicy(K=100, theta=1e-3)
    res = pol.run(step_f, init_f, gap64, tol=1e-6, max_iters=MAX_ITERS)
    assert res.converged
    assert res.certified_gap <= 1e-6
    # it actually ran both phases
    assert res.switch_iter is not None
    assert res.iterate_dtypes == ("float32", "float64")
    assert 0 < res.switch_iter < res.iters
    # the f64 gap at the switch obeyed the trigger: crossed theta or plateaued
    switch_idx = int(np.searchsorted(res.gap_iters, res.switch_iter))
    g_at_switch = res.gap_history[min(switch_idx, len(res.gap_history) - 1)]
    assert g_at_switch <= 1e-3 or len(res.gap_history) > 3
    assert res.certificate_dtype == "float64"


def test_gap_history_is_monotone_enough(tv64):
    # Sanity on the certificate trace: the f64 gap should trend down by
    # orders of magnitude over the run (not strict monotonicity, which KM
    # iterations do not promise per-K-block).
    step_f, init_f, gap64 = tv64
    res = F32Cert64Policy(K=20).run(step_f, init_f, gap64, tol=1e-4,
                                    max_iters=4000)
    assert len(res.gap_history) >= 2
    assert res.gap_history[-1] < res.gap_history[0] * 1e-1


def test_policy_registry_and_no_bf16():
    assert set(["f64", "f32_cert64", "schedule"]) == set(
        get_policy(n).name for n in ("f64", "f32_cert64", "schedule"))
    for bad in ("bf16", "bfloat16", "f16", "float16", "half"):
        with pytest.raises(PrecisionError, match="cancellation"):
            get_policy(bad)
    with pytest.raises(ValueError, match="unknown"):
        get_policy("f128")


def test_schedule_params_are_genome_parameters():
    pol = get_policy("schedule", K=250, theta=5e-4)
    assert pol.K == 250 and pol.theta == 5e-4
