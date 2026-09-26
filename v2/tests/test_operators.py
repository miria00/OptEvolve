"""Prox closed forms vs cvxpy; Grad2D adjoint identity."""
import cvxpy as cp
import jax.numpy as jnp
import numpy as np
import pytest

from atlas.operators.linear import Grad2D
from atlas.operators.prox import (
    grad_quadratic,
    prox_box,
    prox_l1,
    prox_l2sq,
    prox_scaled_l1_conj,
    prox_zero,
    project_linf_ball,
)

TOL = 1e-6
N = 12


def _cvxpy_prox(objective_term, v):
    x = cp.Variable(v.shape[0])
    prob = cp.Problem(cp.Minimize(objective_term(x) + 0.5 * cp.sum_squares(x - v)))
    prob.solve(solver=cp.CLARABEL)
    return x.value


def test_prox_l1_vs_cvxpy(rng):
    v = rng.standard_normal(N)
    step, lam = 0.7, 0.3
    ours = np.asarray(prox_l1(jnp.array(v), step, lam))
    ref = _cvxpy_prox(lambda x: step * lam * cp.norm1(x), v)
    np.testing.assert_allclose(ours, ref, atol=TOL)


def test_prox_l2sq_vs_cvxpy(rng):
    v = rng.standard_normal(N)
    y = rng.standard_normal(N)
    step = 1.3
    ours = np.asarray(prox_l2sq(jnp.array(v), step, jnp.array(y)))
    ref = _cvxpy_prox(lambda x: step * 0.5 * cp.sum_squares(x - y), v)
    np.testing.assert_allclose(ours, ref, atol=TOL)


def test_prox_box_vs_cvxpy(rng):
    v = 2.0 * rng.standard_normal(N)
    lo, hi = -0.5, 0.8
    ours = np.asarray(prox_box(jnp.array(v), lo, hi))
    x = cp.Variable(N)
    prob = cp.Problem(
        cp.Minimize(0.5 * cp.sum_squares(x - v)), [x >= lo, x <= hi]
    )
    prob.solve(solver=cp.CLARABEL)
    np.testing.assert_allclose(ours, x.value, atol=TOL)


def test_project_linf_ball_vs_cvxpy(rng):
    v = 2.0 * rng.standard_normal(N)
    radius = 0.6
    ours = np.asarray(project_linf_ball(jnp.array(v), radius))
    x = cp.Variable(N)
    prob = cp.Problem(
        cp.Minimize(0.5 * cp.sum_squares(x - v)), [cp.norm_inf(x) <= radius]
    )
    prob.solve(solver=cp.CLARABEL)
    np.testing.assert_allclose(ours, x.value, atol=TOL)


def test_prox_scaled_l1_conj_moreau(rng):
    """Moreau identity: prox_{s h*}(v) = v - s prox_{h/s}(v/s) = clip."""
    v = 2.0 * rng.standard_normal(N)
    lam = 0.4
    for s in (0.3, 1.0, 2.5):
        via_moreau = v - s * np.asarray(prox_l1(jnp.array(v) / s, 1.0 / s, lam))
        ours = np.asarray(prox_scaled_l1_conj(jnp.array(v), s, lam))
        np.testing.assert_allclose(ours, via_moreau, atol=1e-12)
        np.testing.assert_allclose(ours, np.clip(v, -lam, lam), atol=1e-12)


def test_prox_zero_and_grad_quadratic(rng):
    v = rng.standard_normal(N)
    y = rng.standard_normal(N)
    np.testing.assert_allclose(np.asarray(prox_zero(jnp.array(v), 0.5)), v)
    np.testing.assert_allclose(
        np.asarray(grad_quadratic(jnp.array(v), jnp.array(y))), v - y
    )


@pytest.mark.parametrize("shape", [(8, 8), (13, 9), (5, 17)])
def test_grad2d_adjoint_identity(shape, rng):
    """<D x, u> == <x, D^T u> on random inputs (exact adjoint)."""
    D = Grad2D(shape=shape)
    x = jnp.array(rng.standard_normal(shape))
    u = jnp.array(rng.standard_normal((2,) + shape))
    lhs = float(jnp.sum(D.forward(x) * u))
    rhs = float(jnp.sum(x * D.adjoint(u)))
    assert abs(lhs - rhs) < 1e-10 * (1.0 + abs(lhs))


def test_grad2d_neumann_and_norm_bound(rng):
    D = Grad2D(shape=(16, 16))
    # constant images are in the kernel (Neumann boundary)
    c = jnp.ones((16, 16))
    assert float(jnp.max(jnp.abs(D.forward(c)))) == 0.0
    # power iteration: ||D||^2 <= 8
    x = jnp.array(rng.standard_normal((16, 16)))
    for _ in range(200):
        x = D.adjoint(D.forward(x))
        x = x / jnp.linalg.norm(x)
    lam_max = float(jnp.sum(x * D.adjoint(D.forward(x))))
    assert lam_max <= 8.0 + 1e-9
    assert D.norm_bound == pytest.approx(np.sqrt(8.0))
