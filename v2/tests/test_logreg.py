"""Elastic-net logistic regression: atoms, duality-gap certificate, certified FBS solve."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)


def _data(seed=0, n=300, p=40):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p)) / np.sqrt(p)
    w = np.zeros(p); w[:5] = rng.normal(size=5) * 3
    y = np.sign(X @ w + 0.3 * rng.normal(size=n)); y[y == 0] = 1
    return X, y


def test_gradient_and_lipschitz_bound():
    from atlas.bench import logreg_cell as lc
    X, y = _data()
    prob = lc.composite(X, y, lam=0.01, alpha=0.5)
    w = jnp.asarray(np.random.default_rng(1).normal(size=40))
    g = np.asarray(prob.f.grad(w))
    eps = 1e-6
    fd = np.array([(float(prob.f.value(w.at[j].add(eps))) - float(prob.f.value(w.at[j].add(-eps)))) / (2 * eps) for j in range(40)])
    assert np.allclose(g, fd, atol=1e-7)
    true_L = np.linalg.norm(X, 2) ** 2 / (4 * X.shape[0])
    assert true_L <= prob.f.props.lipschitz <= 1.05 * true_L


def test_prox_is_the_minimiser():
    from atlas.bench.logreg_cell import ElasticNet
    g = ElasticNet(lam=jnp.asarray(0.7), alpha=jnp.asarray(0.4))
    grid = np.linspace(-5, 5, 200001)
    for v in (-3.0, -0.1, 0.0, 0.2, 2.5):
        obj = g.lam * (g.alpha * np.abs(grid) + 0.5 * (1 - g.alpha) * grid ** 2) * 0.3 + 0.5 * (grid - v) ** 2
        assert abs(float(g.prox(jnp.asarray(v), 0.3)) - grid[np.argmin(obj)]) < 1e-4


@pytest.mark.parametrize("alpha", [0.0, 0.3, 0.9, 1.0])
def test_weak_duality_holds_at_random_points(alpha):
    from atlas.bench import logreg_cell as lc
    from atlas.certify.residuals import logreg_enet_rel_gap
    X, y = _data(2)
    prob = lc.composite(X, y, lam=0.02, alpha=alpha)
    rng = np.random.default_rng(3)
    for _ in range(20):
        assert float(logreg_enet_rel_gap(jnp.asarray(rng.normal(size=40)), prob.f, prob.g)) >= -1e-12


@pytest.mark.parametrize("alpha", [0.5, 1.0])
def test_fbs_certifies_and_satisfies_kkt(alpha):
    from atlas.bench import logreg_cell as lc
    from atlas.certify.residuals import logreg_enet_rel_gap
    from atlas.schemes.base_schemes import Budget, FBSParams, build_scheme, run_built
    X, y = _data(4)
    lam = 0.1 * lc.lam_max(X, y, alpha)
    prob = lc.composite(X, y, lam=lam, alpha=alpha)
    params = FBSParams(step=1.0 / prob.f.props.lipschitz)
    built = build_scheme(prob, "fbs", params)
    assert built.gap_kind == "logreg_gap"
    r = run_built(built, built.cert, params, Budget(max_iters=200000, tol=1e-9, sample_every=1000))
    w = jnp.asarray(r.x)
    assert float(logreg_enet_rel_gap(w, prob.f, prob.g)) <= 1e-9
    grad = np.asarray(prob.f.grad(w)) + lam * (1 - alpha) * np.asarray(w)
    nz = np.abs(np.asarray(w)) > 1e-10
    assert np.allclose(grad[nz] + lam * alpha * np.sign(np.asarray(w)[nz]), 0, atol=1e-5)
    assert np.all(np.abs(grad[~nz]) <= lam * alpha + 1e-5)
    assert 0 < nz.sum() < 40
