"""Screened FISTA certifies the FULL problem, matches the unscreened solution, and re-plans."""
import jax
import pytest
import numpy as np

jax.config.update("jax_enable_x64", True)


def _data(seed=11, n=2000, p=400, rho=0.8):
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n, p)); X = np.empty_like(Z); X[:, 0] = Z[:, 0]
    for j in range(1, p):                       # AR(1) correlated columns: FISTA needs many chunks
        X[:, j] = rho * X[:, j - 1] + np.sqrt(1 - rho ** 2) * Z[:, j]
    w = np.zeros(p); w[:15] = rng.normal(size=15)                   # unscaled: kappa ~ 90
    y = np.sign(X @ w + 0.2 * rng.normal(size=n)); y[y == 0] = 1
    return X, y


def test_screened_fista_matches_unscreened_and_shrinks():
    from atlas.bench import logreg_cell as lc
    from atlas.certify.residuals import logreg_enet_rel_gap
    from atlas.evolve.screened_runner import run_screened
    X, y = _data()
    lam = 0.1 * lc.lam_max(X, y, 0.5)
    on = run_screened(X, y, lam, 0.5, tol=1e-10, K=16)
    off = run_screened(X, y, lam, 0.5, tol=1e-10, K=16, screen=False)
    prob = lc.composite(X, y, lam=lam, alpha=0.5)
    import jax.numpy as jnp
    assert on["converged"] and float(logreg_enet_rel_gap(jnp.asarray(on["x"]), prob.f, prob.g)) <= 1e-10
    # strong convexity (mu = lam (1 - alpha)) bounds each run's distance to w*: ||w - w*||^2 <= 2 gap / mu
    from atlas.certify.residuals import logreg_enet_gap_parts
    mu = lam * 0.5
    rad = lambda w: float(np.sqrt(2.0 * max(float(np.subtract(*logreg_enet_gap_parts(jnp.asarray(w), prob.f, prob.g)[:2])), 0.0) / mu))
    assert np.linalg.norm(on["x"] - off["x"]) <= rad(on["x"]) + rad(off["x"]) + 1e-12
    assert off["evals"] > 64 and on["replans"] >= 1 and on["active_sizes"][-1] < 0.5 * X.shape[1]


def test_diagnoser_names_the_bottlenecks():
    from atlas.evolve.diagnose import diagnose
    ours = {"seconds": 1.534, "evals": 192, "nnz": 197, "p": 2000}
    cuml = {"seconds": 0.755, "evals": 37}
    hw = {"step_GBps": 909, "peak_GBps": 1008, "f32_XTv_GBps": 298, "f64_XTv_GBps": 871}
    d = diagnose(ours, cuml, hw)
    names = {s["symptom"] for s in d["symptoms"]}
    assert {"sparse_solution", "memory_bound_step", "f32_transposed_product_slow", "iteration_gap"} <= names
    assert d["mutations"][0]["ingredient"] == "gap_safe_screening"


def test_bucketed_keep_momentum_certifies_full_problem():
    from atlas.bench import logreg_cell as lc
    from atlas.certify.residuals import logreg_enet_rel_gap
    from atlas.evolve.screened_runner import run_screened
    import jax.numpy as jnp
    X, y = _data()
    lam = 0.1 * lc.lam_max(X, y, 0.5)
    r = run_screened(X, y, lam, 0.5, tol=1e-10, K=16, momentum="keep", buckets=True)
    prob = lc.composite(X, y, lam=lam, alpha=0.5)
    assert r["converged"] and float(logreg_enet_rel_gap(jnp.asarray(r["x"]), prob.f, prob.g)) <= 1e-10
    assert r["replans"] >= 1 and all(w in (400, 200, 100, 50, 25) for w in r["active_sizes"])


def test_float32_iterates_keep_their_dtype_and_certify():
    import jax.numpy as jnp
    from atlas.bench import logreg_cell as lc
    from atlas.evolve.screened_runner import run_screened
    X, y = _data()
    lam = 0.1 * lc.lam_max(X, y, 0.5)
    r = run_screened(X, y, lam, 0.5, tol=1e-6, K=16, momentum="keep", buckets=True, dtype=jnp.float32)
    assert r["converged"] and r["gap"] <= 1e-6 and r["replans"] >= 1


def test_working_set_certifies_full_problem_from_a_small_start():
    import jax.numpy as jnp
    from atlas.bench import logreg_cell as lc
    from atlas.certify.residuals import logreg_enet_rel_gap, logreg_enet_gap_parts
    from atlas.evolve.screened_runner import run_screened
    X, y = _data()
    lam = 0.1 * lc.lam_max(X, y, 0.5)
    ws = run_screened(X, y, lam, 0.5, tol=1e-10, K=16, momentum="keep", working_set=True, ws_init=0.05)
    off = run_screened(X, y, lam, 0.5, tol=1e-10, K=16, screen=False)
    prob = lc.composite(X, y, lam=lam, alpha=0.5)
    assert ws["converged"] and float(logreg_enet_rel_gap(jnp.asarray(ws["x"]), prob.f, prob.g)) <= 1e-10
    assert ws["active_sizes"][0] < X.shape[1] // 4
    mu = lam * 0.5
    rad = lambda w: float(np.sqrt(2.0 * max(float(np.subtract(*logreg_enet_gap_parts(jnp.asarray(w), prob.f, prob.g)[:2])), 0.0) / mu))
    assert np.linalg.norm(ws["x"] - off["x"]) <= rad(ws["x"]) + rad(off["x"]) + 1e-12


@pytest.mark.parametrize("rule", ["power", "gram"])
def test_timed_step_rules_certify_and_gram_is_exact(rule):
    import jax.numpy as jnp
    from atlas.bench import logreg_cell as lc
    from atlas.evolve.screened_runner import run_screened
    X, y = _data()
    lam = 0.1 * lc.lam_max(X, y, 0.5)
    full = run_screened(X, y, lam, 0.5, tol=1e-8, K=16, screen=False, step_rule=rule)
    exact = float(np.linalg.eigvalsh(X.T @ X)[-1]) / (4 * X.shape[0])
    assert full["converged"] and full["L_values"][0] >= (exact if rule == "gram" else 0.99 * exact) * (1 - 1e-9)
    ws = run_screened(X, y, lam, 0.5, tol=1e-8, K=16, momentum="keep", working_set=True, ws_init=0.05, step_rule=rule)
    assert ws["converged"] and ws["L_values"][0] <= full["L_values"][0] * (1 + 1e-9)
