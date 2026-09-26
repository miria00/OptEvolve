"""Gap Safe screening for elastic-net logistic regression: never discards a true nonzero."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("alpha", [0.5, 0.9, 1.0])
def test_screening_is_safe_and_tightens_as_the_gap_closes(alpha):
    from atlas.bench import logreg_cell as lc
    from atlas.certify.residuals import logreg_enet_gap_safe_screen, logreg_enet_rel_gap
    from atlas.schemes.base_schemes import FISTAParams, build_scheme
    rng = np.random.default_rng(7)
    n, p = 800, 120
    X = rng.normal(size=(n, p)) / np.sqrt(p)
    wt = np.zeros(p); wt[:8] = rng.normal(size=8) * 4
    y = np.sign(X @ wt + 0.2 * rng.normal(size=n)); y[y == 0] = 1
    prob = lc.composite(X, y, lam=0.2 * lc.lam_max(X, y, alpha), alpha=alpha)
    built = build_scheme(prob, "fista", FISTAParams(step=1.0 / prob.f.props.lipschitz))
    step = jax.jit(lambda s: built.T(s, 0, built.dyn))
    state, traj = built.state0, []
    for k in range(20000):
        state = step(state)
        if k in (3, 10, 30, 100, 300):
            traj.append(state[0])
    w_star = state[0]
    assert float(logreg_enet_rel_gap(w_star, prob.f, prob.g)) < 1e-11
    zero_star = np.abs(np.asarray(w_star)) < 1e-12
    counts = []
    for w in traj:
        mask, r, gap = logreg_enet_gap_safe_screen(w, prob.f, prob.g)
        mask = np.asarray(mask)
        assert not np.any(mask & ~zero_star), "screened a feature that is nonzero at the optimum"
        counts.append(int(mask.sum()))
    assert counts[-1] >= counts[0] and counts[-1] > 0.5 * zero_star.sum(), (counts, int(zero_star.sum()))
