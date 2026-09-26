"""FISTA scheme: gate, certified solve, and fewer iterations than forward-backward."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)


def _prob(alpha=0.5, seed=5, n=600, p=60):
    from atlas.bench import logreg_cell as lc
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p)) / np.sqrt(p); X[:, :5] *= 8.0          # mildly ill-conditioned
    y = np.sign(X @ rng.normal(size=p) + 0.3 * rng.normal(size=n)); y[y == 0] = 1
    return lc.composite(X, y, lam=0.05 * lc.lam_max(X, y, alpha), alpha=alpha)


def test_gate_rejects_steps_above_one_over_L():
    from atlas.schemes.base_schemes import FISTAParams, build_scheme
    from atlas.typing_ import TypeCheckError
    prob = _prob()
    with pytest.raises(TypeCheckError):
        build_scheme(prob, "fista", FISTAParams(step=1.5 / prob.f.props.lipschitz))


@pytest.mark.parametrize("backend", [None, "f64"])
def test_fista_certifies_with_fewer_iterations_than_fbs(backend):
    from atlas.certify.residuals import logreg_enet_rel_gap
    from atlas.genome import Backend
    from atlas.schemes.base_schemes import Budget, FBSParams, FISTAParams, build_scheme, run_built
    prob = _prob()
    a = 1.0 / prob.f.props.lipschitz
    be = None if backend is None else Backend(precision=backend, K=64)
    res = {}
    for scheme, params in (("fista", FISTAParams(step=a)), ("fbs", FBSParams(step=a))):
        built = build_scheme(prob, scheme, params)
        r = run_built(built, built.cert, params, Budget(max_iters=100000, tol=1e-8, sample_every=500), backend=be)
        assert float(logreg_enet_rel_gap(jnp.asarray(r.x), prob.f, prob.g)) <= 1e-8
        res[scheme] = r.iters
    # with a per-iteration certificate FISTA must need fewer iterations; at cadence K = 64 both may
    # certify inside the same chunk, so counts are quantised to multiples of 64
    if backend is None:
        assert res["fista"] < res["fbs"], res
    else:
        assert res["fista"] <= res["fbs"], res
