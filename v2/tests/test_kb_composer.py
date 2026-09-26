"""KB composer: reproduces existing runners exactly on several bases, gates compositions,
counts cost honestly, certifies new recipes."""
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)
MM = Path(__file__).resolve().parents[1] / "data" / "maros_meszaros"


def _logreg(seed=3, n=800, p=80, alpha=0.5):
    from atlas.bench import logreg_cell as lc
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p)) / np.sqrt(p); X[:, :4] *= 6.0
    y = np.sign(X @ rng.normal(size=p) + 0.3 * rng.normal(size=n)); y[y == 0] = 1
    return lc.composite(X, y, lam=0.05 * lc.lam_max(X, y, alpha), alpha=alpha)


def _policy(problem, scheme, params, tol, cap=50000):
    from atlas.genome import Backend
    from atlas.schemes.base_schemes import Budget, build_scheme, run_built
    return run_built(build_scheme(problem, scheme, params), None, params,
                     Budget(max_iters=cap, tol=tol, sample_every=500), backend=Backend(precision="f64", K=64))


def test_fbs_recipes_reproduce_the_policy_runner():
    from dataclasses import replace
    from atlas.evolve.kb_composer import Recipe, base_params, run_recipe
    prob = _logreg()
    for rec, extra in ((Recipe("fbs", 1.0), {}), (Recipe("fbs", 1.0, anchor=True, restart=("fixed", 64)),
                                                   {"halpern": True, "restart_k": 64})):
        ref = _policy(prob, "fbs", replace(base_params(prob, rec), **extra), 1e-7)
        out = run_recipe(prob, rec, tol=1e-7, max_evals=50000, K=64)
        assert out["evals"] == ref.iters and np.allclose(out["x"], ref.x, rtol=1e-9, atol=1e-11), rec.name()


@pytest.mark.skipif(not (MM / "HS21.mat").exists(), reason="Maros-Meszaros not downloaded")
def test_condat_vu_anchor_restart_recipe_reproduces_the_policy_runner():
    from dataclasses import replace
    from atlas.bench import qp_cell
    from atlas.evolve.kb_composer import Recipe, base_params, run_recipe
    prob = qp_cell.load(str(MM / "HS21.mat"))
    rec = Recipe("condat_vu", 0.98, anchor=True, restart=("fixed", 64))
    ref = _policy(prob, "condat_vu", replace(base_params(prob, rec), halpern=True, restart_k=64), 1e-6)
    out = run_recipe(prob, rec, tol=1e-6, max_evals=50000, K=64)
    assert out["evals"] == ref.iters and np.allclose(out["x"], ref.x, rtol=1e-9, atol=1e-11)


def test_pdhg_recipe_on_a_transport_lp_reproduces_the_policy_runner():
    from atlas.bench import ot
    from atlas.evolve.kb_composer import Recipe, base_params, run_recipe
    prob = ot.transport_composite(6, 5, 1)
    rec = Recipe("pdhg", 0.98)
    ref = _policy(prob, "pdhg", base_params(prob, rec), 1e-6)
    out = run_recipe(prob, rec, tol=1e-6, max_evals=50000, K=64)
    assert out["evals"] == ref.iters and np.allclose(out["x"], ref.x, rtol=1e-9, atol=1e-11)


def test_gates_reject_inadmissible_compositions():
    from atlas.evolve.kb_composer import Recipe, typecheck_recipe
    from atlas.typing_ import TypeCheckError
    prob = _logreg()
    for bad in (Recipe("fista", 1.0, anchor=True), Recipe("fbs", 1.0, rho=1.2), Recipe("fbs", 1.0, rho=1.0),
                Recipe("fbs", 1.0, anchor=True, anderson=5), Recipe("fbs", 2.5), Recipe("pdhg", 0.98)):
        with pytest.raises(TypeCheckError):   # the last: PDHG has no linear map to split on this problem
            typecheck_recipe(bad, prob)
    typecheck_recipe(Recipe("fbs", 1.9, rho=1.0, anchor=True, restart=("adaptive", 0.2)), prob)


def test_anderson_cost_is_counted_honestly():
    from atlas.evolve.kb_composer import Recipe, run_recipe
    prob = _logreg()
    r = Recipe("fbs", 1.0, anderson=5)
    always = run_recipe(prob, r, tol=0.0, max_evals=64, K=64, aa_D=1e300)   # safeguard never rejects
    never = run_recipe(prob, r, tol=0.0, max_evals=64, K=64, aa_D=0.0)      # always falls back
    assert always["evals"] == 1 + 64 and always["aa_accepted"] == 64
    assert never["evals"] == 1 + 128 and never["aa_accepted"] == 0


@pytest.mark.parametrize("rec", [
    dict(base="fbs", step_frac=1.9, anderson=5),
    dict(base="fbs", step_frac=1.9, rho=1.0, anchor=True, restart=("adaptive", 0.2)),
    dict(base="fbs", step_frac=1.0, rho=0.99, anderson=5),
])
def test_new_recipes_certify(rec):
    from atlas.certify.residuals import logreg_enet_rel_gap
    from atlas.evolve.kb_composer import Recipe, novelty, run_recipe
    prob = _logreg()
    out = run_recipe(prob, Recipe(**rec), tol=1e-9, max_evals=200000, K=64)
    assert out["converged"] and float(logreg_enet_rel_gap(jnp.asarray(out["x"]), prob.f, prob.g)) <= 1e-9
    assert "verdict" in novelty(Recipe(**rec))
