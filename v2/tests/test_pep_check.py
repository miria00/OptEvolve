"""Performance estimation of composed recipes agrees with the gates and with known rates."""
import pytest

from atlas.evolve import pep_check

pytestmark = pytest.mark.skipif(not pep_check.available(), reason="PEPit environment not installed")


def test_halpern_reproduces_the_known_rate():
    tau = pep_check.worst_cases([{"alpha": 1.0, "rho": 1.0, "anchor": True, "N": 10}])[0]
    assert tau == pytest.approx((2 / 11) ** 2, rel=1e-3)


def test_pep_agrees_with_the_gates_on_reflection():
    from atlas.evolve.kb_composer import Recipe
    # base 1/2-averaged (a PDHG-like map): rho = 2 is the reflection
    rank = pep_check.pep_rank([Recipe("fbs", 1.0, rho=1.0), Recipe("fbs", 1.0, rho=0.99),
                               Recipe("fbs", 1.0, rho=1.0, anchor=True), Recipe("fbs", 1.0, anderson=5)], alpha_base=0.5)
    reflect_plain, relaxed, reflect_anchored, anderson = rank
    assert reflect_plain["verdict"].startswith("NO worst-case convergence")   # the gate rejects rho alpha = 1 without an anchor
    assert relaxed["verdict"].startswith("worst-case convergent")
    assert reflect_anchored["verdict"].startswith("worst-case convergent")   # the gate admits it under the anchor
    assert anderson["tau"] is None


def test_gate_and_pep_agree_on_the_same_recipes():
    from atlas.evolve.kb_composer import Recipe, typecheck_recipe
    from atlas.typing_ import TypeCheckError
    import numpy as np
    from atlas.bench import logreg_cell as lc
    rng = np.random.default_rng(0); X = rng.normal(size=(200, 20)); y = np.sign(rng.normal(size=200))
    prob = lc.composite(X, y, lam=0.01, alpha=0.5)
    L = prob.f.props.lipschitz
    alpha = 2.0 / (4.0 - 1.9)                      # forward-backward averagedness at a = 1.9/L
    recipes = [Recipe("fbs", 1.9, rho=r, anchor=an) for r in (0.99, 1.0) for an in (False, True)]
    rank = pep_check.pep_rank(recipes, alpha_base=alpha)
    for r, pr in zip(recipes, rank):
        try:
            typecheck_recipe(r, prob); admitted = True
        except TypeCheckError:
            admitted = False
        assert admitted == pr["verdict"].startswith("worst-case convergent"), (r.name(), admitted, pr["verdict"])
