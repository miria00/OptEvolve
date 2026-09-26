"""Davis-Yin splitting (5th base scheme): typecheck, solves vs CVXPY,
generic-composite structural dispatch."""
import numpy as np
import pytest

from atlas import (
    Budget,
    DYSParams,
    TypeCheckError,
    lasso_box_problem,
    solve_dys,
    tv_denoise_problem,
)
from atlas.genome import genome_from_dict, validate
from atlas.typing_ import typecheck_dys


@pytest.fixture(scope="module")
def lasso_box_10x10():
    rng = np.random.default_rng(3)
    y = 0.7 * rng.standard_normal((10, 10)) + 0.3
    return lasso_box_problem(y, lam=0.15, lo=0.0, hi=1.0), y


def _cvxpy_solution(y, lam, lo, hi):
    import cvxpy as cp

    x = cp.Variable(y.shape)
    obj = 0.5 * cp.sum_squares(x - y) + lam * cp.sum(cp.abs(x))
    prob = cp.Problem(cp.Minimize(obj), [x >= lo, x <= hi])
    prob.solve(solver=cp.CLARABEL)
    return np.asarray(x.value), float(prob.value)


def _objective(x, y, lam):
    return float(0.5 * np.sum((x - y) ** 2) + lam * np.sum(np.abs(x)))


def test_dys_cert_side_condition():
    # beta = 1/L_f: a in (0, 2/L); theta = 2/(4 - a*L)
    cert = typecheck_dys(L_smooth=1.0, step=1.0)
    assert cert.satisfied and cert.scheme == "DYS"
    assert cert.averagedness == pytest.approx(2.0 / 3.0)
    assert "Thm 28" in cert.citation and "Davis" in cert.citation
    with pytest.raises(TypeCheckError):
        typecheck_dys(L_smooth=1.0, step=2.0)  # a = 2/L exactly: rejected
    with pytest.raises(TypeCheckError):
        typecheck_dys(L_smooth=1.0, step=-0.1)
    with pytest.raises(TypeCheckError):
        typecheck_dys(L_smooth=1.0, step=float("inf"))


def test_dys_matches_cvxpy(lasso_box_10x10):
    problem, y = lasso_box_10x10
    res = solve_dys(
        problem,
        DYSParams(step=1.0),
        Budget(max_iters=100_000, tol=1e-12, sample_every=50),
    )
    assert res.cert.satisfied
    x_ref, p_ref = _cvxpy_solution(y, 0.15, 0.0, 1.0)
    # x-agreement is limited by CVXPY's own accuracy (~1e-8 objective gap
    # -> ~1e-4 in x via 1-strong convexity); the objective is tighter.
    assert np.max(np.abs(res.x - x_ref)) < 1e-4
    assert _objective(res.x, y, 0.15) <= p_ref + 1e-8
    # iterate respects the box (it is the output of the g-prox then... the
    # extract point is prox_g(z); check feasibility to solver tolerance)
    assert res.x.min() >= -1e-8 and res.x.max() <= 1.0 + 1e-8


def test_dys_with_halpern_and_relaxation(lasso_box_10x10):
    problem, y = lasso_box_10x10
    x_ref, _ = _cvxpy_solution(y, 0.15, 0.0, 1.0)
    res = solve_dys(
        problem,
        DYSParams(step=1.0, rho=1.2),
        Budget(max_iters=100_000, tol=1e-12, sample_every=50),
    )
    assert res.cert.scheme == "DYS+relax"
    assert np.max(np.abs(res.x - x_ref)) < 1e-4
    assert _objective(res.x, y, 0.15) <= _objective(x_ref, y, 0.15) + 1e-8
    res_h = solve_dys(
        problem,
        DYSParams(step=1.0, halpern=True),
        Budget(max_iters=5_000, tol=1e-10, sample_every=50),
    )
    assert res_h.cert.scheme == "DYS+halpern"
    hist = res_h.rel_gap_history
    hist = hist[np.isfinite(hist)]
    assert hist[-1] < hist[0]


def test_dys_rejects_bad_step(lasso_box_10x10):
    problem, _ = lasso_box_10x10
    with pytest.raises(TypeCheckError):
        solve_dys(problem, DYSParams(step=2.0), Budget(max_iters=10))
    with pytest.raises(TypeCheckError):
        solve_dys(problem, DYSParams(step=0.0), Budget(max_iters=10))


def test_dys_structurally_rejected_on_tv(tv_instance_32):
    y, lam = tv_instance_32
    p = tv_denoise_problem(y, lam)
    with pytest.raises(TypeCheckError, match="linear map"):
        solve_dys(p, DYSParams(step=1.0), Budget(max_iters=10))


def test_dys_genome_gate_by_problem_kind():
    g = genome_from_dict({"scheme": "dys", "params": {"tau": 1.0}})
    v_tv = validate(g, problem_kind="tv_denoise")
    assert not v_tv.ok and "inapplicable" in v_tv.error
    v3 = validate(g, problem_kind="three_composite")
    assert v3.ok and v3.cert.scheme == "DYS"
    bad = genome_from_dict({"scheme": "dys", "params": {"tau": 2.5}})
    v_bad = validate(bad, problem_kind="three_composite")
    assert not v_bad.ok and "0 < a < 2/L" in v_bad.error
