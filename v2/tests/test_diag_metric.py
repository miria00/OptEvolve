"""The certified adaptive diagonal metric: its charge bounds the metric change, and it reduces to the scalar rule.

The controller may only move the metric if the change is paid for out of a summable budget. The charge is
eta = rho / (1 - theta) with rho the largest relative increase of any diagonal entry; this test verifies the
matrix inequality M' <= (1 + eta) M that the charge claims, on random admissible metrics.
"""
import numpy as np
import pytest


def _metric(L, T, S):
    return np.block([[np.diag(1.0 / T), -L.T], [-L, np.diag(1.0 / S)]])


def _theta(L, T, S):
    return float(np.linalg.norm(np.diag(np.sqrt(S)) @ L @ np.diag(np.sqrt(T)), 2))


def test_charge_bounds_the_metric_change_on_random_admissible_metrics():
    from atlas.typing_.rules import diag_change_cost
    rng = np.random.default_rng(0)
    tested = 0
    for trial in range(200):
        m, n = int(rng.integers(2, 8)), int(rng.integers(2, 8))
        L = rng.standard_normal((m, n)) * rng.choice([0.1, 1.0, 5.0])
        T, S = np.exp(rng.uniform(-4, 1, n)), np.exp(rng.uniform(-4, 1, m))
        th = _theta(L, T, S)
        if th >= 0.999:
            continue
        Tp = T * (rng.uniform(0.2, 5.0, n) if trial % 2 else np.full(n, rng.uniform(0.2, 5.0)))
        Sp = S * (rng.uniform(0.2, 5.0, m) if trial % 2 else np.full(m, rng.uniform(0.2, 5.0)))
        if _theta(L, Tp, Sp) >= 0.999:
            continue
        eta = diag_change_cost(T, S, Tp, Sp, th)
        M, Mp = _metric(L, T, S), _metric(L, Tp, Sp)
        slack = np.linalg.eigvalsh((1.0 + eta) * M - Mp).min()
        assert slack >= -1e-9 * max(1.0, abs(Mp).max()), f"charge too small: slack {slack:.3e}"
        tested += 1
    assert tested >= 40


def test_uniform_diagonals_reproduce_the_scalar_charge():
    from atlas.backend.adaptive import ratio_change_cost
    from atlas.typing_.rules import diag_change_cost
    t, s, tp, sp, opnorm, n, m = 0.3, 0.7, 0.21, 1.1, 1.3, 5, 4
    theta = np.sqrt(t * s) * opnorm
    scalar = ratio_change_cost(t, s, tp, sp, opnorm)
    diag = diag_change_cost(np.full(n, t), np.full(m, s), np.full(n, tp), np.full(m, sp), theta)
    assert diag == pytest.approx(scalar, rel=1e-12)


def test_gate_rejects_an_unbounded_budget_and_a_scheme_without_a_saddle_metric():
    from atlas.typing_ import TypeCheckError
    from atlas.typing_.rules import typecheck_diag_metric_schedule, typecheck_pdhg
    base = typecheck_pdhg(t=0.1, s=0.1, opnorm=1.0)
    typecheck_diag_metric_schedule(base, "pdhg", eta_budget=10.0, diag_range=1e4, fill=0.98)
    with pytest.raises(TypeCheckError):
        typecheck_diag_metric_schedule(base, "fbs", eta_budget=10.0, diag_range=1e4, fill=0.98)
    with pytest.raises(TypeCheckError):
        typecheck_diag_metric_schedule(base, "pdhg", eta_budget=float("inf"), diag_range=1e4, fill=0.98)
