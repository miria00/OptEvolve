"""The restart gene is REAL: fixed-k Halpern-anchor restarts are consumed
by the solvers, gated by Stage-0, and thread through the cert stack."""
import numpy as np
import pytest

from atlas import Budget, DRSParams, TypeCheckError, solve_drs, tv_denoise_problem
from atlas.genome import genome_from_dict, validate
from atlas.typing_ import typecheck_drs, typecheck_halpern, typecheck_restart


def _problem(tv_instance_32):
    y, lam = tv_instance_32
    return tv_denoise_problem(y, lam)


def test_restart_changes_the_iteration(tv_instance_32):
    """restart_k must actually be consumed: the anchored run with restarts
    produces a DIFFERENT (here: better) trajectory than without."""
    p = _problem(tv_instance_32)
    b = Budget(max_iters=3_000, tol=1e-8, sample_every=10)
    plain = solve_drs(p, DRSParams(step=0.5, halpern=True), b)
    restarted = solve_drs(p, DRSParams(step=0.5, halpern=True, restart_k=50), b)
    assert restarted.cert.scheme == "DRS+halpern+restart50"
    assert "anchor restart" in restarted.cert.citation
    # trajectories differ (the gene is consumed) ...
    n = min(len(plain.rel_gap_history), len(restarted.rel_gap_history))
    assert not np.allclose(
        plain.rel_gap_history[:n], restarted.rel_gap_history[:n]
    )
    # ... and restarting the anchor beats the cold anchor on this instance
    assert (restarted.iters, restarted.rel_gap_history[-1]) < (
        plain.iters, plain.rel_gap_history[-1]
    )


def test_restart_without_halpern_is_a_typecheck_error(tv_instance_32):
    p = _problem(tv_instance_32)
    with pytest.raises(TypeCheckError, match="Halpern"):
        solve_drs(p, DRSParams(step=0.5, restart_k=50), Budget(max_iters=10))


def test_restart_cert_passthrough():
    base = typecheck_halpern(typecheck_drs(0.5))
    cert = typecheck_restart(base, 100)
    # averagedness untouched: restart is a schedule, not an operator change
    assert cert.averagedness == base.averagedness
    assert cert.satisfied and cert.params["restart_k"] == 100
    with pytest.raises(TypeCheckError):
        typecheck_restart(typecheck_drs(0.5), 100)  # no halpern: nothing to reset


def test_genome_gate_restart_rules():
    ok = genome_from_dict(
        {"scheme": "drs", "params": {"tau": 0.5},
         "wrapper": {"halpern": True, "restart": "fixed_k", "restart_k": 50}}
    )
    v = validate(ok)
    assert v.ok and "restart50" in v.cert.scheme
    no_halpern = genome_from_dict(
        {"scheme": "drs", "params": {"tau": 0.5},
         "wrapper": {"halpern": False, "restart": "fixed_k", "restart_k": 50}}
    )
    v2 = validate(no_halpern)
    assert not v2.ok and "halpern" in v2.error.lower()


def test_restart_solver_matches_anchorless_when_k_huge(tv_instance_32):
    """restart_k beyond the budget never fires: identical to plain Halpern."""
    p = _problem(tv_instance_32)
    b = Budget(max_iters=500, tol=1e-10, sample_every=10)
    a = solve_drs(p, DRSParams(step=0.5, halpern=True), b)
    c = solve_drs(p, DRSParams(step=0.5, halpern=True, restart_k=10_000), b)
    np.testing.assert_allclose(a.rel_gap_history, c.rel_gap_history, rtol=0, atol=0)
    np.testing.assert_array_equal(a.x, c.x)
