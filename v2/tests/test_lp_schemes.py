"""Integration: LP composites solve through the GENERALIZED schemes.

B1C built the LP problem side (atlas.bench.lp_cell) and verified the
composite mapping with a standalone reference PDHG; these tests close the
loop: the certified schemes of atlas.schemes.base_schemes consume the
lp_standard composite directly (PDHG via the affine-tilt prox identity,
Condat-Vu with L_f = 0) and terminate on the LP duality-gap certificate
(atlas.certify.residuals.lp_rel_gap).

END-TO-END CONTRACT TEST: a 40x80 netflow LP (40 nodes, 80 arcs) solved
to rel gap 1e-6 with the objective matching HiGHS to 1e-5.
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pytest

from atlas.bench.lp_cell import (
    build_instances,
    from_netflow,
    lp_gap_terms,
    oracle,
    to_composite,
)
from atlas.bench.netflow import gen_regular_flow
from atlas.certify.residuals import lp_rel_gap
from atlas.genome import genome_from_dict
from atlas.schemes.base_schemes import (
    CondatVuParams,
    DRSParams,
    FBSParams,
    PDHGParams,
    solve_condat_vu,
    solve_drs,
    solve_fbs,
    solve_pdhg,
)
from atlas.schemes.compile import solve_genome
from atlas.typing_ import TypeCheckError
from atlas.wrappers.km import Budget


@pytest.fixture(scope="module")
def lp40x80():
    """The named end-to-end instance: 40 nodes x 80 arcs regular netflow."""
    return to_composite(from_netflow(gen_regular_flow(n_nodes=40, seed=7, degree=2)))


def _steps(problem, frac=0.9):
    nb = float(problem.L.norm_bound)
    return frac / nb, frac / nb


def test_e2e_40x80_netflow_pdhg_matches_highs(lp40x80):
    """THE end-to-end test: 40x80 netflow to rel gap 1e-6, objective vs
    HiGHS to 1e-5 (relative)."""
    prob = lp40x80
    assert np.asarray(prob.data["A_dense"]).shape == (40, 80)
    orc = oracle(prob)
    t, s = _steps(prob)
    budget = Budget(max_iters=1_000_000, tol=1e-6, sample_every=1000)
    r = solve_pdhg(prob, PDHGParams(t=t, s=s), budget)
    gap = float(r.rel_gap_history[-1])
    assert gap <= 1e-6
    P = float(np.dot(np.asarray(prob.data["c"]), np.maximum(r.x, 0.0)))
    rel_err = abs(P - orc["P_star"]) / (1.0 + abs(orc["P_star"]))
    assert rel_err <= 1e-5
    # the certificate really is the LP gap (feasibility + duality terms)
    terms = lp_gap_terms(r.x, r.u, prob.data["c"], prob.L, prob.data["b"])
    assert terms["cert"] <= 1e-6


def test_e2e_40x80_netflow_condat_vu_matches_highs(lp40x80):
    prob = lp40x80
    orc = oracle(prob)
    t, s = _steps(prob)
    budget = Budget(max_iters=1_000_000, tol=1e-6, sample_every=1000)
    r = solve_condat_vu(prob, CondatVuParams(t=t, s=s), budget)
    assert float(r.rel_gap_history[-1]) <= 1e-6
    P = float(np.dot(np.asarray(prob.data["c"]), np.maximum(r.x, 0.0)))
    assert abs(P - orc["P_star"]) / (1.0 + abs(orc["P_star"])) <= 1e-5


def test_pdhg_condat_vu_agree_on_dense_lp():
    """With L_f = 0 the Condat-Vu iteration reduces to PDHG through the
    tilt identity; both must reach the certificate and the same value."""
    (prob,), _ = build_instances(1, 0, family="dense", size=24, seed=11)
    t, s = _steps(prob)
    budget = Budget(max_iters=500_000, tol=1e-6, sample_every=1000)
    r1 = solve_pdhg(prob, PDHGParams(t=t, s=s), budget)
    r2 = solve_condat_vu(prob, CondatVuParams(t=t, s=s), budget)
    assert float(r1.rel_gap_history[-1]) <= 1e-6
    assert float(r2.rel_gap_history[-1]) <= 1e-6
    np.testing.assert_allclose(r1.x, r2.x, rtol=0, atol=1e-8)


def test_grid_family_solves():
    (prob,), _ = build_instances(1, 0, family="grid", size=4, seed=5)
    t, s = _steps(prob)
    budget = Budget(max_iters=500_000, tol=1e-6, sample_every=1000)
    r = solve_pdhg(prob, PDHGParams(t=t, s=s), budget)
    assert float(r.rel_gap_history[-1]) <= 1e-6
    orc = oracle(prob)
    P = float(np.dot(np.asarray(prob.data["c"]), np.maximum(r.x, 0.0)))
    assert abs(P - orc["P_star"]) / (1.0 + abs(orc["P_star"])) <= 1e-5


def test_solve_genome_runs_lp(lp40x80):
    """The evolution entry point handles lp_standard composites."""
    prob = lp40x80
    t, s = _steps(prob)
    g = genome_from_dict(
        {"scheme": "pdhg", "params": {"tau": t, "sigma": s, "rho": 1.5}}
    )
    budget = Budget(max_iters=500_000, tol=1e-6, sample_every=1000)
    r = solve_genome(prob, g, budget)
    assert float(r.rel_gap_history[-1]) <= 1e-6
    assert r.cert is not None and r.cert.satisfied


def test_fbs_drs_structurally_rejected_on_lp(lp40x80):
    """f = <c, x> is not strongly convex: the FBS/DRS dual paths must be
    certified construction failures, not crashes."""
    with pytest.raises(TypeCheckError):
        solve_fbs(lp40x80, FBSParams(step=0.1), Budget(max_iters=10, tol=1e-3))
    with pytest.raises(TypeCheckError):
        solve_drs(lp40x80, DRSParams(step=0.1), Budget(max_iters=10, tol=1e-3))


def test_lp_gap_certificate_nonnegative_on_feasible_pair(lp40x80):
    prob = lp40x80
    orc = oracle(prob)
    terms = lp_gap_terms(
        orc["x_star"], -orc["y_star"], prob.data["c"], prob.L, prob.data["b"]
    )
    assert terms["gap_raw"] >= -1e-9  # weak duality on the oracle pair
    assert terms["cert"] <= 1e-6


def test_lp_rel_gap_importable_from_residuals(lp40x80):
    """lp_rel_gap moved to atlas.certify.residuals; the bench alias must
    stay in sync (single implementation)."""
    import atlas.bench.lp_cell as lpc

    assert lpc.lp_rel_gap is lp_rel_gap
