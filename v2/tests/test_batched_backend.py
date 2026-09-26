"""Batched vmap backend (atlas.backend.batched): per-lane equivalence with
the sequential certified path, per-instance masked stopping, precision
rules, and the same-shape contract.

Equivalence protocol: run BOTH paths for a FIXED number of iterations
(tol = 0.0, max_iters = N, K | N) so neither stops early, then compare
the final primal iterate lane by lane at 1e-10 (f64). This pins the
batched step to the sequential step to roundoff, independent of stopping
cadence (the batched path certifies every K iterations, the sequential
km path every iteration).
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pytest
from types import SimpleNamespace

from atlas.backend.batched import batched_solve
from atlas.backend.precision import PrecisionError
from atlas.bench.lp_cell import from_netflow, gen_lp_dense, to_composite
from atlas.bench.netflow import gen_regular_flow
from atlas.certify.residuals import lp_rel_gap, rel_gap
from atlas.ir.spec import tv_denoise_problem
from atlas.schemes.base_schemes import (
    CondatVuParams,
    PDHGParams,
    solve_condat_vu,
    solve_pdhg,
)
from atlas.typing_ import TypeCheckError
from atlas.wrappers.km import Budget

TOL_EQ = 1e-10


def _tv_problems(n=3, size=24, seed=0):
    rng = np.random.default_rng(seed)
    return [
        tv_denoise_problem(rng.uniform(0.0, 1.0, size=(size, size)), lam=0.1)
        for _ in range(n)
    ]


def _lp_problems(n=3, n_nodes=24, seed=0, degree=4):
    return [
        to_composite(from_netflow(gen_regular_flow(
            n_nodes=n_nodes, seed=seed * 1000 + i, degree=degree)))
        for i in range(n)
    ]


def _fixed_budget(n_iters):
    return Budget(max_iters=n_iters, tol=0.0, sample_every=max(n_iters // 4, 1))


def _assert_lanes_match(res, problems, params, solver, n_iters):
    seq_budget = _fixed_budget(n_iters)
    for i, p in enumerate(problems):
        seq = solver(p, params, seq_budget)
        assert seq.iters == n_iters
        diff = float(np.max(np.abs(res.x[i] - np.asarray(seq.x))))
        assert diff <= TOL_EQ, f"lane {i}: max|x_b - x_s| = {diff:g}"
        if seq.u is not None:
            dif_u = float(np.max(np.abs(res.u[i] - np.asarray(seq.u))))
            assert dif_u <= TOL_EQ, f"lane {i}: max|u_b - u_s| = {dif_u:g}"


# ---------------------------------------------------------------------------
# Per-lane equivalence with the sequential path (fixed iterations, f64)
# ---------------------------------------------------------------------------


def test_tv_pdhg_batched_equals_sequential():
    problems = _tv_problems()
    params = PDHGParams(t=0.5, s=0.99 / 4.0)
    n_iters = 200
    res = batched_solve(problems, params, _fixed_budget(n_iters), K=50)
    assert res.batch_iters == n_iters
    assert not res.converged.any()  # tol=0 is unreachable
    _assert_lanes_match(res, problems, params, solve_pdhg, n_iters)


def test_tv_condat_vu_with_relaxation_batched_equals_sequential():
    problems = _tv_problems(seed=7)
    params = CondatVuParams(t=0.05, s=1.0, rho=1.5)
    n_iters = 120
    res = batched_solve(problems, params, _fixed_budget(n_iters), K=40)
    _assert_lanes_match(res, problems, params, solve_condat_vu, n_iters)


def test_lp_netflow_pdhg_batched_equals_sequential():
    problems = _lp_problems()
    nb = max(float(p.L.norm_bound) for p in problems)
    params = PDHGParams(t=0.9 / nb, s=0.9 / nb)
    n_iters = 300
    res = batched_solve(problems, params, _fixed_budget(n_iters), K=100)
    _assert_lanes_match(res, problems, params, solve_pdhg, n_iters)


def test_lp_netflow_halpern_restart_batched_equals_sequential():
    problems = _lp_problems(seed=3)
    nb = max(float(p.L.norm_bound) for p in problems)
    params = PDHGParams(t=0.9 / nb, s=0.9 / nb, halpern=True, restart_k=40)
    n_iters = 200
    res = batched_solve(problems, params, _fixed_budget(n_iters), K=50)
    _assert_lanes_match(res, problems, params, solve_pdhg, n_iters)


def test_lp_dense_pdhg_batched_equals_sequential():
    problems = [
        to_composite(gen_lp_dense(n=30, m=12, seed=s)) for s in (11, 12)
    ]
    nb = max(float(p.L.norm_bound) for p in problems)
    params = PDHGParams(t=0.9 / nb, s=0.9 / nb)
    n_iters = 150
    res = batched_solve(problems, params, _fixed_budget(n_iters), K=50)
    _assert_lanes_match(res, problems, params, solve_pdhg, n_iters)


# ---------------------------------------------------------------------------
# Per-instance stopping via masking
# ---------------------------------------------------------------------------


def test_per_instance_stopping_and_certificates():
    problems = _lp_problems(n=4, n_nodes=24, seed=5)
    nb = max(float(p.L.norm_bound) for p in problems)
    params = PDHGParams(t=0.95 / nb, s=0.95 / nb)
    tol = 1e-4
    res = batched_solve(
        problems, params, Budget(max_iters=200_000, tol=tol), K=200
    )
    assert res.converged.all(), f"gaps {res.certified_gap}"
    # Each lane's certificate holds on the RETURNED iterate, recomputed
    # independently at f64 (the gap that decided its termination).
    for i, p in enumerate(problems):
        g = float(lp_rel_gap(
            res.x[i], res.u[i], p.data["c"], p.L, p.data["b"]))
        assert g <= tol
    assert (res.iters <= res.batch_iters).all()
    assert (res.iters % res.K == 0).all()


def test_frozen_lane_matches_solo_batch():
    # A lane frozen by the done-mask must return EXACTLY the iterate it had
    # at its own convergence check: batch [easy together with others] and
    # batch [easy alone] agree bitwise at the same cadence K.
    problems = _lp_problems(n=3, n_nodes=24, seed=9)
    nb = max(float(p.L.norm_bound) for p in problems)
    params = PDHGParams(t=0.95 / nb, s=0.95 / nb)
    budget = Budget(max_iters=200_000, tol=1e-4)
    res_all = batched_solve(problems, params, budget, K=200)
    assert res_all.converged.all()
    for i in range(len(problems)):
        res_solo = batched_solve([problems[i]], params, budget, K=200)
        assert int(res_solo.iters[0]) == int(res_all.iters[i])
        assert float(np.max(np.abs(res_all.x[i] - res_solo.x[0]))) == 0.0
        assert float(np.max(np.abs(res_all.u[i] - res_solo.u[0]))) == 0.0


def test_tv_batched_converges_at_tol():
    problems = _tv_problems(n=2, size=24, seed=1)
    params = PDHGParams(t=0.5, s=0.99 / 4.0)
    tol = 1e-4
    res = batched_solve(problems, params, Budget(max_iters=50_000, tol=tol),
                        K=100)
    assert res.converged.all()
    for i, p in enumerate(problems):
        g = float(rel_gap(res.x[i], res.u[i], p.data["y"],
                          p.data["lam"], p.L))
        assert g <= tol


# ---------------------------------------------------------------------------
# Precision rules
# ---------------------------------------------------------------------------


def test_f32_cert64_certifies_at_f64():
    problems = _lp_problems(n=2, n_nodes=24, seed=2)
    nb = max(float(p.L.norm_bound) for p in problems)
    params = PDHGParams(t=0.95 / nb, s=0.95 / nb)
    tol = 1e-4
    res = batched_solve(
        problems, params, Budget(max_iters=200_000, tol=tol),
        precision_policy="f32_cert64", K=200,
    )
    assert res.precision == "f32_cert64"
    assert res.iterate_dtype == "float32"
    assert res.certificate_dtype == "float64"
    assert res.certified_gap.dtype == np.float64
    assert res.converged.all()
    for i, p in enumerate(problems):
        g = float(lp_rel_gap(
            res.x[i], res.u[i], p.data["c"], p.L, p.data["b"]))
        assert g <= tol


def test_schedule_policy_refused():
    problems = _lp_problems(n=2)
    params = PDHGParams(t=0.1, s=0.1)
    with pytest.raises(NotImplementedError, match="schedule"):
        batched_solve(problems, params, Budget(), precision_policy="schedule")


def test_sub_f32_precision_refused():
    problems = _lp_problems(n=2)
    params = PDHGParams(t=0.1, s=0.1)
    with pytest.raises(PrecisionError):
        batched_solve(problems, params, Budget(), precision_policy="bf16")


# ---------------------------------------------------------------------------
# Same-shape contract + gate behavior
# ---------------------------------------------------------------------------


def test_shape_mismatch_rejected():
    a = _tv_problems(n=1, size=24)[0]
    b = _tv_problems(n=1, size=32)[0]
    with pytest.raises(ValueError, match="same-shape"):
        batched_solve([a, b], PDHGParams(t=0.5, s=0.2), Budget())


def test_mixed_kind_rejected():
    tv = _tv_problems(n=1)[0]
    lp = _lp_problems(n=1)[0]
    with pytest.raises(ValueError, match="same-shape"):
        batched_solve([tv, lp], PDHGParams(t=0.5, s=0.2), Budget())


def test_stage0_gate_rejects_bad_steps():
    # Level-1 cert is checked against the WORST opnorm in the batch: a
    # stepsize pair violating t*s*||L||^2 < 1 must raise, not clamp.
    problems = _lp_problems(n=2)
    nb = max(float(p.L.norm_bound) for p in problems)
    with pytest.raises(TypeCheckError):
        batched_solve(problems, PDHGParams(t=2.0 / nb, s=2.0 / nb), Budget())


def test_genome_ducktype_adapter():
    # The adapter reads the current public Genome attribute names
    # (scheme / params.tau/sigma/rho / wrapper.halpern/restart_k / backend);
    # duck-typed here so this test does not depend on the in-flux genome
    # module. A real Genome exposes the same attributes.
    problems = _lp_problems(n=2, n_nodes=24, seed=4)
    nb = max(float(p.L.norm_bound) for p in problems)
    genome = SimpleNamespace(
        scheme="pdhg",
        params=SimpleNamespace(tau=0.9 / nb, sigma=0.9 / nb, rho=None),
        wrapper=SimpleNamespace(halpern=True, restart="fixed_k", restart_k=50),
        mix=None,
        backend=SimpleNamespace(precision="f32_cert64", K=100, theta=None,
                                batch=True),
    )
    res = batched_solve(problems, genome,
                        Budget(max_iters=100_000, tol=1e-4))
    assert res.scheme == "pdhg"
    assert res.precision == "f32_cert64"  # taken from the backend gene
    assert res.K == 100
    assert res.converged.all()


def test_mix_genome_refused():
    problems = _lp_problems(n=2)
    genome = SimpleNamespace(
        scheme="mix", params=SimpleNamespace(tau=0.1, sigma=0.1),
        wrapper=None, mix=object(), backend=None,
    )
    with pytest.raises(NotImplementedError, match="mix"):
        batched_solve(problems, genome, Budget())


def test_switch_k_genome_refused():
    """A real switch_k Genome hits the documented mix refusal."""
    from atlas.genome.schema import Genome, GenomeParams, Mix, SubScheme

    problems = _lp_problems(n=2)
    nb = max(float(p.L.norm_bound) for p in problems)
    genome = Genome(
        scheme="pdhg",
        params=GenomeParams(tau=0.9 / nb, sigma=0.9 / nb),
        mix=Mix(
            kind="switch_k", K=50,
            aggressive=SubScheme(scheme="pdhg", tau=1.5 / nb, sigma=1.5 / nb),
            tail=SubScheme(scheme="pdhg", tau=0.9 / nb, sigma=0.9 / nb),
        ),
    )
    with pytest.raises(NotImplementedError, match="mix"):
        batched_solve(problems, genome, Budget())


def test_metric_genome_refused():
    """A real metric (Ruiz) Genome hits the documented rescale refusal."""
    from atlas.genome.schema import Genome, GenomeParams, Metric

    problems = _lp_problems(n=2)
    nb = max(float(p.L.norm_bound) for p in problems)
    genome = Genome(
        scheme="pdhg",
        params=GenomeParams(tau=0.9 / nb, sigma=0.9 / nb),
        metric=Metric(kind="ruiz", iters=5),
    )
    with pytest.raises(NotImplementedError, match="metric"):
        batched_solve(problems, genome, Budget())


# ---------------------------------------------------------------------------
# Timing contract
# ---------------------------------------------------------------------------


def test_timing_fields_and_convention():
    problems = _lp_problems(n=3, n_nodes=24, seed=6)
    nb = max(float(p.L.norm_bound) for p in problems)
    params = PDHGParams(t=0.95 / nb, s=0.95 / nb)
    res = batched_solve(problems, params,
                        Budget(max_iters=100_000, tol=1e-4), K=200)
    assert res.batch_wall_s > 0.0
    assert res.n_instances == 3
    assert res.per_instance_effective_s == pytest.approx(
        res.batch_wall_s / 3.0)
