"""Each scheme reaches rel_gap <= 1e-6 on a 32x32 synthetic TV instance;
typechecks reject violating stepsizes; the duality gap is rigorous
(nonnegative) and monotone-ish."""
import jax.numpy as jnp
import numpy as np
import pytest

from atlas import (
    Budget,
    CondatVuParams,
    DRSParams,
    FBSParams,
    PDHGParams,
    TypeCheckError,
    solve_condat_vu,
    solve_drs,
    solve_fbs,
    solve_pdhg,
    tv_denoise_problem,
)
from atlas.certify.residuals import rel_gap, tv_gap

TOL = 1e-6


def _problem(tv_instance_32):
    y, lam = tv_instance_32
    return tv_denoise_problem(y, lam)


def _check_result(res, problem, tol=TOL):
    y, lam, D = problem.data["y"], problem.data["lam"], problem.L
    x, u = jnp.array(res.x), jnp.array(res.u)
    # dual point feasible, gap rigorous
    assert float(jnp.max(jnp.abs(u))) <= lam + 1e-12
    g = float(tv_gap(x, u, y, lam, D))
    assert g >= -1e-12
    rg = float(rel_gap(x, u, y, lam, D))
    assert rg <= tol
    # gap history: nonnegative everywhere, monotone-ish, big net decrease
    hist = res.rel_gap_history
    hist = hist[np.isfinite(hist)]
    assert np.all(hist >= -1e-12)
    # substantial net decrease (final <= tol is asserted above already)
    assert hist[-1] <= max(hist[0] * 1e-2, 2.0 * tol)
    # monotone-ish: each sampled value below 2x the running minimum so far
    running_min = np.minimum.accumulate(hist)
    assert np.all(hist <= 2.0 * np.maximum(running_min, TOL) + 1e-12)
    # fp residual decreased
    r = res.fp_residual_history
    r = r[np.isfinite(r)]
    assert r[-1] < r[0]


def test_fbs_solves_tv(tv_instance_32):
    p = _problem(tv_instance_32)
    res = solve_fbs(p, FBSParams(step=0.99 * 2.0 / 8.0), Budget(max_iters=400_000, tol=TOL, sample_every=200))
    assert res.cert.satisfied and "2.7" in res.cert.citation
    _check_result(res, p)


def test_drs_solves_tv(tv_instance_32):
    p = _problem(tv_instance_32)
    res = solve_drs(p, DRSParams(step=0.5), Budget(max_iters=20_000, tol=TOL, sample_every=10))
    assert res.cert.satisfied and res.cert.averagedness == 0.5
    _check_result(res, p)


def test_pdhg_solves_tv(tv_instance_32):
    p = _problem(tv_instance_32)
    t = 0.5
    s = 0.99 / (8.0 * t)
    res = solve_pdhg(p, PDHGParams(t=t, s=s), Budget(max_iters=200_000, tol=TOL, sample_every=100))
    assert res.cert.satisfied and "3.3" in res.cert.citation
    _check_result(res, p)


def test_condat_vu_solves_tv(tv_instance_32):
    p = _problem(tv_instance_32)
    t = 0.5
    s = 0.95 * (1.0 - t * 1.0 / 2.0) / (t * 8.0)
    res = solve_condat_vu(p, CondatVuParams(t=t, s=s), Budget(max_iters=200_000, tol=TOL, sample_every=100))
    assert res.cert.satisfied and "3.13" in res.cert.citation
    _check_result(res, p)


def test_schemes_agree(tv_instance_32):
    p = _problem(tv_instance_32)
    res1 = solve_pdhg(p, PDHGParams(t=0.5, s=0.99 / 4.0), Budget(max_iters=200_000, tol=TOL, sample_every=100))
    res2 = solve_drs(p, DRSParams(step=0.5), Budget(max_iters=20_000, tol=TOL, sample_every=10))
    # f is 1-strongly convex: ||x - x*||^2 <= 2*gap for each solver
    assert np.max(np.abs(res1.x - res2.x)) < 1e-2


# ---------------------------------------------------------------------------
# typecheck rejections
# ---------------------------------------------------------------------------


def test_fbs_rejects_large_step(tv_instance_32):
    p = _problem(tv_instance_32)
    with pytest.raises(TypeCheckError):
        solve_fbs(p, FBSParams(step=2.0 / 8.0 + 1e-6), Budget(max_iters=10))
    with pytest.raises(TypeCheckError):
        solve_fbs(p, FBSParams(step=-0.1), Budget(max_iters=10))


def test_drs_rejects_nonpositive_step(tv_instance_32):
    p = _problem(tv_instance_32)
    with pytest.raises(TypeCheckError):
        solve_drs(p, DRSParams(step=0.0), Budget(max_iters=10))


def test_drs_rejects_infinite_step(tv_instance_32):
    """Regression: solve_drs(step=inf) used to run and emit a satisfied cert."""
    p = _problem(tv_instance_32)
    with pytest.raises(TypeCheckError):
        solve_drs(p, DRSParams(step=float("inf")), Budget(max_iters=10))


def test_rho_plus_halpern_is_admitted(tv_instance_32):
    """relax-then-anchor is admissible: the anchor needs only a nonexpansive
    map, so rho in (0, 2] with rho*alpha <= 1 is enough (Sun, Yuan, Zhang &
    Zhao SIOPT 35(2) 2025). DRS is (1/2)-averaged, so rho = 1.5 gives
    rho*alpha = 0.75 <= 1."""
    p = _problem(tv_instance_32)
    res = solve_drs(p, DRSParams(step=1.0, rho=1.5, halpern=True),
                    Budget(max_iters=10))
    assert res.cert.satisfied
    assert res.cert.params["relax_with_anchor"] is True


def test_reflection_needs_the_anchor(tv_instance_32):
    """rho = 2 on a (1/2)-averaged map gives rho*alpha = 1: NONEXPANSIVE but
    not averaged. With the anchor it is admissible; WITHOUT the anchor plain
    KM can cycle, so it must still be refused. This boundary is the whole
    reason the wider regime is gated on `with_anchor`."""
    p = _problem(tv_instance_32)
    ok = solve_drs(p, DRSParams(step=1.0, rho=2.0, halpern=True),
                   Budget(max_iters=10))
    assert ok.cert.satisfied
    with pytest.raises(TypeCheckError):
        solve_drs(p, DRSParams(step=2.0, rho=2.0), Budget(max_iters=10))


def test_pdhg_rejects_ts_violation(tv_instance_32):
    p = _problem(tv_instance_32)
    with pytest.raises(TypeCheckError):
        solve_pdhg(p, PDHGParams(t=0.5, s=1.0 / 4.0), Budget(max_iters=10))  # t*s*8 = 1


def test_condat_vu_rejects_violation(tv_instance_32):
    p = _problem(tv_instance_32)
    # t*L_f/2 + t*s*8 = 0.25 + 0.8 > 1
    with pytest.raises(TypeCheckError):
        solve_condat_vu(p, CondatVuParams(t=0.5, s=0.2), Budget(max_iters=10))


# ---------------------------------------------------------------------------
# wrappers
# ---------------------------------------------------------------------------


def test_fbs_overrelaxed(tv_instance_32):
    p = _problem(tv_instance_32)
    a = 0.9 * 2.0 / 8.0
    alpha = 2.0 / (4.0 - a * 8.0)
    rho = 0.95 / alpha
    res = solve_fbs(p, FBSParams(step=a, rho=rho), Budget(max_iters=400_000, tol=TOL, sample_every=200))
    assert res.cert.scheme == "FBS+relax"
    assert res.cert.averagedness == pytest.approx(rho * alpha)
    _check_result(res, p)


def test_fbs_rejects_bad_relaxation(tv_instance_32):
    p = _problem(tv_instance_32)
    a = 1.9 / 8.0  # alpha = 2/(4-1.9*... close to 1 -> rho=1.9 invalid
    with pytest.raises(TypeCheckError):
        solve_fbs(p, FBSParams(step=a, rho=1.9), Budget(max_iters=10))


def test_wall_time_excludes_jit_compile(tv_instance_32):
    """Regression: wall_time used to include trace+compile (re-paid every
    call), so time-to-tol and fitness were compile-dominated. A 2-iteration
    solve executes in ~1 ms; compile costs ~0.1 s and must land OUTSIDE
    result.wall_time."""
    import time

    p = _problem(tv_instance_32)
    b = Budget(max_iters=2, tol=0.0, sample_every=1)
    t0 = time.perf_counter()
    res = solve_pdhg(p, PDHGParams(t=0.5, s=0.99 / 4.0), b)
    elapsed = time.perf_counter() - t0
    assert res.iters == 2
    assert res.wall_time < elapsed
    # compile overhead (>= tens of ms) is excluded from the reported time
    assert elapsed - res.wall_time > 0.02
    # 2 iterations on a 32x32 patch execute in well under 50 ms
    assert res.wall_time < 0.05


def test_drs_halpern_runs(tv_instance_32):
    """Halpern anchor on the nonexpansive DRS map converges (slower rate)."""
    p = _problem(tv_instance_32)
    res = solve_drs(
        p, DRSParams(step=0.5, halpern=True), Budget(max_iters=3_000, tol=1e-3, sample_every=10)
    )
    assert res.cert.scheme == "DRS+halpern"
    assert "12.2" in res.cert.citation
    hist = res.rel_gap_history
    hist = hist[np.isfinite(hist)]
    assert hist[-1] < hist[0]
