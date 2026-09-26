"""Certified adaptive step ratio: boundary placement, the metric-ordering bound, runner."""
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)
MM = Path(__file__).resolve().parents[1] / "data" / "maros_meszaros"


@pytest.mark.parametrize("L_f", [0.0, 3.7])
def test_boundary_steps_have_the_ratio_and_fill_the_gate(L_f):
    from atlas.backend.adaptive import boundary_steps
    for omega in (1e-3, 0.4, 1.0, 17.0, 3e4):
        t, s = boundary_steps(omega, opnorm=2.5, L_f=L_f, fill=0.98)
        assert t > 0 and s > 0
        assert s / t == pytest.approx(omega ** 2, rel=1e-10)
        assert t * L_f / 2 + t * s * 2.5 ** 2 == pytest.approx(0.98, rel=1e-10)


def test_ratio_change_cost_bounds_the_metric_ordering():
    """(1 + eta) M_n - M_{n+1} must be positive semidefinite for the charged eta."""
    from atlas.backend.adaptive import boundary_steps, ratio_change_cost
    rng = np.random.default_rng(0)
    L = rng.normal(size=(7, 5)); nb = np.linalg.norm(L, 2)

    def M(t, s):
        return np.block([[np.eye(5) / t, -L.T], [-L, np.eye(7) / s]])

    for _ in range(200):
        w0, w1 = np.exp(rng.uniform(-4, 4, size=2))
        Lf = float(rng.choice([0.0, rng.uniform(0.1, 5)]))
        t0, s0 = boundary_steps(w0, nb, Lf, 0.98)
        t1, s1 = boundary_steps(w1, nb, Lf, 0.98)
        eta = ratio_change_cost(t0, s0, t1, s1, nb)
        gap = (1 + eta) * M(t0, s0) - M(t1, s1)
        assert np.linalg.eigvalsh(gap).min() >= -1e-9 * np.abs(gap).max()


def test_gate_rejects_non_saddle_and_infinite_budget():
    from atlas.typing_ import TypeCheckError
    from atlas.typing_.rules import typecheck_metric_schedule, typecheck_pdhg
    base = typecheck_pdhg(t=0.5, s=0.5, opnorm=1.0)
    assert typecheck_metric_schedule(base, "pdhg", 10.0, 100.0, 0.98).satisfied
    for bad in (("fbs", 10.0, 100.0, 0.98), ("pdhg", float("inf"), 100.0, 0.98), ("pdhg", 10.0, 100.0, 1.0)):
        with pytest.raises(TypeCheckError):
            typecheck_metric_schedule(base, *bad)


def _hs(name):
    from atlas.bench import qp_cell
    from atlas.schemes.base_schemes import CondatVuParams, build_scheme
    prob = qp_cell.load(str(MM / f"{name}.mat"))
    Lf, nb = float(prob.f.props.lipschitz), float(prob.L.norm_bound)
    t = 1.0 / Lf; s = 0.49 / (t * nb ** 2)
    return prob, build_scheme(prob, "condat_vu", CondatVuParams(t=t, s=s), check=True)


@pytest.mark.skipif(not (MM / "HS21.mat").exists(), reason="Maros-Meszaros not downloaded")
def test_fixed_rule_matches_the_policy_runner():
    from atlas.backend.adaptive import boundary_steps, run_adaptive
    from atlas.genome import Backend
    from atlas.schemes.base_schemes import Budget, CondatVuParams, build_scheme, run_built
    prob, built = _hs("HS21")
    p = built.cert.params
    omega = (p["s"] / p["t"]) ** 0.5
    t, s = boundary_steps(omega, p["opnorm"], p["L_smooth"], 0.98)
    params = CondatVuParams(t=t, s=s)
    ref = run_built(build_scheme(prob, "condat_vu", params), None, params,
                    Budget(max_iters=20000, tol=1e-6, sample_every=200), backend=Backend(precision="f64", K=64))
    r = run_adaptive(built, built.cert, scheme="condat_vu", tol=1e-6, max_iters=20000, K=64, rule="fixed")
    assert r["iters"] == ref.iters and r["converged"]
    assert np.allclose(r["x"], ref.x, rtol=1e-10, atol=1e-12)


@pytest.mark.skipif(not (MM / "DUALC1.mat").exists(), reason="Maros-Meszaros not downloaded")
def test_pdlp_rule_certifies_and_respects_the_budget():
    from atlas.backend.adaptive import run_adaptive
    from atlas.certify.residuals import qp_rel_kkt
    prob, built = _hs("DUALC1")
    r = run_adaptive(built, built.cert, scheme="condat_vu", tol=1e-4, max_iters=100000, K=64,
                     rule="pdlp", eta_budget=500.0)
    assert r["eta_spent"] <= 500.0
    kkt = float(qp_rel_kkt(jnp.asarray(r["x"]), jnp.asarray(r["u"]), prob.f.P, prob.f.q, prob.L, prob.h.lo, prob.h.hi))
    if r["converged"]:
        assert kkt <= 1e-4
    r0 = run_adaptive(built, built.cert, scheme="condat_vu", tol=1e-4, max_iters=100000, K=64,
                      rule="pdlp", eta_budget=0.0)
    assert r0["n_updates"] == 0          # a zero budget freezes the ratio at the first move
