"""Quasi-Newton primal metric for Condat-Vu on QP: operator, gate, admission, end to end."""
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.sparse as sp

jax.config.update("jax_enable_x64", True)
MM = Path(__file__).resolve().parents[1] / "data" / "maros_meszaros"


def _rand_qp(seed, n=7, m=5):
    from atlas.bench import qp_cell
    rng = np.random.default_rng(seed)
    B = rng.normal(size=(n, n)); P = B.T @ B * rng.uniform(0.5, 20) + 0.05 * np.eye(n)
    A = rng.normal(size=(m, n))
    d = {"P": sp.csc_matrix(P), "q": rng.normal(size=n), "r": 0.0, "A": sp.csc_matrix(A),
         "l": -rng.uniform(0.3, 2, size=m), "u": rng.uniform(0.3, 2, size=m)}
    return qp_cell.to_composite(d), P, A, rng


def test_lbfgs_operator_is_spd_and_satisfies_the_secant_equations():
    from atlas.operators.primal_metric import lbfgs_metric
    _, P, _, rng = _rand_qp(0)
    S = rng.normal(size=(4, 7)); Y = S @ P.T + 0.2 * S
    T = lbfgs_metric(S, Y)
    H = T.dense()
    assert np.allclose(H, H.T, atol=1e-10) and np.linalg.eigvalsh((H + H.T) / 2).min() > 0
    assert np.allclose(H @ Y[-1], S[-1], atol=1e-8)


@pytest.mark.parametrize("seed", range(6))
def test_admitted_iteration_is_averaged_in_the_claimed_metric(seed):
    """F = I + (T - I)/(2 alpha) must be firmly nonexpansive in M = [[T^{-1}, -A^T], [-A, I/s]]."""
    from atlas.operators.primal_metric import krylov_pairs, lbfgs_metric, scale_for_gate
    from atlas.schemes.base_schemes import CondatVuMetricParams, build_scheme
    prob, P, A, rng = _rand_qp(seed)
    if seed % 2:
        S, Y = krylov_pairs(prob.f.P.forward, 7, 3, mu=0.1, seed=seed)
    else:
        S = rng.normal(size=(3, 7)); Y = S @ P.T + 0.1 * S
    met, s, _, _ = scale_for_gate(lbfgs_metric(S, Y), prob.f.P, prob.L, 7, 5)
    built = build_scheme(prob, "condat_vu_metric", CondatVuMetricParams(s=s, metric=met))
    alpha = built.cert.averagedness
    assert 0.5 <= alpha < 1.0
    Tinv = np.linalg.inv(met.dense())
    M = np.block([[Tinv, -A.T], [-A, np.eye(5) / s]])
    assert np.linalg.eigvalsh(M).min() > 0
    for _ in range(25):
        z1, z2 = rng.normal(size=12), rng.normal(size=12)
        def F(z):
            st = (jnp.asarray(z[:7]), jnp.asarray(z[7:]))
            Tz = np.concatenate([np.asarray(v) for v in built.T(st, 0, built.dyn)])
            return z + (Tz - z) / (2 * alpha)
        p, d = F(z1) - F(z2), z1 - z2
        assert p @ M @ p <= (d @ M @ p) * (1 + 1e-9) + 1e-12


def test_gate_rejects_an_oversized_metric():
    from dataclasses import replace
    from atlas.operators.primal_metric import lbfgs_metric, scale_for_gate
    from atlas.schemes.base_schemes import CondatVuMetricParams, build_scheme
    from atlas.typing_ import TypeCheckError
    prob, P, A, rng = _rand_qp(3)
    S = rng.normal(size=(3, 7)); Y = S @ P.T + 0.1 * S
    met, s, _, _ = scale_for_gate(lbfgs_metric(S, Y), prob.f.P, prob.L, 7, 5)
    with pytest.raises(TypeCheckError):
        build_scheme(prob, "condat_vu_metric", CondatVuMetricParams(s=s, metric=replace(met, c=met.c * 2.5)))


@pytest.mark.skipif(not (MM / "HS21.mat").exists(), reason="Maros-Meszaros not downloaded")
def test_krylov_metric_certifies_a_real_qp():
    from atlas.bench import qp_cell
    from atlas.certify.residuals import qp_rel_kkt
    from atlas.genome import Backend
    from atlas.operators.primal_metric import krylov_pairs, lbfgs_metric, scale_for_gate
    from atlas.schemes.base_schemes import Budget, CondatVuMetricParams, build_scheme, run_built
    prob = qp_cell.load(str(MM / "HS21.mat"))
    n, m = prob.shape[0], int(prob.L.forward(jnp.zeros(prob.shape)).shape[0])
    S, Y = krylov_pairs(prob.f.P.forward, n, 2, mu=1e-3)
    met, s, _, _ = scale_for_gate(lbfgs_metric(S, Y), prob.f.P, prob.L, n, m)
    params = CondatVuMetricParams(s=s, metric=met)
    built = build_scheme(prob, "condat_vu_metric", params)
    r = run_built(built, built.cert, params, Budget(max_iters=20000, tol=1e-6, sample_every=200),
                  backend=Backend(precision="f64", K=64))
    kkt = float(qp_rel_kkt(jnp.asarray(r.x), jnp.asarray(r.u), prob.f.P, prob.f.q, prob.L, prob.h.lo, prob.h.hi))
    assert r.iters < 20000 and kkt <= 1e-6, (r.iters, kkt)
