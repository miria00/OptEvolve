"""Metric gene: certified problem rescaling (Ruiz + Pock-Chambolle).

The gene compiles to an equivalently rescaled LP (atlas.schemes.rescale)
with the operator-norm bound of the rescaled matrix recomputed by power
iteration x1.02 and fed to the existing gates; the runtime certificate
and termination are the ORIGINAL problem's f64 duality gap. Tests:
gate accepts/rejects, norm-bound correctness (bound >= true norm on
random matrices), equilibration properties, original-gap certification,
solution match against the unscaled solve and the HiGHS oracle, and the
step-region expansion (steps valid only under the rescaled bound
certify with the gene and are rejected without it).
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pytest

from atlas.bench.lp_cell import (
    LPInstance,
    MatrixLinOp,
    from_netflow,
    gen_lp_dense,
    oracle,
    to_composite,
)
from atlas.bench.netflow import gen_regular_flow
from atlas.certify.residuals import lp_rel_gap
from atlas.genome import content_hash, genome_from_dict, validate
from atlas.schemes.compile import solve_genome
from atlas.schemes.rescale import (
    pock_chambolle_diagonals,
    power_iteration_norm_bound,
    ruiz_equilibrate,
)
from atlas.typing_ import TypeCheckError
from atlas.wrappers.km import Budget

TOL = 1e-5


def _pdhg(tau, sigma, metric=None):
    d = {"scheme": "pdhg", "params": {"tau": tau, "sigma": sigma}}
    if metric is not None:
        d["metric"] = metric
    return genome_from_dict(d)


@pytest.fixture(scope="module")
def lp40():
    return to_composite(
        from_netflow(gen_regular_flow(n_nodes=40, seed=7, degree=2))
    )


@pytest.fixture(scope="module")
def lp_badly_scaled():
    """A dense LP with rows/cols scaled across 4 orders of magnitude.

    Equivalent reformulation of a bounded feasible gen_lp_dense instance
    (x' = C^{-1} x_feas stays feasible; y' = R^{-1} y_aux stays dual
    feasible), so the oracle and duality certificates apply."""
    base = gen_lp_dense(n=24, m=12, seed=3)
    rng = np.random.default_rng(0)
    R = 10.0 ** rng.uniform(-2, 2, size=12)
    C = 10.0 ** rng.uniform(-2, 2, size=24)
    A = R[:, None] * base.A_dense * C[None, :]
    b = R * base.b
    c = C * base.c
    inst = LPInstance(
        c=c, b=b, A_dense=A, L=MatrixLinOp.from_dense(A), name="lp_bad", meta={}
    )
    return to_composite(inst)


# ---------------------------------------------------------------------------
# Stage-0 gate
# ---------------------------------------------------------------------------


class TestMetricGate:
    def test_ruiz_valid_on_lp(self):
        for iters in (0, 1, 10, 30):
            g = _pdhg(0.3, 0.3, {"kind": "ruiz", "iters": iters})
            v = validate(g, problem_kind="lp_netflow")
            assert v.ok, v.error

    def test_ruiz_bad_iters_rejected(self):
        for iters in (-1, 31, 1000):
            g = _pdhg(0.3, 0.3, {"kind": "ruiz", "iters": iters})
            v = validate(g, problem_kind="lp_netflow")
            assert not v.ok and "metric.iters" in v.error
        g = _pdhg(0.3, 0.3, {"kind": "ruiz"})  # iters missing
        assert not validate(g, problem_kind="lp_netflow").ok
        g = _pdhg(0.3, 0.3, {"kind": "ruiz", "iters": 5, "alpha": 1.0})
        v = validate(g, problem_kind="lp_netflow")
        assert not v.ok and "alpha" in v.error

    def test_pc_alpha_range(self):
        for alpha in (0.0, 1.0, 2.0):
            g = _pdhg(0.3, 0.3, {"kind": "pock_chambolle", "alpha": alpha})
            assert validate(g, problem_kind="lp_netflow").ok
        for alpha in (-0.1, 2.1, float("nan")):
            g = _pdhg(0.3, 0.3, {"kind": "pock_chambolle", "alpha": alpha})
            v = validate(g, problem_kind="lp_netflow")
            assert not v.ok and "metric.alpha" in v.error
        g = _pdhg(0.3, 0.3, {"kind": "pock_chambolle", "alpha": 1.0, "iters": 5})
        assert not validate(g, problem_kind="lp_netflow").ok

    def test_bad_kind_rejected_at_parse(self):
        with pytest.raises(Exception):
            _pdhg(0.3, 0.3, {"kind": "jacobi", "iters": 5})

    def test_metric_on_tv_rejected(self):
        g = _pdhg(0.05, 1.0, {"kind": "ruiz", "iters": 5})
        v = validate(g, problem_kind="tv_denoise")
        assert not v.ok and "inapplicable" in v.error

    def test_metric_on_mix_rejected(self):
        d = {
            "scheme": "mix",
            "mix": {
                "kind": "switch_k",
                "K": 100,
                "aggressive": {"scheme": "pdhg", "params": {"tau": 10.0, "sigma": 10.0}},
                "tail": {"scheme": "pdhg", "params": {"tau": 0.3, "sigma": 0.3}},
            },
            "metric": {"kind": "ruiz", "iters": 5},
        }
        v = validate(genome_from_dict(d), problem_kind="lp_netflow")
        assert not v.ok and "BASE genomes" in v.error

    def test_pc_gate_accepts_lemma2_steps(self):
        """Pock-Chambolle guarantees ||D1 A D2|| <= 1, so the gate soundly
        accepts steps sized for that bound even when they violate the
        unscaled constant."""
        g = _pdhg(0.85, 0.85, {"kind": "pock_chambolle", "alpha": 1.0})
        assert validate(g, problem_kind="lp_netflow").ok
        assert not validate(_pdhg(0.85, 0.85), problem_kind="lp_netflow").ok

    def test_hash_distinguishes_metric(self):
        g0 = _pdhg(0.3, 0.3)
        g1 = _pdhg(0.3, 0.3, {"kind": "ruiz", "iters": 5})
        g2 = _pdhg(0.3, 0.3, {"kind": "ruiz", "iters": 10})
        assert len({content_hash(g) for g in (g0, g1, g2)}) == 3


# ---------------------------------------------------------------------------
# Rescaling numerics
# ---------------------------------------------------------------------------


class TestRescaleNumerics:
    def test_norm_bound_upper_bounds_true_norm(self):
        rng = np.random.default_rng(42)
        for i in range(20):
            m = int(rng.integers(3, 30))
            n = int(rng.integers(3, 30))
            A = rng.standard_normal((m, n)) * (10.0 ** rng.uniform(-2, 2))
            nb = power_iteration_norm_bound(A, seed=i)
            true = float(np.linalg.svd(A, compute_uv=False)[0])
            assert nb >= true, (nb, true)
            assert nb <= 1.05 * true  # x1.02 safety, not a wild inflation

    def test_ruiz_balances_row_col_norms(self):
        rng = np.random.default_rng(1)
        A = rng.standard_normal((15, 25))
        A *= 10.0 ** rng.uniform(-3, 3, size=(15, 1))
        A *= 10.0 ** rng.uniform(-3, 3, size=(1, 25))
        d1, d2 = ruiz_equilibrate(A, 15)
        M = A * d1[:, None] * d2[None, :]
        rows = np.max(np.abs(M), axis=1)
        cols = np.max(np.abs(M), axis=0)
        assert np.all(rows > 0.5) and np.all(rows < 2.0)
        assert np.all(cols > 0.5) and np.all(cols < 2.0)

    def test_ruiz_zero_iters_is_identity(self):
        A = np.arange(12.0).reshape(3, 4)
        d1, d2 = ruiz_equilibrate(A, 0)
        assert np.all(d1 == 1.0) and np.all(d2 == 1.0)

    def test_pc_norm_at_most_one(self):
        rng = np.random.default_rng(2)
        for alpha in (0.0, 0.5, 1.0, 1.7, 2.0):
            A = rng.standard_normal((10, 18)) * (
                10.0 ** rng.uniform(-2, 2, size=(10, 1))
            )
            d1, d2 = pock_chambolle_diagonals(A, alpha)
            M = A * d1[:, None] * d2[None, :]
            true = float(np.linalg.svd(M, compute_uv=False)[0])
            assert true <= 1.0 + 1e-10  # Pock-Chambolle Lemma 2


# ---------------------------------------------------------------------------
# Solves: original-gap certification and solution match
# ---------------------------------------------------------------------------


class TestMetricSolve:
    def test_equilibrated_matrix_unchanged(self, lp40):
        """The netflow incidence matrix (entries +-1) is already Ruiz
        equilibrated: the gene must reproduce the plain solve exactly
        (invariance check; d1 = d2 = 1)."""
        nb = float(lp40.L.norm_bound)
        B = Budget(max_iters=200_000, tol=1e-6, sample_every=100)
        r0 = solve_genome(lp40, _pdhg(0.9 / nb, 0.9 / nb), B)
        rm = solve_genome(
            lp40, _pdhg(0.9 / nb, 0.9 / nb, {"kind": "ruiz", "iters": 10}), B
        )
        assert rm.iters == r0.iters
        np.testing.assert_allclose(rm.x, r0.x, rtol=1e-12, atol=1e-12)
        m = rm.extra["metric"]
        assert m["opnorm_rescaled"] == pytest.approx(m["opnorm_original"], rel=1e-9)

    def test_rescaled_solve_certifies_original_gap(self, lp_badly_scaled):
        """Steps sized for the RESCALED norm bound: certifies at TOL on
        the ORIGINAL problem's gap (recomputed independently), objective
        matches HiGHS; the same steps WITHOUT the gene are a certified
        construction failure. This badly scaled instance is unsolvable
        for plain PDHG in this budget, so the gene is load-bearing."""
        prob = lp_badly_scaled
        c = np.asarray(prob.data["c"])
        b = np.asarray(prob.data["b"])
        # recompute the rescaled bound the compiled path will use
        d1, d2 = ruiz_equilibrate(np.asarray(prob.data["A_dense"]), 10)
        nbs = power_iteration_norm_bound(
            np.asarray(prob.data["A_dense"]) * d1[:, None] * d2[None, :]
        )
        g = _pdhg(0.9 / nbs, 0.9 / nbs, {"kind": "ruiz", "iters": 10})
        B = Budget(max_iters=400_000, tol=TOL, sample_every=100)
        res = solve_genome(prob, g, B)
        hist = res.rel_gap_history[np.isfinite(res.rel_gap_history)]
        assert hist[-1] <= TOL
        # ORIGINAL-problem certificate, recomputed from the returned pair
        gap = float(lp_rel_gap(res.x, res.u, c, prob.L, b))
        assert gap <= TOL * 1.01
        # primal iterate maps back to (near) original feasibility
        xp = np.maximum(res.x, 0.0)
        pres = float(np.linalg.norm(prob.data["A_dense"] @ xp - b))
        assert pres <= TOL * (1.0 + float(np.linalg.norm(b))) * 1.01
        # objective matches the HiGHS oracle
        orc = oracle(prob)
        obj = float(np.vdot(xp, c))
        assert obj == pytest.approx(orc["P_star"], abs=1e-3 * (1 + abs(orc["P_star"])))
        # cert records the gene and the recomputed bound
        assert "metric[ruiz]" in res.cert.scheme
        assert res.cert.params["opnorm_rescaled"] == pytest.approx(nbs, rel=1e-9)
        assert "ORIGINAL problem's gap" in res.cert.condition_str
        m = res.extra["metric"]
        assert m["kind"] == "ruiz" and m["iters"] == 10
        assert m["opnorm_rescaled"] < m["opnorm_original"] / 100.0
        # WITHOUT the gene the same steps fail the mandatory typecheck
        with pytest.raises(TypeCheckError):
            solve_genome(prob, _pdhg(0.9 / nbs, 0.9 / nbs), B)

    def test_solution_matches_unscaled_solve(self, lp40):
        """On an instance both configurations can solve, the gene's
        answer agrees with the unscaled solve (objective; primal
        x-distance deliberately not compared, LP optima are non-unique)."""
        nb = float(lp40.L.norm_bound)
        c = np.asarray(lp40.data["c"])
        B = Budget(max_iters=200_000, tol=1e-6, sample_every=100)
        r0 = solve_genome(lp40, _pdhg(0.9 / nb, 0.9 / nb), B)
        rm = solve_genome(
            lp40,
            _pdhg(0.9 / nb, 0.9 / nb, {"kind": "pock_chambolle", "alpha": 1.0}),
            B,
        )
        gap = float(
            lp_rel_gap(rm.x, rm.u, c, lp40.L, np.asarray(lp40.data["b"]))
        )
        assert gap <= 1e-6 * 1.01
        obj0 = float(np.vdot(np.maximum(r0.x, 0.0), c))
        objm = float(np.vdot(np.maximum(rm.x, 0.0), c))
        assert objm == pytest.approx(obj0, abs=1e-4 * (1.0 + abs(obj0)))

    def test_pc_solve_on_badly_scaled(self, lp_badly_scaled):
        prob = lp_badly_scaled
        g = _pdhg(0.85, 0.85, {"kind": "pock_chambolle", "alpha": 1.0})
        assert validate(g, problem_kind="lp_netflow").ok  # Lemma-2 gate bound
        B = Budget(max_iters=400_000, tol=TOL, sample_every=100)
        res = solve_genome(prob, g, B)
        gap = float(
            lp_rel_gap(
                res.x, res.u, np.asarray(prob.data["c"]), prob.L,
                np.asarray(prob.data["b"]),
            )
        )
        assert gap <= TOL * 1.01
        assert res.extra["metric"]["opnorm_rescaled"] <= 1.02 * 1.0 + 1e-9

    def test_metric_with_f32_backend(self, lp40):
        """Regression: ScaledLinOp must preserve the iterate dtype (the
        IncidenceLinOp convention) so the f32-iterate policies run; the
        certificate stays f64 on the ORIGINAL problem's gap."""
        nb = float(lp40.L.norm_bound)
        g = genome_from_dict(
            {
                "scheme": "pdhg",
                "params": {"tau": 0.9 / nb, "sigma": 0.9 / nb},
                "metric": {"kind": "ruiz", "iters": 10},
                "backend": {"precision": "f32_cert64", "K": 100},
            }
        )
        assert validate(g, problem_kind="lp_netflow").ok
        res = solve_genome(
            lp40, g, Budget(max_iters=200_000, tol=1e-4, sample_every=100)
        )
        gap = float(
            lp_rel_gap(
                res.x, res.u, np.asarray(lp40.data["c"]), lp40.L,
                np.asarray(lp40.data["b"]),
            )
        )
        assert gap <= 1e-4 * 1.01
        assert res.extra["backend"]["certificate_dtype"] == "float64"
        assert res.extra["backend"]["iterate_dtypes"] == ["float32"]
        assert res.extra["metric"]["kind"] == "ruiz"

    def test_metric_rejected_on_tv_problem_at_solve(self, tv_instance_32):
        from atlas import tv_denoise_problem

        y, lam = tv_instance_32
        p = tv_denoise_problem(y, lam)
        g = _pdhg(0.05, 1.0, {"kind": "ruiz", "iters": 5})
        with pytest.raises(TypeCheckError):
            solve_genome(p, g, Budget(max_iters=1000, tol=1e-4, sample_every=100))


# ---------------------------------------------------------------------------
# Menus
# ---------------------------------------------------------------------------


def test_random_backend_proposes_metric_on_lp_only():
    from atlas.evolve.random_backend import RandomBackend

    be = RandomBackend(seed=3, problem_kind="lp_netflow")
    genomes = be.ask(200)
    kinds = {g.metric.kind for g in genomes if g.metric is not None}
    assert "ruiz" in kinds
    for g in genomes:
        assert validate(g, problem_kind="lp_netflow").ok
        if g.mix is not None:
            assert g.mix.kind != "switch"
    be_tv = RandomBackend(seed=3, problem_kind="tv_denoise")
    assert all(g.metric is None for g in be_tv.ask(200))


def test_llm_menu_documents_metric():
    from atlas.evolve.llm_backend import MUTATION_MENU
    from atlas.genome import SCHEMA_DOC

    assert "metric" in MUTATION_MENU and "ruiz" in MUTATION_MENU
    assert "ruiz" in SCHEMA_DOC and "pock_chambolle" in SCHEMA_DOC
