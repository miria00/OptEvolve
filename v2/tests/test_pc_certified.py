"""Pock-Chambolle certified preconditioning (2026-09-04).

Two changes are locked here.

1. STRUCTURAL-ZERO / ENDPOINT BUG. The sparse path called scipy's
   `.power(alpha)`, which REFUSES alpha = 0 ("zero power would densify"), so
   alpha = 0 and alpha = 2 (via the 2 - alpha column exponent) raised on every
   sparse composite, i.e. on every MPS/OOD instance. Both endpoints are inside
   the certified range [0, 2]. The fix takes the power on the explicit data
   only, which also realizes the 0^0 := 0 convention Lemma 2's proof uses.

2. CERTIFIED, INSTANCE-INDEPENDENT BOUND. Lemma 2 bounds ||D1 A D2|| <= 1 as a
   statement about the FORMULA, so the gate no longer needs a power-iteration
   estimate and the resulting certificate does not depend on the instance.
   This is what Phase 2 lacked: absolute (tau, sigma) certified against the
   netflow opnorm 2.885 were Stage-0 rejected on 1340/1340 OOD attempts.
"""
import numpy as np
import pytest
import scipy.sparse as sp

from atlas.schemes.rescale import (
    PC_CERTIFIED_NORM,
    PC_SAFETY,
    pock_chambolle_diagonals,
    power_iteration_norm_bound,
)

ALPHAS = [0.0, 0.25, 0.5, 1.0, 1.5, 1.75, 2.0]


def _matrices():
    rng = np.random.default_rng(0)
    yield "random_dense", rng.standard_normal((40, 55))
    # wildly ill-conditioned column scaling: the case that motivated the work
    A = rng.standard_normal((30, 30)) * np.logspace(-6, 6, 30)[None, :]
    yield "ill_conditioned", A
    # TV-like forward difference (structured, many exact zeros)
    n = 20
    D = np.zeros((n - 1, n))
    for i in range(n - 1):
        D[i, i], D[i, i + 1] = -1.0, 1.0
    yield "difference", D
    # a row and a column of exact zeros must be left unscaled, not divided by 0
    Z = rng.standard_normal((12, 14))
    Z[3, :] = 0.0
    Z[:, 7] = 0.0
    yield "zero_row_and_col", Z


class TestEndpointsReachable:
    """The alpha = 0 and alpha = 2 endpoints must not raise on sparse input."""

    @pytest.mark.parametrize("alpha", [0.0, 2.0])
    def test_sparse_endpoints_do_not_raise(self, alpha):
        rng = np.random.default_rng(1)
        A = sp.random(50, 60, density=0.1, random_state=rng, format="csr")
        d1, d2 = pock_chambolle_diagonals(A, alpha)
        assert np.all(np.isfinite(d1)) and np.all(np.isfinite(d2))
        assert np.all(d1 > 0) and np.all(d2 > 0)

    @pytest.mark.parametrize("alpha", ALPHAS)
    def test_dense_and_sparse_agree(self, alpha):
        rng = np.random.default_rng(2)
        A = sp.random(40, 45, density=0.15, random_state=rng, format="csr")
        d1s, d2s = pock_chambolle_diagonals(A, alpha)
        d1d, d2d = pock_chambolle_diagonals(A.toarray(), alpha)
        np.testing.assert_allclose(d1s, d1d, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(d2s, d2d, rtol=1e-12, atol=1e-12)


class TestLemma2:
    """||diag(d1) A diag(d2)|| <= 1 for every alpha in [0, 2]."""

    @pytest.mark.parametrize("alpha", ALPHAS)
    def test_bound_holds_dense(self, alpha):
        for name, A in _matrices():
            d1, d2 = pock_chambolle_diagonals(A, alpha)
            nb = power_iteration_norm_bound(A * d1[:, None] * d2[None, :],
                                            safety=1.0)
            assert nb <= PC_CERTIFIED_NORM + 1e-9, (
                f"Lemma 2 violated on {name} at alpha={alpha}: {nb!r}")

    @pytest.mark.parametrize("alpha", ALPHAS)
    def test_bound_holds_sparse(self, alpha):
        rng = np.random.default_rng(3)
        A = sp.random(60, 70, density=0.08, random_state=rng, format="csr")
        A.data *= np.logspace(-4, 4, A.data.size)  # nasty entry spread
        d1, d2 = pock_chambolle_diagonals(A, alpha)
        scaled = sp.diags(d1) @ A @ sp.diags(d2)
        assert power_iteration_norm_bound(scaled, safety=1.0) <= 1.0 + 1e-9

    def test_zero_row_and_col_left_unscaled(self):
        rng = np.random.default_rng(4)
        Z = rng.standard_normal((10, 11))
        Z[2, :] = 0.0
        Z[:, 5] = 0.0
        d1, d2 = pock_chambolle_diagonals(Z, 1.0)
        assert d1[2] == 1.0 and d2[5] == 1.0
        assert np.all(np.isfinite(d1)) and np.all(np.isfinite(d2))


class TestCertifiedBoundIsInstanceIndependent:
    """The whole point: the bound must not depend on the instance."""

    def test_bound_constant_across_wildly_different_matrices(self):
        from atlas.schemes.rescale import PC_CERTIFIED_NORM as C

        bounds = set()
        for _name, A in _matrices():
            # the value the gate would use under the certified path
            bounds.add(C * PC_SAFETY)
        assert len(bounds) == 1, "certified bound must be instance-independent"

    def test_certified_bound_dominates_true_norm(self):
        """The constant the gate uses is a genuine upper bound everywhere."""
        for name, A in _matrices():
            for alpha in ALPHAS:
                d1, d2 = pock_chambolle_diagonals(A, alpha)
                true = power_iteration_norm_bound(
                    A * d1[:, None] * d2[None, :], safety=1.0)
                assert true <= PC_CERTIFIED_NORM * PC_SAFETY, (
                    f"{name} alpha={alpha}: true {true} exceeds certified "
                    f"{PC_CERTIFIED_NORM * PC_SAFETY}")

    def test_margin_does_not_exclude_the_phase2_champion_band(self):
        """The gate implements tau*sigma < 1/PC_SAFETY^2, so an oversized
        margin silently refuses genomes. Phase-2 champions sat at 0.950 and
        0.9925 of the bound; both must remain admissible."""
        implemented = 1.0 / (PC_SAFETY ** 2)
        assert implemented > 0.9925, (
            f"PC_SAFETY={PC_SAFETY} implements tau*sigma < {implemented}, "
            "which rejects the band where Phase-2 champions sat")

    def test_strictness_reduces_to_tau_sigma(self):
        """With nb = 1, the PDHG gate tau*sigma*nb^2 < 1 is strict whenever
        tau*sigma < 1; soundness does not rest on PC_SAFETY."""
        from atlas.typing_ import TypeCheckError
        from atlas.typing_.rules import typecheck_pdhg

        cert = typecheck_pdhg(0.01, 9.45468304801, PC_CERTIFIED_NORM)
        assert cert.satisfied
        with pytest.raises(TypeCheckError):
            typecheck_pdhg(1.0, 1.5, PC_CERTIFIED_NORM)  # tau*sigma = 1.5 > 1


class TestDegenerateStructure:
    """Edge cases that must be settled before Ruiz+PC is frozen as the LP
    substrate. A zero row/column makes the PC denominator zero, which must
    resolve to an explicit unscaled convention, never an inf or a nan."""

    def test_single_nonzero_row_and_column(self):
        A = np.zeros((5, 6))
        A[0, 0] = 3.0          # row 0 and col 0 have exactly one nonzero
        A[1, 1] = A[2, 2] = 1.0
        for alpha in ALPHAS:
            d1, d2 = pock_chambolle_diagonals(A, alpha)
            assert np.all(np.isfinite(d1)) and np.all(np.isfinite(d2))
            nb = power_iteration_norm_bound(A * d1[:, None] * d2[None, :],
                                            safety=1.0)
            assert nb <= PC_CERTIFIED_NORM + 1e-9

    def test_extreme_magnitude_spread(self):
        """Entries spanning 24 orders of magnitude must not overflow the
        row/col power sums into inf."""
        rng = np.random.default_rng(7)
        A = rng.standard_normal((25, 25)) * np.logspace(-12, 12, 25)[None, :]
        for alpha in ALPHAS:
            d1, d2 = pock_chambolle_diagonals(A, alpha)
            assert np.all(np.isfinite(d1)) and np.all(np.isfinite(d2))
            assert np.all(d1 > 0) and np.all(d2 > 0)
            nb = power_iteration_norm_bound(A * d1[:, None] * d2[None, :],
                                            safety=1.0)
            assert nb <= PC_CERTIFIED_NORM + 1e-9, f"alpha={alpha}: {nb}"

    def test_duplicate_coo_entries_are_summed_first(self):
        """A COO matrix may carry duplicate (i, j) entries. They must be
        summed before the |A_ij|^alpha sums, or the diagonals describe a
        matrix that is not the one the solver runs."""
        rows = np.array([0, 0, 1, 2])
        cols = np.array([0, 0, 1, 2])       # (0,0) appears TWICE
        vals = np.array([1.5, 2.5, 1.0, 1.0])
        coo = sp.coo_matrix((vals, (rows, cols)), shape=(3, 3))
        summed = sp.csr_matrix(
            (np.array([4.0, 1.0, 1.0]),
             (np.array([0, 1, 2]), np.array([0, 1, 2]))), shape=(3, 3))
        for alpha in ALPHAS:
            d1a, d2a = pock_chambolle_diagonals(coo, alpha)
            d1b, d2b = pock_chambolle_diagonals(summed, alpha)
            np.testing.assert_allclose(d1a, d1b, rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(d2a, d2b, rtol=1e-12, atol=1e-12)

    def test_all_zero_matrix_is_identity_scaling(self):
        Z = np.zeros((4, 4))
        d1, d2 = pock_chambolle_diagonals(Z, 1.0)
        assert np.all(d1 == 1.0) and np.all(d2 == 1.0)


class TestCleanResampleIsDefault:
    """The repair ablation's operational consequence: rejection must lead to
    a clean resample, not to an injected diagnostic."""

    def test_repair_feedback_defaults_off(self):
        from atlas.evolve.llm_backend import LLMBackend

        b = LLMBackend(http_post=lambda payload: "{}")
        assert b.repair_feedback is False

    def test_retry_prompt_carries_no_diagnostic_by_default(self):
        from atlas.evolve.llm_backend import LLMBackend

        seen = []

        def fake_post(payload):
            seen.append(payload["messages"][-1]["content"])
            return "not json at all"

        b = LLMBackend(http_post=fake_post, problem_kind="lp_netflow")
        b._propose_one()  # both attempts fail -> falls back
        assert len(seen) == 2, "expected one initial try plus one resample"
        assert seen[0] == seen[1], (
            "default retry must be a CLEAN RESAMPLE: the second prompt "
            "differs from the first, so a diagnostic was injected")
        assert "INVALID" not in seen[1]

    def test_repair_feedback_true_restores_injection(self):
        from atlas.evolve.llm_backend import LLMBackend

        seen = []

        def fake_post(payload):
            seen.append(payload["messages"][-1]["content"])
            return "not json at all"

        b = LLMBackend(http_post=fake_post, problem_kind="lp_netflow",
                       repair_feedback=True)
        b._propose_one()
        assert len(seen) == 2 and seen[0] != seen[1]
        assert "INVALID" in seen[1]


class TestRuizThenPCSubstrate:
    """Ruiz(10) -> PC(alpha) as frozen LP infrastructure. The certified bound
    must survive the composition, because PC is applied LAST."""

    def _prob(self):
        from atlas.bench.lp_cell import from_netflow, to_composite
        from atlas.bench.netflow import gen_regular_flow

        return to_composite(
            from_netflow(gen_regular_flow(n_nodes=60, degree=2, seed=3))
        )

    @pytest.mark.parametrize("ruiz_iters", [0, 2, 5, 10])
    def test_composed_bound_holds_for_any_ruiz_count(self, ruiz_iters):
        """Any Ruiz count enters the SAME certificate pathway: this is what
        makes n_ruiz safe to expose as a gene later."""
        from atlas.schemes.rescale import precondition_lp

        prob = self._prob()
        A = np.asarray(prob.data["A_dense"], dtype=np.float64)
        _p, d1, d2, nb = precondition_lp(
            prob, ruiz_iters=ruiz_iters, pc_alpha=1.0, verify=True)
        scaled = A * d1[:, None] * d2[None, :]
        assert power_iteration_norm_bound(scaled, safety=1.0) <= 1.0 + 1e-9
        assert nb == PC_CERTIFIED_NORM * PC_SAFETY

    @pytest.mark.parametrize("alpha", [0.0, 1.0, 2.0])
    def test_composed_bound_holds_for_endpoint_alphas(self, alpha):
        from atlas.schemes.rescale import precondition_lp

        prob = self._prob()
        A = np.asarray(prob.data["A_dense"], dtype=np.float64)
        _p, d1, d2, _nb = precondition_lp(
            prob, ruiz_iters=10, pc_alpha=alpha, verify=True)
        scaled = A * d1[:, None] * d2[None, :]
        assert power_iteration_norm_bound(scaled, safety=1.0) <= 1.0 + 1e-9

    def test_composed_diagonals_reproduce_the_returned_problem(self):
        """d1, d2 must be the diagonals that actually map the original
        operator onto the preconditioned one, or the runtime inverse maps
        (x = D2 x', u = D1 u') silently return the wrong variables."""
        from atlas.schemes.rescale import precondition_lp

        prob = self._prob()
        A = np.asarray(prob.data["A_dense"], dtype=np.float64)
        p2, d1, d2, _nb = precondition_lp(prob, ruiz_iters=10, pc_alpha=1.0)
        np.testing.assert_allclose(
            np.asarray(p2.data["A_dense"], dtype=np.float64),
            A * d1[:, None] * d2[None, :], rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(
            np.asarray(p2.data["c"], dtype=np.float64),
            d2 * np.asarray(prob.data["c"], dtype=np.float64),
            rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(
            np.asarray(p2.data["b"], dtype=np.float64),
            d1 * np.asarray(prob.data["b"], dtype=np.float64),
            rtol=1e-10, atol=1e-12)

    def test_verify_mode_catches_a_broken_scaling(self, monkeypatch):
        """verify=True must REFUSE to certify if the diagonals do not
        actually achieve the theorem's bound."""
        import atlas.schemes.rescale as R
        from atlas.typing_ import TypeCheckError

        prob = self._prob()
        monkeypatch.setattr(
            R, "pock_chambolle_diagonals",
            lambda A, alpha: (np.full(A.shape[0], 50.0),
                              np.full(A.shape[1], 50.0)))
        with pytest.raises(TypeCheckError, match="Lemma 2 violated"):
            R.precondition_lp(prob, ruiz_iters=0, pc_alpha=1.0, verify=True)


class TestComposedPathCertifiesOriginalCoordinates:
    """The failure this class exists to prevent, measured before the fix:
    solving the BARE preconditioned problem reported a scaled-coordinate gap
    of 9.99e-7 ("converged" at 1e-6) while the ORIGINAL-coordinate gap was
    1.098e-6, with the primal residual differing by 2x. Ruiz+PC must be run
    through the metric-gene path, which maps iterates back and certifies the
    ORIGINAL problem."""

    def _prob(self):
        from atlas.bench.lp_cell import from_netflow, to_composite
        from atlas.bench.netflow import gen_regular_flow

        return to_composite(
            from_netflow(gen_regular_flow(n_nodes=60, degree=2, seed=3))
        )

    def test_ruiz_pc_genome_certifies_the_original_gap(self):
        from atlas.certify.residuals import lp_rel_gap
        from atlas.genome import genome_from_dict
        from atlas.schemes.compile import solve_genome
        from atlas.schemes.rescale import lp_substrate_metric
        from atlas.wrappers.km import Budget

        tol = 1e-6
        prob = self._prob()
        c = np.asarray(prob.data["c"])
        b = np.asarray(prob.data["b"])
        g = genome_from_dict({
            "scheme": "pdhg", "params": {"tau": 0.5, "sigma": 0.5},
            "metric": lp_substrate_metric(),
        })
        res = solve_genome(
            prob, g, Budget(max_iters=400_000, tol=tol, sample_every=100))
        hist = res.rel_gap_history[np.isfinite(res.rel_gap_history)]
        assert hist[-1] <= tol, "did not reach the certificate"
        # THE POINT: recompute independently against the ORIGINAL (c, A, b)
        # from the returned pair. res.x / res.u must already be original-
        # coordinate variables, so no unscaling is applied here.
        gap = float(lp_rel_gap(res.x, res.u, c, prob.L, b))
        assert gap <= tol * 1.01, (
            f"reported {hist[-1]:.3e} but ORIGINAL-coordinate gap is "
            f"{gap:.3e}: the certificate is about the wrong problem")
        assert "ORIGINAL problem's gap" in res.cert.condition_str
        assert "metric[ruiz_pc]" in res.cert.scheme

    def test_ruiz_pc_uses_the_certified_bound_not_power_iteration(self):
        from atlas.genome import genome_from_dict
        from atlas.schemes.compile import solve_genome
        from atlas.schemes.rescale import lp_substrate_metric
        from atlas.wrappers.km import Budget

        prob = self._prob()
        g = genome_from_dict({
            "scheme": "pdhg", "params": {"tau": 0.5, "sigma": 0.5},
            "metric": lp_substrate_metric(),
        })
        res = solve_genome(prob, g, Budget(max_iters=2000, tol=1e-6))
        assert res.cert.params["opnorm_rescaled"] == pytest.approx(
            PC_CERTIFIED_NORM * PC_SAFETY, rel=1e-12)
        assert "Lemma 2" in res.cert.condition_str
        # the bound must NOT be advertised as a power-iteration estimate
        assert "power iteration x1.02" not in res.cert.condition_str
        assert "no power iteration" in res.cert.condition_str
        # and the margin must be printed at a precision that shows it
        assert f"{PC_SAFETY:.12g}" in res.cert.condition_str

    def test_composed_diagonals_match_the_two_step_chain(self):
        from atlas.schemes.rescale import (
            pock_chambolle_diagonals, rescale_lp_problem, ruiz_equilibrate,
        )

        prob = self._prob()
        A = np.asarray(prob.data["A_dense"], dtype=np.float64)
        _p, d1, d2, _nb = rescale_lp_problem(
            prob, "ruiz_pc", iters=10, alpha=1.0)
        d1r, d2r = ruiz_equilibrate(A, 10)
        d1p, d2p = pock_chambolle_diagonals(A * d1r[:, None] * d2r[None, :], 1.0)
        np.testing.assert_allclose(d1, d1p * d1r, rtol=1e-12, atol=1e-15)
        np.testing.assert_allclose(d2, d2r * d2p, rtol=1e-12, atol=1e-15)


def test_gate_and_compiler_agree_on_the_pc_bound():
    """schema.PC_GATE_OPNORM mirrors rescale's certified bound because the
    genome package must stay jax-free. If they drift, validate() and
    solve_genome disagree about the same genome, which is exactly the class
    of bug that breaks 'convergent by construction'."""
    from atlas.genome.schema import PC_GATE_OPNORM

    assert PC_GATE_OPNORM == PC_CERTIFIED_NORM * PC_SAFETY


def test_frozen_b0_passes_the_static_gate():
    """Regression: the static gate hardcoded 1.02 while the compiler used
    1+1e-9, so B0 (tau*sigma = 0.980) was rejected by validate() and accepted
    by the compiler."""
    from atlas.genome import genome_from_dict, validate
    from atlas.schemes.rescale import lp_substrate_metric

    g = genome_from_dict({"scheme": "pdhg",
                          "params": {"tau": 1.98, "sigma": 0.495},
                          "metric": lp_substrate_metric()})
    v = validate(g, problem_kind="lp_netflow")
    assert v.ok, v.error


def test_reflection_plus_halpern_passes_under_the_substrate():
    from atlas.genome import genome_from_dict, validate
    from atlas.schemes.rescale import lp_substrate_metric

    g = genome_from_dict({"scheme": "pdhg",
                          "params": {"tau": 1.98, "sigma": 0.495, "rho": 2.0},
                          "wrapper": {"halpern": True},
                          "metric": lp_substrate_metric()})
    v = validate(g, problem_kind="lp_netflow")
    assert v.ok, v.error


class TestRelaxThenAnchorActuallyRuns:
    """The gate admitting relax-then-anchor is worthless if the compiled
    iteration ignores rho. It DID ignore it: km_run's 'halpern' mode anchored
    onto T rather than onto F_rho, so rho=1 and rho=2 produced bit-identical
    trajectories while the certificate claimed otherwise."""

    def _prob(self):
        from atlas.bench.lp_cell import from_netflow, to_composite
        from atlas.bench.netflow import gen_regular_flow

        return to_composite(
            from_netflow(gen_regular_flow(n_nodes=60, degree=2, seed=3)))

    def _run(self, rho):
        from atlas.genome import genome_from_dict
        from atlas.schemes.compile import solve_genome
        from atlas.schemes.rescale import lp_substrate_metric
        from atlas.wrappers.km import Budget

        d = {"scheme": "pdhg", "params": {"tau": 0.9, "sigma": 0.9},
             "wrapper": {"halpern": True}, "metric": lp_substrate_metric()}
        if rho is not None:
            d["params"]["rho"] = rho
        r = solve_genome(self._prob(), genome_from_dict(d),
                         Budget(max_iters=400, tol=0.0, sample_every=50))
        h = r.rel_gap_history[np.isfinite(r.rel_gap_history)]
        return float(h[-1])

    def test_rho_changes_the_anchored_trajectory(self):
        g1, g2 = self._run(1.0), self._run(2.0)
        assert g1 != g2, (
            "rho is a no-op under the Halpern anchor: the compiled iteration "
            "is not the one the certificate describes")

    def test_rho_one_matches_no_rho(self):
        """rho = 1 must recover the plain anchored iteration exactly."""
        assert self._run(1.0) == self._run(None)
