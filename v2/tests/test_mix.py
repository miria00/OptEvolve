"""Mix genes: avg blends (certified convex combinations) and safeguarded
switch genomes (monitor-gated aggressive steps over a certified fallback)."""
import numpy as np
import pytest

from atlas import Budget, TypeCheckError, solve_genome, tv_denoise_problem
from atlas.genome import canonical_json, content_hash, genome_from_dict, validate

TOL = 1e-6


def _problem(tv_instance_32):
    y, lam = tv_instance_32
    return tv_denoise_problem(y, lam)


def _avg(sa, pa, sb, pb, lam=0.5, wrapper=None, rho=None):
    d = {
        "scheme": "mix",
        "params": {"tau": 1.0, "rho": rho},
        "mix": {
            "kind": "avg",
            "lambda": lam,
            "scheme_a": {"scheme": sa, "params": pa},
            "scheme_b": {"scheme": sb, "params": pb},
        },
    }
    if wrapper:
        d["wrapper"] = wrapper
    return genome_from_dict(d)


def _switch(agg_s, agg_p, fb_s, fb_p, window=25):
    return genome_from_dict(
        {
            "scheme": "mix",
            "mix": {
                "kind": "switch",
                "window": window,
                "aggressive": {"scheme": agg_s, "params": agg_p},
                "fallback": {"scheme": fb_s, "params": fb_p},
            },
        }
    )


# ---------------------------------------------------------------------------
# avg: Stage-0 gate
# ---------------------------------------------------------------------------


class TestAvgGate:
    def test_blend_of_two_valid_genomes_passes(self):
        g = _avg("fbs", {"tau": 0.05}, "fbs", {"tau": 0.2}, lam=0.3)
        v = validate(g)
        assert v.ok, v.error
        assert v.cert.scheme == "avg(FBS,FBS)"
        # combined averagedness = lam*a_a + (1-lam)*a_b
        a_a = 2.0 / (4.0 - 8.0 * 0.05)
        a_b = 2.0 / (4.0 - 8.0 * 0.2)
        assert v.cert.averagedness == pytest.approx(0.3 * a_a + 0.7 * a_b)
        assert "Bauschke" in v.cert.citation

    def test_pdhg_condat_vu_pair_with_identical_steps_passes(self):
        p = {"tau": 0.5, "sigma": 0.99 / 8.0}
        g = _avg("pdhg", p, "condat_vu", p, lam=0.6)
        v = validate(g)
        assert v.ok, v.error
        assert v.cert.scheme == "avg(PDHG,CondatVu)"

    def test_blend_with_invalid_parent_is_rejected(self):
        # parent b violates the FBS side condition (tau >= 0.25)
        g = _avg("fbs", {"tau": 0.05}, "fbs", {"tau": 0.5})
        v = validate(g)
        assert not v.ok
        assert "scheme_b" in v.error and "0 < a < 2/L" in v.error

    def test_cross_family_blend_rejected(self):
        g = _avg("fbs", {"tau": 0.05}, "drs", {"tau": 1.0})
        v = validate(g)
        assert not v.ok and "state space" in v.error

    def test_saddle_pair_with_different_steps_rejected(self):
        g = _avg(
            "pdhg",
            {"tau": 0.5, "sigma": 0.1},
            "condat_vu",
            {"tau": 0.4, "sigma": 0.1},
        )
        v = validate(g)
        assert not v.ok and "IDENTICAL" in v.error

    def test_lambda_range_enforced(self):
        for lam in (0.0, 1.0, -0.2, 1.7):
            g = _avg("fbs", {"tau": 0.05}, "fbs", {"tau": 0.2}, lam=lam)
            assert not validate(g).ok

    def test_canonical_hash_stable_and_distinct(self):
        g1 = _avg("fbs", {"tau": 0.05}, "fbs", {"tau": 0.2}, lam=0.3)
        g2 = _avg("fbs", {"tau": 0.05}, "fbs", {"tau": 0.2}, lam=0.3)
        g3 = _avg("fbs", {"tau": 0.05}, "fbs", {"tau": 0.2}, lam=0.4)
        assert content_hash(g1) == content_hash(g2)
        assert content_hash(g1) != content_hash(g3)
        assert '"kind":"avg"' in canonical_json(g1)


# ---------------------------------------------------------------------------
# avg: runtime
# ---------------------------------------------------------------------------


class TestAvgRuntime:
    def test_blend_converges_on_tv32_to_1e6(self, tv_instance_32):
        p = _problem(tv_instance_32)
        step = {"tau": 0.5, "sigma": 0.99 / 8.0}
        g = _avg("pdhg", step, "condat_vu", step, lam=0.5)
        res = solve_genome(p, g, Budget(max_iters=200_000, tol=TOL, sample_every=100))
        assert res.cert.satisfied
        hist = res.rel_gap_history[np.isfinite(res.rel_gap_history)]
        assert hist[-1] <= TOL
        assert np.all(hist >= -1e-12)  # gap stays rigorous

    def test_same_scheme_blend_converges(self, tv_instance_32):
        p = _problem(tv_instance_32)
        g = _avg("fbs", {"tau": 0.05}, "fbs", {"tau": 0.24}, lam=0.5)
        res = solve_genome(p, g, Budget(max_iters=400_000, tol=TOL, sample_every=200))
        assert res.rel_gap_history[-1] <= TOL

    def test_blend_with_invalid_parent_cannot_run(self, tv_instance_32):
        p = _problem(tv_instance_32)
        g = _avg("fbs", {"tau": 0.05}, "fbs", {"tau": 0.5})
        with pytest.raises(TypeCheckError):
            solve_genome(p, g, Budget(max_iters=10))


# ---------------------------------------------------------------------------
# switch: Stage-0 gate
# ---------------------------------------------------------------------------


class TestSwitchGate:
    def test_valid_fallback_wildly_invalid_aggressive_passes(self):
        g = _switch("fbs", {"tau": 50.0}, "fbs", {"tau": 0.2})
        v = validate(g)
        assert v.ok, v.error
        assert v.cert.scheme.startswith("switch[")
        assert "safeguard" in v.cert.condition_str

    def test_invalid_fallback_rejected(self):
        g = _switch("fbs", {"tau": 0.05}, "fbs", {"tau": 0.5})
        v = validate(g)
        assert not v.ok and "fallback" in v.error

    def test_cross_family_switch_rejected(self):
        g = _switch("drs", {"tau": 1.0}, "fbs", {"tau": 0.2})
        v = validate(g)
        assert not v.ok and "state space" in v.error

    def test_outer_wrappers_forbidden_on_switch(self):
        g = _switch("fbs", {"tau": 5.0}, "fbs", {"tau": 0.2})
        d = {
            "scheme": "mix",
            "params": {"rho": 1.5},
            "mix": {
                "kind": "switch",
                "window": 25,
                "aggressive": {"scheme": "fbs", "params": {"tau": 5.0}},
                "fallback": {"scheme": "fbs", "params": {"tau": 0.2}},
            },
        }
        v = validate(genome_from_dict(d))
        assert not v.ok and "no outer wrappers" in v.error
        assert validate(g).ok

    def test_window_validated(self):
        for w in (0, -3):
            g = _switch("fbs", {"tau": 5.0}, "fbs", {"tau": 0.2}, window=w)
            assert not validate(g).ok


# ---------------------------------------------------------------------------
# switch: runtime (the monitor)
# ---------------------------------------------------------------------------


class TestSwitchRuntime:
    def test_invalid_aggressive_with_valid_fallback_converges(self, tv_instance_32):
        p = _problem(tv_instance_32)
        # aggressive fbs step 40x past the side-condition limit
        g = _switch("fbs", {"tau": 10.0}, "fbs", {"tau": 0.24}, window=25)
        res = solve_genome(p, g, Budget(max_iters=400_000, tol=TOL, sample_every=200))
        assert res.rel_gap_history[-1] <= TOL
        assert res.extra["n_aggressive_accepted"] + res.extra["n_fallback_steps"] == res.iters

    def test_pure_aggressive_divergence_is_caught(self, tv_instance_32):
        """The aggressive map alone stalls far from tol (its stepsize breaks
        the certificate); the monitor must reject nearly all its steps and
        still converge via the fallback."""
        from atlas import FBSParams, solve_fbs
        from atlas.schemes.base_schemes import build_scheme, run_built

        p = _problem(tv_instance_32)
        # run the aggressive map UNGATED (check=False) for a while: no
        # convergence (this is the divergence the monitor must catch)
        built = build_scheme(p, "fbs", FBSParams(step=10.0), check=False)
        res_agg = run_built(
            built, None, FBSParams(step=10.0),
            Budget(max_iters=3_000, tol=TOL, sample_every=10),
        )
        assert res_agg.rel_gap_history[-1] > 1e-2  # nowhere near tol

        g = _switch("fbs", {"tau": 10.0}, "fbs", {"tau": 0.24}, window=10)
        res = solve_genome(p, g, Budget(max_iters=400_000, tol=TOL, sample_every=200))
        assert res.rel_gap_history[-1] <= TOL  # safeguarded: converged
        # the monitor rejected the overwhelming majority of aggressive steps
        assert res.extra["n_fallback_steps"] > 5 * res.extra["n_aggressive_accepted"]

    def test_switch_gap_history_rigorous(self, tv_instance_32):
        p = _problem(tv_instance_32)
        g = _switch("fbs", {"tau": 1.0}, "fbs", {"tau": 0.2}, window=50)
        res = solve_genome(p, g, Budget(max_iters=200_000, tol=TOL, sample_every=100))
        hist = res.rel_gap_history[np.isfinite(res.rel_gap_history)]
        assert np.all(hist >= -1e-12)
        assert hist[-1] <= TOL
