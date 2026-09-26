"""switch_k: the finite-prefix switching primitive.

Gate: the tail must pass the FULL Stage-0 gate; the aggressive block is
structurally validated but exempt from convergence side conditions
("finite prefix, exempt"); 1 <= K <= 5000; shared state space; f64 only;
no outer wrappers. Runtime: a violently divergent aggressive prefix plus
a certified tail still reaches the certificate, with the non-finite
reset logged when it fires; a benign prefix is exercised end to end.
The DEPRECATED residual-window "switch" stays validated and runnable
(covered in tests/test_mix.py) but is absent from the proposal menus
(covered in tests/test_evolve.py and here).
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pytest

from atlas import Budget, solve_genome, tv_denoise_problem
from atlas.bench.lp_cell import from_netflow, to_composite
from atlas.bench.netflow import gen_regular_flow
from atlas.certify.residuals import lp_rel_gap
from atlas.genome import (
    MAX_SWITCH_K,
    canonical_json,
    content_hash,
    genome_from_dict,
    validate,
)

TOL = 1e-6


def _switch_k(agg_s, agg_p, tail_s, tail_p, K=100, extra_top=None):
    d = {
        "scheme": "mix",
        "mix": {
            "kind": "switch_k",
            "K": K,
            "aggressive": {"scheme": agg_s, "params": agg_p},
            "tail": {"scheme": tail_s, "params": tail_p},
        },
    }
    if extra_top:
        d.update(extra_top)
    return genome_from_dict(d)


@pytest.fixture(scope="module")
def lp40():
    return to_composite(
        from_netflow(gen_regular_flow(n_nodes=40, seed=7, degree=2))
    )


# ---------------------------------------------------------------------------
# Stage-0 gate
# ---------------------------------------------------------------------------


class TestSwitchKGate:
    def test_valid_switch_k_accepted_tv(self):
        g = _switch_k("fbs", {"tau": 500.0}, "fbs", {"tau": 0.2}, K=100)
        v = validate(g)
        assert v.ok, v.error
        assert v.cert.scheme == "switch_k[FBS]"

    def test_valid_switch_k_accepted_lp(self):
        g = _switch_k(
            "pdhg",
            {"tau": 1000.0, "sigma": 1000.0},
            "pdhg",
            {"tau": 0.3, "sigma": 0.3},
            K=300,
        )
        v = validate(g, problem_kind="lp_netflow")
        assert v.ok, v.error
        assert v.cert.scheme == "switch_k[PDHG]"

    def test_cert_content(self):
        g = _switch_k("fbs", {"tau": 500.0}, "fbs", {"tau": 0.2}, K=42)
        v = validate(g)
        assert v.ok
        c = v.cert
        assert "finite prefix, exempt" in c.condition_str
        assert "K=42" in c.condition_str
        assert "0 < a < 2/L" in c.condition_str  # tail condition threaded
        assert c.params["K"] == 42
        assert "fbs(tau=500" in c.params["aggressive"]
        # averagedness is the TAIL's (the prefix does not touch the map)
        assert c.averagedness == pytest.approx(2.0 / (4.0 - 8.0 * 0.2))
        assert "finite-prefix switching" in c.citation

    def test_k_out_of_range_rejected(self):
        for K in (0, -5, MAX_SWITCH_K + 1):
            g = _switch_k("fbs", {"tau": 500.0}, "fbs", {"tau": 0.2}, K=K)
            v = validate(g)
            assert not v.ok and "mix.K" in v.error

    def test_k_must_be_int(self):
        for K in (2.5, True):
            with pytest.raises(Exception):
                _switch_k("fbs", {"tau": 500.0}, "fbs", {"tau": 0.2}, K=K)

    def test_invalid_tail_rejected(self):
        # tail violates the FBS side condition (tau >= 0.25 on the TV dual)
        g = _switch_k("fbs", {"tau": 0.05}, "fbs", {"tau": 0.5})
        v = validate(g)
        assert not v.ok
        assert "tail must pass the FULL Stage-0 gate" in v.error

    def test_aggressive_must_be_structurally_well_formed(self):
        g = _switch_k("fbs", {"tau": -1.0}, "fbs", {"tau": 0.2})
        v = validate(g)
        assert not v.ok and "aggressive" in v.error and "positive" in v.error

    def test_mismatched_state_spaces_rejected(self):
        g = _switch_k("drs", {"tau": 5.0}, "fbs", {"tau": 0.2})
        v = validate(g)
        assert not v.ok and "state space" in v.error
        # the saddle pair IS allowed
        g2 = _switch_k(
            "condat_vu",
            {"tau": 100.0, "sigma": 100.0},
            "pdhg",
            {"tau": 0.5, "sigma": 0.1},
        )
        assert validate(g2).ok

    def test_outer_wrappers_rejected(self):
        g = _switch_k(
            "fbs",
            {"tau": 500.0},
            "fbs",
            {"tau": 0.2},
            extra_top={"wrapper": {"halpern": True}},
        )
        v = validate(g)
        assert not v.ok and "no outer wrappers" in v.error

    def test_non_f64_backend_rejected(self):
        g = _switch_k(
            "fbs",
            {"tau": 500.0},
            "fbs",
            {"tau": 0.2},
            extra_top={"backend": {"precision": "f32_cert64", "K": 100}},
        )
        v = validate(g)
        assert not v.ok and "f64" in v.error
        g64 = _switch_k(
            "fbs",
            {"tau": 500.0},
            "fbs",
            {"tau": 0.2},
            extra_top={"backend": {"precision": "f64", "K": 100}},
        )
        assert validate(g64).ok

    def test_aggressive_scheme_applicability(self):
        g = _switch_k("dys", {"tau": 100.0}, "dys", {"tau": 0.5})
        v = validate(g)  # tv_denoise: dys tail cannot certify there
        assert not v.ok
        g2 = _switch_k("fbs", {"tau": 100.0}, "fbs", {"tau": 0.2})
        v2 = validate(g2, problem_kind="lp_netflow")
        assert not v2.ok  # fbs tail cannot certify on the LP cell

    def test_canonical_hash_distinct_in_k(self):
        g1 = _switch_k("fbs", {"tau": 500.0}, "fbs", {"tau": 0.2}, K=100)
        g2 = _switch_k("fbs", {"tau": 500.0}, "fbs", {"tau": 0.2}, K=100)
        g3 = _switch_k("fbs", {"tau": 500.0}, "fbs", {"tau": 0.2}, K=200)
        assert content_hash(g1) == content_hash(g2)
        assert content_hash(g1) != content_hash(g3)
        assert '"kind":"switch_k"' in canonical_json(g1)

    def test_deprecated_switch_still_validates(self):
        g = genome_from_dict(
            {
                "scheme": "mix",
                "mix": {
                    "kind": "switch",
                    "window": 25,
                    "aggressive": {"scheme": "fbs", "params": {"tau": 50.0}},
                    "fallback": {"scheme": "fbs", "params": {"tau": 0.2}},
                },
            }
        )
        assert validate(g).ok


# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------


class TestSwitchKRuntime:
    def test_divergent_prefix_converges_tv32(self, tv_instance_32):
        """Violently large aggressive steps on TV (the TV state stays
        bounded by the prox clips, so no reset expected) + certified
        tail: still reaches rel gap 1e-6."""
        y, lam = tv_instance_32
        p = tv_denoise_problem(y, lam)
        g = _switch_k("fbs", {"tau": 1000.0}, "fbs", {"tau": 0.2}, K=100)
        res = solve_genome(p, g, Budget(max_iters=200_000, tol=TOL, sample_every=100))
        assert res.cert.satisfied
        hist = res.rel_gap_history[np.isfinite(res.rel_gap_history)]
        assert hist[-1] <= TOL
        sk = res.extra["switch_k"]
        assert sk["K"] == 100 and sk["prefix_iters"] == 100
        assert sk["nonfinite_reset"] is False and sk["reset_iter"] is None

    def test_divergent_prefix_reset_fires_lp(self, lp40):
        """Violently divergent PDHG prefix on the LP cell overflows; the
        guard restarts the tail from the last finite state (here the
        initial point) and the solve still certifies at 1e-6, with the
        reset logged."""
        nb = float(lp40.L.norm_bound)
        g = _switch_k(
            "pdhg",
            {"tau": 1000.0, "sigma": 1000.0},
            "pdhg",
            {"tau": 0.9 / nb, "sigma": 0.9 / nb},
            K=300,
        )
        assert validate(g, problem_kind="lp_netflow").ok
        res = solve_genome(lp40, g, Budget(max_iters=200_000, tol=TOL, sample_every=100))
        hist = res.rel_gap_history[np.isfinite(res.rel_gap_history)]
        assert hist[-1] <= TOL
        sk = res.extra["switch_k"]
        assert sk["nonfinite_reset"] is True
        assert isinstance(sk["reset_iter"], int) and 1 <= sk["reset_iter"] <= 300
        assert sk["prefix_iters"] <= 300
        # independent recomputation of the certificate on the returned pair
        gap = float(
            lp_rel_gap(res.x, res.u, lp40.data["c"], lp40.L, lp40.data["b"])
        )
        assert gap <= TOL * 1.01

    def test_benign_prefix_exercised_lp(self, lp40):
        """A mildly over-stepped prefix (slightly outside the certified
        region) that helps early progress: no reset, converges."""
        nb = float(lp40.L.norm_bound)
        ts = 1.02 / (nb * nb)  # t*s*||A||^2 = 1.02, mildly exempt
        g = _switch_k(
            "pdhg",
            {"tau": float(np.sqrt(ts)), "sigma": float(np.sqrt(ts))},
            "pdhg",
            {"tau": 0.9 / nb, "sigma": 0.9 / nb},
            K=200,
        )
        res = solve_genome(lp40, g, Budget(max_iters=200_000, tol=TOL, sample_every=100))
        hist = res.rel_gap_history[np.isfinite(res.rel_gap_history)]
        assert hist[-1] <= TOL
        sk = res.extra["switch_k"]
        assert sk["nonfinite_reset"] is False
        assert sk["prefix_iters"] == 200 and sk["handoff_from_iter"] == 200
        assert res.iters == sk["prefix_iters"] + sk["tail_iters"]

    def test_solve_cert_content(self, lp40):
        nb = float(lp40.L.norm_bound)
        g = _switch_k(
            "pdhg",
            {"tau": 1000.0, "sigma": 1000.0},
            "pdhg",
            {"tau": 0.9 / nb, "sigma": 0.9 / nb},
            K=50,
        )
        res = solve_genome(lp40, g, Budget(max_iters=200_000, tol=TOL, sample_every=100))
        c = res.cert
        assert c.scheme.startswith("switch_k[")
        assert "finite prefix, exempt" in c.condition_str
        assert c.params["K"] == 50
        assert c.satisfied


# ---------------------------------------------------------------------------
# Menus: the old switch is gone, switch_k is in
# ---------------------------------------------------------------------------


def test_menus_propose_switch_k_not_switch():
    from atlas.evolve.llm_backend import MUTATION_MENU
    from atlas.genome import SCHEMA_DOC

    assert "switch_k" in MUTATION_MENU
    assert '"kind":"switch"' not in MUTATION_MENU
    assert "switch_k" in SCHEMA_DOC
    assert '"kind":"switch"' not in SCHEMA_DOC
