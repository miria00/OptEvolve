"""Evolution wing tests: Stage-0 gate, random backend, LLM backend
(mocked HTTP), ask/tell loop, plus one live vLLM integration test.

conftest.py pins JAX to CPU + x64 before any jax import (the genome and
evolve modules themselves are jax-free, but atlas.__init__ pulls schemes)."""
from __future__ import annotations

import json

import pytest

from atlas.genome import (
    Genome,
    GenomeParams,
    Wrapper,
    canonical_json,
    content_hash,
    genome_from_dict,
    validate,
)
from atlas.evolve import LLMBackend, RandomBackend, SearchBackend, evolve

VLLM_URL = "http://localhost:8000/v1"


def _pdhg(tau, sigma, **kw):
    return Genome(scheme="pdhg", params=GenomeParams(tau=tau, sigma=sigma), **kw)


# ---------------------------------------------------------------------------
# Stage-0 gate
# ---------------------------------------------------------------------------


class TestStage0Gate:
    def test_rejects_pdhg_stepsize_product_violation(self):
        # tau*sigma*||L||^2 = 0.5*0.5*8 = 2 >= 1 -> must be REJECTED
        v = validate(_pdhg(0.5, 0.5))
        assert not v.ok
        assert v.cert is None
        assert "t*s*||L||^2 < 1" in v.error  # violated condition fed back

    def test_accepts_valid_pdhg(self):
        # 0.3*0.3*8 = 0.72 < 1
        v = validate(_pdhg(0.3, 0.3))
        assert v.ok and v.error is None
        assert v.cert is not None and v.cert.satisfied
        assert v.cert.scheme == "PDHG"

    def test_rejects_fbs_overstep_and_accepts_valid(self):
        # FBS on the TV dual: a must be < 2/8 = 0.25
        bad = Genome("fbs", GenomeParams(tau=0.3))
        good = Genome("fbs", GenomeParams(tau=0.2))
        assert not validate(bad).ok
        assert "0 < a < 2/L" in validate(bad).error
        assert validate(good).ok

    def test_rejects_condat_vu_violation(self):
        # t*L_f/2 + t*s*||L||^2 = 0.5*0.5 + 0.5*0.3*8 = 1.45 >= 1
        g = Genome("condat_vu", GenomeParams(tau=0.5, sigma=0.3))
        v = validate(g)
        assert not v.ok and "t*L_f/2 + t*s*||L||^2 < 1" in v.error

    def test_rejects_bad_relaxation_no_clamp(self):
        # FBS near the stepsize limit: alpha = 2/(4-8*0.24) ~ 0.96; rho=1.9
        # gives rho*alpha > 1 -> rejected via the combination rule
        g = Genome("fbs", GenomeParams(tau=0.24, rho=1.9))
        v = validate(g)
        assert not v.ok and "rho*alpha" in v.error

    def test_admits_rho_plus_halpern(self):
        """As of 2026-09-05 relax-then-anchor is IN the search space. It was
        excluded, which made the reflection composition (Halpern on 2T - I,
        i.e. r2HPDHG) unrepresentable: the highest-value structure in the
        GPU-LP lineage was outside the grammar by construction."""
        g = _pdhg(0.1, 0.1, wrapper=Wrapper(halpern=True))
        g = Genome(g.scheme, GenomeParams(tau=0.1, sigma=0.1, rho=1.2), g.wrapper)
        v = validate(g)
        assert v.ok, v.error

    def test_admits_reflection_plus_halpern(self):
        """rho = 2 exactly: the reflection, admissible only because an
        anchored step follows."""
        g = _pdhg(0.1, 0.1, wrapper=Wrapper(halpern=True))
        g = Genome(g.scheme, GenomeParams(tau=0.1, sigma=0.1, rho=2.0), g.wrapper)
        assert validate(g).ok

    def test_rejects_missing_sigma_and_bad_restart(self):
        assert not validate(Genome("pdhg", GenomeParams(tau=0.1))).ok
        g = _pdhg(0.1, 0.1, wrapper=Wrapper(restart="fixed_k", restart_k=None))
        assert not validate(g).ok

    def test_structural_rejects_unknown_scheme(self):
        v = validate(Genome("admm", GenomeParams(tau=0.1)))
        assert not v.ok and "scheme" in v.error


class TestCanonicalForm:
    def test_dedup_hash_normalizes_irrelevant_genes(self):
        # sigma is meaningless for drs; restart_k meaningless when "none"
        a = Genome("drs", GenomeParams(tau=1.0, sigma=0.3),
                   Wrapper(restart="none", restart_k=50))
        b = Genome("drs", GenomeParams(tau=1.0, sigma=0.7), Wrapper())
        assert canonical_json(a) == canonical_json(b)
        assert content_hash(a) == content_hash(b)

    def test_distinct_genomes_distinct_hashes(self):
        assert content_hash(_pdhg(0.3, 0.3)) != content_hash(_pdhg(0.3, 0.2))

    def test_roundtrip_from_dict(self):
        d = {"scheme": "PDHG", "params": {"tau": 0.3, "sigma": 0.2},
             "wrapper": {"halpern": True}}
        g = genome_from_dict(d)
        assert g.scheme == "pdhg" and g.wrapper.halpern
        assert validate(g).ok


# ---------------------------------------------------------------------------
# Random backend
# ---------------------------------------------------------------------------


class TestRandomBackend:
    def test_100_draws_all_valid(self):
        be = RandomBackend(seed=42)
        genomes = be.ask(100)
        assert len(genomes) == 100
        for g in genomes:
            v = validate(g)
            assert v.ok, f"invalid random genome {canonical_json(g)}: {v.error}"

    def test_covers_all_schemes_and_is_deterministic(self):
        genomes = RandomBackend(seed=7).ask(100)
        schemes = {g.scheme for g in genomes}
        # the 4 TV-applicable base schemes plus mix genomes (avg/switch_k);
        # dys never appears on the TV cell (structurally inapplicable), and
        # the DEPRECATED residual-window "switch" is never proposed
        assert {"fbs", "drs", "pdhg", "condat_vu", "mix"} <= schemes
        assert "dys" not in schemes
        kinds = {g.mix.kind for g in genomes if g.mix is not None}
        assert kinds == {"avg", "switch_k"}
        again = RandomBackend(seed=7).ask(100)
        assert [canonical_json(g) for g in genomes] == [
            canonical_json(g) for g in again
        ]

    def test_is_a_search_backend(self):
        assert isinstance(RandomBackend(), SearchBackend)


# ---------------------------------------------------------------------------
# LLM backend (mocked HTTP — no server required)
# ---------------------------------------------------------------------------


def _mk_llm(replies, **kw):
    it = iter(replies)
    calls = []

    def fake_post(payload):
        calls.append(payload)
        return next(it)

    be = LLMBackend(http_post=fake_post, seed=3, **kw)
    return be, calls


class TestLLMBackendMocked:
    def test_garbage_reply_falls_back_to_random(self):
        be, calls = _mk_llm(["I am a teapot { not json ]", "still } nonsense {"])
        (g,) = be.ask(1)
        assert validate(g).ok  # fallback proposal is valid
        assert be.n_fallbacks == 1
        assert len(calls) == 2  # initial try + ONE retry, then fallback

    # NOTE (2026-09-04): the default retry is now a CLEAN RESAMPLE, not a
    # diagnostic-injected repair. The paired ablation
    # (scripts/repair_ablation.py, n=200, results/sprint/phase3/) found
    # injection strictly worse than resampling: 52.0% -> 36.0% two-retry
    # yield, McNemar p = 0.0007, effect strongest on genuine gate rejections.
    # The injection path is retained behind repair_feedback=True and is still
    # tested below, so a larger proposer can be re-evaluated later.

    def test_retry_recovers_and_default_prompt_carries_no_diagnostic(self):
        good = json.dumps(
            {"scheme": "pdhg", "params": {"tau": 0.3, "sigma": 0.3},
             "wrapper": {"halpern": False, "restart": "none"}}
        )
        be, calls = _mk_llm(["not json at all", "Sure! Here it is: " + good + " enjoy"])
        (g,) = be.ask(1)
        assert g.scheme == "pdhg" and be.n_fallbacks == 0
        # DEFAULT: the retry is a clean resample of the identical prompt
        assert "INVALID" not in calls[1]["messages"][1]["content"]
        assert calls[0]["messages"][1]["content"] == calls[1]["messages"][1]["content"]

    def test_retry_with_error_feedback_recovers_when_enabled(self):
        good = json.dumps(
            {"scheme": "pdhg", "params": {"tau": 0.3, "sigma": 0.3},
             "wrapper": {"halpern": False, "restart": "none"}}
        )
        be, calls = _mk_llm(["not json at all", "Sure! Here it is: " + good + " enjoy"],
                            repair_feedback=True)
        (g,) = be.ask(1)
        assert g.scheme == "pdhg" and be.n_fallbacks == 0
        assert "INVALID" in calls[1]["messages"][1]["content"]

    def test_stage0_rejection_still_counted_under_clean_resample(self):
        bad = json.dumps({"scheme": "pdhg", "params": {"tau": 0.5, "sigma": 0.5}})
        good = json.dumps({"scheme": "pdhg", "params": {"tau": 0.1, "sigma": 0.1}})
        be, calls = _mk_llm([bad, good])
        (g,) = be.ask(1)
        # the gate still REJECTS and still counts; it just does not lecture
        assert g.params.tau == 0.1 and be.stage0_rejections == 1
        assert "t*s*||L||^2" not in calls[1]["messages"][1]["content"]

    def test_stage0_rejection_feeds_condition_back_when_enabled(self):
        bad = json.dumps({"scheme": "pdhg", "params": {"tau": 0.5, "sigma": 0.5}})
        good = json.dumps({"scheme": "pdhg", "params": {"tau": 0.1, "sigma": 0.1}})
        be, calls = _mk_llm([bad, good], repair_feedback=True)
        (g,) = be.ask(1)
        assert g.params.tau == 0.1 and be.stage0_rejections == 1
        assert "t*s*||L||^2" in calls[1]["messages"][1]["content"]

    def test_json_patch_merges_onto_best(self):
        be, _ = _mk_llm([json.dumps({"params": {"tau": 0.2}})])
        base = _pdhg(0.1, 0.1)
        be.tell(base, -1.0, {"iters": 500})
        (g,) = be.ask(1)
        assert g.scheme == "pdhg" and g.params.tau == 0.2 and g.params.sigma == 0.1

    def test_http_exception_falls_back(self):
        def boom(payload):
            raise ConnectionError("server down")

        be = LLMBackend(http_post=boom, seed=5)
        (g,) = be.ask(1)
        assert validate(g).ok and be.n_fallbacks == 1

    def test_prompt_stays_small(self):
        be, calls = _mk_llm(["garbage", "garbage"])
        for i in range(3):
            be.tell(_pdhg(0.1 + 0.01 * i, 0.1), -float(i), {"iters": i, "gap": 1e-5})
        be.ask(1)
        prompt = calls[0]["messages"][1]["content"]
        assert len(prompt) < 6000  # ~1500 tokens

    def test_payload_carries_deterministic_sampling_seed(self):
        """Regression: the chat payload had no 'seed' field, so --evolve-seed
        never made the LLM proposals reproducible."""
        good = json.dumps({"scheme": "pdhg", "params": {"tau": 0.1, "sigma": 0.1}})
        be, calls = _mk_llm([good, good])
        be.ask(1)
        be.ask(1)
        seeds = [c["seed"] for c in calls]
        assert len(seeds) == 2 and seeds[0] != seeds[1]  # calls sample differently
        # replay: a fresh backend with the same ctor seed issues the same sequence
        be2, calls2 = _mk_llm([good, good])
        be2.ask(1)
        be2.ask(1)
        assert [c["seed"] for c in calls2] == seeds

    def test_reseed_changes_llm_sampling_seed(self):
        """Regression: reseed() used to touch only the random fallback."""
        good = json.dumps({"scheme": "pdhg", "params": {"tau": 0.1, "sigma": 0.1}})
        be, calls = _mk_llm([good, good])
        be.ask(1)
        first = calls[0]["seed"]
        be.reseed(first + 1000)
        be.ask(1)
        assert calls[1]["seed"] == first + 1000 + 2  # base moved with reseed

    def test_archive_dedups_repeated_tells(self):
        """Regression: tell() appended unconditionally, so the loop's elitism
        re-tell flooded the size-8 archive with champion copies."""
        be, _ = _mk_llm([])
        g = _pdhg(0.1, 0.1)
        for _ in range(5):
            be.tell(g, -1.0, {"iters": 100})
        assert len(be.archive) == 1
        be.tell(g, -0.5, {"iters": 90})  # better run of the same genome: replace
        assert len(be.archive) == 1 and be.archive[0][0] == -0.5
        be.tell(_pdhg(0.2, 0.1), -2.0, {"iters": 200})  # distinct genome: append
        assert len(be.archive) == 2


# ---------------------------------------------------------------------------
# Loop
# ---------------------------------------------------------------------------


class TestLoop:
    def test_three_generations_with_stub_fitness(self):
        be = RandomBackend(seed=11)

        def fitness(g):  # peaked at tau = 0.05, higher is better
            return -abs(g.params.tau - 0.05), {"tau": g.params.tau}

        hist = evolve(be, fitness, generations=3, pop_size=6, seed=11)
        gens = hist["generations"]
        assert len(gens) == 3
        for rec in gens:
            assert {"gen", "best_fitness", "best_genome", "n_rejected_stage0",
                    "n_llm_fallbacks"} <= set(rec)
        # elitism keep-1: best fitness monotone non-decreasing
        fits = [rec["best_fitness"] for rec in gens]
        assert fits == sorted(fits)
        assert hist["best_fitness"] == fits[-1]
        assert json.loads(hist["best_genome"])["scheme"] in (
            "fbs", "drs", "pdhg", "condat_vu")
        assert hist["n_unique_evaluated"] >= 1

    def test_loop_gates_invalid_backend_proposals(self):
        class RogueBackend:
            """Emits one invalid genome per ask() among valid ones."""

            def __init__(self):
                self.inner = RandomBackend(seed=2)

            def ask(self, k):
                gs = self.inner.ask(k - 1)
                gs.append(_pdhg(0.5, 0.5))  # 0.5*0.5*8 = 2: invalid
                return gs

            def tell(self, genome, fitness, metrics):
                self.inner.tell(genome, fitness, metrics)

        hist = evolve(RogueBackend(), lambda g: 0.0, generations=2, pop_size=4)
        assert all(rec["n_rejected_stage0"] >= 1 for rec in hist["generations"])

    def test_dedup_skips_reevaluation(self):
        class ConstantBackend:
            def ask(self, k):
                return [_pdhg(0.1, 0.1)] * k

            def tell(self, genome, fitness, metrics):
                pass

        n_calls = 0

        def fitness(g):
            nonlocal n_calls
            n_calls += 1
            return 1.0

        hist = evolve(ConstantBackend(), fitness, generations=3, pop_size=5)
        assert n_calls == 1  # evaluated once, cached thereafter
        assert hist["n_unique_evaluated"] == 1

    def test_loop_with_mocked_llm_backend(self):
        garbage = ["{ nope"] * 100  # every call garbage -> always fallback
        be, _ = _mk_llm(garbage)
        hist = evolve(be, lambda g: -g.params.tau, generations=2, pop_size=3)
        assert len(hist["generations"]) == 2
        assert all(r["n_llm_fallbacks"] == 3 for r in hist["generations"])

    def test_flat_fitness_is_flagged(self):
        """Regression (the exact v1 failure mode): a constant-fitness run used
        to produce a normal-looking history with no warning and no
        per-candidate fitnesses to reconstruct flatness from."""
        be = RandomBackend(seed=13)
        with pytest.warns(RuntimeWarning, match="flat"):
            hist = evolve(be, lambda g: 1.0, generations=3, pop_size=4, seed=13)
        assert hist["flat_fitness_warning"] is True
        assert hist["fitness_spread"] == 0.0
        for rec in hist["generations"]:  # per-candidate fitnesses now recorded
            assert rec["fitnesses"] and all(f == 1.0 for f in rec["fitnesses"])

    def test_varied_fitness_not_flagged(self):
        be = RandomBackend(seed=11)
        hist = evolve(
            be, lambda g: -abs(g.params.tau - 0.05), generations=3, pop_size=6, seed=11
        )
        assert hist["flat_fitness_warning"] is False
        assert hist["fitness_spread"] > 0.0

    def test_elitism_retell_does_not_flood_archive(self):
        """Regression: after 6 generations the archive used to hold up to 7
        copies of the champion, evicting genuine runners-up from the prompt."""
        from atlas.genome import content_hash

        be, _ = _mk_llm(["{ nope"] * 200)  # all garbage -> random fallback
        evolve(be, lambda g: -abs(g.params.tau - 0.05), generations=6, pop_size=3,
               seed=3)
        hashes = [content_hash(g) for _, g, _ in be.archive]
        assert len(hashes) == len(set(hashes)), "archive holds duplicate genomes"
        assert len(hashes) >= 2


# ---------------------------------------------------------------------------
# Live integration (needs the local vLLM server; skips otherwise)
# ---------------------------------------------------------------------------


def _server_reachable() -> bool:
    try:
        import urllib.request

        req = urllib.request.Request(
            VLLM_URL + "/models", headers={"Authorization": "Bearer x"}
        )
        with urllib.request.urlopen(req, timeout=3):
            return True
    except Exception:
        return False


@pytest.mark.integration
@pytest.mark.skipif(not _server_reachable(), reason="vLLM server unreachable")
def test_llm_backend_live_round_trip():
    be = LLMBackend(base_url=VLLM_URL, seed=0, timeout=120.0)
    be.tell(_pdhg(0.3, 0.3), -0.42, {"iters": 800, "rel_gap": 9e-5})
    be.tell(Genome("fbs", GenomeParams(tau=0.2)), -0.61, {"iters": 1500})
    (g,) = be.ask(1)
    v = validate(g)
    assert v.ok, f"live proposal failed the gate: {v.error}"
    assert be.n_llm_calls >= 1
