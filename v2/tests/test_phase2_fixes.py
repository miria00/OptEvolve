"""Regression tests for the Phase-2 centerpiece verification findings
(2026-09-04 fixer pass; see results/sprint/phase2/PHASE2_REPORT.md).

Pins:
  V1: the LLM arm's fallback sampler must NOT share an RNG stream with
      the random arm's proposer at the same replicate seed (the banked
      runs did: every fallback draw was byte-identical to a matched
      random-run candidate, 336/336).
  V3: OOD-medium holdout coverage: all 10 champions (not 5 configs) must
      be present in the canonical holdout_ood_medium.json, and every
      champion OOD row (small + medium) is a sound Stage-0 typecheck
      rejection: 1340/1340.
  V7: repair_rate.json must carry attempt-depth resolved counts and BOTH
      estimators (any-round and strict one-feedback-round); totals must
      reconcile with the raw attempt histograms.
  V9: the analysis certificate sweep counts a converged row with
      cert=None as a FAILURE, never as certified.

Data tests skip when the banked artifacts are absent so the suite stays
green on machines that do not carry results/.
"""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

V2 = Path(__file__).resolve().parent.parent
PHASE2 = V2 / "results" / "sprint" / "phase2"
ANALYSIS = PHASE2 / "analysis"


def _load(p):
    if not p.exists():
        pytest.skip(f"banked artifact missing: {p}")
    with open(p) as fh:
        return json.load(fh)


def _import_from(path, name):
    if not path.exists():
        pytest.skip(f"module missing: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def centerpiece():
    sys.path.insert(0, str(V2 / "scripts"))
    try:
        return _import_from(
            V2 / "scripts" / "run_centerpiece.py", "run_centerpiece_mod"
        )
    finally:
        sys.path.pop(0)


# ---------------------------------------------------------------------------
# V1: fallback-seed decoupling
# ---------------------------------------------------------------------------


def test_fallback_seed_decoupled_between_arms(centerpiece):
    for seed in (42, 43, 44, 45, 46, 0, 7, 1000):
        s_llm = centerpiece.fallback_seed("llm", seed)
        s_rand = centerpiece.fallback_seed("random", seed)
        assert s_llm != s_rand, seed
        # the random arm keeps the historical mapping (reproducibility of
        # the banked Phase-2 random runs)
        assert s_rand == seed + 1
        # and no llm fallback seed may collide with ANY random-arm seed
        # in a plausible replicate range
        assert s_llm not in {s + 1 for s in range(0, 10_000)}


def test_fallback_streams_actually_differ(centerpiece):
    from atlas.genome import canonical_json

    a = centerpiece.CPRandomBackend(
        seed=centerpiece.fallback_seed("llm", 42)
    )
    b = centerpiece.CPRandomBackend(
        seed=centerpiece.fallback_seed("random", 42)
    )
    ga = [canonical_json(g) for g in a.ask(8)]
    gb = [canonical_json(g) for g in b.ask(8)]
    assert ga != gb


# ---------------------------------------------------------------------------
# V9: strict certificate counting in the analysis
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def analysis():
    return _import_from(ANALYSIS / "make_analysis.py", "phase2_analysis_mod")


def test_cert_none_is_a_failure(analysis):
    ok, why = analysis.cert_row_ok({"status": "converged", "cert": None})
    assert ok is False and why == "cert_missing"
    ok, why = analysis.cert_row_ok(
        {"status": "converged", "cert": {"satisfied": False}}
    )
    assert ok is False and why == "cert_unsatisfied"
    ok, why = analysis.cert_row_ok(
        {"status": "converged", "cert": {"satisfied": True}}
    )
    assert ok is True and why is None


def test_canonical_holdouts_have_no_missing_certs(analysis):
    n_conv = n_ok = 0
    for name in ("holdout_within.json", "holdout_ood_small.json",
                 "holdout_ood_medium.json"):
        for r in _load(PHASE2 / name)["rows"]:
            if r["status"] == "converged":
                n_conv += 1
                ok, _ = analysis.cert_row_ok(r)
                n_ok += ok
    assert n_conv == n_ok == 285


# ---------------------------------------------------------------------------
# V3: OOD-medium coverage
# ---------------------------------------------------------------------------

CHAMPIONS = tuple(
    f"{arm}_seed{s}" for arm in ("llm", "random") for s in range(42, 47)
)


def test_ood_medium_covers_all_ten_champions():
    doc = _load(PHASE2 / "holdout_ood_medium.json")
    cfgs = set(doc["config"]["configs"])
    assert set(CHAMPIONS) <= cfgs, sorted(set(CHAMPIONS) - cfgs)
    per_cfg = {}
    for r in doc["rows"]:
        per_cfg.setdefault(r["config"], []).append(r)
    for c in CHAMPIONS:
        rows = per_cfg[c]
        assert len(rows) == 38, (c, len(rows))  # 19 instances x 2 tols
        assert all(r["status"] == "typecheck_rejected" for r in rows), c


def test_champion_ood_rows_are_1340_sound_rejections():
    small = _load(PHASE2 / "holdout_ood_small.json")
    medium = _load(PHASE2 / "holdout_ood_medium.json")
    n = rej = 0
    for doc in (small, medium):
        for r in doc["rows"]:
            if r["config"] in CHAMPIONS:
                n += 1
                rej += r["status"] == "typecheck_rejected"
    assert n == rej == 1340, (n, rej)


# ---------------------------------------------------------------------------
# V7: repair-rate estimators reconcile with raw attempt histograms
# ---------------------------------------------------------------------------


def test_repair_rate_estimators_reconcile():
    rep = _load(ANALYSIS / "repair_rate.json")
    arm = rep["per_arm"]["llm"]
    r1, r2 = arm["repaired_after_1_round"], arm["repaired_after_2_rounds"]
    fb = arm["fallback_after_3_attempts"]
    assert arm["retry_depth"] == 3
    assert r1 + r2 + fb > 0
    assert abs(arm["repair_rate_any_round"] - (r1 + r2) / (r1 + r2 + fb)) < 1e-12
    assert abs(arm["repair_rate_strict_one_round"] - r1 / (r1 + r2 + fb)) < 1e-12
    # cross-check against the raw runs
    tot = {0: 0, 1: 0, 2: 0, "fb": 0}
    for s in range(42, 47):
        d = _load(PHASE2 / f"llm_seed{s}.json")
        for g in d["generations"]:
            for c in g["candidates"]:
                errs = c.get("llm_attempt_errors") or []
                if c["source"] == "llm":
                    tot[min(len(errs), 2)] += 1
                elif c["source"] == "fallback":
                    tot["fb"] += 1
                    # every fallback slot burned exactly 3 LLM attempts
                    assert len(errs) == 3, (s, errs)
    assert tot[1] == r1 and tot[2] == r2 and tot["fb"] == fb
    assert tot[0] == arm["first_try_valid"]


# ---------------------------------------------------------------------------
# V1/V2 disclosure: coupling + budget audit present in gate1_numbers.json
# ---------------------------------------------------------------------------


def test_gate1_numbers_carry_budget_and_coupling_audit():
    g1 = _load(ANALYSIS / "gate1_numbers.json")
    b = g1["budgets"]
    assert "arm_coupling" in b and "per_seed" in b["arm_coupling"]
    for seed in range(42, 47):
        c = b["arm_coupling"]["per_seed"][f"seed{seed}"]
        assert (c["byte_identical_in_matched_random_run"]
                == c["llm_fallback_candidates"])
    # unequal realized budgets are recorded per arm
    llm_evals = b["per_arm"]["llm"]["fresh_evaluations"]
    rand_evals = b["per_arm"]["random"]["fresh_evaluations"]
    assert all(v == 160 for v in rand_evals)
    assert all(v < 160 for v in llm_evals)
    # OOD cap/cap tiers must never present a bare 1.0 time speedup
    for key, row in g1["vs_tuned"].items():
        if row["time_speedup_status"] == "not_measurable_both_at_cap":
            assert row["time_speedup"] is None, key
