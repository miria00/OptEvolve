"""The knowledge base must be well formed and every novelty claim evidenced."""
import atlas.kb as kb


def test_kb_validates():
    assert kb.validate() == []


def test_all_requested_candidates_present():
    ids = {s["id"] for s in kb.schemes()}
    for sid in ("ishikawa", "haugazeau", "anderson", "bregman", "passty", "solodov_svaiter"):
        assert sid in ids


def test_frontier_excludes_mined_schemes():
    # Anderson (AA-PDHG, Aug 2025) and Passty (restarted averaged PDHG) were once
    # listed as unmined in error; the frontier must never reintroduce them.
    lp = kb.frontier("lp")
    assert "anderson" not in lp and "passty" not in lp
    assert {"bregman", "haugazeau", "solodov_svaiter"} <= set(lp)


def test_implemented_entries_resolve():
    assert set(kb.implemented()) == {"krasnoselskii_mann", "halpern", "restart", "haugazeau", "bregman", "variable_metric_schedule", "quasi_newton_metric", "fista", "gap_safe_screening", "anderson", "adaptive_restart"}
