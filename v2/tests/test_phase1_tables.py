"""Regression tests for the Phase-1 banked-data analyses
(assets/make_phase1_tables.py outputs under results/sprint/phase1/).

Pins the 2026-09-03 fixer findings:
  F3: mix-discovery TV rows were labeled 'TV 64x64'; the runs were 128x128
      (labels must come from the raw logs, not a hardcoded string).
  F4: gate summary line 2 said 'repaired 519 to 587 of 883'; the pooled
      numbers are 306 to 374 of 670 (summary must match pooled JSON).
  F6: precision caption/summary said 26 probe solves; the count is 28.
  F5/F8: 3-methods.tex precision wording and draft cruft.

Tests skip when the banked artifacts are absent (e.g. a checkout without
results/), so the suite stays green on machines that do not carry them.
"""
import json
import re
from pathlib import Path

import pytest

V2 = Path(__file__).resolve().parent.parent
PHASE1 = V2 / "results" / "sprint" / "phase1"
SPRINT = V2 / "results" / "sprint"


def _load(p):
    if not p.exists():
        pytest.skip(f"banked artifact missing: {p}")
    with open(p) as fh:
        return json.load(fh)


def _read(p):
    if not p.exists():
        pytest.skip(f"artifact missing: {p}")
    return p.read_text()


def test_mixdisc_tv_cell_label_matches_raw_logs():
    gate = _load(PHASE1 / "gate_repair_rate.json")
    rows = {r["run"]: r for r in gate["per_run"]}
    for s in (42, 43, 44):
        raw_p = SPRINT / "mixdisc" / f"tv_evolve_seed{s}.json"
        raw = _load(raw_p)
        sz = raw["cell_config"]["size"]
        row = rows[f"mix-disc tv s{s}"]
        assert row["cell"] == f"TV {sz}x{sz}", (row["cell"], sz)
    tex = _read(PHASE1 / "gate_table.tex")
    assert "TV 64x64" not in tex


def test_gate_summary_counts_match_pooled_json():
    gate = _load(PHASE1 / "gate_repair_rate.json")
    p2 = gate["pooled"]["clean_2attempt_runs"]
    p3 = gate["pooled"]["mini_er_3attempt_runs"]
    ac = gate["pooled"]["all_clean_runs_combined"]
    # pooled internal consistency
    assert ac["repair_opportunities"] == (
        p2["repair_opportunities"] + p3["repair_opportunities"]
    )
    lo3, hi3 = p3["repair_successes_bounds"]
    assert ac["repair_successes_bounds"] == [
        p2["repair_successes"] + lo3,
        p2["repair_successes"] + hi3,
    ]
    # the prose summary must carry the pooled numbers, not stale hardcodes
    s = gate["summary_5_lines"][1]
    slo, shi = ac["repair_successes_bounds"]
    expected = (
        f"repaired {slo} to {shi} of "
        f"{ac['repair_opportunities']} rejected-and-retried"
    )
    assert expected in s, s
    assert "883" not in s and "519" not in s


def test_gate_summary_rejection_counts_match_pooled_json():
    gate = _load(PHASE1 / "gate_repair_rate.json")
    ac = gate["pooled"]["all_clean_runs_combined"]
    s = gate["summary_5_lines"][0]
    assert f"{ac['llm_calls_total_proposals']} proposal calls" in s
    assert f"rejected {ac['stage0_rejected']}" in s


def test_precision_probe_solve_count_is_recomputable_and_consistent():
    prec = _load(PHASE1 / "precision_table.json")
    n = 0
    for pm in prec["per_machine"].values():
        for row in pm["rows"]:
            n += len(row["per_instance_iters"])
        n += 2  # near-miss stress pair (f64, f32_cert64)
    checked = prec["certificates"]["controlled_probe_solves_checked"]
    assert checked == n
    assert prec["certificates"]["certificates_failed"] == 0
    s = prec["summary_5_lines"][0]
    assert f"0 of {checked} " in s
    assert "of 26 " not in s
    tex = _read(PHASE1 / "precision_table.tex")
    assert f"All {checked} probe solves certified" in tex.replace("\n", " ")
    assert "All 26 probe" not in tex


def test_generated_tex_obeys_style_law():
    for name in ("gate_table.tex", "precision_table.tex"):
        tex = _read(PHASE1 / name)
        assert "---" not in tex and "—" not in tex, name
        assert "\\emph" not in tex and "\\textit" not in tex, name


@pytest.mark.xfail(strict=False, reason=(
    "paper reset to GitHub main 218ac81 on 2026-09-10 (user request); the F5/F8 wording and "
    "cruft fixes exist only in v2/optEvolvev2, not in the canonical methods section"))
def test_methods_tex_precision_wording_and_no_cruft():
    tex = _read(V2 / "OptAtlas_tex_2026" / "3-methods.tex")
    # F5: the ~3% must not be attributed to certificate evaluation
    assert "certificate evaluation costs" not in tex
    assert not re.search(r"3 percent certificate\s+overhead", tex)
    # F8: draft cruft removed
    assert "because duality gap" not in tex
    assert "Backup 5090s" not in tex
    assert "A100" not in tex
    # style law
    assert "—" not in tex
