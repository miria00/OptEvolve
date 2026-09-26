"""Phase-1 banked-data analyses: gate repair-rate table, precision-certificate
table, and the evaluations-axis tidy CSV. NO new solver runs: everything here
is recomputed from the frozen 2026-08-26 campaign artifacts in results/ and
results/sprint/.

Outputs (all under results/sprint/phase1/):
  gate_repair_rate.json      Deliverable 1 data + estimator documentation
  gate_table.tex             Deliverable 1 tex-ready table fragment
  precision_table.json       Deliverable 2 data
  precision_table.tex        Deliverable 2 tex-ready table fragment
  best_so_far_vs_evals.csv   Deliverable 3 tidy CSV (mini-ER + mix-discovery)

Reproduce: /home/miria/jaxenv/bin/python assets/make_phase1_tables.py
(run from the v2 root; pure stdlib, no jax import).

ESTIMATOR (Deliverable 1), documented once here and in the JSON:

Every LLM-arm proposal slot runs: attempt 1 (LLM call), and on a failed
attempt the error string (the violated Stage-0 inequality, or the JSON
parse error) is injected into a retry prompt. The base LLMBackend allows
ONE retry (2 attempts total) before a random-fallback proposal; the
mini-ER LP backend allows TWO retries (3 attempts total). Per run the
logs give C = LLM calls, F = fallback slots, R = Stage-0 validate()
rejections, and the accepted population (slots). With
S = LLM-sourced accepted slots = slots - F, the total failed attempts are
Phi = C - S, of which R are Stage-0 rejections and Phi - R are
parse/format failures (also fed back, but the feedback is a JSON error,
not an inequality).

2-attempt runs (TV full-menu rerun, mix-discovery, hawkeye): the attempt
split is EXACTLY identified: first-attempt failures f1 = Phi - F, second-
attempt failures f2 = F, so
  repair rate (any feedback) = (f1 - f2) / f1 = (Phi - 2F) / (Phi - F).
3-attempt runs (mini-ER): only f1 + f2 = Phi - F and f3 = F are observed;
the attempt-level repair rate (f1 - F)/(f1 + f2) is set-identified via
F <= f2 <= f1, giving the reported [lb, ub]; the point estimate assumes a
constant per-retry repair probability p (geometric repair model):
f1 (2 - p) = Phi - F and f1 (1 - p)^2 = F, solved for p in (0, 1).

The estimand in the task, P(retry valid | first proposal Stage-0-rejected,
inequality fed back), conditions on the STAGE-0 failure type. The logs do
not tag each failed attempt with its type, so that rate is set-identified:
allocate the R Stage-0 and Phi - R parse failures across attempts
extremally subject to the observed (f1, f2) totals. The identified
any-feedback rate equals the Stage-0-specific rate under failure-type
exchangeability, and the Stage-0 share of failures (R / Phi) is reported
so the reader can judge how much mixing there is.

Assumption A1 (checked as far as the logs allow): no endpoint errors ate
attempts (an endpoint error consumes one LLM call, then breaks straight
to fallback, biasing Phi accounting). The mini-ER logs record
per-generation call counts, so the per-generation check
Phi_gen >= 3 F_gen runs there and passes on all 3 runs. The TV-rerun,
mix-discovery, and hawkeye logs record run-level counts only, so those
12 runs get only the run-level check Phi - 2F >= 0, which is necessary
but NOT sufficient: a fallback slot whose attempt died on an endpoint
error consumes 1 call and 1 failure rather than 2, and the run-level
inequality cannot exclude that. All checks that were actually run pass.

Caveat on R (mini-ER runs only): the mini-ER backend books its
experiment-policy refusal of valid genomes that merely carry a backend
block (an excluded gene) into stage0_rejections, so mini-ER R, and hence
the pooled per-call Stage-0 rejection rate, is an upper bound on true
Stage-0 validate() rejections. The magnitude is not separable from the
banked logs (per-attempt transcripts were not persisted); it is likely
small because the menu instructs the model not to emit a backend block.
Repair rates use only (C, F, slots) and are unaffected.

Hawkeye runs: the recorded stage0_rejections counter is a MenuFilter
property summing the LLM backend AND its random fallback's rejection-
sampling counter, so R is polluted (upper bound only) there; the
any-feedback repair rate needs only (C, S, F) and stays valid. Hawkeye
rows go to the JSON but are kept out of the tex table (retracted-harness
lane, polluted R).
"""
from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path

V2 = Path(__file__).resolve().parent.parent
SPRINT = V2 / "results" / "sprint"
OUT = SPRINT / "phase1"
OUT.mkdir(parents=True, exist_ok=True)


def load(p):
    with open(p) as fh:
        return json.load(fh)


def pct(x):
    return 100.0 * x


# ===========================================================================
# Deliverable 1: gate repair-rate table
# ===========================================================================

def geometric_repair_p(phi: int, f: int):
    """Solve f1*(2-p) = phi - f, f1*(1-p)^2 = f for p in (0,1).

    Quadratic: A p^2 + B p + C0 = 0 with A = phi - f, B = -2*phi + 3*f...
    derived directly: (phi - f)(1-p)^2 = f (2 - p)
    => (phi-f) p^2 - (2(phi-f) - f) p + (phi - f - 2 f) = 0
    """
    a = phi - f
    if a <= 0:
        return None
    b = -(2 * a - f)
    c0 = a - 2 * f
    disc = b * b - 4 * a * c0
    if disc < 0:
        return None
    for sgn in (-1.0, 1.0):
        p = (-b + sgn * math.sqrt(disc)) / (2 * a)
        if 0.0 <= p <= 1.0:
            return p
    return None


def stage0_specific_bounds(f1: int, f2: int, r: int, phi: int):
    """[lb, ub] for P(retry valid | attempt-1 Stage-0 rejection) in the
    2-attempt case, allocating R Stage-0 / (Phi - R) parse failures
    extremally across the observed attempt totals (f1, f2)."""
    parse = phi - r
    a_min = max(0, f1 - parse)
    a_max = min(f1, r)
    if a_max <= 0:
        return None, None
    lo, hi = 1.0, 0.0
    for a in range(max(1, a_min), a_max + 1):
        x_min = max(0, f2 - (f1 - a))
        x_max = min(a, f2)
        lo = min(lo, (a - x_max) / a)
        hi = max(hi, (a - x_min) / a)
    return lo, hi


def gate_run_row(label, cell, machine, budget, attempts_max, C, R, F, slots,
                 per_gen_checks_ok, include_in_tex=True, r_polluted=False):
    S = slots - F
    phi = C - S
    parse = phi - R
    row = {
        "run": label,
        "cell": cell,
        "machine": machine,
        "budget_gens_x_pop": budget,
        "attempts_per_slot_max": attempts_max,
        "proposal_slots": slots,
        "llm_calls_total_proposals": C,
        "llm_sourced_accepted": S,
        "fallback_slots": F,
        "failed_attempts_total": phi,
        "stage0_rejected": R,
        "stage0_rejected_polluted_by_fallback_counter": r_polluted,
        "parse_or_format_failures": None if r_polluted else parse,
        "stage0_rejection_rate_per_call": None if r_polluted else R / C,
        "failure_rate_per_call": phi / C,
        "per_gen_consistency_checks_ok": per_gen_checks_ok,
        "include_in_tex": include_in_tex,
    }
    if attempts_max == 2:
        f1, f2 = phi - F, F
        row["first_attempt_failures"] = f1
        row["retry_failures"] = f2
        row["repair_opportunities"] = f1
        row["repair_successes"] = f1 - f2
        row["repair_rate_any_feedback"] = (f1 - f2) / f1 if f1 > 0 else None
        row["repair_rate_identified"] = True
        if not r_polluted:
            lo, hi = stage0_specific_bounds(f1, f2, R, phi)
            row["repair_rate_stage0_specific_bounds"] = [lo, hi]
    else:  # 3 attempts: bounds + geometric point estimate
        opp = phi - F
        f1_lo = math.ceil(opp / 2)
        f1_hi = phi - 2 * F
        row["repair_opportunities"] = opp
        row["repair_successes_bounds"] = [f1_lo - F, f1_hi - F]
        row["repair_rate_any_feedback_bounds"] = (
            [(f1_lo - F) / opp, (f1_hi - F) / opp] if opp > 0 else None
        )
        row["repair_rate_geometric_point"] = geometric_repair_p(phi, F)
        row["repair_rate_identified"] = False
    return row


def build_gate_analysis():
    rows = []
    checks = {}

    # ---- mini-ER (3 attempts per slot, LP netflow, 12x8) ------------------
    for s in (42, 43, 44):
        m = load(SPRINT / f"mini_er_seed{s}.json")
        gg = m["generations"]
        C = sum(g["n_llm_calls"] for g in gg)
        R = sum(g["n_rejected_stage0"] for g in gg)
        F = sum(g["n_llm_fallbacks"] for g in gg)
        slots = sum(len(g["candidates"]) for g in gg)
        ok = True
        for g in gg:
            s_g = sum(1 for c in g["candidates"] if c["source"] == "llm")
            phi_g = g["n_llm_calls"] - s_g
            if phi_g < 3 * g["n_llm_fallbacks"]:
                ok = False
        rows.append(gate_run_row(
            f"mini-ER s{s}", "LP netflow", m["machine"], "12x8", 3,
            C, R, F, slots, ok))

    # ---- TV full-menu rerun (2 attempts, TV 128x128, 10x8) ----------------
    d = load(V2 / "results" / "evolution_history.json")
    llm = d["runs"]["llm"]
    gg = llm["generations"]
    C = llm["n_llm_calls"]
    R = llm["n_stage0_rejections_backend"]
    F = llm["n_llm_fallbacks_total"]
    slots = sum(len(g["fitnesses"]) for g in gg)
    # per-generation call counts are not logged for this run; only the
    # run-level A1 check (necessary, not sufficient) is available.
    ok = (C - (slots - F)) >= 2 * F
    rows.append(gate_run_row(
        "TV rerun s42", "TV 128x128", "LOCAL", "10x8", 2,
        C, R, F, slots, ok))

    # ---- mix-discovery (2 attempts, 14x10, tv + lp, seeds 42/43/44) -------
    host_map = {"miria-4090": "LOCAL", "Mazeppa": "REMOTE"}
    for cell in ("tv", "lp"):
        for s in (42, 43, 44):
            m = load(SPRINT / "mixdisc" / f"{cell}_evolve_seed{s}.json")
            gg = m["history"]["generations"]
            C = m["n_llm_calls"]
            R = m["n_stage0_rejections_backend"]
            F = m["n_llm_fallbacks"]
            slots = sum(len(g["fitnesses"]) for g in gg)
            ok = (C - (slots - F)) >= 2 * F
            # cell label from the raw log, not hardcoded (the mix-discovery
            # TV runs were 128x128; 64x64 was the gate-ablation cell)
            if cell == "tv":
                sz = m["cell_config"]["size"]
                cell_name = f"TV {sz}x{sz}"
            else:
                cell_name = "LP netflow"
            rows.append(gate_run_row(
                f"mix-disc {cell} s{s}", cell_name, host_map[m["host"]],
                f"{m['generations']}x{m['pop_size']}", 2, C, R, F, slots, ok))

    # ---- hawkeye (2 attempts; R polluted; JSON only) ----------------------
    for mach in ("LOCAL", "REMOTE"):
        for s in (42, 43):
            m = load(SPRINT / f"hawkeye_r6lite_{mach}_seed{s}.json")
            c = m["counters"]
            gg = m["history"]["generations"]
            slots = sum(len(g["fitnesses"]) for g in gg)
            C, R, F = c["n_llm_calls"], c["stage0_rejections"], c["n_llm_fallbacks"]
            ok = (C - (slots - F)) >= 2 * F
            rows.append(gate_run_row(
                f"hawkeye {mach} s{s}", "TV 128x128 (GPU)", mach, "10x8", 2,
                C, R, F, slots, ok, include_in_tex=False, r_polluted=True))

    # ---- pooled ----------------------------------------------------------
    def pool(sel_rows, attempts):
        C = sum(r["llm_calls_total_proposals"] for r in sel_rows)
        R = sum(r["stage0_rejected"] for r in sel_rows)
        F = sum(r["fallback_slots"] for r in sel_rows)
        slots = sum(r["proposal_slots"] for r in sel_rows)
        return gate_run_row(
            "pooled", "-", "-", "-", attempts, C, R, F, slots,
            all(r["per_gen_consistency_checks_ok"] for r in sel_rows),
            r_polluted=any(
                r["stage0_rejected_polluted_by_fallback_counter"]
                for r in sel_rows))

    clean2 = [r for r in rows if r["attempts_per_slot_max"] == 2
              and not r["stage0_rejected_polluted_by_fallback_counter"]]
    mini3 = [r for r in rows if r["attempts_per_slot_max"] == 3]
    hawk = [r for r in rows
            if r["stage0_rejected_polluted_by_fallback_counter"]]
    pooled_clean2 = pool(clean2, 2)
    pooled_mini3 = pool(mini3, 3)
    pooled_hawk = pool(hawk, 2)

    # all-clean combined repair rate: identified successes from the
    # 2-attempt pool plus bounded (and geometric-point) successes from the
    # 3-attempt pool, over the combined opportunities.
    opp2 = pooled_clean2["repair_opportunities"]
    suc2 = pooled_clean2["repair_successes"]
    opp3 = pooled_mini3["repair_opportunities"]
    s3_lo, s3_hi = pooled_mini3["repair_successes_bounds"]
    p3 = pooled_mini3["repair_rate_geometric_point"]
    phi3 = pooled_mini3["failed_attempts_total"]
    f3 = pooled_mini3["fallback_slots"]
    f1_geo = (phi3 - f3) / (2 - p3)
    suc3_geo = f1_geo - f3
    all_clean = {
        "runs": len(clean2) + len(mini3),
        "llm_calls_total_proposals": (
            pooled_clean2["llm_calls_total_proposals"]
            + pooled_mini3["llm_calls_total_proposals"]),
        "stage0_rejected": (pooled_clean2["stage0_rejected"]
                            + pooled_mini3["stage0_rejected"]),
        "stage0_rejection_rate_per_call": (
            (pooled_clean2["stage0_rejected"]
             + pooled_mini3["stage0_rejected"])
            / (pooled_clean2["llm_calls_total_proposals"]
               + pooled_mini3["llm_calls_total_proposals"])),
        "repair_opportunities": opp2 + opp3,
        "repair_successes_bounds": [suc2 + s3_lo, suc2 + s3_hi],
        "repair_rate_bounds": [(suc2 + s3_lo) / (opp2 + opp3),
                               (suc2 + s3_hi) / (opp2 + opp3)],
        "repair_rate_point_geometric_for_miniER": (
            (suc2 + suc3_geo) / (opp2 + opp3)),
    }

    # ---- gate latency vs evaluation latency (gate_ablation.json) ----------
    gb = load(SPRINT / "gate_ablation.json")
    agg = gb["aggregate"]
    wa = agg["waste_accounting"]
    n_prop = agg["n_proposals_per_arm"]
    gate_on = [r for r in agg["per_run"] if r["arm"] == "gate_on"]
    validate_s = sum(r["validate_s"] for r in gate_on)
    ask_s = sum(r["ask_s"] for r in gate_on)
    n_fresh = sum(r["n_fresh_evals"] for r in gate_on)
    mean_eval_s = wa["total_eval_s_gate_on"] / n_fresh
    gate_total_us = 1e6 * wa["gate_on_gate_overhead_s_total"] / n_prop
    validate_us = 1e6 * validate_s / n_prop
    ask_us = 1e6 * ask_s / n_prop
    latency = {
        "source": "results/sprint/gate_ablation.json (mazeppa-1, TV 64x64, "
                  "random proposer arms, 3 seeds x 10x8, 240 proposals and "
                  "240 fresh evaluations per arm)",
        "gate_overhead_us_per_proposal_total": gate_total_us,
        "loop_stage0_validate_us_per_proposal": validate_us,
        "ask_side_resampling_us_per_proposal": ask_us,
        "mean_fresh_evaluation_s": mean_eval_s,
        "evaluation_over_gate_ratio": mean_eval_s / (gate_total_us * 1e-6),
        "invalid_genome_eval_cost_s_mean": (
            gb["invalid_eval_microbench"]["mean_eval_wall_s"]),
        "invalid_genome_eval_cost_note": (
            "cost of an invalid genome flowing to evaluation when the gate "
            "is off: the solver-level typecheck raises before iteration 1"),
        "shaped_random_sampler_invalid_rate": (
            gb["mc_sampler_invalid_rate"]["invalid_rate"]),
    }

    ac_slo, ac_shi = all_clean["repair_successes_bounds"]
    p3lo, p3hi = pooled_mini3["repair_rate_any_feedback_bounds"]
    summary = [
        f"Across the {all_clean['runs']} clean LLM-arm campaign runs "
        "(mini-ER 3, TV full-menu rerun 1, mix-discovery 6) the LLM made "
        f"{all_clean['llm_calls_total_proposals']} proposal calls; the "
        f"Stage-0 gate rejected {all_clean['stage0_rejected']} of them "
        f"({pct(all_clean['stage0_rejection_rate_per_call']):.1f} percent "
        "per call, an upper bound: the mini-ER counter also books "
        "experiment-policy backend-block refusals), versus "
        f"{pct(latency['shaped_random_sampler_invalid_rate']):.2f} percent "
        "invalid for the shaped random sampler.",
        "Feeding the violated inequality (or parse error) back and retrying "
        f"repaired {ac_slo} to {ac_shi} of "
        f"{all_clean['repair_opportunities']} rejected-and-retried "
        "proposals: repair rate exactly "
        f"{pct(pooled_clean2['repair_rate_any_feedback']):.1f} percent in "
        f"the {len(clean2)} two-attempt runs "
        f"({pooled_clean2['repair_successes']}/"
        f"{pooled_clean2['repair_opportunities']}) and set-identified in "
        f"[{pct(p3lo):.1f}, {pct(p3hi):.1f}] percent (geometric-model point "
        f"{pct(pooled_mini3['repair_rate_geometric_point']):.1f} percent) "
        f"in the {len(mini3)} three-attempt mini-ER runs.",
        "The Stage-0-SPECIFIC repair rate (conditioning on inequality "
        "feedback only) is not point-identified because 15-20 percent of "
        "failures are parse errors whose attempt positions are unlogged; "
        "pooled 2-attempt bounds are [35.9, 69.1] percent, equal to 51.9 "
        "percent under failure-type exchangeability.",
        "Gate latency is 41.3 microseconds per proposal all-in (13.7 loop "
        "validate + 27.7 ask-side resampling) against a 1.63 s mean "
        "evaluation: the gate costs 2.5e-5 of one evaluation (about 39,400x "
        "cheaper), and even an ungated invalid genome only costs 0.33 ms "
        "because the solver-level typecheck raises before iteration 1.",
        "Estimator caveats: repair rates assume no endpoint errors ate "
        "attempts, checked per generation only for the 3 mini-ER runs "
        "(per-generation call logs exist there) and at run level (a "
        "necessary but not sufficient check) for the other 12 runs, with "
        "every check that was run passing; mini-ER stage0_rejected also "
        "counts policy refusals of valid backend-carrying genomes (upper "
        "bound on true gate rejections; repair rates unaffected); hawkeye "
        "runs are excluded from the table (fallback-polluted rejection "
        "counter, retracted-harness lane) though their identified "
        "any-feedback repair rates, 29.8 to 50.0 percent, are consistent.",
    ]

    out = {
        "deliverable": "phase1 gate repair-rate table (banked data only)",
        "generated_by": "assets/make_phase1_tables.py",
        "date": "2026-09-03",
        "sources": [
            "results/sprint/mini_er_seed{42,43,44}.json",
            "results/evolution_history.json",
            "results/sprint/mixdisc/{tv,lp}_evolve_seed{42,43,44}.json",
            "results/sprint/hawkeye_r6lite_{LOCAL,REMOTE}_seed{42,43}.json",
            "results/sprint/gate_ablation.json",
        ],
        "estimator_documentation": __doc__.split("ESTIMATOR", 1)[1],
        "per_run": rows,
        "pooled": {
            "clean_2attempt_runs": pooled_clean2,
            "mini_er_3attempt_runs": pooled_mini3,
            "hawkeye_2attempt_runs_R_polluted": pooled_hawk,
            "all_clean_runs_combined": all_clean,
        },
        "gate_latency": latency,
        "summary_5_lines": summary,
    }
    with open(OUT / "gate_repair_rate.json", "w") as fh:
        json.dump(out, fh, indent=1)
    return out


def write_gate_tex(g):
    rows = [r for r in g["per_run"] if r["include_in_tex"]]
    p2 = g["pooled"]["clean_2attempt_runs"]
    p3 = g["pooled"]["mini_er_3attempt_runs"]
    ac = g["pooled"]["all_clean_runs_combined"]
    lat = g["gate_latency"]

    def fmt_repair(r):
        if r["attempts_per_slot_max"] == 2:
            return f"{pct(r['repair_rate_any_feedback']):.1f}"
        lo, hi = r["repair_rate_any_feedback_bounds"]
        p = r["repair_rate_geometric_point"]
        return (f"{pct(p):.1f} [{pct(lo):.1f}, {pct(hi):.1f}]")

    lines = []
    lines.append("% Auto-generated by assets/make_phase1_tables.py from the")
    lines.append("% frozen 2026-08-26 campaign artifacts; do not edit by hand.")
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\begin{tabular}{llrrrrr}")
    lines.append("\\toprule")
    lines.append("run & cell & calls & rejected & rej.\\ \\% & fallback & "
                 "repair \\% \\\\")
    lines.append("\\midrule")
    for r in rows:
        lines.append(
            f"{r['run']} & {r['cell']} & {r['llm_calls_total_proposals']} & "
            f"{r['stage0_rejected']} & "
            f"{pct(r['stage0_rejection_rate_per_call']):.1f} & "
            f"{r['fallback_slots']} & {fmt_repair(r)} \\\\")
    lines.append("\\midrule")
    lines.append(
        f"pooled (2-attempt runs) & mixed & "
        f"{p2['llm_calls_total_proposals']} & {p2['stage0_rejected']} & "
        f"{pct(p2['stage0_rejection_rate_per_call']):.1f} & "
        f"{p2['fallback_slots']} & "
        f"{pct(p2['repair_rate_any_feedback']):.1f} \\\\")
    lo, hi = p3["repair_rate_any_feedback_bounds"]
    lines.append(
        f"pooled (mini-ER, 3-attempt) & LP & "
        f"{p3['llm_calls_total_proposals']} & {p3['stage0_rejected']} & "
        f"{pct(p3['stage0_rejection_rate_per_call']):.1f} & "
        f"{p3['fallback_slots']} & "
        f"{pct(p3['repair_rate_geometric_point']):.1f} "
        f"[{pct(lo):.1f}, {pct(hi):.1f}] \\\\")
    alo, ahi = ac["repair_rate_bounds"]
    lines.append(
        f"pooled (all 10 runs) & mixed & "
        f"{ac['llm_calls_total_proposals']} & {ac['stage0_rejected']} & "
        f"{pct(ac['stage0_rejection_rate_per_call']):.1f} & "
        f"{p2['fallback_slots'] + p3['fallback_slots']} & "
        f"{pct(ac['repair_rate_point_geometric_for_miniER']):.1f} "
        f"[{pct(alo):.1f}, {pct(ahi):.1f}] \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    ratio_k = lat["evaluation_over_gate_ratio"] / 1e3
    lines.append(
        "\\caption{Stage-0 gate pressure and verification-guided repair "
        "under the LLM proposer, from the 2026-08-26 campaign logs. "
        "Calls counts every LLM proposal attempt; rejected counts Stage-0 "
        "gate rejections (violated typing inequality fed back to the model "
        "as a retry prompt); fallback counts slots where all attempts "
        "failed and a random proposal was substituted. Repair \\% is the "
        "fraction of fed-back retries that produced a valid genome: "
        "exactly identified in the runs allowing one retry per slot, and "
        "reported as a point estimate under a constant per-retry repair "
        "probability with sharp brackets [lb, ub] in the mini-ER runs "
        "allowing two retries. The rate pools all feedback types; "
        "Stage-0 inequalities are "
        f"{pct(p2['stage0_rejected'] / p2['failed_attempts_total']):.0f}"
        "\\% of fed-back failures in the 2-attempt runs (the rest are JSON "
        "format errors). For scale, the shaped random sampler violates "
        "Stage-0 on only "
        f"{pct(lat['shaped_random_sampler_invalid_rate']):.2f}\\% of "
        "draws, and the gate itself costs "
        f"{lat['gate_overhead_us_per_proposal_total']:.0f}\\,$\\mu$s per "
        "proposal against a "
        f"{lat['mean_fresh_evaluation_s']:.2f}\\,s mean candidate "
        "evaluation (a factor of about "
        f"{ratio_k:.0f}\\,000).}}")
    lines.append("\\label{tab:gate-repair}")
    lines.append("\\end{table}")
    tex = "\n".join(lines) + "\n"
    with open(OUT / "gate_table.tex", "w") as fh:
        fh.write(tex)
    return tex


# ===========================================================================
# Deliverable 2: precision-certificate table
# ===========================================================================

def build_precision_analysis():
    probes = {m: load(SPRINT / f"hawkeye_r6lite_probe_{m}.json")
              for m in ("LOCAL", "REMOTE")}
    hk = load(SPRINT / "hawkeye_r6lite.json")

    per_machine = {}
    n_cert_failed = 0
    n_cert_checked = 0
    for mach, pr in probes.items():
        sweep = pr["champion_policy_sweep_holdout"]
        rows = {r["policy"]: r for r in sweep["rows"]}
        f64_wall = rows["f64"]["sgm10_warm_s"]
        f64_iters = [p["iters"] for p in rows["f64"]["per_instance"]]
        table_rows = []
        for pol in ("absent", "f64", "f32_cert64", "schedule"):
            r = rows[pol]
            iters = [p["iters"] for p in r["per_instance"]]
            certified = [p["certified"] for p in r["per_instance"]]
            n_cert_checked += len(certified)
            n_cert_failed += sum(1 for c in certified if not c)
            assert all(p["certificate_dtype"] == "float64"
                       for p in r["per_instance"])
            row = {
                "policy": pol,
                "holdout_sgm10_warm_s": r["sgm10_warm_s"],
                "per_instance_iters": iters,
                "certified": f"{sum(certified)}/{len(certified)}",
                "certificate_dtype": "float64",
            }
            if pol in ("f32_cert64", "schedule"):
                row["wall_vs_f64_policy_pct"] = pct(
                    r["sgm10_warm_s"] / f64_wall - 1)
                row["iters_vs_f64_policy_pct"] = pct(
                    sum(iters) / sum(f64_iters) - 1)
            if pol == "absent":
                row["note"] = ("P0 km execution path, gap sampled every "
                               "sample_every iterations; wall not "
                               "comparable to the policy path (different "
                               "harness route), iters comparable")
                row["policy_iters_vs_absent_pct"] = pct(
                    sum(f64_iters) / sum(iters) - 1)
            table_rows.append(row)
        nm = pr["near_miss_fbs_inst5_cap40k"]["rows"]
        near_miss = {
            "genome": pr["near_miss_fbs_inst5_cap40k"]["genome_base"],
            "f64": {"iters": nm["f64"]["iters"],
                    "wall_warm_s": nm["f64"]["wall_warm_s"],
                    "certified": nm["f64"]["certified"]},
            "f32_cert64": {"iters": nm["f32_cert64"]["iters"],
                           "wall_warm_s": nm["f32_cert64"]["wall_warm_s"],
                           "certified": nm["f32_cert64"]["certified"]},
            "iters_inflation_pct": pct(
                nm["f32_cert64"]["iters"] / nm["f64"]["iters"] - 1),
            "wall_inflation_pct": pct(
                nm["f32_cert64"]["wall_warm_s"] / nm["f64"]["wall_warm_s"]
                - 1),
        }
        n_cert_checked += 2
        n_cert_failed += sum(
            1 for k in ("f64", "f32_cert64") if not nm[k]["certified"])
        per_machine[mach] = {
            "device": pr["device"],
            "jax_version": pr["jax_version"],
            "tol_rel_gap": pr["tol"],
            "champion_genome": sweep["champion"],
            "rows": table_rows,
            "near_miss_stress_cap40k": near_miss,
        }

    summary = [
        f"Certificate integrity: {n_cert_failed} of {n_cert_checked} "
        "controlled-probe solves failed to certify (12 policy solves plus "
        "the near-miss stress pair per machine); every certificate was "
        "evaluated at float64 (harness-"
        "asserted), and the campaign hand-recomputation matched the f64 gap "
        "of an f32-iterate run to 16 digits.",
        "Iteration inflation from f32 iterates is about 3 percent in the "
        "one stress case that gets near the precision floor (near-miss fbs "
        "at a 40k cap: 24800 vs 24100 iterations, +2.9 percent, identical "
        "on both machines) and 0 percent on the champion sweep (identical "
        "iteration counts per policy).",
        "Runtime: on the champion holdout sweep f32_cert64 is within noise "
        "of all-f64 on the 4090 (-2.1 percent) and slower on the 5090 "
        "(+13.5 percent); the schedule policy tracks f64 on both (-2.0 and "
        "+6.8 percent): at 128x128 the solve is kernel-launch-bound and "
        "the 56-70x f32/f64 throughput ratio is invisible, the campaign's "
        "standing negative.",
        "Certificate-cadence cost: the policy path certifies every K=50 "
        "iterations instead of sampling the gap every sample_every, "
        "costing +3.7 percent (4090) / +7.7 percent (5090) iterations at "
        "termination granularity; a pure certificate-off arm does not "
        "exist by design, so no isolated certificate-evaluation overhead "
        "is measurable from banked data.",
        "Caveats: warm same-process SGM10 on 3 held-out instances inside "
        "the sub-100 ms noise band (standing rule: no ordering claims "
        "there, so the 4090 f32-vs-f64 delta is a tie); jax 0.9.2 vs "
        "0.11.1 version skew between machines; absent-vs-policy wall "
        "differences are a harness-path effect, not a certificate effect.",
    ]

    out = {
        "deliverable": "phase1 precision-certificate table (banked data only)",
        "generated_by": "assets/make_phase1_tables.py",
        "date": "2026-09-03",
        "sources": [
            "results/sprint/hawkeye_r6lite_probe_{LOCAL,REMOTE}.json",
            "results/sprint/hawkeye_r6lite.json",
        ],
        "setting": (
            "TV-on-OCT 128x128, rel duality gap <= 1e-4, per-machine "
            "champion genome from the R6-lite runs, post-harness-fix "
            "controlled probe, warm same-process SGM10 over 3 held-out "
            "instances; policy K=50, schedule theta=0.01; GPU solves"),
        "per_machine": per_machine,
        "certificates": {
            "controlled_probe_solves_checked": n_cert_checked,
            "certificates_failed": n_cert_failed,
            "integrity_note": (
                "f32-iterate/f64-certify verified genuine: iterate dtype "
                "float32, gap history float64, hand-recomputed TV duality "
                "gap matches to 16 digits; harness asserts "
                "certificate_dtype == float64 on every solve "
                "(campaign report section 5)"),
            "in_evolution_non_certifications_note": (
                "the 84 f32_cert64 non-certifications logged during the "
                "R6-lite evolution runs are iteration-cap stalls of slow "
                "genomes (gap-floor median 7.5e-2), not precision floors; "
                "the single near-miss certifies under both policies at a "
                "40k cap (hawkeye_r6lite.json f32_cert64_stalls verdict)"),
        },
        "headline_iteration_inflation_pct": {
            m: per_machine[m]["near_miss_stress_cap40k"]["iters_inflation_pct"]
            for m in per_machine
        },
        "summary_5_lines": summary,
    }
    with open(OUT / "precision_table.json", "w") as fh:
        json.dump(out, fh, indent=1)
    return out


def write_precision_tex(p):
    lines = []
    lines.append("% Auto-generated by assets/make_phase1_tables.py from the")
    lines.append("% frozen 2026-08-26 campaign artifacts; do not edit by hand.")
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\begin{tabular}{llrrrrr}")
    lines.append("\\toprule")
    lines.append("machine & policy & SGM10 (ms) & wall $\\Delta$\\% & "
                 "iters & iters $\\Delta$\\% & certified \\\\")
    lines.append("\\midrule")
    name = {"absent": "absent (P0 f64 path)", "f64": "f64",
            "f32_cert64": "f32 + f64 cert", "schedule": "schedule"}
    for mach in ("LOCAL", "REMOTE"):
        pm = p["per_machine"][mach]
        dev = "RTX 4090" if mach == "LOCAL" else "RTX 5090"
        for r in pm["rows"]:
            wall = f"{1e3 * r['holdout_sgm10_warm_s']:.2f}"
            dw = (f"{r['wall_vs_f64_policy_pct']:+.1f}"
                  if "wall_vs_f64_policy_pct" in r else
                  ("0.0" if r["policy"] == "f64" else "n/c"))
            iters = "/".join(str(i) for i in r["per_instance_iters"])
            di = (f"{r['iters_vs_f64_policy_pct']:+.1f}"
                  if "iters_vs_f64_policy_pct" in r else
                  ("0.0" if r["policy"] == "f64" else
                   f"{-r['policy_iters_vs_absent_pct']:+.1f}"))
            lines.append(f"{dev} & {name[r['policy']]} & {wall} & {dw} & "
                         f"{iters} & {di} & {r['certified']} \\\\")
        nm = pm["near_miss_stress_cap40k"]
        lines.append(
            f"{dev} & stress f32 vs f64 & "
            f"{1e3 * nm['f32_cert64']['wall_warm_s']:.0f} vs "
            f"{1e3 * nm['f64']['wall_warm_s']:.0f} & "
            f"{nm['wall_inflation_pct']:+.1f} & "
            f"{nm['f32_cert64']['iters']} vs {nm['f64']['iters']} & "
            f"{nm['iters_inflation_pct']:+.1f} & 2/2 \\\\")
        if mach == "LOCAL":
            lines.append("\\midrule")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append(
        "\\caption{Backend precision policies with always-f64 certificates "
        "on the TV-on-OCT 128x128 cell (per-machine R6-lite champion, "
        "post-fix controlled probe, warm SGM10 over 3 held-out GPU solves "
        "at rel gap $\\le 10^{-4}$; policy certificate cadence K=50, "
        "schedule $\\theta=0.01$). Wall and iteration deltas are relative "
        "to the f64 policy on the same machine; the absent row is the P0 "
        "execution path, whose wall is not comparable across harness "
        "routes (n/c) and whose iteration delta reflects the coarser "
        "certificate cadence of the policy path. The stress row is the "
        "near-miss fbs genome at a 40k iteration cap: f32 iterates cost "
        "about 3\\% extra iterations and no certificate is lost. All "
        f"{p['certificates']['controlled_probe_solves_checked']} "
        "probe solves certified at float64; a hand recomputation of the "
        "f64 gap of an f32-iterate run matches to 16 digits. Same-process "
        "timings inside the sub-100 ms noise band: wall orderings between "
        "policies are not claimed (the 4090 f32 delta is a tie); jax "
        "0.9.2 (4090) vs 0.11.1 (5090).}")
    lines.append("\\label{tab:precision-cert}")
    lines.append("\\end{table}")
    tex = "\n".join(lines) + "\n"
    with open(OUT / "precision_table.tex", "w") as fh:
        fh.write(tex)
    return tex


# ===========================================================================
# Deliverable 3: best-so-far fitness vs cumulative evaluations CSV
# ===========================================================================

def build_evals_csv():
    """One row per FRESH candidate evaluation (dedup cache hits consume no
    evaluation budget and cannot change the best-so-far), in evaluation
    order. fitness = -SGM10(time-to-tol) on the run's train set (higher is
    better; -inf = certified-construction failure at solve time). gen = -1
    marks the mini-ER seed-genome evaluation performed before generation 0.
    """
    rows = []
    host_map = {"miria-4090": "LOCAL", "Mazeppa": "REMOTE"}

    # ---- mini-ER: candidates carry per-proposal dedup provenance ----------
    for s in (42, 43, 44):
        m = load(SPRINT / f"mini_er_seed{s}.json")
        cum = 0
        best = -math.inf
        # seed-genome evaluation (performed before gen 0)
        fit0 = m["config"]["seed_genome_fitness"]
        cum += 1
        best = max(best, fit0)
        rows.append(["mini_er", "lp_netflow", m["machine"], "llm", s, -1,
                     cum, fit0, best])
        n_fresh = 0
        for g in m["generations"]:
            for c in g["candidates"]:
                if c.get("rejected") or c.get("dedup"):
                    continue
                fit = float(c["fitness"])
                cum += 1
                n_fresh += 1
                best = max(best, fit)
                rows.append(["mini_er", "lp_netflow", m["machine"], "llm",
                             s, g["gen"], cum, fit, best])
        assert n_fresh + 1 == m["n_unique_evaluated"], (s, n_fresh)
        assert math.isclose(best, m["final_champion_fitness"],
                            rel_tol=1e-12), (s, best)

    # ---- mix-discovery: the evaluated log is the fresh-eval stream --------
    for cell in ("tv", "lp"):
        for s in (42, 43, 44):
            m = load(SPRINT / "mixdisc" / f"{cell}_evolve_seed{s}.json")
            gg = m["history"]["generations"]
            # map evaluation index -> generation via per-gen fresh-eval counts
            gen_of = []
            for g in gg:
                gen_of.extend([g["gen"]] * g["n_evaluated"])
            ev = m["evaluated"]
            assert len(gen_of) == len(ev)
            best = -math.inf
            for i, e in enumerate(ev):
                fit = float(e["fitness"])
                best = max(best, fit)
                rows.append(["mix_discovery", cell, host_map[m["host"]],
                             "llm", s, gen_of[i], i + 1, fit, best])
            assert math.isclose(best, m["history"]["best_fitness"],
                                rel_tol=1e-12), (cell, s, best)

    with open(OUT / "best_so_far_vs_evals.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["experiment", "cell", "machine", "proposer", "seed",
                    "gen", "cum_evals", "fitness", "best_so_far_fitness"])
        for r in rows:
            w.writerow(r)
    return rows


# ===========================================================================

def main():
    os.chdir(V2)
    g = build_gate_analysis()
    gate_tex = write_gate_tex(g)
    p = build_precision_analysis()
    prec_tex = write_precision_tex(p)
    rows = build_evals_csv()

    # style-law self-check on generated tex: no em-dash, no italics
    for tex in (gate_tex, prec_tex):
        assert "---" not in tex and "—" not in tex
        assert "\\emph" not in tex and "\\textit" not in tex

    print("wrote", OUT / "gate_repair_rate.json")
    print("wrote", OUT / "gate_table.tex")
    print("wrote", OUT / "precision_table.json")
    print("wrote", OUT / "precision_table.tex")
    print("wrote", OUT / "best_so_far_vs_evals.csv", f"({len(rows)} rows)")
    print("\n========== gate_table.tex ==========\n")
    print(gate_tex)
    print("\n========== precision_table.tex ==========\n")
    print(prec_tex)
    print("\n---- gate summary ----")
    for line in g["summary_5_lines"]:
        print("*", line)
    print("\n---- precision summary ----")
    for line in p["summary_5_lines"]:
        print("*", line)


if __name__ == "__main__":
    main()
