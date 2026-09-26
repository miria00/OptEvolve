"""Aggregate the four Hawkeye R6-lite runs into results/sprint/
hawkeye_r6lite.json: per-machine champions, the cross-hardware divergence
table, stall data, and the verbatim caveats. Pure file IO, no jax.

Usage: python scripts/hawkeye_r6lite_aggregate.py
"""
from __future__ import annotations

import json
import math
import os
import time

V2_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPRINT = os.path.join(V2_ROOT, "results", "sprint")

CAVEATS_VERBATIM = [
    "jax 0.9.2 on LOCAL vs 0.11.1 on REMOTE (version skew; this is R6-LITE, "
    "preliminary, the paper-grade run repeats on matched versions)",
    "LOCAL GPU shared with vLLM (contention noted; keep vLLM idle during "
    "timed solves by pausing between generations if needed)",
]


def _load(machine, seed):
    p = os.path.join(SPRINT, f"hawkeye_r6lite_{machine}_seed{seed}.json")
    with open(p) as fh:
        return json.load(fh)


def _num(v):
    return float("inf") if v == "inf" else (
        float("-inf") if v == "-inf" else (float("nan") if v == "nan" else v))


def _policy_str(genome_json):
    g = json.loads(genome_json)
    be = g.get("backend")
    if be is None:
        return "absent (P0 f64 km path)"
    s = f"{be['precision']} K={be['K']}"
    if be.get("theta") is not None:
        s += f" theta={be['theta']:.4g}"
    return s


def _row(run):
    ch = run["champion"]
    g = json.loads(ch["genome"])
    return {
        "machine": run["machine"],
        "seed": run["config"]["evolve_seed"],
        "device": run["device"],
        "jax": run["jax_version"],
        "champion_genome": ch["genome"],
        "champion_scheme": g["scheme"],
        "champion_precision_policy": _policy_str(ch["genome"]),
        "train_sgm10_s": _num(ch["train_sgm10_s"]),
        "holdout_sgm10_s": _num(ch["holdout"]["sgm10_time_s"]),
        "default_all_f64_holdout_sgm10_s": _num(
            run["default_all_f64"]["holdout"]["sgm10_time_s"]),
        "champion_f64_holdout_sgm10_s": _num(
            run["champion_f64"]["holdout"]["sgm10_time_s"]),
        "speedup_vs_all_f64_default": _num(
            run["speedups_holdout"]["champion_vs_default_f64"]),
        "speedup_vs_champion_forced_f64": _num(
            run["speedups_holdout"]["champion_vs_champion_f64"]),
        "n_unique_evaluated": run["counters"]["n_unique_evaluated"],
        "stage0_rejections": run["counters"]["stage0_rejections"],
        "llm_fallbacks": run["counters"]["n_llm_fallbacks"],
        "wall_time_s": run["wall_time_s"],
    }


def main():
    runs = {(m, s): _load(m, s) for m in ("LOCAL", "REMOTE") for s in (42, 43)}
    rows = [_row(r) for r in runs.values()]

    # per-machine champion = the seed-run champion with the better
    # held-out SGM10 (ties broken by train fitness)
    machine_champ = {}
    for m in ("LOCAL", "REMOTE"):
        cand = [r for r in rows if r["machine"] == m]
        cand.sort(key=lambda r: (r["holdout_sgm10_s"], r["train_sgm10_s"]))
        machine_champ[m] = cand[0]

    # backend-gene usage across each run's evaluated population
    backend_usage = {}
    stalls = []
    for (m, s), r in runs.items():
        gens = r["history"]["generations"]
        n_backend_champ_gens = sum(
            1 for g in gens
            if g["best_genome"] and json.loads(g["best_genome"]).get("backend"))
        backend_usage[f"{m}_seed{s}"] = {
            "generations_with_backend_gene_champion": n_backend_champ_gens,
            "final_champion_policy": _policy_str(r["champion"]["genome"]),
        }
        for st in r["stall_log"]:
            st2 = dict(st)
            st2["machine"], st2["seed"] = m, s
            stalls.append(st2)

    f32_stalls = [s for s in stalls if s["precision"] in ("f32_cert64",)]

    probes = {}
    for m in ("LOCAL", "REMOTE"):
        p = os.path.join(SPRINT, f"hawkeye_r6lite_probe_{m}.json")
        if os.path.exists(p):
            with open(p) as fh:
                probes[m] = json.load(fh)

    hw = {}
    for m in ("LOCAL", "REMOTE"):
        with open(os.path.join(SPRINT, f"hardware_{m}.json")) as fh:
            hw[m] = json.load(fh)
        hw[m]["f32_over_f64_ratio"] = round(
            hw[m]["tflops_f32"] / hw[m]["tflops_f64"], 1)

    def _is_f32_heavy(policy):
        return ("f32_cert64" in policy) or ("schedule" in policy)

    lc, rc = machine_champ["LOCAL"], machine_champ["REMOTE"]
    divergence = {
        "prediction": "both GPUs are FP64-poor consumer silicon (measured "
                      "f32/f64 throughput ratios LOCAL "
                      f"{hw['LOCAL']['f32_over_f64_ratio']}x, REMOTE "
                      f"{hw['REMOTE']['f32_over_f64_ratio']}x), so both "
                      "champions should go f32-heavy; the interesting "
                      "comparison is the schedule parameters (K, theta) "
                      "against memory bandwidth",
        "local_champion_f32_heavy": _is_f32_heavy(
            lc["champion_precision_policy"]),
        "remote_champion_f32_heavy": _is_f32_heavy(
            rc["champion_precision_policy"]),
        "policies_differ_between_machines": (
            lc["champion_precision_policy"] != rc["champion_precision_policy"]),
        "outcome": "PREDICTION NOT CONFIRMED, BUT THE EVOLUTIONARY "
                   "EVIDENCE IS RETRACTED AS A HARNESS ARTIFACT "
                   "(correction 2026-08-26): all four champions carry NO "
                   "backend block, i.e. the all-f64 P0 km execution path. "
                   "However, the recorded runs executed under a harness "
                   "bug: the policy execution path re-traced and "
                   "re-compiled its chunk runner on EVERY solve and timed "
                   "the compile, so every backend-gene genome paid a "
                   "constant 0.07-0.12 s penalty per evaluation (3-4x the "
                   "km solve time at 128x128) and evolution could never "
                   "retain a backend gene REGARDLESS of hardware. The bug "
                   "is fixed (atlas.backend.precision now caches AOT "
                   "compiles like the km path and wall_time excludes "
                   "compile); the post-fix probe shows the same champion "
                   "under an f64 policy now times FASTER than the km path "
                   "on both machines, i.e. under the fixed harness the "
                   "selection pressure points the OTHER way. The "
                   "evolutionary claim 'hardware did not select precision "
                   "genes' is therefore UNSUPPORTED by these runs and "
                   "needs a re-run under the fixed harness (folds into "
                   "the planned matched-jax paper-grade R6).",
        "mechanism": "What SURVIVES is the physical, like-for-like "
                     "conclusion: within the policy path (identical "
                     "harness), f32 iterates buy nothing at 128x128 "
                     "because the solve is kernel-launch-bound, not "
                     "FLOP-bound. Post-fix probe, same champion "
                     "iteration, warm-timed holdout SGM10: 4090 f64 "
                     "0.01136 vs f32_cert64 0.01113 (2 percent, inside "
                     "noise); 5090 f64 0.00483 vs f32_cert64 0.00548 "
                     "(f32 slightly SLOWER). The 56-70x FP64 throughput "
                     "penalty is invisible at this problem size. Note "
                     "the post-fix policy-vs-km gap (policy ~4x FASTER: "
                     "fewer in-graph history writes and a cheaper "
                     "certificate cadence) means absent-vs-present "
                     "backend comparisons remain harness-sensitive in "
                     "both directions; only within-policy-path "
                     "comparisons are statements about silicon. The "
                     "paper-grade R6 needs larger instances (FLOP-bound "
                     "regime) or the batched vmap backend for the f32 "
                     "lane to have anything physical to win.",
        "schedule_parameter_comparison": "No discovered schedules to "
                                         "compare: neither machine's "
                                         "champion carries a schedule "
                                         "gene, so the planned K/theta vs "
                                         "memory-bandwidth comparison has "
                                         "no data at this problem size.",
    }

    out = {
        "experiment": "hawkeye_r6lite_cross_hardware",
        "status": "R6-LITE, preliminary; evolutionary backend-gene negative RETRACTED as a harness artifact (2026-08-26), see divergence_assessment.outcome",
        "caveats_verbatim": CAVEATS_VERBATIM,
        "additional_caveats": [
            "vLLM contention handling: the four runs were scheduled "
            "sequentially (REMOTE seeds 42, 43 first, then LOCAL 42, 43), "
            "so no remote proposal traffic hit the shared 4090 while LOCAL "
            "was timing solves; within a generation the LLM ask() completes "
            "before any timed evaluation starts, so vLLM is idle during "
            "timed solves by construction of the loop",
            "policy-path timing (CORRECTED 2026-08-26): during the "
            "recorded evolution runs, solves with a backend block "
            "re-traced and re-compiled their chunk runner on every solve "
            "and the timed wall included that compile, a per-solve "
            "penalty of roughly 0.07-0.12 s (3-4x the km-path solve time "
            "at 128x128) that biased fitness against every backend-gene "
            "genome; the previously stated mitigation ('the warm re-run "
            "makes the two comparable') was FALSE, the warm re-run paid "
            "full compile again. Fixed in atlas.backend.precision "
            "(AOT-compile cached across solves, wall excludes compile); "
            "posthoc_controlled_probe now holds POST-FIX numbers; the "
            "evolutionary backend-gene negative is retracted, see "
            "divergence_assessment.outcome",
            "champion selection: the divergence_table champion per "
            "machine is the better of the two seed runs BY HELD-OUT "
            "SGM10, so its holdout column and speedups are max-over-2-"
            "seeds statistics on the same 3 held-out instances "
            "(optimistically selected; per-seed values are in runs: "
            "LOCAL s42 1.58x / s43 4.41x, REMOTE s42 4.56x / s43 10.4x). "
            "The selected LOCAL 4.41x did reproduce in a fresh N=3 GPU "
            "re-measurement (4.47x, tight), but treat the table as "
            "best-seed, not expectation",
            "the speedup columns measure STEPSIZE TUNING against a "
            "textbook default under the all-f64 km path (both champions "
            "carry no precision gene, policy ABSENT); they are NOT a "
            "hardware or precision effect",
            "speedup_vs_champion_forced_f64 is trivially about 1.0 in all "
            "four runs because every champion carries no backend block "
            "(the forced-f64 twin is the identical genome)",
            "fitness timing convention: warm re-run wall time to "
            "certification (first solve warms compile caches; only solves "
            "that certified are re-run and timed); non-certifying solves "
            "score inf under strict SGM10",
        ],
        "hardware_records": hw,
        "runs": rows,
        "divergence_table": {
            "columns": ["machine", "champion (scheme + precision policy)",
                        "holdout SGM10 s", "all-f64 default SGM10 s",
                        "speedup vs all-f64 default",
                        "speedup vs champion-forced-f64"],
            "selection_rule": "per-machine champion = the seed-run champion "
                              "with the better held-out SGM10 (ties broken "
                              "by train fitness): a MAX-OVER-2-SEEDS "
                              "statistic on the same 3 held-out instances; "
                              "per-seed values in runs",
            "LOCAL": machine_champ["LOCAL"],
            "REMOTE": machine_champ["REMOTE"],
        },
        "divergence_assessment": divergence,
        "backend_gene_usage": backend_usage,
        "f32_cert64_stalls": {
            "n": len(f32_stalls),
            "records": f32_stalls,
            "note": "instances where an f32_cert64 genome failed to certify "
                    "at rel gap <= 1e-4 within the iteration cap, with the "
                    "gap floor reached: Hawkeye-lane DATA, not failure",
            "verdict": "NO confirmed accumulation stalls at the 1e-4 tier. "
                       "Gap-floor quantiles of the 84 f32_cert64 "
                       "non-certifications: min 9.4e-4, median 7.5e-2, max "
                       "5.2e-1: these are slow-genome iteration-CAP stalls, "
                       "not precision floors. The single near-miss (fbs "
                       "tau=0.00612 rho=1.67, K=25, floor 9.351e-4 at the "
                       "10k cap, identical on both machines) was re-run "
                       "with a 40k cap: it certifies under BOTH f32_cert64 "
                       "(24800 iters) and its f64 twin (24100 iters) on "
                       "both machines, so f32 cost about 3 percent extra "
                       "iterations and no certificate was lost to "
                       "accumulation.",
        },
        "posthoc_controlled_probe": probes,
        "all_stall_records": stalls,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    dest = os.path.join(SPRINT, "hawkeye_r6lite.json")
    with open(dest, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print("wrote", dest)
    for m in ("LOCAL", "REMOTE"):
        c = machine_champ[m]
        print(f"{m}: {c['champion_scheme']} | {c['champion_precision_policy']}"
              f" | holdout {c['holdout_sgm10_s']} s | default "
              f"{c['default_all_f64_holdout_sgm10_s']} s | speedup "
              f"{c['speedup_vs_all_f64_default']}")


if __name__ == "__main__":
    main()
