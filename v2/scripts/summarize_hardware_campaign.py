"""Create a reviewable report and standalone figures from completed pilots."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import sqlite3

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/hardware_codex")
    args = ap.parse_args()
    root = Path(args.root)
    runs = {}
    for name in ("oct_cpu_v1", "oct_4090_v1", "oct_5090_v1", "lp_5090_v2"):
        p = root / name / "benchmark.json"
        if p.exists():
            d = json.loads(p.read_text())
            if d["status"] == "complete":
                runs[name] = d
    lines = ["# OptEvolve hardware implementation and diagnostic results", "",
        "Figures: [OCT timings](hardware_times.pdf), "
        "[joint solver/kernel comparison](joint_kernel_times.pdf). "
        "Validation logs are in `validation/`.", "",
        "These are exploratory measurements, not the frozen final paper evaluation. "
        "The OCT diagnostic uses 3 training and 3 held-out 128x128 patches, "
        "selected once with seed 20260906. Each measurement uses 3 warm repeats. "
        "Reported time is synchronized solver execution, including numerical "
        "certificate checks and excluding compilation/preprocessing. "
        "Every returned result was checked independently in NumPy/SciPy.", "",
        "LP uses only air05, trento1, and neos8 from the frozen calibration "
        "partition, with a 20,000-iteration diagnostic cap. The original "
        "pre-registration and final holdout remain untouched.", "",
        "## Training-selected execution policies", "",
        "The algorithm catalog contains existing textbook and banked policies "
        "for OCT, and frozen B0 plus expert wrapper settings for LP. Selection "
        "from this catalog is not autonomous algorithm discovery.", "",
        "| Run | Tolerance | Selected algorithm / execution | Holdout solves | Holdout seconds | Same algorithm legacy / selected |",
        "|---|---|---|---:|---:|---:|"]
    flat, summary = [], {}
    for name, run in runs.items():
        summary[name] = {}
        for tol, block in run["tolerances"].items():
            selected = block["joint_selected"]
            informative = selected["train"]["n_solved"] > 0
            legacy = next(r for r in block["rows"] if r["algorithm"] == selected["algorithm"] and r["policy"] == "legacy")
            h = selected["holdout"]
            comparable = h["n_solved"] == h["n"] == legacy["holdout"]["n_solved"]
            speedup = legacy["holdout"]["penalized_sgm1_s"] / h["penalized_sgm1_s"] if comparable and informative else None
            display_speed = f"{speedup:.2f}x" if speedup else "not comparable"
            description = f"{selected['algorithm']} / {selected['policy']}" if informative else "no training solves; selection uninformative"
            hold_desc = f"{h['n_solved']}/{h['n']}" if informative else "not selected"
            time_desc = f"{h['penalized_sgm1_s']:.6f}" if informative else "not selected"
            lines.append(f"| {name} | {tol} | {description} | {hold_desc} | {time_desc} | {display_speed} |")
            summary[name][tol] = {"selected_algorithm": selected["algorithm"],
                "selected_policy": selected["policy"], "holdout": h,
                "same_algorithm_legacy_speedup": speedup}
            summary[name][tol]["selection_status"] = "selected_on_training_only" if informative else "uninformative_all_non_solves"
            if not informative:
                summary[name][tol].update(selected_algorithm=None, selected_policy=None, holdout=None)
            for row in block["rows"]:
                for partition in ("train", "holdout"):
                    for point in row[partition]["rows"]:
                        flat.append({"run": name, "tol": tol, "partition": partition,
                            "algorithm": row["algorithm"], "policy": row["policy"],
                            "instance": point["instance_index"], "certified": point["certified"],
                            "solver_s": point["solver_s"], "iterations": point["iters"][0],
                            "certificate_checks": point["checks"],
                            "repeat_min_s": min(point["solver_samples_s"]),
                            "repeat_max_s": max(point["solver_samples_s"]),
                            "warm_call_s": point["warm_call_s"],
                            "prepare_s": point["prepare_s"], "cold_call_s": point["cold_call_s"]})
    if flat:
        with (root / "measurements.csv").open("w") as f:
            writer = csv.DictWriter(f, fieldnames=list(flat[0]))
            writer.writeheader()
            writer.writerows(flat)
    lines += ["", "## Cross-device transfer", "",
        "Rows are selected on training data on the source device, then evaluated "
        "using the destination device's already recorded holdout measurements. "
        "Values are time / destination-selected time; greater than 1 is a transfer penalty."]
    oct_runs = [k for k in runs if k.startswith("oct_")]
    for tol in ("0.0001", "1e-06"):
        lines += ["", f"Tolerance {tol}:", "",
                  "| Selected on | " + " | ".join(oct_runs) + " |",
                  "|---|" + "---:|" * len(oct_runs)]
        for source in oct_runs:
            chosen = runs[source]["tolerances"][tol]["joint_selected"]
            values = []
            for dest in oct_runs:
                b = runs[dest]["tolerances"][tol]
                r = next(r for r in b["rows"] if (r["algorithm"], r["policy"]) ==
                         (chosen["algorithm"], chosen["policy"]))
                h, best = r["holdout"], b["joint_selected"]["holdout"]
                values.append(f"{h['penalized_sgm1_s']/best['penalized_sgm1_s']:.2f}" if h["n_solved"] == h["n"] else "non-solve")
            lines.append("| " + source + " | " + " | ".join(values) + " |")
    pilot_path = root / "pilot_cpu_v1/pilot_0.0001.json"
    if pilot_path.exists():
        p = json.loads(pilot_path.read_text())
        lines += ["", "## Proposer context pilot", "",
            f"Status: {p['status']}. Two seeds per arm, target 8 fresh evaluations "
            "per seed (plus a shared baseline), CPU target, tolerance 1e-4. "
            "This is an implementation pilot, not statistical evidence of "
            "LLM superiority or a hardware-context effect.", "",
            "| Arm | Seed | Fresh evals | LLM calls | Fallbacks | Holdout solves | Holdout seconds |",
            "|---|---:|---:|---:|---:|---:|---:|"]
        for r in p["runs"]:
            h = r["holdout"]
            lines.append(f"| {r['arm']} | {r['seed']} | {r['fresh_evaluations']} | {r['llm_calls']} | {r['fallbacks']} | {h['n_solved']}/{h['n']} | {h['penalized_sgm1_s']:.6f} |")
    lines += ["", "## Joint solver and generated CUDA kernel diagnostic", "",
        "This second LP diagnostic uses comp07-2idx and neos-1171737 for "
        "selection, with neos-1171448 and neos-4300652-rahue held out from "
        "selection. All four belong to the frozen calibration partition and "
        "were chosen for previous solvability, so this is not a representative "
        "solve-rate evaluation. Tolerance 1e-6, 20,000 iterations, three repeats.", "",
        "Generated FP64 CSR templates fuse a matvec with its vector update. "
        "The two dependent updates remain separate launches. Thread-per-row "
        "and warp-per-row specializations pass reference admission before "
        "timing; original-coordinate certificates remain independent."]
    for joint_name in ("joint_lp_4090_v2", "joint_lp_5090_v1"):
        jp = root / joint_name / "benchmark.json"
        if not jp.exists():
            continue
        joint = json.loads(jp.read_text())
        if joint["status"] != "complete":
            continue
        block = joint["tolerances"]["1e-06"]
        # Derived from the already measured factorial, without retuning holdout.
        block["reverse_sequential_selected"] = min(
            (r for r in block["rows"] if r["policy"] == block["backend_only_selected"]["policy"]),
            key=lambda r: (-r["train"]["n_solved"], r["train"]["penalized_sgm1_s"]))
        summary[joint_name] = block
        lines += ["", f"### {joint_name}", "",
            "Selections are over an exhaustively measured finite catalog. "
            "They are not budget-matched evolutionary search arms.", "",
            "| Selection | Algorithm / implementation | Training solves | Holdout solves | Penalized holdout seconds |",
            "|---|---|---:|---:|---:|"]
        for label, key in (("Algorithm only", "algorithm_only_selected"),
                           ("Backend only", "backend_only_selected"),
                           ("Sequential", "sequential_selected"),
                           ("Reverse sequential", "reverse_sequential_selected"),
                           ("Joint", "joint_selected")):
            r = block[key]
            t, h = r["train"], r["holdout"]
            lines.append(f"| {label} | {r['algorithm']} / {r['policy']} | {t['n_solved']}/{t['n']} | {h['n_solved']}/{h['n']} | {h['penalized_sgm1_s']:.6f} |")
        same = all(block["joint_selected"][k] == block["sequential_selected"][k]
                   for k in ("algorithm", "policy"))
        lines += ["", "Joint and sequential selection chose the same recipe; "
                  "this catalog provides no evidence of a joint-search advantage." if same else
                  "Joint and sequential selection differed. Inspect held-out performance; "
                  "a different selection alone is not evidence of an advantage.", "",
                  "Fixed B0 algorithm, showing kernel/cadence effects:", "",
                  "| Implementation | Holdout solves | Penalized holdout seconds |",
                  "|---|---:|---:|"]
        for r in sorted(block["rows"], key=lambda r: r["policy"]):
            if r["algorithm"] == "frozen_B0":
                h = r["holdout"]
                lines.append(f"| {r['policy']} | {h['n_solved']}/{h['n']} | {h['penalized_sgm1_s']:.6f} |")
    db = root / "nsys_lp4090_final.sqlite"
    if not db.exists():
        db = root / "nsys_lp4090_nodes.sqlite"
    if db.exists():
        con = sqlite3.connect(db)
        names = ("base_step_block", "certificate_only", "solver_K1", "solver_K50", "solver_K200")
        captures = con.execute("SELECT start,end,text FROM NVTX_EVENTS").fetchall()
        trace_summary = {}
        for start, end, name in captures:
            if name not in names:
                continue
            rows = con.execute("SELECT s.value,COUNT(*),SUM(k.end-k.start) FROM "
                "CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName "
                "WHERE k.start>=? AND k.end<=? GROUP BY s.value", (start, end)).fetchall()
            total = sum(r[2] for r in rows)
            trace_summary[name] = {"kernel_count": sum(r[1] for r in rows),
                "gpu_kernel_sum_ms": total / 1e6,
                "scatter_kernel_share": sum(r[2] for r in rows if "scatter" in r[0]) / total,
                "kernels": rows}
        summary["nsight"] = trace_summary
        lines += ["", "## Nsight kernel evidence", "",
            "CUDA graph node tracing enabled. Counts cover 3 repetitions of "
            "200 iterations (or 3 standalone certificate calls). These are "
            "instrumented diagnostics; do not use their wall times as speedup claims.", "",
            "| Phase | Actual kernel executions | Scatter kernel time share |",
            "|---|---:|---:|"]
        for k, v in trace_summary.items():
            lines.append(f"| {k} | {v['kernel_count']} | {v['scatter_kernel_share']:.1%} |")
    custom_db = root / "nsys_lp4090_cuda.sqlite"
    if custom_db.exists():
        con = sqlite3.connect(custom_db)
        start, end = con.execute("SELECT start,end FROM NVTX_EVENTS WHERE text='base_step_block'").fetchone()
        rows = con.execute("SELECT s.value,COUNT(*),SUM(k.end-k.start) FROM "
            "CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName "
            "WHERE k.start>=? AND k.end<=? GROUP BY s.value", (start, end)).fetchall()
        summary["nsight_cuda"] = {"kernel_count": sum(r[1] for r in rows), "kernels": rows}
        lines += ["", "The generated `csr_thread128` base loop on trento1 emitted "
            f"{sum(r[1] for r in rows)} kernel executions over 600 iterations: "
            "two CUDA updates plus one loop-counter kernel per iteration. "
            "The XLA base loop emitted seven per iteration. Fewer launches did "
            "not make this variant faster on trento1; the serial dual-row update "
            "dominated its trace. The generated implementation is therefore "
            "an optional measured allele, never a mandatory replacement."]
    lines += ["", "## Interpretation and limits", "",
        "- The held-out OCT sample is only three images; repeated timings do not increase the number of independent images.",
        "- Both GPUs selecting the same recipe is not evidence of GPU-to-GPU specialization.",
        "- Pure f32 can lose certified solves at 1e-6. Numerical checking remains f64; an f64 check is not a floating-point convergence proof.",
        "- Joint catalog selection and sequential selection may choose the same recipe; report this rather than claiming a co-design interaction.",
        "- LP uses a shorter diagnostic iteration cap than the old assay. Its solve rates cannot replace the pre-registered calibration result.",
        "- Compilation and preparation are reported separately. Reuse/amortization assumptions matter for deployment.",
        "- The OCT and initial LP policy campaigns use XLA kernels. The separate joint LP diagnostic uses newly generated CUDA templates; no arbitrary LLM-generated kernel or new algorithm family is claimed.",
        "- The 4090 retained the existing vLLM allocation; benchmark runs preceded this campaign's LLM requests.", ""]
    (root / "REPORT.md").write_text("\n".join(lines))
    (root / "summary.json").write_text(json.dumps(summary, indent=2))
    if oct_runs:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        for ax, tol in zip(axes, ("0.0001", "1e-06")):
            xs = np.arange(len(oct_runs))
            for offset, policy in ((-.25, "legacy"), (0, "f64_K50"), (.25, "selected")):
                values = []
                for run in oct_runs:
                    b = runs[run]["tolerances"][tol]
                    r = b["joint_selected"] if policy == "selected" else next(r for r in b["rows"] if r["algorithm"] == "banked_policy" and r["policy"] == policy)
                    values.append(r["holdout"]["penalized_sgm1_s"] * 1000)
                ax.bar(xs + offset, values, width=.24, label=policy)
            ax.set_xticks(xs, [r.replace("oct_", "").replace("_v1", "") for r in oct_runs])
            ax.set_ylabel("Held-out SGM-1 solver time (ms)")
            ax.set_title(f"OCT 128x128, tolerance {tol}")
            ax.legend(frameon=False)
        fig.savefig(root / "hardware_times.png", dpi=180)
        fig.savefig(root / "hardware_times.pdf")
    joint_names = [n for n in ("joint_lp_4090_v2", "joint_lp_5090_v1") if n in summary]
    if joint_names:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
        menu = ["legacy", "f64_K50", "f64_K200", "csr_thread128_K50",
                "csr_thread128_K200", "csr_warp128_K50", "csr_warp128_K200",
                "csr_warp256_K50", "csr_warp256_K200"]
        algorithms = ["frozen_B0", "expert_relaxation"]
        fig, axes = plt.subplots(len(joint_names), 1, figsize=(13, 3*len(joint_names)),
                                 squeeze=False, constrained_layout=True)
        for ax, name in zip(axes[:, 0], joint_names):
            values = np.zeros((2, len(menu)))
            points = {}
            for r in summary[name]["rows"]:
                i, j = algorithms.index(r["algorithm"]), menu.index(r["policy"])
                values[i, j] = r["holdout"]["penalized_sgm1_s"]
                points[i, j] = r["holdout"]
            im = ax.imshow(values, cmap="YlOrRd", norm=LogNorm(vmin=.3, vmax=12), aspect="auto")
            for (i, j), h in points.items():
                ax.text(j, i, f"{values[i,j]:.2f}s\n{h['n_solved']}/{h['n']}",
                        ha="center", va="center", color="white" if values[i,j] > 3 else "black", fontsize=9)
            ax.set_yticks(range(2), ["B0", "B0 + relaxation"])
            ax.set_xticks(range(len(menu)), [p.replace("csr_", "").replace("f64_", "XLA ") for p in menu],
                          rotation=25, ha="right")
            ax.set_title(name.replace("joint_lp_", "").replace("_v1", "").replace("_v2", "") +
                         ": held-out LP diagnostic, tolerance 1e-6 (2 instances)")
            fig.colorbar(im, ax=ax, label="Penalized time (s)")
        fig.savefig(root / "joint_kernel_times.png", dpi=180)
        fig.savefig(root / "joint_kernel_times.pdf")
    print(root / "REPORT.md")


if __name__ == "__main__":
    main()
