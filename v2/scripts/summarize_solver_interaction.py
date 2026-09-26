"""Summarize completed development factorials without reselecting on validation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from atlas.bench.factorial import select_catalog


def summarize(run_dir):
    run_dir = Path(run_dir)
    protocol = json.loads((run_dir / "protocol.json").read_text())
    data = json.loads((run_dir / "data.json").read_text())
    results = json.loads((run_dir / "factorial.json").read_text())
    if results["status"] != "complete":
        raise ValueError(f"unfinished run: {run_dir}")
    out = {"run": run_dir.name, "device": protocol["device"],
           "protocol_version": protocol["version"],
           "catalog_campaign_s": results["campaign_s"], "tolerances": {}}
    for tol, block in results["tolerances"].items():
        choices = select_catalog(block["rows"], *protocol["baseline"])
        if choices != {k: tuple(v) for k, v in block["selections"].items()}:
            raise AssertionError("recorded selections disagree with training-only reconstruction")
        catalog = {(r["algorithm"], r["policy"]): r for r in block["rows"]}
        base = catalog[tuple(protocol["baseline"])]["validation"]
        measured = {}
        for arm, key in choices.items():
            r = catalog[key]
            measured[arm] = {"recipe": list(key), "train_solved": r["train"]["n_solved"],
                            "validation_solved": r["validation"]["n_solved"],
                            "n_validation": r["validation"]["n"],
                            "validation_penalized_sgm1_s": r["validation"]["penalized_sgm1_s"],
                            "baseline_over_selected": base["penalized_sgm1_s"] / r["validation"]["penalized_sgm1_s"]}
        out["tolerances"][tol] = measured
    audit = run_dir / "patient_group_audit.json"
    out["patient_audit"] = json.loads(audit.read_text()) if audit.exists() else data.get("group_audit")
    return out, protocol, data, results


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--paper-out", help="optional directory for generated LaTeX table")
    args = ap.parse_args()
    runs = [summarize(p) for p in args.runs]
    inputs = [[m["input_sha256"] for m in d["instances"]] for _, _, d, _ in runs]
    if any(x != inputs[0] for x in inputs[1:]):
        raise ValueError("device comparison requires identical input arrays")
    for _, p, d, _ in runs[1:]:
        for key in ("algorithms", "policies", "baseline", "tolerances"):
            if p[key] != runs[0][1][key]:
                raise ValueError(f"protocol mismatch: {key}")
        for key in ("train", "validation"):
            if d[key] != runs[0][2][key]:
                raise ValueError(f"partition mismatch: {key}")
        if p["source_sha256"] != runs[0][1]["source_sha256"]:
            raise ValueError("device comparison requires identical captured source hashes")
        for key in ("size", "n_train", "n_validation", "max_iters", "repeats",
                    "penalty_s", "data_seed", "order_seed", "synthetic"):
            if p["args"][key] != runs[0][1]["args"][key]:
                raise ValueError(f"measurement setting mismatch: {key}")
        if p["jax"] != runs[0][1]["jax"]:
            raise ValueError("device comparison requires matching JAX versions")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps([s for s, _, _, _ in runs], indent=2))
    lines = ["**Solver/execution interaction: development results**", "",
             "These runs exhaustively measure a finite catalog. They are not budget-matched search trials.",
             "All device runs use identical input hashes, candidate definitions, partitions and captured source hashes.",
             "Twelve 256x256 OCT development images, six train and six validation; two tolerances; 5,000-iteration cap; three warm repeats.",
             "The baseline is the banked Condat-Vu solver with compiled f64 execution and K=64, not legacy execution.",
             "A 120-second non-solve penalty enters SGM-1. Penalized values with failures are not actual mean solve times.", "",
             "| Device | Tolerance | Selection | Recipe | Validation solves | Penalized SGM-1 (s) | Baseline / selected |",
             "|---|---|---|---|---:|---:|---:|"]
    for summary, _, _, _ in runs:
        for tol, choices in summary["tolerances"].items():
            for arm in ("fixed", "algorithm_only", "implementation_only", "algorithm_then_implementation",
                        "implementation_then_algorithm", "joint"):
                v = choices[arm]
                lines.append(f"| {summary['device']} | {tol} | {arm} | {' / '.join(v['recipe'])} | "
                             f"{v['validation_solved']}/{v['n_validation']} | {v['validation_penalized_sgm1_s']:.6f} | "
                             f"{v['baseline_over_selected']:.3f}x |")
    lines += ["", "**Joint versus sequential**", ""]
    for summary, _, _, _ in runs:
        for tol, choices in summary["tolerances"].items():
            j = choices["joint"]
            for arm in ("algorithm_then_implementation", "implementation_then_algorithm"):
                s = choices[arm]
                ratio = s["validation_penalized_sgm1_s"] / j["validation_penalized_sgm1_s"]
                lines.append(f"- {summary['device']}, tolerance {tol}: {arm} / joint = {ratio:.4f}x; "
                             f"same recipe = {s['recipe'] == j['recipe']}; validation solves "
                             f"{s['validation_solved']} versus {j['validation_solved']}.")
    lines += ["", "**Recorded catalog costs**", ""]
    for summary, _, _, _ in runs:
        lines.append(f"- {summary['device']}: {summary['catalog_campaign_s']:.1f} seconds for the catalog loop, "
                     "including warmup and repeated solves; excludes data loading and the preceding operator profiles. "
                     "Per-instance preparation, cold calls and warm calls are retained in factorial.json.")
    lines += ["", "**Limits and provenance**", "",
              "- Training measurements alone determine selections; all validation configurations are published for development diagnosis.",
              "- The v1 protocol incorrectly stated that patient identifiers were unavailable. The separate filename-derived group audit finds six train and six validation groups with zero overlap; it does not modify the inputs or original protocol.",
              "- CPU timings overlap light development activity and a draft PDF build; device allocations are captured in protocol.json. These are workstation diagnostics, not final isolated benchmark runs.",
              "- A tiny difference between training-selected recipes may be measurement noise. No joint-search superiority or LLM contribution is inferred from this catalog.",
              "- Larger independent datasets, matched-budget search runs, competent automatic tuning and final holdout evaluation remain outstanding.",
              "- Generated native kernels are absent from this TV catalog; both mathematical schemes use the existing XLA reference implementation.",
              "", "[Interaction heatmap](interaction_heatmap.pdf) shows the full validation catalog, including incomplete solves.", ""]
    (out / "REPORT.md").write_text("\n".join(lines))
    if args.paper_out:
        paper_out = Path(args.paper_out)
        paper_out.mkdir(parents=True, exist_ok=True)
        tex = [r"% Generated by scripts/summarize_solver_interaction.py from completed runs.",
               r"\begin{table}[t]", r"\centering\small",
               r"\begin{tabular}{llrrrr}", r"\toprule",
               r"device & tolerance & base (s) & impl. (s) & joint (s) & solves \\",
               r"\midrule"]
        for s, _, _, _ in runs:
            device = "CPU" if "cpu" in s["device"].lower() else "RTX " + s["device"].split()[-1]
            for tol, choices in s["tolerances"].items():
                b, p, j = (choices[k] for k in ("fixed", "implementation_only", "joint"))
                exponent = "-4" if float(tol) == 1e-4 else "-6" if float(tol) == 1e-6 else None
                toltex = f"$10^{{{exponent}}}$" if exponent else tol
                counts = f"{b['validation_solved']}/{p['validation_solved']}/{j['validation_solved']}"
                tex.append(f"{device} & {toltex} & {b['validation_penalized_sgm1_s']:.5f} & "
                           f"{p['validation_penalized_sgm1_s']:.5f} & {j['validation_penalized_sgm1_s']:.5f} & {counts} " + r"\\")
        tex += [r"\bottomrule", r"\end{tabular}",
                r"\caption{Larger-patch development diagnostic: six training and six validation OCT images at $256\times256$. Entries are penalized SGM-1 seconds, with a 120-second non-solve penalty and 5,000-iteration cap. Solves lists baseline/implementation-only/joint counts, each out of six. The baseline is compiled banked Condat--V\~u with float64 and $K=64$. Selections use training data only. This is a finite catalog, not a matched-budget search experiment.}",
                r"\label{tab:development-interaction}", r"\end{table}"]
        (paper_out / "development_interaction.tex").write_text("\n".join(tex) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import LogNorm
    tolerances = list(runs[0][3]["tolerances"])
    fig, axes = plt.subplots(len(runs), len(tolerances), squeeze=False,
                             figsize=(14, 3.5 * len(runs)), constrained_layout=True)
    for i, (summary, protocol, _, results) in enumerate(runs):
        algs, policies = list(protocol["algorithms"]), list(protocol["policies"])
        for j, tol in enumerate(tolerances):
            ax = axes[i, j]
            catalog = {(r["algorithm"], r["policy"]): r["validation"] for r in results["tolerances"][tol]["rows"]}
            baseline = catalog[tuple(protocol["baseline"])]["penalized_sgm1_s"]
            vals = np.array([[catalog[a, p]["penalized_sgm1_s"] / baseline for p in policies] for a in algs])
            incomplete = np.array([[catalog[a, p]["n_solved"] != catalog[a, p]["n"] for p in policies] for a in algs])
            cmap = plt.get_cmap("RdYlGn_r").copy(); cmap.set_bad("#dedede")
            ax.imshow(np.ma.array(vals, mask=incomplete), cmap=cmap, norm=LogNorm(.5, 2.), aspect="auto")
            for x, a in enumerate(algs):
                for y, p in enumerate(policies):
                    r = catalog[a, p]
                    text = f"{r['n_solved']}/{r['n']}" if incomplete[x, y] else f"{vals[x,y]:.2f}"
                    ax.text(y, x, text, ha="center", va="center", fontsize=8)
            ax.set_xticks(range(len(policies)), [p.replace("f32_cert64", "f32").replace("schedule", "mixed") for p in policies], rotation=60, ha="right")
            ax.set_yticks(range(len(algs)), [a.replace("condat_vu", "CV").replace("pdhg", "PDHG") for a in algs])
            ax.set_title(f"{summary['device']} | tolerance {tol}")
    fig.suptitle("Development validation catalog: time / banked f64-K64 baseline\nRatios below 1 are faster; gray cells report solves under the iteration cap", fontsize=12)
    fig.savefig(out / "interaction_heatmap.pdf")
    fig.savefig(out / "interaction_heatmap.png", dpi=180)
    plt.close(fig)
    print(out / "REPORT.md")


if __name__ == "__main__":
    main()
