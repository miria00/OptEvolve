"""Phase 2 two-distance holdout evaluation (REPORTED SEPARATELY, never
blended into one number):

  (i)  WITHIN-distribution: 10 fresh netflow instances (cell seed 7007,
       degree-2 regular, sizes matched to the train range).
  (ii) OOD: Mittelmann + MIPLIB small tier, a deterministic stratified
       (by nnz) subsample of 48 instances, PLUS the full 19-instance
       medium tier flagged separately (tier "ood_medium").

Per (config, instance, tolerance): iterations-to-certificate, wall
seconds, certificate status. Tolerances 1e-4 AND 1e-6 with runtime caps
60 s / 120 s (cap hits logged as failures; caps and truncations logged;
Level-1 typecheck failures on an instance recorded as
"typecheck_rejected", the honest partial-OOD-transfer outcome).
Aggregates: SGM-10 (PAR-1 at the cap) + solve rate per config per tier
per tolerance. Process-parallel across (config, instance, tol) tasks.

Designed for reuse by the ablation phase: --genomes accepts arbitrary
genome JSON lists in any of these shapes:
  [{"name": ..., "genome": {...}}, ...]      (list of named genomes)
  {"name1": {...genome...}, "name2": {...}}  (dict name -> genome)
  a run_centerpiece.py output JSON            (its final_champion is used)
--tuned takes a tune_baseline.py output (its overall_best is added as
"tuned_pdhg"); vanilla PDHG is always added unless --no-vanilla.

Usage:
  env -u LD_LIBRARY_PATH JAX_PLATFORMS=cpu python scripts/eval_holdout.py \
      --genomes results/sprint/phase2/centerpiece_llm_seed42.json \
      --tuned results/sprint/phase2/tuned_baseline.json \
      --machine LOCAL --workers 20 \
      --out results/sprint/phase2/holdout_eval.json
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

from atlas.bench.mps_cell import DATA_ROOT, MANIFEST_PATH, load_manifest  # noqa: E402
from atlas.bench.runner import sanitize_json  # noqa: E402
from atlas.genome import canonical_json, genome_from_dict  # noqa: E402
from phase2_common import (  # noqa: E402
    HOLDOUT_CAPS_S,
    TEXTBOOK_PER_INSTANCE,
    aggregate_holdout,
    holdout_task,
    machine_record,
    vanilla_pdhg,
    within_holdout_specs,
    worker_init,
)

DEFAULT_OOD_SEED = 20260903
DEFAULT_OOD_SMALL_N = 48


# ---------------------------------------------------------------------------
# Config (genome list) loading
# ---------------------------------------------------------------------------


def _name_from_path(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def load_genome_configs(paths: list[str]) -> list[tuple[str, str]]:
    """[(name, canonical_genome_json)] from arbitrary genome JSON files."""
    configs: list[tuple[str, str]] = []

    def add(name: str, gd) -> None:
        if isinstance(gd, str):
            gd = json.loads(gd)
        g = genome_from_dict(gd)
        configs.append((name, canonical_json(g)))

    for path in paths:
        with open(path) as fh:
            doc = json.load(fh)
        if isinstance(doc, list):
            for i, entry in enumerate(doc):
                add(entry.get("name", f"{_name_from_path(path)}_{i}"), entry["genome"])
        elif isinstance(doc, dict) and "final_champion" in doc:
            add(f"champion[{_name_from_path(path)}]", doc["final_champion"])
        elif isinstance(doc, dict) and "genome" in doc:
            add(doc.get("name", _name_from_path(path)), doc["genome"])
        elif isinstance(doc, dict):
            for name, gd in doc.items():
                add(name, gd)
        else:
            raise ValueError(f"unrecognized genome file shape: {path}")
    return configs


# ---------------------------------------------------------------------------
# OOD instance selection
# ---------------------------------------------------------------------------


def stratified_small_subsample(
    entries: list[dict], n: int, seed: int
) -> list[dict]:
    """Deterministic stratified-by-nnz subsample: sort by (nnz, name),
    split into n contiguous strata, draw one entry per stratum with
    rng(seed)."""
    pool = sorted(entries, key=lambda e: (e["nnz"], e["name"]))
    if n >= len(pool):
        return pool
    rng = np.random.default_rng(seed)
    picked = []
    for stratum in np.array_split(np.arange(len(pool)), n):
        picked.append(pool[int(rng.choice(stratum))])
    return picked


def ood_specs(args) -> tuple[list[dict], dict]:
    entries = [e for e in load_manifest(args.manifest) if e.get("usable")]
    small = [e for e in entries if e.get("holdout_tier") == "small"]
    medium = [e for e in entries if e.get("holdout_tier") == "medium"]
    if args.no_small:
        chosen_small = []
        selection = "excluded (--no-small)"
    elif args.ood_smallest:
        chosen_small = sorted(small, key=lambda e: (e["nnz"], e["name"]))[
            : args.ood_small_n
        ]
        selection = "smallest-by-nnz (smoke mode)"
    else:
        chosen_small = stratified_small_subsample(
            small, args.ood_small_n, args.ood_seed
        )
        selection = "stratified-by-nnz, one draw per stratum"
    specs = [
        {
            "kind": "mps",
            "path": os.path.join(args.data_root, e["file"]),
            "name": e["name"],
            "tier": "ood_small",
            "nnz": e["nnz"],
        }
        for e in chosen_small
    ]
    if not args.no_medium:
        specs += [
            {
                "kind": "mps",
                "path": os.path.join(args.data_root, e["file"]),
                "name": e["name"],
                "tier": "ood_medium",
                "nnz": e["nnz"],
            }
            for e in sorted(medium, key=lambda e: (e["nnz"], e["name"]))
        ]
    sel_rec = {
        "small_tier_total": len(small),
        "small_selected": [e["name"] for e in chosen_small],
        "small_selection": selection,
        "ood_seed": args.ood_seed,
        "medium_included": not args.no_medium,
        "medium_selected": [] if args.no_medium else [
            e["name"] for e in sorted(medium, key=lambda e: (e["nnz"], e["name"]))
        ],
    }
    return specs, sel_rec


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--genomes",
        action="append",
        default=[],
        help="genome JSON file (repeatable); see module docstring for shapes",
    )
    ap.add_argument(
        "--tuned",
        default=None,
        help="tune_baseline.py output JSON; adds its overall_best as 'tuned_pdhg'",
    )
    ap.add_argument("--no-vanilla", action="store_true")
    ap.add_argument(
        "--no-textbook-per-instance",
        action="store_true",
        help="drop the per-instance textbook-rule baseline "
        "(tau=sigma=0.9/||A||_instance)",
    )
    ap.add_argument("--within-n", type=int, default=10)
    ap.add_argument("--ood-small-n", type=int, default=DEFAULT_OOD_SMALL_N)
    ap.add_argument("--ood-seed", type=int, default=DEFAULT_OOD_SEED)
    ap.add_argument(
        "--ood-smallest",
        action="store_true",
        help="take the N smallest-by-nnz small-tier instances (smoke mode) "
        "instead of the stratified subsample",
    )
    ap.add_argument("--no-medium", action="store_true")
    ap.add_argument(
        "--no-small",
        action="store_true",
        help="drop the ood_small tier entirely (medium-only OOD run)",
    )
    ap.add_argument("--no-ood", action="store_true")
    ap.add_argument("--tols", default="1e-4,1e-6")
    ap.add_argument("--manifest", default=MANIFEST_PATH)
    ap.add_argument("--data-root", default=DATA_ROOT)
    ap.add_argument("--machine", choices=("LOCAL", "REMOTE"), default="LOCAL")
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    t_start = time.time()
    tols = [float(t) for t in args.tols.split(",") if t.strip()]
    for t in tols:
        if t not in HOLDOUT_CAPS_S:
            raise SystemExit(
                f"tol {t:g} has no registered runtime cap "
                f"(known: {sorted(HOLDOUT_CAPS_S)})"
            )

    configs = load_genome_configs(args.genomes)
    if args.tuned:
        with open(args.tuned) as fh:
            tuned_doc = json.load(fh)
        configs.append(
            ("tuned_pdhg", canonical_json(
                genome_from_dict(json.loads(tuned_doc["overall_best"]["genome"]))
            ))
        )
    if not args.no_vanilla:
        configs.append(("vanilla_pdhg", canonical_json(vanilla_pdhg())))
    if not args.no_textbook_per_instance:
        # The textbook step RULE (tau = sigma = 0.9/||A||_instance): the
        # runnable identical-handicap baseline on OOD instances whose
        # operator norm differs from the train cell's (absolute-stepsize
        # genomes are typecheck-rejected there, which is logged, not hidden).
        configs.append(("textbook_pdhg_per_instance", TEXTBOOK_PER_INSTANCE))
    if not configs:
        raise SystemExit("no genome configs given (use --genomes/--tuned)")
    names = [n for n, _ in configs]
    if len(set(names)) != len(names):
        raise SystemExit(f"duplicate config names: {names}")

    specs = within_holdout_specs()[: args.within_n]
    sel_rec = None
    if not args.no_ood:
        mps, sel_rec = ood_specs(args)
        specs = specs + mps

    tasks = []
    for name, gj in configs:
        for idx in range(len(specs)):
            for tol in tols:
                tasks.append((name, gj, idx, tol, HOLDOUT_CAPS_S[tol]))
    print(
        f"[{args.machine}] {len(configs)} configs x {len(specs)} instances "
        f"x {len(tols)} tols = {len(tasks)} tasks, {args.workers} workers",
        flush=True,
    )

    rows = []
    if args.workers > 1:
        with mp.get_context("spawn").Pool(
            processes=args.workers,
            initializer=worker_init,
            initargs=(specs, None),
        ) as pool:
            for i, row in enumerate(pool.imap_unordered(holdout_task, tasks, 1)):
                rows.append(row)
                print(
                    f"  [{i + 1}/{len(tasks)}] {row['config']} "
                    f"{row['instance']} tol={row['tol']:g}: {row['status']}"
                    + (
                        f" it={row['iters_to_cert']} t={row['time_to_cert_s']:.3g}s"
                        if row["status"] == "converged"
                        else ""
                    ),
                    flush=True,
                )
    else:
        worker_init(specs)
        for i, task in enumerate(tasks):
            row = holdout_task(task)
            rows.append(row)
            print(
                f"  [{i + 1}/{len(tasks)}] {row['config']} {row['instance']} "
                f"tol={row['tol']:g}: {row['status']}",
                flush=True,
            )

    rows.sort(key=lambda r: (r["config"], r["tier"], r["tol"], r["instance"]))
    aggregates = aggregate_holdout(rows)

    out = {
        "experiment": "phase2_holdout_eval",
        **machine_record(args.machine),
        "config": {
            "configs": {
                n: (gj if gj == TEXTBOOK_PER_INSTANCE else json.loads(gj))
                for n, gj in configs
            },
            "tols": tols,
            "runtime_caps_s": {f"{k:g}": v for k, v in HOLDOUT_CAPS_S.items()},
            "within_holdout_cell_seed": 7007,
            "within_instances": [s["name"] for s in specs if s["kind"] == "netflow"],
            "ood_selection": sel_rec,
            "workers": args.workers,
            "note": (
                "two-distance holdout: tier netflow_within (within-"
                "distribution) and tiers ood_small / ood_medium (OOD "
                "Mittelmann+MIPLIB) are REPORTED SEPARATELY and must never "
                "be blended into one number"
            ),
        },
        "aggregates": aggregates,
        "rows": rows,
        "total_wall_s": time.time() - t_start,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(sanitize_json(out), fh, indent=2, allow_nan=False)
        fh.write("\n")
    print(
        f"[{args.machine}] holdout eval DONE in {out['total_wall_s']:.0f}s "
        f"-> {args.out}",
        flush=True,
    )
    for key, agg in aggregates.items():
        print(
            f"  {key}: solve_rate {agg['solve_rate']:.2f} "
            f"sgm10 {agg['sgm10_time_s_par1_at_cap']:.4g}s "
            f"(certified {agg['n_certified']}/{agg['n_instances']})",
            flush=True,
        )


if __name__ == "__main__":
    main()
