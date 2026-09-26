"""Phase 2 TUNED-PARENT bar: random-search tuning of single-scheme PDHG.

60-iteration random search over single-scheme PDHG (tau, sigma, rho)
ONLY (no wrapper genes, no metric, no mix), on the SAME train set and
the SAME fitness as run_centerpiece.py, for 3 tuning seeds. Output: the
best genome per tuning seed + the overall best, plus the best
textbook-default across all 5 base schemes on the train set (the
structurally inapplicable ones are recorded as gate rejections, not
hidden).

Usage:
  env -u LD_LIBRARY_PATH JAX_PLATFORMS=cpu python scripts/tune_baseline.py \
      --machine LOCAL --out results/sprint/phase2/tuned_baseline.json
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

from atlas.bench.runner import sanitize_json  # noqa: E402
from atlas.genome import (  # noqa: E402
    Genome,
    GenomeParams,
    canonical_json,
    content_hash,
    validate,
)
from phase2_common import (  # noqa: E402
    OP2,
    TRAIN_NODES,
    eval_genome_train,
    eval_textbook_defaults,
    machine_record,
    train_specs,
    worker_init,
)

TUNE_ITERS = 60
TUNING_SEEDS = (101, 102, 103)


def draw_pdhg(rng: np.random.Generator) -> Genome:
    """One random single-scheme PDHG candidate over (tau, sigma, rho).
    Rejection-sampled against the Stage-0 gate by the caller (never
    clamped)."""
    tau = float(np.exp(rng.uniform(np.log(1e-2), np.log(10.0))))
    s_max = 0.98 / (OP2 * tau)
    sigma = float(
        np.exp(rng.uniform(np.log(s_max * 1e-4), np.log(s_max)))
    )
    rho = float(rng.uniform(0.05, 1.95)) if rng.uniform() < 0.5 else None
    return Genome(
        scheme="pdhg", params=GenomeParams(tau=tau, sigma=sigma, rho=rho)
    )


def tune_one_seed(seed: int, iters: int, pool, cache: dict) -> dict:
    rng = np.random.default_rng(seed)
    best_g, best_fit, best_m = None, float("-inf"), None
    n_gate_rejections = 0
    history = []
    for it in range(iters):
        # rejection sampling against the gate (reject, never clamp)
        for _try in range(1000):
            g = draw_pdhg(rng)
            v = validate(g, problem_kind="lp_netflow")
            if v.ok:
                break
            n_gate_rejections += 1
        else:
            raise RuntimeError("no valid PDHG draw in 1000 tries")
        h = content_hash(g)
        if h in cache:
            fit, m = cache[h]
            dedup = True
        else:
            fit, m = eval_genome_train(g, len(TRAIN_NODES), pool=pool)
            cache[h] = (fit, m)
            dedup = False
        history.append(
            {
                "iter": it,
                "genome": canonical_json(g),
                "fitness": fit,
                "dedup": dedup,
                "sgm10_time_s": m.get("sgm10_time_s"),
                "n_failed_instances": m.get("n_failed_instances"),
            }
        )
        if fit > best_fit:
            best_g, best_fit, best_m = g, fit, m
    return {
        "tuning_seed": seed,
        "iters": iters,
        "n_gate_rejections": n_gate_rejections,
        "best_genome": canonical_json(best_g),
        "best_fitness": best_fit,
        "best_metrics": best_m,
        "history": history,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=TUNE_ITERS)
    ap.add_argument(
        "--tuning-seeds",
        default=",".join(str(s) for s in TUNING_SEEDS),
        help="comma-separated tuning seeds (default 101,102,103)",
    )
    ap.add_argument("--machine", choices=("LOCAL", "REMOTE"), default="LOCAL")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    seeds = [int(s) for s in args.tuning_seeds.split(",") if s.strip()]
    t_start = time.time()
    specs = train_specs()
    pool = None
    if args.workers > 1:
        pool = mp.get_context("spawn").Pool(
            processes=args.workers,
            initializer=worker_init,
            initargs=(specs, None),
        )
    else:
        worker_init(specs)

    print(f"[{args.machine}] textbook defaults (all 5 schemes) ...", flush=True)
    textbook = eval_textbook_defaults(pool=pool)
    print(
        f"  best textbook default: {textbook['best_textbook_default']} "
        f"fitness {textbook['best_textbook_fitness']:.6g}",
        flush=True,
    )

    cache: dict = {}  # shared across tuning seeds (dedup only, seeds stay independent draws)
    per_seed = []
    for s in seeds:
        t0 = time.time()
        rec = tune_one_seed(s, args.iters, pool, cache)
        per_seed.append(rec)
        print(
            f"  tuning seed {s}: best fitness {rec['best_fitness']:.6g} "
            f"({time.time() - t0:.0f}s, {rec['n_gate_rejections']} gate rejections)",
            flush=True,
        )

    overall = max(per_seed, key=lambda r: r["best_fitness"])
    out = {
        "experiment": "phase2_tuned_baseline",
        **machine_record(args.machine),
        "config": {
            "search_space": "single-scheme PDHG (tau, sigma, rho) only",
            "iters_per_seed": args.iters,
            "tuning_seeds": seeds,
            "train_nodes": list(TRAIN_NODES),
            "cell_seed": 7,
            "degree": 2,
            "tol_rel_gap": 1e-4,
            "n_repeats_median": 3,
            "gate_opnorm_sq": OP2,
            "workers": args.workers,
        },
        "textbook_defaults_train": textbook,
        "per_seed": per_seed,
        "overall_best": {
            "tuning_seed": overall["tuning_seed"],
            "genome": overall["best_genome"],
            "fitness": overall["best_fitness"],
            "metrics": overall["best_metrics"],
        },
        "n_unique_evaluated": len(cache),
        "total_wall_s": time.time() - t_start,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if pool is not None:
        pool.close()
        pool.join()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(sanitize_json(out), fh, indent=2, allow_nan=False)
        fh.write("\n")
    print(
        f"[{args.machine}] tuned baseline DONE in {out['total_wall_s']:.0f}s "
        f"-> {args.out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
