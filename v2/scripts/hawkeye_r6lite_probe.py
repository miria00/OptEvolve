"""Post-hoc controlled precision probe for Hawkeye R6-lite.

Two questions the evolution runs left open, answered with matched pairs:
(A) On each machine's holdout instances, what does each execution policy
    cost for the SAME champion iteration (absent km path vs policy path
    f64 / f32_cert64 / schedule)? Separates the policy-path overhead from
    any f32 arithmetic gain at 128x128.
(B) The near-miss fbs genome (f32_cert64, gap floor 9.35e-4 at the 10k
    cap on train inst5 of BOTH machines): does its f64 twin certify under
    a 4x larger cap while f32 plateaus? That is the accumulation-stall
    verdict.

Usage: python scripts/hawkeye_r6lite_probe.py --machine LOCAL|REMOTE
Writes results/sprint/hawkeye_r6lite_probe_{machine}.json
"""
from __future__ import annotations

import os

os.environ["JAX_PLATFORMS"] = "cuda"
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import argparse
import json
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import jax  # noqa: E402

from atlas.bench.kermany import load_patches  # noqa: E402
from atlas.bench.metrics import sgm  # noqa: E402
from atlas.bench.runner import sanitize_json  # noqa: E402
from atlas.genome import genome_from_dict, canonical_json  # noqa: E402
from atlas.ir.spec import tv_denoise_problem  # noqa: E402
from atlas.schemes.compile import solve_genome  # noqa: E402
from atlas.wrappers.km import Budget  # noqa: E402

V2_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPRINT = os.path.join(V2_ROOT, "results", "sprint")

CHAMPIONS = {
    "LOCAL": {"scheme": "pdhg",
              "params": {"tau": 0.0914301979418, "sigma": 0.549199698856,
                         "alpha": None, "rho": 1.56194351561},
              "wrapper": {"halpern": False, "restart": "none",
                          "restart_k": None},
              "mix": None, "backend": None},
    "REMOTE": {"scheme": "pdhg",
               "params": {"tau": 0.0233575586243, "sigma": 4.20958242511,
                          "alpha": None, "rho": None},
               "wrapper": {"halpern": False, "restart": "none",
                           "restart_k": None},
               "mix": None, "backend": None},
}

NEAR_MISS_FBS = {
    "scheme": "fbs",
    "params": {"tau": 0.0061220196945, "sigma": None, "alpha": None,
               "rho": 1.66951443508},
    "wrapper": {"halpern": False, "restart": "none", "restart_k": None},
    "mix": None, "backend": None,
}

POLICIES = [
    None,
    {"precision": "f64", "K": 50, "theta": None, "batch": False},
    {"precision": "f32_cert64", "K": 50, "theta": None, "batch": False},
    {"precision": "schedule", "K": 50, "theta": 1e-2, "batch": False},
]


def timed(problem, g, budget):
    r1 = solve_genome(problem, g, budget)
    r2 = solve_genome(problem, g, budget)
    gaps = np.asarray(r2.rel_gap_history, float)
    be = (r2.extra or {}).get("backend") if r2.extra else None
    return {
        "iters": int(r2.iters),
        "final_rel_gap": float(gaps[-1]),
        "gap_floor": float(np.nanmin(gaps)),
        "wall_cold_s": float(r1.wall_time),
        "wall_warm_s": float(r2.wall_time),
        "certified": bool(gaps[-1] <= budget.tol),
        "certificate_dtype": be["certificate_dtype"] if be else "float64",
        "switch_iter": be["switch_iter"] if be else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--machine", required=True, choices=["LOCAL", "REMOTE"])
    args = ap.parse_args()
    dev = jax.devices()[0]
    assert dev.platform != "cpu"
    print(f"[probe] {dev.device_kind} jax={jax.__version__}", flush=True)

    patches = load_patches(9, size=128, split="test", seed=7)
    problems = [tv_denoise_problem(p, lam=0.1) for p in patches]
    train, holdout = problems[:6], problems[6:]
    tol = 1e-4

    out = {"machine": args.machine, "device": dev.device_kind,
           "jax_version": jax.__version__, "tol": tol}

    # (A) champion iteration under each policy, holdout
    base = CHAMPIONS[args.machine]
    budget = Budget(max_iters=10_000, tol=tol, sample_every=50)
    rows = []
    for pol in POLICIES:
        g = genome_from_dict({**base, "backend": pol})
        recs = [timed(pr, g, budget) for pr in holdout]
        times = [r["wall_warm_s"] if r["certified"] else float("inf")
                 for r in recs]
        name = "absent" if pol is None else pol["precision"]
        rows.append({"policy": name, "policy_full": pol,
                     "sgm10_warm_s": sgm(times), "per_instance": recs})
        print(f"  [A] {name:12s} sgm10={sgm(times):.4g}", flush=True)
    out["champion_policy_sweep_holdout"] = {
        "champion": canonical_json(genome_from_dict(base)), "rows": rows}

    # (B) near-miss fbs genome: f32_cert64 vs f64 twin, train inst5, 40k cap
    budget_b = Budget(max_iters=40_000, tol=tol, sample_every=50)
    b_rows = {}
    for prec in ("f32_cert64", "f64"):
        g = genome_from_dict({**NEAR_MISS_FBS,
                              "backend": {"precision": prec, "K": 25,
                                          "theta": None, "batch": False}})
        rec = timed(train[5], g, budget_b)
        b_rows[prec] = rec
        print(f"  [B] {prec:11s} certified={rec['certified']} "
              f"floor={rec['gap_floor']:.3e} iters={rec['iters']}", flush=True)
    out["near_miss_fbs_inst5_cap40k"] = {
        "genome_base": canonical_json(genome_from_dict(NEAR_MISS_FBS)),
        "rows": b_rows,
    }

    dest = os.path.join(SPRINT, f"hawkeye_r6lite_probe_{args.machine}.json")
    with open(dest, "w") as fh:
        json.dump(sanitize_json(out), fh, indent=2, allow_nan=False)
    print("[probe] wrote", dest, flush=True)


if __name__ == "__main__":
    main()
