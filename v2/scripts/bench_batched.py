"""GPU micro-benchmark for the batched vmap backend (Phase-1 deliverable).

Cells:
  lp : batch of 32 same-size netflow LPs (500 nodes, degree 4, ~1000 arcs),
       vanilla PDHG t = s = 0.95 / max ||A||.
  tv : batch of 16 TV 128x128 instances on real Kermany OCT patches,
       PDHG oracle-default params t = 0.5, s = 0.99/4.

Modes (each run in its OWN process so JAX_PLATFORMS is clean):
  seq     : one certified solve per instance through the sequential
            precision-policy path (atlas.schemes.base_schemes.run_built with
            a backend gene), SAME certificate cadence K as the batched path;
            wall = sum of per-instance exec walls (compile excluded).
  batched : atlas.backend.batched_solve on the whole batch; wall =
            batch_wall_s (ONE clock, compile excluded).

Precisions: f64 and f32_cert64 (certificates always f64).

TIMING CONVENTION: every wall excludes AOT compile. Each config is run
twice back to back in the same process; the SECOND (warm-cache) run is the
headline number, both are recorded. The factorial's backend comparison
uses the batched batch_wall vs the sequential summed wall at matched
instance sets; us/iter columns are derived and labeled.

Usage:
  python scripts/bench_batched.py --all --out results/sprint/phase1
      runs every (cell, mode, platform, precision) combo as subprocesses
      (gpu platforms only if CUDA is visible) and merges into
      results/sprint/phase1/batched_bench.json
  python scripts/bench_batched.py --cell lp --mode seq --platform cpu \
      --precision f64 --out results/sprint/phase1   (one combo, one process)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

V2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LP_BATCH = 32
LP_NODES = 500
LP_DEGREE = 4
TV_BATCH = 16
TV_SIZE = 128
TOL = 1e-4
LP_MAX_ITERS = 100_000
TV_MAX_ITERS = 50_000
LP_K = 200
TV_K = 100
SEED = 0
REPEATS = 2

CELLS = ("lp", "tv")
MODES = ("seq", "batched")
PRECISIONS = ("f64", "f32_cert64")


def _build_problems(cell: str):
    if cell == "lp":
        from atlas.bench.lp_cell import from_netflow, to_composite
        from atlas.bench.netflow import gen_regular_flow

        return [
            to_composite(from_netflow(gen_regular_flow(
                n_nodes=LP_NODES, seed=SEED * 10007 + i, degree=LP_DEGREE)))
            for i in range(LP_BATCH)
        ]
    from atlas.bench.tv_cell import build_instances

    return build_instances(TV_BATCH, size=TV_SIZE, lam=0.1, seed=SEED)


def _params_and_budget(cell: str, problems):
    from atlas.schemes.base_schemes import PDHGParams
    from atlas.wrappers.km import Budget

    if cell == "lp":
        nb = max(float(p.L.norm_bound) for p in problems)
        params = PDHGParams(t=0.95 / nb, s=0.95 / nb)
        return params, Budget(max_iters=LP_MAX_ITERS, tol=TOL), LP_K
    params = PDHGParams(t=0.5, s=0.99 / 4.0)
    return params, Budget(max_iters=TV_MAX_ITERS, tol=TOL), TV_K


def _run_seq(problems, params, budget, precision, K) -> dict:
    """Sequential certified solves, one per instance, policy path (same
    cadence K as the batched path); returns summed wall + iters."""
    from types import SimpleNamespace

    from atlas.schemes.base_schemes import build_scheme, run_built

    backend = SimpleNamespace(precision=precision, K=K, theta=None,
                              batch=False)
    walls, iters, gaps, conv, compile_s = [], [], [], [], 0.0
    for p in problems:
        built = build_scheme(p, "pdhg", params, check=True)
        r = run_built(built, built.cert, params, budget, backend=backend)
        walls.append(float(r.wall_time))
        iters.append(int(r.iters))
        g = float(r.rel_gap_history[-1])
        gaps.append(g)
        conv.append(g <= budget.tol)
        compile_s += float(r.extra["compile_time_s"])
    return {
        "wall_s": float(sum(walls)),
        "sum_iters": int(sum(iters)),
        "batch_iters": None,
        "n_converged": int(sum(conv)),
        "max_final_gap": float(max(gaps)),
        "compile_s": float(compile_s),
        "per_instance_walls_s": walls,
    }


def _run_batched(problems, params, budget, precision, K) -> dict:
    from atlas.backend.batched import batched_solve

    r = batched_solve(problems, params, budget, precision_policy=precision,
                      K=K)
    return {
        "wall_s": float(r.batch_wall_s),
        "sum_iters": int(r.iters.sum()),
        "batch_iters": int(r.batch_iters),
        "n_converged": int(r.converged.sum()),
        "max_final_gap": float(r.certified_gap.max()),
        "compile_s": float(r.compile_time_s),
        "per_instance_effective_s": float(r.per_instance_effective_s),
    }


def run_one(cell: str, mode: str, precision: str) -> dict:
    import jax

    problems = _build_problems(cell)
    params, budget, K = _params_and_budget(cell, problems)
    B = len(problems)
    runs = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        r = (_run_seq if mode == "seq" else _run_batched)(
            problems, params, budget, precision, K)
        r["wall_total_incl_compile_s"] = time.perf_counter() - t0
        runs.append(r)
    warm = runs[-1]
    out = {
        "cell": cell,
        "mode": mode,
        "precision": precision,
        "platform": jax.devices()[0].platform,
        "device_kind": jax.devices()[0].device_kind,
        "B": B,
        "K": K,
        "tol": budget.tol,
        "max_iters": budget.max_iters,
        "params": {"t": params.t, "s": params.s},
        "warm": warm,
        "cold": runs[0],
        "iters": warm["sum_iters"],
        "total_wall_s": warm["wall_s"],
    }
    # us/iter, labeled by what an "iteration" is in each mode:
    # seq: one instance advanced one step; batched: us_per_batch_iter is one
    # LOCKSTEP step of all B lanes, us_per_lane_iter_effective divides by B.
    if mode == "seq":
        out["us_per_iter"] = 1e6 * warm["wall_s"] / max(warm["sum_iters"], 1)
    else:
        out["us_per_batch_iter"] = (
            1e6 * warm["wall_s"] / max(warm["batch_iters"], 1))
        out["us_per_lane_iter_effective"] = (
            1e6 * warm["wall_s"] / max(warm["batch_iters"] * B, 1))
    return out


def _part_path(out_dir, cell, mode, platform, precision):
    return os.path.join(
        out_dir, "parts", f"{cell}_{mode}_{platform}_{precision}.json")


def main_child(args):
    os.environ["JAX_PLATFORMS"] = "cuda" if args.platform == "gpu" else "cpu"
    sys.path.insert(0, V2)
    res = run_one(args.cell, args.mode, args.precision)
    path = _part_path(args.out, args.cell, args.mode, args.platform,
                      args.precision)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"[bench_batched] wrote {path}: wall={res['total_wall_s']:.4f}s "
          f"iters={res['iters']} conv={res['warm']['n_converged']}/{res['B']}")


def main_all(args):
    platforms = ["cpu"]
    if args.gpu:
        platforms.append("gpu")
    combos = [
        (cell, mode, plat, prec)
        for cell in CELLS
        for prec in PRECISIONS
        for plat in platforms
        for mode in MODES
    ]
    for cell, mode, plat, prec in combos:
        part = _part_path(args.out, cell, mode, plat, prec)
        if args.resume and os.path.exists(part):
            print(f"[bench_batched] skip existing {part}")
            continue
        cmd = [sys.executable, os.path.abspath(__file__),
               "--cell", cell, "--mode", mode, "--platform", plat,
               "--precision", prec, "--out", args.out]
        print("[bench_batched] RUN", " ".join(cmd), flush=True)
        rc = subprocess.run(cmd, cwd=V2).returncode
        if rc != 0:
            print(f"[bench_batched] FAILED rc={rc}: {cell} {mode} {plat} "
                  f"{prec}", flush=True)
    merge(args.out)


def merge(out_dir):
    import glob
    import platform as _pl

    parts = {}
    for path in sorted(glob.glob(os.path.join(out_dir, "parts", "*.json"))):
        with open(path) as fh:
            r = json.load(fh)
        key = f"{r['cell']}.{r['mode']}_{r['platform']}.{r['precision']}"
        parts[key] = r
    doc = {
        "meta": {
            "host": _pl.node(),
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "lp": {"B": LP_BATCH, "n_nodes": LP_NODES, "degree": LP_DEGREE,
                   "K": LP_K, "max_iters": LP_MAX_ITERS},
            "tv": {"B": TV_BATCH, "size": TV_SIZE, "K": TV_K,
                   "max_iters": TV_MAX_ITERS},
            "tol": TOL,
            "repeats": REPEATS,
            "timing_convention": (
                "walls exclude AOT compile; headline = warm (2nd) run; "
                "seq wall = sum over instances; batched wall = one batch "
                "clock (the factorial's backend-arm number)"),
        },
        "runs": parts,
    }
    out_path = os.path.join(out_dir, "batched_bench.json")
    with open(out_path, "w") as fh:
        json.dump(doc, fh, indent=1)
    print(f"[bench_batched] merged {len(parts)} parts -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--gpu", action="store_true",
                    help="with --all: include the gpu platform combos")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--merge-only", action="store_true")
    ap.add_argument("--cell", choices=CELLS)
    ap.add_argument("--mode", choices=MODES)
    ap.add_argument("--platform", choices=("cpu", "gpu"))
    ap.add_argument("--precision", choices=PRECISIONS)
    ap.add_argument("--out", default=os.path.join(
        V2, "results", "sprint", "phase1"))
    args = ap.parse_args()
    if args.merge_only:
        merge(args.out)
        return
    if args.all:
        main_all(args)
        return
    if not all([args.cell, args.mode, args.platform, args.precision]):
        ap.error("either --all or all of --cell/--mode/--platform/--precision")
    main_child(args)


if __name__ == "__main__":
    main()
