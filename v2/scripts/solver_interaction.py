"""Development-only TV solver/precision/cadence factorial with strong controls.

New output directories are mandatory. No LP data or final test partition is
loaded. The finite catalog is measured exhaustively, not presented as a
budget-matched evolutionary comparison. Set JAX_PLATFORMS before invocation.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import numpy as np
from atlas.backend.hardware import operator_profile, prepare_base
from atlas.bench.factorial import select_catalog, check_group_split
from atlas.bench.kermany import load_patches
from atlas.genome import Backend, Genome, GenomeParams, genome_to_dict
from atlas.ir.spec import tv_denoise_problem
from hardware_campaign import Evaluator, source_hashes, write_json


def algorithms():
    return {f"{s}_{label}": Genome(s, GenomeParams(tau=t, sigma=u, rho=r))
            for s in ("condat_vu", "pdhg")
            for label, t, u, r in (("banked", .05, 1., 1.9),
                                   ("balanced", .25, .25, 1.5))}


def policies(kernels=()):
    menu = {f"{p}_K{k}": Backend(precision=p, K=k, theta=.01)
            for p in ("f64", "f32_cert64", "schedule") for k in (16, 64, 256)}
    from atlas.backend.cuda_tv import VARIANTS
    for kernel in kernels:
        if kernel == "jax":
            continue
        if kernel not in VARIANTS:
            raise ValueError(f"unsupported TV kernel: {kernel}")
        menu.update({f"{kernel}_K{k}": Backend(K=k, kernel=kernel) for k in (16,64,256)})
    return menu


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--n-train", type=int, default=6)
    ap.add_argument("--n-validation", type=int, default=6)
    ap.add_argument("--max-iters", type=int, default=5000)
    ap.add_argument("--tolerances", default="0.0001,0.000001")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--penalty-s", type=float, default=120.)
    ap.add_argument("--data-seed", type=int, default=20260907)
    ap.add_argument("--order-seed", type=int, default=718)
    ap.add_argument("--synthetic", action="store_true", help="smoke only")
    ap.add_argument("--kernels", default="jax", help="comma-separated native TV variants to add to XLA policies")
    ap.add_argument("--algorithm-filter", help="comma-separated algorithm catalog names")
    ap.add_argument("--policy-filter", help="comma-separated policy catalog names; must retain f64_K64")
    args = ap.parse_args()
    if min(args.size, args.n_train, args.n_validation, args.max_iters, args.repeats) < 1:
        raise ValueError("sizes, counts and iteration limit must be positive")
    tolerances = [float(t) for t in args.tolerances.split(",")]
    if not all(np.isfinite(t) and t > 0 for t in tolerances):
        raise ValueError("tolerances must be positive finite values")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    algs, backends = algorithms(), policies(args.kernels.split(","))
    if args.algorithm_filter:
        algs = {a: algs[a] for a in args.algorithm_filter.split(",")}
    if args.policy_filter:
        backends = {b: backends[b] for b in args.policy_filter.split(",")}
    if "condat_vu_banked" not in algs or "f64_K64" not in backends:
        raise ValueError("the fixed baseline must remain in the catalog")
    for label in ("banked", "balanced"):
        if (f"pdhg_{label}" in algs) != (f"condat_vu_{label}" in algs):
            raise ValueError("retain both schemes for each selected parameter setting")
    menu = [(a, b) for a in algs for b in backends]
    order = np.random.default_rng(args.order_seed).permutation(len(menu)).tolist()
    hashes = source_hashes()
    hashes[str(Path(__file__).relative_to(ROOT))] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    try:
        gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu",
                                       "--format=csv"], text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        gpu = None
    protocol = {
        "version": "solver-interaction-development-v3", "args": vars(args),
        "host": platform.node(), "device": jax.devices()[0].device_kind,
        "jax": jax.__version__, "source_sha256": hashes, "gpu_snapshot": gpu,
        "status": "development only; exhaustive finite catalog, not search efficiency",
        "data": "Kermany train split; image sampling; filename-derived patient groups checked disjoint before timing",
        "baseline": ["condat_vu_banked", "f64_K64"],
        "algorithms": {a: genome_to_dict(g) for a, g in algs.items()},
        "policies": {b: genome_to_dict(replace(next(iter(algs.values())), backend=p))["backend"]
                     for b, p in backends.items()},
        "order": [menu[i] for i in order], "tolerances": tolerances,
        "timing": "three warm synchronized solver timings after one complete warmup; original-coordinate f64 checks included; independent NumPy check outside timer",
        "censoring": "exact iteration cap; fixed 120-second penalty by default, not a wall-time deadline",
        "selection": "maximize train solves, then minimize train SGM-1, then lexical tie break; validation excluded",
        "inference": "repeated timings are not independent data; this pilot does not establish population generalization",
    }
    if args.synthetic:
        protocol["data"] = "synthetic smoke data only"
    # Commit the complete candidate and measurement contract before loading data.
    write_json(out / "protocol.json", protocol)
    n = args.n_train + args.n_validation
    if args.synthetic:
        patches = np.random.default_rng(args.data_seed).random((n, args.size, args.size))
        meta = [{"index": i, "synthetic": True} for i in range(n)]
    else:
        patches, meta = load_patches(n, size=args.size, split="train", seed=args.data_seed, return_meta=True)
    problems = [tv_denoise_problem(y, .1) for y in patches]
    for p, m in zip(problems, meta):
        m["input_sha256"] = hashlib.sha256(np.asarray(p.f.y).tobytes()).hexdigest()
        m["source_partition"] = "synthetic_smoke" if args.synthetic else "dataset_train_development_only"
    train = list(range(args.n_train))
    validation = list(range(args.n_train, n))
    group_audit = None if args.synthetic else check_group_split(meta, train, validation)
    write_json(out / "data.json", {"instances": meta, "train": train, "validation": validation,
                                   "group_audit": group_audit, "final_holdout_loaded": False})
    # Structural eligibility and actual numerical-map difference are checked
    # before timing. PDHG uses the fidelity prox; CV uses its forward gradient.
    built = {a: prepare_base(problems[0], g)[0] for a, g in algs.items()}
    differences = {}
    for label in ("banked", "balanced"):
        if f"pdhg_{label}" not in built:
            continue
        left, right = built[f"pdhg_{label}"], built[f"condat_vu_{label}"]
        state = jax.tree_util.tree_map(lambda x: x + .1, left.state0)
        x = left.T(state, 0, left.dyn)
        y = right.T(state, 0, right.dyn)
        diff = max(float(np.max(np.abs(np.asarray(v) - np.asarray(w))))
                   for v, w in zip(jax.tree_util.tree_leaves(x), jax.tree_util.tree_leaves(y)))
        if diff <= 1e-12:
            raise AssertionError("the supposedly distinct solver maps coincide")
        differences[label] = diff
    write_json(out / "map_difference.json", differences)
    print("MAP_DIFFERENCE", differences, flush=True)
    profiles = {a: operator_profile(problems[0], g, hlo_dir=out / "hlo" / a)
                for a, g in algs.items()}
    write_json(out / "profiles.json", profiles)
    evaluator = Evaluator(args, problems)
    result = {"status": "running", "tolerances": {}}
    start = time.perf_counter()
    for tol in tolerances:
        rows = []
        result["tolerances"][str(tol)] = {"rows": rows}
        for idx in order:
            a, b = menu[idx]
            genome = replace(algs[a], backend=backends[b])
            measured = evaluator.evaluate(genome, train, tol)
            rows.append({"algorithm": a, "policy": b, "genome": genome_to_dict(genome), "train": measured})
            print(f"TRAIN tol={tol} {a}/{b} solved={measured['n_solved']}/{len(train)} time={measured['penalized_sgm1_s']:.6f}", flush=True)
            write_json(out / "factorial.json", result)
        choices = select_catalog(rows, "condat_vu_banked", "f64_K64")
        result["tolerances"][str(tol)]["selections"] = choices
        # Selection is fixed before any validation evaluation at this tolerance.
        write_json(out / "factorial.json", result)
        by_pair = {(r["algorithm"], r["policy"]): r for r in rows}
        for idx in reversed(order):
            a, b = menu[idx]
            genome = replace(algs[a], backend=backends[b])
            measured = evaluator.evaluate(genome, validation, tol)
            by_pair[a, b]["validation"] = measured
            print(f"VALIDATION tol={tol} {a}/{b} solved={measured['n_solved']}/{len(validation)} time={measured['penalized_sgm1_s']:.6f}", flush=True)
            write_json(out / "factorial.json", result)
    result.update(status="complete", campaign_s=time.perf_counter() - start)
    write_json(out / "factorial.json", result)
    print("COMPLETE", result["campaign_s"], flush=True)


if __name__ == "__main__":
    main()
