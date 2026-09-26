"""Trace TV update, relaxation and certificate kernels on frozen development data.

Nsight timings diagnose costs; use the uninstrumented comparison for speedups.
The input is the first training image of a previously recorded development run.
"""
from __future__ import annotations

import argparse
import ctypes
import ctypes.util
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time

from hardware_campaign import (ROOT, prepare_base, write_json, source_hashes,
                               Backend, Budget, run_built)
from atlas.bench.kermany import load_patches
from atlas.genome import Genome, GenomeParams, genome_to_dict
from atlas.ir.spec import tv_denoise_problem
import jax
import jax.numpy as jnp
import numpy as np


def development_input(run):
    run = Path(run)
    protocol = json.loads((run / "protocol.json").read_text())
    data = json.loads((run / "data.json").read_text())
    a = protocol["args"]
    patches, meta = load_patches(a["n_train"] + a["n_validation"],
        size=a["size"], split="train", seed=a["data_seed"], return_meta=True)
    i = data["train"][0]
    digest = hashlib.sha256(patches[i].tobytes()).hexdigest()
    if digest != data["instances"][i]["input_sha256"]:
        raise ValueError("frozen development input hash mismatch")
    return tv_denoise_problem(patches[i], .1), {**meta[i], "input_sha256": digest,
                                               "source": str(run), "fold": "train"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--development-run", default=str(ROOT / "results/attack_plan_20260906/tv_4090_v1"))
    ap.add_argument("--kernels", default="jax")
    ap.add_argument("--schemes", default="pdhg,condat_vu")
    ap.add_argument("--steps", type=int, default=256)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--nsys", action="store_true")
    args = ap.parse_args()
    if min(args.steps, args.repeats) < 1:
        raise ValueError("positive steps and repeats required")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    p, meta = development_input(args.development_run)
    functions, preparation = {}, {}
    for scheme in args.schemes.split(","):
        for kernel in args.kernels.split(","):
            g = Genome(scheme, GenomeParams(tau=.05, sigma=1., rho=1.9),
                       backend=Backend(K=64, kernel=kernel))
            start = time.perf_counter()
            built, cert, params, _ = prepare_base(p, g)
            state = jax.tree_util.tree_map(jnp.asarray, built.state0)
            dyn = jax.tree_util.tree_map(jnp.asarray, built.dyn)
            label = f"{scheme}/{kernel}"
            modes = ("base", "relaxed", "relaxed_unfused") if built.relaxed_T is not None else ("base", "relaxed")
            for mode in modes:
                def block(s, d, b=built, mode_=mode):
                    def step(k, z):
                        if mode_ == "relaxed" and b.relaxed_T is not None:
                            return b.relaxed_T(z, k, d, 1.9)
                        tz = b.T(z, k, d)
                        return jax.tree_util.tree_map(lambda old, new: (1.-1.9)*old + 1.9*new, z, tz) if mode_ != "base" else tz
                    return jax.lax.fori_loop(0, args.steps, step, s)
                exe = jax.jit(block).lower(state, dyn).compile()
                name = label + f"/{mode}_block"
                (out / (name.replace("/", "_") + ".hlo.txt")).write_text(exe.as_text())
                functions[name] = lambda e=exe, s=state, d=dyn: e(s, d)
            for cadence in (64, 256):
                backend = replace(g.backend, K=cadence)
                def solve(b=built, c=cert, params_=params, backend_=backend):
                    return run_built(b, c, params_, Budget(max_iters=args.steps, tol=0.), backend=backend_).x
                functions[f"{label}/solver_K{cadence}"] = solve
            if kernel == "jax":
                metric = jax.jit(built.metric).lower(state, dyn).compile()
                functions[f"{label}/certificate"] = lambda e=metric, s=state, d=dyn: e(s, d)
            preparation[label] = {"genome": genome_to_dict(g), "prepare_compile_s": time.perf_counter()-start,
                                  "kernel_evidence": getattr(built, "kernel_evidence", None)}
    for fn in functions.values():
        for _ in range(3):
            jax.block_until_ready(fn())
    runtime = nvtx = None
    if args.nsys:
        runtime = ctypes.CDLL(ctypes.util.find_library("cudart"))
        nvtx = ctypes.CDLL(ctypes.util.find_library("nvToolsExt"))
        nvtx.nvtxRangePushA.argtypes = [ctypes.c_char_p]
        if runtime.cudaProfilerStart():
            raise RuntimeError("cudaProfilerStart failed")
    samples = {}
    try:
        for name, fn in functions.items():
            if nvtx:
                nvtx.nvtxRangePushA(name.encode())
            times = []
            for _ in range(args.repeats):
                t0 = time.perf_counter()
                jax.block_until_ready(fn())
                times.append(time.perf_counter()-t0)
            samples[name] = times
            if nvtx:
                nvtx.nvtxRangePop()
    finally:
        if runtime:
            runtime.cudaProfilerStop()
    write_json(out / "capture.json", {
        "args": vars(args), "device": jax.devices()[0].device_kind,
        "jax": jax.__version__, "data": meta, "source_sha256": source_hashes(),
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "preparation": preparation, "samples_s": samples,
        "note": "instrumented diagnostic" if args.nsys else "warm synchronized microbenchmark"})
    print(out / "capture.json", flush=True)


if __name__ == "__main__":
    main()
