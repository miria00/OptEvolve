"""Nsight capture of named solver/SpMV/certificate phases, after compilation.

Run with nsys profile --trace=cuda,nvtx --sample=none
--capture-range=cudaProfilerApi --capture-range-end=stop. NVTX ranges label
actual GPU kernels; optimized HLO counts are never called launch counts.
"""
from __future__ import annotations
import argparse
import ctypes
import ctypes.util
from pathlib import Path
import time

from hardware_campaign import (ROOT, algorithms, load_data, prepare_base,
                               write_json, replace, Backend, Budget, run_built)
import jax
import jax.numpy as jnp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--cell", choices=("lp", "oct", "toy_tv"), default="lp")
    ap.add_argument("--lp-names", default="trento1")
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--kernel", default="jax")
    args = ap.parse_args()
    args.n_train, args.n_holdout, args.data_seed = 1, 0, 20260906
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    problems, metadata = load_data(args)
    p = problems[0]
    g = next(iter(algorithms(args.cell).values()))
    g = replace(g, backend=Backend(K=50, kernel=args.kernel))
    built, cert, params, _ = prepare_base(p, g)
    state = jax.tree_util.tree_map(jnp.asarray, built.state0)
    dyn = jax.tree_util.tree_map(jnp.asarray, built.dyn)
    def steps(s, d):
        return jax.lax.fori_loop(0, args.steps, lambda k, z: built.T(z, k, d), s)
    exe_step = jax.jit(steps).lower(state, dyn).compile()
    state = exe_step(state, dyn)
    exe_cert = jax.jit(built.metric).lower(state, dyn).compile()
    x = jnp.ones(p.shape, jnp.float64)
    u = p.L.forward(x)
    exe_fwd = jax.jit(p.L.forward).lower(x).compile()
    exe_adj = jax.jit(p.L.adjoint).lower(u).compile()
    functions = {
        "base_step_block": lambda: exe_step(state, dyn),
        "certificate_only": lambda: exe_cert(state, dyn),
        "original_spmv": lambda: exe_fwd(x),
        "original_adjoint": lambda: exe_adj(u),
    }
    budget = Budget(max_iters=args.steps, tol=0.)
    for k in (1, 50, 200):
        backend = Backend(K=k, kernel=args.kernel)
        def solve(b=backend):
            return run_built(built, cert, params, budget, backend=b).x
        functions[f"solver_K{k}"] = solve
    for fn in functions.values():
        jax.block_until_ready(fn())
    runtime = ctypes.CDLL(ctypes.util.find_library("cudart"))
    nvtx = ctypes.CDLL(ctypes.util.find_library("nvToolsExt"))
    nvtx.nvtxRangePushA.argtypes = [ctypes.c_char_p]
    samples = {}
    status = runtime.cudaProfilerStart()
    if status:
        raise RuntimeError(f"cudaProfilerStart failed: {status}")
    try:
        for name, fn in functions.items():
            nvtx.nvtxRangePushA(name.encode())
            times = []
            for _ in range(args.repeats):
                t0 = time.perf_counter()
                result = fn()
                jax.block_until_ready(result)
                times.append(time.perf_counter() - t0)
            samples[name] = times
            nvtx.nvtxRangePop()
    finally:
        runtime.cudaProfilerStop()
    write_json(out / "capture.json", {
        "device": jax.devices()[0].device_kind, "jax": jax.__version__,
        "args": vars(args), "data": metadata, "samples_s": samples,
        "note": "instrumented timings; use hardware_campaign for speedup claims"})


if __name__ == "__main__":
    main()
