"""Measured hardware context for certified solver search.

Hawkeye-inspired executable examples pair an operation with measured cost.
These are calibration measurements, never predictions of guaranteed speedup.
HLO instruction counts describe compiler IR, not measured GPU kernel launches.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import time

import jax
import jax.numpy as jnp
import numpy as np

from atlas.genome import Genome, content_hash
from atlas.schemes.compile import _prepare_with_metric, genome_solver_params
from atlas.schemes.base_schemes import build_scheme, _wrap_cert


STRATEGIES = [
    {"name": "certificate_cadence", "gene": "backend.K",
     "choices": [1, 10, 50, 200],
     "evidence": "check cost, check count, detection delay, solve time",
     "constraint": "final original-coordinate f64 check; exact iteration cap",
     "loses_when": "delayed detection costs more than avoided checks"},
    {"name": "precision", "gene": "backend.precision",
     "choices": ["f64", "f32_cert64", "schedule"],
     "evidence": "measured step cost, gap floor, certified solve time",
     "constraint": "f64 original data for checking; f32 can stagnate",
     "loses_when": "conversion or refinement cost exceeds arithmetic savings"},
    {"name": "sparse_matvec", "gene": "backend.kernel (LP PDHG, f64 only)",
     "choices": ["jax", "csr_thread128", "csr_warp128", "csr_warp256"],
     "evidence": "forward and adjoint time, matrix nnz and row distribution",
     "constraint": "reference agreement and adjoint identity",
     "status": "bounded CUDA/FFI CSR update templates; numerical admission required"},
    {"name": "fusion", "gene": "backend.kernel (LP PDHG, f64 only)",
     "evidence": "optimized HLO and optional device trace",
     "constraint": "HLO fusion count is not a measured launch count",
     "status": "two CUDA update kernels; compiler may add loop bookkeeping"},
]


def prepare_base(problem, genome: Genome, prescaled=None):
    """Use the same preparation and certificates as the production compiler."""
    if genome.scheme == "mix":
        raise ValueError("hardware pilot supports base schemes and wrappers")
    if genome.metric is not None:
        return _prepare_with_metric(problem, genome, prescaled=prescaled)
    scheme, params = genome_solver_params(genome)
    built = build_scheme(problem, scheme, params, check=True)
    cert = _wrap_cert(built.cert, params.rho, params.halpern, params.restart_k,
                      outer=getattr(params, "outer", None))
    if getattr(genome.backend, "kernel", "jax") != "jax":
        from atlas.backend.native import install_kernel
        built = install_kernel(problem, genome, built)
    return built, cert, params, None


def _measure(fn, args, repeats, label, hlo_dir):
    t0 = time.perf_counter()
    exe = jax.jit(fn).lower(*args).compile()
    compile_s = time.perf_counter() - t0
    jax.block_until_ready(exe(*args))
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = exe(*args)
        jax.block_until_ready(result)
        samples.append((time.perf_counter() - t0) * 1e6)
    hlo = exe.as_text()
    record = {
        "median_us": float(np.median(samples)), "samples_us": samples,
        "compile_s": compile_s,
        "hlo_sha256": hashlib.sha256(hlo.encode()).hexdigest(),
        "hlo_fusion_instructions": len(re.findall(r" = .*?\bfusion\(", hlo)),
        "hlo_custom_call_instructions": len(re.findall(r" = .*?custom-call\(", hlo)),
        "gpu_launch_count": None,
    }
    if hlo_dir:
        dest = Path(hlo_dir) / f"{label}.hlo.txt"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(hlo)
        record["hlo_file"] = str(dest)
    return record


def operator_profile(problem, genome, *, repeats=9, block_iters=100,
                     hlo_dir=None):
    built, _, _, _ = prepare_base(problem, genome)
    dyn64 = jax.tree_util.tree_map(jnp.asarray, built.dyn)
    state64 = jax.tree_util.tree_map(jnp.asarray, built.state0)
    ops = {}
    for name, dtype in (("f64", jnp.float64), ("f32", jnp.float32)):
        if dtype == jnp.float32 and getattr(genome.backend, "kernel", "jax") != "jax":
            continue
        def cast(x):
            a = jnp.asarray(x)
            return a.astype(dtype) if jnp.issubdtype(a.dtype, jnp.floating) else a
        state = jax.tree_util.tree_map(cast, state64)
        dyn = jax.tree_util.tree_map(cast, dyn64)

        def block(s, d):
            return jax.lax.fori_loop(0, block_iters,
                                    lambda i, z: built.T(z, i, d), s)
        rec = _measure(block, (state, dyn), repeats, f"base_step_{name}", hlo_dir)
        rec["block_iters"] = block_iters
        rec["per_step_us"] = rec["median_us"] / block_iters
        ops[f"base_step_{name}"] = rec
    ops["certificate_f64"] = _measure(
        built.metric, (state64, dyn64), repeats, "certificate_f64", hlo_dir)
    # Original operator: these matvecs enter original-coordinate certificates.
    if problem.L is not None:
        x = jnp.ones(problem.shape, jnp.float64)
        u = problem.L.forward(x)
        for label, fn, arg in (("original_forward", problem.L.forward, x),
                               ("original_adjoint", problem.L.adjoint, u)):
            ops[label] = _measure(fn, (arg,), repeats, label, hlo_dir)
    meta = dict(problem.data.get("meta", {}))
    matrix = problem.data.get("A_csr")
    if matrix is not None:
        row_nnz = np.diff(matrix.tocsr().indptr)
        meta.update(rows=matrix.shape[0], cols=matrix.shape[1], nnz=matrix.nnz,
                    row_nnz_quantiles=np.quantile(row_nnz, [0, .5, .9, 1]).tolist())
    device = jax.devices()[0]
    return {
        "version": 1, "kind": device.device_kind, "platform": device.platform,
        "jax_version": jax.__version__, "problem": problem.name,
        "shape": list(problem.shape), "features": meta,
        "genome_hash": content_hash(genome), "operators": ops,
        "strategies": STRATEGIES,
        "timing_scope": "warm synchronized calls; base steps amortized over a block",
        "interpretation": "standalone costs are not additive kernel shares",
    }


def context_block(profile, mode="measured"):
    """Three controlled prompt conditions, with no automatic device probing."""
    if mode == "none" or not profile:
        return None
    if mode not in ("identity", "measured"):
        raise ValueError("hardware context must be none, identity, or measured")
    context = {k: profile.get(k) for k in ("kind", "platform", "shape", "features")}
    if mode == "measured":
        context["operator_costs_us"] = {
            k: v.get("per_step_us", v["median_us"])
            for k, v in profile.get("operators", {}).items()}
        context["strategies"] = profile.get("strategies", STRATEGIES)
        context["instruction"] = (
            "Use measured workload costs, not peak TFLOPS, to propose a policy. "
            "Compare fewer checks against delayed convergence detection. "
            "The f32 benefit is empirical and may vanish at tight tolerance.")
    return "Target hardware evidence: " + json.dumps(context, sort_keys=True)
