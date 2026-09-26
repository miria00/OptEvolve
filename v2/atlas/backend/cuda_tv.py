"""Bounded FP64 TV stencil generation and numerical admission for both schemes."""
from __future__ import annotations

import ctypes
from dataclasses import replace
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import jax
import jax.numpy as jnp
import numpy as np

from atlas.typing_ import TypeCheckError

VARIANTS = {"tv_fused128": (128, False), "tv_fused256": (256, False),
            "tv_recip256": (256, True)}
_TEMPLATE_CHECKS = {}


def check_compatible(problem, genome):
    """Reject unsupported atoms/geometry before any CUDA compilation."""
    from atlas.operators.linear import Grad2D
    from atlas.operators.prox import QuadraticDataFidelity, ScaledL1, Zero
    if (genome.scheme not in ("pdhg", "condat_vu") or
        genome.backend.precision != "f64" or genome.metric is not None or
        type(problem.L) is not Grad2D or
        type(problem.f) is not QuadraticDataFidelity or
        type(problem.g) is not Zero or type(problem.h) is not ScaledL1 or
        problem.name != "tv_denoise" or tuple(problem.L.shape) != tuple(problem.shape)):
        raise TypeCheckError("Generated TV kernels require canonical Grad2D TV PDHG/Condat-Vu, f64, and no metric transform")


def generated_source():
    source = Path(__file__).with_name("kernels").joinpath("tv_update.cu").read_text()
    handlers = []
    for name, (threads, reciprocal) in VARIANTS.items():
        handlers.append(f"""
XLA_FFI_DEFINE_HANDLER_SYMBOL({name}, (LaunchTV<{threads}, {str(reciprocal).lower()}>),
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>()
      .Attr<int64_t>("proximal").Attr<double>("rho")
      .Arg<ffi::Buffer<ffi::F64>>().Arg<ffi::Buffer<ffi::F64>>()
      .Arg<ffi::Buffer<ffi::F64>>().Arg<ffi::Buffer<ffi::F64>>()
      .Arg<ffi::Buffer<ffi::F64>>().Arg<ffi::Buffer<ffi::F64>>()
      .Arg<ffi::Buffer<ffi::F64>>()
      .Ret<ffi::Buffer<ffi::F64>>().Ret<ffi::Buffer<ffi::F64>>(),
      {{ffi::Traits::kCmdBufferCompatible}});
""")
    return source.replace("// HANDLERS_GENERATED_HERE", "\n".join(handlers))


@lru_cache(maxsize=1)
def load_kernels():
    if jax.devices()[0].platform != "gpu":
        raise ValueError("Generated TV kernels require CUDA")
    source = generated_source()
    nvcc = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
    version = subprocess.check_output([nvcc, "--version"], text=True)
    # Normal IEEE fused multiply-add is permitted, as in optimized XLA.
    # Fast division, flush-to-zero and --use_fast_math remain disabled.
    # Build options enter the cache identity as well as the artifact manifest.
    flags = ["-std=c++17", "-O3", "--shared", "-Xcompiler", "-fPIC",
             "--fmad=true", "-gencode=arch=compute_80,code=compute_80"]
    identity = source + version + jax.__version__ + jax.ffi.include_dir() + json.dumps(flags)
    digest = hashlib.sha256(identity.encode()).hexdigest()
    root = Path(os.environ.get("OPTEVOLVE_TV_KERNEL_CACHE", str(Path.home()/".cache/optevolve/cuda_tv"))) / digest
    root.mkdir(parents=True, exist_ok=True)
    cu, so = root/"tv_update.cu", root/"tv_update.so"
    cu.write_text(source)
    command = [nvcc, *flags, "-I"+jax.ffi.include_dir(), str(cu), "-o", str(so)]
    if not so.exists():
        start = time.perf_counter()
        proc = subprocess.run(command, text=True, capture_output=True)
        (root/"compile.log").write_text(proc.stdout+proc.stderr)
        if proc.returncode:
            raise RuntimeError(f"CUDA TV build failed: {root/'compile.log'}")
        (root/"build.json").write_text(json.dumps({"command": command,
            "compile_s": time.perf_counter()-start, "nvcc": version,
            "source_sha256": hashlib.sha256(source.encode()).hexdigest()}, indent=2))
    lib = ctypes.CDLL(str(so))
    targets = {}
    for name in VARIANTS:
        target = f"optevolve_{digest[:16]}_{name}"
        jax.ffi.register_ffi_target(target, jax.ffi.pycapsule(getattr(lib, name)), platform="CUDA")
        targets[name] = target
    return lib, targets, {"build_digest": digest, "artifact_dir": str(root)}


def call_update(target, x, u, y, t, s, lam, proximal, rho=1., inverse=None):
    if x.dtype != jnp.float64 or u.dtype != jnp.float64:
        raise ValueError("Generated TV stencil accepts only float64 iterates")
    return tuple(jax.ffi.ffi_call(target, (
        jax.ShapeDtypeStruct(x.shape, x.dtype), jax.ShapeDtypeStruct(u.shape, u.dtype)))(
            x, u, y, jnp.asarray(t), jnp.asarray(s), jnp.asarray(lam),
            jnp.asarray(1./(1.+t) if inverse is None else inverse),
            proximal=np.int64(proximal), rho=np.float64(rho)))


def install_kernel(problem, genome, built):
    check_compatible(problem, genome)
    name = genome.backend.kernel
    if name not in VARIANTS:
        raise TypeCheckError(f"Unknown generated TV kernel: {name}")
    _, targets, artifact = load_kernels()
    target = targets[name]
    if target not in _TEMPLATE_CHECKS:
        from atlas.backend.unit_tests import run_check
        def implementation(x, u, y, t, s, lam, proximal, rho):
            a, b = call_update(target, *map(jnp.asarray, (x, u, y)), t, s, lam, proximal, rho)
            return np.concatenate((np.asarray(a).ravel(), np.asarray(b).ravel()))
        report = run_check("tv_update", implementation)
        if not report.passed:
            raise TypeCheckError(f"TV template admission rejected: {report}")
        _TEMPLATE_CHECKS[target] = vars(report)
    proximal = genome.scheme == "pdhg"
    atom = "phi" if proximal else "f"
    # Compute once during preparation, not once per pixel or solver iteration.
    # This is a numerically admitted reciprocal rewrite, not bitwise division.
    dyn = {**built.dyn, "tv_inverse": jnp.asarray(1./(1.+float(built.dyn["t"])), dtype=jnp.float64)}
    def relaxed(state, k, d, rho):
        return call_update(target, *state, d[atom].y, d["t"], d["s"], d["h"].lam, proximal, rho, d["tv_inverse"])
    def T(state, k, d):
        return relaxed(state, k, d, 1.)
    candidate = replace(built, T=T, relaxed_T=relaxed, dyn=dyn,
                        static_key=("cuda_tv", name, artifact["build_digest"], built.static_key))
    rho = 1. if genome.params.rho is None else float(genome.params.rho)
    rng, worst, start = np.random.default_rng(7103), 0., time.perf_counter()
    ref = jax.jit(built.T)
    raw, rel = jax.jit(T), jax.jit(lambda st, d: relaxed(st, 0, d, rho))
    for i in range(4):
        state = built.state0 if i == 0 else tuple(jnp.asarray(rng.normal(size=v.shape)) for v in built.state0)
        expected = ref(state, 0, built.dyn)
        pairs = [(raw(state, 0, dyn), expected),
                 (rel(state, dyn), tuple((1.-rho)*s+rho*t for s,t in zip(state, expected)))]
        for actual, expected_ in pairs:
            for a, e in zip(actual, expected_):
                a, e = np.asarray(a), np.asarray(e)
                if not np.isfinite(a).all() or not np.isfinite(e).all():
                    raise TypeCheckError("Nonfinite TV admission result")
                np.testing.assert_allclose(a, e, rtol=2e-12, atol=2e-12)
                worst = max(worst, float(np.max(np.abs(a-e))))
    candidate.kernel_evidence = {**artifact, "variant": name,
        "registry_check": _TEMPLATE_CHECKS[target], "admission_states": 4,
        "admission_maps": ["base", "relaxed"], "max_abs_error": worst,
        "admission_s": time.perf_counter()-start,
        "scope": "numerical admission, not formal floating-point proof"}
    return candidate
