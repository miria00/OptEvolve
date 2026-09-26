"""Generated FP64 CUDA/FFI implementations of the standard LP PDHG map.

The algorithm gate runs first. Bounded template variants then pass numerical
reference checks before timed evaluation. These checks are not a formal proof
of floating-point refinement. The original-coordinate certificate is untouched.
"""
from __future__ import annotations

import ctypes
from dataclasses import replace
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
import scipy.sparse as sp

VARIANTS = {"csr_thread128": (1, 128), "csr_warp128": (32, 128),
            "csr_warp256": (32, 256)}
_LIBRARIES = {}
_ADMITTED = {}
_TEMPLATE_CHECKS = {}


def generated_source():
    template = Path(__file__).with_name("kernels").joinpath("lp_update.cu").read_text()
    handlers = []
    for name, (lanes, threads) in VARIANTS.items():
        handlers.append(f"""
XLA_FFI_DEFINE_HANDLER_SYMBOL({name}, (Launch<{lanes}, {threads}>),
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>()
      .Attr<int64_t>("dual")
      .Arg<ffi::Buffer<ffi::F64>>().Arg<ffi::Buffer<ffi::S32>>()
      .Arg<ffi::Buffer<ffi::S32>>().Arg<ffi::Buffer<ffi::F64>>()
      .Arg<ffi::Buffer<ffi::F64>>().Arg<ffi::Buffer<ffi::F64>>()
      .Arg<ffi::Buffer<ffi::F64>>().Arg<ffi::Buffer<ffi::F64>>()
      .Ret<ffi::Buffer<ffi::F64>>(), {{ffi::Traits::kCmdBufferCompatible}});
""")
    return template.replace("// HANDLERS_GENERATED_HERE", "\n".join(handlers))


def load_kernels():
    if jax.devices()[0].platform != "gpu":
        raise ValueError("Generated LP kernels require a CUDA device")
    source = generated_source()
    nvcc = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
    version = subprocess.check_output([nvcc, "--version"], text=True)
    identity = source + version + jax.__version__ + jax.ffi.include_dir()
    digest = hashlib.sha256(identity.encode()).hexdigest()
    if digest in _LIBRARIES:
        return _LIBRARIES[digest]
    root = Path(os.environ.get("OPTEVOLVE_KERNEL_CACHE",
                    str(Path.home() / ".cache/optevolve/cuda_lp"))) / digest
    root.mkdir(parents=True, exist_ok=True)
    cu, so = root / "lp_update.cu", root / "lp_update.so"
    cu.write_text(source)
    command = [nvcc, "-std=c++17", "-O3", "--shared", "-Xcompiler", "-fPIC",
               "--fmad=false", "-gencode=arch=compute_80,code=compute_80",
               "-I" + jax.ffi.include_dir(), str(cu), "-o", str(so)]
    if not so.exists():
        t0 = time.perf_counter()
        proc = subprocess.run(command, text=True, capture_output=True)
        (root / "compile.log").write_text(proc.stdout + proc.stderr)
        if proc.returncode:
            raise RuntimeError(f"CUDA build failed: {root / 'compile.log'}")
        (root / "build.json").write_text(json.dumps({"command": command,
            "compile_s": time.perf_counter() - t0, "nvcc": version,
            "source_sha256": hashlib.sha256(source.encode()).hexdigest()}, indent=2))
    lib = ctypes.CDLL(str(so))
    targets = {}
    for name in VARIANTS:
        target = "optevolve_" + digest[:16] + "_" + name
        jax.ffi.register_ffi_target(target, jax.ffi.pycapsule(getattr(lib, name)),
                                    platform="CUDA")
        targets[name] = target
    result = (lib, targets, {"build_digest": digest, "artifact_dir": str(root)})
    _LIBRARIES[digest] = result  # retain the library for the handler lifetime
    return result


def _csr(matrix):
    matrix = sp.csr_matrix(matrix, dtype=np.float64, copy=True)
    matrix.sum_duplicates()
    matrix.sort_indices()
    matrix.check_format(full_check=True)
    if max(*matrix.shape, matrix.nnz) > np.iinfo(np.int32).max:
        raise ValueError("Generated CSR kernel supports int32 indices only")
    if not np.isfinite(matrix.data).all():
        raise ValueError("Generated CSR kernel requires finite matrix entries")
    return tuple(jnp.asarray(x) for x in (matrix.data,
        matrix.indices.astype(np.int32), matrix.indptr.astype(np.int32)))


def install_kernel(problem, genome, built):
    """Replace only the LP base update, preserving wrappers and certificates."""
    name = getattr(genome.backend, "kernel", "jax")
    if name == "jax":
        return built
    if name not in VARIANTS:
        raise ValueError(f"Unknown generated kernel: {name}")
    if genome.scheme != "pdhg" or problem.name != "lp_standard":
        raise ValueError("Generated CSR update requires standard-form LP PDHG")
    if genome.backend.precision != "f64":
        raise ValueError("Generated CSR update currently supports f64 only")
    # Identity of the prepared reference operator distinguishes instances.
    cache_key = (built.static_key, name, genome.params)
    if cache_key in _ADMITTED:
        return _ADMITTED[cache_key]
    _, targets, artifact = load_kernels()
    target = targets[name]
    if target not in _TEMPLATE_CHECKS:
        from atlas.backend.unit_tests import run_check
        from atlas.typing_ import TypeCheckError
        def implementation(A, v, old, base, cost, step, dual):
            return jax.ffi.ffi_call(target, jax.ShapeDtypeStruct(base.shape, np.float64))(
                *_csr(A), *map(jnp.asarray, (v, old, base, cost, step)), dual=np.int64(dual))
        report = run_check("lp_csr_update", implementation)
        if not report.passed:
            raise TypeCheckError(f"Generated kernel admission rejected: {report}")
        _TEMPLATE_CHECKS[target] = vars(report)
    A = problem.data.get("A_csr")
    if A is None:
        A = problem.data["A_dense"]
    A = sp.csr_matrix(A)
    c, b = np.asarray(problem.data["c"]), np.asarray(problem.data["b"])
    if genome.metric is not None:
        d1, d2 = np.asarray(built.dyn["metric_d1"]), np.asarray(built.dyn["metric_d2"])
        A = sp.diags(d1) @ A @ sp.diags(d2)
        c, b = d2 * c, d1 * b
    dyn = {**built.dyn, "cuda_A": _csr(A), "cuda_AT": _csr(A.T),
           "cuda_c": jnp.asarray(c), "cuda_b": jnp.asarray(b)}

    def T(state, k, d):
        x, u = state
        if x.dtype != jnp.float64 or u.dtype != jnp.float64:
            raise ValueError("Generated CSR update accepts only float64 states")
        xp = jax.ffi.ffi_call(target, jax.ShapeDtypeStruct(x.shape, x.dtype))(
            *d["cuda_AT"], u, u, x, d["cuda_c"], jnp.asarray(d["t"]),
            dual=np.int64(0))
        up = jax.ffi.ffi_call(target, jax.ShapeDtypeStruct(u.shape, u.dtype))(
            *d["cuda_A"], xp, x, u, d["cuda_b"], jnp.asarray(d["s"]),
            dual=np.int64(1))
        return xp, up

    candidate = replace(built, T=T, dyn=dyn,
                        static_key=("cuda_lp", name, artifact["build_digest"], built.static_key))
    # Admission outside fitness: zero and three deterministic random states.
    ref, got = jax.jit(built.T), jax.jit(T)
    rng = np.random.default_rng(7001)
    max_error = 0.
    t0 = time.perf_counter()
    for i in range(4):
        state = built.state0 if i == 0 else tuple(
            jnp.asarray(rng.normal(size=s.shape)) for s in built.state0)
        expected = ref(state, 0, built.dyn)
        actual = got(state, 0, dyn)
        for a, e in zip(actual, expected):
            a, e = np.asarray(a), np.asarray(e)
            if not np.isfinite(a).all() or not np.isfinite(e).all():
                raise ValueError("Non-finite generated-kernel admission result")
            np.testing.assert_allclose(a, e, rtol=2e-10, atol=2e-10,
                err_msg="Generated LP kernel rejected by reference admission")
            max_error = max(max_error, float(np.max(np.abs(a-e), initial=0.)))
    candidate.kernel_evidence = {**artifact, "variant": name,
        "registry_check": _TEMPLATE_CHECKS[target],
        "admission_states": 4, "max_abs_error": max_error,
        "admission_s": time.perf_counter() - t0,
        "scope": "numerical reference admission, not formal FP refinement"}
    _ADMITTED[cache_key] = candidate
    return candidate
