"""Hardware detection and a cached matmul TFLOPS probe (backend lane).

detect_device() returns a JSON-serializable record

    {kind, platform, memory_gb, tflops_f32, tflops_f64, probe: {...}}

where kind is jax.devices()[0].device_kind of the probed device, platform is
its jax platform string, memory_gb is total physical memory of that device,
and tflops_f32 / tflops_f64 are measured square-matmul throughputs from a
short (about 2 second) probe. Results are cached per process, so repeated
calls (e.g. from the Context Builder on every genome evaluation) pay the
probe cost once.

CPU-pinned processes: atlas.__init__ pins JAX_PLATFORMS=cpu on import so the
vLLM server keeps the GPU. detect_device(prefer_gpu=True) still reports the
GPU on such a machine by launching a short subprocess with JAX_PLATFORMS=cuda
and XLA preallocation disabled; probe matrices are small (peak well under
1 GB) so the probe coexists with a vLLM server holding most of the card.
If no GPU is visible (or the subprocess fails) the record describes the CPU.

The f64/f32 TFLOPS ratio in this record is the hardware asymmetry the
Hawkeye-style backend genes are meant to exploit: consumer cards run f64 at
roughly 1/60 of f32 throughput, data-center cards at about 1/2.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time

# Probe sizes chosen so each timed matmul is milliseconds-scale on the target
# class of device and total memory stays small (f32 8192^2 * 3 = 768 MB;
# 4096 undersaturates a 4090/5090 by roughly 30 percent).
_PROBE_SIZES = {
    "gpu": {"float32": 8192, "float64": 4096},
    "cpu": {"float32": 1536, "float64": 1024},
}

_CACHE: dict[str, dict] = {}


def _matmul_tflops(dtype_name: str, n: int, target_seconds: float = 0.8,
                   min_reps: int = 3) -> float:
    """Measured TFLOPS of repeated n x n matmul at the given dtype.

    Compiles first (AOT, outside the timed region, per v1 lesson 4), then
    repeats until at least target_seconds have elapsed. FLOP count per
    matmul is 2 n^3.
    """
    import jax
    import jax.numpy as jnp

    dtype = jnp.dtype(dtype_name)
    a = jax.random.normal(jax.random.PRNGKey(0), (n, n), dtype=dtype)
    b = jax.random.normal(jax.random.PRNGKey(1), (n, n), dtype=dtype)
    f = jax.jit(lambda a, b: a @ b)
    f(a, b).block_until_ready()  # trace + compile, untimed

    reps = 0
    t0 = time.perf_counter()
    while True:
        f(a, b).block_until_ready()
        reps += 1
        elapsed = time.perf_counter() - t0
        if reps >= min_reps and elapsed >= target_seconds:
            break
        if elapsed > 4.0 * target_seconds:
            break
    return 2.0 * float(n) ** 3 * reps / elapsed / 1e12


def _cpu_memory_gb() -> float:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
    except (ValueError, OSError):
        return float("nan")


def _gpu_memory_gb() -> float:
    """Total physical GPU memory via nvidia-smi (JAX memory_stats reports the
    XLA pool limit, not the card), falling back to memory_stats."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode == 0 and out.stdout.strip():
            return float(out.stdout.strip().splitlines()[0]) / 1024.0
    except (OSError, ValueError, subprocess.TimeoutExpired):
        pass
    try:
        import jax
        stats = jax.devices()[0].memory_stats()
        if stats and "bytes_limit" in stats:
            return stats["bytes_limit"] / 2**30
    except Exception:
        pass
    return float("nan")


def _gpu_visible() -> bool:
    if shutil.which("nvidia-smi") is None:
        return False
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True,
                             text=True, timeout=10)
        return out.returncode == 0 and "GPU" in out.stdout
    except (OSError, subprocess.TimeoutExpired):
        return False


def _probe_current() -> dict:
    """Probe the process-default JAX device (whatever JAX_PLATFORMS gave us)."""
    import jax

    dev = jax.devices()[0]
    platform = dev.platform
    is_gpu = platform != "cpu"
    sizes = _PROBE_SIZES["gpu" if is_gpu else "cpu"]
    t32 = _matmul_tflops("float32", sizes["float32"])
    t64 = _matmul_tflops("float64", sizes["float64"])
    mem = _gpu_memory_gb() if is_gpu else _cpu_memory_gb()
    return {
        "kind": dev.device_kind,
        "platform": platform,
        "memory_gb": round(mem, 2),
        "tflops_f32": round(t32, 3),
        "tflops_f64": round(t64, 3),
        "probe": {
            "n_f32": sizes["float32"],
            "n_f64": sizes["float64"],
            "jax_version": jax.__version__,
            "x64_enabled": bool(jax.config.jax_enable_x64),
        },
    }


def _subprocess_gpu_probe(timeout: float = 300.0) -> dict | None:
    """Probe the GPU from a fresh subprocess with JAX_PLATFORMS=cuda.

    Used when this process is CPU-pinned (the LOCAL default). Preallocation
    is disabled so the probe only takes the few hundred MB it actually needs
    next to a resident vLLM server.
    """
    env = dict(os.environ)
    env["JAX_PLATFORMS"] = "cuda"
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    try:
        out = subprocess.run(
            [sys.executable, "-m", "atlas.backend.device", "--json"],
            env=env, capture_output=True, text=True, timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    text = out.stdout
    start = text.find("{")
    if start < 0:
        return None
    try:
        rec = json.loads(text[start:])
    except json.JSONDecodeError:
        return None
    if rec.get("platform") == "cpu":
        return None  # subprocess silently fell back; not a GPU record
    return rec


def detect_device(prefer_gpu: bool = True, force_refresh: bool = False) -> dict:
    """Return the (cached) device record for this machine.

    prefer_gpu=True (default): if the process default device is a GPU, probe
    it in-process; if the process is CPU-pinned but nvidia-smi shows a GPU,
    probe it in a subprocess with the GPU temporarily allowed; otherwise
    probe the CPU. prefer_gpu=False always probes the process-default device.
    """
    import jax

    inproc_gpu = jax.devices()[0].platform != "cpu"
    want = "gpu" if (inproc_gpu or (prefer_gpu and _gpu_visible())) else "cpu"
    if not force_refresh and want in _CACHE:
        return _CACHE[want]

    if want == "gpu" and not inproc_gpu:
        rec = _subprocess_gpu_probe()
        if rec is None:
            want = "cpu"
            if not force_refresh and want in _CACHE:
                return _CACHE[want]
            rec = _probe_current()
    else:
        rec = _probe_current()

    _CACHE[want] = rec
    return rec


if __name__ == "__main__":
    # Subprocess / CLI entrypoint: probe the process-default device and print
    # the record as JSON on stdout (the last JSON object in stdout is parsed
    # by _subprocess_gpu_probe, so keep stdout clean).
    as_json = "--json" in sys.argv
    record = _probe_current()
    print(json.dumps(record) if as_json else json.dumps(record, indent=2))
