"""The atlas package must enforce CPU + float64 at import time, so an
entrypoint that forgets JAX_PLATFORMS/jax_enable_x64 cannot silently run
the pipeline in float32 (regression: only entrypoints set the flags; a
forgetful caller got float32 arrays and an unreachable 1e-9 oracle)."""
import os
import subprocess
import sys

V2_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_atlas_import_enforces_cpu_and_x64():
    # Strip every JAX_* var: the library itself must pin the platform/dtype.
    env = {k: v for k, v in os.environ.items() if not k.startswith("JAX_")}
    code = (
        "import atlas, jax, jax.numpy as jnp\n"
        "assert jax.config.jax_enable_x64, 'x64 not enabled by atlas import'\n"
        "assert jax.default_backend() == 'cpu', jax.default_backend()\n"
        "a = jnp.asarray([1.0], dtype=jnp.float64)\n"
        "assert a.dtype == jnp.float64, a.dtype\n"
    )
    r = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        cwd=V2_ROOT,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert r.returncode == 0, f"stdout={r.stdout}\nstderr={r.stderr}"
