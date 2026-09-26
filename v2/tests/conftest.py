"""Force CPU + float64 BEFORE any jax import (GPU is owned by vLLM)."""
import os

_gpu = os.environ.get("OPTEVOLVE_TEST_GPU")
# "1": GPU tests on CUDA alone. "both": one process holding CUDA AND CPU,
# which the cross-device cache and device-gene tests need; before this
# option no pytest process could hold two backends, so those tests could
# only ever skip. Anything else: CPU only, the default suite.
os.environ["JAX_PLATFORMS"] = {"1": "cuda", "both": "cuda,cpu"}.get(_gpu, "cpu")

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402
import pytest  # noqa: E402


@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(0)


@pytest.fixture(scope="session")
def tv_instance_32():
    """32x32 synthetic piecewise-smooth image + noise, and lam."""
    r = np.random.default_rng(7)
    H = W = 32
    xx, yy = np.meshgrid(np.linspace(0, 1, W), np.linspace(0, 1, H))
    img = 0.6 * (xx > 0.5) + 0.3 * (yy > 0.35) + 0.2 * np.sin(4 * xx)
    noisy = img + 0.08 * r.standard_normal((H, W))
    return noisy.astype(np.float64), 0.1
