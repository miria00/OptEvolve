"""Compile caches must never hand one program to inputs it was not built for."""
import jax
import jax.numpy as jnp
import pytest

import atlas  # noqa: F401  (x64 and platform setup)
import atlas.backend.precision as P
from atlas.wrappers.km import _abstract


def _cpu_and_gpu():
    try:
        return jax.devices("cpu")[0], jax.devices("cuda")[0]
    except RuntimeError:
        return None


def test_cache_key_records_placement():
    key = _abstract(jnp.ones(4))
    assert key[1][0][2][0][0] == jax.devices()[0].platform


@pytest.mark.skipif(_cpu_and_gpu() is None, reason="needs cpu and cuda backends in one process")
def test_cache_key_distinguishes_devices():
    cpu, gpu = _cpu_and_gpu()
    assert _abstract(jax.device_put(jnp.ones(4), cpu)) != _abstract(jax.device_put(jnp.ones(4), gpu))


def test_run_device_policy_does_not_reuse_a_closure_executable():
    # Regression for the 2026-09-10 bug: with static_key=None two steps that
    # differ only inside their closures produced the same key, and the second
    # silently ran the first's program (rho=1.0 returned rho=1.9's 640 iters).
    x0 = jnp.ones(8, dtype=jnp.float64)
    dyn = {"_": jnp.zeros(())}
    gap = lambda s, g: jnp.sum(jnp.abs(s))
    fast = P.run_device_policy(lambda z, i, d: 0.5 * z, x0, dyn, gap, dyn,
                               K=4, max_iters=40, tol=1e-6, static_key=None)
    slow = P.run_device_policy(lambda z, i, d: 0.9 * z, x0, dyn, gap, dyn,
                               K=4, max_iters=40, tol=1e-6, static_key=None)
    assert fast.iters == 24 and slow.iters == 40


def test_every_cache_uses_one_key_function():
    # A second copy of the key function, blind to placement, is exactly how a
    # GPU solve got a CPU-compiled program after the first copy was fixed.
    import atlas.wrappers.km as km
    from atlas import _cache_keys
    assert P._abstract is km._abstract is _cache_keys.abstract
