"""Process-parallel fitness evaluation (multiprocessing pool, CPU).

Usage:
    from atlas.evolve.parallel import make_pool
    with make_pool(n_workers=8) as pool:
        hist = evolve(backend, fitness_fn, generations, pop_size, pool=pool)

Requirements: `fitness_fn` and the Genome dataclasses must be picklable
(module-level functions; genomes are plain frozen dataclasses and always
pickle). Each worker process is initialized with JAX_PLATFORMS=cpu BEFORE
any jax import (the GPU belongs to the vLLM server) and inherits the
persistent XLA compilation cache dir, so workers share compiled kernels
across processes and runs.

Default start method is "spawn": fork after jax initialization is unsafe
(XLA runtime threads do not survive fork). "fork" is accepted for
pure-Python fitness functions that never touch jax.
"""
from __future__ import annotations

import multiprocessing as mp
import os


def _worker_init(extra_env=None):
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false")
    # Workers run one genome each; keep intra-op threading modest so
    # n_workers processes do not oversubscribe the cores.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    if extra_env:
        os.environ.update(extra_env)


def make_pool(n_workers: int, ctx: str = "spawn", extra_env: dict | None = None):
    """A multiprocessing.Pool whose workers are safe for jax-on-CPU
    fitness evaluation. Use as a context manager."""
    if n_workers < 1:
        raise ValueError("n_workers must be >= 1")
    return mp.get_context(ctx).Pool(
        processes=n_workers, initializer=_worker_init, initargs=(extra_env,)
    )


# -- picklable demo fitness (used by tests; a real fitness lives with its
#    cell, e.g. scripts/run_tv_cell.py's worker) ---------------------------


def demo_fitness(genome):
    """Pure-Python stand-in fitness: peaked at tau = 0.05 (higher better)."""
    if genome.scheme != "mix":
        tau = genome.params.tau
    elif genome.mix.kind == "avg":
        tau = genome.mix.a.tau
    elif genome.mix.kind == "switch_k":
        tau = genome.mix.tail.tau
    else:  # deprecated residual-window switch
        tau = genome.mix.fallback.tau
    return -abs(tau - 0.05), {"tau": tau}
