"""The penalty parameter, selected by measurement instead of by a constant.

A hand-set rho = 1 was calibrated on one family and DIVERGES on another
(gap 30.4 on LM22).  A scale-derived formula, rho = tr(A^T A)/n, fixes that
family and is WORSE than the constant on the one the constant came from.  So
there is no transferable formula, which is the same shape as the lasso
engine's finding that min(n,p) is a correct bound on the active-set size and a
poor estimate of it.

The fix there was to measure the quantity cheaply rather than bound it, and
the same fix applies here.  Run each candidate rho for a short burst, read the
actual gap decrease, keep the winner AND ITS ITERATE, and continue.  The probe
is not overhead charged against the solve; it is the first phase of the solve,
which is the design note the lasso agent records for its own probe.

Reported against a per-instance oracle over the same candidate set, so the
number is a regret, not a speedup against a strawman.
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fcls_core import fw_gap_and_feas


def build(dtype="f64"):
    os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.22")
    import jax, jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    dev = jax.devices("gpu")[0]

    def proj(V):
        n = V.shape[0]
        U = jnp.sort(V, axis=0)[::-1]
        css = jnp.cumsum(U, axis=0) - 1.0
        ind = jnp.arange(1, n + 1, dtype=V.dtype).reshape(-1, 1)
        r = n - 1 - jnp.argmax(((U - css / ind) > 0)[::-1], axis=0)
        th = jnp.take_along_axis(css, r[None, :], axis=0)[0] / (r + 1).astype(V.dtype)
        return jnp.maximum(V - th, 0.0)

    @jax.jit
    def chunk(Z, U, M, B, rho, K, al):
        def body(i, s):
            Z, U = s
            X = M @ (B + rho * (Z - U))
            XA = al * X + (1.0 - al) * Z
            Zn = proj(XA + U)
            return (Zn, U + XA - Zn)
        return jax.lax.fori_loop(0, K, body, (Z, U))
    return jax, jnp, dev, chunk


def solve_fixed(jax, jnp, dev, chunk, A, B, AtA, AtB, rho, target,
                burst=50, max_bursts=200, al=1.6, Z0=None, U0=None, t0=None):
    n = A.shape[1]
    if t0 is None:
        t0 = time.perf_counter()
    with jax.default_device(dev):
        M = jnp.linalg.inv(jnp.asarray(AtA) + rho * jnp.eye(n))
        Z = jnp.asarray(Z0) if Z0 is not None else jnp.full((n, B.shape[1]), 1.0 / n)
        U = jnp.asarray(U0) if U0 is not None else jnp.zeros((n, B.shape[1]))
        Bd = jnp.asarray(AtB)
        Z, U = chunk(Z, U, M, Bd, rho, 1, al); Z.block_until_ready()
        for k in range(max_bursts):
            Z, U = chunk(Z, U, M, Bd, rho, burst, al); Z.block_until_ready()
            gw, _, feas, _ = fw_gap_and_feas(A, B, np.asarray(Z, dtype=np.float64))
            if max(gw, feas) <= target:
                return time.perf_counter() - t0, (k + 1) * burst, gw
    return None, (max_bursts * burst), gw


def probe_then_solve(jax, jnp, dev, chunk, A, B, AtA, AtB, cands, target,
                     probe_iters=30, al=1.6):
    """Short burst per candidate, keep the winner and its iterate, continue."""
    n = A.shape[1]
    t0 = time.perf_counter()
    best = None
    with jax.default_device(dev):
        Bd = jnp.asarray(AtB); AtAd = jnp.asarray(AtA)
        for rho in cands:
            M = jnp.linalg.inv(AtAd + rho * jnp.eye(n))
            Z = jnp.full((n, B.shape[1]), 1.0 / n); U = jnp.zeros_like(Z)
            Z, U = chunk(Z, U, M, Bd, rho, probe_iters, al); Z.block_until_ready()
            gw, _, feas, _ = fw_gap_and_feas(A, B, np.asarray(Z, dtype=np.float64))
            sc = max(gw, feas)
            if best is None or sc < best[0]:
                best = (sc, rho, np.asarray(Z, dtype=np.float64),
                        np.asarray(U, dtype=np.float64))
    t_probe = time.perf_counter() - t0
    # the probe's iterate is kept: the diagnostic is the first phase of the solve
    t_tot, iters, gw = solve_fixed(jax, jnp, dev, chunk, A, B, AtA, AtB, best[1],
                                   target, Z0=best[2], U0=best[3], t0=t0)
    return t_tot, t_probe, best[1], iters, gw
