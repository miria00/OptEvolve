"""Fitness for the TV cell: a genome is evaluated by actually solving.

This is the seam that was missing.  The genome's scheme and stepsize genes
ARE the Condat-Vu parameters the regime-2 arm runs, so a candidate genome is
evaluated by running that arm on a clinical OCT patch and returning the
negated time to a certified primal-dual gap.  Higher is better; a candidate
that never certifies inside the budget scores -inf and is never clamped.

The iteration is compiled ONCE and the genes are passed as device scalars, so
the search pays one compilation, not one per candidate.
"""
from __future__ import annotations
import math, os, time
import numpy as np

# Clearing this is load bearing: with LD_LIBRARY_PATH set, JAX silently
# falls back to CPU or fails to find a GPU platform at all.  Doing it here
# rather than in the caller means a nohup'd run cannot inherit the trap,
# which is exactly how the first C/L/K ablation died.
os.environ.pop("LD_LIBRARY_PATH", None)
os.environ["JAX_PLATFORMS"] = "cuda,cpu"
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.22")
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tv_core import certify

_DEV = jax.devices("gpu")[0]


def _grad(x):
    gx = jnp.zeros_like(x).at[:-1, :].set(x[1:, :] - x[:-1, :])
    gy = jnp.zeros_like(x).at[:, :-1].set(x[:, 1:] - x[:, :-1])
    return gx, gy


def _div(px, py):
    d = jnp.zeros_like(px).at[0, :].set(px[0, :])
    d = d.at[1:-1, :].set(px[1:-1, :] - px[:-2, :]).at[-1, :].set(-px[-2, :])
    e = jnp.zeros_like(py).at[:, 0].set(py[:, 0])
    e = e.at[:, 1:-1].set(py[:, 1:-1] - py[:, :-2]).at[:, -1].set(-py[:, -2])
    return d + e


@jax.jit
def _chunk(x, px, py, y, lam, tau, sig, rho, K):
    def body(i, s):
        x, px, py = s
        xn = x - tau * ((x - y) - _div(px, py))
        xn = x + rho * (xn - x)
        xb = 2.0 * xn - x
        gx, gy = _grad(xb)
        qx, qy = px + sig * gx, py + sig * gy
        nrm = jnp.sqrt(qx * qx + qy * qy)
        sc = jnp.minimum(1.0, lam / jnp.maximum(nrm, 1e-30))
        return (xn, qx * sc, qy * sc)
    return jax.lax.fori_loop(0, K, body, (x, px, py))


class TVCell:
    """One OCT patch, one accuracy target, one compiled kernel."""

    def __init__(self, npy, lam=0.1, target=1e-6, max_iter=60000, chunk=500,
                 budget_s=8.0):
        self.y = np.load(npy)
        self.lam, self.target = lam, target
        self.max_iter, self.chunk, self.budget_s = max_iter, chunk, budget_s
        self.n_eval = 0

    def __call__(self, genome):
        """genome -> negated time to a certified gap (higher is better)."""
        p = genome.params
        tau = float(p.tau)
        sig = float(p.sigma) if p.sigma is not None else 1.0 / (8.0 * tau)
        rho = float(p.rho) if p.rho is not None else 1.0
        # the Stage-0 gate has already checked the side condition; this is
        # only a guard against a non-finite gene reaching the kernel
        if not all(map(math.isfinite, (tau, sig, rho))) or tau <= 0 or sig <= 0:
            return -math.inf, {"reason": "non-finite gene"}
        self.n_eval += 1
        t0 = time.perf_counter()
        with jax.default_device(_DEV):
            yd = jnp.asarray(self.y)
            x = yd; px = jnp.zeros_like(yd); py = jnp.zeros_like(yd)
            args = (jnp.asarray(self.lam), jnp.asarray(tau),
                    jnp.asarray(sig), jnp.asarray(rho))
            it = 0
            while it < self.max_iter and time.perf_counter() - t0 < self.budget_s:
                x, px, py = _chunk(x, px, py, yd, *args, self.chunk)
                x.block_until_ready(); it += self.chunk
                g, P, G = certify(np.asarray(x, dtype=np.float64), self.y, self.lam,
                                  np.asarray(px, dtype=np.float64),
                                  np.asarray(py, dtype=np.float64))
                if not math.isfinite(g):
                    return -math.inf, {"reason": "diverged", "iters": it}
                if g <= self.target:
                    el = time.perf_counter() - t0
                    return -el, {"t": el, "iters": it, "gap": g}
        return -math.inf, {"reason": "no certificate in budget",
                           "iters": it, "gap": float(g)}
