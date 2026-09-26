import jax
import jax.numpy as jnp

def project(v, lo, hi, a, b):
    """Single-step semismooth Newton: projects v onto {x: lo<=x<=hi, a'x=b}.
    Initial guess is the unconstrained KKT multiplier; one Newton correction
    for the active set. Sufficient within a converging outer solver."""
    a2 = a * a
    mu = (a @ v - b) / a2.sum()
    r = v - mu * a
    x = jnp.clip(r, lo, hi)
    d = (a2 * ((r > lo) & (r < hi))).sum() + 1e-30
    mu += (a @ x - b) / d
    return jnp.clip(v - mu * a, lo, hi)