import jax
import jax.numpy as jnp


# EVOLVE-BLOCK-START
def project(v, lo, hi, a, b):
    """Euclidean projection onto {lo <= x <= hi, a'x = b}: 60-step bisection on the multiplier mu.

    x(mu) = clip(v - mu * a, lo, hi) and a'x(mu) is nonincreasing in mu, so bisection on a'x(mu) = b converges.
    """
    def phi(mu):
        return jnp.sum(a * jnp.clip(v - mu * a, lo, hi)) - b

    scale = jnp.maximum(jnp.max(jnp.abs(v)) / jnp.maximum(jnp.max(jnp.abs(a)), 1e-30), 1.0)
    lo_mu, hi_mu = -1e3 * scale, 1e3 * scale

    def body(_, state):
        l_, h_ = state
        m = 0.5 * (l_ + h_)
        return jnp.where(phi(m) > 0, m, l_), jnp.where(phi(m) > 0, h_, m)

    l_, h_ = jax.lax.fori_loop(0, 60, body, (lo_mu, hi_mu))
    return jnp.clip(v - 0.5 * (l_ + h_) * a, lo, hi)
# EVOLVE-BLOCK-END
