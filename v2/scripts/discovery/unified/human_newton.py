import jax
import jax.numpy as jnp


def project(v, lo, hi, a, b):
    """Exact projection onto {lo <= x <= hi, a'x = b} by safeguarded semismooth Newton on the multiplier.

    phi(mu) = a'clip(v - mu a, lo, hi) - b is piecewise linear and nonincreasing with slope -sum_{free} a_i^2.
    A Newton step lands exactly on the root once mu is in the root's linear piece.

    The bracket is computed EXACTLY from the breakpoints rather than guessed, which is what a first version got wrong
    (a +-1e3*scale guess left 16 steps too few to reach the right piece; the certificate rejected it at error 0.76):
      h = max_i (v_i - lo_i)/a_i : every coordinate sits at lo above it, so phi(h) = a'lo - b <= 0 by feasibility;
      l = min over finite breakpoints, lowered to the exact root of the linear tail when some hi_i = +inf, so
          phi(l) >= 0.
    Written by hand to test whether the synthesis spec is reachable at all.
    """
    big = 1e300
    a2 = a * a
    hi_fin = jnp.isfinite(hi)
    bp_lo = (v - lo) / a
    bp_hi = jnp.where(hi_fin, (v - jnp.where(hi_fin, hi, 0.0)) / a, big)
    h0 = jnp.max(bp_lo)
    mu_min = jnp.minimum(jnp.min(bp_hi), jnp.min(bp_lo))
    C0 = jnp.sum(jnp.where(hi_fin, a * jnp.where(hi_fin, hi, 0.0), a * v))
    S0 = -jnp.sum(jnp.where(hi_fin, 0.0, a2))
    mu_lin = jnp.where(S0 < 0, (b - C0) / jnp.where(S0 < 0, S0, -1.0), mu_min)
    l0 = jnp.minimum(mu_min, mu_lin)

    def body(_, st):
        l_, h_, mu = st
        x = jnp.clip(v - mu * a, lo, hi)
        phi = jnp.dot(a, x) - b
        l_ = jnp.where(phi > 0, mu, l_)
        h_ = jnp.where(phi > 0, h_, mu)
        free = (x > lo) & (x < hi)
        slope = jnp.sum(jnp.where(free, a2, 0.0))
        mu_n = mu + phi / jnp.where(slope > 0, slope, 1.0)
        ok = (slope > 0) & (mu_n >= l_) & (mu_n <= h_)
        mu_next = jnp.where(phi == 0.0, mu, jnp.where(ok, mu_n, 0.5 * (l_ + h_)))
        return l_, h_, mu_next

    _, _, mu = jax.lax.fori_loop(0, 20, body, (l0, h0, 0.5 * (l0 + h0)))
    return jnp.clip(v - mu * a, lo, hi)
