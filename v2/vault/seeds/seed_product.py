"""Seed for synthesizing the PRODUCT projection: exact projection onto the product over blocks k of
{lo <= x <= hi, a_k'x in [b_lo_k, b_hi_k]} (blocks given by integer ids bid in 0..K-1, K static). Library version:
bracket from breakpoints, bisection, safeguarded Newton polish, all blocks at once with segment ops."""
import jax
import jax.numpy as jnp


def project(v, lo, hi, a, bid, b_lo, b_hi, K, iters=60, expand=40, polish=2):
    """Exact projection onto the product over blocks k of {lo <= x <= hi, a_k'x in [b_lo_k, b_hi_k]}, all blocks at once.

    Per block, phi_k(mu) = a_k'clip(v - mu a_k, lo, hi) is nonincreasing in mu for coefficients of ANY sign (its slope
    is -sum over free coordinates of a^2). The target is the band point nearest a_k'clip(v), so mu = 0 is the answer
    when that is already inside the band. Bracket: the finite breakpoints, widened by doubling until the signs are
    right (unbounded sides make phi linear beyond the last breakpoint); then bisection; then safeguarded Newton steps
    that land exactly on the root's linear piece.
    """
    seg = lambda z: jax.ops.segment_sum(z, bid, num_segments=K)
    target = jnp.clip(seg(a * jnp.clip(v, lo, hi)), b_lo, b_hi)

    def phi(mu):
        return seg(a * jnp.clip(v - mu[bid] * a, lo, hi)) - target

    bp1 = jnp.where(jnp.isfinite(lo), (v - jnp.where(jnp.isfinite(lo), lo, 0.0)) / a, jnp.nan)
    bp2 = jnp.where(jnp.isfinite(hi), (v - jnp.where(jnp.isfinite(hi), hi, 0.0)) / a, jnp.nan)
    mins = jnp.minimum(jnp.where(jnp.isnan(bp1), jnp.inf, bp1), jnp.where(jnp.isnan(bp2), jnp.inf, bp2))
    maxs = jnp.maximum(jnp.where(jnp.isnan(bp1), -jnp.inf, bp1), jnp.where(jnp.isnan(bp2), -jnp.inf, bp2))
    mlo = jax.ops.segment_min(mins, bid, num_segments=K)
    mhi = jax.ops.segment_max(maxs, bid, num_segments=K)
    width = 1.0 + (mhi - mlo)
    l0, h0 = mlo - width, mhi + width

    def widen(_, st):
        l_, h_ = st
        w = h_ - l_
        return jnp.where(phi(l_) < 0, l_ - w, l_), jnp.where(phi(h_) > 0, h_ + w, h_)

    l_, h_ = jax.lax.fori_loop(0, expand, widen, (l0, h0))

    def bisect(_, st):
        l_, h_ = st
        mid = 0.5 * (l_ + h_)
        pos = phi(mid) > 0
        return jnp.where(pos, mid, l_), jnp.where(pos, h_, mid)

    l_, h_ = jax.lax.fori_loop(0, iters, bisect, (l_, h_))
    mu = 0.5 * (l_ + h_)
    a2 = a * a

    def newton(_, mu):
        x = jnp.clip(v - mu[bid] * a, lo, hi)
        r = seg(a * x) - target
        slope = seg(jnp.where((x > lo) & (x < hi), a2, 0.0))
        mu_n = mu + r / jnp.where(slope > 0, slope, 1.0)
        ok = (slope > 0) & (mu_n >= l_) & (mu_n <= h_)
        return jnp.where(ok, mu_n, mu)

    mu = jax.lax.fori_loop(0, polish, newton, mu)
    return jnp.clip(v - mu[bid] * a, lo, hi)
