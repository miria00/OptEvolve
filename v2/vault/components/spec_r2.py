import jax
import jax.numpy as jnp

def project(v, lo, hi, a, b):
    """Safeguarded semismooth Newton: 2 unrolled iters + while_loop fallback.
    Precomputes a2=a*a to save n muls/iter; no S guard (inf/NaN rejected by
    bracket check, triggering bisection). phi is always non-increasing since
    its slope is -sum(a_i^2 * free_i) <= 0 regardless of sign of a."""
    a2 = a * a
    # Newton iter 1 (mu=0 => raw=v)
    c = jnp.clip(v, lo, hi)
    p = jnp.dot(a, c) - b
    S = jnp.dot(a2, (v > lo) & (v < hi))
    mc = p / S
    mlo = jnp.where(p > 0, 0., -1e10)
    mhi = jnp.where(p > 0, 1e10, 0.)
    m1 = jnp.where((mc >= mlo) & (mc <= mhi), mc, .5 * (mlo + mhi))
    # Newton iter 2
    r = v - m1 * a
    c = jnp.clip(r, lo, hi)
    p = jnp.dot(a, c) - b
    S = jnp.dot(a2, (r > lo) & (r < hi))
    mc = m1 + p / S
    mlo = jnp.where(p > 0, m1, mlo)
    mhi = jnp.where(p > 0, mhi, m1)
    m2 = jnp.where((mc >= mlo) & (mc <= mhi), mc, .5 * (mlo + mhi))
    # Final evaluation & convergence check
    r = v - m2 * a
    c = jnp.clip(r, lo, hi)
    p = jnp.dot(a, c) - b
    done = jnp.abs(p) <= 1e-10
    # Fallback while_loop for difficult cases
    def cond(s):
        i, _, _, _, q = s
        return (jnp.abs(q) > 1e-10) & (i < 100)
    def body(s):
        i, m, lo_, hi_, q = s
        r = v - m * a
        S = jnp.dot(a2, (r > lo) & (r < hi))
        nc = m + q / S
        lo_ = jnp.where(q > 0, m, lo_)
        hi_ = jnp.where(q > 0, hi_, m)
        mn = jnp.where((nc >= lo_) & (nc <= hi_), nc, .5 * (lo_ + hi_))
        return (i + 1, mn, lo_, hi_, jnp.dot(a, jnp.clip(v - mn * a, lo, hi)) - b)
    def slow():
        _, mf, _, _, _ = jax.lax.while_loop(cond, body, (2, m2, mlo, mhi, p))
        return jnp.clip(v - mf * a, lo, hi)
    return jax.lax.cond(done, lambda: c, slow)