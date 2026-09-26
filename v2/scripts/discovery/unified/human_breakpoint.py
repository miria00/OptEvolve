import jax
import jax.numpy as jnp


def project(v, lo, hi, a, b):
    """Exact projection onto {lo <= x <= hi, a'x = b} by sorting the 2n breakpoints of the multiplier (O(n log n)).

    x_i(mu) = clip(v_i - mu a_i, lo_i, hi_i). phi(mu) = sum a_i x_i(mu) is piecewise linear and nonincreasing.
    Coordinate i is at hi for mu <= (v_i - hi_i)/a_i, linear in between, at lo for mu >= (v_i - lo_i)/a_i.
    Sweep the sorted breakpoints keeping phi = C + S*mu, find the segment that contains b, solve it linearly.
    Written by hand as the human-authored arm of the operator-library experiment.
    """
    n = v.shape[0]
    big = 1e300
    hi_fin = jnp.isfinite(hi)
    # initial state for mu below every finite breakpoint
    C0 = jnp.sum(jnp.where(hi_fin, a * hi, a * v))
    S0 = jnp.sum(jnp.where(hi_fin, 0.0, -a * a))
    bp_hi = jnp.where(hi_fin, (v - jnp.where(hi_fin, hi, 0.0)) / a, -big)
    dC_hi = jnp.where(hi_fin, a * v - a * jnp.where(hi_fin, hi, 0.0), 0.0)
    dS_hi = jnp.where(hi_fin, -a * a, 0.0)
    bp_lo = (v - lo) / a
    dC_lo = a * lo - a * v
    dS_lo = a * a
    bp = jnp.concatenate([bp_hi, bp_lo])
    dC = jnp.concatenate([dC_hi, dC_lo])
    dS = jnp.concatenate([dS_hi, dS_lo])
    order = jnp.argsort(bp)
    bp, dC, dS = bp[order], dC[order], dS[order]
    Cpre = C0 + jnp.concatenate([jnp.zeros(1), jnp.cumsum(dC)])     # state after the first e events, e = 0..2n
    Spre = S0 + jnp.concatenate([jnp.zeros(1), jnp.cumsum(dS)])
    phi_at = Cpre[:-1] + Spre[:-1] * bp                               # phi at each breakpoint (continuous)
    cnt = jnp.sum(phi_at >= b)
    Cs, Ss = Cpre[cnt], Spre[cnt]
    mu_lin = (b - Cs) / jnp.where(Ss == 0.0, -1.0, Ss)
    mu_flat = jnp.where(cnt > 0, bp[jnp.maximum(cnt - 1, 0)], bp[0])
    mu = jnp.where(Ss == 0.0, mu_flat, mu_lin)
    return jnp.clip(v - mu * a, lo, hi)
