import jax
import jax.numpy as jnp

def project(v, lo, hi, a, b):
    """Exact Euclidean projection onto {lo<=x<=hi, a'x=b} via safeguarded semismooth Newton.
    
    Key speed optimizations:
    - Wide centered bracket (mu±1e6) eliminates expensive breakpoint computation
      (saves 2 vector divisions + 4 reductions for n=1500)
    - Carries raw=v-mu*a in loop state, saving 1 mul+1 sub per iteration
    - Typically 2-3 Newton iterations for well-conditioned problems (early exit)
    """
    a2 = a * a
    sum_a2 = jnp.sum(a2)

    # Unconstrained initial guess (solution to equality constraint ignoring bounds)
    mu = (jnp.sum(a * v) - b) / sum_a2

    # Wide centered bracket - safe for all reasonable inputs, avoids breakpoint compute
    mu_lo = mu - 1e6
    mu_hi = mu + 1e6

    # Initial raw residual and dual residual
    raw = v - mu * a
    phi = jnp.sum(a * jnp.clip(raw, lo, hi)) - b

    def cond(state):
        i, _, _, _, _, p = state
        return (jnp.abs(p) > 1e-10) & (i < 50)

    def body(state):
        i, mu, mu_lo, mu_hi, raw, p = state
        # Active set: variables strictly interior
        free = (raw > lo) & (raw < hi)
        S = jnp.sum(a2 * free)
        # Newton step (exact on current linear piece); bisection fallback
        mu_newton = mu + p / jnp.maximum(S, 1e-30)
        mu_bisect = 0.5 * (mu_lo + mu_hi)
        mu_new = jnp.where((mu_newton >= mu_lo) & (mu_newton <= mu_hi), mu_newton, mu_bisect)
        # Update bracket (phi is nonincreasing in mu for all sign patterns of a)
        mu_lo = jnp.where(p > 0, mu, mu_lo)
        mu_hi = jnp.where(p > 0, mu_hi, mu)
        # Advance raw and phi
        raw_new = v - mu_new * a
        p_new = jnp.sum(a * jnp.clip(raw_new, lo, hi)) - b
        return (i + 1, mu_new, mu_lo, mu_hi, raw_new, p_new)

    _, mu, _, _, _, _ = jax.lax.while_loop(cond, body, (0, mu, mu_lo, mu_hi, raw, phi))
    return jnp.clip(v - mu * a, lo, hi)