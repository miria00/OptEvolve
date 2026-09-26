import jax
import jax.numpy as jnp

def project(v, lo, hi, a, b):
    """Euclidean projection onto {lo<=x<=hi, a'x=b} via safeguarded semismooth Newton.
    Carries x in loop state to eliminate redundant clip per iteration.
    Tight bracket from breakpoints; bisection fallback ensures global convergence."""
    a2 = a * a

    # Unconstrained initial guess
    mu = (jnp.dot(a, v) - b) / jnp.dot(a, a)

    # Tight bracket: root lies in [min(all breakpoints), max(all breakpoints)]
    bp_lo = (v - lo) / a
    bp_hi = jnp.where(jnp.isinf(hi), jnp.where(a > 0, 1e18, -1e18), (v - hi) / a)
    mu_lo = jnp.minimum(jnp.min(bp_lo), jnp.min(bp_hi))
    mu_hi = jnp.maximum(jnp.max(bp_lo), jnp.max(bp_hi))
    mu = jnp.clip(mu, mu_lo, mu_hi)

    # Initial x and phi (carried in state to avoid recomputation)
    x = jnp.clip(v - mu * a, lo, hi)
    phi_val = jnp.dot(a, x) - b

    def cond(state):
        i, _, _, _, p = state
        return (jnp.abs(p) > 1e-10) & (i < 40)

    def body(state):
        i, mu, (mu_lo, mu_hi), x, p = state
        # x is carried from previous iteration (saves one clip)
        free = (x > lo) & (x < hi)
        S = jnp.sum(a2 * free)
        S_safe = jnp.where(S > 1e-30, S, 1.0)
        mu_newton = mu + p / S_safe
        mu_bisect = 0.5 * (mu_lo + mu_hi)
        mu_new = jnp.where((mu_newton >= mu_lo) & (mu_newton <= mu_hi), mu_newton, mu_bisect)
        mu_lo_new = jnp.where(p > 0, mu, mu_lo)
        mu_hi_new = jnp.where(p > 0, mu_hi, mu)
        x_new = jnp.clip(v - mu_new * a, lo, hi)
        p_new = jnp.dot(a, x_new) - b
        return i + 1, mu_new, (mu_lo_new, mu_hi_new), x_new, p_new

    _, mu, _, x, _ = jax.lax.while_loop(cond, body, (0, mu, (mu_lo, mu_hi), x, phi_val))
    return x