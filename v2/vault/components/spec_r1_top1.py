import jax
import jax.numpy as jnp

def project(v, lo, hi, a, b):
    """Euclidean projection onto {lo <= x <= hi, a'x = b} via safeguarded semismooth Newton.
    
    Uses Newton on the piecewise-linear dual function phi(mu) = a'clip(v-mu*a, lo, hi) - b.
    Each Newton step solves the current linear piece exactly. Safeguarded by bisection
    when Newton leaves the bracket. Stops when |phi| < 1e-10 (sufficient for 1e-9 x-precision).
    Typically converges in 2-5 iterations for well-conditioned problems.
    """
    a2 = a * a
    sum_a2 = jnp.sum(a2)
    
    # Unconstrained initial guess
    mu = (jnp.sum(a * v) - b) / sum_a2
    
    # Initial bracket (tight for a > 0)
    # Breakpoints: (v-lo)/a and (v-hi)/a
    bp_lo = (v - lo) / a
    bp_hi = jnp.where(jnp.isinf(hi), jnp.where(a > 0, 1e18, -1e18), (v - hi) / a)
    # For the bracket, we want mu_lo < mu* < mu_hi
    # For a > 0: mu* is between min(bp_hi) and max(bp_lo)
    # Use a safe bracket
    all_bp = jnp.minimum(bp_lo, bp_hi)
    span = jnp.maximum(jnp.max(all_bp) - jnp.min(all_bp), 1.0)
    mu_lo = jnp.min(all_bp) - span
    mu_hi = jnp.max(all_bp) + span
    
    # Clamp initial guess to bracket
    mu = jnp.clip(mu, mu_lo, mu_hi)
    
    # Compute initial phi
    x = jnp.clip(v - mu * a, lo, hi)
    phi_val = jnp.sum(a * x) - b
    
    def cond(state):
        i, _, _, phi_val = state
        return (jnp.abs(phi_val) > 1e-10) & (i < 50)
    
    def body(state):
        i, mu, bracket, phi_val = state
        mu_lo, mu_hi = bracket
        x = jnp.clip(v - mu * a, lo, hi)
        free = (x > lo) & (x < hi)
        S = jnp.sum(a2 * free)
        S_safe = jnp.maximum(S, 1e-30)
        mu_newton = mu + phi_val / S_safe
        mu_bisect = 0.5 * (mu_lo + mu_hi)
        in_bracket = (mu_newton >= mu_lo) & (mu_newton <= mu_hi)
        mu_new = jnp.where(in_bracket, mu_newton, mu_bisect)
        
        mu_lo_new = jnp.where(phi_val > 0, mu, mu_lo)
        mu_hi_new = jnp.where(phi_val > 0, mu_hi, mu)
        
        x_new = jnp.clip(v - mu_new * a, lo, hi)
        phi_new = jnp.sum(a * x_new) - b
        
        return i + 1, mu_new, (mu_lo_new, mu_hi_new), phi_new
    
    _, mu, _, _ = jax.lax.while_loop(cond, body, (0, mu, (mu_lo, mu_hi), phi_val))
    return jnp.clip(v - mu * a, lo, hi)