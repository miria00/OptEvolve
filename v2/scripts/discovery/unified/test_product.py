"""Certify product_project against an independent per-block reference (float64 numpy bisection with a verified bracket
and a final exact solve on the root's linear piece), on mixed-sign coefficients, infinite bounds, equality and slab
blocks, and degenerate ties."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, jax, jax.numpy as jnp
from uni import product_project


def ref_block(v, lo, hi, a, bl, bu):
    x0 = np.clip(v, lo, hi)
    t = np.clip(a @ x0, bl, bu)
    phi = lambda mu: a @ np.clip(v - mu * a, lo, hi) - t
    l, h = -1.0, 1.0
    while phi(l) < 0: l *= 2
    while phi(h) > 0: h *= 2
    for _ in range(400):
        mid = 0.5 * (l + h)
        if phi(mid) > 0: l = mid
        else: h = mid
    mu = 0.5 * (l + h)
    for _ in range(3):
        x = np.clip(v - mu * a, lo, hi); free = (x > lo) & (x < hi); S = np.sum(a[free] ** 2)
        if S > 0:
            mn = mu + phi(mu) / S
            if l - 1e-12 <= mn <= h + 1e-12: mu = mn
    return np.clip(v - mu * a, lo, hi)


rng = np.random.default_rng(0)
worst = 0.0
for trial in range(40):
    K = int(rng.integers(1, 60))
    sizes = rng.integers(2, 20, K)
    n = int(sizes.sum()); bid = np.repeat(np.arange(K), sizes)
    a = rng.standard_normal(n) * rng.choice([1e-2, 1, 1e2], n)
    a[np.abs(a) < 1e-3] = 0.5
    lo = np.where(rng.random(n) < 0.2, -np.inf, rng.standard_normal(n) - 1)
    hi = np.where(rng.random(n) < 0.3, np.inf, lo + rng.random(n) * 3 + 0.01)
    hi = np.where(np.isinf(lo) & np.isinf(hi), 1.0, hi)
    xf = np.where(np.isfinite(lo) & np.isfinite(hi), lo + rng.random(n) * (hi - lo), np.where(np.isfinite(lo), lo + rng.random(n), hi - rng.random(n)))
    s = np.bincount(bid, weights=a * xf, minlength=K)
    slab = rng.random(K) < 0.4
    bl = s - np.where(slab, rng.random(K), 0.0); bu = s + np.where(slab, rng.random(K), 0.0)
    v = rng.standard_normal(n) * rng.choice([0.1, 3, 100])
    if trial % 7 == 0:
        v = xf.copy()                                                      # already feasible: must return v
    x = np.asarray(product_project(jnp.asarray(v), jnp.asarray(lo), jnp.asarray(hi), jnp.asarray(a), jnp.asarray(bid),
                                   jnp.asarray(bl), jnp.asarray(bu), K))
    xr = np.concatenate([ref_block(v[bid == k], lo[bid == k], hi[bid == k], a[bid == k], bl[k], bu[k]) for k in range(K)])
    order = np.argsort(bid, kind="stable")
    err = np.max(np.abs(x[order] - xr)) / (1 + np.max(np.abs(xr)))
    sums = np.bincount(bid, weights=a * x, minlength=K)
    feas = max(np.max(np.maximum(lo - x, 0)), np.max(np.where(np.isinf(hi), 0, np.maximum(x - hi, 0))),
               np.max(np.maximum(bl - sums, 0) / (1 + np.abs(bl))), np.max(np.maximum(sums - bu, 0) / (1 + np.abs(bu))))
    worst = max(worst, err)
    if err > 1e-9 or feas > 1e-8:
        print("FAIL trial", trial, "K", K, "err %.2e feas %.2e" % (err, feas))
print("worst relative error %.2e over 40 trials" % worst)
