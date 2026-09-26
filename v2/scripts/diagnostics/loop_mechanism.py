"""Is the device-loop penalty per certificate-check, or per outer-loop trip?

Two explanations are on the table and both cannot be right. My earlier
microbenchmark had ZERO certificate evaluations and still showed 2.02x, which
rules out a pure per-check cost. The workflow's HLO evidence showed no
unrolling difference at K=256, which rules out the unrolling story. The
reconciling hypothesis is a fixed cost per OUTER TRIP, which in the solver
coincides with per-check because a check fires once per trip.

This separates them: vary the number of outer trips at fixed total iterations,
with and without a certificate in the loop body.
"""
import sys, time, json
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

def jax_step(x, u, y, t, s, lam, inv, rho):
    h, w = x.shape
    u0, u1 = u[0], u[1]
    Z1 = jnp.zeros((1, w)); Z2 = jnp.zeros((h, 1))
    v0 = jnp.concatenate([Z1, u0[:-1]], 0) - jnp.concatenate([u0[:-1], Z1], 0)
    v1 = jnp.concatenate([Z2, u1[:, :-1]], 1) - jnp.concatenate([u1[:, :-1], Z2], 1)
    pr = ((x - t*(v0+v1)) + t*y) * inv
    bar = 2.*pr - x
    d0 = jnp.concatenate([bar[1:] - bar[:-1], Z1], 0)
    d1 = jnp.concatenate([bar[:, 1:] - bar[:, :-1], Z2], 1)
    a = jnp.clip(u0 + s*d0, -lam, lam); b = jnp.clip(u1 + s*d1, -lam, lam)
    return ((1.-rho)*x + rho*pr,
            jnp.stack([(1.-rho)*u0+rho*a, (1.-rho)*u1+rho*b]))

def timeit(fn, args, reps=15, warm=4):
    for _ in range(warm): jax.block_until_ready(fn(*args))
    t0 = time.perf_counter()
    for _ in range(reps): jax.block_until_ready(fn(*args))
    return (time.perf_counter()-t0)/reps

t, s, lam, rho = 0.05, 1.0, 0.1, 1.9
inv = 1.0/(1.0+t); TOT = 512
rows = []
for n in (128, 256):
    rng = np.random.default_rng(0)
    y = jnp.asarray(rng.random((n, n))); x = jnp.asarray(rng.random((n, n)))
    u = jnp.asarray(rng.standard_normal((2, n, n))*0.1)
    step = lambda a_, b_: jax_step(a_, b_, y, t, s, lam, inv, rho)
    gap = lambda a_, b_: jnp.sum(a_*a_) + jnp.sum(b_*b_)   # a cheap stand-in certificate
    for K in (16, 32, 64, 128, 256, 512):
        trips = TOT // K
        for traced in (False, True):
            for cert in (False, True):
                def body_factory(traced=traced, cert=cert, K=K):
                    def outer(x0, u0_):
                        def body(c):
                            xx, uu, k = c
                            if traced:
                                end = jnp.minimum(k+K, TOT)
                                xx, uu = jax.lax.fori_loop(k, end, lambda i, d: step(*d), (xx, uu))
                            else:
                                xx, uu = jax.lax.fori_loop(0, K, lambda i, d: step(*d), (xx, uu))
                                end = k + K
                            if cert:
                                g = gap(xx, uu)
                                xx = xx + 0.0*g          # keep the certificate live
                            return (xx, uu, end)
                        return jax.lax.while_loop(lambda c: c[2] < TOT, body, (x0, u0_, jnp.int32(0)))[:2]
                    return outer
                dt = timeit(jax.jit(body_factory()), (x, u))
                rows.append({"n": n, "K": K, "trips": trips, "traced": traced,
                             "cert": cert, "ns_per_iter": dt/TOT*1e9})
    print(f"n={n} done", flush=True)

print(f"\n{'n':>5s} {'K':>5s} {'trips':>6s} | " +
      " ".join(f"{lbl:>12s}" for lbl in ("static", "static+cert", "traced", "traced+cert")) +
      f" | {'traced-static':>13s} {'cert cost':>10s}")
for n in (128, 256):
    for K in (16, 32, 64, 128, 256, 512):
        g = lambda tr, ce: next(r["ns_per_iter"] for r in rows
                                if r["n"]==n and r["K"]==K and r["traced"]==tr and r["cert"]==ce)
        a, b, c, d = g(False,False), g(False,True), g(True,False), g(True,True)
        trips = TOT//K
        print(f"{n:5d} {K:5d} {trips:6d} | {a:12.0f} {b:12.0f} {c:12.0f} {d:12.0f} "
              f"| {c-a:13.0f} {b-a:10.0f}")
Path('/home/miria/OptEvolve/v2/results/finish_20260910/loop_mechanism.json').write_text(json.dumps(rows, indent=1))
print("\nIf (traced - static) scales with TRIPS at fixed total iterations, the cost")
print("is per outer trip. If it is flat in trips, it is per iteration.")
print("The 'cert cost' column isolates the certificate, which is a separate term.")
