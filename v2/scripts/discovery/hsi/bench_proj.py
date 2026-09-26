"""Per-device cost of the simplex projection, measured properly.

Nine repetitions, median reported with min and max, because an earlier
single-shot reading of this same quantity was taken while another job held the
GPU and inverted the ordering.  The machine's own load is recorded beside the
timing.
"""
import json, os, sys, time
import numpy as np

n, m, REPS, INNER = 498, 2000, 9, 50
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
dev = jax.devices("gpu")[0] if os.environ.get("DEVKIND", "gpu") == "gpu" else jax.devices("cpu")[0]

V = np.random.default_rng(0).standard_normal((n, m))
M = np.random.default_rng(1).standard_normal((n, n))


def mk_bisect(nb):
    def f(V):
        vmax = jnp.max(V, axis=0)
        def st(i, b):
            lo, hi = b
            mid = 0.5 * (lo + hi)
            gt = jnp.sum(jnp.maximum(V - mid, 0.0), axis=0) > 1.0
            return (jnp.where(gt, mid, lo), jnp.where(gt, hi, mid))
        lo, hi = jax.lax.fori_loop(0, nb, st, (vmax - 1.0, vmax))
        return jnp.maximum(V - 0.5 * (lo + hi), 0.0)
    return f


def psort(V):
    U = jnp.sort(V, axis=0)[::-1]
    css = jnp.cumsum(U, axis=0) - 1.0
    ind = jnp.arange(1, n + 1, dtype=V.dtype).reshape(-1, 1)
    r = n - 1 - jnp.argmax(((U - css / ind) > 0)[::-1], axis=0)
    th = jnp.take_along_axis(css, r[None, :], axis=0)[0] / (r + 1).astype(V.dtype)
    return jnp.maximum(V - th, 0.0)


out = {"device": str(dev), "loadavg": os.getloadavg()[0], "reps": REPS}
with jax.default_device(dev):
    Vd, Md = jnp.asarray(V), jnp.asarray(M)
    cases = [("gemm", None), ("sort", psort), ("bisect32", mk_bisect(32)),
             ("bisect48", mk_bisect(48)), ("bisect64", mk_bisect(64))]
    for nm, fn in cases:
        if nm == "gemm":
            f = jax.jit(lambda A, X: jax.lax.fori_loop(0, INNER, lambda i, s: A @ s, X))
            call = lambda: f(Md, Vd)
        else:
            f = jax.jit(lambda X, fn=fn: jax.lax.fori_loop(0, INNER, lambda i, s: fn(s), X))
            call = lambda: f(Vd)
        r = call(); r.block_until_ready()
        ts = []
        for _ in range(REPS):
            t = time.perf_counter(); r = call(); r.block_until_ready()
            ts.append((time.perf_counter() - t) / INNER)
        ts = sorted(ts)
        err = None
        if fn is not None:
            P = np.asarray(fn(Vd), dtype=np.float64)
            err = max(float(abs(P.sum(0) - 1).max()), float(-P.min()))
        out[nm] = {"median_s": ts[len(ts)//2], "min_s": ts[0], "max_s": ts[-1],
                   "spread": ts[-1] / ts[0], "feas_err": err}
print(json.dumps(out, indent=1))
