"""Our arm: batched FISTA over the simplex, device and precision as genes.

Everything before the first iteration (Gram, correlation, Lipschitz constant,
host-to-device transfer, compilation) is inside the clock, because the
baselines pay their setup too.  The reported time is time to a CERTIFIED
worst-pixel Frank-Wolfe gap, recomputed in float64 on the host from the
returned X, never the solver's own stopping rule.
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fcls_core import fw_gap_and_feas


def build(device, dtype):
    os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu" if device == "gpu" else "cpu")
    # The 4090 and the H100 both carry a resident vLLM server, so the default
    # 75 percent preallocation fails.  Allocate on demand instead.
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.28")
    import jax, jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    dev = jax.devices("gpu" if device == "gpu" else "cpu")[0]
    dt = jnp.float32 if dtype == "f32" else jnp.float64

    def proj_simplex(V):
        n = V.shape[0]
        U = jnp.sort(V, axis=0)[::-1]
        css = jnp.cumsum(U, axis=0) - 1.0
        ind = jnp.arange(1, n + 1, dtype=V.dtype).reshape(-1, 1)
        cond = (U - css / ind) > 0
        rho = n - 1 - jnp.argmax(cond[::-1], axis=0)
        theta = jnp.take_along_axis(css, rho[None, :], axis=0)[0] / (rho + 1).astype(V.dtype)
        return jnp.maximum(V - theta, 0.0)

    @jax.jit
    def chunk(X, Z, t, DtD, DtY, step, K):
        def body(i, s):
            X, Z, t = s
            G = DtD @ Z - DtY
            Xn = proj_simplex(Z - step * G)
            tn = 0.5 * (1.0 + jnp.sqrt(1.0 + 4.0 * t * t))
            Zn = Xn + ((t - 1.0) / tn) * (Xn - X)
            return (Xn, Zn, tn)
        return jax.lax.fori_loop(0, K, body, (X, Z, t))

    return jax, jnp, dev, dt, chunk


def run(D, Y, device, dtype, targets, max_iter=200000, check_every=50, budget_s=600.0):
    jax, jnp, dev, dt, chunk = build(device, dtype)
    n, m = D.shape[1], Y.shape[1]
    hit = {}
    t0 = time.perf_counter()
    with jax.default_device(dev):
        Dd = jnp.asarray(D, dtype=dt)
        Yd = jnp.asarray(Y, dtype=dt)
        DtD = Dd.T @ Dd
        DtY = Dd.T @ Yd
        # Lipschitz constant by power iteration on DtD, x1.02 guard.
        v = jnp.ones((n, 1), dtype=dt)
        for _ in range(60):
            v = DtD @ v
            v = v / jnp.linalg.norm(v)
        L = float(jnp.vdot(v, DtD @ v).real) * 1.02
        step = jnp.asarray(1.0 / L, dtype=dt)
        X = jnp.full((n, m), 1.0 / n, dtype=dt)
        Z = X
        t = jnp.asarray(1.0, dtype=dt)
        X, Z, t = chunk(X, Z, t, DtD, DtY, step, 1)   # compile inside the clock
        X.block_until_ready()
        it = 1
        while it < max_iter and (time.perf_counter() - t0) < budget_s:
            X, Z, t = chunk(X, Z, t, DtD, DtY, step, check_every)
            X.block_until_ready()
            it += check_every
            el = time.perf_counter() - t0
            Xh = np.asarray(X, dtype=np.float64)
            g, gmed, feas, fsum = fw_gap_and_feas(D, Y, Xh)
            for tg in targets:
                if str(tg) not in hit and g <= tg and feas <= max(tg, 1e-9):
                    hit[str(tg)] = {"t": el, "iters": it, "gap": g, "feas": feas, "obj": fsum}
            if all(str(tg) in hit for tg in targets):
                break
    return {"device": device, "dtype": dtype, "L": L, "iters": it,
            "final_gap": g, "final_gap_median": gmed, "feas": feas, "obj": fsum,
            "wall_total": time.perf_counter() - t0, "hit": hit}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mat", default="/home/miria/data/hsi/Cuprite.mat")
    ap.add_argument("--device", default="gpu")
    ap.add_argument("--dtype", default="f64")
    ap.add_argument("--pixels", type=int, default=0)
    ap.add_argument("--budget", type=float, default=600.0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    import scipy.io as sio
    d = sio.loadmat(a.mat)
    D = np.asarray(d["D"], dtype=np.float64)
    Y = np.asarray(d["Y"], dtype=np.float64)
    if a.pixels:
        Y = Y[:, : a.pixels]
    r = run(D, Y, a.device, a.dtype, [1e-3, 1e-4, 1e-6, 1e-8], budget_s=a.budget)
    r["pixels"] = Y.shape[1]
    print(json.dumps(r, indent=1))
    if a.out:
        open(a.out, "w").write(json.dumps(r, indent=1))
