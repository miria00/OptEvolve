"""Our arm, v2: batched FISTA over the simplex with two genes added.

  restart  (wrapper gene)  adaptive gradient restart, which is what a
                           poorly conditioned FISTA needs and what the
                           genome's restart wrapper exists to supply.
  restrict (layer 4)       a pixel whose certified Frank-Wolfe gap is already
                           under target is retired from the batch, and the
                           batch is compacted.  This is the working-set move:
                           the heuristic is that a retired pixel stays
                           converged, and the full recheck at the end is the
                           re-admission pass that makes it exact.

Timing includes Gram, correlation, Lipschitz estimate, transfer and
compilation.  Acceptance is a float64 host recomputation of the worst-pixel
relative Frank-Wolfe gap over ALL pixels, retired ones included.
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fcls_core import fw_gap_and_feas


def build(device, dtype):
    os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu" if device == "gpu" else "cpu")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.28")
    import jax, jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    dev = jax.devices("gpu" if device == "gpu" else "cpu")[0]
    dt = jnp.float32 if dtype == "f32" else jnp.float64

    def proj(V):
        n = V.shape[0]
        U = jnp.sort(V, axis=0)[::-1]
        css = jnp.cumsum(U, axis=0) - 1.0
        ind = jnp.arange(1, n + 1, dtype=V.dtype).reshape(-1, 1)
        rho = n - 1 - jnp.argmax(((U - css / ind) > 0)[::-1], axis=0)
        theta = jnp.take_along_axis(css, rho[None, :], axis=0)[0] / (rho + 1).astype(V.dtype)
        return jnp.maximum(V - theta, 0.0)

    @jax.jit
    def chunk(X, Z, t, DtD, DtY, step, K):
        def body(i, s):
            X, Z, t = s
            G = DtD @ Z - DtY
            Xn = proj(Z - step * G)
            # adaptive gradient restart: restart where the momentum direction
            # and the step disagree, per pixel.
            bad = (jnp.sum((Z - Xn) * (Xn - X), axis=0) > 0)
            tn = jnp.where(bad, 1.0, 0.5 * (1.0 + jnp.sqrt(1.0 + 4.0 * t * t)))
            beta = jnp.where(bad, 0.0, (t - 1.0) / tn)
            Zn = Xn + beta * (Xn - X)
            return (Xn, Zn, tn)
        return jax.lax.fori_loop(0, K, body, (X, Z, t))

    @jax.jit
    def gaps(X, DtD, DtY, f):
        G = DtD @ X - DtY
        gap = jnp.sum(G * X, axis=0) - jnp.min(G, axis=0)
        return gap / (1.0 + jnp.abs(f))

    return jax, jnp, dev, dt, chunk, gaps


def run(D, Y, device, dtype, targets, max_iter=400000, check_every=100, budget_s=600.0):
    jax, jnp, dev, dt, chunk, gaps = build(device, dtype)
    n, m = D.shape[1], Y.shape[1]
    hit, log = {}, []
    t0 = time.perf_counter()
    with jax.default_device(dev):
        Dd = jnp.asarray(D, dtype=dt)
        Yd = jnp.asarray(Y, dtype=dt)
        DtD = Dd.T @ Dd
        DtY_full = Dd.T @ Yd
        yy = jnp.sum(Yd * Yd, axis=0)
        v = jnp.ones((n, 1), dtype=dt)
        for _ in range(60):
            v = DtD @ v
            v = v / jnp.linalg.norm(v)
        L = float(jnp.vdot(v, DtD @ v).real) * 1.02
        step = jnp.asarray(1.0 / L, dtype=dt)

        Xfull = np.full((n, m), 1.0 / n, dtype=np.float64)
        alive = np.arange(m)
        X = jnp.full((n, m), 1.0 / n, dtype=dt)
        Z, t = X, jnp.ones((m,), dtype=dt)
        DtY, yyv = DtY_full, yy
        tgt_i, it = 0, 0
        X, Z, t = chunk(X, Z, t, DtD, DtY, step, 1)
        X.block_until_ready(); it += 1

        while it < max_iter and (time.perf_counter() - t0) < budget_s and tgt_i < len(targets):
            X, Z, t = chunk(X, Z, t, DtD, DtY, step, check_every)
            X.block_until_ready(); it += check_every
            # per-pixel objective for the alive set, for the relative gap
            R = Dd @ X - Yd[:, alive] if alive.size != m else Dd @ X - Yd
            f = 0.5 * jnp.sum(R * R, axis=0)
            g = np.asarray(gaps(X, DtD, DtY, f), dtype=np.float64)
            Xfull[:, alive] = np.asarray(X, dtype=np.float64)
            el = time.perf_counter() - t0
            # full, honest recheck over every pixel including retired ones
            gw, gmed, feas, fsum = fw_gap_and_feas(D, Y, Xfull)
            log.append({"it": it, "t": el, "alive": int(alive.size), "worst": gw})
            while tgt_i < len(targets) and gw <= targets[tgt_i] and feas <= 1e-9:
                hit[str(targets[tgt_i])] = {"t": el, "iters": it, "gap": gw,
                                            "feas": feas, "obj": fsum,
                                            "alive": int(alive.size)}
                tgt_i += 1
            if tgt_i >= len(targets):
                break
            # retire pixels already under the NEXT target with a safety margin
            keep = np.where(g > targets[tgt_i] * 0.1)[0]
            if keep.size and keep.size < alive.size * 0.9:
                alive = alive[keep]
                ki = jnp.asarray(keep)
                X = jnp.take(X, ki, axis=1)
                Z = jnp.take(Z, ki, axis=1)
                DtY = jnp.take(DtY_full, jnp.asarray(alive), axis=1)
                t = jnp.ones((alive.size,), dtype=dt)
            elif keep.size == 0:
                break
    return {"device": device, "dtype": dtype, "L": L, "iters": it, "pixels": m,
            "final_worst_gap": gw, "final_median_gap": gmed, "feas": feas,
            "obj": fsum, "wall_total": time.perf_counter() - t0,
            "hit": hit, "trace": log[-6:]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mat", default="/home/miria/data/hsi/Cuprite.mat")
    ap.add_argument("--device", default="gpu"); ap.add_argument("--dtype", default="f64")
    ap.add_argument("--pixels", type=int, default=0)
    ap.add_argument("--budget", type=float, default=600.0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    import scipy.io as sio
    d = sio.loadmat(a.mat)
    D = np.asarray(d["D"], dtype=np.float64); Y = np.asarray(d["Y"], dtype=np.float64)
    if a.pixels: Y = Y[:, : a.pixels]
    r = run(D, Y, a.device, a.dtype, [1e-3, 1e-4, 1e-6, 1e-8], budget_s=a.budget)
    print(json.dumps(r, indent=1))
    if a.out: open(a.out, "w").write(json.dumps(r, indent=1))
