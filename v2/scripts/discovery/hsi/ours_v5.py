"""Our arm, final: relocation (ADMM on one shared inverse) + the certified
box-and-hyperplane projection + a working set over pixels.

Checking is itself priced.  The per-pixel gap is computed on the device and
only confirmed in float64 on the host when the device says a target is within
reach, so the acceptance check does not dominate the clock it is measuring.
The confirmation is over EVERY pixel, retired ones included.
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fcls_core import fw_gap_and_feas


def build(device, dtype, nbis):
    os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu" if device == "gpu" else "cpu")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.28")
    import jax, jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    dev = jax.devices("gpu" if device == "gpu" else "cpu")[0]
    dt = jnp.float32 if dtype == "f32" else jnp.float64

    def proj(V):
        """Exact projection onto the unit simplex, per column.

        Measured on both devices against a safeguarded bisection on the dual
        scalar: sort wins on the RTX 4090 (0.454 ms against 1.219 ms at 48
        bisection steps) and on the CPU (13.3 ms against 58.3 ms), so the
        bisection form was measured and declined.  The operator gene is
        settled here by measurement, not by the assumption that a GPU
        dislikes sorting.
        """
        n = V.shape[0]
        U = jnp.sort(V, axis=0)[::-1]
        css = jnp.cumsum(U, axis=0) - 1.0
        ind = jnp.arange(1, n + 1, dtype=V.dtype).reshape(-1, 1)
        r = n - 1 - jnp.argmax(((U - css / ind) > 0)[::-1], axis=0)
        theta = jnp.take_along_axis(css, r[None, :], axis=0)[0] / (r + 1).astype(V.dtype)
        return jnp.maximum(V - theta, 0.0)

    @jax.jit
    def chunk(Z, U, M, B, rho, K, alpha):
        def body(i, s):
            Z, U = s
            X = M @ (B + rho * (Z - U))
            XA = alpha * X + (1.0 - alpha) * Z
            Zn = proj(XA + U)
            return (Zn, U + XA - Zn)
        return jax.lax.fori_loop(0, K, body, (Z, U))

    @jax.jit
    def devgap(Z, DtD, B, yy):
        G = DtD @ Z - B
        quad = jnp.sum(Z * (DtD @ Z), axis=0)
        f = 0.5 * (quad - 2.0 * jnp.sum(B * Z, axis=0) + yy)
        return (jnp.sum(G * Z, axis=0) - jnp.min(G, axis=0)) / (1.0 + jnp.abs(f))

    return jax, jnp, dev, dt, chunk, devgap


def run(D, Y, device, dtype, targets, rho=1.0, alpha=1.6, nbis=48,
        max_iter=2000000, check_every=500, budget_s=1800.0):
    jax, jnp, dev, dt, chunk, devgap = build(device, dtype, nbis)
    n, m = D.shape[1], Y.shape[1]
    hit, log = {}, []
    t0 = time.perf_counter()
    with jax.default_device(dev):
        Dd = jnp.asarray(D, dtype=dt); Yd = jnp.asarray(Y, dtype=dt)
        DtD = Dd.T @ Dd
        Bfull = Dd.T @ Yd
        yyf = jnp.sum(Yd * Yd, axis=0)
        M = jnp.linalg.inv(DtD + rho * jnp.eye(n, dtype=dt))
        Zfull = np.full((n, m), 1.0 / n, dtype=np.float64)
        alive = np.arange(m)
        Z = jnp.full((n, m), 1.0 / n, dtype=dt); U = jnp.zeros((n, m), dtype=dt)
        B, yy = Bfull, yyf
        rh = jnp.asarray(rho, dtype=dt); al = jnp.asarray(alpha, dtype=dt)
        Z, U = chunk(Z, U, M, B, rh, 1, al); Z.block_until_ready()
        it, tgt_i = 1, 0
        while it < max_iter and (time.perf_counter() - t0) < budget_s and tgt_i < len(targets):
            Z, U = chunk(Z, U, M, B, rh, check_every, al)
            gdev = np.asarray(devgap(Z, DtD, B, yy), dtype=np.float64)
            it += check_every
            el = time.perf_counter() - t0
            if gdev.max() <= targets[tgt_i] * 2.0 or (it % (check_every * 20) == 1):
                Zfull[:, alive] = np.asarray(Z, dtype=np.float64)
                gw, gmed, feas, fsum = fw_gap_and_feas(D, Y, Zfull)
                log.append({"it": it, "t": round(el, 2), "alive": int(alive.size),
                            "worst": gw, "med": gmed})
                while tgt_i < len(targets) and gw <= targets[tgt_i] and feas <= 1e-9:
                    hit[str(targets[tgt_i])] = {"t": el, "iters": it, "gap": gw,
                                                "feas": feas, "obj": fsum,
                                                "alive": int(alive.size)}
                    tgt_i += 1
                if tgt_i >= len(targets):
                    break
            keep = np.where(gdev > targets[tgt_i] * 0.01)[0]
            if 0 < keep.size <= alive.size * 0.7:
                Zfull[:, alive] = np.asarray(Z, dtype=np.float64)
                alive = alive[keep]; ki = jnp.asarray(keep)
                Z = jnp.take(Z, ki, axis=1); U = jnp.take(U, ki, axis=1)
                av = jnp.asarray(alive)
                B = jnp.take(Bfull, av, axis=1); yy = jnp.take(yyf, av)
            elif keep.size == 0:
                Zfull[:, alive] = np.asarray(Z, dtype=np.float64); break
        Zfull[:, alive] = np.asarray(Z, dtype=np.float64)
        gw, gmed, feas, fsum = fw_gap_and_feas(D, Y, Zfull)
        while tgt_i < len(targets) and gw <= targets[tgt_i] and feas <= 1e-9:
            hit[str(targets[tgt_i])] = {"t": time.perf_counter() - t0, "iters": it,
                                        "gap": gw, "feas": feas, "obj": fsum,
                                        "alive": int(alive.size)}
            tgt_i += 1
    return {"arm": "ours_admm_ws", "device": device, "dtype": dtype, "rho": rho,
            "alpha": alpha, "nbis": nbis, "iters": it, "pixels": m,
            "final_worst_gap": gw, "final_median_gap": gmed, "feas": feas,
            "obj": fsum, "wall_total": time.perf_counter() - t0,
            "hit": hit, "trace": log[-8:]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mat", default="/home/miria/data/hsi/Cuprite.mat")
    ap.add_argument("--device", default="gpu"); ap.add_argument("--dtype", default="f64")
    ap.add_argument("--pixels", type=int, default=0)
    ap.add_argument("--rho", type=float, default=1.0)
    ap.add_argument("--alpha", type=float, default=1.6)
    ap.add_argument("--nbis", type=int, default=48)
    ap.add_argument("--budget", type=float, default=1800.0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    import scipy.io as sio
    d = sio.loadmat(a.mat)
    D = np.asarray(d["D"], dtype=np.float64); Y = np.asarray(d["Y"], dtype=np.float64)
    if a.pixels: Y = Y[:, : a.pixels]
    r = run(D, Y, a.device, a.dtype, [1e-3, 1e-4, 1e-6, 1e-8],
            rho=a.rho, alpha=a.alpha, nbis=a.nbis, budget_s=a.budget)
    print(json.dumps(r, indent=1))
    if a.out: open(a.out, "w").write(json.dumps(r, indent=1))
