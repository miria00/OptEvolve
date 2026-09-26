"""Our arm for isotropic TV deblurring: Condat-Vu, device and precision genes.

f = 1/2||Ax-y||^2 + mu/2||x||^2 is smooth with L_f = ||A||^2 + mu, A is a
circular blur applied by FFT, g is the pixel box, h is lam||.||_{2,1} on the
gradient.  Both operators the iteration needs, an FFT pair and a local
difference stencil, are GPU native, and there is no packaged solver for this
problem to defer to.
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deblur_core import make_kernel, blur, certify


def build(device, dtype, box):
    os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu" if device == "gpu" else "cpu")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.11")
    import jax, jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    dev = jax.devices("gpu" if device == "gpu" else "cpu")[0]
    dt = jnp.float32 if dtype == "f32" else jnp.float64

    def grad(x):
        gx = jnp.zeros_like(x).at[:-1, :].set(x[1:, :] - x[:-1, :])
        gy = jnp.zeros_like(x).at[:, :-1].set(x[:, 1:] - x[:, :-1])
        return gx, gy

    def div(px, py):
        d = jnp.zeros_like(px).at[0, :].set(px[0, :])
        d = d.at[1:-1, :].set(px[1:-1, :] - px[:-2, :]).at[-1, :].set(-px[-2, :])
        e = jnp.zeros_like(py).at[:, 0].set(py[:, 0])
        e = e.at[:, 1:-1].set(py[:, 1:-1] - py[:, :-2]).at[:, -1].set(-py[:, -2])
        return d + e

    @jax.jit
    def chunk(x, px, py, y, Kf, lam, mu, tau, sig, K):
        n0, n1 = x.shape
        def body(i, s):
            x, px, py = s
            Ax = jnp.fft.irfft2(jnp.fft.rfft2(x) * Kf, s=(n0, n1))
            r = Ax - y
            gf = jnp.fft.irfft2(jnp.fft.rfft2(r) * jnp.conj(Kf), s=(n0, n1)) + mu * x
            dtp = -div(px, py)
            xn = x - tau * (gf + dtp)
            if box:
                xn = jnp.clip(xn, 0.0, 1.0)
            xb = 2.0 * xn - x
            gx, gy = grad(xb)
            qx, qy = px + sig * gx, py + sig * gy
            nrm = jnp.sqrt(qx * qx + qy * qy)
            sc = jnp.minimum(1.0, lam / jnp.maximum(nrm, 1e-30))
            return (xn, qx * sc, qy * sc)
        return jax.lax.fori_loop(0, K, body, (x, px, py))

    return jax, jnp, dev, dt, chunk


def run(y, Kf, lam, mu, device, dtype, targets, tau=0.25, sig=0.45, box=True,
        max_iter=4000000, check_every=100, budget_s=300.0):
    jax, jnp, dev, dt, chunk = build(device, dtype, box)
    hit, log = {}, []
    t0 = time.perf_counter()
    with jax.default_device(dev):
        yd = jnp.asarray(y, dtype=dt)
        Kd = jnp.asarray(Kf, dtype=jnp.complex128 if dtype == "f64" else jnp.complex64)
        x = yd
        px = jnp.zeros_like(yd); py = jnp.zeros_like(yd)
        args = (jnp.asarray(lam, dt), jnp.asarray(mu, dt),
                jnp.asarray(tau, dt), jnp.asarray(sig, dt))
        x, px, py = chunk(x, px, py, yd, Kd, *args, 1); x.block_until_ready()
        it, tgt_i = 1, 0
        while it < max_iter and (time.perf_counter() - t0) < budget_s and tgt_i < len(targets):
            x, px, py = chunk(x, px, py, yd, Kd, *args, check_every)
            x.block_until_ready(); it += check_every
            el = time.perf_counter() - t0
            g, P, G = certify(np.asarray(x, dtype=np.float64), y, Kf, lam, mu,
                              np.asarray(px, dtype=np.float64),
                              np.asarray(py, dtype=np.float64))
            log.append({"it": it, "t": round(el, 3), "gap": g})
            while tgt_i < len(targets) and g <= targets[tgt_i]:
                hit[str(targets[tgt_i])] = {"t": el, "iters": it, "gap": g, "P": P}
                tgt_i += 1
    return {"arm": "ours_condat_vu_deblur", "device": device, "dtype": dtype,
            "shape": list(y.shape), "lam": lam, "mu": mu, "box": box,
            "iters": it, "final_gap": g, "primal": P,
            "wall_total": time.perf_counter() - t0, "hit": hit, "trace": log[-3:]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--npy", required=True)
    ap.add_argument("--lam", type=float, default=0.02)
    ap.add_argument("--mu", type=float, default=1e-6)
    ap.add_argument("--sigma", type=float, default=2.0)
    ap.add_argument("--device", default="gpu"); ap.add_argument("--dtype", default="f64")
    ap.add_argument("--tau", type=float, default=0.25); ap.add_argument("--sig", type=float, default=0.45)
    ap.add_argument("--budget", type=float, default=300.0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    clean = np.load(a.npy.replace("_noisy", "_clean"))
    Kf = make_kernel(clean.shape, sigma=a.sigma)
    rng = np.random.default_rng(0)
    y = blur(clean, Kf) + 0.01 * rng.standard_normal(clean.shape)
    r = run(y, Kf, a.lam, a.mu, a.device, a.dtype, [1e-3, 1e-4, 1e-6, 1e-8],
            tau=a.tau, sig=a.sig, budget_s=a.budget)
    print(json.dumps(r, indent=1))
    if a.out: open(a.out, "w").write(json.dumps(r, indent=1))
