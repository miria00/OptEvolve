"""Our arm for TV-regularized Poisson deconvolution: PDHG on K = [A; D].

The formulation gene is the whole story here.  The Poisson data term has no
Lipschitz gradient, so the smooth-plus-proximable split that works for the
Gaussian case is inapplicable; relocating the data term behind the linear map
leaves g = the non-negativity indicator, whose projection is a clip, and both
dual blocks have closed-form proxes.  Both operators are GPU native: A is an
FFT pair, D is a local stencil.
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pois_core import make_kernel, make_data, primal


def build(device, dtype):
    os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu" if device == "gpu" else "cpu")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.22")
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
    def chunk(x, xb, ua, px, py, y, Kf, lam, tau, sig, K):
        n0, n1 = x.shape
        def body(i, s):
            x, xb, ua, px, py = s
            Axb = jnp.fft.irfft2(jnp.fft.rfft2(xb) * Kf, s=(n0, n1))
            v = ua + sig * Axb
            uan = 0.5 * (v + 1.0 - jnp.sqrt((v - 1.0) ** 2 + 4.0 * sig * y))
            gx, gy = grad(xb)
            qx, qy = px + sig * gx, py + sig * gy
            nrm = jnp.sqrt(qx * qx + qy * qy)
            sc = jnp.minimum(1.0, lam / jnp.maximum(nrm, 1e-30))
            pxn, pyn = qx * sc, qy * sc
            Atu = jnp.fft.irfft2(jnp.fft.rfft2(uan) * jnp.conj(Kf), s=(n0, n1))
            xn = jnp.maximum(x - tau * (Atu - div(pxn, pyn)), 0.0)
            return (xn, 2.0 * xn - x, uan, pxn, pyn)
        return jax.lax.fori_loop(0, K, body, (x, xb, ua, px, py))

    return jax, jnp, dev, dt, chunk


def run(y, Kf, lam, device, dtype, targets, fstar, tau=0.33, sig=0.33,
        max_iter=4000000, check_every=200, budget_s=300.0):
    jax, jnp, dev, dt, chunk = build(device, dtype)
    hit, log = {}, []
    t0 = time.perf_counter()
    with jax.default_device(dev):
        yd = jnp.asarray(y, dt)
        Kd = jnp.asarray(Kf, jnp.complex128 if dtype == "f64" else jnp.complex64)
        x = jnp.maximum(yd, 1e-3); xb = x
        ua = jnp.zeros_like(yd); px = jnp.zeros_like(yd); py = jnp.zeros_like(yd)
        args = (yd, Kd, jnp.asarray(lam, dt), jnp.asarray(tau, dt), jnp.asarray(sig, dt))
        x, xb, ua, px, py = chunk(x, xb, ua, px, py, *args, 1); x.block_until_ready()
        it, tgt_i = 1, 0
        while it < max_iter and (time.perf_counter() - t0) < budget_s and tgt_i < len(targets):
            x, xb, ua, px, py = chunk(x, xb, ua, px, py, *args, check_every)
            x.block_until_ready(); it += check_every
            el = time.perf_counter() - t0
            P = primal(np.asarray(x, dtype=np.float64), y, Kf, lam)
            rel = (P - fstar) / (1.0 + abs(fstar))
            log.append({"it": it, "t": round(el, 3), "rel": rel, "P": P})
            while tgt_i < len(targets) and rel <= targets[tgt_i]:
                hit[str(targets[tgt_i])] = {"t": el, "iters": it, "rel": rel, "P": P}
                tgt_i += 1
    return {"arm": "ours_pdhg_poisson", "device": device, "dtype": dtype,
            "shape": list(y.shape), "lam": lam, "iters": it, "final_rel": rel,
            "primal": P, "wall_total": time.perf_counter() - t0,
            "hit": hit, "trace": log[-2:]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--lam", type=float, default=0.05)
    ap.add_argument("--peak", type=float, default=100.0)
    ap.add_argument("--device", default="gpu"); ap.add_argument("--dtype", default="f64")
    ap.add_argument("--budget", type=float, default=120.0)
    ap.add_argument("--ref", type=float, default=None)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    clean = np.load(f"/home/miria/data/tv/0801_{a.size}_clean.npy")
    Kf = make_kernel(clean.shape, sigma=2.0)
    y, _ = make_data(clean, Kf, peak=a.peak, seed=0)
    fstar = a.ref if a.ref is not None else float(
        np.load(f"/home/miria/data/poisson/ref_{a.size}.npy"))
    r = run(y, Kf, a.lam, a.device, a.dtype, [1e-3, 1e-4, 1e-6, 1e-8], fstar,
            budget_s=a.budget)
    print(json.dumps(r, indent=1))
    if a.out: open(a.out, "w").write(json.dumps(r, indent=1))
