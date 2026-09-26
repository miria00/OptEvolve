"""Our arm for isotropic TV: Condat-Vu with device and precision as genes.

f = 1/2||x-y||^2 (smooth, L_f = 1), g = 0 or the pixel box (exact clip),
h = lam||.||_{2,1} composed with the gradient (||D||^2 <= 8).  This is a base
scheme of the genome, so what is evolved here is the stepsize pair, the
over-relaxation, the device and the precision, not a new algorithm.

Timing includes transfer and compilation.  Acceptance is a float64 host
recomputation of the relative primal-dual gap from the returned iterate and a
feasible dual point, never the solver's own stopping rule.
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tv_core import certify


def build(device, dtype, box):
    os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu" if device == "gpu" else "cpu")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.28")
    import jax, jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    dev = jax.devices("gpu" if device == "gpu" else "cpu")[0]
    dt = jnp.float32 if dtype == "f32" else jnp.float64

    def grad(x):
        gx = jnp.zeros_like(x).at[:-1, :].set(x[1:, :] - x[:-1, :])
        gy = jnp.zeros_like(x).at[:, :-1].set(x[:, 1:] - x[:, :-1])
        return gx, gy

    def div(px, py):
        d = jnp.zeros_like(px)
        d = d.at[0, :].set(px[0, :])
        d = d.at[1:-1, :].set(px[1:-1, :] - px[:-2, :])
        d = d.at[-1, :].set(-px[-2, :])
        e = jnp.zeros_like(py)
        e = e.at[:, 0].set(py[:, 0])
        e = e.at[:, 1:-1].set(py[:, 1:-1] - py[:, :-2])
        e = e.at[:, -1].set(-py[:, -2])
        return d + e

    @jax.jit
    def chunk(x, px, py, y, lam, tau, sig, rho, K):
        def body(i, s):
            x, px, py = s
            dtp = -div(px, py)
            xn = x - tau * ((x - y) + dtp)
            if box:
                xn = jnp.clip(xn, 0.0, 1.0)
            xn = x + rho * (xn - x)
            xb = 2.0 * xn - x
            gx, gy = grad(xb)
            qx, qy = px + sig * gx, py + sig * gy
            nrm = jnp.sqrt(qx * qx + qy * qy)
            sc = jnp.minimum(1.0, lam / jnp.maximum(nrm, 1e-30))
            return (xn, qx * sc, qy * sc)
        return jax.lax.fori_loop(0, K, body, (x, px, py))

    return jax, jnp, dev, dt, chunk


def run(y, lam, device, dtype, targets, tau=0.25, sig=0.45, rho=1.0, box=False,
        max_iter=2000000, check_every=100, budget_s=600.0):
    jax, jnp, dev, dt, chunk = build(device, dtype, box)
    hit, log = {}, []
    t0 = time.perf_counter()
    with jax.default_device(dev):
        yd = jnp.asarray(y, dtype=dt)
        x = yd
        px = jnp.zeros_like(yd); py = jnp.zeros_like(yd)
        lm = jnp.asarray(lam, dtype=dt); ta = jnp.asarray(tau, dtype=dt)
        sg = jnp.asarray(sig, dtype=dt); rh = jnp.asarray(rho, dtype=dt)
        x, px, py = chunk(x, px, py, yd, lm, ta, sg, rh, 1)
        x.block_until_ready()
        it, tgt_i = 1, 0
        while it < max_iter and (time.perf_counter() - t0) < budget_s and tgt_i < len(targets):
            x, px, py = chunk(x, px, py, yd, lm, ta, sg, rh, check_every)
            x.block_until_ready(); it += check_every
            el = time.perf_counter() - t0
            xh = np.asarray(x, dtype=np.float64)
            g, P, G = certify(xh, y, lam, np.asarray(px, dtype=np.float64),
                              np.asarray(py, dtype=np.float64), box)
            log.append({"it": it, "t": round(el, 3), "gap": g, "P": P})
            while tgt_i < len(targets) and g <= targets[tgt_i]:
                hit[str(targets[tgt_i])] = {"t": el, "iters": it, "gap": g, "P": P}
                tgt_i += 1
    return {"arm": "ours_condat_vu", "device": device, "dtype": dtype, "box": box,
            "tau": tau, "sig": sig, "rho": rho, "iters": it, "shape": list(y.shape),
            "lam": lam, "final_gap": g, "primal": P, "dual": G,
            "wall_total": time.perf_counter() - t0, "hit": hit, "trace": log[-3:]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--npy", default="")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--lam", type=float, default=0.1)
    ap.add_argument("--device", default="gpu"); ap.add_argument("--dtype", default="f64")
    ap.add_argument("--tau", type=float, default=0.25); ap.add_argument("--sig", type=float, default=0.45)
    ap.add_argument("--rho", type=float, default=1.0)
    ap.add_argument("--box", action="store_true")
    ap.add_argument("--budget", type=float, default=300.0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    if a.npy:
        y = np.load(a.npy)
    else:
        raise SystemExit("pass --npy <noisy image>.npy (prepared by prep_images.py)")
    r = run(y, a.lam, a.device, a.dtype, [1e-3, 1e-4, 1e-6, 1e-8],
            tau=a.tau, sig=a.sig, rho=a.rho, box=a.box, budget_s=a.budget)
    print(json.dumps(r, indent=1))
    if a.out: open(a.out, "w").write(json.dumps(r, indent=1))
