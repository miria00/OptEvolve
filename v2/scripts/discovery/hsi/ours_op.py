"""Our arm, v3: relocation + certified projection, run as DRS/ADMM.

    min_x  1/2||Dx-y||^2 + iota_simplex(z),   x = z

The formulation gene is what matters here.  Every pixel shares the same D, so
relocating the simplex into g leaves an f whose proximal map is
(D^T D + rho I)^{-1}, ONE 498x498 inverse for all 47,750 pixels.  The
iteration is then a single GEMM plus the exact box-and-hyperplane projection
this project already certified.  A first-order method on the same problem
cannot use that shared factorization and needs ~10^5 iterations; this needs
hundreds.

Timing covers Gram, inverse, transfer and compilation.  Acceptance is a
float64 host recomputation of the worst-pixel relative Frank-Wolfe gap on z,
which is exactly feasible by construction.
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fcls_core import fw_gap_and_feas


def build(device, dtype, proj_kind='sort', nbis=48):
    os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu" if device == "gpu" else "cpu")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.28")
    import jax, jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    dev = jax.devices("gpu" if device == "gpu" else "cpu")[0]
    dt = jnp.float32 if dtype == "f32" else jnp.float64

    def proj_sort(V):
        n = V.shape[0]
        U = jnp.sort(V, axis=0)[::-1]
        css = jnp.cumsum(U, axis=0) - 1.0
        ind = jnp.arange(1, n + 1, dtype=V.dtype).reshape(-1, 1)
        r = n - 1 - jnp.argmax(((U - css / ind) > 0)[::-1], axis=0)
        th = jnp.take_along_axis(css, r[None, :], axis=0)[0] / (r + 1).astype(V.dtype)
        return jnp.maximum(V - th, 0.0)

    def proj_bisect(V):
        vmax = jnp.max(V, axis=0)
        def st(i, b):
            lo, hi = b
            mid = 0.5 * (lo + hi)
            s = jnp.sum(jnp.maximum(V - mid, 0.0), axis=0)
            gt = s > 1.0
            return (jnp.where(gt, mid, lo), jnp.where(gt, hi, mid))
        lo, hi = jax.lax.fori_loop(0, nbis, st, (vmax - 1.0, vmax))
        return jnp.maximum(V - 0.5 * (lo + hi), 0.0)

    proj = proj_sort if proj_kind == "sort" else proj_bisect

    @jax.jit
    def chunk(Z, U, M, B, rho, K, alpha):
        def body(i, s):
            Z, U = s
            X = M @ (B + rho * (Z - U))
            XA = alpha * X + (1.0 - alpha) * Z          # over-relaxation gene
            Zn = proj(XA + U)
            Un = U + XA - Zn
            return (Zn, Un)
        return jax.lax.fori_loop(0, K, body, (Z, U))

    return jax, jnp, dev, dt, chunk


def run(D, Y, device, dtype, targets, rho=1.0, alpha=1.6, proj_kind='sort',
        nbis=48, max_iter=200000, check_every=25, budget_s=600.0):
    jax, jnp, dev, dt, chunk = build(device, dtype, proj_kind, nbis)
    n, m = D.shape[1], Y.shape[1]
    hit, log = {}, []
    t0 = time.perf_counter()
    with jax.default_device(dev):
        Dd = jnp.asarray(D, dtype=dt)
        Yd = jnp.asarray(Y, dtype=dt)
        DtD = Dd.T @ Dd
        B = Dd.T @ Yd
        # one inverse for every pixel: this is what relocation buys
        M = jnp.linalg.inv(DtD + rho * jnp.eye(n, dtype=dt))
        Z = jnp.full((n, m), 1.0 / n, dtype=dt)
        U = jnp.zeros((n, m), dtype=dt)
        rh = jnp.asarray(rho, dtype=dt); al = jnp.asarray(alpha, dtype=dt)
        Z, U = chunk(Z, U, M, B, rh, 1, al)
        Z.block_until_ready()
        it, tgt_i = 1, 0
        while it < max_iter and (time.perf_counter() - t0) < budget_s and tgt_i < len(targets):
            Z, U = chunk(Z, U, M, B, rh, check_every, al)
            Z.block_until_ready(); it += check_every
            el = time.perf_counter() - t0
            Zh = np.asarray(Z, dtype=np.float64)
            gw, gmed, feas, fsum = fw_gap_and_feas(D, Y, Zh)
            log.append({"it": it, "t": round(el, 3), "worst": gw, "med": gmed})
            while tgt_i < len(targets) and gw <= targets[tgt_i] and feas <= 1e-9:
                hit[str(targets[tgt_i])] = {"t": el, "iters": it, "gap": gw,
                                            "feas": feas, "obj": fsum}
                tgt_i += 1
    return {"device": device, "dtype": dtype, "rho": rho, "alpha": alpha,
            "proj": proj_kind, "nbis": nbis,
            "iters": it, "pixels": m, "final_worst_gap": gw,
            "final_median_gap": gmed, "feas": feas, "obj": fsum,
            "wall_total": time.perf_counter() - t0, "hit": hit, "trace": log[-4:]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mat", default="/home/miria/data/hsi/Cuprite.mat")
    ap.add_argument("--device", default="gpu"); ap.add_argument("--dtype", default="f64")
    ap.add_argument("--pixels", type=int, default=0)
    ap.add_argument("--rho", type=float, default=1.0)
    ap.add_argument("--alpha", type=float, default=1.6)
    ap.add_argument("--budget", type=float, default=600.0)
    ap.add_argument("--proj", default="sort")
    ap.add_argument("--nbis", type=int, default=48)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    import scipy.io as sio
    d = sio.loadmat(a.mat)
    D = np.asarray(d["D"], dtype=np.float64); Y = np.asarray(d["Y"], dtype=np.float64)
    if a.pixels: Y = Y[:, : a.pixels]
    r = run(D, Y, a.device, a.dtype, [1e-3, 1e-4, 1e-6, 1e-8],
            rho=a.rho, alpha=a.alpha, proj_kind=a.proj, nbis=a.nbis,
            budget_s=a.budget)
    print(json.dumps(r, indent=1))
    if a.out: open(a.out, "w").write(json.dumps(r, indent=1))
