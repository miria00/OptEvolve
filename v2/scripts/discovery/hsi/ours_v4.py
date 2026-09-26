"""Our arm, v4 = relocation (ADMM on a shared inverse)
                + the certified box-and-hyperplane projection, bisection form
                + a working set over pixels.

Three genes, each doing measurable work:

  formulation   the simplex is relocated into g, so f's proximal map is one
                498x498 inverse shared by all 47,750 pixels.  A first-order
                method cannot use it.
  operator      the projection finds theta with sum(max(v-theta,0)) = 1 by
                safeguarded bisection on a bracket of width 1, which is the
                certified operator's form.  It replaces a full sort per
                column, which is what a GPU is worst at.
  restrict      a pixel whose recomputed gap is already far under target is
                retired and the batch is compacted.  The final acceptance is
                recomputed over EVERY pixel, retired ones included, so the
                retirement is a heuristic wrapped in an exact check.
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
        """Projection onto the unit simplex, per column, by bisection on the
        dual scalar.  sum_j max(v_j - theta, 0) = 1 is decreasing in theta and
        is bracketed by [max(v)-1, max(v)], a width-1 interval."""
        vmax = jnp.max(V, axis=0)
        lo = vmax - 1.0
        hi = vmax
        def step(i, b):
            lo, hi = b
            mid = 0.5 * (lo + hi)
            s = jnp.sum(jnp.maximum(V - mid, 0.0), axis=0)
            gt = s > 1.0
            return (jnp.where(gt, mid, lo), jnp.where(gt, hi, mid))
        lo, hi = jax.lax.fori_loop(0, nbis, step, (lo, hi))
        return jnp.maximum(V - 0.5 * (lo + hi), 0.0)

    @jax.jit
    def chunk(Z, U, M, B, rho, K, alpha):
        def body(i, s):
            Z, U = s
            X = M @ (B + rho * (Z - U))
            XA = alpha * X + (1.0 - alpha) * Z
            Zn = proj(XA + U)
            return (Zn, U + XA - Zn)
        return jax.lax.fori_loop(0, K, body, (Z, U))

    return jax, jnp, dev, dt, chunk


def run(D, Y, device, dtype, targets, rho=1.0, alpha=1.6, nbis=60,
        max_iter=400000, check_every=25, budget_s=600.0):
    jax, jnp, dev, dt, chunk = build(device, dtype, nbis)
    n, m = D.shape[1], Y.shape[1]
    hit, log = {}, []
    t0 = time.perf_counter()
    with jax.default_device(dev):
        Dd = jnp.asarray(D, dtype=dt); Yd = jnp.asarray(Y, dtype=dt)
        DtD = Dd.T @ Dd
        Bfull = Dd.T @ Yd
        M = jnp.linalg.inv(DtD + rho * jnp.eye(n, dtype=dt))
        Zfull = np.full((n, m), 1.0 / n, dtype=np.float64)
        alive = np.arange(m)
        Z = jnp.full((n, m), 1.0 / n, dtype=dt); U = jnp.zeros((n, m), dtype=dt)
        B = Bfull
        rh = jnp.asarray(rho, dtype=dt); al = jnp.asarray(alpha, dtype=dt)
        Z, U = chunk(Z, U, M, B, rh, 1, al); Z.block_until_ready()
        it, tgt_i = 1, 0
        while it < max_iter and (time.perf_counter() - t0) < budget_s and tgt_i < len(targets):
            Z, U = chunk(Z, U, M, B, rh, check_every, al)
            Z.block_until_ready(); it += check_every
            Zfull[:, alive] = np.asarray(Z, dtype=np.float64)
            el = time.perf_counter() - t0
            gw, gmed, feas, fsum = fw_gap_and_feas(D, Y, Zfull)
            log.append({"it": it, "t": round(el, 3), "alive": int(alive.size),
                        "worst": gw, "med": gmed})
            while tgt_i < len(targets) and gw <= targets[tgt_i] and feas <= 1e-9:
                hit[str(targets[tgt_i])] = {"t": el, "iters": it, "gap": gw,
                                            "feas": feas, "obj": fsum,
                                            "alive": int(alive.size)}
                tgt_i += 1
            if tgt_i >= len(targets):
                break
            # retire pixels two decades under the next target
            G = D.T @ (D @ Zfull[:, alive]) - D.T @ Y[:, alive]
            R = D @ Zfull[:, alive] - Y[:, alive]
            f = 0.5 * np.einsum('ij,ij->j', R, R)
            gi = (np.einsum('ij,ij->j', G, Zfull[:, alive]) - G.min(axis=0)) / (1.0 + np.abs(f))
            keep = np.where(gi > targets[tgt_i] * 0.01)[0]
            if 0 < keep.size <= alive.size * 0.8:
                alive = alive[keep]; ki = jnp.asarray(keep)
                Z = jnp.take(Z, ki, axis=1); U = jnp.take(U, ki, axis=1)
                B = jnp.take(Bfull, jnp.asarray(alive), axis=1)
            elif keep.size == 0:
                break
    return {"device": device, "dtype": dtype, "rho": rho, "alpha": alpha,
            "nbis": nbis, "iters": it, "pixels": m, "final_worst_gap": gw,
            "final_median_gap": gmed, "feas": feas, "obj": fsum,
            "wall_total": time.perf_counter() - t0, "hit": hit, "trace": log[-3:]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mat", default="/home/miria/data/hsi/Cuprite.mat")
    ap.add_argument("--device", default="gpu"); ap.add_argument("--dtype", default="f64")
    ap.add_argument("--pixels", type=int, default=0)
    ap.add_argument("--rho", type=float, default=1.0)
    ap.add_argument("--alpha", type=float, default=1.6)
    ap.add_argument("--nbis", type=int, default=60)
    ap.add_argument("--budget", type=float, default=600.0)
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
