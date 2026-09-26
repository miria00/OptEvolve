"""Our arm for batched QP layers, with the operator chain fused algebraically.

The unfused iteration is three GEMMs per step:

    Z  = (rho (S - W) G - q) M        (B,m)(m,n) then (B,n)(n,n)
    GZ = Z G^T                        (B,n)(n,m)
    S  = min(GZ + W, h);  W += GZ - S

but Z is never needed until the very end.  Composing the chain collapses it to
a single GEMM against a precomputed m-by-m operator:

    P = G M G^T   (m,m, built once)      C = q M G^T   (B,m, built once)
    GZ = rho (S - W) P - C               (B,m)(m,m)

This is the same kind of move as the temporal fusion tile, one level up: it
trades launches and HBM traffic for a smaller resident operator, and it is a
gene rather than an implementation detail because whether it pays depends on
m against n and on the device.  Both forms are implemented so the choice is
measured, not assumed.
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qp_core import make_problem, score


def build(device, dtype, fuse):
    os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu" if device == "gpu" else "cpu")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.22")
    import jax, jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    dev = jax.devices("gpu" if device == "gpu" else "cpu")[0]
    dt = jnp.float32 if dtype == "f32" else jnp.float64

    if fuse:
        @jax.jit
        def chunk(S, W, P, C, h, rho, K):
            def body(i, s):
                S, W = s
                GZ = rho * ((S - W) @ P) - C
                Sn = jnp.minimum(GZ + W, h[None, :])
                return (Sn, W + GZ - Sn)
            return jax.lax.fori_loop(0, K, body, (S, W))
    else:
        @jax.jit
        def chunk(S, W, G, M, qs, h, rho, K):
            def body(i, s):
                S, W = s
                Z = (rho * ((S - W) @ G) - qs) @ M
                GZ = Z @ G.T
                Sn = jnp.minimum(GZ + W, h[None, :])
                return (Sn, W + GZ - Sn)
            return jax.lax.fori_loop(0, K, body, (S, W))
    return jax, jnp, dev, dt, chunk


def run(Q, G, h, qs, fstar, device, dtype, targets, fuse=True, rho=1.0,
        max_iter=2000000, check_every=200, budget_s=300.0):
    jax, jnp, dev, dt, chunk = build(device, dtype, fuse)
    n, m, B = Q.shape[0], h.size, qs.shape[0]
    hit, log = {}, []
    t0 = time.perf_counter()
    with jax.default_device(dev):
        Qd = jnp.asarray(Q, dt); Gd = jnp.asarray(G, dt)
        hd = jnp.asarray(h, dt); qd = jnp.asarray(qs, dt)
        M = jnp.linalg.inv(Qd + rho * (Gd.T @ Gd))
        S = jnp.zeros((B, m), dt); W = jnp.zeros((B, m), dt)
        rh = jnp.asarray(rho, dt)
        if fuse:
            MGt = M @ Gd.T
            P = Gd @ MGt
            C = qd @ MGt
            args = (P, C, hd, rh)
        else:
            args = (Gd, M, qd, hd, rh)
        S, W = chunk(S, W, *args, 1); S.block_until_ready()
        it, tgt_i = 1, 0
        while it < max_iter and (time.perf_counter() - t0) < budget_s and tgt_i < len(targets):
            S, W = chunk(S, W, *args, check_every)
            S.block_until_ready(); it += check_every
            el = time.perf_counter() - t0
            Zj = (rh * ((S - W) @ Gd) - qd) @ M
            Zh = np.asarray(Zj, dtype=np.float64)
            gw, gmed, feas, _ = score(Q, G, h, qs, Zh, fstar)
            acc = max(gw, feas)
            log.append({"it": it, "t": round(el, 3), "rel": gw, "feas": feas})
            while tgt_i < len(targets) and acc <= targets[tgt_i]:
                hit[str(targets[tgt_i])] = {"t": el, "iters": it, "rel": gw, "feas": feas}
                tgt_i += 1
    return {"arm": "ours_admm_qp", "fuse": fuse, "device": device, "dtype": dtype,
            "n": n, "m": m, "B": B, "iters": it, "rho": rho,
            "final_rel": gw, "final_feas": feas,
            "wall_total": time.perf_counter() - t0, "hit": hit, "trace": log[-2:]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=64); ap.add_argument("--m", type=int, default=48)
    ap.add_argument("--B", type=int, default=1024)
    ap.add_argument("--device", default="gpu"); ap.add_argument("--dtype", default="f64")
    ap.add_argument("--fuse", type=int, default=1)
    ap.add_argument("--budget", type=float, default=120.0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    Q, G, h, qs = make_problem(n=a.n, m=a.m, B=a.B, seed=0)
    ref = np.load(f"/home/miria/data/qp/ref_n{a.n}_m{a.m}_B{a.B}.npy")
    r = run(Q, G, h, qs, ref, a.device, a.dtype, [1e-3, 1e-4, 1e-6, 1e-8],
            fuse=bool(a.fuse), budget_s=a.budget)
    print(json.dumps(r, indent=1))
    if a.out: open(a.out, "w").write(json.dumps(r, indent=1))
