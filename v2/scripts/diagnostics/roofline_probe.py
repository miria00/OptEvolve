"""Roofline predictor for temporal fusion: can arithmetic intensity pick the tile?

Condition number says nothing about per-iteration cost. The quantity that does
is arithmetic intensity against the device ridge point. This probe measures the
device's achievable f64 bandwidth and f64 FLOP rate, computes the intensity of
the TV update under each (B, T) fusion tile, and predicts a ranking. The
prediction is then scored against the measured end-to-end sweep, so the model
is falsifiable rather than decorative.
"""
import argparse, json, time
from pathlib import Path
import numpy as np, jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

# ---- FLOP and byte accounting for one PDHG/Condat-Vu TV iteration -----------
# Counted from atlas/backend/kernels/tv_update.cu, per output pixel.
#   primal : v0(1 sub) v1(1 sub) adj(1 add) num(2 mul,2 add) *inv(1 mul)   = 8
#   bar    : 2*pr - x                                                       = 2
#   duals  : d0,d1 each (bar_nbr - bar) with bar_nbr = 2*pr-x -> 2*(2+1)    = 6
#   clip   : 2 x (1 mul-add + 2 cmp)                                        = 6
#   relax  : 3 arrays x (2 mul + 1 add)                                     = 9
FLOPS_PER_PIXEL_ITER = 31
# Unfused traffic per pixel per iteration: read x,u0,u1,y ; write xp,up0,up1.
BYTES_PER_PIXEL_ITER = 7 * 8


def measure_bandwidth(n_mb=512, reps=20):
    """Achievable f64 streaming bandwidth via a read-2 write-1 triad."""
    n = int(n_mb * 1e6 / 8)
    a = jnp.asarray(np.random.default_rng(0).random(n))
    b = jnp.asarray(np.random.default_rng(1).random(n))
    f = jax.jit(lambda x, y: x + 2.0 * y)
    for _ in range(3):
        jax.block_until_ready(f(a, b))
    t0 = time.perf_counter()
    for _ in range(reps):
        jax.block_until_ready(f(a, b))
    dt = (time.perf_counter() - t0) / reps
    return 3 * n * 8 / dt / 1e9          # GB/s, 2 reads + 1 write


def measure_f64_flops(n=4096, reps=10):
    """Achievable f64 FLOP rate from a large square matmul (2n^3 flops)."""
    rng = np.random.default_rng(0)
    A = jnp.asarray(rng.standard_normal((n, n)))
    B = jnp.asarray(rng.standard_normal((n, n)))
    f = jax.jit(lambda x, y: x @ y)
    for _ in range(3):
        jax.block_until_ready(f(A, B))
    t0 = time.perf_counter()
    for _ in range(reps):
        jax.block_until_ready(f(A, B))
    dt = (time.perf_counter() - t0) / reps
    return 2 * n**3 / dt / 1e12          # TFLOP/s


def tile_intensity(B, T):
    """Arithmetic intensity of one fused tile, per useful output pixel-iteration.

    A block loads (B+2T)^2 cells and writes back B^2, doing T iterations. HBM
    traffic is therefore amortised over B^2 * T useful pixel-iterations, while
    the arithmetic is paid on the shrinking valid region, which is redundant by
    the halo factor.
    """
    S = B + 2 * T
    # 4 arrays in (x,u0,u1,y), 3 out (x,u0,u1)
    bytes_tile = (4 * S * S + 3 * B * B) * 8
    useful = B * B * T
    bytes_per_useful = bytes_tile / useful
    redundancy = (S * S) / (B * B)
    flops_per_useful = FLOPS_PER_PIXEL_ITER * redundancy
    return flops_per_useful / bytes_per_useful, bytes_per_useful, flops_per_useful


def roofline_time(ai, flops_per_useful, bw_GBs, tflops):
    """Seconds per useful pixel-iteration under the roofline."""
    peak = min(tflops * 1e12, bw_GBs * 1e9 * ai)   # attainable FLOP/s
    return flops_per_useful / peak


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device-label", default="4090")
    ap.add_argument("--tiles", default="8x4,16x2,16x4,24x4,32x2,32x3,32x4")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    bw = measure_bandwidth()
    tf = measure_f64_flops()
    ridge = tf * 1e12 / (bw * 1e9)
    print(f"device {a.device_label}: achievable f64 bandwidth {bw:.1f} GB/s, "
          f"f64 {tf:.2f} TFLOP/s, ridge point {ridge:.3f} flop/byte")
    unfused_ai = FLOPS_PER_PIXEL_ITER / BYTES_PER_PIXEL_ITER
    print(f"unfused TV iteration intensity {unfused_ai:.3f} flop/byte -> "
          f"{'MEMORY' if unfused_ai < ridge else 'COMPUTE'}-bound\n")

    rows = []
    print(f"{'tile':>8s} {'S':>4s} {'redund':>7s} {'AI':>7s} {'regime':>8s} {'pred_ns':>9s}")
    for spec in a.tiles.split(","):
        B, T = (int(v) for v in spec.split("x"))
        ai, bpu, fpu = tile_intensity(B, T)
        t = roofline_time(ai, fpu, bw, tf)
        S = B + 2 * T
        rows.append({"tile": spec, "B": B, "T": T, "S": S,
                     "redundancy": (S*S)/(B*B), "ai": ai,
                     "pred_s_per_pixel_iter": t})
        print(f"{spec:>8s} {S:4d} {(S*S)/(B*B):7.2f} {ai:7.3f} "
              f"{'MEM' if ai < ridge else 'COMP':>8s} {t*1e9:9.4f}")
    if a.out:
        Path(a.out).write_text(json.dumps({
            "device": a.device_label, "bandwidth_GBs": bw, "f64_TFLOPs": tf,
            "ridge_flop_per_byte": ridge, "unfused_ai": unfused_ai,
            "flops_per_pixel_iter": FLOPS_PER_PIXEL_ITER,
            "bytes_per_pixel_iter": BYTES_PER_PIXEL_ITER, "tiles": rows,
        }, indent=1))
        print("\nwrote", a.out)
