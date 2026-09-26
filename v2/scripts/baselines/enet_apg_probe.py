"""BALLPARK PROBE, not a discovered method: textbook accelerated proximal
gradient (Nesterov constant momentum for strongly convex composite, FISTA
family) on the SAME elastic-net objective as enet_bench.py. It answers one
question only: is a full-gradient first-order method even in the same range as
coordinate-descent author code on these datasets?

Records a time-to-objective curve in ONE run: jitted blocks of 25 iterations
with static loop bounds, synchronised between blocks, objective evaluated on
the host with the timer paused. Load, device transfer and step-size estimation
are timed separately and reported.
"""
import sys, time, json, argparse
import numpy as np, scipy.sparse as sp
import jax, jax.numpy as jnp
from jax.experimental import sparse as jsp
from sklearn.datasets import load_svmlight_file
jax.config.update("jax_enable_x64", True)
ap = argparse.ArgumentParser()
ap.add_argument("path"); ap.add_argument("frac", type=float); ap.add_argument("r", type=float)
ap.add_argument("--max-iters", type=int, default=4000); ap.add_argument("--out", default="")
ap.add_argument("--center", action="store_true")
A = ap.parse_args()
dev = jax.devices()[0]
t0 = time.perf_counter(); X, y = load_svmlight_file(A.path); t_load = time.perf_counter() - t0
X = sp.csr_matrix(X); y = np.asarray(y, float); n, d = X.shape
if A.center: y = y - y.mean()
lam_max = np.abs(X.T @ y).max() / n; a = A.frac * lam_max; l1 = a * A.r; l2 = a * (1 - A.r)
def P(w):
    res = y - X @ w
    return 0.5 * res @ res / n + l1 * np.abs(w).sum() + 0.5 * l2 * w @ w
t0 = time.perf_counter()
Xj = jsp.BCOO.from_scipy_sparse(X); XTj = jsp.BCOO.from_scipy_sparse(sp.csr_matrix(X.T)); yj = jnp.asarray(y)
jax.block_until_ready((Xj.data, XTj.data, yj)); t_xfer = time.perf_counter() - t0
@jax.jit
def power(Xj, XTj, v):
    def body(i, v):
        u = XTj @ (Xj @ v); return u / jnp.linalg.norm(u)
    v = jax.lax.fori_loop(0, 60, body, v)
    return jnp.linalg.norm(XTj @ (Xj @ v))
v0 = jnp.ones(d) / np.sqrt(d)
jax.block_until_ready(power(Xj, XTj, v0))
t0 = time.perf_counter(); sig2 = float(power(Xj, XTj, v0)); t_step = time.perf_counter() - t0
L = 1.05 * (sig2 / n) + l2; mu = l2
beta = (np.sqrt(L) - np.sqrt(mu)) / (np.sqrt(L) + np.sqrt(mu))
B = 25
@jax.jit
def block(Xj, XTj, yj, w, wp):
    def body(i, c):
        w, wp = c
        z = w + beta * (w - wp)
        g = XTj @ (Xj @ z - yj) / n + l2 * z
        v = z - g / L
        return (jnp.sign(v) * jnp.maximum(jnp.abs(v) - l1 / L, 0.0), w)
    return jax.lax.fori_loop(0, B, body, (w, wp))
w = jnp.zeros(d); wp = jnp.zeros(d)
jax.block_until_ready(block(Xj, XTj, yj, w, wp))  # compile, excluded
curve = []; t_exec = 0.0; it = 0; stall = 0; prevP = np.inf
while it < A.max_iters:
    t0 = time.perf_counter(); w, wp = block(Xj, XTj, yj, w, wp); jax.block_until_ready(w)
    t_exec += time.perf_counter() - t0; it += B
    p = P(np.asarray(w)); curve.append((it, t_exec, p))
    stall = stall + 1 if prevP - p <= 1e-13 * abs(p) else 0
    prevP = p
    if stall >= 4: break
name = A.path.split("/")[-1]
print(f"{name} on {dev.platform}: n={n} d={d}  kappa=L/mu={L/mu:.1f}  load {t_load:.1f}s  transfer {t_xfer:.3f}s  step-size {t_step*1e3:.1f}ms")
nnz_w = int((np.abs(np.asarray(w)) > 1e-12).sum())
print(f"   final P={curve[-1][2]:.12f} after {curve[-1][0]} iters, exec {curve[-1][1]:.3f}s, nnz(w)={nnz_w}, {1e3*curve[-1][1]/curve[-1][0]:.3f} ms/iter")
if A.out:
    json.dump({"dataset": name, "platform": dev.platform, "n": n, "d": d, "a": a, "r": A.r,
               "kappa": L / mu, "t_load": t_load, "t_transfer": t_xfer, "t_stepsize": t_step,
               "curve": curve}, open(A.out, "w"), indent=1)
