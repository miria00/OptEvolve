"""Is the elastic-net logistic step memory bound? Time X w, X^T v and the logistic
gradient on the GPU in float64 and float32 and report achieved bandwidth.
Data: epsilon test split (real). Bytes counted: one full read of X per product."""
import argparse, json, sys, time
import numpy as np, jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
ap = argparse.ArgumentParser()
ap.add_argument("--data", default="/home/miria/OptEvolve/v2/data/enet_libsvm/epsilon_test")
ap.add_argument("--out", default="/home/miria/OptEvolve/v2/results/logreg_20260911/bandwidth_4090.json")
a = ap.parse_args()
assert jax.devices()[0].platform == "gpu", f"not on a GPU: {jax.devices()}"
D = a.data
X = np.load(D + "/X.npy"); y = np.load(D + "/y.npy").astype(np.float64)
n, p = X.shape
rng = np.random.default_rng(0)
out = {"device": str(jax.devices()[0]), "n": n, "p": p, "rows": []}
for dt in (jnp.float64, jnp.float32):
    Xg = jnp.asarray(X, dtype=dt); yg = jnp.asarray(y, dtype=dt)
    w = jnp.asarray(rng.normal(size=p), dtype=dt); v = jnp.asarray(rng.normal(size=n), dtype=dt)
    fns = {
        "Xw": (jax.jit(lambda X, w: X @ w), (Xg, w), 1),
        "XTv": (jax.jit(lambda X, v: X.T @ v), (Xg, v), 1),
        "grad": (jax.jit(lambda X, w, y: -(X.T @ (y * jax.nn.sigmoid(-y * (X @ w)))) / X.shape[0]), (Xg, w, yg), 2),
    }
    for name, (f, args, passes) in fns.items():
        jax.block_until_ready(f(*args))
        ts = []
        for _ in range(30):
            t0 = time.perf_counter(); jax.block_until_ready(f(*args)); ts.append(time.perf_counter() - t0)
        t = float(np.median(ts)); b = passes * n * p * np.dtype(dt).itemsize
        row = {"op": name, "dtype": np.dtype(dt).name, "ms": 1e3 * t, "GBps": b / t / 1e9, "nominal_passes": passes}
        out["rows"].append(row); print(f"{np.dtype(dt).name:8s} {name:5s} {1e3 * t:7.3f} ms  {b / t / 1e9:7.1f} GB/s (counting {passes} read(s) of X)", flush=True)
    del Xg
json.dump(out, open(a.out, "w"), indent=1)
