"""Per-case-class breakdown of the operator certificate for one candidate (offline diagnosis, not used in search)."""
import sys, os, json
os.environ.setdefault("OPCHECK_TARGET", "")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib.util
import numpy as np, jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import opcheck
from atlas.operators.prox import project_box_hyperplane

spec = importlib.util.spec_from_file_location("cand", sys.argv[1]); mod = importlib.util.module_from_spec(spec)
mod.__dict__.update({"jax": jax, "jnp": jnp, "np": np}); spec.loader.exec_module(mod)
cand = jax.jit(mod.project)
ref = jax.jit(lambda v, lo, hi, a, b: project_box_hyperplane(v, lo, hi, a, b, iters=200))
rng = np.random.default_rng(0)
cases = opcheck.cases(rng)
labels = []
for n in (7, 64, 1000, 10000):
    for hk in ("one", "inf", "caps"):
        for sc in (0.1, 3.0, 50.0):
            labels.append(f"pos n={n} hi={hk} scale={sc}")
for n in (64, 2000):
    for sc in (0.1, 3.0, 50.0):
        labels.append(f"MIXED-SIGN n={n} scale={sc}")
labels += ["target"] * (len(cases) - len(labels) - 2) + ["simplex hi=inf", "degenerate tie"]
worst = {}
for lab, (v, lo, hi, a, b) in zip(labels, cases):
    args = tuple(jnp.asarray(z) for z in (v, lo, hi, a)) + (jnp.asarray(b),)
    x, xr = np.asarray(cand(*args)), np.asarray(ref(*args))
    err = float(np.max(np.abs(x - xr)) / (1 + np.max(np.abs(xr)))) if np.all(np.isfinite(x)) else float("inf")
    key = lab.split(" scale")[0]
    worst[key] = max(worst.get(key, 0.0), err)
for k, e in worst.items():
    print("  %-28s %s" % (k, "ok" if e <= 1e-9 else "FAIL %.2e" % e))
