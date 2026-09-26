"""Is the composer's flattened state (ravel_pytree) what makes PDHG on transport ~13x slower per
step than the policy runner? Time the same PDHG map four ways on one DOTmark pair:
  step      one jitted application of T on the pytree state (no loop);
  tree      a 64-step fori_loop carrying the pytree state (x, u), as the policy runner does;
  flat      a 64-step fori_loop carrying the concatenated vector, unravel -> T -> ravel each step,
            as atlas/evolve/kb_composer.py run_recipe does;
  flat+norm the flat loop also computing ||F(v) - v|| each step, as the adaptive-restart body does.
All four compute identical iterates."""
import argparse, json, sys, time
from dataclasses import replace
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree
sys.path.insert(0, ".")
from atlas.bench import dotmark
from atlas.evolve import kb_composer as kc

ap = argparse.ArgumentParser()
ap.add_argument("--instance", default="CauchyDensity:32:1001-1002"); ap.add_argument("--kappa", type=float, default=30.0)
ap.add_argument("--K", type=int, default=64); ap.add_argument("--reps", type=int, default=20); ap.add_argument("--out", default=None)
a = ap.parse_args()
assert jax.devices()[0].platform == "gpu", f"not on a GPU: {jax.devices()}"
cls, res, pr = a.instance.split(":"); i1, i2 = map(int, pr.split("-"))
prob = dotmark.composite(cls, int(res), i1, i2, operator="reduction")
ratio = float(np.linalg.norm(np.asarray(prob.data["c"])) / np.linalg.norm(np.asarray(prob.data["b"]))) / a.kappa
r = next(replace(r, ratio=ratio) for r in kc.enumerate_recipes(bases=("pdhg",)) if "+" not in r.name())
built, _, _ = kc.typecheck_recipe(r, prob)
T = built.T
dyn = jax.tree_util.tree_map(jnp.asarray, built.dyn)
s0 = jax.tree_util.tree_map(jnp.asarray, built.state0)
v0, unravel = ravel_pytree(s0)
K = a.K


def flatF(v, d):
    return ravel_pytree(T(unravel(v), 0, d))[0]


variants = {
    "step": (lambda s, d: T(s, 0, d), s0, 1),
    "tree": (lambda s, d: jax.lax.fori_loop(0, K, lambda i, s: T(s, 0, d), s), s0, K),
    "flat": (lambda v, d: jax.lax.fori_loop(0, K, lambda i, v: flatF(v, d), v), v0, K),
    "flat+norm": (lambda c, d: jax.lax.fori_loop(
        0, K, lambda i, c: (lambda Fv: (Fv, c[1] + jnp.linalg.norm(Fv - c[0])))(flatF(c[0], d)), c), (v0, jnp.asarray(0.0)), K),
}
out = {"instance": a.instance, "device": str(jax.devices()[0]), "K": K, "rows": {}}
ref = None
for name, (f, x0, steps) in variants.items():
    exe = jax.jit(f).lower(x0, dyn).compile()
    x = exe(x0, dyn); jax.block_until_ready(x)
    ts = []
    for _ in range(a.reps):
        t0 = time.perf_counter(); x = exe(x, dyn); jax.block_until_ready(x); ts.append(time.perf_counter() - t0)
    us = 1e6 * float(np.median(ts)) / steps
    out["rows"][name] = us
    print(f"  {name:10s} {us:9.1f} us/step", flush=True)
# identical iterates: K tree steps == K flat steps from the same start
xt = jax.jit(variants["tree"][0])(s0, dyn); xf = unravel(jax.jit(variants["flat"][0])(v0, dyn))
diff = max(float(jnp.max(jnp.abs(p - q))) for p, q in zip(jax.tree_util.tree_leaves(xt), jax.tree_util.tree_leaves(xf)))
print(f"  max |tree - flat| after {K} steps: {diff:.2e}")
out["tree_flat_maxdiff"] = diff
if a.out:
    json.dump(out, open(a.out, "w"), indent=1)
print("FLATTEN_PROBE_DONE")
