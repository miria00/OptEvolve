"""Build the opcheck TARGET for spec-driven synthesis: the absorbed block of a close-loop instance and REAL prox inputs.

The prox inputs are recovered from solver states: Condat-Vu updates x+ = prox_{t g}(x - t (grad f(x) + L'y)), so
v = x - t (P x + q + A_keep' y) restricted to the block. The script checks that projecting v with bisection reproduces
the next iterate's block; if not, it stops rather than hand the synthesizer the wrong inputs.
"""
import argparse, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, jax, jax.numpy as jnp, scipy.sparse as sp
from uni import make_instance, permute_instance, diagnose, plan_block, build_formulation
from atlas.schemes.base_schemes import CondatVuParams
from atlas.schemes.compile import build_scheme
from atlas.operators.prox import project_box_hyperplane

ap = argparse.ArgumentParser()
ap.add_argument("--case", default="knapsack_ridge:1500:0.003")
ap.add_argument("--out", required=True)
a = ap.parse_args()
fam, size, rho = a.case.split(":")[0], int(a.case.split(":")[1]), float(a.case.split(":")[2])
d = permute_instance(make_instance(fam, size, seed=0, rho=rho), 7)          # same encoding as close_loop.py
plan, why = plan_block(d, diagnose(d))
form = build_formulation(d, "reloc", plan=plan, impl="bisect")
prob = form.prob
Lf, nb = float(prob.f.props.lipschitz), float(prob.L.norm_bound)
t = 1.0 / Lf
built = build_scheme(prob, "condat_vu", CondatVuParams(t=t, s=0.49 / (t * nb ** 2)), check=True)
T, dyn = built.T, jax.tree_util.tree_map(jnp.asarray, built.dyn)
v = jax.tree_util.tree_map(jnp.asarray, built.state0)
print("state leaves:", [np.shape(z) for z in jax.tree_util.tree_leaves(v)])
step = jax.jit(lambda z, i: T(z, i, dyn))
P, q, A = sp.csr_matrix(prob.data["P"]), np.asarray(prob.data["q"]), sp.csr_matrix(prob.data["A"])
nx = plan["n_x"]
V, errs = [], []
for it in range(1, 1601):
    x, y = np.asarray(v[0]), np.asarray(v[1])
    vin = x - t * (P @ x + q + A.T @ y)
    v = step(v, it)
    if it in (2, 5, 10, 20, 40, 80, 160, 320, 640, 1000, 1300, 1600):
        blk = vin[:nx]
        xb = np.asarray(project_box_hyperplane(jnp.asarray(blk), jnp.asarray(plan["lo"]), jnp.asarray(plan["hi"]),
                                               jnp.asarray(plan["a"]), plan["b_lo"], iters=200))
        errs.append(float(np.max(np.abs(xb - np.asarray(v[0])[:nx]))))
        V.append(blk)
print("max |P(v) - next x block| over captured iterates:", max(errs), errs[:4])
if max(errs) > 1e-8:
    sys.exit("prox inputs NOT identified; refusing to write a target")
np.savez(a.out, lo=plan["lo"], hi=plan["hi"], a=plan["a"], b=plan["b_lo"], V=np.stack(V), case=a.case)
print("wrote", a.out, "n_x", nx, "V", np.stack(V).shape, "a sign", (plan["a"] > 0).mean())
