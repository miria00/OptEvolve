"""Per-iteration time t of our OT PDHG variants on one GPU (DOTmark pairs).

N (iterations to the certificate) is hardware-invariant and comes from the CPU
calibration; this measures only t, so GPU time-to-solution is predicted as
N x t and verified separately once N is fixed. tol = 0 forces the full count.
"""
import json, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import jax
jax.config.update("jax_enable_x64", True)
from atlas.bench import dotmark
from atlas.genome import Backend
from atlas.schemes.base_schemes import Budget, PDHGParams, build_scheme, run_built

out = {"device": str(jax.devices("cuda")[0]), "rows": []}
ITERS = 2048
for res, op in ((32, "incidence"), (32, "reduction"), (64, "incidence"), (64, "reduction")):
    prob = dotmark.composite("CauchyDensity", res, 1009, 1010, operator=op)
    for scheme, nb in (("pdhg", float(prob.L.norm_bound)), ("pdhg_entropic", 2 ** 0.5)):
        t, s = 0.99 / nb, 0.99 / nb
        params = PDHGParams(t=t, s=s)
        built = build_scheme(prob, scheme, params)
        for dev in ("gpu",):
            r = run_built(built, built.cert, params, Budget(max_iters=ITERS, tol=0.0, sample_every=512),
                          backend=Backend(precision="f64", K=64, device=dev))
            row = {"res": res, "operator": op, "arcs": res ** 4, "scheme": scheme, "device": dev, "iters": int(r.iters),
                   "exec_s": float(r.wall_time), "us_per_iter": 1e6 * float(r.wall_time) / max(int(r.iters), 1)}
            out["rows"].append(row); print(row, flush=True)
json.dump(out, open("results/ot_dotmark_20260910/gpu_periter.json", "w"), indent=1)
