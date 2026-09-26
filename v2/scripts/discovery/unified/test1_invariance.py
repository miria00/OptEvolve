"""TEST 1 - is N a property of the formulation, independent of how the projection is implemented?

Same instance, same relocated formulation, different projection implementations:
  sort (simplex only, exact), bisect60, bisect100 (exact to machine precision), dykstra-k (generic, inexact).
Prediction if the N x t factorization holds: every exact implementation gives the same N (to the K-iteration
resolution); Dykstra-k matches only once its projection error is small enough, which measures eps*.
Encodings are permuted. Projection error is measured against a 200-step bisection reference on inputs drawn
around the reference solution, so eps is reported in the units the iteration actually sees.
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import jax, jax.numpy as jnp
from uni import (make_instance, permute_instance, diagnose, plan_block, build_formulation, reference, measure,
                 project_block)

ap = argparse.ArgumentParser()
ap.add_argument("--families", default="portfolio,svm_dual,meb_dual,knapsack_ridge,budget_band")
ap.add_argument("--size", type=int, default=2000)
ap.add_argument("--seeds", default="0,1,2")
ap.add_argument("--dykstra", default="1,3,10,30,100,300")
ap.add_argument("--K", type=int, default=10)
ap.add_argument("--cap", type=int, default=50000)
ap.add_argument("--out", required=True)
a = ap.parse_args()


def proj_error(plan, impl, k, x_ref, trials=20, seed=0):
    rng = np.random.default_rng(seed)
    xr = x_ref[plan["T"]]
    scale = max(1e-3, float(np.std(xr)) * 3.0)
    args = (jnp.asarray(plan["lo"]), jnp.asarray(plan["hi"]), jnp.asarray(plan["a"]),
            jnp.asarray(plan["b_lo"]), jnp.asarray(plan["b_hi"]))
    errs = []
    for _ in range(trials):
        vv = jnp.asarray(xr + rng.standard_normal(xr.size) * scale)
        ref = project_block(vv, plan["kind"], "bisect", *args, iters=200, k=0)
        got = project_block(vv, plan["kind"], impl, *args, iters=100 if impl == "bisect" else 60, k=k)
        errs.append(float(jnp.max(jnp.abs(got - ref)) / (1.0 + jnp.max(jnp.abs(ref)))))
    return max(errs)


rows = []
out = open(a.out, "w")
print("  family          seed  impl         eps_proj   N        vs_exact  t_iter_us", flush=True)
for fam in a.families.split(","):
    for seed in [int(s) for s in a.seeds.split(",")]:
        d = permute_instance(make_instance(fam, a.size, seed=seed, rho=1e-2), 100 + seed)
        g = diagnose(d)
        plan, why = plan_block(d, g)
        if plan is None:
            print("  %-15s %-5d NO PLAN: %s" % (fam, seed, why), flush=True); continue
        obj_ref, _ = reference(d)
        if obj_ref is None:
            print("  %-15s %-5d reference failed" % (fam, seed), flush=True); continue
        # a high-accuracy x for the projection-error inputs: the exact relocated solve at tight tolerance
        impls = ([("sort", 0)] if plan["kind"] == "simplex" else []) + [("bisect", 0), ("bisect100", 0)] + \
                [("dykstra", int(k)) for k in a.dykstra.split(",")]
        N_exact = None
        x_for_eps = None
        for impl, k in impls:
            iters = 100 if impl == "bisect100" else 60
            base_impl = "bisect" if impl == "bisect100" else impl
            # an inexact operator either matches the exact N or it does not; 20x the exact N is a clean verdict and
            # stops a non-converging Dykstra run from grinding to the global cap at milliseconds per iteration
            cap = max(20 * N_exact, 2000) if (impl == "dykstra" and N_exact) else a.cap
            try:
                form = build_formulation(d, "reloc", plan=plan, impl=base_impl, iters=iters, k=k)
                m = measure(form, d, obj_ref, tol=1e-6, cap=cap, K=a.K)
            except Exception as e:
                print("  %-15s %-5d %-12s ERROR %s" % (fam, seed, impl, str(e)[:100]), flush=True); continue
            if x_for_eps is None:
                x_for_eps = m["probe"]["x"]
            eps = proj_error(plan, base_impl, k, x_for_eps) if impl != "sort" else 0.0
            if impl in ("sort", "bisect") and N_exact is None and m["N"]:
                N_exact = m["N"]
            label = impl + (str(k) if impl == "dykstra" else "")
            ratio = (m["N"] / N_exact) if (m["N"] and N_exact) else None
            rec = {"family": fam, "seed": seed, "size": a.size, "kind": plan["kind"], "impl": label, "eps": eps,
                   "N": m["N"], "censored": m["censored"], "iters_run": m["iters_run"], "t_iter": m["t_iter"],
                   "setup_s": m["setup_s"], "rv": m["row_violation"], "gap": m["obj_gap"], "N_exact": N_exact}
            out.write(json.dumps(rec) + "\n"); out.flush()
            print("  %-15s %-5d %-12s %-10.2e %-8s %-9s %.1f" % (fam, seed, label, eps,
                  m["N"] if m["N"] else ">%d" % m["iters_run"], ("%.2f" % ratio) if ratio else "-",
                  1e6 * (m["t_iter"] or 0)), flush=True)
print("TEST1_DONE", flush=True)
