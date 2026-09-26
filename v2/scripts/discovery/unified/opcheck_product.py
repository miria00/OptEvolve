"""Certificate for a PRODUCT projection candidate: project(v, lo, hi, a, bid, b_lo, b_hi, K) onto the product over
blocks of {lo <= x <= hi, a_k'x in [b_lo_k, b_hi_k]}. Reference: the library product projection run with 400 bisection
steps (itself certified to 1.3e-14 against an independent per-block reference, test_product.py). Clauses: many small
blocks, one huge block, mixed-sign coefficients, infinite bounds, equality and slab rows, feasible inputs, unsorted
block ids. A test, not a proof. Prints one JSON line (same fields as opcheck.py; no timing: callers time in-solver)."""
import importlib.util, json, os, sys, traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from uni import product_project

EPS = float(os.environ.get("OPCHECK_EPS", "1e-9"))


def make(rng, K, size_lo, size_hi, signs, inf_frac, slab_frac, feasible_input=False, shuffle=False):
    sizes = rng.integers(size_lo, size_hi + 1, K)
    n = int(sizes.sum())
    bid = np.repeat(np.arange(K), sizes)
    a = np.abs(rng.standard_normal(n)) + 0.1
    if signs:
        a *= np.where(rng.random(n) < 0.5, -1.0, 1.0)
    lo = rng.standard_normal(n) - 1.0
    hi = lo + rng.random(n) * 3 + 0.01
    hi = np.where(rng.random(n) < inf_frac, np.inf, hi)
    xf = np.where(np.isfinite(hi), lo + rng.random(n) * np.where(np.isfinite(hi), hi - lo, 1.0), lo + rng.random(n))
    s = np.bincount(bid, weights=a * xf, minlength=K)
    slab = rng.random(K) < slab_frac
    bl, bu = s - np.where(slab, rng.random(K), 0.0), s + np.where(slab, rng.random(K), 0.0)
    v = xf.copy() if feasible_input else rng.standard_normal(n) * rng.choice([0.1, 3.0, 100.0])
    if shuffle:
        p = rng.permutation(n)
        v, lo, hi, a, bid = v[p], lo[p], hi[p], a[p], bid[p]
    return v, lo, hi, a, bid, bl, bu, K


def main(path):
    res = {"compiles": 0.0, "max_err": 1.0, "max_feas": 1.0, "error": "", "failing_clauses": {}}
    try:
        spec = importlib.util.spec_from_file_location("cand", path)
        mod = importlib.util.module_from_spec(spec)
        mod.__dict__.update({"jax": jax, "jnp": jnp, "np": np})
        spec.loader.exec_module(mod)
        rng = np.random.default_rng(0)
        clauses = []
        for t in range(3):
            clauses.append(("many small blocks, a>0", make(rng, 400, 2, 17, False, 0.0, 0.0)))
            clauses.append(("many small blocks, mixed-sign a", make(rng, 400, 2, 17, True, 0.0, 0.0)))
            clauses.append(("infinite upper bounds", make(rng, 200, 2, 10, True, 0.4, 0.0)))
            clauses.append(("slab rows", make(rng, 200, 2, 10, True, 0.2, 0.6)))
            clauses.append(("one huge block", make(rng, 1, 5000, 5000, True, 0.1, 0.0)))
            clauses.append(("feasible inputs", make(rng, 300, 2, 12, True, 0.2, 0.3, feasible_input=True)))
            clauses.append(("unsorted block ids", make(rng, 300, 2, 12, True, 0.2, 0.3, shuffle=True)))
            clauses.append(("single-element blocks", make(rng, 500, 1, 1, True, 0.2, 0.3)))
        worst, feas = {}, 0.0
        for name, (v, lo, hi, a, bid, bl, bu, K) in clauses:
            args = tuple(jnp.asarray(z) for z in (v, lo, hi, a, bid, bl, bu))
            x = np.asarray(mod.project(*args, K))
            xr = np.asarray(product_project(*args, K, iters=400, expand=60, polish=3))
            err = float(np.max(np.abs(x - xr)) / (1 + np.max(np.abs(xr)))) if x.shape == xr.shape and np.all(np.isfinite(x)) else 1.0
            sums = np.bincount(bid, weights=a * x, minlength=K)
            f = max(float(np.max(np.maximum(lo - x, 0))), float(np.max(np.where(np.isinf(hi), 0, np.maximum(x - hi, 0)))),
                    float(np.max(np.maximum(bl - sums, 0) / (1 + np.abs(bl)))), float(np.max(np.maximum(sums - bu, 0) / (1 + np.abs(bu)))))
            worst[name] = max(worst.get(name, 0.0), err)
            feas = max(feas, f)
        res.update({"compiles": 1.0, "max_err": max(worst.values()), "max_feas": feas,
                    "failing_clauses": {k: v for k, v in worst.items() if not v <= EPS}})
    except Exception:
        res["error"] = traceback.format_exc()[-600:]
    res["certified"] = 1.0 if res["compiles"] == 1.0 and res["max_err"] <= EPS and res["max_feas"] <= 1e-7 else 0.0
    print(json.dumps(res))


if __name__ == "__main__":
    main(sys.argv[1])
