"""Certify a candidate projection onto {lo <= x <= hi, a'x = b} and time it against the current operator.

Runs in the solver environment (jaxenv2), called as a subprocess by the SkyDiscover evaluator, which runs in its own
isolated environment. Prints one JSON line.

Contract for a candidate file: define project(v, lo, hi, a, b) using jax / jax.numpy, jit-compatible (static shapes,
no Python branching on traced values). lo, hi are arrays (hi may be +inf), a is a positive array, b a scalar, and the
set is non-empty. Coefficients of a may have EITHER sign (a real family, svm_dual, has both).
Must return the Euclidean projection of v.

Certification is numerical, in float64, and is a TEST not a proof:
  exactness : max |x - x_ref| / (1 + max|x_ref|) over randomized and edge-case inputs, x_ref = 200-step bisection
  feasibility: box violation and |a'x - b| / (1 + |b|)
Timing is RELATIVE: candidate vs the library's 60-step bisection, same device, same process, so contention on a
shared GPU cancels to first order.

Timing is AMORTIZED inside one jitted loop of R calls, as in the solver: timing single jitted calls from Python adds a
dispatch overhead of tens of microseconds that swamps operators in the tens of microseconds (the synthesis targets).
The loop accumulates a weighted checksum of every output coordinate so the compiler cannot prune any of the work.

If OPCHECK_TARGET (an .npz with lo, hi, a, b, V) is set, the block extracted from the target instance is added to the
certification cases and the per-call cost is also measured on it (speedup_target): the spec is about that block on
that device, not about a synthetic one.
"""
import importlib.util, json, os, sys, time, traceback
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from atlas.operators.prox import project_box_hyperplane

EPS_STAR = float(os.environ.get("OPCHECK_EPS", "1e-9"))
REPS = int(os.environ.get("OPCHECK_REPS", "200"))
TARGET = os.environ.get("OPCHECK_TARGET", "")


def looped(f, reps=REPS):
    """One XLA program running f reps times on rotating inputs, as it would run inside the solver loop."""
    def run(V, W, lo, hi, a, b):
        nV = V.shape[0]
        def body(i, acc):
            return acc + jnp.vdot(W, f(V[i % nV], lo, hi, a, b))
        return jax.lax.fori_loop(0, reps, body, jnp.zeros((), V.dtype))
    return jax.jit(run)


def per_call(fl, args, budget_s=0.6):
    jax.block_until_ready(fl(*args))
    reps, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < budget_s:
        jax.block_until_ready(fl(*args))
        reps += 1
    return (time.perf_counter() - t0) / (reps * REPS)


class _SkipTiming(Exception):
    pass


CLAUSES = []                                                  # contract clause of each case, parallel to cases()


def cases(rng):
    out = []
    CLAUSES.clear()
    for n in (7, 64, 1000, 10000):
        for hi_kind in ("one", "inf", "caps"):
            a = np.abs(rng.standard_normal(n)) + 0.1
            lo = np.zeros(n)
            hi = np.ones(n) if hi_kind == "one" else (np.full(n, np.inf) if hi_kind == "inf"
                                                      else rng.random(n) * 2 + 0.05)
            xf = lo + rng.random(n) * np.minimum(hi - lo, 1.0)
            b = float(a @ xf)
            for scale in (0.1, 3.0, 50.0):
                v = rng.standard_normal(n) * scale
                out.append((v, lo, hi, a, b))
                CLAUSES.append(f"a>0, n={n}, hi={'+inf' if hi_kind == 'inf' else ('1' if hi_kind == 'one' else 'mixed finite')}")
    for n in (64, 2000):                                      # MIXED-SIGN coefficients, as in svm_dual's d'alpha = 0
        a = np.where(rng.random(n) < 0.5, -1.0, 1.0) * (np.abs(rng.standard_normal(n)) + 0.1)
        lo, hi = np.zeros(n), np.ones(n)
        b = float(a @ (rng.random(n)))
        for scale in (0.1, 3.0, 50.0):
            out.append((rng.standard_normal(n) * scale, lo, hi, a, b))
            CLAUSES.append(f"mixed-sign a, n={n}")
    if TARGET:                                                # the block extracted from the target instance
        t = np.load(TARGET)
        for i in range(min(6, t["V"].shape[0])):
            out.append((t["V"][i], t["lo"], t["hi"], t["a"], float(t["b"])))
            CLAUSES.append("deployment block (solver inputs)")
    n = 500                                                   # simplex special case and a degenerate tie case
    out.append((rng.standard_normal(n), np.zeros(n), np.full(n, np.inf), np.ones(n), 1.0))
    CLAUSES.append("simplex (a=1, hi=+inf)")
    out.append((np.ones(n), np.zeros(n), np.ones(n), np.ones(n), 250.0))
    CLAUSES.append("degenerate ties")
    return out


def main(path):
    res = {"compiles": 0.0, "max_err": 1.0, "max_feas": 1.0, "speedup_1k": 0.0, "speedup_10k": 0.0,
           "cand_us_1k": None, "ref_us_1k": None, "error": ""}
    try:
        spec = importlib.util.spec_from_file_location("cand", path)
        mod = importlib.util.module_from_spec(spec)
        mod.__dict__.update({"jax": jax, "jnp": jnp, "np": np})     # prelude: code recovered without its import lines
        spec.loader.exec_module(mod)
        cand = jax.jit(mod.project)
        ref200 = jax.jit(lambda v, lo, hi, a, b: project_box_hyperplane(v, lo, hi, a, b, iters=200))
        rng = np.random.default_rng(0)
        errs, feas = [], []
        per_clause = {}
        for ci, (v, lo, hi, a, b) in enumerate(cases(rng)):
            args = tuple(jnp.asarray(z) for z in (v, lo, hi, a)) + (jnp.asarray(b),)
            x = np.asarray(cand(*args))
            xr = np.asarray(ref200(*args))
            if x.shape != xr.shape or not np.all(np.isfinite(x)):
                errs.append(1.0); feas.append(1.0); continue
            errs.append(float(np.max(np.abs(x - xr)) / (1.0 + np.max(np.abs(xr)))))
            boxv = float(max(np.max(np.maximum(lo - x, 0)), np.max(np.where(np.isinf(hi), 0, np.maximum(x - hi, 0)))))
            feas.append(max(boxv, abs(float(a @ x) - b) / (1.0 + abs(b))))
            per_clause[CLAUSES[ci]] = max(per_clause.get(CLAUSES[ci], 0.0), errs[-1])
        res["failing_clauses"] = {k: v for k, v in per_clause.items() if not v <= EPS_STAR}
        res["compiles"] = 1.0
        res["max_err"], res["max_feas"] = max(errs), max(feas)
        if os.environ.get("OPCHECK_NO_TIMING"):
            raise _SkipTiming()
        c_loop = looped(mod.project)
        r_loop = looped(lambda v, lo, hi, a, b: project_box_hyperplane(v, lo, hi, a, b, iters=60))
        sets = []
        for n, key in ((1000, "1k"), (10000, "10k")):
            a = np.abs(rng.standard_normal(n)) + 0.1
            b = float(a @ (rng.random(n) * 0.5))
            sets.append((key, np.stack([rng.standard_normal(n) * 3.0 for _ in range(8)]), np.zeros(n), np.ones(n), a, b))
        if TARGET:
            t = np.load(TARGET)
            sets.append(("target", t["V"], t["lo"], t["hi"], t["a"], float(t["b"])))
        for key, V, lo, hi, a, b in sets:
            W = rng.standard_normal(V.shape[1])
            args = tuple(jnp.asarray(z) for z in (V, W, lo, hi, a)) + (jnp.asarray(b),)
            tc = min(per_call(c_loop, args), per_call(c_loop, args))
            tr = min(per_call(r_loop, args), per_call(r_loop, args))
            tc, tr = min(tc, per_call(c_loop, args)), min(tr, per_call(r_loop, args))
            res[f"speedup_{key}"] = tr / tc
            res[f"cand_us_{key}"], res[f"ref_us_{key}"] = 1e6 * tc, 1e6 * tr
    except _SkipTiming:
        pass                                                  # certificate only: the caller times in the solver
    except Exception:
        res["error"] = traceback.format_exc()[-600:]
    correct = res["compiles"] == 1.0 and res["max_err"] <= EPS_STAR and res["max_feas"] <= 1e-7
    res["certified"] = 1.0 if correct else 0.0
    print(json.dumps(res))


if __name__ == "__main__":
    main(sys.argv[1])
