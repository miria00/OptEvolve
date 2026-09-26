"""SkyDiscover evaluator for spec-driven operator synthesis, v2: certificate feedback names the violated contract clauses.
 Runs in the isolated SkyDiscover env and shells out to the
solver env for certification, so neither environment's packages touch the other.

The SPEC comes from the cost model, not from open-ended fitness: an operator is only useful if it is certified exact
AND at least SPEC_SPEEDUP times cheaper per call than the library operator, the per-call budget derived from c* in the
break-even experiment. The score is therefore shaped around the spec:
  - does not compile or not exact          -> small partial credit (so the search can climb out of broken code)
  - certified but slower than the spec      -> 1 + progress toward the spec
  - certified and meets the spec            -> 2 + log2 of the speedup beyond it
The artifact carries the failing condition as text, which SkyDiscover's context builder feeds back to the model.
"""
import json, math, os, subprocess

SOLVER_PY = "/home/miria/jaxenv2/bin/python"
OPCHECK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "opcheck.py")
SPEC_SPEEDUP = float(os.environ.get("SPEC_SPEEDUP", "3.0"))
# the cost model's full ladder for the target instance family (rung label, required speedup); feedback names the rungs
LADDER = [tuple(x.split("=")) for x in os.environ.get("SPEC_LADDER", "").split(",") if "=" in x]
LOCK = os.environ.get("SPEC_LOCK", "/tmp/claude-1000/opcheck.lock")   # one timing run at a time on the device


def evaluate(program_path):
    env = dict(os.environ)
    env.pop("LD_LIBRARY_PATH", None)
    env.setdefault("JAX_PLATFORMS", "cuda,cpu")
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    try:
        import fcntl
        os.makedirs(os.path.dirname(LOCK), exist_ok=True)
        with open(LOCK, "w") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            p = subprocess.run([SOLVER_PY, OPCHECK, program_path], capture_output=True, text=True, timeout=600, env=env)
        line = [l for l in p.stdout.strip().splitlines() if l.startswith("{")][-1]
        r = json.loads(line)
    except Exception as e:
        return {"combined_score": 0.0, "certified": 0.0, "error_text": f"checker crashed: {e}"}
    if r.get("compiles", 0.0) < 1.0:
        score, why = 0.05, "does not compile or crashed: " + r.get("error", "")[-300:]
    elif r.get("certified", 0.0) < 1.0:
        score = 0.1 + 0.4 * max(0.0, min(1.0, -math.log10(max(r["max_err"], 1e-16)) / 9.0))
        fails = r.get("failing_clauses") or {}
        listed = "; ".join(f"{k} (error {v:.1e})" for k, v in sorted(fails.items(), key=lambda kv: -kv[1])[:8])
        why = (f"NOT EXACT: max relative error {r['max_err']:.2e} (need <= 1e-9), feasibility {r['max_feas']:.2e}. "
               f"Contract clauses violated: {listed or 'see error'}")
    else:
        if r.get("speedup_target"):
            s = r["speedup_target"]
            where = (f"on the target block: {r['cand_us_target']:.1f} us per call vs {r['ref_us_target']:.1f} us for "
                     f"60-step bisection (synthetic n=1k: {r['speedup_1k']:.2f}x, n=10k: {r['speedup_10k']:.2f}x)")
        else:
            s = math.sqrt(max(r["speedup_1k"], 1e-6) * max(r["speedup_10k"], 1e-6))
            where = "geometric mean over synthetic n=1k and n=10k"
        met = [lab for lab, need in LADDER if s >= float(need)]
        unmet = [(lab, float(need)) for lab, need in LADDER if s < float(need)]
        ladder_txt = (f" Ladder rungs met: {met or 'none'}; next rung {unmet[0][0]} needs {unmet[0][1]:.2f}x."
                      if LADDER and unmet else (" All ladder rungs met." if LADDER else ""))
        if s < SPEC_SPEEDUP:
            score = 1.0 + max(0.0, s) / SPEC_SPEEDUP
            why = f"exact, but speedup {s:.2f}x is below the spec {SPEC_SPEEDUP:.2f}x ({where}).{ladder_txt}"
        else:
            score = 2.0 + math.log2(s / SPEC_SPEEDUP)
            why = f"MEETS SPEC: exact and {s:.2f}x faster than 60-step bisection ({where}).{ladder_txt}"
    out = {"combined_score": float(score), "certified": float(r.get("certified", 0.0)),
           "max_err": float(r.get("max_err", 1.0)), "speedup_1k": float(r.get("speedup_1k", 0.0)),
           "speedup_10k": float(r.get("speedup_10k", 0.0)), "speedup_target": float(r.get("speedup_target") or 0.0),
           "cand_us_target": float(r.get("cand_us_target") or 0.0)}
    try:
        from skydiscover.evaluation.evaluation_result import EvaluationResult
        return EvaluationResult(metrics=out, artifacts={"feedback": why})
    except Exception:
        return out
