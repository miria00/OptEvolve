"""SkyDiscover evaluator v3 for the agent's evolve action: certificate over the contract, then IN-SOLVER cost.

Spec comes from the router's feedback: the relocated formulation's per-iteration budget on this device for this case
(SPEC_CASE, SPEC_FORMULATION, SPEC_BUDGET_PER_ITER seconds). Score:
  does not run                      0.05
  not exact                         0.1 .. 0.5 by digits of accuracy; feedback names the violated contract clauses
  certified, over budget            1 + budget / per_iter  (< 2)
  certified, meets budget           2 + log2(budget / per_iter)
"""
import json, math, os, subprocess

PY = "/home/miria/jaxenv2/bin/python"
HERE = os.path.dirname(os.path.abspath(__file__))
CASE = os.environ.get("SPEC_CASE")
FORM = os.environ.get("SPEC_FORMULATION", "reloc")
BUDGET = float(os.environ.get("SPEC_BUDGET_PER_ITER", "0"))
LOCK = os.environ.get("SPEC_LOCK", "/tmp/claude-1000/opcheck.lock")


def _run(args, extra_env):
    env = dict(os.environ, **extra_env)
    env.pop("LD_LIBRARY_PATH", None)
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    p = subprocess.run([PY] + args, capture_output=True, text=True, timeout=900, env=env)
    return json.loads([l for l in p.stdout.strip().splitlines() if l.startswith("{")][-1])


def evaluate(program_path):
    try:
        import fcntl
        with open(LOCK, "w") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            cert = _run([os.path.join(HERE, os.environ.get("CERT_SCRIPT", "opcheck.py")), program_path],
                        {"OPCHECK_NO_TIMING": "1"})
            cal = _run([os.path.join(HERE, "insolver_cal.py"), CASE, FORM, program_path], {}) \
                if cert.get("certified") == 1.0 else None
    except Exception as e:
        return {"combined_score": 0.0, "certified": 0.0, "error_text": f"evaluator crashed: {e}"}
    if cert.get("compiles", 0.0) < 1.0:
        score, why = 0.05, "does not compile or crashed: " + (cert.get("error") or "")[-300:]
    elif cert.get("certified", 0.0) < 1.0:
        score = 0.1 + 0.4 * max(0.0, min(1.0, -math.log10(max(cert["max_err"], 1e-16)) / 9.0))
        fails = cert.get("failing_clauses") or {}
        listed = "; ".join(f"{k} (error {v:.1e})" for k, v in sorted(fails.items(), key=lambda kv: -kv[1])[:8])
        why = f"NOT EXACT: max relative error {cert['max_err']:.2e} (need <= 1e-9). Violated clauses: {listed}"
    elif not cal or not cal.get("ok"):
        score, why = 0.6, "certified, but the in-solver run failed: " + str((cal or {}).get("error"))[-300:]
    else:
        r = BUDGET / cal["per_iter_s"]
        score = 1.0 + r if r < 1.0 else 2.0 + math.log2(r)
        why = (f"certified; in-solver per-iteration cost {1e6 * cal['per_iter_s']:.1f} us vs budget "
               f"{1e6 * BUDGET:.1f} us on {CASE} ({'MEETS' if r >= 1 else 'over'} the spec); setup {cal['setup_s']:.3f} s")
    out = {"combined_score": float(score), "certified": float(cert.get("certified", 0.0)),
           "per_iter_us": float(1e6 * cal["per_iter_s"]) if cal and cal.get("ok") else 0.0}
    try:
        from skydiscover.evaluation.evaluation_result import EvaluationResult
        return EvaluationResult(metrics=out, artifacts={"feedback": why})
    except Exception:
        return out
