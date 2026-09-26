"""SkyDiscover evaluator for the BLACK-BOX arm: fitness is end-to-end solve time on the target instance only.

Score: does not run -> 0.05; wrong or non-converged answer -> 0.1; accepted -> T_fixed / T, the speedup over the best
fixed formulation (the time the agent is asked to beat). Same lock as the spec arm so the two never time concurrently.
"""
import json, os, subprocess

SOLVER_PY = "/home/miria/jaxenv2/bin/python"
E2E = os.path.join(os.path.dirname(os.path.abspath(__file__)), "e2e_eval.py")
T_FIXED = float(os.environ.get("E2E_T_FIXED", "0.242"))
T_BISECT = float(os.environ.get("E2E_T_BISECT", "0.805"))
LOCK = os.environ.get("SPEC_LOCK", "/tmp/claude-1000/opcheck.lock")


def evaluate(program_path):
    env = dict(os.environ)
    env.pop("LD_LIBRARY_PATH", None)
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    try:
        import fcntl
        with open(LOCK, "w") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            p = subprocess.run([SOLVER_PY, E2E, program_path], capture_output=True, text=True, timeout=900, env=env)
        r = json.loads([l for l in p.stdout.strip().splitlines() if l.startswith("{")][-1])
    except Exception as e:
        return {"combined_score": 0.0, "accepted": 0.0, "error_text": f"evaluator crashed: {e}"}
    if r.get("error"):
        score, why = 0.05, "crashed in the solver: " + r["error"][-300:]
    elif not r.get("converged"):
        score, why = 0.1, (f"solver did not reach an accepted answer within 20000 iterations "
                           f"(row violation {r.get('rv')}, objective gap {r.get('gap')})")
    else:
        score = T_FIXED / r["total_s"]
        why = (f"accepted: solved in {r['total_s']:.3f} s ({r['N']} iterations). The time to beat is {T_FIXED:.3f} s "
               f"(best fixed formulation); with the current 60-step bisection operator it takes {T_BISECT:.3f} s.")
    out = {"combined_score": float(score), "accepted": float(bool(r.get("converged"))),
           "total_s": float(r.get("total_s") or 0.0), "N": float(r.get("N") or 0.0)}
    try:
        from skydiscover.evaluation.evaluation_result import EvaluationResult
        return EvaluationResult(metrics=out, artifacts={"feedback": why})
    except Exception:
        return out
