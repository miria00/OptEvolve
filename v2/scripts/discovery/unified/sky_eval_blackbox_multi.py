"""SkyDiscover evaluator for the AGENT-ALONE arm scored on a suite (see e2e_eval_multi.py). Score = geometric-mean
end-to-end speedup over the best standard formulation across the suite (failures count as 10x slower). Feedback lists
per-case outcomes so the agent can see where it fails, which is exactly what an agent without the tool can see."""
import json, os, subprocess

PY = "/home/miria/jaxenv2/bin/python"
HERE = os.path.dirname(os.path.abspath(__file__))
LOCK = os.environ.get("SPEC_LOCK", "/tmp/claude-1000/opcheck.lock")


def evaluate(program_path):
    env = dict(os.environ)
    env.pop("LD_LIBRARY_PATH", None)
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    try:
        import fcntl
        with open(LOCK, "w") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            p = subprocess.run([PY, os.path.join(HERE, "e2e_eval_multi.py"), program_path], capture_output=True,
                               text=True, timeout=1800, env=env)
        r = json.loads([l for l in p.stdout.strip().splitlines() if l.startswith("{")][-1])
    except Exception as e:
        return {"combined_score": 0.0, "failed": 99.0, "error_text": f"evaluator crashed: {e}"}
    why = "; ".join(f"{c['case']}: " + (f"solved {c['T']:.3f} s ({c['ratio']:.2f}x vs standard)" if c.get("accepted")
                                         else "FAILED (no accepted answer)") for c in r["cases"])
    out = {"combined_score": float(r["geo_ratio"]), "failed": float(r["failed"])}
    try:
        from skydiscover.evaluation.evaluation_result import EvaluationResult
        return EvaluationResult(metrics=out, artifacts={"feedback": why})
    except Exception:
        return out
