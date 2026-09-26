"""SimpleTES evaluator for the exact projection onto {lo <= x <= hi, a'x = b}, reusing the SkyDiscover evaluators.

SimpleTES contract: evaluate(filepath) -> JSON-serializable dict with a finite combined_score, run in a fresh
subprocess per candidate. Every metric key is printed verbatim into later prompts, so the SkyDiscover text feedback
artifact is carried as metrics["feedback"]. A truthy metrics["error"] makes SimpleTES score the candidate -inf, keep it
out of trajectory history, and fold the message into its failure-pattern summary.

Mode (env SIMPLETES_ARM_EVAL):
  blackbox_multi  sky_eval_blackbox_multi.evaluate, unchanged (end-to-end time over the E2E_CASES suite, GPU, lock)
  insolver        sky_eval_v3.evaluate, unchanged (certificate, then in-solver per-iteration cost, GPU, lock)
  stub            certificate only on CPU (opcheck.py, JAX_PLATFORMS=cpu, OPCHECK_NO_TIMING=1), scored like the
                  certificate branch of sky_eval_v3 with certified = 1.0; for plumbing checks, never for results

SIMPLETES_ARM_FAILURE_AS_ERROR=1 (default) maps "never produced an answer" outcomes (crash, does not compile, or every
suite case failed) to metrics["error"], which is how SimpleTES's own task evaluators report failures. Set it to 0 to
keep the SkyDiscover floor scores (0.05 / 0.1) as ordinary finite scores instead.
"""
import importlib.util
import json
import math
import os
import subprocess
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
UNIFIED = os.path.normpath(os.path.join(HERE, "..", "unified"))
JAX_PY = "/home/miria/jaxenv2/bin/python"
MODES = ("blackbox_multi", "insolver", "stub")


def _install_evaluation_result_shim():
    # The SkyDiscover evaluators return EvaluationResult(metrics, artifacts={"feedback": ...}) only when skydiscover
    # imports, and a bare metrics dict otherwise, which in this venv would silently drop the feedback text. A minimal
    # stand-in keeps their code unmodified and preserves the artifact.
    try:
        import skydiscover.evaluation.evaluation_result  # noqa: F401
        return
    except ImportError:
        pass

    class EvaluationResult:
        def __init__(self, metrics, artifacts=None):
            self.metrics, self.artifacts = metrics, artifacts or {}

    names = ("skydiscover", "skydiscover.evaluation", "skydiscover.evaluation.evaluation_result")
    for name in names:
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules[names[-1]].EvaluationResult = EvaluationResult


def _clean_child_environment():
    # SimpleTES puts its repo on PYTHONPATH for evaluator subprocesses. Inherited by the jaxenv2 children, it would make
    # them import SimpleTES's sitecustomize (and its litellm dependency chain), so it is removed before delegating.
    for key in ("PYTHONPATH", "SIMPLETES_CAPTURE_CONSTRUCTION_PATH", "SIMPLETES_SHARED_CONSTRUCTION_PATH",
                "SIMPLETES_SHARED_CONSTRUCTION_MAX_BYTES"):
        os.environ.pop(key, None)


def _load_sky(module_name):
    spec = importlib.util.spec_from_file_location(f"arm_{module_name}", os.path.join(UNIFIED, module_name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _split(result):
    if hasattr(result, "metrics"):
        return dict(result.metrics), str((result.artifacts or {}).get("feedback", ""))
    metrics = dict(result)
    return metrics, str(metrics.pop("error_text", ""))


def _stub_certificate(program_path):
    env = dict(os.environ, JAX_PLATFORMS="cpu", CUDA_VISIBLE_DEVICES="", OPCHECK_NO_TIMING="1",
               XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1",
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    env.pop("LD_LIBRARY_PATH", None)
    # Lowest CPU priority: the host also dispatches kernels for timing-sensitive GPU experiments.
    p = subprocess.run([JAX_PY, os.path.join(UNIFIED, "opcheck.py"), program_path], capture_output=True, text=True,
                       timeout=900, env=env, preexec_fn=lambda: os.nice(19))
    lines = [l for l in p.stdout.strip().splitlines() if l.startswith("{")]
    if not lines:
        return {"combined_score": 0.0, "certified": 0.0}, f"evaluator crashed: {p.stderr[-300:]}"
    cert = json.loads(lines[-1])
    if cert.get("compiles", 0.0) < 1.0:
        return ({"combined_score": 0.05, "certified": 0.0},
                "does not compile or crashed: " + (cert.get("error") or "")[-300:])
    if cert.get("certified", 0.0) < 1.0:
        score = 0.1 + 0.4 * max(0.0, min(1.0, -math.log10(max(cert["max_err"], 1e-16)) / 9.0))
        fails = cert.get("failing_clauses") or {}
        listed = "; ".join(f"{k} (error {v:.1e})" for k, v in sorted(fails.items(), key=lambda kv: -kv[1])[:8])
        return ({"combined_score": float(score), "certified": 0.0},
                f"NOT EXACT: max relative error {cert['max_err']:.2e} (need <= 1e-9). Violated clauses: {listed}")
    return {"combined_score": 1.0, "certified": 1.0}, "certified (CPU stub: no timing measured)"


def _is_failure(mode, metrics, feedback):
    if feedback.startswith("evaluator crashed"):
        return True
    if mode in ("insolver", "stub"):
        return feedback.startswith("does not compile or crashed")
    # blackbox_multi: every included suite case failed, so the geometric mean sits exactly at the 0.1 failure ratio.
    return metrics.get("failed", 0.0) >= 1.0 and abs(metrics.get("combined_score", 0.0) - 0.1) < 1e-9


def evaluate(program_path):
    mode = os.environ.get("SIMPLETES_ARM_EVAL", "")
    try:
        if mode not in MODES:
            raise ValueError(f"SIMPLETES_ARM_EVAL must be one of {MODES}, got {mode!r}")
        _clean_child_environment()
        if mode == "stub":
            metrics, feedback = _stub_certificate(program_path)
        else:
            _install_evaluation_result_shim()
            module = "sky_eval_blackbox_multi" if mode == "blackbox_multi" else "sky_eval_v3"
            metrics, feedback = _split(_load_sky(module).evaluate(program_path))
    except Exception as exc:
        metrics, feedback = {"combined_score": 0.0}, f"evaluator crashed: {type(exc).__name__}: {exc}"
    metrics = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
    metrics["feedback"] = feedback
    if os.environ.get("SIMPLETES_ARM_FAILURE_AS_ERROR", "1") == "1" and _is_failure(mode, metrics, feedback):
        metrics["error"] = feedback or "candidate failed"
    return metrics
