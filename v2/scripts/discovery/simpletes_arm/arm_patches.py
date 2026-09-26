"""Accounting and interface shims applied to SimpleTES at launch (run_arm.py). SimpleTES source stays unmodified.

1. Every generation request is logged to SIMPLETES_ARM_TOKEN_LOG (one JSON line per LLM call) whether or not
   --save-llm-io is on, because SimpleTES only records token usage for candidates that pass code extraction, so its own
   checkpoints undercount the LLM budget of failed candidates.
2. Sampling knobs SimpleTES does not expose (top_p, extra_body) are injected from the environment so the arm can match
   the SkyDiscover arm's sampling (temperature 1.0, top_p 0.95) or disable thinking for a cheap dry run.
3. Policy-side LLM calls (reflection, llm_* selectors) are logged too, so an equal-budget comparison can count them.
4. Optional extraction fallback: SimpleTES requires EVOLVE-BLOCK markers in the answer. The shared proxy salvages
   programs from the reasoning trace as a bare ```python block without markers; SkyDiscover accepts those, SimpleTES
   would reject them, so the fallback accepts the last fenced block that defines project(...).

The generation worker runs in a spawn-context process pool. It is referenced by module-level name, so the replacement
must live in an importable module (this one); spawned children inherit sys.path and the environment from the parent.
"""
from __future__ import annotations

import json
import os
import re
import time

TOKEN_LOG_ENV = "SIMPLETES_ARM_TOKEN_LOG"
TOP_P_ENV = "SIMPLETES_ARM_TOP_P"
EXTRA_BODY_ENV = "SIMPLETES_ARM_EXTRA_BODY"
FALLBACK_ENV = "SIMPLETES_ARM_MARKER_FALLBACK"

_FENCE_RE = re.compile(r"```[^\n`]*\r?\n(.*?)```", re.DOTALL)
_PROJECT_DEF_RE = re.compile(r"^def project\(", re.MULTILINE)


def _log(record: dict) -> None:
    path = os.environ.get(TOKEN_LOG_ENV)
    if not path:
        return
    record = {"t": round(time.time(), 3), "pid": os.getpid(), **record}
    # O_APPEND keeps short lines from concurrent pool workers intact without a lock.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(fd, (json.dumps(record, default=str) + "\n").encode())
    finally:
        os.close(fd)


def _usage_dict(resp) -> dict:
    usage = getattr(resp, "usage", None)
    if usage is None:
        return {}
    out = {k: getattr(usage, k, None) for k in ("prompt_tokens", "completion_tokens", "total_tokens")}
    details = getattr(usage, "completion_tokens_details", None)
    if details is not None and getattr(details, "reasoning_tokens", None) is not None:
        out["reasoning_tokens"] = details.reasoning_tokens
    return {k: v for k, v in out.items() if v is not None}


def _sampling_overrides(call_kwargs: dict) -> dict:
    kwargs = dict(call_kwargs)
    if os.environ.get(TOP_P_ENV):
        kwargs["top_p"] = float(os.environ[TOP_P_ENV])
    if os.environ.get(EXTRA_BODY_ENV):
        kwargs["extra_body"] = json.loads(os.environ[EXTRA_BODY_ENV])
    return kwargs


def worker_generate_logged(model, messages, call_kwargs, track_io):
    """Drop-in for simpletes.llm.litellm_client._worker_generate with per-call logging and sampling overrides."""
    from litellm import completion
    from simpletes.llm import litellm_client as lc

    kwargs = _sampling_overrides(call_kwargs)
    t0 = time.time()
    try:
        resp = completion(model=model, messages=messages, **kwargs)
    except Exception as exc:
        _log({"kind": "generate", "ok": False, "wall_s": round(time.time() - t0, 2), "error": str(exc)[:300]})
        return ("error", lc._worker_error_info(exc))
    choices = getattr(resp, "choices", None) or []
    msg = getattr(choices[0], "message", None) if choices else None
    text = (getattr(msg, "content", None) if msg is not None else None) or ""
    usage = _usage_dict(resp)
    _log({"kind": "generate", "ok": True, "wall_s": round(time.time() - t0, 2),
          "finish": getattr(choices[0], "finish_reason", None) if choices else None,
          "content_chars": len(text), "has_markers": "EVOLVE-BLOCK-START" in text, **usage})
    if not choices:
        return ("ok", ("", "", None))
    raw_output = lc._build_raw_output(msg, text)
    return ("ok", (text, raw_output, (usage or None) if track_io else None))


def _worker_generate_batch_refused(*args, **kwargs):
    # n>1 batched requests would bypass per-call accounting; the launcher always streams k candidates as k requests.
    raise RuntimeError("batched n>1 generation is disabled in the SimpleTES arm (use streamed k candidates)")


def _wrap_async_completion(fn, kind):
    async def wrapped(*args, **kwargs):
        t0 = time.time()
        try:
            resp = await fn(*args, **kwargs)
        except Exception as exc:
            _log({"kind": kind, "ok": False, "wall_s": round(time.time() - t0, 2), "error": str(exc)[:300]})
            raise
        _log({"kind": kind, "ok": True, "wall_s": round(time.time() - t0, 2), **_usage_dict(resp)})
        return resp
    return wrapped


def _wrap_sync_completion(fn, kind):
    def wrapped(*args, **kwargs):
        t0 = time.time()
        try:
            resp = fn(*args, **kwargs)
        except Exception as exc:
            _log({"kind": kind, "ok": False, "wall_s": round(time.time() - t0, 2), "error": str(exc)[:300]})
            raise
        _log({"kind": kind, "ok": True, "wall_s": round(time.time() - t0, 2), **_usage_dict(resp)})
        return resp
    return wrapped


def install() -> None:
    """Apply the shims in the launcher process before the engine (and its process pool) is created."""
    import litellm
    from simpletes import generator
    from simpletes.llm import litellm_client
    from simpletes.policies import llm_refine

    litellm_client._worker_generate = worker_generate_logged
    litellm_client._worker_generate_batch = _worker_generate_batch_refused
    # Reflection (TrajectoryPolicyBase._llm_generate) imports acompletion at call time; llm_refine binds completion at
    # import time. Both are auxiliary LLM calls outside --max-generations, so they are logged under their own kinds.
    litellm.acompletion = _wrap_async_completion(litellm.acompletion, "policy_reflection")
    llm_refine.completion = _wrap_sync_completion(llm_refine.completion, "policy_selector")

    original_extract = generator.extract_code_detailed

    def extract_with_fallback(llm_output, evolve_context=None):
        code, reason = original_extract(llm_output, evolve_context)
        if code is not None or reason != "missing_evolve_block_markers" or evolve_context is None:
            return code, reason
        if os.environ.get(FALLBACK_ENV, "1") != "1":
            return code, reason
        blocks = [b for b in _FENCE_RE.findall(llm_output or "") if _PROJECT_DEF_RE.search(b)]
        if not blocks:
            return code, reason
        merged = evolve_context.merge_with_evolved_block(blocks[-1])
        _log({"kind": "extract_fallback", "ok": merged is not None})
        return (merged, "arm_fallback_unmarked_project_block") if merged else (None, reason)

    generator.extract_code_detailed = extract_with_fallback
