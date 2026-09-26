"""LLM-guided mutation backend against the local vLLM endpoint.

Prompt = current best genomes + their metrics + the mutation menu + the
JSON schema/validity card (atlas.genome.SCHEMA_DOC). The model replies
with a full JSON genome or a JSON patch (a partial object without
"scheme", merged onto the current best). Defensive by design for a weak
1.5B model: strict first-balanced-{...} extraction, schema validation,
Stage-0 gate, ONE retry with the error message injected, then a
RandomBackend fallback proposal - the loop never crashes on model garbage.

The HTTP call is plain requests against an OpenAI-compatible
/chat/completions route; inject `http_post` to mock it in unit tests.
"""
from __future__ import annotations

import json
from typing import Callable, List, Optional

from atlas.genome import (
    SCHEMA_DOC,
    Genome,
    GenomeFormatError,
    canonical_json,
    content_hash,
    genome_from_dict,
    genome_to_dict,
    validate,
)
from atlas.evolve.random_backend import RandomBackend

MUTATION_MENU = """Mutation menu (pick ONE or TWO small moves):
- switch base scheme (fbs/drs/pdhg/condat_vu; dys only on problems
  without a linear map, so NOT on the TV cell)
- scale tau or sigma (e.g. x0.5, x2, x10)
- rebalance tau vs sigma keeping the product fixed
- set/unset over-relaxation rho (e.g. 1.5, 1.8)
- toggle wrapper.halpern (rho allowed if nonexpansive)
- with halpern on: restart "fixed_k" + restart_k (25..400) resets the
  anchor every k iterations (often faster than the plain anchor)
- MIX two schemes: {"scheme":"mix","mix":{"kind":"avg","lambda":0.5,
  "scheme_a":{...},"scheme_b":{...}}} blends lambda*T_a+(1-lambda)*T_b;
  parents must be the same scheme, or pdhg+condat_vu with IDENTICAL
  (tau, sigma)
- FINITE-PREFIX SWITCH: {"scheme":"mix","mix":{"kind":"switch_k","K":300,
  "aggressive":{...},"tail":{...}}}: aggressive scheme for the first K
  iterations (big steps allowed, side conditions exempt), then the tail
  (must be fully valid) forever; same-scheme or pdhg/condat_vu branches
- METRIC gene (LP cells only, base pdhg/condat_vu genomes):
  "metric": {"kind":"ruiz","iters":10} (equilibrate the constraint
  matrix) or {"kind":"pock_chambolle","alpha":1.0}; solves the rescaled
  problem, certificate stays on the original gap; helps on badly scaled
  instances
- adjust alpha in [0,1]
- BACKEND gene (execution policy; on hardware with a large f32/f64
  throughput gap prefer f32 iterates): "backend": {"precision":
  "f64"|"f32_cert64"|"schedule", "K": 100, "theta": 0.01 (schedule only,
  in (tolerance, 1))}; the certificate ALWAYS stays f64; bf16/f16 are
  refused; switch_k genomes take only "f64"; omit the block for the
  plain f64 baseline"""

_SYSTEM = (
    "You tune parameters of certified convex-optimization solvers "
    "(TV denoising and LP cells). You reply with EXACTLY ONE JSON object "
    "and no other text."
)


class LLMBackend:
    """SearchBackend: LLM-proposed mutations, gated and fallback-protected."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        model: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        api_key: str = "EMPTY",
        temperature: float = 0.8,
        max_tokens: int = 400,
        timeout: float = 60.0,
        seed: int = 0,
        fallback: Optional[RandomBackend] = None,
        http_post: Optional[Callable[[dict], str]] = None,
        archive_size: int = 8,
        n_prompt_best: int = 3,
        problem_kind: str = "tv_denoise",
        hardware: Optional[dict] = None,
        repair_feedback: bool = False,
        hardware_context: str = "measured",
        fallback_seed_offset: int = 100003,
        chat_template_kwargs: Optional[dict] = None,
    ):
        # REPAIR FEEDBACK IS OFF BY DEFAULT (2026-09-04). The controlled
        # ablation (scripts/repair_ablation.py, n=200 paired rejection
        # contexts on Qwen2.5-Coder-1.5B-Instruct) found that injecting the
        # rejection diagnostic and retrying is WORSE than discarding the
        # reply and resampling the clean prompt:
        #   no feedback   35.0% first retry / 52.0% within two
        #   generic       21.5%             / 42.0%
        #   exact error   21.5%             / 36.0%
        # McNemar exact two-sided, exact vs none: p = 0.0016 / 0.0007, and
        # the effect is STRONGEST on the 112 genuine gate rejections
        # (p = 0.012 / 0.0055). The exact violated inequality is also
        # indistinguishable from a generic "invalid configuration" string
        # (p = 1 / 0.126), so the diagnostic content buys nothing here.
        # Set True to reproduce the pre-ablation behavior or to re-test on a
        # larger proposer, where the conclusion may well differ.
        self.repair_feedback = repair_feedback
        self.chat_template_kwargs = chat_template_kwargs
        # Context Builder hardware record (atlas.backend.device.detect_device
        # dict): the model must SEE the target hardware for backend genes to
        # be anything but blind (CONTRACT.md TODO 6). None = no hardware line.
        self.hardware = hardware
        if hardware_context not in ("none", "identity", "measured"):
            raise ValueError("hardware_context must be none, identity, or measured")
        self.hardware_context = hardware_context
        self.problem_kind = problem_kind
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        # Base seed for LLM sampling: every request carries a deterministic
        # per-call "seed" (base + call counter), so identical runs replay
        # identical proposals on seed-honoring servers (e.g. vLLM).
        self.seed = seed
        self.fallback_seed_offset = int(fallback_seed_offset)
        self.fallback = (
            fallback
            if fallback is not None
            else RandomBackend(seed=seed + self.fallback_seed_offset,
                               problem_kind=problem_kind)
        )
        self._http_post = http_post if http_post is not None else self._default_http_post
        self.archive_size = archive_size
        self.n_prompt_best = n_prompt_best
        # archive of (fitness, genome, metrics), best first
        self.archive: list[tuple[float, Genome, dict]] = []
        # counters (the loop reads per-generation deltas)
        self.stage0_rejections = 0
        self.n_fallbacks = 0
        self.n_llm_calls = 0
        self.token_usage = {"prompt_tokens": 0, "completion_tokens": 0}
        self.last_transcript: list[dict] = []  # [{"prompt":..., "reply":...}, ...]

    def reseed(self, seed: int) -> None:
        self.seed = seed
        self.fallback.reseed(seed + self.fallback_seed_offset)

    # -- HTTP --------------------------------------------------------------

    def _default_http_post(self, payload: dict) -> str:
        import requests

        r = requests.post(
            self.base_url + "/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=self.timeout,
        )
        r.raise_for_status()
        response = r.json()
        for key in self.token_usage:
            self.token_usage[key] += int(response.get("usage", {}).get(key, 0))
        msg = response["choices"][0]["message"]
        content = msg.get("content")
        # REASONING-MODEL TRAP (2026-09-24). A thinking model can spend the
        # whole budget in its reasoning channel and return content=None, which
        # is indistinguishable from "the model said nothing" downstream.  The
        # answer is usually still in the reasoning channel, so salvage it.
        # Applied in the transport, so every arm of every ablation gets it.
        if not content:
            content = msg.get("reasoning_content") or msg.get("reasoning") or ""
        return content

    def _chat(self, user_msg: str) -> str:
        self.n_llm_calls += 1
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            # Deterministic per-call sampling seed (OpenAI-compatible field,
            # honored by vLLM): same run -> same seed sequence -> same
            # proposals; distinct calls still sample differently.
            "seed": self.seed + self.n_llm_calls,
        }
        # Thinking models must be told to stop thinking, or they exhaust the
        # token budget before emitting an answer.  Harmless on models that do
        # not read it.
        if self.chat_template_kwargs:
            payload["chat_template_kwargs"] = dict(self.chat_template_kwargs)
        reply = self._http_post(payload)
        self.last_transcript.append({"prompt": user_msg, "reply": reply})
        return reply

    # -- prompt ------------------------------------------------------------

    def _best_block(self) -> str:
        if not self.archive:
            return "No evaluated genomes yet: propose a sensible first genome."
        lines = []
        for i, (fit, g, metrics) in enumerate(self.archive[: self.n_prompt_best], 1):
            m = {k: metrics[k] for k in list(metrics)[:4]}  # keep prompt small
            lines.append(f"{i}. fitness={fit:.6g} metrics={json.dumps(m)}")
            lines.append(f"   genome: {canonical_json(g)}")
        return "Current best genomes (fitness HIGHER is better):\n" + "\n".join(lines)

    def _hardware_block(self) -> Optional[str]:
        if not self.hardware or self.hardware_context == "none":
            return None
        hw = self.hardware
        if "operators" in hw or self.hardware_context == "identity":
            from atlas.backend.hardware import context_block
            return context_block(hw, self.hardware_context)
        t32, t64 = hw.get("tflops_f32"), hw.get("tflops_f64")
        ratio = (
            f"{t32 / t64:.0f}x" if t32 and t64 and t64 > 0 else "unknown"
        )
        return (
            "Target hardware: kind="
            f"{hw.get('kind')!r}, platform={hw.get('platform')!r}, "
            f"memory_gb={hw.get('memory_gb')}, tflops_f32={t32}, "
            f"tflops_f64={t64} (f32/f64 throughput ratio {ratio}). "
            "A large ratio favors backend precision 'f32_cert64' or "
            "'schedule'; certificates always run at f64."
        )

    def _build_prompt(self, feedback: Optional[str]) -> str:
        hw_block = self._hardware_block()
        parts = ([hw_block] if hw_block else []) + [
            SCHEMA_DOC,
            self._best_block(),
            MUTATION_MENU,
            "Propose ONE mutated genome likely to improve fitness. Reply with "
            "ONLY the JSON genome object (or a partial JSON patch of the best "
            "genome).",
        ]
        if feedback:
            parts.append(
                f"Your previous reply was INVALID: {feedback}\n"
                "Fix it and reply with ONLY one corrected JSON genome object."
            )
        return "\n\n".join(parts)

    # -- parsing -----------------------------------------------------------

    @staticmethod
    def _extract_json_block(text: str) -> Optional[str]:
        """First balanced {...} block, string-and-escape aware."""
        start = text.find("{")
        if start < 0:
            return None
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            c = text[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
            elif c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        return None

    def _parse_reply(self, reply: str):
        """-> (genome, None) or (None, error_message)."""
        block = self._extract_json_block(reply)
        if block is None:
            return None, "no JSON object found in the reply"
        try:
            d = json.loads(block)
        except json.JSONDecodeError as e:
            return None, f"malformed JSON: {e}"
        if not isinstance(d, dict):
            return None, "reply JSON must be an object"
        if "scheme" not in d:  # JSON patch onto the current best
            if not self.archive:
                return None, "a partial patch needs a base genome; send a full genome"
            base = genome_to_dict(self.archive[0][1])
            for key in ("params", "wrapper"):
                sub = d.get(key)
                if isinstance(sub, dict):
                    base[key].update(sub)
            for key in set(d) - {"params", "wrapper"}:
                base[key] = d[key]
            if base.get("mix") is not None:
                base["scheme"] = "mix"
            d = base
        try:
            g = genome_from_dict(d)
        except GenomeFormatError as e:
            return None, str(e)
        v = validate(g, problem_kind=self.problem_kind)
        if not v.ok:
            self.stage0_rejections += 1
            return None, v.error
        return g, None

    # -- SearchBackend -----------------------------------------------------

    def _propose_one(self) -> Genome:
        # Default path: reject -> discard -> CLEAN RESAMPLE (no diagnostic
        # injected). See the repair_feedback note in __init__ for the ablation
        # that settled this.
        feedback = None
        for _attempt in range(2):  # initial try + ONE resample (or repair)
            try:
                reply = self._chat(self._build_prompt(feedback))
            except Exception as e:  # server down, timeout, bad payload, ...
                break
            g, err = self._parse_reply(reply)
            if g is not None:
                return g
            feedback = err if self.repair_feedback else None
        self.n_fallbacks += 1
        return self.fallback.ask(1)[0]

    def ask(self, k: int) -> List[Genome]:
        self.last_transcript = []
        return [self._propose_one() for _ in range(k)]

    def tell(self, genome: Genome, fitness: float, metrics: dict) -> None:
        # Dedup by canonical content hash: re-telling the same genome (the
        # loop's elitism re-tells the champion every generation) must not
        # flood the archive with copies and evict genuine runners-up - the
        # prompt's "current best genomes" block needs DISTINCT genomes.
        fitness = float(fitness)
        h = content_hash(genome)
        for i, (fit0, g0, _m0) in enumerate(self.archive):
            if content_hash(g0) == h:
                if fitness > fit0:
                    self.archive[i] = (fitness, genome, dict(metrics))
                break
        else:
            self.archive.append((fitness, genome, dict(metrics)))
        self.archive.sort(key=lambda t: t[0], reverse=True)
        del self.archive[self.archive_size :]
