"""Repair-feedback ablation (P0): does the EXACT violated inequality teach the
proposer to return to the certified region, or would any rejection signal do?

Phase 2 banked a descriptive 43%-within-two-retries repair rate. That number
alone cannot distinguish "the verifier teaches" from "resampling sometimes
works". This runs the controlled version.

PAIRED DESIGN. We first collect N rejection CONTEXTS from the real Phase-2
proposer (same model, same prompt card, same archive). Each context is a base
prompt plus the invalid reply it produced plus the exact classified error.
Every context is then replayed under three feedback conditions:

  exact    the precise violated inequality / format error (Phase-2 behavior)
  generic  "invalid configuration" and nothing more
  none     no corrective feedback at all: the base prompt is simply resampled

All three share the context, the model, the temperature, the archive and the
per-attempt sampling SEED, so the prompt text is the only thing that varies.
The repair seed always differs from the seed that produced the rejection, so
the `none` arm is a genuine resample rather than a replay of the same reply.

Reports first-retry and two-retry repair probability per condition with Wilson
95% intervals, plus McNemar exact tests on the paired arms.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from atlas.genome import Genome, canonical_json, genome_from_dict  # noqa: E402

import run_centerpiece as rc  # noqa: E402
from phase2_common import policy_violation  # noqa: E402

GENERIC_FEEDBACK = "invalid configuration"
CONDITIONS = ("exact", "generic", "none")


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval; correct at the small n and extreme p we expect."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value on discordant pairs (b, c).

    b = A-only successes, c = B-only successes. Under H0 each discordant pair
    is a fair coin, so the count is Binomial(b + c, 1/2).
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2.0**n)
    return min(1.0, 2.0 * tail)


class ProbeBackend(rc.CPLLMBackend):
    """CPLLMBackend with an explicit per-call seed, so paired arms match."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.force_seed: int | None = None

    def _chat(self, user_msg: str) -> str:
        self.n_llm_calls += 1
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": rc.CP_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "seed": (
                self.force_seed
                if self.force_seed is not None
                else self.seed + self.n_llm_calls
            ),
        }
        # Thinking models exhaust the budget in the reasoning channel and
        # return content=None unless told otherwise.  Applied to EVERY arm.
        if getattr(self, "chat_template_kwargs", None):
            payload["chat_template_kwargs"] = dict(self.chat_template_kwargs)
        reply = self._http_post(payload)
        self.last_transcript.append({"prompt": user_msg, "reply": reply})
        return reply

    def prompt_for(self, feedback):
        return self._build_prompt(feedback)

    def classify(self, reply):
        return self._parse_classified(reply)


def build_archive(backend: ProbeBackend) -> None:
    """Seed the prompt archive with the Phase-2 starting point plus two
    evolved genomes, so the rejection contexts look like real generations."""
    entries = [
        # (genome dict, fitness) - vanilla seed and two mid-search genomes
        ({"scheme": "pdhg", "params": {"tau": 0.311958874053,
                                       "sigma": 0.311958874053},
          "wrapper": {"halpern": False, "restart": "none"}}, -0.7352),
        ({"scheme": "pdhg", "params": {"tau": 0.05, "sigma": 2.33183230454,
                                       "rho": 1.5},
          "wrapper": {"halpern": False, "restart": "none"}}, -0.1146),
        ({"scheme": "pdhg", "params": {"tau": 0.01, "sigma": 9.45468304801},
          "wrapper": {"halpern": False, "restart": "none"}}, -0.0822),
    ]
    for d, fit in entries:
        g = genome_from_dict(d)
        backend.tell(g, fit, {"sgm10_time_s": -fit, "solve_rate": 1.0})


def collect_contexts(backend: ProbeBackend, n_want: int, seed0: int,
                     verbose: bool = True) -> list[dict]:
    """Drive the real proposer until n_want invalid replies are collected."""
    contexts: list[dict] = []
    base_prompt = backend.prompt_for(None)
    call = 0
    while len(contexts) < n_want:
        call += 1
        backend.force_seed = seed0 + call
        try:
            reply = backend._chat(base_prompt)
        except Exception as e:  # noqa: BLE001
            print(f"  endpoint error: {e}", flush=True)
            time.sleep(2.0)
            continue
        g, err, klass = backend.classify(reply)
        if g is None:
            contexts.append({
                "idx": len(contexts),
                "reject_seed": seed0 + call,
                "error": err,
                "class": klass,
                "reply_head": reply[:200],
            })
            if verbose and len(contexts) % 10 == 0:
                print(f"  collected {len(contexts)}/{n_want} "
                      f"({call} calls)", flush=True)
    if verbose:
        print(f"  collected {len(contexts)} rejections from {call} calls "
              f"(rejection rate {len(contexts)/call:.1%})", flush=True)
    return contexts


def run_condition(backend: ProbeBackend, ctx: dict, cond: str,
                  repair_seed: int, max_retries: int) -> dict:
    """Replay one rejection context under one feedback condition."""
    err = ctx["error"]
    attempts = []
    for attempt in range(max_retries):
        if cond == "exact":
            fb = err
        elif cond == "generic":
            fb = GENERIC_FEEDBACK
        else:  # none: no corrective feedback whatsoever
            fb = None
        backend.force_seed = repair_seed + attempt
        try:
            reply = backend._chat(backend.prompt_for(fb))
        except Exception as e:  # noqa: BLE001
            attempts.append({"ok": False, "class": "endpoint", "error": str(e)})
            break
        g, err_new, klass = backend.classify(reply)
        attempts.append({
            "ok": g is not None,
            "class": klass,
            "error": err_new,
            "genome": canonical_json(g) if g is not None else None,
        })
        if g is not None:
            return {"condition": cond, "attempts": attempts,
                    "first_ok": attempt == 0, "any_ok": True,
                    "n_attempts": attempt + 1}
        err = err_new  # exact arm follows the NEW error on the next retry
    return {"condition": cond, "attempts": attempts,
            "first_ok": False, "any_ok": False, "n_attempts": len(attempts)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-contexts", type=int, default=120)
    ap.add_argument("--max-retries", type=int, default=2)
    ap.add_argument("--seed", type=int, default=20260904)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    ap.add_argument("--reasoning-effort", default="")
    ap.add_argument("--out", default="results/sprint/phase3/repair_ablation.json")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    ctk = ({"reasoning_effort": args.reasoning_effort}
           if args.reasoning_effort else None)
    backend = ProbeBackend(
        chat_template_kwargs=ctk,
        base_url=args.base_url, model=args.model,
        temperature=args.temperature, seed=args.seed,
        problem_kind="lp_netflow",
    )
    build_archive(backend)
    print(f"archive seeded with {len(backend.archive)} genomes; "
          f"model={args.model} temp={args.temperature}", flush=True)

    print(f"\n[1/2] collecting {args.n_contexts} rejection contexts", flush=True)
    contexts = collect_contexts(backend, args.n_contexts, args.seed * 7)
    by_class: dict[str, int] = {}
    for c in contexts:
        by_class[c["class"]] = by_class.get(c["class"], 0) + 1
    print(f"  rejection classes: {by_class}", flush=True)

    print(f"\n[2/2] replaying each context under {len(CONDITIONS)} conditions "
          f"(<= {args.max_retries} retries each)", flush=True)
    rows = []
    t0 = time.time()
    for i, ctx in enumerate(contexts):
        # One repair seed per context, SHARED by all conditions and always
        # different from the seed that produced the rejection.
        repair_seed = args.seed * 13 + 1000 * (i + 1)
        rec = {"context": ctx, "results": {}}
        for cond in CONDITIONS:
            rec["results"][cond] = run_condition(
                backend, ctx, cond, repair_seed, args.max_retries)
        rows.append(rec)
        if (i + 1) % 10 == 0:
            el = time.time() - t0
            print(f"  {i+1}/{len(contexts)} contexts  ({el:.0f}s, "
                  f"{el/(i+1):.1f}s/context)", flush=True)

    # ---- aggregate --------------------------------------------------------
    summary = {}
    for cond in CONDITIONS:
        first = sum(r["results"][cond]["first_ok"] for r in rows)
        anyok = sum(r["results"][cond]["any_ok"] for r in rows)
        n = len(rows)
        summary[cond] = {
            "n": n,
            "first_retry_repairs": first,
            "first_retry_rate": first / n if n else 0.0,
            "first_retry_ci95": wilson(first, n),
            "two_retry_repairs": anyok,
            "two_retry_rate": anyok / n if n else 0.0,
            "two_retry_ci95": wilson(anyok, n),
            "valid_proposals_recovered": anyok,
        }
    pairs = {}
    for a, b in (("exact", "generic"), ("exact", "none"), ("generic", "none")):
        for metric in ("first_ok", "any_ok"):
            ba = sum(r["results"][a][metric] and not r["results"][b][metric]
                     for r in rows)
            bb = sum(r["results"][b][metric] and not r["results"][a][metric]
                     for r in rows)
            pairs[f"{a}_vs_{b}::{metric}"] = {
                "a_only": ba, "b_only": bb,
                "mcnemar_p_two_sided": mcnemar_exact(ba, bb),
            }

    out = {
        "config": vars(args),
        "n_contexts": len(rows),
        "rejection_classes": by_class,
        "summary": summary,
        "paired_tests": pairs,
        "rows": rows,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)

    print("\n===== REPAIR-FEEDBACK ABLATION =====")
    print(f"contexts: {len(rows)}   classes: {by_class}")
    print(f"{'condition':10s} {'1st retry':>22s} {'within 2 retries':>24s}")
    for cond in CONDITIONS:
        s = summary[cond]
        lo1, hi1 = s["first_retry_ci95"]
        lo2, hi2 = s["two_retry_ci95"]
        print(f"{cond:10s} {s['first_retry_rate']:6.1%} "
              f"[{lo1:.1%},{hi1:.1%}]{'':>3s} "
              f"{s['two_retry_rate']:6.1%} [{lo2:.1%},{hi2:.1%}]")
    print("\npaired McNemar (two-sided exact):")
    for k, v in pairs.items():
        print(f"  {k:28s} a_only={v['a_only']:3d} b_only={v['b_only']:3d} "
              f"p={v['mcnemar_p_two_sided']:.4g}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
