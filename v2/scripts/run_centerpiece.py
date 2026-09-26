"""Phase 2 CENTERPIECE: one evolution run (one arm, one seed) on the
netflow LP train cell.

Protocol (locked; see scripts/phase2_common.py):
- Seed population = vanilla PDHG textbook steps ONLY (tau = sigma =
  0.9/||A||). Named solvers NEVER appear in the menu.
- Menu genes: tau, sigma, rho, halpern, restart_k, metric(ruiz),
  switch_k. The backend gene, avg mix and deprecated residual-window
  switch are EXCLUDED by protocol; LLM proposals carrying them are
  POLICY REFUSALS, booked separately from convergence-condition
  (Stage-0 gate) rejections.
- 16 generations x 10 proposals, elitism 1.
- TRAIN = 8 netflow instances, degree-2 regular, 300..800 nodes, cell
  seed 7 (identical for every arm/seed).
- Fitness = -SGM-10 of median-of-3 time-to-rel-gap<=1e-4, f64 CPU.
- Per-generation logging: n_llm_calls, refusals-vs-rejections split,
  cumulative CANDIDATE EVALUATIONS (Stage-0 rejects and dedup cache
  hits consume no solver budget and do NOT count), full per-candidate
  provenance and genomes.

Usage:
  env -u LD_LIBRARY_PATH JAX_PLATFORMS=cpu python scripts/run_centerpiece.py \
      --arm llm --seed 42 --machine LOCAL \
      --base-url http://localhost:8000/v1 \
      --out results/sprint/phase2/centerpiece_llm_seed42.json
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from atlas.bench.runner import sanitize_json  # noqa: E402
from atlas.evolve.llm_backend import LLMBackend  # noqa: E402
from atlas.evolve.random_backend import RandomBackend  # noqa: E402
from atlas.genome import (  # noqa: E402
    Genome,
    Metric,
    canonical_json,
    content_hash,
    validate,
)
from phase2_common import (  # noqa: E402
    MECHANISMS,
    OP2,
    T0,
    TRAIN_NODES,
    eval_genome_train,
    genes_active,
    machine_record,
    policy_violation,
    train_specs,
    vanilla_pdhg,
    worker_init,
)

GENERATIONS = 16
POP_SIZE = 10
LP_SCHEMES = ("pdhg", "condat_vu")
_RUIZ_ITERS = (5, 10, 20)

# Offset separating the LLM arm's fallback-sampler RNG stream from the
# random arm's proposer stream at the same replicate seed. Before
# 2026-09-04 both used args.seed + 1, which made the arms POSITIVELY
# CORRELATED by construction: every fallback draw in an LLM run was
# byte-identical to a candidate of the matched random run (336/336 across
# the banked Phase-2 runs). Fixed per Phase-2 verification finding V1;
# regression-locked in tests/test_phase2_fixes.py.
LLM_FALLBACK_SEED_OFFSET = 100003


def fallback_seed(arm: str, seed: int) -> int:
    """Seed of the CPRandomBackend used as proposer (random arm) or as
    LLM-fallback sampler (llm arm). The two streams MUST differ at equal
    replicate seed so the arms are independent replicates."""
    if arm == "llm":
        return seed + LLM_FALLBACK_SEED_OFFSET
    return seed + 1

CP_SCHEMA_DOC = f"""Genome JSON schema (standard-form LP cell, PDHG family):
{{"scheme": "pdhg"|"condat_vu"|"mix",
 "params": {{"tau": number>0 (primal step),
            "sigma": number>0 (dual step),
            "rho": null or number in (0,2) (over-relaxation)}},
 "wrapper": {{"halpern": true|false,
             "restart": "none"|"fixed_k",
             "restart_k": null or integer>0 (needs restart="fixed_k" AND
                          halpern=true; resets the Halpern anchor every k)}},
 "metric": null
   | {{"kind":"ruiz", "iters": integer in [0,30]}} (equilibrate the
     constraint matrix; the certificate stays on the ORIGINAL gap),
 "mix": null
   | {{"kind":"switch_k", "K": integer in [1,5000],
      "aggressive": {{"scheme":..., "params":{{"tau":..., "sigma":...}}}},
      "tail":      {{"scheme":..., "params":{{"tau":..., "sigma":...}}}}}}}}
Validity (Stage-0 gate for this LP cell; ||A||^2 = {OP2:.4f}, f linear so
L_f = 0):
- pdhg:      tau*sigma*{OP2:.4f} < 1
- condat_vu: tau*sigma*{OP2:.4f} < 1 (identical: L_f = 0)
- fbs, drs, dys: structurally inapplicable to this LP; REJECTED
- rho in (0,2); rho and halpern are mutually exclusive
- restart "fixed_k" requires halpern=true (anchor restart)
- metric: base pdhg/condat_vu genomes only (not on a mix genome)
- mix switch_k: the AGGRESSIVE scheme runs the first K iterations (its
  stepsizes MAY break the side condition: big steps), then the TAIL runs
  forever from where the prefix stopped; the tail must be FULLY valid;
  branches pdhg/condat_vu; no outer rho/halpern/restart on a mix genome
EXCLUDED from this experiment (any of these is refused): a "backend"
block, mix kind "avg", mix kind "switch", metric kind "pock_chambolle"."""

CP_MENU = """Mutation menu (pick ONE or TWO small moves):
- scale tau or sigma (e.g. x0.5, x2, x10)
- rebalance tau vs sigma keeping the product fixed (the product is capped
  by tau*sigma*||A||^2 < 1; the RATIO tau/sigma is free and matters)
- set/unset over-relaxation rho (e.g. 1.5, 1.8, 1.95)
- toggle wrapper.halpern (mutually exclusive with rho)
- with halpern on: restart "fixed_k" + restart_k (25..400) resets the
  anchor every k iterations (often much faster than the plain anchor)
- add/tune "metric": {"kind":"ruiz","iters":10} (rescales the problem;
  helps when the instance is badly scaled)
- FINITE-PREFIX SWITCH: {"scheme":"mix","mix":{"kind":"switch_k","K":300,
  "aggressive":{"scheme":"pdhg","params":{"tau":...,"sigma":...}},
  "tail":{"scheme":"pdhg","params":{"tau":...,"sigma":...}}}}; the
  aggressive branch may break the stepsize condition (big steps!), the
  tail must be valid

COMMON MISTAKES (each is REJECTED):
- a "mix" block without "scheme":"mix"
- "rho" set together with "halpern":true (choose ONE)
- "restart":"fixed_k" with "halpern":false (restart needs the anchor)
- "metric" on a mix genome (base genomes only)
- tau*sigma too big: tau*sigma must stay below 1/8.33 = 0.12
Pick ONE mechanism per proposal; do not stack them all."""

CP_SYSTEM = (
    "You tune parameters of certified first-order solvers (PDHG family) "
    "for standard-form linear programs min c'x s.t. Ax=b, x>=0. You reply "
    "with EXACTLY ONE JSON object and no other text. Write plain JSON "
    "numbers only (no arithmetic expressions like 0.3*2). Keep the JSON "
    "COMPACT on a single line."
)


# ---------------------------------------------------------------------------
# Proposal backends (centerpiece menu)
# ---------------------------------------------------------------------------


class CPRandomBackend(RandomBackend):
    """Random arm / LLM fallback over the centerpiece menu: LP stepsize
    ranges, no avg mix (p_mix=0), no backend gene, ruiz-only metric."""

    def __init__(self, seed: int):
        super().__init__(
            seed=seed,
            schemes=LP_SCHEMES,
            problem_kind="lp_netflow",
            p_alpha=0.0,
            p_mix=0.0,  # avg excluded by protocol
            p_switch=0.10,
            p_backend=0.0,
            p_metric=0.15,
        )

    def _draw_steps(self, scheme: str, aggressive: bool = False):
        scale = 20.0 if aggressive else 1.0
        tau = self._log_uniform(1e-2, 10.0)
        s_max = scale * 0.98 / (OP2 * tau)
        sigma = self._log_uniform(s_max * 1e-4, s_max)
        return tau, sigma

    def _draw_metric(self):
        if self.rng.uniform() >= self.p_metric:
            return None
        return Metric(kind="ruiz", iters=int(self.rng.choice(_RUIZ_ITERS)))


class CPLLMBackend(LLMBackend):
    """LLM mutation backend with the centerpiece prompt card and the
    POLICY-vs-GATE-vs-FORMAT rejection split (Phase-1 F7/F9)."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.n_policy_refusals = 0
        self.n_gate_rejections = 0
        self.n_format_errors = 0
        self.attempt_log: list[dict] = []  # per-proposal attempt errors

    def _chat(self, user_msg: str) -> str:
        self.n_llm_calls += 1
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": CP_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "seed": self.seed + self.n_llm_calls,
        }
        reply = self._http_post(payload)
        self.last_transcript.append({"prompt": user_msg, "reply": reply})
        return reply

    def _build_prompt(self, feedback):
        parts = [
            CP_SCHEMA_DOC,
            self._best_block(),
            CP_MENU,
            "Propose ONE mutated genome likely to improve fitness. Reply "
            "with ONLY the JSON genome object (or a partial JSON patch of "
            "the best genome).",
        ]
        if feedback:
            parts.append(
                f"Your previous reply was INVALID: {feedback}\n"
                "Fix it and reply with ONLY one corrected JSON genome object."
            )
        return "\n\n".join(parts)

    def _parse_classified(self, reply: str):
        """-> (genome|None, error|None, class in {format, policy, gate})."""
        from atlas.genome import GenomeFormatError, genome_from_dict, genome_to_dict

        block = self._extract_json_block(reply)
        if block is None:
            return None, "no JSON object found in the reply", "format"
        try:
            d = json.loads(block)
        except json.JSONDecodeError as e:
            return None, f"malformed JSON: {e}", "format"
        if not isinstance(d, dict):
            return None, "reply JSON must be an object", "format"
        if "scheme" not in d:  # JSON patch onto the current best
            if not self.archive:
                return (
                    None,
                    "a partial patch needs a base genome; send a full genome",
                    "format",
                )
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
            return None, str(e), "format"
        pv = policy_violation(g)
        if pv is not None:
            return None, pv, "policy"
        v = validate(g, problem_kind=self.problem_kind)
        if not v.ok:
            return None, v.error, "gate"
        return g, None, None

    def _propose_one(self):
        # THREE attempts before the random fallback, so the run stays
        # LLM-guided. As of 2026-09-04 the retries are CLEAN RESAMPLES by
        # default, not diagnostic-injected repairs: the paired ablation
        # (scripts/repair_ablation.py, n=200) found feedback injection
        # strictly worse than resampling (52.0% -> 36.0% two-retry yield),
        # with the effect strongest on genuine gate rejections. Pass
        # repair_feedback=True to restore the old behavior.
        feedback = None
        attempts = []
        for _attempt in range(3):
            try:
                reply = self._chat(self._build_prompt(feedback))
            except Exception as e:
                attempts.append({"class": "endpoint", "error": str(e)})
                break
            g, err, klass = self._parse_classified(reply)
            if g is not None:
                self.attempt_log.append({"attempts": attempts, "accepted": "llm"})
                return g
            if klass == "format":
                self.n_format_errors += 1
            elif klass == "policy":
                self.n_policy_refusals += 1
            else:
                self.n_gate_rejections += 1
                self.stage0_rejections += 1
            attempts.append({"class": klass, "error": err})
            feedback = err if self.repair_feedback else None
        self.n_fallbacks += 1
        self.attempt_log.append({"attempts": attempts, "accepted": "fallback"})
        return self.fallback.ask(1)[0]


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


def run(args) -> dict:
    t_start = time.time()
    specs = train_specs()
    n_inst = len(specs)
    pool = None
    if args.workers > 1:
        pool = mp.get_context("spawn").Pool(
            processes=args.workers,
            initializer=worker_init,
            initargs=(specs, None),
        )
    else:
        worker_init(specs)

    seed_genome = vanilla_pdhg()
    assert validate(seed_genome, problem_kind="lp_netflow").ok

    fallback = CPRandomBackend(seed=fallback_seed(args.arm, args.seed))
    if args.arm == "llm":
        backend = CPLLMBackend(
            base_url=args.base_url,
            seed=args.seed,
            fallback=fallback,
            problem_kind="lp_netflow",
            max_tokens=700,
        )
    else:
        backend = fallback

    tag = f"[{args.machine} {args.arm} seed {args.seed}]"
    print(f"{tag} evaluating vanilla-PDHG seed ...", flush=True)
    fit0, m0 = eval_genome_train(seed_genome, n_inst, pool=pool)
    backend.tell(seed_genome, fit0, m0)
    cache = {content_hash(seed_genome): (fit0, m0)}
    champion, champ_fit, champ_m = seed_genome, fit0, m0
    print(
        f"{tag} seed fitness {fit0:.6g} "
        f"(sgm iters {m0['sgm10_iters']:.1f}, fails {m0['n_failed_instances']})",
        flush=True,
    )

    first_pop: dict = {k: None for k in MECHANISMS}
    first_champ: dict = {k: None for k in MECHANISMS}
    generations = []
    trajectory = [
        {"gen": -1, "cumulative_evaluations": 1, "champion_fitness": fit0}
    ]
    truncated = False
    wall_guard_s = args.wall_guard_min * 60.0

    is_llm = args.arm == "llm"

    for gen in range(args.generations):
        gen_t0 = time.time()
        if time.time() - t_start > wall_guard_s:
            truncated = True
            break
        c0 = {
            "llm_calls": backend.n_llm_calls if is_llm else 0,
            "fallbacks": backend.n_fallbacks if is_llm else 0,
            "policy": backend.n_policy_refusals if is_llm else 0,
            "gate": backend.n_gate_rejections if is_llm else 0,
            "format": backend.n_format_errors if is_llm else 0,
            "rand_gate": (
                backend.fallback.stage0_rejections
                if is_llm
                else backend.stage0_rejections
            ),
        }
        entries = []
        for _ in range(args.pop_size):
            if is_llm:
                fb_before = backend.n_fallbacks
                g = backend._propose_one()
                source = (
                    "fallback" if backend.n_fallbacks > fb_before else "llm"
                )
                prov = backend.attempt_log[-1]["attempts"] if backend.attempt_log else []
            else:
                g = backend.ask(1)[0]
                source = "random"
                prov = []
            # Belt-and-braces: gate + policy re-check before spending budget.
            v = validate(g, problem_kind="lp_netflow")
            pv = policy_violation(g)
            if not v.ok or pv is not None:
                entries.append(
                    {
                        "genome": canonical_json(g),
                        "source": source,
                        "rejected": True,
                        "rejection_class": "policy" if pv else "gate",
                        "error": pv or v.error,
                        "llm_attempt_errors": prov,
                    }
                )
                continue
            h = content_hash(g)
            if h in cache:
                fit, m = cache[h]
                dedup = True
            else:
                fit, m = eval_genome_train(g, n_inst, pool=pool)
                cache[h] = (fit, m)
                dedup = False
            backend.tell(g, fit, m)
            act = genes_active(g)
            for k in MECHANISMS:
                if act[k] and first_pop[k] is None:
                    first_pop[k] = gen
            entries.append(
                {
                    "genome": canonical_json(g),
                    "source": source,
                    "rejected": False,
                    "dedup": dedup,
                    "fitness": fit,
                    "genes_active": act,
                    "llm_attempt_errors": prov,
                    "metrics": (
                        {
                            k: m[k]
                            for k in (
                                "sgm10_time_s",
                                "sgm10_iters",
                                "n_failed_instances",
                            )
                            if k in m
                        }
                        if "error" not in m
                        else {"error": m["error"]}
                    ),
                }
            )
            if fit > champ_fit:
                champion, champ_fit, champ_m = g, fit, m
        backend.tell(champion, champ_fit, champ_m)  # elitism keep-1
        c_act = genes_active(champion)
        for k in MECHANISMS:
            if c_act[k] and first_champ[k] is None:
                first_champ[k] = gen
        cum_evals = len(cache)
        gen_rec = {
            "gen": gen,
            "champion": canonical_json(champion),
            "champion_fitness": champ_fit,
            "champion_sgm10_iters": champ_m.get("sgm10_iters"),
            "champion_genes_active": c_act,
            "candidates": entries,
            "cumulative_evaluations": cum_evals,
            "n_llm_calls": (backend.n_llm_calls - c0["llm_calls"]) if is_llm else 0,
            "n_llm_fallbacks": (backend.n_fallbacks - c0["fallbacks"]) if is_llm else 0,
            "stage0_rejections": {
                "policy_refusals": (
                    (backend.n_policy_refusals - c0["policy"]) if is_llm else 0
                ),
                "convergence_condition_rejections": (
                    (backend.n_gate_rejections - c0["gate"])
                    if is_llm
                    else (backend.stage0_rejections - c0["rand_gate"])
                ),
                "format_errors": (
                    (backend.n_format_errors - c0["format"]) if is_llm else 0
                ),
                "fallback_sampler_gate_rejections": (
                    (backend.fallback.stage0_rejections - c0["rand_gate"])
                    if is_llm
                    else None
                ),
            },
            "wall_s": time.time() - gen_t0,
        }
        generations.append(gen_rec)
        trajectory.append(
            {
                "gen": gen,
                "cumulative_evaluations": cum_evals,
                "champion_fitness": champ_fit,
            }
        )
        print(
            f"{tag} gen {gen}: champ fit {champ_fit:.6g} "
            f"evals {cum_evals} genes {[k for k in MECHANISMS if c_act[k]]} "
            f"({time.time() - gen_t0:.0f}s)",
            flush=True,
        )

    champ_cert = validate(champion, problem_kind="lp_netflow").cert
    out = {
        "experiment": "phase2_centerpiece",
        "arm": args.arm,
        "replicate_seed": args.seed,
        **machine_record(args.machine),
        "config": {
            "cell_seed": 7,
            "train_nodes": list(TRAIN_NODES),
            "degree": 2,
            "tol_rel_gap": 1e-4,
            "max_iters": 200_000,
            "sample_every": 100,
            "n_repeats_median": 3,
            "generations": args.generations,
            "pop_size": args.pop_size,
            "elitism": 1,
            "workers": args.workers,
            "seed_genome": canonical_json(seed_genome),
            "seed_genome_fitness": fit0,
            "fallback_sampler_seed": fallback_seed(args.arm, args.seed),
            "fallback_seed_decoupled_from_random_arm": args.arm == "llm",
            "gate_opnorm_sq": OP2,
            "textbook_step": T0,
            "llm_model": backend.model if is_llm else None,
            "llm_base_url": args.base_url if is_llm else None,
            "menu_genes": [
                "tau",
                "sigma",
                "rho",
                "halpern",
                "restart_k",
                "metric_ruiz",
                "switch_k",
            ],
            "genes_excluded_by_protocol": [
                "backend",
                "mix_avg",
                "mix_switch_deprecated",
                "metric_pock_chambolle",
            ],
        },
        "generations": generations,
        "truncated_by_wall_guard": truncated,
        "first_active_in_population": first_pop,
        "first_active_in_champion": first_champ,
        "fitness_trajectory_vs_evaluations": trajectory,
        "cumulative_evaluations_total": len(cache),
        "final_champion": canonical_json(champion),
        "final_champion_fitness": champ_fit,
        "final_champion_metrics": champ_m,
        "final_champion_genes_active": genes_active(champion),
        "final_champion_cert": champ_cert.to_dict() if champ_cert else None,
        "counters_total": {
            "n_llm_calls": backend.n_llm_calls if is_llm else 0,
            "n_llm_fallbacks": backend.n_fallbacks if is_llm else 0,
            "policy_refusals": backend.n_policy_refusals if is_llm else 0,
            "convergence_condition_rejections": (
                backend.n_gate_rejections
                if is_llm
                else backend.stage0_rejections
            ),
            "format_errors": backend.n_format_errors if is_llm else 0,
            "fallback_sampler_gate_rejections": (
                backend.fallback.stage0_rejections if is_llm else None
            ),
        },
        "total_wall_s": time.time() - t_start,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if pool is not None:
        pool.close()
        pool.join()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(sanitize_json(out), fh, indent=2, allow_nan=False)
        fh.write("\n")
    print(f"{tag} DONE in {out['total_wall_s']:.0f}s -> {args.out}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=("llm", "random"), required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--machine", choices=("LOCAL", "REMOTE"), default="LOCAL")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--generations", type=int, default=GENERATIONS)
    ap.add_argument("--pop-size", type=int, default=POP_SIZE)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--wall-guard-min", type=float, default=300.0)
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
