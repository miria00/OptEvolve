"""MINI-E-R: rediscovery experiment on the netflow LP cell.

Question: seeded with ONLY vanilla PDHG (textbook stepsizes, no wrapper
genes), does LLM-guided evolution over the typed genome space rediscover
the mechanisms of the human GPU-LP lineage (over-relaxation, restart,
Halpern anchoring, safeguarded switching / averaged mixing), and in what
order?

Cell: standard-form LP min c'x s.t. Ax=b, x>=0 on degree-2 netflow
instances (atlas.bench.netflow), 6 train (40x80 .. 100x200 nodes x arcs)
+ 3 held-out (46x92, 70x140, 94x188), cell seed 7. Fitness = -SGM-10 of
median-of-3 time-to-rel-gap<=1e-4 on train (PAR-1 wall-time cap on
failures). Stage-0 gate: problem_kind="lp_netflow". Genes open to
mutation: stepsizes (tau, sigma), over-relaxation rho, Halpern anchor,
fixed-k anchor restart, avg mix, safeguarded switch. The backend
(precision) gene is EXCLUDED from this experiment by protocol.

Budget: 12 generations x 8 proposals, hard wall guard 42 min.

Usage:
  JAX_PLATFORMS=cpu python scripts/run_mini_er.py --seed 42 \
      --base-url http://localhost:8000/v1 --machine LOCAL \
      --out results/sprint/mini_er_seed42.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from atlas.bench.lp_cell import from_netflow, to_composite  # noqa: E402
from atlas.bench.metrics import DEFAULT_SGM_SHIFT, sgm, time_to_tol  # noqa: E402
from atlas.bench.netflow import gen_regular_flow  # noqa: E402
from atlas.bench.runner import sanitize_json  # noqa: E402
from atlas.evolve.llm_backend import LLMBackend  # noqa: E402
from atlas.evolve.random_backend import RandomBackend  # noqa: E402
from atlas.genome import (  # noqa: E402
    OPNORM_LP,
    Genome,
    GenomeParams,
    canonical_dict,
    canonical_json,
    content_hash,
    validate,
)
from atlas.schemes.compile import solve_genome  # noqa: E402
from atlas.typing_ import TypeCheckError  # noqa: E402
from atlas.wrappers.km import Budget  # noqa: E402

# ---------------------------------------------------------------------------
# Protocol constants
# ---------------------------------------------------------------------------

CELL_SEED = 7
TRAIN_NODES = (40, 52, 64, 76, 88, 100)
HOLDOUT_NODES = (46, 70, 94)
DEGREE = 2
TOL = 1e-4
MAX_ITERS = 200_000
SAMPLE_EVERY = 100
N_REPEATS = 3
GENERATIONS = 12
POP_SIZE = 8
WALL_GUARD_S = 42 * 60
LP_SCHEMES = ("pdhg", "condat_vu")
T0 = 0.9 / OPNORM_LP  # textbook tau = sigma = 0.9 / ||A||

OP2 = OPNORM_LP**2  # 8.3232

LP_SCHEMA_DOC = f"""Genome JSON schema (standard-form LP cell, PDHG family):
{{"scheme": "pdhg"|"condat_vu"|"mix",
 "params": {{"tau": number>0 (primal step),
            "sigma": number>0 (dual step),
            "rho": null or number in (0,2) (over-relaxation)}},
 "wrapper": {{"halpern": true|false,
             "restart": "none"|"fixed_k",
             "restart_k": null or integer>0 (needs restart="fixed_k" AND
                          halpern=true; resets the Halpern anchor every k)}},
 "mix": null
   | {{"kind":"avg", "lambda": number in (0,1),
      "scheme_a": {{"scheme":..., "params":{{"tau":..., "sigma":...}}}},
      "scheme_b": {{"scheme":..., "params":{{"tau":..., "sigma":...}}}}}}
   | {{"kind":"switch", "window": integer>=1,
      "aggressive": {{"scheme":..., "params":{{"tau":..., "sigma":...}}}},
      "fallback":  {{"scheme":..., "params":{{"tau":..., "sigma":...}}}}}}}}
Validity (Stage-0 gate for this LP cell; ||A||^2 = {OP2:.4f}, f linear so
L_f = 0):
- pdhg:      tau*sigma*{OP2:.4f} < 1
- condat_vu: tau*sigma*{OP2:.4f} < 1 (identical: L_f = 0)
- fbs, drs, dys: structurally inapplicable to this LP; REJECTED
- rho and halpern are mutually exclusive; rho in (0,2) with rho*0.5 < 1
- restart "fixed_k" requires halpern=true (anchor restart)
- mix avg: lambda*T_a + (1-lambda)*T_b; BOTH parents valid; same scheme,
  or pdhg+condat_vu with IDENTICAL (tau, sigma)
- mix switch: fallback must be valid; aggressive params MAY violate the
  stepsize condition (big steps); a candidate aggressive step is accepted
  only when the certified gap beats the best certified gap of the last
  `window` iterations, else the certified fallback step runs; no outer
  rho/halpern/restart on a switch genome
- do NOT emit a "backend" block in this experiment."""

LP_MENU = """Mutation menu (pick ONE or TWO small moves):
- scale tau or sigma (e.g. x0.5, x2, x10)
- rebalance tau vs sigma keeping the product fixed (the product is capped
  by tau*sigma*||A||^2 < 1; the RATIO tau/sigma is free and matters)
- set/unset over-relaxation rho (e.g. 1.5, 1.8, 1.95)
- toggle wrapper.halpern (mutually exclusive with rho)
- with halpern on: restart "fixed_k" + restart_k (25..400) resets the
  anchor every k iterations (often much faster than the plain anchor)
- MIX two schemes: {"scheme":"mix","mix":{"kind":"avg","lambda":0.5,
  "scheme_a":{...},"scheme_b":{...}}}
- SAFEGUARDED SWITCH: {"scheme":"mix","mix":{"kind":"switch","window":25,
  "aggressive":{...},"fallback":{...}}}; the aggressive branch may break
  the stepsize condition (big steps!), the fallback must be valid

COMMON MISTAKES (each is REJECTED):
- a "mix" block without "scheme":"mix"
- "rho" set together with "halpern":true (choose ONE)
- "restart":"fixed_k" with "halpern":false (restart needs the anchor)
- avg parents pdhg/condat_vu with different tau or sigma (must be equal)
- tau*sigma too big: tau*sigma must stay below 1/8.33 = 0.12
Pick ONE mechanism per proposal; do not stack them all."""

LP_SYSTEM = (
    "You tune parameters of certified first-order solvers (PDHG family) "
    "for standard-form linear programs min c'x s.t. Ax=b, x>=0. You reply "
    "with EXACTLY ONE JSON object and no other text. Write plain JSON "
    "numbers only (no arithmetic expressions like 0.3*2). Keep the JSON "
    "COMPACT on a single line."
)


# ---------------------------------------------------------------------------
# LP-specific proposal backends
# ---------------------------------------------------------------------------


class LPRandomBackend(RandomBackend):
    """Random control/fallback over the LP-valid region (pdhg/condat_vu)."""

    def __init__(self, seed: int):
        super().__init__(
            seed=seed,
            schemes=LP_SCHEMES,
            problem_kind="lp_netflow",
            p_alpha=0.0,
            p_mix=0.10,
            p_switch=0.08,
            p_backend=0.0,
        )

    def _draw_steps(self, scheme: str, aggressive: bool = False):
        scale = 20.0 if aggressive else 1.0
        tau = self._log_uniform(1e-2, 10.0)
        s_max = scale * 0.98 / (OP2 * tau)
        sigma = self._log_uniform(s_max * 1e-4, s_max)
        return tau, sigma


class LPLLMBackend(LLMBackend):
    """LLM mutation backend with the LP prompt card; backend gene refused."""

    def _chat(self, user_msg: str) -> str:
        self.n_llm_calls += 1
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": LP_SYSTEM},
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
            LP_SCHEMA_DOC,
            self._best_block(),
            LP_MENU,
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

    def _parse_reply(self, reply):
        g, err = super()._parse_reply(reply)
        if g is not None and g.backend is not None:
            self.stage0_rejections += 1
            return None, (
                "the 'backend' block is excluded from this experiment; "
                "remove it"
            )
        return g, err

    def _propose_one(self):
        # Weak-model accommodation: TWO repair retries (base class: one)
        # before the random fallback, so the run stays LLM-guided.
        feedback = None
        for _attempt in range(3):
            try:
                reply = self._chat(self._build_prompt(feedback))
            except Exception as e:
                feedback = f"endpoint error: {e}"
                break
            g, err = self._parse_reply(reply)
            if g is not None:
                return g
            feedback = err
        self.n_fallbacks += 1
        return self.fallback.ask(1)[0]


# ---------------------------------------------------------------------------
# Cell + fitness
# ---------------------------------------------------------------------------


def build_cell():
    probs, shapes = [], []
    for i, nn in enumerate(TRAIN_NODES + HOLDOUT_NODES):
        nf = gen_regular_flow(
            n_nodes=nn, seed=CELL_SEED * 10007 + i, degree=DEGREE
        )
        probs.append(to_composite(from_netflow(nf)))
        shapes.append((nn, nf.n_arcs))
    n_tr = len(TRAIN_NODES)
    return probs[:n_tr], probs[n_tr:], shapes[:n_tr], shapes[n_tr:]


def eval_genome(g: Genome, instances) -> tuple[float, dict]:
    """(fitness, metrics): -SGM-10 of per-instance median-of-N_REPEATS
    time-to-tol; failures capped at their budgeted wall time (PAR-1).
    Instance-level TypeCheckError = certified construction failure = -inf."""
    budget = Budget(max_iters=MAX_ITERS, tol=TOL, sample_every=SAMPLE_EVERY)
    times, iters_at, fails, gaps = [], [], 0, []
    for prob in instances:
        reps_t, reps_i = [], []
        failed = False
        for _rep in range(N_REPEATS):
            try:
                r = solve_genome(prob, g, budget)
            except TypeCheckError as e:
                return float("-inf"), {"error": str(e)}
            gap = float(r.rel_gap_history[-1])
            if gap <= TOL:
                t, it = time_to_tol(r, TOL, sample_every=SAMPLE_EVERY)
                reps_t.append(float(t))
                reps_i.append(int(it))
            else:
                failed = True
                reps_t.append(float(r.wall_time))  # PAR-1 cap
                reps_i.append(int(r.iters))
            gaps.append(gap)
        if failed:
            fails += 1
        times.append(float(np.median(reps_t)))
        iters_at.append(int(np.median(reps_i)))
    fit = -sgm(times, DEFAULT_SGM_SHIFT)
    return fit, {
        "sgm10_time_s": -fit,
        "sgm10_iters": sgm(iters_at, DEFAULT_SGM_SHIFT),
        "n_failed_instances": fails,
        "per_instance_time_s": times,
        "per_instance_iters": iters_at,
    }


def genes_active(g: Genome) -> dict:
    d = canonical_dict(g)
    rho = d["params"]["rho"]
    return {
        "over_relaxation": rho is not None and rho != 1.0,
        "halpern": bool(d["wrapper"]["halpern"]),
        "restart": d["wrapper"]["restart"] == "fixed_k",
        "mix_avg": d["mix"] is not None and d["mix"]["kind"] == "avg",
        "mix_switch": d["mix"] is not None and d["mix"]["kind"] == "switch",
        "scheme_changed": d["scheme"] != "pdhg",
    }


MECHANISMS = (
    "over_relaxation",
    "halpern",
    "restart",
    "mix_avg",
    "mix_switch",
    "scheme_changed",
)


# ---------------------------------------------------------------------------
# The seeded evolution loop (mirrors atlas.evolve.loop semantics + seeding)
# ---------------------------------------------------------------------------


def run_replicate(seed: int, base_url: str, machine: str, out_path: str):
    t_start = time.time()
    train, holdout, tr_shapes, ho_shapes = build_cell()

    seed_genome = Genome(
        scheme="pdhg", params=GenomeParams(tau=T0, sigma=T0)
    )
    assert validate(seed_genome, problem_kind="lp_netflow").ok

    fallback = LPRandomBackend(seed=seed + 1)
    lb = LPLLMBackend(
        base_url=base_url,
        seed=seed,
        fallback=fallback,
        problem_kind="lp_netflow",
        max_tokens=800,
    )

    print(f"[{machine} seed {seed}] evaluating vanilla-PDHG seed ...")
    fit0, m0 = eval_genome(seed_genome, train)
    lb.tell(seed_genome, fit0, m0)
    cache = {content_hash(seed_genome): (fit0, m0)}
    champion, champ_fit, champ_m = seed_genome, fit0, m0
    print(f"  seed fitness {fit0:.6g} (sgm iters {m0['sgm10_iters']:.1f})")

    first_pop: dict = {k: None for k in MECHANISMS}
    first_champ: dict = {k: None for k in MECHANISMS}
    generations = []
    truncated = False

    for gen in range(GENERATIONS):
        gen_t0 = time.time()
        if time.time() - t_start > WALL_GUARD_S:
            truncated = True
            break
        rej0, fb0, llm0 = lb.stage0_rejections, lb.n_fallbacks, lb.n_llm_calls
        entries = []
        for _ in range(POP_SIZE):
            fb_before = lb.n_fallbacks
            g = lb._propose_one()
            source = "fallback" if lb.n_fallbacks > fb_before else "llm"
            v = validate(g, problem_kind="lp_netflow")
            if not v.ok or g.backend is not None:
                entries.append(
                    {
                        "genome": canonical_json(g),
                        "source": source,
                        "rejected": True,
                        "error": v.error or "backend block excluded",
                    }
                )
                continue
            h = content_hash(g)
            if h in cache:
                fit, m = cache[h]
                dedup = True
            else:
                fit, m = eval_genome(g, train)
                cache[h] = (fit, m)
                dedup = False
            lb.tell(g, fit, m)
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
                    "metrics": {
                        k: m[k]
                        for k in (
                            "sgm10_time_s",
                            "sgm10_iters",
                            "n_failed_instances",
                        )
                        if k in m
                    }
                    if "error" not in m
                    else {"error": m["error"]},
                }
            )
            if fit > champ_fit:
                champion, champ_fit, champ_m = g, fit, m
        lb.tell(champion, champ_fit, champ_m)  # elitism keep-1
        c_act = genes_active(champion)
        for k in MECHANISMS:
            if c_act[k] and first_champ[k] is None:
                first_champ[k] = gen
        generations.append(
            {
                "gen": gen,
                "champion": canonical_json(champion),
                "champion_fitness": champ_fit,
                "champion_sgm10_iters": champ_m.get("sgm10_iters"),
                "champion_genes_active": c_act,
                "candidates": entries,
                "n_rejected_stage0": lb.stage0_rejections - rej0,
                "n_llm_fallbacks": lb.n_fallbacks - fb0,
                "n_llm_calls": lb.n_llm_calls - llm0,
                "wall_s": time.time() - gen_t0,
            }
        )
        print(
            f"  gen {gen}: champ fit {champ_fit:.6g} "
            f"genes {[k for k in MECHANISMS if c_act[k]]} "
            f"({time.time() - gen_t0:.0f}s)"
        )

    # ---- held-out comparison ---------------------------------------------
    textbook = {
        "pdhg_t0.9": seed_genome,
        "pdhg_t0.99": Genome(
            scheme="pdhg",
            params=GenomeParams(tau=0.99 / OPNORM_LP, sigma=0.99 / OPNORM_LP),
        ),
        "condat_vu_t0.9": Genome(
            scheme="condat_vu", params=GenomeParams(tau=T0, sigma=T0)
        ),
    }
    tb_train = {
        name: eval_genome(g, train) for name, g in textbook.items()
    }
    best_tb_name = max(tb_train, key=lambda n: tb_train[n][0])
    holdout_rows = {}
    for name, g in [
        ("champion", champion),
        ("vanilla_pdhg_seed", seed_genome),
        (f"best_textbook_default[{best_tb_name}]", textbook[best_tb_name]),
    ]:
        fit_ho, m_ho = eval_genome(g, holdout)
        holdout_rows[name] = {
            "genome": canonical_json(g),
            "holdout_sgm10_time_s": None
            if not math.isfinite(fit_ho)
            else -fit_ho,
            "metrics": m_ho,
        }

    out = {
        "experiment": "mini_er_lp_netflow",
        "replicate_seed": seed,
        "machine": machine,
        "hostname": platform.node(),
        "jax_platform": "cpu",
        "config": {
            "cell_seed": CELL_SEED,
            "train_nodes_x_arcs": tr_shapes,
            "holdout_nodes_x_arcs": ho_shapes,
            "degree": DEGREE,
            "tol_rel_gap": TOL,
            "max_iters": MAX_ITERS,
            "sample_every": SAMPLE_EVERY,
            "n_repeats_median": N_REPEATS,
            "generations": GENERATIONS,
            "pop_size": POP_SIZE,
            "seed_genome": canonical_json(seed_genome),
            "seed_genome_fitness": fit0,
            "gate_opnorm_sq": OP2,
            "llm_model": lb.model,
            "llm_base_url": base_url,
            "genes_excluded": ["backend"],
        },
        "generations": generations,
        "truncated_by_wall_guard": truncated,
        "first_active_in_population": first_pop,
        "first_active_in_champion": first_champ,
        "final_champion": canonical_json(champion),
        "final_champion_fitness": champ_fit,
        "n_unique_evaluated": len(cache),
        "textbook_defaults_train": {
            n: {"fitness": f, "metrics": m} for n, (f, m) in tb_train.items()
        },
        "best_textbook_default": best_tb_name,
        "holdout_comparison": holdout_rows,
        "total_wall_s": time.time() - t_start,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(sanitize_json(out), fh, indent=2, allow_nan=False)
        fh.write("\n")
    print(f"[{machine} seed {seed}] DONE in {out['total_wall_s']:.0f}s -> {out_path}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--machine", default="LOCAL")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    run_replicate(args.seed, args.base_url, args.machine, args.out)


if __name__ == "__main__":
    main()
