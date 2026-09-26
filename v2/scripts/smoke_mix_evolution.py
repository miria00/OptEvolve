"""Sprint smoke test: mix genes end-to-end through BOTH proposal backends.

Runs a 3-generation evolution on TV 64x64 (real OCT patches) where the
scheme menu includes dys (structurally inapplicable on TV: exercises the
Stage-0 rejection path), avg blends and safeguarded switch genomes, once
with the RandomBackend and once with the LLMBackend against the live vLLM
endpoint. Confirms that at least one mix genome is proposed, passes the
gate, and is evaluated. Writes results/sprint/smoke_mix_evolution.json.

Usage: JAX_PLATFORMS=cpu python scripts/smoke_mix_evolution.py
"""
from __future__ import annotations

import json
import os
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from atlas.bench.metrics import DEFAULT_SGM_SHIFT, sgm, time_to_tol  # noqa: E402
from atlas.bench.tv_cell import build_instances  # noqa: E402
from atlas.evolve.llm_backend import LLMBackend  # noqa: E402
from atlas.evolve.loop import evolve  # noqa: E402
from atlas.evolve.random_backend import RandomBackend  # noqa: E402
from atlas.genome import Genome, canonical_json  # noqa: E402
from atlas.schemes.compile import genome_cap_scheme, solve_genome  # noqa: E402
from atlas.typing_ import TypeCheckError  # noqa: E402
from atlas.wrappers.km import Budget  # noqa: E402

MENU = ("fbs", "drs", "pdhg", "condat_vu", "dys")  # dys included: TV rejects it
GENERATIONS = 3
POP_SIZE = 8
TOL = 1e-4
SIZE = 64
N_TRAIN = 2
SEED = 42
CAPS = {"fbs": 40_000, "drs": 4_000, "pdhg": 40_000, "condat_vu": 40_000,
        "dys": 40_000, "mix": 40_000}


def make_fitness(instances, log):
    def fitness_fn(g: Genome):
        cap = CAPS.get(genome_cap_scheme(g), 40_000)
        budget = Budget(max_iters=cap, tol=TOL, sample_every=100)
        times, fails = [], 0
        for prob in instances:
            try:
                r = solve_genome(prob, g, budget)
            except TypeCheckError as e:
                log.append({"genome": canonical_json(g), "error": str(e)})
                return float("-inf"), {"error": str(e)}
            gap = float(r.rel_gap_history[-1])
            if gap <= TOL:
                t, _ = time_to_tol(r, TOL, sample_every=100)
                times.append(float(t))
            else:
                fails += 1
                times.append(float(r.wall_time))
        fit = -sgm(times, DEFAULT_SGM_SHIFT)
        is_mix = g.scheme == "mix"
        log.append({"genome": canonical_json(g), "fitness": fit,
                    "is_mix": is_mix, "failures": fails})
        return fit, {"sgm_time": -fit, "failures": fails, "is_mix": is_mix}

    return fitness_fn


def mix_stats(evaluated):
    n_mix = sum(1 for e in evaluated if e.get("is_mix"))
    kinds = {}
    for e in evaluated:
        if e.get("is_mix"):
            g = json.loads(e["genome"])
            kinds[g["mix"]["kind"]] = kinds.get(g["mix"]["kind"], 0) + 1
    return n_mix, kinds


def main():
    t0 = time.time()
    instances = build_instances(N_TRAIN, size=SIZE, seed=SEED)
    out = {"config": {"generations": GENERATIONS, "pop_size": POP_SIZE,
                      "tol": TOL, "size": SIZE, "n_train": N_TRAIN,
                      "menu": list(MENU), "seed": SEED}}

    # ---- Random proposal backend ------------------------------------------
    log_r: list = []
    rb = RandomBackend(seed=SEED, schemes=MENU, p_mix=0.25, p_switch=0.15,
                       p_backend=0.15)
    hist_r = evolve(rb, make_fitness(instances, log_r), GENERATIONS, POP_SIZE,
                    seed=SEED)
    n_mix_r, kinds_r = mix_stats(log_r)
    out["random"] = {
        "history": hist_r,
        "stage0_rejections": rb.stage0_rejections,
        "n_mix_evaluated": n_mix_r,
        "mix_kinds_evaluated": kinds_r,
        "n_evaluated": len([e for e in log_r if "fitness" in e]),
    }
    print(f"[random] best={hist_r['best_fitness']:.4g} "
          f"mix evaluated={n_mix_r} kinds={kinds_r} "
          f"stage0_rejections={rb.stage0_rejections}")

    # ---- LLM proposal backend (live vLLM) ---------------------------------
    log_l: list = []
    fallback = RandomBackend(seed=SEED + 1, schemes=MENU, p_mix=0.25,
                             p_switch=0.15, p_backend=0.15)
    try:
        from atlas.backend.device import detect_device

        hardware = detect_device(prefer_gpu=False)  # the CPU the solves run on
    except Exception:
        hardware = None
    lb = LLMBackend(seed=SEED, fallback=fallback, hardware=hardware)
    hist_l = evolve(lb, make_fitness(instances, log_l), GENERATIONS, POP_SIZE,
                    seed=SEED)
    n_mix_l, kinds_l = mix_stats(log_l)
    out["llm"] = {
        "history": hist_l,
        "stage0_rejections": lb.stage0_rejections,
        "n_llm_calls": lb.n_llm_calls,
        "n_fallbacks": lb.n_fallbacks,
        "n_mix_evaluated": n_mix_l,
        "mix_kinds_evaluated": kinds_l,
        "n_evaluated": len([e for e in log_l if "fitness" in e]),
    }
    print(f"[llm]    best={hist_l['best_fitness']:.4g} "
          f"mix evaluated={n_mix_l} kinds={kinds_l} "
          f"llm_calls={lb.n_llm_calls} fallbacks={lb.n_fallbacks} "
          f"stage0_rejections={lb.stage0_rejections}")

    out["wall_time_s"] = time.time() - t0
    out["ok"] = bool(n_mix_r >= 1 and n_mix_l >= 1)
    dest = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "results", "sprint", "smoke_mix_evolution.json")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    from atlas.bench.runner import sanitize_json

    with open(dest, "w") as fh:
        json.dump(sanitize_json(out), fh, indent=2, allow_nan=False)
    print("wrote", dest, "ok =", out["ok"], f"({out['wall_time_s']:.0f}s)")
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
