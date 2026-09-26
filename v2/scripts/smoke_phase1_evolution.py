"""Phase-1 integration smoke: 3-generation evolution on a small netflow LP
with the full gene menu, switch_k and metric included.

Checks, per gene kind (switch_k, metric):
  - at least one genome PROPOSED by the backend,
  - at least one genome passes the Stage-0 GATE (and a deliberately
    invalid genome of the same kind is rejected by the gate),
  - at least one genome EVALUATED end-to-end via solve_genome.

Writes results/sprint/phase1/smoke_evolution.json.

Usage: JAX_PLATFORMS=cpu python scripts/smoke_phase1_evolution.py
"""
from __future__ import annotations

import json
import os
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from atlas.bench.lp_cell import from_netflow, gate_opnorm, to_composite  # noqa: E402
from atlas.bench.metrics import time_to_tol  # noqa: E402
from atlas.bench.netflow import gen_regular_flow  # noqa: E402
from atlas.evolve.loop import evolve  # noqa: E402
from atlas.evolve.random_backend import RandomBackend  # noqa: E402
from atlas.genome import (  # noqa: E402
    Genome,
    GenomeParams,
    Metric,
    Mix,
    SubScheme,
    canonical_json,
    validate,
)
from atlas.schemes.compile import solve_genome  # noqa: E402
from atlas.typing_ import TypeCheckError  # noqa: E402
from atlas.wrappers.km import Budget  # noqa: E402

GENERATIONS = 3
POP_SIZE = 8
TOL = 1e-4
SEED = 42
OUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "sprint", "phase1", "smoke_evolution.json",
)


def genome_kinds(g: Genome):
    kinds = []
    if g.mix is not None and g.mix.kind == "switch_k":
        kinds.append("switch_k")
    if g.metric is not None:
        kinds.append("metric")
    if not kinds:
        kinds.append("base")
    return kinds


def main() -> int:
    t0 = time.time()
    instances = [
        to_composite(from_netflow(gen_regular_flow(n_nodes=24, seed=s, degree=2)))
        for s in (0, 1)
    ]
    opnorm = gate_opnorm(instances)

    counts = {k: {"proposed": 0, "gate_passed": 0, "evaluated": 0,
                  "converged": 0}
              for k in ("switch_k", "metric", "base")}
    eval_log = []

    def fitness_fn(g: Genome):
        kinds = genome_kinds(g)
        v = validate(g, problem_kind="lp_netflow", lp_opnorm=opnorm)
        for k in kinds:
            counts[k]["proposed"] += 1
            if v.ok:
                counts[k]["gate_passed"] += 1
        if not v.ok:
            return float("-inf"), {"gate": v.error}
        budget = Budget(max_iters=40_000, tol=TOL, sample_every=100)
        times, ok = [], True
        for prob in instances:
            try:
                r = solve_genome(prob, g, budget)
            except TypeCheckError as e:
                return float("-inf"), {"error": str(e)}
            gap = float(r.rel_gap_history[-1])
            if gap <= TOL:
                t, _ = time_to_tol(r, TOL, sample_every=100)
                times.append(float(t))
            else:
                ok = False
                times.append(10.0)
        for k in kinds:
            counts[k]["evaluated"] += 1
            if ok:
                counts[k]["converged"] += 1
        fit = -sum(times) / len(times)
        eval_log.append({"kinds": kinds, "fitness": fit,
                         "genome": canonical_json(g)})
        return fit, {"converged": ok}

    backend = RandomBackend(
        seed=SEED, problem_kind="lp_netflow",
        p_mix=0.05, p_switch=0.30, p_metric=0.45,
    )
    hist = evolve(backend, fitness_fn, generations=GENERATIONS,
                  pop_size=POP_SIZE, seed=SEED, problem_kind="lp_netflow")

    # Gate rejection check: deliberately invalid genomes of each kind.
    bad_switch = Genome(
        scheme="mix", params=GenomeParams(tau=0.1, sigma=0.1),
        mix=Mix(kind="switch_k", K=999_999,
                aggressive=SubScheme(scheme="pdhg", tau=0.5, sigma=0.5),
                tail=SubScheme(scheme="pdhg", tau=0.1, sigma=0.1)),
    )
    bad_metric = Genome(
        scheme="pdhg", params=GenomeParams(tau=0.1, sigma=0.1),
        metric=Metric(kind="ruiz", iters=99),
    )
    v_sw = validate(bad_switch, problem_kind="lp_netflow", lp_opnorm=opnorm)
    v_me = validate(bad_metric, problem_kind="lp_netflow", lp_opnorm=opnorm)
    gate_rejections = {
        "switch_k_invalid_rejected": (not v_sw.ok, v_sw.error),
        "metric_invalid_rejected": (not v_me.ok, v_me.error),
    }

    ok = all(
        counts[k]["proposed"] >= 1 and counts[k]["gate_passed"] >= 1
        and counts[k]["evaluated"] >= 1
        for k in ("switch_k", "metric")
    ) and not v_sw.ok and not v_me.ok

    out = {
        "protocol": {"generations": GENERATIONS, "pop_size": POP_SIZE,
                     "tol": TOL, "seed": SEED, "cell": "netflow n_nodes=24 x2",
                     "menu": "pdhg/condat_vu + rho/halpern/restart_k + avg "
                             "+ switch_k + metric(ruiz|pock_chambolle)"},
        "counts": counts,
        "gate_rejections": gate_rejections,
        "best_fitness": hist["best_fitness"],
        "best_genome": hist["best_genome"],
        "n_unique_evaluated": hist["n_unique_evaluated"],
        "flat_fitness_warning": hist["flat_fitness_warning"],
        "wall_s": time.time() - t0,
        "PASS": ok,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(json.dumps({"counts": counts, "PASS": ok,
                      "best_fitness": hist["best_fitness"],
                      "wall_s": round(out["wall_s"], 1)}, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
