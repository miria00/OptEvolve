"""Phase 2 CENTERPIECE shared protocol: cell, fitness, workers, helpers.

Shared by scripts/run_centerpiece.py, scripts/tune_baseline.py and
scripts/eval_holdout.py (and reusable by the ablation phase). Pattern
follows scripts/run_mini_er.py; process-parallel evaluation follows
atlas.evolve.parallel (spawn pool, CPU-pinned workers).

PROTOCOL CONSTANTS (fixed across ALL arms/seeds; do not edit mid-phase):
- TRAIN: 8 netflow instances, degree-2 regular, sizes 300..800 nodes,
  cell seed 7 (instance i uses seed 7*10007 + i).
- WITHIN-distribution holdout: 10 fresh netflow instances, cell seed
  7007, sizes matched to the train range.
- Fitness: -SGM-10 of per-instance median-of-3 time-to-rel-gap<=1e-4 at
  f64 on CPU; failures capped at their budgeted wall time (PAR-1);
  instance-level TypeCheckError = certified construction failure = -inf.
- Genome menu: tau, sigma, rho, halpern, restart_k, metric(ruiz),
  switch_k. Named solvers NEVER appear in the menu; the backend gene,
  the avg mix and the deprecated residual-window switch are EXCLUDED
  from this phase by protocol (policy refusals, booked separately from
  convergence-condition rejections).
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_V2_ROOT = os.path.dirname(_SCRIPTS_DIR)
for _p in (_V2_ROOT, _SCRIPTS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

from atlas.bench.metrics import DEFAULT_SGM_SHIFT, sgm, time_to_tol  # noqa: E402
from atlas.genome import (  # noqa: E402
    OPNORM_LP,
    Genome,
    GenomeParams,
    canonical_dict,
    canonical_json,
    content_hash,
    genome_from_dict,
    validate,
)
from atlas.typing_ import TypeCheckError  # noqa: E402
from atlas.wrappers.km import Budget  # noqa: E402

# ---------------------------------------------------------------------------
# Protocol constants
# ---------------------------------------------------------------------------

CELL_SEED = 7
DEGREE = 2
TRAIN_NODES = (300, 371, 443, 514, 586, 657, 729, 800)  # linspace(300,800,8)
HOLDOUT_CELL_SEED = 7007
WITHIN_HOLDOUT_NODES = (
    300, 356, 411, 467, 522, 578, 633, 689, 744, 800
)  # linspace(300,800,10), matched to the train range

TOL_TRAIN = 1e-4
MAX_ITERS = 200_000
SAMPLE_EVERY = 100
N_REPEATS = 3

T0 = 0.9 / OPNORM_LP  # textbook tau = sigma = 0.9 / ||A||
OP2 = OPNORM_LP**2  # 8.3232 (1.02^2 * 8, degree-2 regular netflow gate)

# Holdout evaluation caps (Discipline block): runtime cap per solve,
# cap hits logged as failures; PAR-1 aggregation at the cap.
HOLDOUT_CAPS_S = {1e-4: 60.0, 1e-6: 120.0}
HOLDOUT_TOLS = (1e-4, 1e-6)
HOLDOUT_PROBE_ITERS = 3000
HOLDOUT_HARD_MAX_ITERS = 5_000_000
HOLDOUT_SAMPLE_EVERY = 100

# Sentinel config "genome": the textbook step rule re-instantiated per
# instance (tau = sigma = 0.9/||A||_instance); see holdout_task.
TEXTBOOK_PER_INSTANCE = "TEXTBOOK_PER_INSTANCE"

MECHANISMS = (
    "over_relaxation",
    "halpern",
    "restart",
    "metric",
    "switch_k",
    "scheme_changed",
)


# ---------------------------------------------------------------------------
# Instance specs (picklable; workers build lazily and cache)
# ---------------------------------------------------------------------------


def train_specs() -> list[dict]:
    return [
        {
            "kind": "netflow",
            "n_nodes": int(nn),
            "seed": CELL_SEED * 10007 + i,
            "degree": DEGREE,
            "name": f"netflow_train_{nn}n",
        }
        for i, nn in enumerate(TRAIN_NODES)
    ]


def within_holdout_specs() -> list[dict]:
    return [
        {
            "kind": "netflow",
            "n_nodes": int(nn),
            "seed": HOLDOUT_CELL_SEED * 10007 + i,
            "degree": DEGREE,
            "name": f"netflow_within_{nn}n",
        }
        for i, nn in enumerate(WITHIN_HOLDOUT_NODES)
    ]


def build_problem(spec: dict):
    """Instance spec -> lp_standard CompositeProblem (heavy; workers cache)."""
    if spec["kind"] == "netflow":
        from atlas.bench.lp_cell import from_netflow, to_composite
        from atlas.bench.netflow import gen_regular_flow

        nf = gen_regular_flow(
            n_nodes=spec["n_nodes"], seed=spec["seed"], degree=spec["degree"]
        )
        return to_composite(from_netflow(nf))
    if spec["kind"] == "mps":
        from atlas.bench.mps_cell import load_standard_form, to_composite

        prob = to_composite(load_standard_form(spec["path"]))
        prob.data["meta"].update(name=spec["name"], tier=spec.get("tier"))
        return prob
    raise ValueError(f"unknown instance spec kind {spec['kind']!r}")


# ---------------------------------------------------------------------------
# Genome helpers
# ---------------------------------------------------------------------------


def vanilla_pdhg() -> Genome:
    """The seed population: vanilla PDHG, textbook tau = sigma = 0.9/||A||."""
    return Genome(scheme="pdhg", params=GenomeParams(tau=T0, sigma=T0))


def genes_active(g: Genome) -> dict:
    d = canonical_dict(g)
    rho = d["params"]["rho"]
    return {
        "over_relaxation": rho is not None and rho != 1.0,
        "halpern": bool(d["wrapper"]["halpern"]),
        "restart": d["wrapper"]["restart"] == "fixed_k",
        "metric": d["metric"] is not None,
        "switch_k": d["mix"] is not None and d["mix"]["kind"] == "switch_k",
        "scheme_changed": d["scheme"] != "pdhg",
    }


def policy_violation(g: Genome) -> str | None:
    """Genes EXCLUDED from this phase by protocol (POLICY refusals, booked
    separately from convergence-condition rejections; Phase-1 F7/F9)."""
    if g.backend is not None:
        return "the 'backend' block is excluded from this experiment; remove it"
    if g.mix is not None and g.mix.kind == "avg":
        return (
            "the 'avg' mix is excluded from this experiment (provably "
            "vacuous on the LP cell); use a base genome or 'switch_k'"
        )
    if g.mix is not None and g.mix.kind == "switch":
        return (
            "the residual-window 'switch' mix is deprecated; use the "
            "finite-prefix 'switch_k' form"
        )
    if g.metric is not None and g.metric.kind != "ruiz":
        return (
            "only the {'kind':'ruiz','iters':N} metric gene is in this "
            "experiment's menu; remove or replace the metric block"
        )
    return None


def genome_json_to_genome(genome_dict_or_json) -> Genome:
    d = genome_dict_or_json
    if isinstance(d, str):
        d = json.loads(d)
    return genome_from_dict(d)


# ---------------------------------------------------------------------------
# Fitness evaluation (train protocol): serial + process-parallel paths
# ---------------------------------------------------------------------------

_WORKER_SPECS: list[dict] = []
_WORKER_PROBLEMS: dict[int, object] = {}


def worker_init(specs: list[dict], extra_env: dict | None = None) -> None:
    """Spawn-pool initializer: CPU pinning + the instance spec table."""
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    if extra_env:
        os.environ.update(extra_env)
    global _WORKER_SPECS, _WORKER_PROBLEMS
    _WORKER_SPECS = list(specs)
    _WORKER_PROBLEMS = {}


def _get_problem(idx: int):
    prob = _WORKER_PROBLEMS.get(idx)
    if prob is None:
        prob = build_problem(_WORKER_SPECS[idx])
        _WORKER_PROBLEMS[idx] = prob
    return prob


def fitness_task(args) -> dict:
    """One (genome, train instance) fitness sub-task: N_REPEATS solves,
    median time-to-tol with PAR-1 capping. Returns a plain dict."""
    genome_json, idx = args
    from atlas.schemes.compile import solve_genome

    g = genome_json_to_genome(genome_json)
    prob = _get_problem(idx)
    budget = Budget(max_iters=MAX_ITERS, tol=TOL_TRAIN, sample_every=SAMPLE_EVERY)
    reps_t, reps_i = [], []
    failed = False
    for _rep in range(N_REPEATS):
        try:
            r = solve_genome(prob, g, budget)
        except TypeCheckError as e:
            return {"idx": idx, "error": str(e)}
        gap = float(r.rel_gap_history[-1])
        if gap <= TOL_TRAIN:
            t, it = time_to_tol(r, TOL_TRAIN, sample_every=SAMPLE_EVERY)
            reps_t.append(float(t))
            reps_i.append(int(it))
        else:
            failed = True
            reps_t.append(float(r.wall_time))  # PAR-1 cap
            reps_i.append(int(r.iters))
    return {
        "idx": idx,
        "median_time_s": float(np.median(reps_t)),
        "median_iters": int(np.median(reps_i)),
        "failed": bool(failed),
    }


def eval_genome_train(g: Genome, n_instances: int, pool=None) -> tuple[float, dict]:
    """(fitness, metrics) on the train protocol. `pool` (optional) is a
    make_pool(...) pool whose workers were initialized with
    worker_init(train_specs()); serial fallback builds instances in-process.
    Semantics identical to run_mini_er.eval_genome."""
    gj = canonical_json(g)
    tasks = [(gj, i) for i in range(n_instances)]
    if pool is not None:
        rows = pool.map(fitness_task, tasks)
    else:
        global _WORKER_SPECS
        if not _WORKER_SPECS:
            worker_init(train_specs())
        rows = [fitness_task(t) for t in tasks]
    rows = sorted(rows, key=lambda r: r["idx"])
    for r in rows:
        if "error" in r:
            return float("-inf"), {"error": r["error"]}
    times = [r["median_time_s"] for r in rows]
    iters_at = [r["median_iters"] for r in rows]
    fails = sum(1 for r in rows if r["failed"])
    fit = -sgm(times, DEFAULT_SGM_SHIFT)
    return fit, {
        "sgm10_time_s": -fit,
        "sgm10_iters": sgm(iters_at, DEFAULT_SGM_SHIFT),
        "n_failed_instances": fails,
        "per_instance_time_s": times,
        "per_instance_iters": iters_at,
    }


# ---------------------------------------------------------------------------
# Holdout evaluation task (eval_holdout + ablation reuse)
# ---------------------------------------------------------------------------


def _round_to_sample(n: int, sample_every: int) -> int:
    return max(sample_every, int(math.ceil(n / sample_every)) * sample_every)


def holdout_task(args) -> dict:
    """One (config, instance, tolerance) holdout evaluation.

    args = (config_name, genome_json, idx, tol, cap_s).
    Wall-cap protocol (caps and truncations LOGGED, cap hits = failures):
      1. probe solve (HOLDOUT_PROBE_ITERS); if it certifies, done;
      2. else extrapolate the iteration rate and re-solve from scratch
         with max_iters = min(HARD_MAX, rate * cap_s);
      3. converged with time-to-certificate <= cap  -> "converged";
         converged past the cap                     -> "cap_time" failure;
         budget exhausted at the wall-derived bound -> "cap_time" failure;
         budget exhausted at HARD_MAX iterations    -> "cap_iters" failure;
         Level-1 typecheck failure on the instance  -> "typecheck_rejected".
    """
    config_name, genome_json, idx, tol, cap_s = args
    from atlas.schemes.compile import solve_genome

    prob = _get_problem(idx)
    if genome_json == TEXTBOOK_PER_INSTANCE:
        # The textbook STEP RULE applied to this instance's own operator
        # norm (tau = sigma = 0.9/||A||): the fair runnable baseline on
        # instances whose ||A|| differs from the train cell's.
        step = 0.9 / float(prob.L.norm_bound)
        g = Genome(scheme="pdhg", params=GenomeParams(tau=step, sigma=step))
    else:
        g = genome_json_to_genome(genome_json)
    spec = _WORKER_SPECS[idx]
    base = {
        "config": config_name,
        "genome": canonical_json(g),
        "instance": spec.get("name", str(idx)),
        "instance_idx": idx,
        "tier": spec.get("tier", "netflow_within"),
        "tol": tol,
        "cap_s": cap_s,
    }
    t_task0 = time.time()
    try:
        probe = solve_genome(
            prob,
            g,
            Budget(
                max_iters=HOLDOUT_PROBE_ITERS,
                tol=tol,
                sample_every=HOLDOUT_SAMPLE_EVERY,
            ),
        )
    except TypeCheckError as e:
        base.update(
            status="typecheck_rejected",
            error=str(e),
            wall_s=None,
            iters_to_cert=None,
            time_to_cert_s=None,
            final_rel_gap=None,
            task_wall_s=time.time() - t_task0,
        )
        return base

    def finish(r, allowed_iters, probe_wall, rate):
        gap = float(r.rel_gap_history[-1])
        cert = r.cert.to_dict() if r.cert is not None else None
        if gap <= tol:
            t, it = time_to_tol(r, tol, sample_every=HOLDOUT_SAMPLE_EVERY)
            status = "converged" if t <= cap_s else "cap_time"
            base.update(
                status=status,
                time_to_cert_s=float(t),
                iters_to_cert=int(it),
            )
        else:
            status = (
                "cap_iters"
                if allowed_iters >= HOLDOUT_HARD_MAX_ITERS
                else "cap_time"
            )
            base.update(status=status, time_to_cert_s=None, iters_to_cert=None)
        base.update(
            wall_s=float(r.wall_time),
            iters_run=int(r.iters),
            allowed_iters=int(allowed_iters),
            probe_wall_s=None if probe_wall is None else float(probe_wall),
            probe_rate_iters_per_s=None if rate is None else float(rate),
            final_rel_gap=gap,
            cert=cert,
            task_wall_s=time.time() - t_task0,
        )
        return base

    probe_gap = float(probe.rel_gap_history[-1])
    if probe_gap <= tol:
        return finish(probe, HOLDOUT_PROBE_ITERS, None, None)
    probe_wall = max(float(probe.wall_time), 1e-9)
    rate = float(probe.iters) / probe_wall
    allowed = min(
        HOLDOUT_HARD_MAX_ITERS,
        _round_to_sample(int(rate * cap_s), HOLDOUT_SAMPLE_EVERY),
    )
    r = solve_genome(
        prob,
        g,
        Budget(max_iters=allowed, tol=tol, sample_every=HOLDOUT_SAMPLE_EVERY),
    )
    return finish(r, allowed, probe_wall, rate)


def aggregate_holdout(rows: list[dict]) -> dict:
    """SGM-10 + solve rate per (config, tier, tol). Failures (any cap or
    typecheck rejection) enter the SGM at the runtime cap (PAR-1); they
    are also counted per failure class. Iteration SGM covers certified
    solves only (n reported)."""
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault((r["config"], r["tier"], r["tol"]), []).append(r)
    out: dict[str, dict] = {}
    for (config, tier, tol), rs in sorted(groups.items()):
        times, iters_ok = [], []
        n_conv = n_cap_time = n_cap_iters = n_reject = 0
        for r in rs:
            if r["status"] == "converged":
                n_conv += 1
                times.append(float(r["time_to_cert_s"]))
                iters_ok.append(int(r["iters_to_cert"]))
            else:
                times.append(float(r["cap_s"]))  # PAR-1 at the runtime cap
                if r["status"] == "cap_time":
                    n_cap_time += 1
                elif r["status"] == "cap_iters":
                    n_cap_iters += 1
                else:
                    n_reject += 1
        key = f"{config}|{tier}|tol={tol:g}"
        out[key] = {
            "config": config,
            "tier": tier,
            "tol": tol,
            "n_instances": len(rs),
            "solve_rate": n_conv / len(rs),
            "sgm10_time_s_par1_at_cap": sgm(times, DEFAULT_SGM_SHIFT),
            "sgm10_iters_certified_only": (
                sgm(iters_ok, DEFAULT_SGM_SHIFT) if iters_ok else None
            ),
            "n_certified": n_conv,
            "n_cap_time": n_cap_time,
            "n_cap_iters": n_cap_iters,
            "n_typecheck_rejected": n_reject,
        }
    return out


# ---------------------------------------------------------------------------
# Textbook defaults (all 5 schemes; inapplicable ones recorded as rejected)
# ---------------------------------------------------------------------------


def textbook_defaults() -> dict[str, Genome]:
    """Textbook-parameter genome per base scheme. On the LP cell only
    pdhg/condat_vu pass the gate; fbs/drs/dys are structurally
    inapplicable and their gate rejections are recorded, not hidden."""
    return {
        "pdhg": Genome(scheme="pdhg", params=GenomeParams(tau=T0, sigma=T0)),
        "condat_vu": Genome(
            scheme="condat_vu", params=GenomeParams(tau=T0, sigma=T0)
        ),
        "fbs": Genome(scheme="fbs", params=GenomeParams(tau=0.125)),
        "drs": Genome(scheme="drs", params=GenomeParams(tau=1.0)),
        "dys": Genome(scheme="dys", params=GenomeParams(tau=1.0)),
    }


def eval_textbook_defaults(pool=None) -> dict:
    """Fitness of every textbook default on the train set + the best name."""
    rows: dict[str, dict] = {}
    best_name, best_fit = None, float("-inf")
    for name, g in textbook_defaults().items():
        v = validate(g, problem_kind="lp_netflow")
        if not v.ok:
            rows[name] = {
                "genome": canonical_json(g),
                "gate_rejected": True,
                "error": v.error,
            }
            continue
        fit, m = eval_genome_train(g, len(TRAIN_NODES), pool=pool)
        rows[name] = {
            "genome": canonical_json(g),
            "gate_rejected": False,
            "fitness": fit,
            "metrics": m,
        }
        if fit > best_fit:
            best_name, best_fit = name, fit
    return {
        "per_scheme": rows,
        "best_textbook_default": best_name,
        "best_textbook_fitness": best_fit,
    }


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------


def machine_record(machine: str) -> dict:
    import platform

    return {
        "machine": machine,
        "hostname": platform.node(),
        "jax_platform": "cpu",
        "python": sys.version.split()[0],
    }
