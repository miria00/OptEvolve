"""MIX-DISCOVERY experiment driver: can a certified mixed/switched method
beat the best conventional (single-scheme) method on held-out instances?

Two cells, full gene menu (5 schemes incl. dys, avg blends, safeguarded
switch, rho, halpern + fixed-k restart; the backend/precision gene is
DISABLED so every phenotype runs the exact P0 f64 path, for fairness
against the f64 baselines):

  tv : TV-on-OCT, 128x128 Kermany patches, 6 train / 3 held-out, seed 7.
  lp : standard-form LP on degree-2 regular min-cost netflow (the
       canonical lp_netflow gate family, n_nodes=500 -> 1000 arcs),
       6 train / 3 held-out, instance seeds 7*10007+i. NOTE: no prior
       E-R run artifact exists on either machine (checked 2026-08-26);
       sizes were fixed by pilot timing so one evolve run fits the
       45-minute cap (balanced-default PDHG needs 39k-126k iters,
       0.25-0.8 s per instance on the 4090 box CPU).

Modes:
  evolve   : one evolution run (LLM-guided proposals with random
             fallback), 14 generations x 10 proposals by default.
  tune-lp  : the LP conventional bar: textbook defaults + 60 iterations
             of random search over SINGLE-scheme genomes (no mix genes).
  finalize : evaluate all champions + baselines on the held-out split
             (N=3 timing repeats, one machine), classify phenotypes,
             apply the pre-registered win condition, write
             results/sprint/mix_discovery.json.

Win condition (pre-registered): a genome whose PHENOTYPE genuinely mixes
(avg with lambda in (0.05, 0.95) and operator-distinct parents, or a
switch whose aggressive branch fires on >10% of accepted steps) beats the
conventional bar on held-out SGM by >10% in ALL 3 evolve seeds; anything
less is reported as a tie/loss.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import socket
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np

from atlas.bench.kermany import load_patches
from atlas.bench.lp_cell import from_netflow, to_composite
from atlas.bench.metrics import sgm, time_to_tol
from atlas.bench.netflow import gen_regular_flow
from atlas.bench.runner import sanitize_json
from atlas.evolve.llm_backend import LLMBackend
from atlas.evolve.loop import evolve
from atlas.evolve.random_backend import RandomBackend
from atlas.genome import (
    SCHEMA_DOC,
    Genome,
    GenomeParams,
    canonical_json,
    genome_from_dict,
    validate,
)
from atlas.ir.spec import tv_denoise_problem
from atlas.schemes.compile import genome_cap_scheme, solve_genome
from atlas.typing_ import TypeCheckError
from atlas.wrappers.km import Budget

V2_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(V2_ROOT, "results", "sprint", "mixdisc")
FINAL_JSON = os.path.join(V2_ROOT, "results", "sprint", "mix_discovery.json")

FULL_MENU = ("fbs", "drs", "pdhg", "condat_vu", "dys")

CELLS = {
    "tv": {
        "problem_kind": "tv_denoise",
        "size": 128,
        "lam": 0.1,
        "n_train": 6,
        "n_holdout": 3,
        "instance_seed": 7,
        "tol": 1e-4,
        "sample_every": 50,
        # run_tv_cell.py convention: roughly equal wall-time per solve
        "caps": {"fbs": 8000, "drs": 1200, "pdhg": 8000, "condat_vu": 8000,
                 "dys": 8000},
    },
    "lp": {
        "problem_kind": "lp_netflow",
        "family": "regular_deg2",
        "n_nodes": 500,
        "degree": 2,
        "n_train": 6,
        "n_holdout": 3,
        "instance_seed": 7,
        "tol": 1e-4,
        "sample_every": 100,
        # plain/avg iterations ~6 us at n=500; the switch monitor runs the
        # f64 certificate every candidate step (~2x/iter): half cap =
        # roughly equal wall-time budget
        "caps": {"pdhg": 400_000, "condat_vu": 400_000, "fbs": 400_000,
                 "drs": 400_000, "dys": 400_000, "switch": 200_000},
    },
}

TV_DOC = (
    "Problem cell: TV denoising of a 128x128 OCT patch, "
    "min_x 0.5||x-y||^2 + 0.1||Dx||_1, ||D|| <= sqrt(8). Applicable "
    "schemes: fbs (dual, step in (0,0.25)), drs, pdhg (tau*sigma*8 < 1), "
    "condat_vu (tau/2 + tau*sigma*8 < 1). dys is structurally "
    "inapplicable here. The backend/precision gene is DISABLED in this "
    "experiment: always omit the 'backend' block."
)

LP_DOC = (
    "Problem cell: standard-form LP min c'x s.t. Ax=b, x>=0 (min-cost "
    "network flow on degree-2 regular graphs, 500 nodes / 1000 arcs), "
    "||A|| <= 2.885. ONLY pdhg and condat_vu are structurally applicable "
    "(f is linear so fbs/drs/dys are rejected). Side condition: "
    "tau*sigma*||A||^2 < 1 with ||A||^2 <= 8.33. The balanced default "
    "tau=sigma=0.343 needs 40k-130k iterations to reach rel gap 1e-4; "
    "asymmetric tau vs sigma, over-relaxation rho, halpern anchor with "
    "fixed-k restarts, avg blends (identical (tau,sigma) required) and "
    "safeguarded switch genomes (aggressive branch may break the side "
    "condition) are all available. The backend/precision gene is "
    "DISABLED in this experiment: always omit the 'backend' block."
)

MENU_TV = """Mutation menu (pick ONE or TWO small moves):
- switch base scheme (fbs/drs/pdhg/condat_vu; dys is invalid on this cell)
- scale tau or sigma (e.g. x0.5, x2, x10)
- rebalance tau vs sigma keeping the product fixed
- set/unset over-relaxation rho (e.g. 1.5, 1.8; mutually exclusive with halpern)
- toggle wrapper.halpern; with halpern on: restart "fixed_k" + restart_k
  (25..400) resets the anchor every k iterations
- MIX two schemes: {"scheme":"mix","mix":{"kind":"avg","lambda":0.5,
  "scheme_a":{...},"scheme_b":{...}}} blends lambda*T_a+(1-lambda)*T_b;
  parents must be the same scheme, or pdhg+condat_vu with IDENTICAL
  (tau, sigma)
- SAFEGUARDED SWITCH: {"scheme":"mix","mix":{"kind":"switch","window":25,
  "aggressive":{...},"fallback":{...}}}; the aggressive branch may break
  the stepsize side conditions (big steps!), the fallback must be valid;
  same-scheme or pdhg/condat_vu branches
Do NOT emit a "backend" block (disabled in this experiment)."""

MENU_LP = """Mutation menu (pick ONE or TWO small moves):
- base scheme pdhg or condat_vu only (fbs/drs/dys are invalid on this cell)
- scale tau or sigma (e.g. x0.5, x2, x10)
- rebalance tau vs sigma keeping the product fixed (primal-dual step
  asymmetry is often the biggest lever on LP)
- set/unset over-relaxation rho (e.g. 1.5, 1.8; mutually exclusive with halpern)
- toggle wrapper.halpern; with halpern on: restart "fixed_k" + restart_k
  (25..400) resets the anchor every k iterations (the HPR-LP trick)
- MIX: {"scheme":"mix","mix":{"kind":"avg","lambda":0.5,"scheme_a":{...},
  "scheme_b":{...}}}; parents same scheme or pdhg+condat_vu, ALWAYS with
  IDENTICAL (tau, sigma) on this cell
- SAFEGUARDED SWITCH: {"scheme":"mix","mix":{"kind":"switch","window":25,
  "aggressive":{...},"fallback":{...}}}; the aggressive branch may break
  tau*sigma*||A||^2 < 1 (big steps!), the fallback must satisfy it
Do NOT emit a "backend" block (disabled in this experiment)."""


# ---------------------------------------------------------------------------
# Instances
# ---------------------------------------------------------------------------


def build_cell(cell: str):
    cfg = CELLS[cell]
    if cell == "tv":
        n = cfg["n_train"] + cfg["n_holdout"]
        patches = load_patches(n, size=cfg["size"], split="test",
                               seed=cfg["instance_seed"])
        problems = [tv_denoise_problem(p, lam=cfg["lam"]) for p in patches]
    else:
        n = cfg["n_train"] + cfg["n_holdout"]
        problems = [
            to_composite(from_netflow(gen_regular_flow(
                n_nodes=cfg["n_nodes"],
                seed=cfg["instance_seed"] * 10007 + i,
                degree=cfg["degree"],
            )))
            for i in range(n)
        ]
    return problems[: cfg["n_train"]], problems[cfg["n_train"]:]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def cap_for(g: Genome, caps: dict) -> int:
    if g.scheme == "mix" and g.mix is not None and g.mix.kind == "switch" \
            and "switch" in caps:
        return caps["switch"]
    return caps[genome_cap_scheme(g)]


def eval_genome(g: Genome, instances, tol, caps, sample_every) -> dict:
    budget = Budget(max_iters=cap_for(g, caps), tol=tol,
                    sample_every=sample_every)
    rows, strict_times, iters_l, n_fail = [], [], [], 0
    for ii, problem in enumerate(instances):
        try:
            result = solve_genome(problem, g, budget)
        except TypeCheckError as e:
            rows.append({"instance": ii, "error": f"TypeCheckError: {e}",
                         "converged": False})
            strict_times.append(float("inf"))
            iters_l.append(0)
            n_fail += 1
            continue
        gap = float(result.rel_gap_history[-1])
        converged = gap <= tol
        t_tol, _ = time_to_tol(result, tol, sample_every=sample_every)
        rows.append({
            "instance": ii,
            "iters": int(result.iters),
            "wall_time_s": float(result.wall_time),
            "rel_gap_achieved": gap,
            "time_to_tol_s": float(t_tol) if converged else None,
            "converged": bool(converged),
            "extra": result.extra,
        })
        strict_times.append(float(t_tol) if converged else float("inf"))
        iters_l.append(int(result.iters))
        if not converged:
            n_fail += 1
    return {
        "genome": canonical_json(g),
        "sgm10_time_s": sgm(strict_times),
        "sgm10_iters": sgm(iters_l),
        "failure_rate": n_fail / max(len(instances), 1),
        "rows": rows,
    }


def genome_fitness(g: Genome, instances, tol, caps, sample_every):
    ev = eval_genome(g, instances, tol, caps, sample_every)
    fit = -ev["sgm10_time_s"] if math.isfinite(ev["sgm10_time_s"]) else -math.inf
    is_mix = g.scheme == "mix"
    metrics = {
        "sgm10_time_s": (round(ev["sgm10_time_s"], 4)
                         if math.isfinite(ev["sgm10_time_s"]) else float("inf")),
        "failure_rate": ev["failure_rate"],
        "sgm10_iters": round(ev["sgm10_iters"], 1),
        "is_mix": is_mix,
        "mix_kind": (g.mix.kind if is_mix and g.mix is not None else None),
    }
    return fit, metrics


# -- process-parallel machinery (spawn-safe, mirrors run_tv_cell.py) --------

_POOL_CTX: dict = {}


def _worker_init(cell: str, tv_patches, lam):
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import jax as _jax

    _jax.config.update("jax_enable_x64", True)
    cfg = CELLS[cell]
    if cell == "tv":
        instances = [tv_denoise_problem(p, lam=lam) for p in tv_patches]
    else:
        instances = [
            to_composite(from_netflow(gen_regular_flow(
                n_nodes=cfg["n_nodes"],
                seed=cfg["instance_seed"] * 10007 + i,
                degree=cfg["degree"],
            )))
            for i in range(cfg["n_train"])
        ]
    _POOL_CTX.update(instances=instances, tol=cfg["tol"], caps=cfg["caps"],
                     sample_every=cfg["sample_every"])


def _fitness_worker(g: Genome):
    return genome_fitness(g, _POOL_CTX["instances"], _POOL_CTX["tol"],
                          _POOL_CTX["caps"], _POOL_CTX["sample_every"])


class LoggingPool:
    """Wraps a multiprocessing pool so every (genome, fitness, metrics)
    triple evaluated through it is recorded parent-side."""

    def __init__(self, pool, log: list):
        self.pool = pool
        self.log = log

    def map(self, fn, genomes):
        genomes = list(genomes)
        outs = list(self.pool.map(fn, genomes))
        for g, out in zip(genomes, outs):
            fit, metrics = out if isinstance(out, tuple) else (float(out), {})
            self.log.append({"genome": canonical_json(g),
                             "fitness": float(fit), "metrics": metrics})
        return outs


# ---------------------------------------------------------------------------
# LLM backend subclass: cell doc, custom menu, backend gene disabled
# ---------------------------------------------------------------------------


class MixDiscLLM(LLMBackend):
    def __init__(self, cell_doc: str, menu: str, **kw):
        super().__init__(**kw)
        self.cell_doc = cell_doc
        self.menu = menu

    def _build_prompt(self, feedback):
        parts = [
            self.cell_doc,
            SCHEMA_DOC,
            self._best_block(),
            self.menu,
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
            return None, ("the backend/precision gene is disabled in this "
                          "experiment; omit the 'backend' block")
        return g, err

    def ask(self, k):
        out = super().ask(k)
        # fallback proposals can still carry backend genes only if the
        # fallback backend was misconfigured; guard hard here.
        assert all(g.backend is None for g in out)
        return out


def make_backends(cell: str, evolve_seed: int, llm_url: str):
    cfg = CELLS[cell]
    fallback = RandomBackend(
        seed=evolve_seed + 1,
        problem_kind=cfg["problem_kind"],
        schemes=FULL_MENU,
        p_alpha=0.0,
        p_backend=0.0,
    )
    llm = MixDiscLLM(
        cell_doc=TV_DOC if cell == "tv" else LP_DOC,
        menu=MENU_TV if cell == "tv" else MENU_LP,
        base_url=llm_url,
        seed=evolve_seed,
        fallback=fallback,
        problem_kind=cfg["problem_kind"],
        hardware=None,
    )
    return llm


# ---------------------------------------------------------------------------
# Phenotype classification
# ---------------------------------------------------------------------------


def mix_genotype(cell: str, gd: dict):
    """Genotype-level mix classification of a canonical genome dict."""
    m = gd.get("mix")
    if gd.get("scheme") != "mix" or not m:
        return None
    if m["kind"] == "avg":
        a, b = m["scheme_a"], m["scheme_b"]
        distinct = (a["scheme"] != b["scheme"]
                    or a["params"]["tau"] != b["params"]["tau"]
                    or a["params"]["sigma"] != b["params"]["sigma"])
        if cell == "lp" and {a["scheme"], b["scheme"]} == {"pdhg", "condat_vu"}:
            # L_f = 0 on the LP cell: condat_vu's gradient step on linear f
            # IS pdhg's prox step (both v - tau*c), so the pair is one and
            # the same operator; the blend is phenotypically trivial.
            same_steps = (a["params"]["tau"] == b["params"]["tau"]
                          and a["params"]["sigma"] == b["params"]["sigma"])
            distinct = distinct and not same_steps
        return {"kind": "avg", "lambda": m["lambda"],
                "lambda_in_range": 0.05 < m["lambda"] < 0.95,
                "distinct_parents": bool(distinct),
                "genuine_candidate": bool(distinct and 0.05 < m["lambda"] < 0.95)}
    return {"kind": "switch", "window": m["window"],
            "genuine_candidate": True}  # settled by runtime acceptance stats


def switch_acceptance(rows) -> float | None:
    acc = rej = 0
    for r in rows:
        ex = r.get("extra") or {}
        if "n_aggressive_accepted" in ex:
            acc += int(ex["n_aggressive_accepted"])
            rej += int(ex["n_fallback_steps"])
    tot = acc + rej
    return (acc / tot) if tot > 0 else None


def phenotype_mixing(cell: str, gd: dict, holdout_rows) -> dict:
    info = mix_genotype(cell, gd)
    if info is None:
        return {"is_mix": False, "genuine": False}
    if info["kind"] == "avg":
        return {"is_mix": True, "genuine": info["genuine_candidate"],
                **info}
    frac = switch_acceptance(holdout_rows)
    return {"is_mix": True,
            "genuine": frac is not None and frac > 0.10,
            "aggressive_accept_fraction": frac, **info}


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------


def mode_evolve(args):
    cell = args.cell
    cfg = CELLS[cell]
    os.makedirs(OUT_DIR, exist_ok=True)
    train, _ = build_cell(cell)
    print(f"[mixdisc:{cell}] {len(train)} train instances built")

    log: list = []
    pool = None
    if args.workers > 0:
        import multiprocessing as mp

        tv_patches = ([np.asarray(p.data["y"]) for p in train]
                      if cell == "tv" else None)
        raw = mp.get_context("spawn").Pool(
            args.workers, initializer=_worker_init,
            initargs=(cell, tv_patches, cfg.get("lam")),
        )
        pool = LoggingPool(raw, log)
        fitness = _fitness_worker
        print(f"[mixdisc:{cell}] {args.workers} spawn workers")
    else:
        def fitness(g):
            fit, metrics = genome_fitness(g, train, cfg["tol"], cfg["caps"],
                                          cfg["sample_every"])
            log.append({"genome": canonical_json(g), "fitness": fit,
                        "metrics": metrics})
            return fit, metrics

    backend = make_backends(cell, args.evolve_seed, args.llm_url)
    t0 = time.perf_counter()
    hist = evolve(
        backend, fitness, args.generations, args.pop_size,
        seed=args.evolve_seed, problem_kind=cfg["problem_kind"],
        pool=pool,
    )
    wall = time.perf_counter() - t0
    if pool is not None:
        pool.pool.close()
        pool.pool.join()

    n_mix = sum(1 for e in log if (e["metrics"] or {}).get("is_mix"))
    out = {
        "experiment": "mix_discovery_evolve",
        "cell": cell,
        "cell_config": cfg,
        "evolve_seed": args.evolve_seed,
        "generations": args.generations,
        "pop_size": args.pop_size,
        "backend": "llm+random_fallback",
        "llm_model": backend.model,
        "host": socket.gethostname(),
        "wall_time_s": wall,
        "n_llm_calls": getattr(backend, "n_llm_calls", None),
        "n_llm_fallbacks": backend.n_fallbacks,
        "n_stage0_rejections_backend": backend.stage0_rejections,
        "n_mix_evaluated": n_mix,
        "history": hist,
        "evaluated": log,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    path = os.path.join(OUT_DIR, f"{cell}_evolve_seed{args.evolve_seed}.json")
    with open(path, "w") as f:
        json.dump(sanitize_json(out), f, indent=1)
    print(f"[mixdisc:{cell}] seed {args.evolve_seed}: best fitness "
          f"{hist['best_fitness']:.4f} wall {wall:.0f}s -> {path}")
    print(f"[mixdisc:{cell}] best genome: {hist['best_genome']}")


def lp_textbook_defaults():
    opn = 1.02 * math.sqrt(8.0)
    t = 0.99 / opn
    return {
        "pdhg": Genome(scheme="pdhg", params=GenomeParams(tau=t, sigma=t)),
        "condat_vu": Genome(scheme="condat_vu",
                            params=GenomeParams(tau=t, sigma=t)),
    }


def tv_defaults():
    return {
        "fbs": Genome(scheme="fbs", params=GenomeParams(tau=0.125)),
        "drs": Genome(scheme="drs", params=GenomeParams(tau=1.0)),
        "pdhg": Genome(scheme="pdhg",
                       params=GenomeParams(tau=0.5, sigma=0.99 / 4.0)),
        "condat_vu": Genome(scheme="condat_vu",
                            params=GenomeParams(tau=0.5, sigma=0.99 * 0.1875)),
    }


def mode_tune_lp(args):
    cfg = CELLS["lp"]
    os.makedirs(OUT_DIR, exist_ok=True)
    train, _ = build_cell("lp")
    rb = RandomBackend(
        seed=args.tune_seed, problem_kind="lp_netflow", schemes=FULL_MENU,
        p_alpha=0.0, p_backend=0.0, p_mix=0.0, p_switch=0.0,
    )
    rows = []
    candidates = {"default_" + k: g for k, g in lp_textbook_defaults().items()}
    for i, g in enumerate(rb.ask(args.tune_iters)):
        candidates[f"rs_{i:02d}"] = g
    t0 = time.perf_counter()
    for name, g in candidates.items():
        ev = eval_genome(g, train, cfg["tol"], cfg["caps"], cfg["sample_every"])
        rows.append({"name": name, "genome": ev["genome"],
                     "train_sgm10_s": ev["sgm10_time_s"],
                     "failure_rate": ev["failure_rate"],
                     "sgm10_iters": ev["sgm10_iters"]})
        print(f"  [{name}] sgm10={ev['sgm10_time_s']:.4g} "
              f"fail={ev['failure_rate']:.2f}")
    finite = [r for r in rows if math.isfinite(r["train_sgm10_s"])]
    best = min(finite, key=lambda r: r["train_sgm10_s"])
    out = {
        "experiment": "lp_conventional_bar",
        "cell_config": cfg,
        "tune_seed": args.tune_seed,
        "tune_iters": args.tune_iters,
        "note": ("single-scheme genomes only (p_mix=p_switch=0); wrappers "
                 "rho/halpern/restart allowed: they are conventional named "
                 "techniques (over-relaxation, Halpern anchor)"),
        "host": socket.gethostname(),
        "wall_time_s": time.perf_counter() - t0,
        "rows": rows,
        "best": best,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    path = os.path.join(OUT_DIR, "lp_tuned_baseline.json")
    with open(path, "w") as f:
        json.dump(sanitize_json(out), f, indent=1)
    print(f"[tune-lp] best {best['name']} train sgm10 "
          f"{best['train_sgm10_s']:.4g} -> {path}")


# ---------------------------------------------------------------------------
# Finalize
# ---------------------------------------------------------------------------

SEED42_TV_CHAMPION = Genome(
    scheme="condat_vu", params=GenomeParams(tau=0.05, sigma=1.0, rho=1.9)
)


def _holdout_eval(g: Genome, holdout, cfg, repeats: int) -> dict:
    per_rep = []
    rows_last = None
    for _ in range(repeats):
        ev = eval_genome(g, holdout, cfg["tol"], cfg["caps"],
                         cfg["sample_every"])
        per_rep.append(ev["sgm10_time_s"])
        rows_last = ev["rows"]
    finite = [t for t in per_rep if math.isfinite(t)]
    return {
        "genome": canonical_json(g),
        "holdout_sgm10_s_repeats": per_rep,
        "holdout_sgm10_s_mean": (float(np.mean(finite)) if len(finite) == len(per_rep)
                                 else float("inf")),
        "holdout_sgm10_s_min": min(per_rep),
        "holdout_sgm10_s_max": max(per_rep),
        "failure_rate": (sum(1 for r in rows_last if not r["converged"])
                         / len(rows_last)),
        "rows": rows_last,
    }


def _gate_record(g: Genome, kind: str) -> dict:
    v = validate(g, problem_kind=kind)
    return {"ok": v.ok, "error": v.error,
            "cert": (v.cert.to_dict() if v.cert is not None else None)}


def _num(x) -> float:
    """Undo sanitize_json's non-finite-float stringification."""
    if isinstance(x, str):
        return {"inf": math.inf, "-inf": -math.inf}.get(x, math.nan)
    return float(x)


def _best_from_logs(entries, want_mix, cell):
    """Best entry by fitness among mix/non-mix evaluated genomes."""
    best = None
    for e in entries:
        e = dict(e, fitness=_num(e["fitness"]))
        gd = json.loads(e["genome"])
        is_mix = gd.get("scheme") == "mix"
        if want_mix == "genuine_mix":
            info = mix_genotype(cell, gd)
            ok = (info is not None and info.get("genuine_candidate"))
        elif want_mix == "mix":
            ok = is_mix
        else:
            ok = not is_mix
        if not ok or not math.isfinite(e["fitness"]):
            continue
        if best is None or e["fitness"] > best["fitness"]:
            best = e
    return best


def mode_finalize(args):
    repeats = args.repeats
    result = {
        "experiment": "MIX-DISCOVERY",
        "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "eval_host": socket.gethostname(),
        "eval_note": ("all held-out numbers in this file were measured on "
                      "this one host, CPU, f64, serial, N="
                      f"{repeats} timing repeats; evolution fitness was "
                      "measured on the run host recorded per run"),
        "win_condition": ("genuine-mix phenotype beats the conventional bar "
                          "on held-out SGM10 by >10% in all 3 evolve seeds"),
        "cells": {},
    }
    for cell in ("tv", "lp"):
        cfg = CELLS[cell]
        kind = cfg["problem_kind"]
        train, holdout = build_cell(cell)
        cellres = {"config": cfg, "baselines": {}, "runs": {},
                   "mix_champions": {}}

        # -- conventional baselines ---------------------------------------
        if cell == "tv":
            defaults = tv_defaults()
            # pick best default on TRAIN
            tr = {}
            for name, g in defaults.items():
                ev = eval_genome(g, train, cfg["tol"], cfg["caps"],
                                 cfg["sample_every"])
                tr[name] = ev["sgm10_time_s"]
            best_def = min(tr, key=lambda k: tr[k])
            conv = {
                "best_default": defaults[best_def],
                "seed42_evolved_champion": SEED42_TV_CHAMPION,
            }
            cellres["baselines"]["train_default_sgm10_s"] = tr
        else:
            defaults = lp_textbook_defaults()
            tr = {}
            for name, g in defaults.items():
                ev = eval_genome(g, train, cfg["tol"], cfg["caps"],
                                 cfg["sample_every"])
                tr[name] = ev["sgm10_time_s"]
            best_def = min(tr, key=lambda k: tr[k])
            with open(os.path.join(OUT_DIR, "lp_tuned_baseline.json")) as f:
                tuned = json.load(f)
            conv = {
                "best_default": defaults[best_def],
                "tuned_random_search_60": genome_from_dict(
                    json.loads(tuned["best"]["genome"])),
            }
            cellres["baselines"]["train_default_sgm10_s"] = tr
            cellres["baselines"]["tuned_baseline_meta"] = {
                "name": tuned["best"]["name"],
                "train_sgm10_s": tuned["best"]["train_sgm10_s"],
                "tune_iters": tuned["tune_iters"],
                "tune_seed": tuned["tune_seed"],
            }

        # -- load evolution runs ------------------------------------------
        seeds = args.seeds
        all_entries = []
        for s in seeds:
            path = os.path.join(OUT_DIR, f"{cell}_evolve_seed{s}.json")
            with open(path) as f:
                run = json.load(f)
            entries = run["evaluated"]
            all_entries.extend(entries)
            cellres["runs"][str(s)] = {
                "host": run["host"],
                "wall_time_s": run["wall_time_s"],
                "best_fitness": run["history"]["best_fitness"],
                "best_genome": run["history"]["best_genome"],
                "n_unique_evaluated": run["history"]["n_unique_evaluated"],
                "n_mix_evaluated": run["n_mix_evaluated"],
                "n_llm_fallbacks": run["n_llm_fallbacks"],
                "n_stage0_rejections_backend":
                    run["n_stage0_rejections_backend"],
                "per_gen_best": [g["best_fitness"]
                                 for g in run["history"]["generations"]],
                "flat_fitness_warning":
                    run["history"].get("flat_fitness_warning"),
            }

        # third conventional bar: best single-scheme genome found by the
        # evolution runs themselves
        best_single = _best_from_logs(all_entries, "single", cell)
        if best_single is not None:
            conv["best_evolved_single_scheme"] = genome_from_dict(
                json.loads(best_single["genome"]))

        # -- held-out evaluation ------------------------------------------
        print(f"[finalize:{cell}] evaluating "
              f"{len(conv)} baselines + mix champions on holdout "
              f"(N={repeats} repeats) ...")
        for name, g in conv.items():
            ev = _holdout_eval(g, holdout, cfg, repeats)
            ev["gate"] = _gate_record(g, kind)
            cellres["baselines"][name] = ev
            print(f"    {name:28s} holdout sgm10 "
                  f"{ev['holdout_sgm10_s_mean']:.4g} "
                  f"[{ev['holdout_sgm10_s_min']:.4g}, "
                  f"{ev['holdout_sgm10_s_max']:.4g}]")

        bar_name = min(
            (n for n in conv),
            key=lambda n: cellres["baselines"][n]["holdout_sgm10_s_mean"],
        )
        bar = cellres["baselines"][bar_name]["holdout_sgm10_s_mean"]
        cellres["conventional_bar"] = {"name": bar_name,
                                       "holdout_sgm10_s": bar}

        # -- per-seed genuine-mix champions --------------------------------
        per_seed_verdicts = []
        for s in seeds:
            path = os.path.join(OUT_DIR, f"{cell}_evolve_seed{s}.json")
            with open(path) as f:
                run = json.load(f)
            beste = _best_from_logs(run["evaluated"], "genuine_mix", cell)
            if beste is None:
                per_seed_verdicts.append({
                    "seed": s, "found_genuine_mix_candidate": False})
                cellres["mix_champions"][str(s)] = None
                continue
            g = genome_from_dict(json.loads(beste["genome"]))
            ev = _holdout_eval(g, holdout, cfg, repeats)
            ev["gate"] = _gate_record(g, kind)
            ev["train_fitness"] = beste["fitness"]
            ev["phenotype"] = phenotype_mixing(
                cell, json.loads(beste["genome"]), ev["rows"])
            m = ev["holdout_sgm10_s_mean"]
            beats = (math.isfinite(m) and m < bar * 0.90
                     and ev["phenotype"]["genuine"])
            per_seed_verdicts.append({
                "seed": s,
                "found_genuine_mix_candidate": True,
                "holdout_sgm10_s_mean": m,
                "margin_vs_bar": (None if not math.isfinite(m)
                                  else (bar - m) / bar),
                "phenotype_genuine": ev["phenotype"]["genuine"],
                "beats_bar_by_10pct": bool(beats),
            })
            cellres["mix_champions"][str(s)] = ev
            print(f"    mix champion seed {s}: holdout "
                  f"{m:.4g} vs bar {bar:.4g} "
                  f"genuine={ev['phenotype']['genuine']}")

        wins = [v.get("beats_bar_by_10pct", False) for v in per_seed_verdicts]
        all_found = all(v.get("found_genuine_mix_candidate")
                        for v in per_seed_verdicts)
        if all_found and all(wins):
            verdict = "WIN: genuine mix beats the conventional bar by >10% in all seeds"
        else:
            # distinguish tie vs loss on the best mix champion
            ms = [v.get("holdout_sgm10_s_mean") for v in per_seed_verdicts
                  if v.get("holdout_sgm10_s_mean") is not None
                  and math.isfinite(v.get("holdout_sgm10_s_mean"))]
            if not ms:
                verdict = "NO GENUINE MIX CANDIDATE with finite held-out time"
            else:
                best_m = min(ms)
                if best_m < bar * 0.9:
                    verdict = ("NO WIN: at least one seed's mix champion "
                               "fails the >10% margin (not seed-consistent)")
                elif best_m <= bar * 1.1:
                    verdict = "TIE: best mix within 10% of the conventional bar"
                else:
                    verdict = "LOSS: mixes do not beat tuned single schemes"
        cellres["per_seed_verdicts"] = per_seed_verdicts
        cellres["verdict"] = verdict
        print(f"[finalize:{cell}] VERDICT: {verdict}")
        result["cells"][cell] = cellres

    def _ser(o):
        if isinstance(o, Genome):
            return json.loads(canonical_json(o))
        raise TypeError(str(type(o)))

    with open(FINAL_JSON, "w") as f:
        json.dump(sanitize_json(json.loads(json.dumps(result, default=_ser))),
                  f, indent=1)
    print(f"wrote {FINAL_JSON}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,
                    choices=("evolve", "tune-lp", "finalize"))
    ap.add_argument("--cell", choices=("tv", "lp"))
    ap.add_argument("--evolve-seed", type=int, default=42)
    ap.add_argument("--generations", type=int, default=14)
    ap.add_argument("--pop-size", type=int, default=10)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--llm-url", default="http://localhost:8000/v1")
    ap.add_argument("--tune-seed", type=int, default=91)
    ap.add_argument("--tune-iters", type=int, default=60)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    args = ap.parse_args()
    if args.mode == "evolve":
        if not args.cell:
            ap.error("--cell required for evolve")
        mode_evolve(args)
    elif args.mode == "tune-lp":
        mode_tune_lp(args)
    else:
        mode_finalize(args)


if __name__ == "__main__":
    main()
