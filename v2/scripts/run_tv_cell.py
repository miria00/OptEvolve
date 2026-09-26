#!/usr/bin/env python
"""End-to-end TV-denoising demo on real Kermany OCT patches.

Pipeline (plan P0 slice, papers/optimization_atlas_v2_plan.md):
  (a) build N train + M held-out instances from REAL clinical OCT B-scans
      (the raw patch IS y; the speckle is real measurement noise),
  (b) compute/load high-accuracy oracle references (cached),
  (c) BASELINE: the 4 base schemes with default typechecked params,
  (d) EVOLUTION at identical evaluation budget, twice:
      LLM-guided (live vLLM Qwen2.5-Coder-1.5B) vs random search (control),
      fitness = -SGM-10 of time-to-rel_gap<=tol across the train instances,
  (e) held-out evaluation of the three champions + results files:
      results/baseline_card.json, results/evolution_history.json,
      results/comparison.md; figures go to assets/evolution_curves.png.

Budget note: iteration caps are per scheme so every solve gets a roughly
equal WALL-TIME budget (a DRS iteration carries an inner CG solve and is
~10x the cost of a PDHG/FBS iteration). All defaults converge well within
their caps. The genome space now includes avg blends, safeguarded switch
genomes and REAL fixed-k Halpern-anchor restarts (all consumed by
atlas.schemes.compile.solve_genome). Only `alpha` remains a reserved
inert gene; the phenotype cache below dedups genomes that differ only in
alpha.

Run from V2 root:
  /home/miria/jaxenv/bin/python scripts/run_tv_cell.py [--workers N]
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import argparse
import dataclasses
import json
import math
import time

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np

from atlas.bench.kermany import load_patches
from atlas.bench.metrics import sgm, time_to_tol
from atlas.bench.runner import run_cell, sanitize_json, save_card
from atlas.bench.tv_cell import oracle
from atlas.evolve.llm_backend import LLMBackend
from atlas.evolve.loop import evolve
from atlas.evolve.random_backend import RandomBackend
from atlas.genome import Genome, canonical_dict, canonical_json
from atlas.ir.spec import tv_denoise_problem
from atlas.schemes.base_schemes import (
    CondatVuParams,
    DRSParams,
    FBSParams,
    PDHGParams,
)
from atlas.schemes.compile import genome_cap_scheme, solve_genome
from atlas.typing_ import TypeCheckError
from atlas.wrappers.km import Budget

V2_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(V2_ROOT, "results")
# Repo convention: results/ holds data (JSON cards, history, comparison.md);
# assets/ holds ALL plotting code and generated figures.
ASSETS_DIR = os.path.join(V2_ROOT, "assets")

# Per-scheme iteration caps: roughly equal wall-time budget per solve
# (~5-8 s worst case on a 128x128 patch, CPU f64). DRS iters carry an
# inner CG solve (~10x cost), hence the smaller cap. Mix genomes are
# budgeted under the slowest scheme they contain (genome_cap_scheme).
ITER_CAPS = {"fbs": 8000, "drs": 1200, "pdhg": 8000, "condat_vu": 8000,
             "dys": 8000}
SAMPLE_EVERY = 50

# Default typechecked params (textbook choices; every side condition holds):
#   fbs  : dual smoothness L = ||D||^2 = 8, step 1/L = 0.125 in (0, 0.25)
#   drs  : step 1.0 (any a > 0)
#   pdhg : t*s*||D||^2 = 0.99 < 1
#   cv   : t*L_f/2 + t*s*||D||^2 = 0.9925 < 1
DEFAULTS = {
    "fbs": FBSParams(step=0.125),
    "drs": DRSParams(step=1.0),
    "pdhg": PDHGParams(t=0.5, s=0.99 / 4.0),
    "condat_vu": CondatVuParams(t=0.5, s=0.99 * 0.1875),
}


def phenotype_key(g: Genome) -> str:
    """Canonical json with the (still-inert) alpha gene nulled: genomes
    differing only in alpha share one evaluation."""
    d = canonical_dict(g)
    d["params"]["alpha"] = None
    return json.dumps(d, sort_keys=True, separators=(",", ":"))


def eval_genome(g: Genome, instances, tol: float) -> dict:
    """Run one genome over instances via solve_genome (handles base
    schemes, avg blends, switch genomes, halpern restarts); STRICT SGM
    summary (a solve that never reaches tol scores inf)."""
    cap = ITER_CAPS[genome_cap_scheme(g)]
    budget = Budget(max_iters=cap, tol=tol, sample_every=SAMPLE_EVERY)
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
        rel_gap_achieved = float(result.rel_gap_history[-1])
        converged = rel_gap_achieved <= tol
        t_tol, iter_at = time_to_tol(result, tol, sample_every=SAMPLE_EVERY)
        rows.append({
            "instance": ii,
            "iters": int(result.iters),
            "wall_time_s": float(result.wall_time),
            "rel_gap_achieved": rel_gap_achieved,
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


def eval_params(scheme: str, params, instances, tol: float) -> dict:
    """Run one (scheme, params) phenotype over instances; compact summary."""
    card = run_cell(
        [(scheme, params)],
        instances,
        tol=tol,
        max_iters=ITER_CAPS[scheme],
        sample_every=SAMPLE_EVERY,
    )
    agg = card["aggregates"][0]
    # STRICT times: a solve that never reaches tol has time-to-tol = inf
    # (poisons the SGM). run_cell's PAR-1 aggregate (failures capped at
    # budgeted wall time) is kept alongside as auxiliary data; fitness
    # must NOT use it - under tight iteration caps PAR-1 makes a fast
    # FAILURE look cheaper than a slow success.
    strict_times = [
        r["time_to_tol_s"] if r.get("converged") else float("inf")
        for r in card["rows"]
    ]
    return {
        "scheme": scheme,
        "params": dataclasses.asdict(params),
        "sgm10_time_s": sgm(strict_times),
        "sgm10_time_par1_s": agg["sgm10_time_s"],
        "sgm10_iters": agg["sgm10_iters"],
        "failure_rate": agg["failure_rate"],
        "rows": card["rows"],
    }


def _genome_fitness(g: Genome, instances, tol: float):
    ev = eval_genome(g, instances, tol)
    fit = -ev["sgm10_time_s"] if math.isfinite(ev["sgm10_time_s"]) else -math.inf
    metrics = {
        "sgm10_time_s": round(ev["sgm10_time_s"], 4)
        if math.isfinite(ev["sgm10_time_s"])
        else float("inf"),
        "failure_rate": ev["failure_rate"],
        "sgm10_iters": round(ev["sgm10_iters"], 1),
    }
    return fit, metrics


def make_fitness_fn(train_instances, tol: float, log_prefix: str):
    """fitness = -SGM-10 time-to-tol over train instances (higher better).

    Phenotype-level cache: genomes differing only in the inert alpha gene
    share one evaluation.
    """
    cache: dict = {}
    counter = {"n": 0}

    def fitness_fn(g: Genome):
        key = phenotype_key(g)
        if key not in cache:
            fit, metrics = _genome_fitness(g, train_instances, tol)
            cache[key] = (fit, metrics)
            counter["n"] += 1
            print(
                f"    [{log_prefix} eval {counter['n']:3d}] "
                f"sgm10={metrics['sgm10_time_s']:>8} "
                f"fail={metrics['failure_rate']:.2f} {fmt_genome(canonical_json(g))}"
            )
        return cache[key]

    fitness_fn.cache = cache
    return fitness_fn


# -- process-parallel fitness (speed lever; --workers N) --------------------
# Spawn-safe: the initializer rebuilds the train problems in each worker
# from the raw patch arrays; the worker function is module-level so it
# pickles by name (the __main__ guard keeps re-import side-effect free).

_POOL_CTX: dict = {}


def _tv_worker_init(patches, lam, tol):
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    import jax as _jax

    _jax.config.update("jax_enable_x64", True)
    _POOL_CTX["instances"] = [tv_denoise_problem(p, lam=lam) for p in patches]
    _POOL_CTX["tol"] = tol


def _tv_fitness_worker(g: Genome):
    return _genome_fitness(g, _POOL_CTX["instances"], _POOL_CTX["tol"])


def genome_dict_from_canonical(cj: str | None) -> dict | None:
    return None if cj is None else json.loads(cj)


def _fmt_sub(s: dict | None) -> str:
    if not s:
        return "?"
    p = s.get("params", {})
    out = s["scheme"] + f"(tau={p.get('tau'):.4g}"
    if p.get("sigma") is not None:
        out += f",sigma={p['sigma']:.4g}"
    return out + ")"


def fmt_genome(cj: str | None) -> str:
    """Render EVERY non-null gene of the canonical genome, including mix
    blocks. The only remaining inert gene is alpha (trailing *)."""
    if cj is None:
        return "-"
    d = json.loads(cj)
    p = d["params"]
    w = d["wrapper"]
    parts = [d["scheme"]]
    m = d.get("mix")
    if m:
        if m["kind"] == "avg":
            parts.append(
                f"avg[l={m['lambda']:.3g}] {_fmt_sub(m['scheme_a'])}+"
                f"{_fmt_sub(m['scheme_b'])}"
            )
        else:
            parts.append(
                f"switch[w={m['window']}] agg={_fmt_sub(m['aggressive'])} "
                f"fb={_fmt_sub(m['fallback'])}"
            )
    for k in ("tau", "sigma", "rho"):
        if p.get(k) is not None:
            parts.append(f"{k}={p[k]:.4g}")
    if p.get("alpha") is not None:
        parts.append(f"alpha={p['alpha']:.4g}*")  # * = inert (reserved) gene
    if w["halpern"]:
        parts.append("halpern")
    if w.get("restart", "none") != "none":
        parts.append(f"restart={w['restart']}:{w.get('restart_k')}")
    return " ".join(parts)


def fmt_s(v: float) -> str:
    return "inf" if not math.isfinite(v) else f"{v:.3f}"


def plot_curves(hist_llm, hist_rand, default_sgm, default_name, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    SURFACE, GRID = "#fcfcfb", "#e4e3dd"
    TEXT1, TEXT2 = "#262625", "#6b6a63"
    C_LLM, C_RAND = "#2a78d6", "#eb6834"  # categorical slots 1-2, validated

    def series(hist):
        pts = [
            (h["gen"] + 1, -h["best_fitness"])
            for h in hist["generations"]
            if math.isfinite(h["best_fitness"])
        ]
        return [p[0] for p in pts], [p[1] for p in pts]

    gens_l, best_l = series(hist_llm)
    gens_r, best_r = series(hist_rand)

    fig, ax = plt.subplots(figsize=(7.6, 4.4), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    ax.axhline(default_sgm, color=TEXT2, lw=1.4, ls=(0, (4, 3)), zorder=1)
    ax.annotate(
        f"default {default_name}  {default_sgm:.3f}s",
        xy=(0.99, default_sgm),
        xycoords=("axes fraction", "data"),
        ha="right",
        va="bottom",
        fontsize=8.5,
        color=TEXT2,
    )
    ax.plot(gens_l, best_l, color=C_LLM, lw=2, marker="o", ms=4.5, zorder=3,
            label="LLM-guided (Qwen2.5-Coder-1.5B)")
    ax.plot(gens_r, best_r, color=C_RAND, lw=2, marker="o", ms=4.5, zorder=3,
            label="random search (control)")
    for gens, best, c, dy in ((gens_l, best_l, C_LLM, -9), (gens_r, best_r, C_RAND, 9)):
        if not gens:
            continue
        ax.annotate(
            f"{best[-1]:.3f}s",
            xy=(gens[-1], best[-1]),
            xytext=(4, dy),
            textcoords="offset points",
            fontsize=8.5,
            color=TEXT1,
            va="center",
        )

    ax.set_xlabel("generation", color=TEXT1, fontsize=10)
    ax.set_ylabel("best SGM-10 time-to-tol (s)  =  −fitness", color=TEXT1, fontsize=10)
    ax.set_title(
        "Evolved solver configs on real OCT TV denoising "
        "(train SGM, rel gap ≤ 1e-4)",
        color=TEXT1,
        fontsize=11,
        pad=10,
    )
    ax.set_xticks(range(1, len(hist_llm["generations"]) + 1))
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.tick_params(colors=TEXT2, labelsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    leg = ax.legend(loc="upper right", frameon=False, fontsize=9)
    for t in leg.get_texts():
        t.set_color(TEXT1)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-train", type=int, default=6)
    ap.add_argument("--n-holdout", type=int, default=3)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--lam", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--generations", type=int, default=8)
    ap.add_argument("--pop-size", type=int, default=6)
    ap.add_argument("--evolve-seed", type=int, default=42)
    ap.add_argument("--oracle-tol", type=float, default=1e-9)
    ap.add_argument("--skip-llm", action="store_true")
    ap.add_argument("--hardware-profile", default=None,
                    help="operator profile JSON produced by hardware_campaign.py")
    ap.add_argument("--hardware-context", choices=("none", "identity", "measured"),
                    default="measured")
    ap.add_argument("--workers", type=int, default=0,
                    help="process-parallel fitness evaluation (0 = serial)")
    args = ap.parse_args()

    t_start = time.perf_counter()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ---- (a) instances: one draw of n_train + n_holdout DISTINCT scans ----
    n_total = args.n_train + args.n_holdout
    print(f"[1/6] loading {n_total} real OCT patches ({args.size}x{args.size}) ...")
    patches, meta = load_patches(
        n_total, size=args.size, split="test", seed=args.seed, return_meta=True
    )
    problems = [tv_denoise_problem(p, lam=args.lam) for p in patches]
    train, holdout = problems[: args.n_train], problems[args.n_train :]
    print(f"      labels: {[m['label_name'] for m in meta]}")

    # ---- (b) oracle references (cached) ----
    print(f"[2/6] oracle references (rel gap <= {args.oracle_tol:g}, cached) ...")
    oracle_summary = []
    for i, pr in enumerate(problems):
        o = oracle(pr, tol=args.oracle_tol)
        oracle_summary.append(
            {
                "instance": i,
                "split": "train" if i < args.n_train else "holdout",
                "label": meta[i]["label_name"],
                "P_star": o["P_star"],
                "rel_gap": o["rel_gap"],
                "iters": o["iters"],
                "cache_hit": o["cache_hit"],
                "key": o["key"],
            }
        )
        print(
            f"      inst {i} [{meta[i]['label_name']:6s}] P*={o['P_star']:.6f} "
            f"gap={o['rel_gap']:.1e} iters={o['iters']} cache_hit={o['cache_hit']}"
        )

    # ---- (c) baseline: 4 defaults on train ----
    print("[3/6] baseline: 4 default schemes on train ...")
    baseline = {}
    for scheme, params in DEFAULTS.items():
        ev = eval_params(scheme, params, train, args.tol)
        baseline[scheme] = ev
        print(
            f"      {scheme:9s} sgm10={fmt_s(ev['sgm10_time_s'])}s "
            f"iters(sgm)={ev['sgm10_iters']:.0f} fail={ev['failure_rate']:.2f}"
        )
    best_default = min(baseline, key=lambda s: baseline[s]["sgm10_time_s"])

    baseline_card = {
        "cell": "tv_oct_kermany2017",
        "description": "Baseline: 4 default typechecked schemes on real OCT TV denoising",
        "tolerance_rel_gap": args.tol,
        "iter_caps": ITER_CAPS,
        "sample_every": SAMPLE_EVERY,
        "instances": {
            "n_train": args.n_train,
            "n_holdout": args.n_holdout,
            "size": args.size,
            "lam": args.lam,
            "seed": args.seed,
            "labels": [m["label_name"] for m in meta],
        },
        "oracle": {"tol": args.oracle_tol, "references": oracle_summary},
        "schemes": {
            s: {k: ev[k] for k in ("params", "sgm10_time_s", "sgm10_time_par1_s",
                                   "sgm10_iters", "failure_rate", "rows")}
            for s, ev in baseline.items()
        },
        "best_default": best_default,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    save_card(baseline_card, os.path.join(RESULTS_DIR, "baseline_card.json"))

    # ---- (d) evolution: identical budget, LLM vs random control ----
    evo_budget = dict(generations=args.generations, pop_size=args.pop_size)
    runs = {}

    pool = None
    if args.workers > 0:
        # Spawn workers that rebuild the train problems from the raw patch
        # arrays (see _tv_worker_init); genomes and (fit, metrics) tuples
        # pickle across the pool boundary.
        import multiprocessing as mp

        train_patches = [np.asarray(pr.data["y"]) for pr in train]
        pool = mp.get_context("spawn").Pool(
            args.workers,
            initializer=_tv_worker_init,
            initargs=(train_patches, args.lam, args.tol),
        )
        print(f"      [parallel] {args.workers} spawn workers for fitness eval")

    def _evolve_kwargs():
        return dict(pool=pool) if pool is not None else {}

    if not args.skip_llm:
        print(f"[4/6] evolution (LLM-guided, live vLLM) {evo_budget} ...")
        llm_backend = LLMBackend(
            seed=args.evolve_seed,
            fallback=RandomBackend(seed=args.evolve_seed + 100003, p_alpha=0.0),
            hardware=(json.load(open(args.hardware_profile))
                      if args.hardware_profile else None),
            hardware_context=args.hardware_context,
        )
        fit_llm = (
            _tv_fitness_worker if pool is not None
            else make_fitness_fn(train, args.tol, "llm")
        )
        t0 = time.perf_counter()
        hist_llm = evolve(
            llm_backend, fit_llm, seed=args.evolve_seed, **evo_budget,
            **_evolve_kwargs(),
        )
        hist_llm["wall_time_s"] = time.perf_counter() - t0
        hist_llm["n_llm_calls"] = llm_backend.n_llm_calls
        hist_llm["n_llm_fallbacks_total"] = llm_backend.n_fallbacks
        hist_llm["n_stage0_rejections_backend"] = llm_backend.stage0_rejections
        runs["llm"] = hist_llm
        print(
            f"      best fitness {hist_llm['best_fitness']:.4f} "
            f"({-hist_llm['best_fitness']:.3f}s) | llm_calls={llm_backend.n_llm_calls} "
            f"fallbacks={llm_backend.n_fallbacks} "
            f"stage0_rej={llm_backend.stage0_rejections}"
        )

    print(f"[5/6] evolution (random control) {evo_budget} ...")
    rand_backend = RandomBackend(seed=args.evolve_seed, p_alpha=0.0)
    fit_rand = (
        _tv_fitness_worker if pool is not None
        else make_fitness_fn(train, args.tol, "rnd")
    )
    t0 = time.perf_counter()
    hist_rand = evolve(rand_backend, fit_rand, seed=args.evolve_seed, **evo_budget,
                       **_evolve_kwargs())
    hist_rand["wall_time_s"] = time.perf_counter() - t0
    hist_rand["n_stage0_rejections_backend"] = rand_backend.stage0_rejections
    runs["random"] = hist_rand
    print(
        f"      best fitness {hist_rand['best_fitness']:.4f} "
        f"({-hist_rand['best_fitness']:.3f}s) "
        f"stage0_rej={rand_backend.stage0_rejections}"
    )

    # ---- (e) held-out evaluation of the champions ----
    print("[6/6] held-out evaluation (3 unseen patches) ...")
    if pool is not None:
        pool.close()
        pool.join()
    from atlas.genome import genome_from_dict

    heldout = {}
    ev_def = eval_params(best_default, DEFAULTS[best_default], holdout, args.tol)
    heldout["default"] = {
        "label": best_default + " (default)",
        "scheme": best_default,
        "params": dataclasses.asdict(DEFAULTS[best_default]),
        "sgm10_time_s": ev_def["sgm10_time_s"],
        "failure_rate": ev_def["failure_rate"],
        "sgm10_iters": ev_def["sgm10_iters"],
    }
    for name in ("random", "llm"):
        if name not in runs:
            continue
        cj = runs[name].get("best_genome")
        if cj is None:
            continue
        g = genome_from_dict(json.loads(cj))
        ev = eval_genome(g, holdout, args.tol)
        heldout[name] = {
            "label": fmt_genome(cj),
            "genome": json.loads(cj),
            "sgm10_time_s": ev["sgm10_time_s"],
            "failure_rate": ev["failure_rate"],
            "sgm10_iters": ev["sgm10_iters"],
        }
    for name, h in heldout.items():
        print(
            f"      {name:8s} {h['label']:40s} heldout "
            f"sgm10={fmt_s(h['sgm10_time_s'])}s fail={h['failure_rate']:.2f}"
        )

    wall_total = time.perf_counter() - t_start

    # ---- results files ----
    history_out = {
        "budget": {**evo_budget, "tol": args.tol, "iter_caps": ITER_CAPS,
                   "n_train": args.n_train, "n_holdout": args.n_holdout,
                   "evolve_seed": args.evolve_seed},
        "fitness": "-SGM-10(time-to-rel_gap<=tol) over train instances; "
        "PAR-1 cap at budgeted wall time for non-converged solves; "
        "inf (fitness -inf) for Stage-0/typecheck failures",
        "runs": runs,
        "heldout": heldout,
        "baseline_train_sgm10": {s: baseline[s]["sgm10_time_s"] for s in baseline},
        "total_wall_time_s": wall_total,
    }
    hist_path = os.path.join(RESULTS_DIR, "evolution_history.json")
    with open(hist_path, "w") as fh:
        # sanitize_json: strict RFC-8259 output (no bare Infinity/NaN tokens)
        json.dump(sanitize_json(history_out), fh, indent=2, allow_nan=False)
        fh.write("\n")

    # comparison.md
    def fmt_params(scheme: str, params) -> str:
        d = {k: v for k, v in dataclasses.asdict(params).items()
             if v not in (None, False) and not k.startswith("cg_")}
        return scheme + " " + " ".join(f"{k}={v:.4g}" if isinstance(v, float)
                                       else f"{k}={v}" for k, v in d.items())

    rows = [
        (
            f"default ({best_default})",
            baseline[best_default]["sgm10_time_s"],
            fmt_params(best_default, DEFAULTS[best_default]),
            heldout["default"]["sgm10_time_s"],
            heldout["default"]["failure_rate"],
        )
    ]
    for name, label in (("random", "random-evolved"), ("llm", "LLM-evolved")):
        if name in runs:
            rows.append(
                (
                    label,
                    -runs[name]["best_fitness"],
                    fmt_genome(runs[name].get("best_genome")),
                    heldout[name]["sgm10_time_s"] if name in heldout else float("inf"),
                    heldout[name]["failure_rate"] if name in heldout else 1.0,
                )
            )

    lines = [
        "# TV-on-OCT cell: default vs evolved solver configurations",
        "",
        f"Real Kermany2017 OCT B-scan patches ({args.size}x{args.size}, lam={args.lam}); "
        f"tolerance: rigorous relative duality gap <= {args.tol:g}. "
        f"Train = {args.n_train} patches, held-out = {args.n_holdout} unseen patches. "
        f"Evolution budget (identical for both runs): {args.generations} generations x "
        f"{args.pop_size} proposals, seed {args.evolve_seed}.",
        "",
        "| config | train SGM-10 time (s) | best genome | held-out SGM-10 time (s) | held-out failures |",
        "|---|---|---|---|---|",
    ]
    for label, tr, genome, ho, fr in rows:
        lines.append(
            f"| {label} | {fmt_s(tr)} | `{genome}` | {fmt_s(ho)} | {fr:.2f} |"
        )
    lines.append("")
    lines.append("All four baseline schemes (train SGM-10 s): "
                 + ", ".join(f"{s} {fmt_s(baseline[s]['sgm10_time_s'])}"
                             for s in baseline) + ".")
    lines.append("")
    lines.append("The only gene marked `*` (alpha) is reserved/inert; mix, "
                 "switch, restart and rho genes are all consumed by the "
                 "compiled solvers.")
    for name, hist in runs.items():
        if hist.get("flat_fitness_warning"):
            lines.append("")
            lines.append(
                f"**WARNING ({name} run): flat/near-flat fitness** - spread "
                f"{hist.get('fitness_spread', 0.0):.3g} across "
                f"{hist.get('n_unique_evaluated', 0)} unique genomes; the "
                f"ranking above is dominated by timing noise, not solver "
                f"quality."
            )
    comparison_path = os.path.join(RESULTS_DIR, "comparison.md")
    with open(comparison_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")

    # curves (figures go to assets/ per repo convention)
    os.makedirs(ASSETS_DIR, exist_ok=True)
    curves_path = os.path.join(ASSETS_DIR, "evolution_curves.png")
    if "llm" in runs:
        plot_curves(
            runs["llm"],
            hist_rand,
            baseline[best_default]["sgm10_time_s"],
            best_default,
            curves_path,
        )

    print(f"\nTotal wall time: {wall_total:.1f}s")
    print(f"Wrote: {os.path.join(RESULTS_DIR, 'baseline_card.json')}")
    print(f"       {hist_path}")
    print(f"       {comparison_path}")
    print(f"       {curves_path}")


if __name__ == "__main__":
    main()
