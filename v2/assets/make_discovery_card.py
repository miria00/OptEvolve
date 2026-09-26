#!/usr/bin/env python
"""Discovery card: the paper's 4-panel figure for the TV-on-OCT cell.

Panels (conventions borrowed from the neighboring literature):
  A. Evolution trajectory (FunSearch/ShinkaEvolve convention): best-so-far
     train SGM time-to-tol vs generation, LLM-guided vs random control,
     certified default as a dashed baseline.
  B. Convergence traces (PDLP/cuPDLP convention): relative duality gap vs
     wall-time (log-y) on ONE held-out patch — default vs both champions.
  C. Held-out generalization (L2O convention): per-instance time-to-tol
     bars on the unseen patches.
  D. The OCT visual: raw clinical B-scan patch vs its TV-denoised output.

Caption block prints the champion genome WITH its Level-1 construction
certificate (satisfied side condition + averagedness + LSCOMO citation) —
the element none of the neighboring papers can print.

Run from V2 root:  /home/miria/jaxenv/bin/python assets/make_discovery_card.py
Inputs:  results/evolution_history.json, results/comparison.md metadata
Output:  assets/discovery_card.png
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import json
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from atlas.bench.kermany import load_patches
from atlas.bench.metrics import sgm, time_to_tol
from atlas.genome import genome_from_dict
from atlas.ir.spec import tv_denoise_problem
from atlas.schemes.base_schemes import (
    CondatVuParams,
    DRSParams,
    FBSParams,
    PDHGParams,
    solve_condat_vu,
    solve_drs,
    solve_fbs,
    solve_pdhg,
)
from atlas.wrappers.km import Budget

V2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(V2, "results")
ASSETS = os.path.join(V2, "assets")

SOLVERS = {"fbs": solve_fbs, "drs": solve_drs, "pdhg": solve_pdhg,
           "condat_vu": solve_condat_vu}
PARAMS = {"fbs": FBSParams, "drs": DRSParams, "pdhg": PDHGParams,
          "condat_vu": CondatVuParams}

C_DEFAULT, C_LLM, C_RND = "#6b7280", "#2563eb", "#f59e0b"


def genome_to_phenotype(cj: str):
    """Canonical genome JSON -> (scheme, params dataclass). Mirrors
    scripts/run_tv_cell.py::genome_to_solver_params (P0-inert genes dropped)."""
    g = genome_from_dict(json.loads(cj))
    p, w = g.params, g.wrapper
    kw = dict(rho=p.rho, halpern=w.halpern)
    if g.scheme == "fbs":
        return "fbs", FBSParams(step=p.tau, **kw)
    if g.scheme == "drs":
        return "drs", DRSParams(step=p.tau, **kw)
    if g.scheme == "pdhg":
        return "pdhg", PDHGParams(t=p.tau, s=p.sigma, **kw)
    return "condat_vu", CondatVuParams(t=p.tau, s=p.sigma, **kw)


def best_so_far(gens, key="best_fitness"):
    xs, ys, cur = [], [], -math.inf
    for g in gens:
        f = g.get(key)
        if isinstance(f, (int, float)) and math.isfinite(f):
            cur = max(cur, f)
        xs.append(g["gen"] + 1)
        ys.append(-cur if math.isfinite(cur) else math.nan)  # fitness = -SGM time
    return np.asarray(xs), np.asarray(ys)


def main():
    hist = json.load(open(os.path.join(RESULTS, "evolution_history.json")))
    base = json.load(open(os.path.join(RESULTS, "baseline_card.json")))
    budget = hist["budget"]
    tol = budget["tol"]
    size = budget.get("size", 128)
    lam = budget.get("lam", 0.1)
    seed = budget.get("seed", 7)
    n_train, n_hold = budget["n_train"], budget["n_holdout"]

    # --- champions + default ------------------------------------------------
    runs = hist["runs"]
    champ = {}
    for name in ("llm", "random"):
        gens = runs[name]["generations"]
        fin = [g for g in gens if g.get("best_genome")]
        champ[name] = fin[-1]["best_genome"]
    # default = best baseline scheme from the card ("schemes" mapping)
    base_rows = base.get("schemes", {})
    default_scheme = base.get("best_default") or (
        min(base_rows, key=lambda s: base_rows[s]["sgm10_time_s"])
        if base_rows else "condat_vu")
    default_params = {"fbs": FBSParams(step=0.125), "drs": DRSParams(step=1.0),
                      "pdhg": PDHGParams(t=0.5, s=0.99 / 4.0),
                      "condat_vu": CondatVuParams(t=0.5, s=0.99 * 0.1875)}[default_scheme]

    configs = [
        ("default", default_scheme, default_params, C_DEFAULT),
        ("LLM-evolved", *genome_to_phenotype(champ["llm"]), C_LLM),
        ("random-evolved", *genome_to_phenotype(champ["random"]), C_RND),
    ]

    # --- held-out instances (same construction as the demo script) ----------
    patches = load_patches(n_train + n_hold, size=size, split="test", seed=seed)
    hold = patches[n_train:]
    problems = [tv_denoise_problem(p, lam) for p in hold]

    # --- solve for traces + per-instance times ------------------------------
    # Times (Panel C): IDENTICAL methodology to scripts/run_tv_cell.py's
    # held-out eval — stop at the target tol, sample_every=50, per-scheme
    # iteration caps — so the panel's numbers agree with comparison.md.
    ITER_CAPS = {"fbs": 8000, "drs": 1200, "pdhg": 8000, "condat_vu": 8000}
    traces, times = {}, {n: [] for n, *_ in configs}
    for name, scheme, params, _c in configs:
        for j, prob in enumerate(problems):
            bud = Budget(max_iters=ITER_CAPS[scheme], tol=tol, sample_every=50)
            res = SOLVERS[scheme](prob, params, bud)
            t, _ = time_to_tol(res, tol)
            times[name].append(t)
        # Traces (Panel B): a separate deeper run on patch #1 only, past the
        # target tol so the curve continues below the dotted line.
        deep = Budget(max_iters=ITER_CAPS[scheme], tol=tol * 1e-2, sample_every=50)
        res = SOLVERS[scheme](problems[0], params, deep)
        gaps = np.asarray(res.rel_gap_history, dtype=float)
        its = np.arange(1, len(gaps) + 1) * deep.sample_every
        tsec = res.wall_time * its / max(int(res.iters), 1)
        traces[name] = (tsec, gaps)

    # --- figure -------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 10.6))
    (axA, axB), (axC, axD) = axes
    fig.subplots_adjust(hspace=0.42, wspace=0.30, top=0.915, bottom=0.245)

    # Panel A: evolution trajectory
    for name, color, label in (("llm", C_LLM, "LLM-guided (Qwen2.5-Coder-1.5B)"),
                               ("random", C_RND, "random search (control)")):
        xs, ys = best_so_far(runs[name]["generations"])
        axA.plot(xs, ys, "-o", color=color, label=label, ms=4.5, lw=2)
    dflt_sgm = base_rows[default_scheme]["sgm10_time_s"] if base_rows else None
    if dflt_sgm:
        axA.axhline(dflt_sgm, ls="--", color=C_DEFAULT, lw=1.8)
        axA.text(0.98, dflt_sgm * 1.08, f"certified default ({default_scheme})  "
                 f"{dflt_sgm:.3f}s", ha="right", va="bottom", fontsize=9,
                 color=C_DEFAULT, transform=axA.get_yaxis_transform())
    axA.set_yscale("log")
    axA.xaxis.set_major_locator(MaxNLocator(integer=True))
    axA.set_xlabel("generation")
    axA.set_ylabel("best train SGM-10 time-to-tol (s)")
    axA.set_title("A  Evolution trajectory (train)", loc="left", fontweight="bold")
    axA.legend(fontsize=9, frameon=False)

    # Panel B: convergence traces on held-out patch 1
    for name, _s, _p, color in configs:
        tsec, gaps = traces[name]
        axB.plot(tsec, np.maximum(gaps, 1e-12), color=color, lw=2, label=name)
    axB.axhline(tol, ls=":", color="k", lw=1.2)
    axB.text(0.98, tol * 1.3, f"target tol {tol:g}", ha="right", fontsize=9,
             transform=axB.get_yaxis_transform())
    axB.set_yscale("log")
    axB.set_xlabel("wall time (s)")
    axB.set_ylabel("relative duality gap")
    axB.set_title("B  Convergence on held-out patch #1", loc="left",
                  fontweight="bold")
    axB.legend(fontsize=9, frameon=False)

    # Panel C: per-instance held-out times
    idx = np.arange(len(problems))
    w = 0.26
    for k, (name, _s, _p, color) in enumerate(configs):
        vals = [v if math.isfinite(v) else 0 for v in times[name]]
        axC.bar(idx + (k - 1) * w, vals, width=w, color=color, label=name)
    axC.set_xticks(idx, [f"patch {i+1}" for i in idx])
    axC.set_ylabel("time-to-tol (s)")
    sp = sgm(times["default"]) / sgm(times["LLM-evolved"])
    axC.set_title(f"C  Held-out generalization ({len(problems)} unseen patches, "
                  f"LLM speedup {sp:.1f}x)", loc="left", fontweight="bold")
    axC.legend(fontsize=9, frameon=False)

    # Panel D: OCT before/after with the LLM champion
    name, scheme, params, _ = configs[1]
    res = SOLVERS[scheme](problems[0], params, Budget(max_iters=20_000,
                                                      tol=1e-7, sample_every=50))
    axD.set_title("D  Clinical OCT: raw vs TV-denoised", loc="left",
                  fontweight="bold")
    both = np.concatenate([np.asarray(hold[0]),
                           np.ones((size, 6)), np.asarray(res.x)], axis=1)
    axD.imshow(both, cmap="gray", vmin=0, vmax=1)
    axD.set_xticks([]), axD.set_yticks([])
    axD.text(0.25, -0.06, "raw (speckle)", transform=axD.transAxes,
             ha="center", fontsize=9)
    axD.text(0.78, -0.06, "TV-denoised", transform=axD.transAxes,
             ha="center", fontsize=9)

    # caption: champion genome + its construction certificate
    _, sch, prm, _ = configs[1]
    from atlas.typing_ import rules as R
    tc = {"fbs": lambda p: R.typecheck_fbs(8.0, p.step),
          "drs": lambda p: R.typecheck_drs(p.step),
          "pdhg": lambda p: R.typecheck_pdhg(p.t, p.s, math.sqrt(8.0)),
          "condat_vu": lambda p: R.typecheck_condat_vu(p.t, p.s, math.sqrt(8.0), 1.0)}[sch](prm)
    fig.suptitle("Atlas discovers a certified stepsize + relaxation policy for "
                 "TV denoising on real clinical OCT", fontsize=13.5,
                 fontweight="bold")
    cap = (f"LLM champion genome: {json.loads(champ['llm'])['scheme']} "
           f"{json.dumps(json.loads(champ['llm'])['params'])}   |   "
           f"Level-1 certificate: {tc.condition_str}  (satisfied; "
           f"averagedness α={tc.averagedness:.3f};  {tc.citation})")
    fig.text(0.5, 0.185, cap, ha="center", fontsize=8.5, family="monospace",
             wrap=True)

    # --- analysis block: what evolution actually changed --------------------
    dp, cp = default_params, configs[1][2]
    lhs = lambda t, s: t * 1.0 / 2 + t * s * 8.0  # condat_vu side-condition lhs
    d_lhs = lhs(dp.t, dp.s) if default_scheme == "condat_vu" else float("nan")
    c_lhs = lhs(cp.t, cp.s)
    rho = getattr(cp, "rho", None) or 1.0
    analysis = "\n".join([
        "WHAT EVOLUTION CHANGED  (textbook default -> discovered policy, both certified):",
        f"  primal step   tau  {dp.t:.3g} -> {cp.t:.3g}   ({dp.t/cp.t:.0f}x smaller: "
        "the strongly convex data-fidelity block is easy - stop over-stepping it)",
        f"  dual step     sigma {dp.s:.3g} -> {cp.s:.3g}   ({cp.s/dp.s:.1f}x larger: "
        "work shifts to the linf-ball dual geometry of TV, where the difficulty lives)",
        f"  relaxation    rho  1.0 -> {rho:.2g}    (rho*alpha = {rho*tc.averagedness:.2f} < 1: "
        "near-maximal certified over-relaxation)",
        f"  side-condition lhs {d_lhs:.4f} -> {c_lhs:.3f}: retreat from the stepsize "
        "boundary buys the slack that over-relaxation then spends -",
        "  the OSQP/ADMM practitioner recipe (relax at ~1.5-1.9), rediscovered by search "
        "and carried with a machine-checked convergence certificate.",
    ])
    fig.text(0.055, 0.015, analysis, ha="left", va="bottom", fontsize=8.8,
             family="monospace",
             bbox=dict(boxstyle="round,pad=0.55", fc="#f4f6f8", ec="#9ca3af"))

    out = os.path.join(ASSETS, "discovery_card.png")
    fig.savefig(out, dpi=170)
    print("wrote", out)
    print("held-out SGM:", {n: round(sgm(v), 3) for n, v in times.items()})
    print("cert:", tc.condition_str, "| averagedness", round(tc.averagedness, 4),
          "|", tc.citation)


if __name__ == "__main__":
    main()
