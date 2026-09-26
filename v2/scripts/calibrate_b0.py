"""Calibrate B0 on the frozen calibration set, then pick the cap.

B0 = Ruiz(10) + PC(best alpha) + best static PDHG. The substrate is applied
IDENTICALLY here and to every later configuration, so no comparison can be
won by rediscovering numerical hygiene.

PARAMETERIZATION. Under the PC substrate the certified bound is 1, so the
gate tau*sigma*||L'||^2 < 1 is tau*sigma < 1/PC_SAFETY^2. We search the PDLP
parameterization directly:

    tau = eta / omega,   sigma = eta * omega,   tau*sigma = eta^2

so eta in (0, 1) is the step scale (the whole admissible range, with the
gate reducing to eta < 1/PC_SAFETY) and omega > 0 is the primal weight.

TWO SEPARATE STAGES, deliberately not fused (fusing them lets you fiddle
until you like the number):
  STAGE A  tune (eta, omega, alpha) under a GENEROUS fixed dev cap.
           Coordinate descent: (eta, omega) at alpha=1, then alpha, then
           refine (eta, omega). Stability is checked by re-scoring the top
           configs on bootstrap resamples of the calibration set.
  STAGE B  with B0 FROZEN, choose the smallest cap from the PREDECLARED grid
           for which B0 solves 60-80% of calibration at 1e-6. If no grid
           value lands in the band, that is a clean "cell calibration
           failed", not a new knob.

Also reports solve rate by nnz tertile (a global cap that is really a size
filter shows up as 100/70/0) and records preprocessing time separately.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

# DEVICE. atlas/__init__.py pins JAX_PLATFORMS=cpu (setdefault) to reserve the
# GPU for the co-resident vLLM server. Calibration uses no LLM, and the CPU
# path is ~7x slower at 94k nnz and ~20x at 313k nnz (measured), so default to
# the GPU here unless the caller says otherwise. Must run BEFORE atlas is
# imported, because setdefault only yields to an already-set value.
os.environ.setdefault("JAX_PLATFORMS", "cuda")
# The LP cell target is DECLARED hardware. 4090 hosts the vLLM proposer
# (15.3 GB resident) and a full-budget run OOMed in its ~9 GB remainder;
# solver evaluation belongs on the uncontended 5090.
TARGET_GPU_SUBSTRING = os.environ.get("ATLAS_TARGET_GPU", "5090")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from atlas.bench import mps_cell  # noqa: E402
from atlas.genome import genome_from_dict  # noqa: E402
from atlas.schemes.compile import solve_genome  # noqa: E402
from atlas.schemes.rescale import PC_SAFETY, lp_substrate_metric  # noqa: E402
from atlas.typing_ import TypeCheckError  # noqa: E402
from atlas.wrappers.km import Budget  # noqa: E402

SPLIT = os.path.join(ROOT, "results", "sprint", "phase3", "cell_split.json")
OUT = os.path.join(ROOT, "results", "sprint", "phase3", "b0_calibration.json")
CAP_GRID_S = [5.0, 10.0, 20.0, 40.0, 80.0]      # PREDECLARED
TARGET_BAND = (0.60, 0.80)                       # PREDECLARED
ETAS = [0.5, 0.7, 0.9, 0.99]
OMEGAS = [0.25, 0.5, 1.0, 2.0, 4.0]
ALPHAS = [0.0, 0.5, 1.0, 1.5, 2.0]
SGM_SHIFT = 10.0


def sgm(vals, shift=SGM_SHIFT):
    v = np.asarray([x for x in vals if np.isfinite(x)], dtype=float)
    if v.size == 0:
        return float("inf")
    return float(np.exp(np.mean(np.log(v + shift))) - shift)


def genome(eta, omega, alpha):
    return genome_from_dict({
        "scheme": "pdhg",
        "params": {"tau": eta / omega, "sigma": eta * omega},
        "metric": lp_substrate_metric(pc_alpha=alpha),
    })


def load_all(rows, verbose=True):
    probs, prep = {}, {}
    for e in rows:
        t0 = time.time()
        try:
            std = mps_cell.load_standard_form(
                os.path.join(ROOT, "data", "lp_suites", e["file"]))
            probs[e["name"]] = mps_cell.to_composite(std)
            prep[e["name"]] = time.time() - t0
        except Exception as ex:  # noqa: BLE001
            if verbose:
                print(f"  ! load {e['name']}: {type(ex).__name__}", flush=True)
    return probs, prep


def evaluate(probs, eta, omega, alpha, tol, max_iters, cap_s):
    """-> (n_solved, sgm_time, rows). Unsolved instances take the CAP, never
    -inf: a flat landscape is the failure mode this design exists to avoid."""
    g = genome(eta, omega, alpha)
    rows, n_ok = [], 0
    for name, prob in probs.items():
        t0 = time.time()
        try:
            r = solve_genome(prob, g, Budget(max_iters=max_iters, tol=tol,
                                             sample_every=250))
            el = time.time() - t0
            h = r.rel_gap_history[np.isfinite(r.rel_gap_history)]
            gap = float(h[-1]) if h.size else float("nan")
            ok = bool(gap <= tol) and el <= cap_s
            n_ok += ok
            rows.append({"name": name, "ok": ok, "iters": int(r.iters),
                         "wall": el, "gap": gap})
        except TypeCheckError as ex:
            rows.append({"name": name, "ok": False, "rejected": str(ex)[:120]})
    times = [r["wall"] if r.get("ok") else cap_s for r in rows]
    censored = [r["gap"] for r in rows
                if not r.get("ok") and np.isfinite(r.get("gap", np.nan))]
    med_gap = float(np.median(censored)) if censored else None
    for r in rows:
        r["median_terminal_gap_of_config"] = med_gap
    return n_ok, sgm(times), rows


def better(a, b):
    """Lexicographic: more certified solves first, then lower SGM time."""
    return (a[0] > b[0]) or (a[0] == b[0] and a[1] < b[1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--screen-max-iters", type=int, default=20_000,
                    help="budget for RANKING configs in stages A1-A3")
    ap.add_argument("--dev-max-iters", type=int, default=200_000,
                    help="budget for characterizing the FINALISTS and B0")
    ap.add_argument("--n-finalists", type=int, default=3)
    ap.add_argument("--dev-cap-s", type=float, default=120.0)
    ap.add_argument("--bootstrap", type=int, default=200)
    args = ap.parse_args()

    import subprocess

    import jax
    import jaxlib

    devs = jax.devices()
    try:
        gpu_name = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True).strip().splitlines()[0]
    except Exception:  # noqa: BLE001
        gpu_name = "unknown"
    env = {"devices": [str(d) for d in devs], "gpu_name": gpu_name,
           "jax": jax.__version__, "jaxlib": jaxlib.__version__,
           "jax_platforms": os.environ.get("JAX_PLATFORMS"),
           "target_gpu_substring": TARGET_GPU_SUBSTRING,
           "iterate_precision": "float64",
           "certificate_precision": "float64",
           "host": os.uname().nodename}
    print("ENVIRONMENT", flush=True)
    for k, v in env.items():
        print(f"  {k:26s} {v}", flush=True)
    if TARGET_GPU_SUBSTRING and TARGET_GPU_SUBSTRING not in gpu_name:
        sys.exit(f"\nFATAL: declared target GPU {TARGET_GPU_SUBSTRING!r} not "
                 f"present (found {gpu_name!r}). Timings from a different "
                 "device are not comparable; set ATLAS_TARGET_GPU to override.")

    split = json.load(open(SPLIT))
    rows = split["calibration"]["instances"]
    print(f"calibration set n={len(rows)} sha={split['calibration']['sha256'][:16]}",
          flush=True)
    probs, prep = load_all(rows)
    # Warm the kernel cache so no first-JIT compile lands inside a timed solve.
    print("warming JIT for each instance", flush=True)
    _wg = genome(0.9, 1.0, 1.0)
    t_warm = time.time()
    for _n, _p in probs.items():
        try:
            solve_genome(_p, _wg, Budget(max_iters=2, tol=0.0, sample_every=1))
        except Exception:  # noqa: BLE001
            pass
    print(f"  warm-up {time.time() - t_warm:.1f}s", flush=True)
    print(f"loaded {len(probs)}; preprocessing total {sum(prep.values()):.1f}s "
          f"(median {np.median(list(prep.values())):.2f}s)", flush=True)
    print(f"gate under PC substrate: eta < {1.0 / PC_SAFETY:.9f}\n", flush=True)

    result = {"config": vars(args), "environment": env,
              "preprocess_s": prep, "stages": {}}
    t_start = time.time()

    # ---- STAGE A -----------------------------------------------------
    print("STAGE A1: (eta, omega) at alpha=1", flush=True)
    best, best_key = None, (-1, float("inf"))
    a1 = []
    for eta in ETAS:
        for om in OMEGAS:
            n, s, _rw = evaluate(probs, eta, om, 1.0, args.tol,
                                 args.screen_max_iters, args.dev_cap_s)
            a1.append({"eta": eta, "omega": om, "alpha": 1.0,
                       "n_solved": n, "sgm_s": s,
                       "median_terminal_gap": _rw[0].get(
                           "median_terminal_gap_of_config") if _rw else None})
            print(f"  eta={eta:<5} omega={om:<5} solved {n:>2}/{len(probs)} "
                  f"sgm={s:8.3f}s", flush=True)
            if better((n, s), best_key):
                best_key, best = (n, s), (eta, om, 1.0)
    result["stages"]["A1"] = a1
    print(f"  -> best {best}  {best_key}\n", flush=True)

    print("STAGE A2: alpha sweep at the best (eta, omega)", flush=True)
    a2 = []
    for al in ALPHAS:
        n, s, _rw = evaluate(probs, best[0], best[1], al, args.tol,
                             args.screen_max_iters, args.dev_cap_s)
        a2.append({"eta": best[0], "omega": best[1], "alpha": al,
                   "n_solved": n, "sgm_s": s,
                   "median_terminal_gap": _rw[0].get(
                       "median_terminal_gap_of_config") if _rw else None})
        print(f"  alpha={al:<5} solved {n:>2}/{len(probs)} sgm={s:8.3f}s",
              flush=True)
        if better((n, s), best_key):
            best_key, best = (n, s), (best[0], best[1], al)
    result["stages"]["A2"] = a2
    print(f"  -> best {best}  {best_key}\n", flush=True)

    print("STAGE A3: refine (eta, omega) at the best alpha", flush=True)
    a3 = []
    for eta in ETAS:
        for om in OMEGAS:
            if (eta, om) == (best[0], best[1]):
                continue
            n, s, _rw = evaluate(probs, eta, om, best[2], args.tol,
                                 args.screen_max_iters, args.dev_cap_s)
            a3.append({"eta": eta, "omega": om, "alpha": best[2],
                       "n_solved": n, "sgm_s": s,
                       "median_terminal_gap": _rw[0].get(
                           "median_terminal_gap_of_config") if _rw else None})
            if better((n, s), best_key):
                best_key, best = (n, s), (eta, om, best[2])
    result["stages"]["A3"] = a3

    # SCREEN-THEN-CONFIRM. Stages A1-A3 rank on a truncated budget, which
    # could favour a config that starts fast and stalls. So take the top
    # finalists by screening score and re-rank them at the FULL dev budget
    # before freezing anything.
    # FINALIST RULE, fixed before any full-budget outcome is observed.
    # Screening on 20k iters would discard a config that is mediocre early
    # and excellent late, which is EXACTLY the behaviour hypothesised to
    # matter at 1e-6. So the finalists are:
    #   (a) top n by screening lexicographic score,
    #   (b) the best 2 by MEDIAN TERMINAL GAP among configs that solved
    #       fewest, i.e. best late progress despite poor solve count,
    #   (c) the most aggressive admissible endpoint (max eta).
    pool = a1 + a2 + a3
    pool.sort(key=lambda r: (-r["n_solved"], r["sgm_s"]))
    seen, finalists = set(), []

    def _add(k):
        if k not in seen:
            seen.add(k)
            finalists.append(k)

    for r in pool[:args.n_finalists]:
        _add((r["eta"], r["omega"], r["alpha"]))
    by_gap = [r for r in pool if r.get("median_terminal_gap") is not None]
    by_gap.sort(key=lambda r: r["median_terminal_gap"])
    for r in by_gap[:2]:
        _add((r["eta"], r["omega"], r["alpha"]))
    aggressive = max(pool, key=lambda r: r["eta"])
    _add((aggressive["eta"], aggressive["omega"], aggressive["alpha"]))
    print(f"STAGE A3b: re-ranking {len(finalists)} finalists at the FULL "
          f"budget ({args.dev_max_iters} iters)", flush=True)
    best_key = (-1, float("inf"))
    a3b = []
    for (eta, om, al) in finalists:
        n, s, _ = evaluate(probs, eta, om, al, args.tol,
                           args.dev_max_iters, args.dev_cap_s)
        a3b.append({"eta": eta, "omega": om, "alpha": al,
                    "n_solved": n, "sgm_s": s})
        print(f"  eta={eta:<5} omega={om:<5} alpha={al:<4} solved {n:>2}/"
              f"{len(probs)} sgm={s:8.3f}s", flush=True)
        if better((n, s), best_key):
            best_key, best = (n, s), (eta, om, al)
    result["stages"]["A3b_full_budget"] = a3b
    eta0, om0, al0 = best
    print(f"  -> B0 FROZEN: eta={eta0} omega={om0} alpha={al0} "
          f"(tau={eta0/om0:.6g}, sigma={eta0*om0:.6g})  {best_key}\n", flush=True)
    result["B0"] = {"eta": eta0, "omega": om0, "alpha": al0,
                    "tau": eta0 / om0, "sigma": eta0 * om0,
                    "ruiz_iters": 10, "dev_n_solved": best_key[0],
                    "dev_sgm_s": best_key[1]}

    # ---- stability ----------------------------------------------------
    print("STAGE A4: bootstrap stability of the frozen B0", flush=True)
    n_b0, _s, rows_b0 = evaluate(probs, eta0, om0, al0, args.tol,
                                 args.dev_max_iters, args.dev_cap_s)
    oks = np.array([1.0 if r.get("ok") else 0.0 for r in rows_b0])
    rng = np.random.default_rng(0)
    boot = [float(np.mean(rng.choice(oks, size=oks.size, replace=True)))
            for _ in range(args.bootstrap)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"  dev solve rate {oks.mean():.1%}  bootstrap 95% [{lo:.1%}, {hi:.1%}]",
          flush=True)
    result["B0"]["bootstrap_solve_rate_ci95"] = [float(lo), float(hi)]
    result["B0"]["dev_rows"] = rows_b0

    # ---- STAGE B: cap selection from the predeclared grid --------------
    print("\nSTAGE B: cap selection from the PREDECLARED grid", flush=True)
    capsel, chosen = [], None
    for cap in CAP_GRID_S:
        n_ok = sum(1 for r in rows_b0
                   if r.get("ok") and r.get("wall", 1e9) <= cap)
        rate = n_ok / max(1, len(rows_b0))
        capsel.append({"cap_s": cap, "n_solved": n_ok, "rate": rate})
        inband = TARGET_BAND[0] <= rate <= TARGET_BAND[1]
        print(f"  cap {cap:>5.0f}s  solved {n_ok:>2}/{len(rows_b0)}  "
              f"{rate:5.1%} {'  <== IN BAND' if inband else ''}", flush=True)
        if inband and chosen is None:
            chosen = cap
    result["cap_selection"] = {"grid": CAP_GRID_S, "band": list(TARGET_BAND),
                              "rows": capsel, "chosen_cap_s": chosen}
    if chosen is None:
        print("\n  CELL CALIBRATION FAILED: no predeclared cap lands in "
              f"{TARGET_BAND[0]:.0%}-{TARGET_BAND[1]:.0%}.", flush=True)
    else:
        print(f"\n  CHOSEN CAP: {chosen:.0f}s", flush=True)

    # ---- parameter robustness: is B0 on a knife edge? ------------------
    print("\nSTAGE C: parameter robustness around the frozen B0", flush=True)
    robust = []
    for lab, (e_, o_, a_) in (
        ("eta x0.95", (eta0 * 0.95, om0, al0)),
        ("eta*      ", (eta0, om0, al0)),
        ("eta x1.05", (min(eta0 * 1.05, 1.0 / PC_SAFETY * 0.999999), om0, al0)),
        ("omega x0.5", (eta0, om0 * 0.5, al0)),
        ("omega x2  ", (eta0, om0 * 2.0, al0)),
    ):
        n, s, _ = evaluate(probs, e_, o_, a_, args.tol,
                           args.dev_max_iters, args.dev_cap_s)
        robust.append({"label": lab.strip(), "eta": e_, "omega": o_,
                       "alpha": a_, "n_solved": n, "sgm_s": s})
        print(f"  {lab}  eta={e_:.4f} omega={o_:.4f}  solved {n:>2}/"
              f"{len(probs)}  sgm={s:8.3f}s", flush=True)
    result["parameter_robustness"] = robust

    # ---- size tertiles -------------------------------------------------
    nnz = {e["name"]: e["nnz"] for e in rows}
    order = sorted((r for r in rows_b0 if "name" in r),
                   key=lambda r: nnz.get(r["name"], 0))
    k = max(1, len(order) // 3)
    tert = {}
    for i, lab in enumerate(("small", "mid", "large")):
        chunk = order[i * k:(i + 1) * k] if lab != "large" else order[2 * k:]
        solved = [r for r in chunk if r.get("ok")]
        cens = [r for r in chunk if not r.get("ok")
                and np.isfinite(r.get("gap", np.nan))]
        rate = len(solved) / max(1, len(chunk))
        med_t = float(np.median([r["wall"] for r in solved])) if solved else None
        med_g = float(np.median([r["gap"] for r in cens])) if cens else None
        tert[lab] = {"n": len(chunk), "solve_rate": rate,
                     "median_time_solved_s": med_t,
                     "median_terminal_gap_censored": med_g,
                     "censored_gaps": sorted(r["gap"] for r in cens)}
        print(f"  tertile {lab:<6} n={len(chunk):>2}  solved {len(solved)}/"
              f"{len(chunk)} ({rate:5.1%})  median t(solved)="
              f"{('%.2fs' % med_t) if med_t is not None else '  n/a':>8}  "
              f"median terminal gap(censored)="
              f"{('%.2e' % med_g) if med_g is not None else 'n/a'}", flush=True)
    result["size_tertiles"] = tert
    if tert["small"]["solve_rate"] > 0.9 and tert["large"]["solve_rate"] < 0.1:
        print("  WARNING: the global cap is acting as a SIZE FILTER; consider "
              "a predeclared size-normalized cap.", flush=True)

    result["wall_total_s"] = time.time() - t_start
    json.dump(result, open(OUT, "w"), indent=1, default=float)
    print(f"\nwrote {OUT}  ({result['wall_total_s']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
