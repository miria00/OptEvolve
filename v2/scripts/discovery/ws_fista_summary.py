"""Grid-level summary of the working-set FISTA ablation, with the cuML reference recomputed UNIFORMLY.

The ablation driver attaches whatever single baseline file it was given; the project standard is the
FASTER cuML precision that actually reaches the target, taking its fastest and median passing run. This
recomputes that for every instance from the baselines directory, so no instance is compared against the
slower precision by accident.

Per instance the contrasts already carry a paired interval over repetitions. Across instances this
reports the geometric mean with a t interval on the per-instance log ratios, which is the right summary
for a claim about a problem CLASS rather than one problem.
"""
import argparse, glob, json, math, os, statistics, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="results/ws_fista_20260913")
ap.add_argument("--baselines", default="results/generalisation_20260911/baselines")
ap.add_argument("--out", default=None)
a = ap.parse_args()


def cuml_reference(iid, bdir):
    """Fastest and median passing run of the faster cuML precision that reaches the target."""
    best = None
    for tag, label in (("cuml", "float32"), ("cuml64", "float64")):
        cs = [json.load(open(f))["cuml"] for f in sorted(glob.glob(os.path.join(bdir, f"{iid}_{tag}_rep*.json")))]
        ok = [c["seconds"] for c in cs if c.get("passed")]
        if ok and (best is None or min(ok) < best["fastest"]):
            best = {"precision": label, "fastest": min(ok), "median": statistics.median(ok),
                    "passing": len(ok), "runs": len(cs)}
    return best


def gm_interval(vals, conf=0.95):
    """Geometric mean of per-instance ratios with a t interval on their logs."""
    if len(vals) < 2:
        return {"n": len(vals), "geo": (vals[0] if vals else float("nan")), "lo": None, "hi": None}
    from scipy import stats
    lg = [math.log(v) for v in vals]
    m, sd, n = statistics.fmean(lg), statistics.stdev(lg), len(lg)
    h = float(stats.t.ppf(0.5 * (1 + conf), n - 1)) * sd / math.sqrt(n)
    return {"n": n, "geo": math.exp(m), "lo": math.exp(m - h), "hi": math.exp(m + h)}


KEYS = ["reduced-set L vs full L (momentum reset)", "reduced-set L vs full L (momentum kept)",
        "momentum kept vs reset (full L)", "momentum kept vs reset (reduced-set L)",
        "both ingredients vs neither", "both ingredients vs plain FISTA"]
rows, per_contrast = [], {k: [] for k in KEYS}
for f in sorted(glob.glob(os.path.join(a.dir, "*.json"))):
    d = json.load(open(f))
    if "arms" not in d or "data" not in d:      # skip our own SUMMARY.json and any non-instance file
        continue
    ds = os.path.basename(d["data"].rstrip("/"))
    iid = f"{ds}_a{d['alpha']}_r{d['lam_frac']}"
    arms = {k: v for k, v in d["arms"].items() if v.get("converged")}
    if not arms:
        continue
    best = min(arms, key=lambda k: arms[k]["median_wall_s"])
    ref = cuml_reference(iid, a.baselines)
    r = {"id": iid, "n": d["n"], "p": d["p"], "alpha": d["alpha"], "lam_frac": d["lam_frac"],
         "best_arm": best, "warm": arms[best]["median_wall_s"],
         # "repeat" = solve + compile with the executable cache WARM, i.e. a steady-state repeated solve.
         # It is NOT a cold solve: since the 2026-09-13 cache fix it collapses onto the solve loop. The
         # genuine cold number is "first", the first solve on an empty cache (absent in pre-fix runs).
         "repeat": arms[best]["median_wall_incl_compile_s"],
         "first": arms[best].get("first_solve_total_s"),
         "evals": arms[best]["evals"], "cuml": ref,
         "contrasts": {k: v["geo_ratio"] for k, v in (d.get("contrasts") or {}).items()},
         "contrast_verdicts": {k: v["verdict"] for k, v in (d.get("contrasts") or {}).items()}}
    if ref:
        r["warm_vs_cuml"] = ref["fastest"] / r["warm"]
        r["warm_vs_cuml_median"] = ref["median"] / r["warm"]
        r["repeat_vs_cuml"] = ref["fastest"] / r["repeat"]
        if r["first"]:
            r["first_vs_cuml"] = ref["fastest"] / r["first"]
    rows.append(r)
    for k in KEYS:
        if k in r["contrasts"]:
            per_contrast[k].append(r["contrasts"][k])

print(f"{'instance':28s} {'best arm':18s} {'loop':>7s} {'repeat':>7s} {'first':>7s} {'cuML':>7s} "
      f"{'prec':>7s} {'loop x':>6s} {'rpt x':>6s} {'1st x':>6s}")
for r in rows:
    c = r.get("cuml") or {}
    fs = r.get("first")
    print(f"{r['id']:28s} {r['best_arm']:18s} {r['warm']:7.4f} {r['repeat']:7.4f} "
          f"{(fs if fs else float('nan')):7.4f} {c.get('fastest', float('nan')):7.4f} "
          f"{c.get('precision', '-'):>7s} {r.get('warm_vs_cuml', float('nan')):6.2f} "
          f"{r.get('repeat_vs_cuml', float('nan')):6.2f} {r.get('first_vs_cuml', float('nan')):6.2f}")

print(f"\nCROSS-INSTANCE CONTRASTS (geometric mean over instances, 95% t interval on the log ratios):")
for k in KEYS:
    v = per_contrast[k]
    if not v:
        continue
    g = gm_interval(v)
    lo = f"[{g['lo']:.3f}, {g['hi']:.3f}]" if g["lo"] else "(single instance)"
    incon = sum(1 for r in rows if r["contrast_verdicts"].get(k) == "inconclusive")
    print(f"  {k:46s} {g['geo']:6.3f}x {lo:18s} n={g['n']}  inconclusive within-instance on {incon}")

def _report(label, key, note):
    v = [r[key] for r in rows if r.get(key)]
    if not v:
        return
    g = gm_interval(v)
    iv = f" [{g['lo']:.2f}, {g['hi']:.2f}]" if g["lo"] else ""
    print(f"  {label:24s} geo {g['geo']:.2f}x{iv}  wins {sum(1 for x in v if x > 1)}/{len(v)}   {note}")


if any(r.get("warm_vs_cuml") for r in rows):
    print("\nvs cuML (faster passing precision that reaches the target, its fastest run):")
    _report("SOLVE LOOP", "warm_vs_cuml", "the loop alone")
    _report("REPEATED SOLVE", "repeat_vs_cuml", "loop + compile with the cache warm: what a caller gets from the 2nd solve on")
    _report("FIRST SOLVE", "first_vs_cuml", "empty cache, pays tracing and compilation once")
    print("  Three different claims. Say which one any quoted number is.")
if a.out:
    json.dump({"rows": rows, "contrasts": {k: gm_interval(v) for k, v in per_contrast.items() if v}},
              open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}")
