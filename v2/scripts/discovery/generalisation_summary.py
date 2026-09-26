"""Held-out summary of the generalisation stage: our frozen portfolio against cuML in float32 and float64
(fastest and median of its passing runs, target relative gap 1e-6 on our certificate). Instances where a
cuML precision never reaches the target at any tolerance are counted separately, with its best gap."""
import glob, json, math, os, statistics, sys
G = sys.argv[1] if len(sys.argv) > 1 else "results/generalisation_20260911"
P = json.load(open(os.path.join(G, "portfolio.json")))
gm = lambda xs: math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def ours(rec, label="dispatched"):
    v = rec["times"][label]
    v = rec["times"][v["same_as"]] if "same_as" in v else v
    return v.get("seconds") if v.get("converged") else None


def ours_repeat(rec, label="dispatched"):
    """Solve + compilation with the executable cache warm, i.e. a STEADY-STATE REPEATED solve.

    It is not a cold solve. Since the 2026-09-13 executable cache this collapses onto the solve loop,
    and these records contain NO first-solve number at all: time_config does an untimed warm-up before
    its timed repetitions, so the warm-up absorbs the compilation. Absent in pre-2026-09-13 records.
    """
    v = rec["times"][label]
    v = rec["times"][v["same_as"]] if "same_as" in v else v
    return v.get("seconds_incl_compile") if v.get("converged") else None


def cuml(iid, tag):
    cs = [json.load(open(f))["cuml"] for f in glob.glob(os.path.join(G, "baselines", f"{iid}_{tag}_rep*.json"))]
    ok = [c["seconds"] for c in cs if c.get("passed")]
    gaps = [t["gap"] for c in cs for t in c.get("tries", []) if "gap" in t]
    return {"runs": len(cs), "min": min(ok) if ok else None, "median": statistics.median(ok) if ok else None,
            "best_gap": min(gaps) if gaps else None}


print(f"portfolio {P['sha256'][:16]} frozen {P['frozen_utc']}; members {len(P['members'])}; rule {json.dumps(P['rule'])[:200]}")
recs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(G, "heldout", "*.json")))]
rows = []
for r in recs:
    iid = r["instance"]["id"]
    rows.append({"id": iid, "ours": ours(r), "single": ours(r, "single_best"), "plain": ours(r, "plain_fista"),
                 "ours_rep": ours_repeat(r), "f32": cuml(iid, "cuml"), "f64": cuml(iid, "cuml64")})
print(f"{'instance':28s} {'ours':>8s} {'cuML32 min':>11s} {'cuML64 min':>11s}  notes")
for x in rows:
    f = lambda c: f"{c['min']:.4f}" if c["min"] else (f"FAIL {c['best_gap']:.0e}" if c["best_gap"] else "n/a")
    print(f"{x['id']:28s} {x['ours'] if x['ours'] else float('nan'):8.4f} {f(x['f32']):>11s} {f(x['f64']):>11s}")
def best_cuml(x):
    """Per instance, the faster cuML precision that reaches the target: its fastest run and its median."""
    c = [x[t] for t in ("f32", "f64") if x[t]["min"]]
    if not c:
        return None, None
    b = min(c, key=lambda v: v["min"])
    return b["min"], b["median"]


for label, key in (("dispatched", "ours"), ("single best", "single"), ("plain FISTA", "plain")):
    both = [(x, *best_cuml(x)) for x in rows if x[key] and best_cuml(x)[0]]
    if both:
        sp = [bm / x[key] for x, bm, _ in both]; sm = [md / x[key] for x, _, md in both]
        print(f"{label:12s} vs best cuML precision per instance: {len(both)} instances; geo speedup vs fastest "
              f"{gm(sp):.2f}x, vs median {gm(sm):.2f}x; wins {sum(v > 1 for v in sp)}/{len(both)}; "
              f"range {min(sp):.2f}x to {max(sp):.2f}x")
for label, key in (("dispatched", "ours"), ("single best", "single"), ("plain FISTA", "plain")):
    for tag in ("f32", "f64"):
        both = [x for x in rows if x[key] and x[tag]["min"]]
        fail = [x for x in rows if x[key] and x[tag]["runs"] and not x[tag]["min"]]
        if not both and not fail:
            continue
        print(f"{label:12s} vs cuML {tag}: {len(both)} instances both certify; geo speedup vs fastest "
              f"{gm([x[tag]['min'] / x[key] for x in both]):.2f}x, vs median {gm([x[tag]['median'] / x[key] for x in both]):.2f}x; "
              f"wins {sum(x[tag]['min'] > x[key] for x in both)}/{len(both)}; cuML never reaches the target on {len(fail)} where we do")


# The repeated-solve comparison, reported separately because it is a DIFFERENT CLAIM from the solve loop.
rep = [(x, *best_cuml(x)) for x in rows if x.get("ours_rep") and best_cuml(x)[0]]
if rep:
    sp = [bm / x["ours_rep"] for x, bm, _ in rep]
    print(f"\ndispatched, REPEATED SOLVE (loop + compile, executable cache warm: what a caller gets from "
          f"the 2nd solve on): {len(rep)} instances; geo {gm(sp):.2f}x vs cuML's fastest; "
          f"wins {sum(v > 1 for v in sp)}/{len(rep)}")
    print("  These records carry NO first-solve number: the warm-up solve absorbs compilation, so a cold "
          "single solve is NOT measured here.")
else:
    print("\n(no repeated-solve figures: records predate the 2026-09-13 compile accounting)")
