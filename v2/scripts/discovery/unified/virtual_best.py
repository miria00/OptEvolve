"""VIRTUAL BEST: speedup of a method against the best of K baseline solvers, cell by cell, plus certified-cell counts.

A cell is one (instance, accuracy target tol). Input is JSONL, one record per solver run judged at one tol:
    {"instance": str, "tol": float, "solver": str, "t_s": float, "accepted": bool}
with optional "eps" (the setting the solver was run at), "rep", "rv", "gap". test_baselines.py writes this format; our
own method is recorded the same way under any solver name ("ours" by default) and is never part of the virtual best.

Per (cell, solver): accepted reps of one eps setting are reduced to their MEDIAN time, then the solver's cell time is the
MIN over eps settings. Taking the min over eps lets a baseline use the fastest setting that our checker accepts at tol,
which is generous to the baselines; --matched-eps keeps only runs at eps == tol (the untuned protocol). Acceptance is
always OUR checker's verdict in the record, never a solver status.

    virtual best(cell)  = min over baselines of the solver's cell time (among solvers accepted at that cell)
    speedup(cell)       = t_virtual_best / t_method, on cells BOTH certify; summarized by the geometric mean
    certified cells     = cells with an accepted record, per solver, for the method, and for the virtual best
Cells only the method certifies, or only the virtual best certifies, are counted separately and never enter the mean
(an infinite or zero ratio is not a speedup). The per-tolerance breakdown shows how the margin moves with the target; when
records carry rv and gap it also reports the accuracy margin log10(tol / max(rv, gap)) of the method and of the winner.

    python virtual_best.py records.jsonl [more.jsonl ...] --method ours [--baselines a,b] [--matched-eps] [--json out]
    python virtual_best.py --selftest
"""
import argparse, json, math, sys
from collections import defaultdict


def load_records(paths):
    recs = []
    for p in paths:
        with open(p) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    recs.append(json.loads(line))
    return recs


def _ok_time(r):
    t = r.get("t_s")
    return bool(r.get("accepted")) and t is not None and math.isfinite(float(t)) and float(t) > 0


def _tol_key(t):
    return float(f"{float(t):.6g}")         # 1e-06 and 0.000001 from different writers are the same rung


def cell_times(recs, matched_eps=False):
    """{(instance, tol): {solver: (t_s, record_of_that_time)}} over accepted runs, median over reps, min over eps."""
    reps = defaultdict(list)
    for r in recs:
        tol = _tol_key(r["tol"])
        if matched_eps and r.get("eps") is not None and _tol_key(r["eps"]) != tol:
            continue
        if _ok_time(r):
            eps = None if r.get("eps") is None else _tol_key(r["eps"])
            reps[(str(r["instance"]), tol, str(r["solver"]), eps)].append(r)
    out = defaultdict(dict)
    for (inst, tol, solver, _eps), rs in reps.items():
        rs = sorted(rs, key=lambda r: float(r["t_s"]))
        med = rs[(len(rs) - 1) // 2]             # lower median, so the time is one that was actually measured
        cur = out[(inst, tol)].get(solver)
        if cur is None or float(med["t_s"]) < cur[0]:
            out[(inst, tol)][solver] = (float(med["t_s"]), med)
    return out


def _geomean(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else None


def _acc_margin(r, tol):
    if r is None or r.get("rv") is None or r.get("gap") is None:
        return None
    worst = max(float(r["rv"]), float(r["gap"]))
    return math.log10(tol / worst) if worst > 0 else float("inf")


def _median(xs):
    xs = sorted(x for x in xs if x is not None)
    return None if not xs else (xs[len(xs) // 2] if len(xs) % 2 else 0.5 * (xs[len(xs) // 2 - 1] + xs[len(xs) // 2]))


def virtual_best(recs, method="ours", baselines=None, matched_eps=False, exclude=()):
    """Summary dict for `method` against the virtual best of `baselines` (default: every other solver in the records)."""
    all_cells = sorted({(str(r["instance"]), _tol_key(r["tol"])) for r in recs})
    solvers = sorted({str(r["solver"]) for r in recs})
    if baselines is None:
        baselines = [s for s in solvers if s != method and s not in exclude]
    ct = cell_times(recs, matched_eps=matched_eps)
    rows = []
    for cell in all_cells:
        times = ct.get(cell, {})
        base = {s: times[s] for s in baselines if s in times}
        vb = min(base.items(), key=lambda kv: kv[1][0]) if base else None
        me = times.get(method)
        rows.append({"instance": cell[0], "tol": cell[1],
                     "vb_solver": vb[0] if vb else None, "t_vb": vb[1][0] if vb else None,
                     "t_method": me[0] if me else None,
                     "speedup": (vb[1][0] / me[0]) if (vb and me) else None,
                     "acc_margin_method": _acc_margin(me[1], cell[1]) if me else None,
                     "acc_margin_vb": _acc_margin(vb[1][1], cell[1]) if vb else None,
                     "certified": {s: s in times for s in list(baselines) + [method]}})

    def block(rs):
        sp = [r["speedup"] for r in rs if r["speedup"] is not None]
        wins = defaultdict(int)
        for r in rs:
            if r["vb_solver"]:
                wins[r["vb_solver"]] += 1
        return {"cells": len(rs),
                "certified": {s: sum(r["certified"][s] for r in rs) for s in list(baselines) + [method]},
                "certified_vb": sum(r["t_vb"] is not None for r in rs),
                "both": len(sp), "method_only": sum(r["t_method"] is not None and r["t_vb"] is None for r in rs),
                "vb_only": sum(r["t_method"] is None and r["t_vb"] is not None for r in rs),
                "geomean_speedup": _geomean(sp), "median_speedup": _median(sp),
                "min_speedup": min(sp) if sp else None, "max_speedup": max(sp) if sp else None,
                "method_faster": sum(s > 1 for s in sp), "vb_faster": sum(s < 1 for s in sp),
                "vb_winner_counts": dict(wins),
                "median_acc_margin_method": _median([r["acc_margin_method"] for r in rs if r["speedup"] is not None]),
                "median_acc_margin_vb": _median([r["acc_margin_vb"] for r in rs if r["speedup"] is not None])}

    tols = sorted({r["tol"] for r in rows}, reverse=True)
    return {"method": method, "baselines": list(baselines), "matched_eps": matched_eps,
            "overall": block(rows), "per_tol": {t: block([r for r in rows if r["tol"] == t]) for t in tols},
            "rows": rows}


def _f(x, fmt="{:.3g}"):
    return "-" if x is None else fmt.format(x)


def report(s, per_cell=False, file=sys.stdout):
    p = lambda *a: print(*a, file=file)
    p(f"method {s['method']!r} vs virtual best of {s['baselines']}"
      f"{'  (matched eps only)' if s['matched_eps'] else '  (min over eps settings)'}")
    hdr = (f"  {'tol':>8s} {'cells':>5s} {'cert_me':>7s} {'cert_VB':>7s} {'both':>5s} {'me_only':>7s} {'VB_only':>7s} "
           f"{'geomean':>8s} {'median':>8s} {'min':>8s} {'max':>8s} {'me>VB':>5s} {'VB>me':>5s} {'accM_me':>7s} "
           f"{'accM_VB':>7s}")
    p(hdr)
    for label, b in list(s["per_tol"].items()) + [("all", s["overall"])]:
        p(f"  {label if isinstance(label, str) else format(label, 'g'):>8s} {b['cells']:5d} "
          f"{b['certified'][s['method']]:7d} {b['certified_vb']:7d} {b['both']:5d} {b['method_only']:7d} "
          f"{b['vb_only']:7d} {_f(b['geomean_speedup']):>8s} {_f(b['median_speedup']):>8s} "
          f"{_f(b['min_speedup']):>8s} {_f(b['max_speedup']):>8s} {b['method_faster']:5d} {b['vb_faster']:5d} "
          f"{_f(b['median_acc_margin_method'], '{:.2f}'):>7s} {_f(b['median_acc_margin_vb'], '{:.2f}'):>7s}")
    p("  accM = median log10(tol / max(rowviol, objgap)) over cells both certify: digits of accuracy to spare")
    p("  certified cells per solver: " + ", ".join(f"{k}={v}" for k, v in s["overall"]["certified"].items())
      + f", VIRTUAL_BEST={s['overall']['certified_vb']}")
    p("  virtual-best winner counts: " + ", ".join(f"{k}={v}" for k, v in sorted(s["overall"]["vb_winner_counts"].items())))
    if per_cell:
        for r in s["rows"]:
            p(f"    {r['instance'][:28]:28s} {r['tol']:8g} VB={str(r['vb_solver']):9s} t_VB={_f(r['t_vb'])} "
              f"t_me={_f(r['t_method'])} speedup={_f(r['speedup'])}")


def selftest():
    """Tiny synthetic set with a hand-computed answer."""
    R = lambda i, t, s, ts, ok, **kw: {"instance": i, "tol": t, "solver": s, "t_s": ts, "accepted": ok, **kw}
    recs = [
        # A @ 1e-2: s1 reps 0.9 / 1.0 / 1.2 -> lower median 1.0; s2 2.0; ours 0.5 -> VB s1 1.0, speedup 2
        R("A", 1e-2, "s1", 1.2, True, eps=1e-2), R("A", 1e-2, "s1", 0.9, True, eps=1e-2),
        R("A", 1e-2, "s1", 1.0, True, eps=1e-2), R("A", 1e-2, "s2", 2.0, True, eps=1e-2),
        R("A", 1e-2, "ours", 0.5, True),
        # A @ 1e-6: s1 4.0, s2 rejected, ours 1.0 -> speedup 4
        R("A", 1e-6, "s1", 4.0, True, eps=1e-6), R("A", 1e-6, "s2", 0.1, False, eps=1e-6),
        R("A", 1e-6, "ours", 1.0, True),
        # B @ 1e-2: s1 error record; s2 fast-but-rejected at eps 1e-2, accepted at eps 1e-3 (3.0); ours 6.0 -> 0.5
        R("B", 1e-2, "s1", None, False, eps=1e-2), R("B", 1e-2, "s2", 1.0, False, eps=1e-2),
        R("B", 1e-2, "s2", 3.0, True, eps=1e-3), R("B", 1e-2, "ours", 6.0, True),
        # B @ 1e-6: no baseline certifies, ours does -> method-only
        R("B", 1e-6, "s1", 9.0, False, eps=1e-6), R("B", 1e-6, "ours", 2.0, True),
        # C @ 1e-2: only s1 -> VB-only; C @ 1e-6: nobody
        R("C", 1e-2, "s1", 1.0, True, eps=1e-2), R("C", 1e-2, "ours", 0.3, False),
        R("C", 1e-6, "s1", 1.0, False, eps=1e-6), R("C", 0.000001, "ours", 1.0, False),
    ]
    s = virtual_best(recs, method="ours")
    o = s["overall"]
    assert s["baselines"] == ["s1", "s2"], s["baselines"]
    assert o["cells"] == 6 and o["both"] == 3 and o["method_only"] == 1 and o["vb_only"] == 1, o
    assert o["certified"] == {"s1": 3, "s2": 2, "ours": 4} and o["certified_vb"] == 4, o
    assert abs(o["geomean_speedup"] - 4 ** (1 / 3)) < 1e-12, o["geomean_speedup"]      # (2 * 4 * 0.5) ** (1/3)
    assert o["method_faster"] == 2 and o["vb_faster"] == 1 and o["vb_winner_counts"] == {"s1": 3, "s2": 1}, o
    assert abs(s["per_tol"][1e-2]["geomean_speedup"] - 1.0) < 1e-12                    # sqrt(2 * 0.5)
    assert abs(s["per_tol"][1e-6]["geomean_speedup"] - 4.0) < 1e-12
    # untuned protocol: s2's accepted run at eps 1e-3 no longer counts for the 1e-2 cell, so B @ 1e-2 is method-only
    m = virtual_best(recs, method="ours", matched_eps=True)["overall"]
    assert m["both"] == 2 and m["method_only"] == 2 and abs(m["geomean_speedup"] - math.sqrt(8)) < 1e-12, m
    # restricting the field changes the virtual best: without s1, A's cells go to s2 or nobody
    r = virtual_best(recs, method="ours", baselines=["s2"])["overall"]
    assert r["both"] == 2 and abs(r["geomean_speedup"] - math.sqrt(4 * 0.5)) < 1e-12, r
    report(s, per_cell=True)
    print("SELFTEST OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", nargs="*")
    ap.add_argument("--method", default="ours")
    ap.add_argument("--baselines", default="", help="comma list; default every solver in the records except --method")
    ap.add_argument("--exclude", default="", help="comma list of solvers to leave out of the default baseline field")
    ap.add_argument("--matched-eps", action="store_true", help="only runs with eps == tol (untuned baselines)")
    ap.add_argument("--per-cell", action="store_true")
    ap.add_argument("--json", default="")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        sys.exit(0)
    if not a.jsonl:
        ap.error("give at least one JSONL file, or --selftest")
    s = virtual_best(load_records(a.jsonl), method=a.method,
                     baselines=[b for b in a.baselines.split(",") if b] or None, matched_eps=a.matched_eps,
                     exclude=tuple(b for b in a.exclude.split(",") if b))
    report(s, per_cell=a.per_cell)
    if a.json:
        with open(a.json, "w") as fh:
            json.dump({**s, "per_tol": {f"{k:g}": v for k, v in s["per_tol"].items()}}, fh, indent=1)
