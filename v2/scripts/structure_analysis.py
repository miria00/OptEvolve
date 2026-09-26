"""Answer the two structure questions from the sweep JSONs, with numbers.

(a) Does theta = t*s*||L||^2 ALONE predict which admissible setting minimises
    time-to-certificate, or does the true condition number kappa_plus add
    predictive power?
(b) Does the ADMISSIBLE REGION move with family and size as expected?

Usage:
  python scripts/structure_analysis.py results/structure_20260909/grid_4090.json \
      results/structure_20260909/regular_4090.json \
      --rawgrid results/structure_20260909/rawgrid_4090.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np


def geomean(xs):
    xs = [x for x in xs if x is not None and x > 0]
    return float(math.exp(sum(math.log(x) for x in xs) / len(xs))) if xs else None


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.size < 3:
        return None
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    den = math.sqrt(float((ra ** 2).sum() * (rb ** 2).sum()))
    return float((ra * rb).sum() / den) if den > 0 else None


def pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.size < 3:
        return None
    a = a - a.mean(); b = b - b.mean()
    den = math.sqrt(float((a ** 2).sum() * (b ** 2).sum()))
    return float((a * b).sum() / den) if den > 0 else None


def load(paths):
    insts, runs = [], []
    for p in paths:
        d = json.loads(Path(p).read_text())
        insts += d["instances"]
        runs += d["runs"]
    return insts, runs


def cell_table(insts):
    """One row per cell; per-instance quantities are GEOMEAN-aggregated
    (the degree-2 regular cells have genuinely instance-dependent kappa)."""
    byc = {}
    for r in insts:
        byc.setdefault(r["cell_id"], []).append(r)
    cells = {}
    for cid, group in byc.items():
        r = group[0]
        eq = r.get("equilibration") or {}
        pw = r.get("primal_weight") or {}
        g = lambda k: geomean([x.get(k) for x in group])
        ge = lambda k: geomean([(x.get("equilibration") or {}).get(k)
                                for x in group])
        gp = lambda k: geomean([(x.get("primal_weight") or {}).get(k)
                                for x in group])
        cells[cid] = {
            "cell_id": cid, "family": r["family"], "size": r["size"],
            "degree": r.get("degree"),
            "n": int(np.prod(r["shape"])) if r["shape"] else None,
            "rows": eq.get("rows"), "cols": eq.get("cols"), "nnz": eq.get("nnz"),
            "norm_bound": g("norm_bound"), "sigma_max": g("sigma_max"),
            "sigma_min_plus": g("sigma_min_plus"),
            "kappa_plus": g("kappa_plus"), "kappa_method": r["kappa_method"],
            "kappa_min": min([x["kappa_plus"] for x in group
                              if x.get("kappa_plus")] or [float("nan")]),
            "kappa_max": max([x["kappa_plus"] for x in group
                              if x.get("kappa_plus")] or [float("nan")]),
            "n_inst": len(group),
            "rank": r["structural_rank"],
            "omega0": gp("omega0"),
            "rho_row": ge("rho_row"), "rho_col": ge("rho_col"),
            "rho_entry": ge("rho_entry"),
        }
    return cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--rawgrid", default=None)
    ap.add_argument("--metric", default="exec_s", choices=("exec_s", "N"))
    args = ap.parse_args()

    insts, runs = load(args.paths)
    cells = cell_table(insts)
    order = sorted(cells, key=lambda c: (cells[c]["family"], cells[c]["degree"] or 0,
                                         cells[c]["size"]))
    M = args.metric

    print("=" * 118)
    print("CELL STRUCTURE (measured)")
    print("=" * 118)
    hdr = (f"{'cell':<18}{'rows':>6}{'cols':>7}{'nnz':>8}{'||L||_gate':>11}"
           f"{'sigma_max':>11}{'s_min+':>9}{'kappa+':>10}{'method':>15}"
           f"{'1/||L||^2':>11}{'omega0':>9}")
    print(hdr)
    for cid in order:
        c = cells[cid]
        print(f"{cid:<18}{c['rows'] or 0:>6}{c['cols'] or 0:>7}{c['nnz'] or 0:>8}"
              f"{c['norm_bound']:>11.5f}{(c['sigma_max'] or float('nan')):>11.5f}"
              f"{(c['sigma_min_plus'] or float('nan')):>9.5f}"
              f"{(c['kappa_plus'] or float('nan')):>10.4f}{c['kappa_method']:>15}"
              f"{1.0/c['norm_bound']**2:>11.6f}{(c['omega0'] or float('nan')):>9.4f}")

    # ---------------- table of metric vs (theta, t) ----------------
    thetas = sorted({r["theta_nominal"] for r in runs})
    tau_t = {}
    for r in runs:
        tau_t.setdefault(r["t_label"], []).append(r["t"])
    taus = sorted(tau_t, key=lambda k: float(np.median(tau_t[k])))
    tab = {}
    for cid in order:
        for th in thetas:
            for tl in taus:
                sub = [r for r in runs if r["cell_id"] == cid
                       and r["theta_nominal"] == th and r["t_label"] == tl]
                if not sub:
                    continue
                if not sub[0]["gate_admit"]:
                    tab[(cid, th, tl)] = None
                    continue
                tab[(cid, th, tl)] = {
                    "metric": geomean([r[M] for r in sub]),
                    "exec_s": geomean([r["exec_s"] for r in sub]),
                    "N": geomean([r["N"] for r in sub]),
                    "r": float(np.median([r["ratio_t_over_s"] for r in sub])),
                    "t": float(np.median([r["t"] for r in sub])),
                    "s": float(np.median([r["s"] for r in sub])),
                    "tL": float(np.median([r["t"] * r["opnorm_gate"] for r in sub])),
                    "capped": any(not r["converged"] for r in sub),
                    "n": len(sub),
                }

    print("\n" + "=" * 118)
    print(f"Q1  {M} vs (theta, t)  -- geometric mean over instances, "
          f"* = hit the iteration cap")
    print("=" * 118)
    within_theta = []
    for cid in order:
        c = cells[cid]
        print(f"\n  {cid}  (||L||={c['norm_bound']:.4f}, "
              f"kappa+={c['kappa_plus'] and round(c['kappa_plus'], 3)})")
        print("    " + f"{'theta \\ t':<11}" + "".join(f"{t:>11s}" for t in taus)
              + f"{'  spread':>10}{'  argmin t':>11}")
        for th in thetas:
            row, vals, keyed = [], [], []
            for tl in taus:
                e = tab.get((cid, th, tl), "missing")
                if e == "missing":
                    row.append(f"{'-':>11}"); continue
                if e is None:
                    row.append(f"{'REJECT':>11}"); continue
                v = e["metric"]
                vals.append(v); keyed.append((v, tl))
                txt = (f"{v:10.4f}" if M == "exec_s" else f"{int(round(v)):10d}")
                row.append(txt + ("*" if e["capped"] else " "))
            if len(vals) > 1:
                sp = max(vals) / min(vals)
                within_theta.append((sp, cid, th))
                best_tl = min(keyed)[1]
                print("    " + f"{th:<11g}" + "".join(row)
                      + f"{sp:>9.2f}x{best_tl:>11s}")
            else:
                print("    " + f"{th:<11g}" + "".join(row) + f"{'-':>10}{'-':>11}")

    print("\n  -- within-theta spread (theta held EXACTLY fixed, only r = t/s moves) --")
    within_theta.sort(reverse=True)
    for sp, cid, th in within_theta[:8]:
        print(f"     {cid:<18} theta={th:<6g}  spread {sp:.2f}x")
    if within_theta:
        allsp = [w[0] for w in within_theta]
        print(f"     max {max(allsp):.2f}x   median {np.median(allsp):.2f}x   "
              f"min {min(allsp):.2f}x   over {len(allsp)} (cell,theta) rows")

    # ---------------- best setting per cell ----------------
    print("\n" + "=" * 118)
    print(f"Q2  BEST ADMISSIBLE SETTING PER CELL (argmin {M})")
    print("=" * 118)
    best = {}
    print(f"{'cell':<18}{'theta*':>8}{'t*':>10}{'s*':>10}{'r*=t/s':>10}"
          f"{'t*||L||':>9}{'N*':>9}{'exec*_s':>10}{'kappa+':>9}")
    for cid in order:
        keys = [k for k in tab if k[0] == cid and tab[k]]
        if not keys:
            continue
        bk = min(keys, key=lambda k: tab[k]["metric"])
        e = tab[bk]
        best[cid] = (bk, e["metric"])
        print(f"{cid:<18}{bk[1]:>8g}{e['t']:>10.5f}{e['s']:>10.5f}{e['r']:>10.4g}"
              f"{e['tL']:>9.4f}{int(round(e['N'])):>9d}{e['exec_s']:>10.4f}"
              f"{(cells[cid]['kappa_plus'] or float('nan')):>9.3f}")

    # ---------------- predictor regret ----------------
    print("\n" + "=" * 118)
    print("Q3  REGRET OF EACH PREDICTOR  (regret = metric(prediction)/metric(oracle))")
    print("=" * 118)
    settings = sorted({(k[1], k[2]) for k in tab if tab[k]},
                      key=lambda z: (z[0], str(z[1])))

    def regrets_for(pick):
        """pick(cid) -> (theta, t_label) or None."""
        out = {}
        for cid in best:
            key = pick(cid)
            if key is None:
                return None
            e = tab.get((cid, key[0], key[1]))
            if not e:
                return None
            out[cid] = e["metric"] / best[cid][1]
        return out

    def show(name, reg, extra=""):
        if not reg:
            print(f"  {name:<52} n/a")
            return None
        mx = max(reg.values()); gm = geomean(list(reg.values()))
        worst = max(reg, key=reg.get)
        print(f"  {name:<52} max {mx:6.3f}x   geomean {gm:6.3f}x"
              f"   worst cell {worst}{extra}")
        return mx, gm

    # P0: theta alone -- adversarial choice of t inside the best theta
    p0 = {}
    for cid in best:
        th_star = best[cid][0][1]
        vals = [tab[k]["metric"] for k in tab
                if k[0] == cid and k[1] == th_star and tab[k]]
        p0[cid] = max(vals) / best[cid][1]
    show("P0  theta* known, t chosen adversarially in that row", p0)

    p0m = {}
    for cid in best:
        th_star = best[cid][0][1]
        vals = [tab[k]["metric"] for k in tab
                if k[0] == cid and k[1] == th_star and tab[k]]
        p0m[cid] = geomean(vals) / best[cid][1]
    show("P0b theta* known, t chosen uniformly at random (geomean)", p0m)

    # P1: single global (theta, t) chosen with full oracle knowledge
    scored = []
    for th, tl in settings:
        reg = regrets_for(lambda cid, th=th, tl=tl: (th, tl))
        if reg:
            scored.append((max(reg.values()), geomean(list(reg.values())), th, tl))
    scored.sort()
    if scored:
        print("\n  best SINGLE GLOBAL (theta, t) settings, ranked by worst-case regret:")
        for mx, gm, th, tl in scored[:6]:
            print(f"      theta={th:<6g} t={tl:<6s}   max {mx:6.3f}x   geomean {gm:6.3f}x")
        bal = [z for z in scored if z[3] == "bal"]
        if bal:
            mx, gm, th, tl = min(bal)
            print(f"      CATALOG BALANCED r=1 (t==s), best theta={th:g}: "
                  f"max {mx:.3f}x  geomean {gm:.3f}x   <-- what the gate alone gives")

    # P2 / P3: leave-one-cell-out fits of log2 t* on a structure feature
    def loo_fit(feature_name, getter):
        rows = [(cid, getter(cells[cid])) for cid in best]
        rows = [(cid, v) for cid, v in rows if v is not None and v > 0]
        if len(rows) < 4:
            return None
        reg, preds = {}, {}
        for held, _ in rows:
            xs = [math.log2(v) for cid, v in rows if cid != held]
            ys = [math.log2(tab[best[cid][0]]["t"]) for cid, v in rows if cid != held]
            A = np.vstack([xs, np.ones(len(xs))]).T
            coef, *_ = np.linalg.lstsq(A, np.asarray(ys), rcond=None)
            xh = math.log2(dict(rows)[held])
            t_pred = 2.0 ** float(coef[0] * xh + coef[1])
            th_star_global = scored[0][2] if scored else 0.9
            cand = [k for k in tab if k[0] == held and k[1] == th_star_global and tab[k]]
            if not cand:
                return None
            k_pick = min(cand, key=lambda k: abs(math.log(tab[k]["t"]) - math.log(t_pred)))
            reg[held] = tab[k_pick]["metric"] / best[held][1]
            preds[held] = (t_pred, tab[k_pick]["t"], k_pick[2])
        return reg, preds

    # P-const-loo: the FAIR baseline -- one constant (theta, t) fitted on the
    # other cells only, i.e. exactly the information budget the kappa-aware
    # predictor gets, minus kappa.
    if scored:
        reg = {}
        for held in best:
            cand = []
            for th, tl in settings:
                rs = []
                ok = True
                for cid in best:
                    if cid == held:
                        continue
                    e = tab.get((cid, th, tl))
                    if not e:
                        ok = False
                        break
                    rs.append(e["metric"] / best[cid][1])
                if ok and rs:
                    cand.append((max(rs), geomean(rs), th, tl))
            if not cand:
                reg = None
                break
            cand.sort()
            _, _, th, tl = cand[0]
            e = tab.get((held, th, tl))
            if not e:
                reg = None
                break
            reg[held] = e["metric"] / best[held][1]
        show("P-loo CONSTANT (theta,t) fitted on the other cells", reg)

    # 2-feature LOO: does kappa add anything ON TOP of ||L|| ?
    def loo_fit_multi(name, getters):
        rows = []
        for cid in best:
            v = [g(cells[cid]) for g in getters]
            if any(x is None or x <= 0 for x in v):
                continue
            rows.append((cid, [math.log2(x) for x in v]))
        if len(rows) < len(getters) + 3:
            return None
        th_star_global = scored[0][2] if scored else 0.9
        reg = {}
        for held, xh in rows:
            X = np.asarray([r[1] for r in rows if r[0] != held])
            y = np.asarray([math.log2(tab[best[r[0]][0]]["t"])
                            for r in rows if r[0] != held])
            A = np.hstack([X, np.ones((X.shape[0], 1))])
            coef, *_ = np.linalg.lstsq(A, y, rcond=None)
            t_pred = 2.0 ** float(np.dot(coef[:-1], xh) + coef[-1])
            cand = [k for k in tab if k[0] == held and k[1] == th_star_global and tab[k]]
            if not cand:
                return None
            kp = min(cand, key=lambda k: abs(math.log(tab[k]["t"]) - math.log(t_pred)))
            reg[held] = tab[kp]["metric"] / best[held][1]
        return reg

    print("\n  INCREMENTAL TEST -- does kappa_plus add anything on top of ||L||?")
    for nm, gs in (("||L||", [lambda c: c["norm_bound"]]),
                   ("||L|| + kappa_plus", [lambda c: c["norm_bound"],
                                           lambda c: c["kappa_plus"]]),
                   ("||L|| + kappa + n_cols", [lambda c: c["norm_bound"],
                                               lambda c: c["kappa_plus"],
                                               lambda c: c["cols"]])):
        r = loo_fit_multi(nm, gs)
        show(f"P-loo2 t* ~ {nm}", r)

    # within-family: is the optimum stable across sizes inside one family?
    print("\n  WITHIN-FAMILY STABILITY of the optimum (one family's modal best "
          "setting applied to every cell of that family):")
    fams = {}
    for cid in best:
        c = cells[cid]
        key = (c["family"], c["degree"] if c["family"] == "regular" else None)
        fams.setdefault(key, []).append(cid)
    for key, cids in sorted(fams.items(), key=lambda z: str(z[0])):
        cand = []
        for th, tl in settings:
            rs = [tab[(cid, th, tl)]["metric"] / best[cid][1] for cid in cids
                  if tab.get((cid, th, tl))]
            if len(rs) == len(cids):
                cand.append((max(rs), geomean(rs), th, tl))
        if not cand:
            continue
        cand.sort()
        mx, gm, th, tl = cand[0]
        kmin = min(cells[c]["kappa_plus"] for c in cids)
        kmax = max(cells[c]["kappa_plus"] for c in cids)
        print(f"    {str(key):<22} cells={len(cids)}  kappa+ range "
              f"{kmin:8.2f}..{kmax:8.2f} ({kmax/kmin:5.2f}x)  one setting "
              f"(theta={th:g}, t={tl}) costs max {mx:.3f}x, geomean {gm:.3f}x")

    # LOO prediction of the DIFFICULTY N* itself
    print("\n  LOO prediction of the ORACLE cost N* (log-log fit, "
          "reported as the geomean/max multiplicative error):")
    for nm, g in (("kappa_plus", lambda c: c["kappa_plus"]),
                  ("||L||_gate", lambda c: c["norm_bound"]),
                  ("n_cols", lambda c: c["cols"]),
                  ("1/sigma_min_plus", lambda c: (1.0 / c["sigma_min_plus"])
                   if c["sigma_min_plus"] else None)):
        rows = [(cid, g(cells[cid])) for cid in best]
        rows = [(cid, v) for cid, v in rows if v and v > 0]
        if len(rows) < 5:
            continue
        errs = []
        for held, xh in rows:
            xs = [math.log(v) for cid, v in rows if cid != held]
            ys = [math.log(tab[best[cid][0]]["N"]) for cid, v in rows if cid != held]
            A = np.vstack([xs, np.ones(len(xs))]).T
            coef, *_ = np.linalg.lstsq(A, np.asarray(ys), rcond=None)
            pred = math.exp(float(coef[0] * math.log(xh) + coef[1]))
            act = tab[best[held][0]]["N"]
            errs.append(max(pred / act, act / pred))
        print(f"    N* ~ {nm:<18} LOO error: geomean {geomean(errs):.3f}x  "
              f"max {max(errs):.3f}x")

    print("\n  leave-one-cell-out predictors of t* (theta pinned at the best global "
          f"theta={scored[0][2] if scored else 'n/a'}; the predicted t is snapped to "
          "the nearest swept t):")
    for nm, g in (("kappa_plus", lambda c: c["kappa_plus"]),
                  ("||L||_gate", lambda c: c["norm_bound"]),
                  ("n_cols", lambda c: c["cols"]),
                  ("kappa_plus/||L||", lambda c: (c["kappa_plus"] / c["norm_bound"])
                   if c["kappa_plus"] else None)):
        res = loo_fit(nm, g)
        if res is None:
            print(f"  P-loo {nm:<46} n/a")
            continue
        reg, preds = res
        show(f"P-loo t* ~ {nm}", reg)

    # ---------------- variance decomposition of log(time) ----------------
    print("\n" + "=" * 118)
    print("Q3b VARIANCE DECOMPOSITION of log(%s) over ADMITTED settings" % M)
    print("=" * 118)
    rows = [(cid, th, tl, tab[(cid, th, tl)])
            for cid in order for th in thetas for tl in taus
            if tab.get((cid, th, tl))]
    if len(rows) > 10:
        y = np.log(np.asarray([e["metric"] for _, _, _, e in rows]))
        cids = [r[0] for r in rows]
        ths = np.asarray([r[1] for r in rows], float)
        lr = np.log2(np.asarray([e["r"] for _, _, _, e in rows]))
        ltL = np.log2(np.asarray([e["tL"] for _, _, _, e in rows]))
        uc = sorted(set(cids))
        Dc = np.asarray([[1.0 if c == u else 0.0 for u in uc] for c in cids])

        def r2(X):
            X = np.hstack([X, np.ones((X.shape[0], 1))])
            coef, *_ = np.linalg.lstsq(X, y, rcond=None)
            resid = y - X @ coef
            return 1.0 - float((resid ** 2).sum() / ((y - y.mean()) ** 2).sum())

        print(f"  n = {len(rows)} (cell, theta, t) admitted settings")
        print(f"  R^2  cell dummies only                 = {r2(Dc):.4f}")
        print(f"  R^2  cell + log2(theta)                = {r2(np.hstack([Dc, np.log2(ths)[:, None]])):.4f}")
        print(f"  R^2  cell + log2(theta) + log2(r)      = {r2(np.hstack([Dc, np.log2(ths)[:, None], lr[:, None]])):.4f}")
        print(f"  R^2  cell + log2(theta) + log2(r)+r^2  = "
              f"{r2(np.hstack([Dc, np.log2(ths)[:, None], lr[:, None], (lr**2)[:, None]])):.4f}")
        print(f"  R^2  cell + log2(theta) + log2(t||L||) + sq = "
              f"{r2(np.hstack([Dc, np.log2(ths)[:, None], ltL[:, None], (ltL**2)[:, None]])):.4f}")
        # within-cell: how much of the spread does theta alone explain?
        print("\n  within-cell R^2 of log(%s):" % M)
        print(f"    {'cell':<18}{'theta only':>12}{'theta+log r':>14}{'theta+logr+sq':>16}")
        for cid in order:
            sel = [i for i, c in enumerate(cids) if c == cid]
            if len(sel) < 8:
                continue
            yy = y[sel]
            th_i = np.log2(ths[sel])[:, None]
            lr_i = lr[sel][:, None]

            def r2c(X):
                X = np.hstack([X, np.ones((X.shape[0], 1))])
                coef, *_ = np.linalg.lstsq(X, yy, rcond=None)
                res = yy - X @ coef
                return 1.0 - float((res ** 2).sum() / ((yy - yy.mean()) ** 2).sum())
            print(f"    {cid:<18}{r2c(th_i):>12.4f}{r2c(np.hstack([th_i, lr_i])):>14.4f}"
                  f"{r2c(np.hstack([th_i, lr_i, lr_i**2])):>16.4f}")

    # ---------------- does theta_true differ from theta_gate ----------------
    print("\n" + "=" * 118)
    print("Q4  GATE NORM vs TRUE NORM, AND CORRELATIONS")
    print("=" * 118)
    for cid in order:
        c = cells[cid]
        if c["sigma_max"]:
            print(f"  {cid:<18} ||L||_gate/sigma_max = "
                  f"{c['norm_bound']/c['sigma_max']:.5f}  "
                  f"=> nominal theta=1 is a TRUE product of "
                  f"{(c['sigma_max']/c['norm_bound'])**2:.5f}")
    xs = [cells[c]["kappa_plus"] for c in best if cells[c]["kappa_plus"]]
    ys = [math.log2(tab[best[c][0]]["r"]) for c in best if cells[c]["kappa_plus"]]
    zs = [tab[best[c][0]]["N"] for c in best if cells[c]["kappa_plus"]]
    ts = [math.log2(tab[best[c][0]]["t"]) for c in best if cells[c]["kappa_plus"]]
    nb = [cells[c]["norm_bound"] for c in best if cells[c]["kappa_plus"]]
    nc = [cells[c]["cols"] for c in best if cells[c]["kappa_plus"]]
    if len(xs) >= 3:
        print(f"\n  n_cells with kappa_plus = {len(xs)}")
        print(f"  Spearman(kappa_plus, log2 r*)   = {spearman(xs, ys)}")
        print(f"  Spearman(kappa_plus, log2 t*)   = {spearman(xs, ts)}")
        print(f"  Spearman(kappa_plus, N*)        = {spearman(xs, zs)}")
        print(f"  Spearman(||L||_gate, N*)        = {spearman(nb, zs)}")
        print(f"  Spearman(n_cols,     N*)        = {spearman(nc, zs)}")
        print(f"  Pearson(log kappa_plus, log N*) = "
              f"{pearson([math.log(v) for v in xs], [math.log(v) for v in zs])}")
        A = np.vstack([[math.log(v) for v in xs], np.ones(len(xs))]).T
        coef, res, *_ = np.linalg.lstsq(A, np.log(np.asarray(zs)), rcond=None)
        print(f"  fit  log N* = {coef[0]:.4f} * log kappa_plus + {coef[1]:.4f}"
              f"   => N* ~ kappa_plus^{coef[0]:.3f}")

    # ---------------- Q5: admissible region in raw coordinates ----------------
    if args.rawgrid:
        rinsts, rruns = load([args.rawgrid])
        rcells = cell_table(rinsts)
        rorder = sorted(rcells, key=lambda c: (rcells[c]["family"],
                                               rcells[c]["degree"] or 0,
                                               rcells[c]["size"]))
        print("\n" + "=" * 118)
        print("Q5  ADMISSIBLE REGION IN RAW (t, s) COORDINATES  "
              "(identical catalog on every cell)")
        print("=" * 118)
        taus_r = sorted({r["t"] for r in rruns})
        sigs_r = sorted({r["s"] for r in rruns})
        print(f"  raw catalog: t in {taus_r}")
        print(f"               s in {sigs_r}")
        print(f"\n{'cell':<18}{'||L||_gate':>11}{'||L||^2':>10}{'ts_max':>10}"
              f"{'#admit':>8}{'#total':>8}{'frac':>8}   admits (0.25,0.25)?")
        for cid in rorder:
            sub = [r for r in rruns if r["cell_id"] == cid]
            adm = sum(1 for r in sub if r["gate_admit"])
            c = rcells[cid]
            q = [r for r in sub if abs(r["t"] - 0.25) < 1e-12
                 and abs(r["s"] - 0.25) < 1e-12]
            qq = ("-" if not q else
                  (f"YES lhs={q[0]['gate_lhs']:.4f}" if q[0]["gate_admit"]
                   else f"NO  lhs={q[0]['gate_lhs']:.4f}"))
            print(f"{cid:<18}{c['norm_bound']:>11.5f}{c['norm_bound']**2:>10.4f}"
                  f"{1.0/c['norm_bound']**2:>10.6f}{adm:>8}{len(sub):>8}"
                  f"{adm/len(sub):>8.3f}   {qq}")

        print("\n  per-cell admit map (rows t, cols s; A = admitted, . = rejected)")
        for cid in rorder:
            sub = [r for r in rruns if r["cell_id"] == cid]
            print(f"\n   {cid}   ||L||={rcells[cid]['norm_bound']:.4f}")
            print("     " + f"{'t \\ s':<8}" + "".join(f"{s:>8g}" for s in sigs_r))
            for t in taus_r:
                row = ""
                for s in sigs_r:
                    m = [r for r in sub if abs(r["t"] - t) < 1e-12
                         and abs(r["s"] - s) < 1e-12]
                    row += f"{('A' if m and m[0]['gate_admit'] else '.'):>8}"
                print("     " + f"{t:<8g}" + row)

        # which raw pairs flip verdict across cells
        pairs = sorted({(r["t"], r["s"]) for r in rruns})
        flips = []
        for t, s in pairs:
            vs = {}
            for cid in rorder:
                m = [r for r in rruns if r["cell_id"] == cid
                     and abs(r["t"] - t) < 1e-12 and abs(r["s"] - s) < 1e-12]
                if m:
                    vs[cid] = m[0]["gate_admit"]
            if len(set(vs.values())) > 1:
                flips.append((t, s, vs))
        print(f"\n  {len(flips)}/{len(pairs)} raw (t,s) pairs FLIP verdict across cells:")
        for t, s, vs in flips:
            yes = [c for c in vs if vs[c]]
            no = [c for c in vs if not vs[c]]
            print(f"    t={t:<6g} s={s:<6g}  admitted on {len(yes)} cells, "
                  f"rejected on {len(no)}: rejected = {no}")

        # best raw setting per cell (time-to-certificate), if solved
        solved = [r for r in rruns if r.get("exec_s")]
        if solved:
            print("\n  best RAW (t,s) per cell by measured time-to-certificate:")
            print(f"{'cell':<18}{'t*':>8}{'s*':>8}{'r*':>10}{'lhs*':>9}"
                  f"{'N*':>9}{'exec*_s':>10}")
            for cid in rorder:
                sub = {}
                for r in solved:
                    if r["cell_id"] != cid:
                        continue
                    sub.setdefault((r["t"], r["s"]), []).append(r)
                if not sub:
                    continue
                agg = {k: (geomean([x["exec_s"] for x in v]),
                           geomean([x["N"] for x in v]),
                           v[0]["gate_lhs"]) for k, v in sub.items()}
                bk = min(agg, key=lambda k: agg[k][0])
                e = agg[bk]
                print(f"{cid:<18}{bk[0]:>8g}{bk[1]:>8g}{bk[0]/bk[1]:>10.4g}"
                      f"{e[2]:>9.4f}{int(round(e[1])):>9d}{e[0]:>10.4f}")
    print("=" * 118)


if __name__ == "__main__":
    main()
