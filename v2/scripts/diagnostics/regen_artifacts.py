"""Regenerate and PERSIST the numbers that currently exist only as prose.

The claim ledger found that the fANOVA decomposition and the kernel admission
result are quoted in REPORT.md files with no machine-readable output archived
anywhere. A reviewer asking for the fit could not be answered. This recomputes
both from the raw factorials and writes them to results/artifacts_20260910/.
"""
import json, math, statistics as st
from pathlib import Path
R = Path('/home/miria/OptEvolve/v2/results')
OUT = R / 'artifacts_20260910'; OUT.mkdir(exist_ok=True)

def cells(path, tol, alg=None):
    f = json.load(open(path)); out = {}
    for r in f['tolerances'][tol]['rows']:
        if alg and r['algorithm'] != alg: continue
        rows = r['train']['rows']
        out[(r['algorithm'], r['policy'])] = dict(
            N=st.mean([q['iters'][0] for q in rows]),
            t=st.mean([q['solver_s'] / q['iters'][0] * 1e6 for q in rows]),
            sgm1=r['train']['penalized_sgm1_s'], solved=r['train']['n_solved'],
            per_instance_N=[q['iters'][0] for q in rows])
    return out

def fanova(y):
    """Exact ANOVA decomposition of a full factorial. y maps tuple -> value."""
    ks = list(y); nfac = len(ks[0])
    levels = [sorted({k[i] for k in ks}) for i in range(nfac)]
    gm = st.mean(y.values())
    def m1(i, v): return st.mean([x for k, x in y.items() if k[i] == v]) - gm
    def m2(i, j, a, b): return st.mean([x for k, x in y.items() if k[i] == a and k[j] == b]) - gm
    main = [{v: m1(i, v) for v in levels[i]} for i in range(nfac)]
    inter = {}
    for i in range(nfac):
        for j in range(i + 1, nfac):
            inter[(i, j)] = {(a, b): m2(i, j, a, b) - main[i][a] - main[j][b]
                             for a in levels[i] for b in levels[j]}
    res = {}
    for k, v in y.items():
        acc = v - gm - sum(main[i][k[i]] for i in range(nfac))
        for (i, j), d in inter.items(): acc -= d[(k[i], k[j])]
        res[k] = acc
    n = len(y)
    ss = lambda d: sum(x * x for x in d.values()) * (n // len(d))
    parts = {}
    for i in range(nfac): parts[f"main_{i}"] = ss(main[i])
    for (i, j), d in inter.items(): parts[f"inter_{i}x{j}"] = ss(d)
    parts["residual"] = sum(x * x for x in res.values())
    tot = sum(parts.values())
    return {k: 100 * v / tot for k, v in parts.items()}, gm, n

report = {"generated_from": "raw factorial JSONs", "decompositions": {}}

# A. algorithm x precision x cadence, on log penalised SGM-1 time (TV n=256, 4090)
src = R / 'tv_ffi_20260906/cadence_4090_v1/factorial.json'
for tol in ('0.0001', '1e-06'):
    c = cells(src, tol)
    y = {}
    for (alg, pol), v in c.items():
        prec, K = pol.rsplit('_K', 1)
        y[(alg, prec, int(K))] = math.log(v['sgm1'])
    pct, gm, n = fanova(y)
    report["decompositions"][f"alg_x_prec_x_cadence__logSGM1__tol{tol}"] = {
        "source": str(src.relative_to(R)), "n_cells": n,
        "factors": ["algorithm", "precision", "cadence"],
        "percent_variance": {
            "algorithm": pct["main_0"], "precision": pct["main_1"], "cadence": pct["main_2"],
            "algorithm_x_precision": pct["inter_0x1"], "algorithm_x_cadence": pct["inter_0x2"],
            "precision_x_cadence": pct["inter_1x2"], "three_way_residual": pct["residual"]}}

# B. execution x precision x cadence, on log per-iteration cost
SRC = {('device_loop','f64'): ('tv_ffi_20260906/cadence_4090_v1/factorial.json','f64_K%d'),
       ('device_loop','f32'): ('tv_ffi_20260906/f32row_4090/factorial.json','f32_cert64_K%d'),
       ('host_phases','f64'): ('tv_ffi_20260906/f64host_4090/factorial.json','schedule_K%d'),
       ('host_phases','f32'): ('tv_ffi_20260906/px999999_4090/factorial.json','schedule_K%d')}
for tol in ('0.0001', '1e-06'):
    y = {}; raw = {}
    for (ex, pr), (path, pat) in SRC.items():
        c = cells(R / path, tol, alg='condat_vu_banked')
        for K in (16, 64, 256):
            key = ('condat_vu_banked', pat % K)
            if key in c:
                y[(ex, pr, K)] = math.log(c[key]['t'])
                raw[f"{ex}|{pr}|K{K}"] = c[key]['t']
    if len(y) == 12:
        pct, gm, n = fanova(y)
        report["decompositions"][f"exec_x_prec_x_cadence__logT__tol{tol}"] = {
            "sources": {f"{k[0]}|{k[1]}": v[0] for k, v in SRC.items()}, "n_cells": n,
            "per_iteration_us": raw,
            "percent_variance": {
                "execution": pct["main_0"], "precision": pct["main_1"], "cadence": pct["main_2"],
                "execution_x_precision": pct["inter_0x1"], "execution_x_cadence": pct["inter_0x2"],
                "precision_x_cadence": pct["inter_1x2"], "three_way_residual": pct["residual"]}}

(OUT / 'fanova.json').write_text(json.dumps(report, indent=1))
print("wrote", OUT / 'fanova.json')
for name, d in report["decompositions"].items():
    print(f"\n{name}  (n={d['n_cells']})")
    for k, v in d["percent_variance"].items(): print(f"    {k:26s} {v:6.2f}%")
