"""Scaled version of pc_gate_probe: does PC scaling clear the Phase-2 OOD
gate wipeout across the whole OOD tier, not just app1-1?

Phase 2 banked 1340/1340 champion OOD rows as Stage-0 rejections. This
re-runs the gate arithmetic for the Phase-2 champion under PC scaling on
every small+medium tier instance and reports the pass rate, plus a
Lemma-2 audit (max observed ||D1 A D2|| over all instances and alphas).
"""
import json
import os
import sys
import time

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from atlas.bench import mps_cell
from atlas.schemes.rescale import (
    pock_chambolle_diagonals,
    power_iteration_norm_bound,
)

TAU, SIGMA = 0.01, 9.45468304801  # llm_seed43 champion
TS = TAU * SIGMA
ALPHAS = [0.0, 1.0, 2.0]
ROOT = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "data", "lp_suites")
TIER = sys.argv[1] if len(sys.argv) > 1 else "small"
LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else 10**9

man = json.load(open(os.path.join(ROOT, "MANIFEST.json")))
small_max = man["small_tier_nnz"]
med_max = man["medium_tier_nnz"]
rows = []
for e in man["instances"]:
    nnz = e.get("nnz") or 0
    if TIER == "small" and nnz <= small_max:
        rows.append(e)
    elif TIER == "medium" and small_max < nnz <= med_max:
        rows.append(e)
rows.sort(key=lambda e: e.get("nnz") or 0)
rows = rows[:LIMIT]
print(f"tier={TIER}  instances={len(rows)}  "
      f"(small_tier_nnz<={small_max}, medium<={med_max})", flush=True)
print(f"champion tau*sigma={TS:.6g}\n", flush=True)

out = []
worst_lemma = 0.0
n_raw_pass = n_pc_pass = 0
t0 = time.time()
for i, e in enumerate(rows):
    path = os.path.join(ROOT, e["file"])
    name = e["name"]
    try:
        std = mps_cell.load_standard_form(path)
        A = std.A.tocsr()
    except Exception as ex:  # noqa: BLE001
        print(f"[{i+1}/{len(rows)}] {name}: LOAD FAIL {type(ex).__name__}", flush=True)
        continue
    op_raw = power_iteration_norm_bound(A)
    gate_raw = TS * op_raw**2
    raw_ok = gate_raw < 1.0
    n_raw_pass += raw_ok
    per_alpha = {}
    all_pc_ok = True
    for a in ALPHAS:
        d1, d2 = pock_chambolle_diagonals(A, a)
        A_s = sp.diags(d1) @ A @ sp.diags(d2)
        op_true = power_iteration_norm_bound(A_s, safety=1.0)
        gate = TS * power_iteration_norm_bound(A_s) ** 2
        worst_lemma = max(worst_lemma, op_true)
        per_alpha[a] = (op_true, gate, gate < 1.0)
        all_pc_ok &= gate < 1.0
    n_pc_pass += all_pc_ok
    out.append({"name": name, "nnz": int(e.get("nnz") or 0),
                "op_raw": op_raw, "gate_raw": gate_raw, "raw_pass": bool(raw_ok),
                "pc": {str(k): [v[0], v[1], v[2]] for k, v in per_alpha.items()},
                "pc_pass_all_alpha": bool(all_pc_ok)})
    print(f"[{i+1}/{len(rows)}] {name:22s} nnz={e.get('nnz'):>8} "
          f"raw ||A||={op_raw:9.3f} gate={gate_raw:11.3f} "
          f"{'PASS' if raw_ok else 'REJECT':6s} | "
          f"PC a=1 ||A'||={per_alpha[1.0][0]:.6f} gate={per_alpha[1.0][1]:.5f} "
          f"{'PASS' if per_alpha[1.0][2] else 'REJECT'}", flush=True)

n = len(out)
print(f"\n===== SUMMARY tier={TIER} ({n} instances, {time.time()-t0:.0f}s) =====")
print(f"  UNSCALED  champion passes gate: {n_raw_pass}/{n} "
      f"({100*n_raw_pass/max(n,1):.1f}%)")
print(f"  PC-SCALED champion passes gate at ALL alphas: {n_pc_pass}/{n} "
      f"({100*n_pc_pass/max(n,1):.1f}%)")
print(f"  LEMMA-2 AUDIT: max ||D1 A D2|| over all instances x alphas = "
      f"{worst_lemma:.9f}  {'OK (<=1)' if worst_lemma <= 1+1e-9 else 'VIOLATED'}")
outdir = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "results", "sprint", "phase3")
os.makedirs(outdir, exist_ok=True)
outpath = os.path.join(outdir, f"pc_ood_{TIER}.json")
json.dump({"summary": {"tier": TIER, "n": n,
                       "unscaled_pass": n_raw_pass, "pc_pass_all_alpha": n_pc_pass,
                       "max_scaled_opnorm": worst_lemma, "alphas": ALPHAS,
                       "tau": TAU, "sigma": SIGMA},
           "instances": out}, open(outpath, "w"), indent=1)
print(f"wrote {outpath}")
