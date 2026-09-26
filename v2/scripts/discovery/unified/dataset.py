"""STEP 3 data - (instance, candidate formulation) -> features, N, costs.

Candidates per instance: std_raw, std_ruiz, reloc with an exact projection, reloc with Dykstra-3 and Dykstra-30.
The Dykstra candidates matter: they are the cases where absorbing the block is NOT automatically right (an inexact
or slow projection can cost more than it saves), which is what keeps the selection problem from being a threshold.
Every instance is permuted. Output is JSONL, one line per (instance, candidate); failures are recorded, not dropped.
"""
import argparse, json, os, sys, time, traceback, socket
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from uni import (make_instance, permute_instance, diagnose, plan_block, build_formulation, reference, measure,
                 features)

ap = argparse.ArgumentParser()
ap.add_argument("--grid", choices=("small", "large"), required=True)
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--nshards", type=int, default=1)
ap.add_argument("--out", required=True)
ap.add_argument("--certified-only", action="store_true",
                help="drop the inexact Dykstra candidates: they fail certification and are not admissible anyway")
a = ap.parse_args()

if a.grid == "small":
    sizes, seeds, cap, K = (500, 1000, 2000), (0, 1), 150000, 50
    big = {}
else:
    sizes, seeds, cap, K = (10000, 30000), (0, 1), 60000, 100
    big = {"knapsack_ridge": (8000, 20000)}

grid = []
for fam in ("portfolio", "svm_dual", "meb_dual", "knapsack_ridge", "budget_band"):
    for n in big.get(fam, sizes):
        for seed in seeds:
            if fam == "portfolio":
                for k in (max(2, int(np.sqrt(n))), max(10, n // 10)):
                    grid.append((fam, n, seed, {"k": k}))
            elif fam == "knapsack_ridge":
                for rho in ((0.0, 1e-3, 1e-1) if a.grid == "small" else (0.0, 1e-2)):
                    grid.append((fam, n, seed, {"rho": rho}))
            else:
                grid.append((fam, n, seed, {}))
grid = [g for i, g in enumerate(grid) if i % a.nshards == a.shard]
print(f"shard {a.shard}/{a.nshards}: {len(grid)} instances on {socket.gethostname()}", flush=True)

CANDS = [("std_raw", {}), ("std_ruiz", {}), ("reloc", {"impl": "exact"}), ("reloc", {"impl": "dykstra", "k": 3}),
         ("reloc", {"impl": "dykstra", "k": 30})]
if a.certified_only:
    CANDS = CANDS[:3]
# RESUME: skip (instance, candidate) pairs already written, so a restart does not redo finished work
done = set()
if os.path.exists(a.out):
    for line in open(a.out):
        try:
            r = json.loads(line)
            if "cand" in r and "error" not in r:
                done.add((r["family"], r["size"], r["seed"], json.dumps(r["params"], sort_keys=True), r["cand"]))
        except Exception:
            pass
out = open(a.out, "a")
for fam, n, seed, params in grid:
    meta = {"family": fam, "size": n, "seed": seed, "params": params, "grid": a.grid, "host": socket.gethostname()}
    try:
        d = permute_instance(make_instance(fam, n, seed=seed, **params), 1000 + seed)
        diag = diagnose(d)
        plan, why = plan_block(d, diag)
        if plan is None:
            out.write(json.dumps({**meta, "error": f"no plan: {why}"}) + "\n"); out.flush(); continue
        r0 = time.perf_counter()
        obj_ref, ref_t = reference(d)
        if obj_ref is None:
            out.write(json.dumps({**meta, "error": f"reference: {ref_t}"}) + "\n"); out.flush(); continue
        meta.update({"kind": plan["kind"], "diag": diag, "obj_ref": obj_ref, "ref_s": ref_t,
                     "n_var": int(d["P"].shape[0])})
    except Exception as e:
        out.write(json.dumps({**meta, "error": traceback.format_exc()[-400:]}) + "\n"); out.flush(); continue
    for cand, kw in CANDS:
        kw = dict(kw)
        if kw.get("impl") == "exact":
            kw["impl"] = "sort" if plan["kind"] == "simplex" else "bisect"
        name_guess = cand if cand != "reloc" else "reloc:" + kw["impl"] + (str(kw["k"]) if kw.get("impl") == "dykstra" else "")
        if (fam, n, seed, json.dumps(params, sort_keys=True), name_guess) in done:
            continue
        try:
            form = build_formulation(d, cand, plan=plan, **kw)
            # standard candidates at large n almost always run to the cap; checking them every 1000 iterations instead
            # of every K cuts host-side acceptance checks 10x on the H100's slow CPU, at 1000-iteration N resolution
            K_c = 1000 if (a.grid == "large" and cand.startswith("std")) else K
            m = measure(form, d, obj_ref, tol=1e-6, cap=cap, K=K_c, probe_iters=200)
            feat = features(form, d, plan, m, probe_iters=200)
            rec = {**meta, "cand": form.name, "N": m["N"], "censored": m["censored"], "iters_run": m["iters_run"],
                   "setup_s": m["setup_s"], "t_iter": m["t_iter"], "rv": m["row_violation"], "gap": m["obj_gap"],
                   "features": feat}
        except Exception as e:
            rec = {**meta, "cand": cand + ":" + str(kw), "error": traceback.format_exc()[-400:]}
        out.write(json.dumps(rec) + "\n"); out.flush()
        print(f"  {fam:15s} n={n:<6d} s={seed} {str(params):16s} {rec.get('cand','?'):18s} "
              f"N={rec.get('N') or ('>' + str(rec.get('iters_run')) if 'iters_run' in rec else 'ERR')}", flush=True)
print("DATASET_DONE", flush=True)
