"""Is the exact solver still viable at this DOTmark resolution? Times POT's network simplex on one pair and
nothing else, so the caller can cap it with `timeout` and record the cap as the regime boundary."""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import ot
from atlas.bench import dotmark

ap = argparse.ArgumentParser()
ap.add_argument("--res", type=int, required=True); ap.add_argument("--cls", default="CauchyDensity")
ap.add_argument("--pair", default="1001-1002"); ap.add_argument("--out", required=True)
a = ap.parse_args()
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
i1, i2 = map(int, a.pair.split("-"))
t0 = time.perf_counter(); A, B, C = dotmark.pair(a.cls, a.res, i1, i2); t_load = time.perf_counter() - t0
print(f"res {a.res}: {A.size} x {B.size} supports, cost matrix {C.nbytes / 1e9:.2f} GB, loaded in {t_load:.1f}s", flush=True)
t0 = time.perf_counter(); G, log = ot.emd(A, B, C, numItermax=10 ** 8, log=True); t = time.perf_counter() - t0
rec = {"res": a.res, "class": a.cls, "pair": a.pair, "supports": int(A.size), "cost_gb": float(C.nbytes / 1e9),
       "t_s": t, "cost": float(log["cost"]), "warning": log.get("warning")}
json.dump(rec, open(a.out, "w"), indent=1)
print(f"POT exact: {t:.1f}s cost {log['cost']:.6g} warning {log.get('warning')}", flush=True)
print("POT_PROBE_DONE")
