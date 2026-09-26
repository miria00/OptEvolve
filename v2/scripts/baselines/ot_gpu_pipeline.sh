#!/bin/bash
# OT on DOTmark: calibrate on the GPU (N is device-invariant), freeze the ratio,
# measure clean single-core CPU t, evaluate on the GPU against POT emd timed on
# a dedicated P-core (the host thread of this process is pinned to CPU 0).
set -u
cd /home/miria/OptEvolve/v2
OUT=results/ot_dotmark_20260910
PY=/home/miria/jaxenv2/bin/python
H=scripts/baselines/ot_headtohead.py
GPUENV="env -u LD_LIBRARY_PATH JAX_PLATFORMS=cuda,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=1"
echo "[1] calibrate on GPU, res 32"
taskset -c 0 $GPUENV $PY $H --mode calibrate --device gpu --operator reduction --cap 50000 \
  --out $OUT/calib_gpu_res32.json 2>&1 | grep -v -i warn
read RP RE < <($PY - <<'PYE'
import json
rows = json.load(open("results/ot_dotmark_20260910/calib_gpu_res32.json"))["rows"]
best = {}
for sc in ("pdhg", "pdhg_entropic"):
    rs = [r for r in rows if r["scheme"] == sc]
    ok = [r for r in rs if r["passed"]]
    pick = min(ok, key=lambda r: r["iters"]) if ok else min(rs, key=lambda r: r["cert"])
    best[sc] = pick["ratio"]
print(best["pdhg"], best["pdhg_entropic"])
PYE
)
echo "[2] frozen ratios: pdhg $RP, entropic $RE"
echo "[3] clean single-core CPU per-iteration time, res 32"
taskset -c 0 env -u LD_LIBRARY_PATH JAX_PLATFORMS=cpu XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" OMP_NUM_THREADS=1 \
  $PY - <<'PYE' 2>&1 | grep -v -i warn | tee $OUT/cpu_periter_res32.txt
import sys; sys.path.insert(0, ".")
import jax; jax.config.update("jax_enable_x64", True)
from atlas.bench import dotmark
from atlas.genome import Backend
from atlas.schemes.base_schemes import Budget, PDHGParams, build_scheme, run_built
for op in ("incidence", "reduction"):
    prob = dotmark.composite("CauchyDensity", 32, 1009, 1010, operator=op)
    for scheme, nb in (("pdhg", float(prob.L.norm_bound)), ("pdhg_entropic", 2 ** 0.5)):
        params = PDHGParams(t=0.99 / nb, s=0.99 / nb)
        built = build_scheme(prob, scheme, params)
        r = run_built(built, built.cert, params, Budget(max_iters=256, tol=0.0, sample_every=128),
                      backend=Backend(precision="f64", K=64, device="cpu"))
        print(f"cpu P-core res 32 {op:9s} {scheme:14s} {1e6 * r.wall_time / r.iters:9.1f} us/iter")
PYE
echo "[4] evaluate on GPU, res 32, 30 test pairs"
taskset -c 0 $GPUENV $PY $H --mode eval --device gpu --operator reduction --cap 50000 \
  --ratio-pdhg $RP --ratio-entropic $RE --out $OUT/eval_gpu_res32.json 2>&1 | grep -v -i warn
echo "[5] evaluate on GPU, res 64, one pair per class (ratio transferred from res 32)"
taskset -c 0 $GPUENV $PY $H --mode eval --device gpu --operator reduction --cap 50000 --res 64 --pairs 1001-1002 \
  --ratio-pdhg $RP --ratio-entropic $RE --out $OUT/eval_gpu_res64.json 2>&1 | grep -v -i warn
echo "[done]"
