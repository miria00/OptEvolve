"""Phase-1 integration smoke: batched GPU solve of 8 same-shape netflow
LPs with per-instance f64 certificates.

Intended for mazeppa-1 (RTX 5090); runs on any JAX default device. Does
NOT set JAX_PLATFORMS: run it in an environment where the GPU is the
default device.

Writes results/sprint/phase1/smoke_batched_gpu.json.

Usage: python scripts/smoke_phase1_batched_gpu.py
"""
from __future__ import annotations

import json
import os
import sys
import time

V2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, V2)

import jax  # noqa: E402
import numpy as np  # noqa: E402

from atlas.backend import batched_solve  # noqa: E402
from atlas.bench.lp_cell import from_netflow, to_composite  # noqa: E402
from atlas.bench.netflow import gen_regular_flow  # noqa: E402
from atlas.genome import Genome, GenomeParams, Wrapper  # noqa: E402
from atlas.wrappers.km import Budget  # noqa: E402

TOL = 1e-4
N = 8
N_NODES = 64
OUT = os.path.join(V2, "results", "sprint", "phase1", "smoke_batched_gpu.json")


def main() -> int:
    dev = jax.devices()[0]
    problems = [
        to_composite(from_netflow(gen_regular_flow(n_nodes=N_NODES, seed=s,
                                                   degree=2)))
        for s in range(N)
    ]
    nb = max(float(p.L.norm_bound) for p in problems)
    genome = Genome(
        scheme="pdhg",
        params=GenomeParams(tau=0.9 / nb, sigma=0.9 / nb),
        wrapper=Wrapper(halpern=True, restart="fixed_k", restart_k=150),
    )
    t0 = time.time()
    res = batched_solve(problems, genome,
                        Budget(max_iters=300_000, tol=TOL), K=100)
    wall = time.time() - t0

    gaps = np.asarray(res.certified_gap)
    conv = np.asarray(res.converged)
    ok = (bool(conv.all()) and bool((gaps <= TOL).all())
          and res.certificate_dtype == "float64"
          and res.n_instances == N)

    out = {
        "device": str(dev),
        "platform": dev.platform,
        "device_kind": getattr(dev, "device_kind", "?"),
        "n_instances": res.n_instances,
        "cell": f"netflow n_nodes={N_NODES} degree=2, seeds 0..{N-1}",
        "genome": {"scheme": "pdhg", "tau": 0.9 / nb, "sigma": 0.9 / nb,
                   "halpern": True, "restart_k": 150},
        "converged": conv.tolist(),
        "certified_gap_per_lane_f64": gaps.tolist(),
        "iters_per_lane": np.asarray(res.iters).tolist(),
        "batch_iters": int(res.batch_iters),
        "batch_wall_s": res.batch_wall_s,
        "compile_time_s": res.compile_time_s,
        "outer_wall_s": wall,
        "precision": res.precision,
        "certificate_dtype": res.certificate_dtype,
        "construction_cert": str(res.cert),
        "PASS": ok,
        "gpu": dev.platform == "gpu",
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
