"""Phase-1 integration smoke: one Mittelmann instance via mps_cell, solved
by solve_genome PDHG to rel gap 1e-4; original-scale objective compared to
the HiGHS oracle.

Writes results/sprint/phase1/smoke_mps.json.

Usage: JAX_PLATFORMS=cpu python scripts/smoke_phase1_mps.py [name]
"""
from __future__ import annotations

import json
import os
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")

V2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, V2)

from atlas.bench.mps_cell import (  # noqa: E402
    DATA_ROOT,
    build_instances,
    load_manifest,
    oracle_mps,
    original_objective_from_meta,
)
from atlas.genome import Genome, GenomeParams, Wrapper  # noqa: E402
from atlas.schemes.compile import solve_genome  # noqa: E402
from atlas.wrappers.km import Budget  # noqa: E402

TOL = 1e-4
NAME = sys.argv[1] if len(sys.argv) > 1 else "ran10x10a"
OUT = os.path.join(V2, "results", "sprint", "phase1", "smoke_mps.json")


def main() -> int:
    t0 = time.time()
    probs = build_instances([NAME])
    assert len(probs) == 1, f"instance {NAME!r} not found/usable"
    prob = probs[0]
    meta = prob.data["meta"]
    nb = float(prob.L.norm_bound)

    genome = Genome(
        scheme="pdhg",
        params=GenomeParams(tau=0.9 / nb, sigma=0.9 / nb),
        wrapper=Wrapper(halpern=True, restart="fixed_k", restart_k=200),
    )
    res = solve_genome(prob, genome,
                       Budget(max_iters=500_000, tol=TOL, sample_every=100))
    gap = float(res.rel_gap_history[-1])

    import numpy as np
    x = np.asarray(res.x)
    c = np.asarray(prob.data["c"])
    std_obj = float(c @ x)
    orig_obj = original_objective_from_meta(meta, std_obj)

    entry = next(e for e in load_manifest() if e["name"] == NAME)
    oracle = oracle_mps(os.path.join(DATA_ROOT, entry["file"]))
    oracle_obj = float(oracle["P_star"])

    rel_err = abs(orig_obj - oracle_obj) / max(1.0, abs(oracle_obj))
    ok = gap <= TOL and rel_err <= 5e-4

    out = {
        "instance": NAME,
        "shape": {"rows": entry["rows"], "cols": entry["cols"],
                  "nnz": entry["nnz"]},
        "genome": {"scheme": "pdhg", "tau": 0.9 / nb, "sigma": 0.9 / nb,
                   "halpern": True, "restart_k": 200},
        "norm_bound": nb,
        "iters": int(res.iters),
        "rel_gap": gap,
        "objective_original_scale": orig_obj,
        "highs_oracle_objective": oracle_obj,
        "rel_objective_error": rel_err,
        "wall_s": time.time() - t0,
        "PASS": ok,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
