"""Numerical-parity probe for env_parity.json (Phase 1, 2026-09-03).

Solves the TV 64x64 PDHG default (the run_tv_cell.py baseline params) on 2
deterministic Kermany test-split instances at f64 on CPU, prints iteration
counts and final rel gaps as JSON, and dumps full trajectories to
<prefix>_i{0,1}.npz for elementwise comparison across venvs.

Usage (run once per venv, then compare the npz files elementwise):
    JAX_PLATFORMS=cpu <venv>/bin/python scripts/parity_probe.py <prefix>

Recorded reference (results/sprint/phase1/env_parity.json): iterations
1191 / 810 identical under jax 0.9.2 (jaxenv) and 0.11.1 (jaxenv2 and
mazeppa-1); max abs deviation across x, u, fp and gap histories 3.6e-16.
"""
import json
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from atlas.bench.tv_cell import build_instances
from atlas.schemes.base_schemes import PDHGParams, solve_pdhg
from atlas.wrappers.km import Budget

prefix = sys.argv[1] if len(sys.argv) > 1 else "parity"
insts = build_instances(2, size=64, lam=0.1, seed=0)
budget = Budget(max_iters=8000, tol=1e-4, sample_every=10)
out = {}
for i, prob in enumerate(insts):
    r = solve_pdhg(prob, PDHGParams(t=0.5, s=0.99 / 4.0), budget)
    out[f"inst{i}"] = dict(
        iters=int(r.iters),
        rel_gap=float(np.asarray(r.rel_gap_history)[-1]),
    )
    np.savez(
        f"{prefix}_i{i}.npz",
        x=np.asarray(r.x),
        u=np.asarray(r.u),
        fp=np.asarray(r.fp_residual_history),
        gap=np.asarray(r.rel_gap_history),
    )
print(json.dumps(out))
