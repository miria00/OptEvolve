"""Cell runner: run schemes/genomes over instances, emit a benchmark card.

`run_cell(genomes_or_params, instances, tol)` runs every genome on every
instance through the core solvers (Level-1 typecheck enforced inside
solve; a TypeCheckError is recorded as a certified construction failure,
not a crash) and assembles a JSON-serializable card: per-instance rows
{scheme, params, iters, wall_time_s, rel_gap_achieved, cert summary}
plus aggregates {SGM-10 time, failure rate} per genome. `save_card`
writes it as JSON.
"""
from __future__ import annotations

import dataclasses
import json
import os
import platform
import time as _time

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import numpy as np

from atlas.bench.metrics import DEFAULT_SGM_SHIFT, sgm, time_to_tol
from atlas.schemes.base_schemes import (
    CondatVuParams,
    DRSParams,
    FBSParams,
    PDHGParams,
    solve_condat_vu,
    solve_drs,
    solve_fbs,
    solve_pdhg,
)
from atlas.typing_ import TypeCheckError
from atlas.wrappers.km import Budget

SCHEMES = {
    "fbs": (FBSParams, solve_fbs),
    "drs": (DRSParams, solve_drs),
    "pdhg": (PDHGParams, solve_pdhg),
    "condat_vu": (CondatVuParams, solve_condat_vu),
}
_PARAMS_TO_SCHEME = {cls: name for name, (cls, _) in SCHEMES.items()}

DEFAULT_PDHG = PDHGParams(t=0.5, s=0.99 / 4.0)  # t*s*||D||^2 = 0.99 < 1


def _normalize_genome(g):
    """Accept a params dataclass, ('scheme', params), or {'scheme','params'}."""
    if type(g) in _PARAMS_TO_SCHEME:
        return _PARAMS_TO_SCHEME[type(g)], g
    if isinstance(g, dict):
        name = str(g["scheme"]).lower()
        if name not in SCHEMES:
            raise KeyError(f"unknown scheme {name!r} (have {sorted(SCHEMES)})")
        params = g.get("params", {})
        if isinstance(params, dict):
            params = SCHEMES[name][0](**params)
        return name, params
    if isinstance(g, (tuple, list)) and len(g) == 2:
        name = str(g[0]).lower()
        if name not in SCHEMES:
            raise KeyError(f"unknown scheme {name!r} (have {sorted(SCHEMES)})")
        params = g[1]
        if isinstance(params, dict):
            params = SCHEMES[name][0](**params)
        return name, params
    raise TypeError(f"cannot interpret genome {g!r}")


def _normalize_genomes(genomes_or_params):
    if type(genomes_or_params) in _PARAMS_TO_SCHEME or isinstance(
        genomes_or_params, (dict, tuple)
    ):
        genomes_or_params = [genomes_or_params]
    return [_normalize_genome(g) for g in genomes_or_params]


def _instance_summary(problem, i: int) -> dict:
    return {
        "index": i,
        "name": problem.name,
        "shape": list(problem.shape),
        "lam": problem.data.get("lam"),
    }


def _hardware() -> dict:
    hw = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "jax_backend": jax.default_backend(),
        "jax_version": jax.__version__,
        "x64": bool(jax.config.jax_enable_x64),
        "note": "CPU-only (JAX_PLATFORMS=cpu); GPU reserved for vLLM server",
    }
    # Full detect_device record (matmul throughput probe, a few seconds,
    # cached per process) is opt-in: set ATLAS_CARD_DEVICE=1 for cards that
    # feed the cross-hardware divergence table. The cheap identity record is
    # always included.
    try:
        if os.environ.get("ATLAS_CARD_DEVICE") == "1":
            from atlas.backend.device import detect_device

            hw["device"] = detect_device(
                prefer_gpu=(jax.default_backend() == "gpu")
            )
        else:
            dev = jax.devices()[0]
            hw["device"] = {
                "kind": dev.device_kind,
                "platform": dev.platform,
                "note": "identity only; ATLAS_CARD_DEVICE=1 for the "
                "throughput-probed record",
            }
    except Exception as e:  # a card must never fail on the probe
        hw["device_error"] = str(e)
    return hw


def run_cell(
    genomes_or_params,
    instances,
    tol: float = 1e-4,
    max_iters: int = 100_000,
    sample_every: int = 100,
    cell_name: str = "tv_oct",
) -> dict:
    """Run each genome on each instance; return the benchmark card dict.

    Rows: scheme, params, iters, wall_time_s, rel_gap_achieved,
    time_to_tol_s, converged, cert summary (Level-1 ConstructionCert; a
    violated side condition is recorded as converged=False with the
    TypeCheckError message). Aggregates per genome: SGM-10 of
    time-to-tol (failures capped at their budgeted wall time, PAR-1),
    SGM-10 iters, failure rate.
    """
    genomes = _normalize_genomes(genomes_or_params)
    budget = Budget(max_iters=max_iters, tol=tol, sample_every=sample_every)
    rows = []
    aggregates = []
    for gi, (scheme, params) in enumerate(genomes):
        _, solver = SCHEMES[scheme]
        params_dict = dataclasses.asdict(params)
        g_times, g_iters, g_fail = [], [], 0
        for ii, problem in enumerate(instances):
            row = {
                "genome_index": gi,
                "scheme": scheme,
                "params": params_dict,
                "instance": _instance_summary(problem, ii),
            }
            try:
                result = solver(problem, params, budget)
            except TypeCheckError as e:
                row.update(
                    {
                        "iters": 0,
                        "wall_time_s": 0.0,
                        "rel_gap_achieved": None,
                        "time_to_tol_s": None,
                        "converged": False,
                        "cert": None,
                        "error": f"TypeCheckError: {e}",
                    }
                )
                rows.append(row)
                g_fail += 1
                # Construction failure: the genome is invalid on this cell;
                # poison the SGM (inf) rather than reward zero time.
                g_times.append(float("inf"))
                g_iters.append(0)
                continue
            rel_gap_achieved = float(result.rel_gap_history[-1])
            converged = rel_gap_achieved <= tol
            t_tol, iter_at = time_to_tol(result, tol, sample_every=sample_every)
            row.update(
                {
                    "iters": int(result.iters),
                    "wall_time_s": float(result.wall_time),
                    "rel_gap_achieved": rel_gap_achieved,
                    "final_fp_residual": float(result.fp_residual_history[-1]),
                    "time_to_tol_s": None if not converged else float(t_tol),
                    "iters_to_tol": iter_at,
                    "converged": bool(converged),
                    "cert": result.cert.to_dict() if result.cert is not None else None,
                }
            )
            rows.append(row)
            if converged:
                g_times.append(float(t_tol))
            else:
                g_fail += 1
                g_times.append(float(result.wall_time))  # PAR-1 cap
            g_iters.append(int(result.iters))
        aggregates.append(
            {
                "genome_index": gi,
                "scheme": scheme,
                "params": params_dict,
                "n_instances": len(instances),
                "sgm10_time_s": sgm(g_times, DEFAULT_SGM_SHIFT),
                "sgm10_iters": sgm(g_iters, DEFAULT_SGM_SHIFT),
                "failure_rate": g_fail / max(len(instances), 1),
            }
        )
    card = {
        "cell": cell_name,
        "tolerance_rel_gap": tol,
        "budget": {"max_iters": max_iters, "sample_every": sample_every},
        "n_genomes": len(genomes),
        "n_instances": len(instances),
        "rows": rows,
        "aggregates": aggregates,
        "hardware": _hardware(),
        "timestamp": _time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    json.dumps(card)  # fail fast if not serializable
    return card


def sanitize_json(obj):
    """Recursively replace non-finite floats (inf/-inf/nan) with the strings
    "inf"/"-inf"/"nan" so the result is strict RFC-8259 JSON. Python's
    json.dump would otherwise emit bare Infinity/NaN tokens that JSON.parse
    and most non-Python tooling reject."""
    if isinstance(obj, float):
        if obj != obj:
            return "nan"
        if obj == float("inf"):
            return "inf"
        if obj == float("-inf"):
            return "-inf"
        return obj
    if isinstance(obj, dict):
        return {k: sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_json(v) for v in obj]
    return obj


def save_card(card: dict, path: str) -> str:
    """Write a benchmark card as pretty, STRICT (RFC-8259) JSON; non-finite
    floats are stringified. Returns the absolute path."""
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(sanitize_json(card), fh, indent=2, allow_nan=False)
        fh.write("\n")
    return path
