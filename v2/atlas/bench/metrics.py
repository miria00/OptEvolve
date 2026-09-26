"""Benchmark metrics: time-to-tol, shifted geometric mean, speedup.

Conventions (standard in the optimization-benchmarking literature,
cf. Mittelmann benchmarks): a solve that never reaches tol gets
time-to-tol = inf; aggregate with the shifted geometric mean SGM-10
(shift = 10 s) after replacing failures with a cap (PAR-1: the budgeted
wall time), and report the failure rate separately.
"""
from __future__ import annotations

import numpy as np

DEFAULT_SGM_SHIFT = 10.0


def time_to_tol(result_or_history, tol: float, *, iters=None, wall_time_s=None,
                sample_every: int = 100):
    """Estimated wall seconds to first reach rel_gap <= tol.

    Accepts either a SolveResult-like object (with .rel_gap_history,
    .iters, .wall_time) or a raw history array plus explicit `iters`,
    `wall_time_s`, `sample_every`. The history is sampled every
    `sample_every` iterations (bucket-final, core km_run convention), so
    the crossing iteration is bracketed at (i+1)*sample_every and time
    is prorated from the measured wall time; no re-run needed.

    PRECISION-POLICY CADENCE (fixed 2026-08-26, sprint finding): a result
    from the backend precision path writes ONE history entry per backend.K
    iterations, not per sample_every; its true evaluation iterations are
    recorded in result.extra["backend"]["gap_iters"]. When that record is
    present and consistent with the history length it is authoritative;
    assuming sample_every cadence understated time-to-tol by a factor
    K/sample_every ("free certificates" for backend genomes).

    Returns (t_seconds, iter_at). If tol is never reached: (inf, None).
    """
    gap_iters = None
    if hasattr(result_or_history, "rel_gap_history"):
        history = np.asarray(result_or_history.rel_gap_history, dtype=float)
        iters = int(result_or_history.iters)
        wall_time_s = float(result_or_history.wall_time)
        extra = getattr(result_or_history, "extra", None)
        if isinstance(extra, dict):
            backend = extra.get("backend")
            if isinstance(backend, dict):
                gi = backend.get("gap_iters")
                if gi is not None and len(gi) == history.size:
                    gap_iters = np.asarray(gi, dtype=np.int64)
    else:
        history = np.asarray(result_or_history, dtype=float)
        if iters is None or wall_time_s is None:
            raise ValueError("raw-history form needs iters= and wall_time_s=")
    with np.errstate(invalid="ignore"):
        hit = np.nonzero(history <= tol)[0]  # NaN compares False
    if hit.size == 0:
        return float("inf"), None
    if gap_iters is not None:
        iter_at = min(int(gap_iters[int(hit[0])]), iters)
    else:
        iter_at = min((int(hit[0]) + 1) * sample_every, iters)
    t = wall_time_s * iter_at / max(iters, 1)
    return float(t), int(iter_at)


def sgm(times, shift: float = DEFAULT_SGM_SHIFT) -> float:
    """Shifted geometric mean: exp(mean(log(t + shift))) - shift.

    SGM-10 is sgm(times, shift=10.0) with times in seconds. Any inf
    (unreplaced failure) makes the result inf; empty input gives nan.
    """
    t = np.asarray(list(times), dtype=float)
    if t.size == 0:
        return float("nan")
    if np.any(np.isnan(t)) or np.any(t < 0):
        raise ValueError("times must be nonnegative and non-NaN")
    if np.any(np.isinf(t)):
        return float("inf")
    return float(np.exp(np.mean(np.log(t + shift))) - shift)


def speedup_vs_baseline(times, baseline_times, shift: float = DEFAULT_SGM_SHIFT) -> float:
    """SGM speedup of `times` over `baseline_times`: sgm(base)/sgm(times).

    > 1 means `times` is faster. inf-failures propagate: a failing
    candidate gives 0.0-ish (sgm=inf in the denominator), a failing
    baseline gives inf.
    """
    s_new = sgm(times, shift)
    s_base = sgm(baseline_times, shift)
    if np.isinf(s_new):
        return 0.0
    if s_new <= 0 and s_base <= 0:
        return 1.0
    if s_new <= 0:
        return float("inf")
    return float(s_base / s_new)
