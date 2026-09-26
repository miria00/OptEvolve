"""Paired comparison of two arms measured on the SAME repetitions.

Comparing per-arm medians throws away the pairing, and overlapping per-arm ranges are not a
significance test: two arms can overlap almost completely and still differ on every matched pair.
The paired standard deviation cannot be recovered from the two per-arm standard deviations, because
it depends on how the timings covary, so the differences themselves have to be kept.
"""
from __future__ import annotations

import math
import statistics


def _tcrit(df: int, conf: float) -> float:
    from scipy import stats
    return float(stats.t.ppf(0.5 * (1.0 + conf), df))


def paired_interval(x, y, conf: float = 0.95) -> dict:
    """Two-sided t interval for the mean of d_r = x_r - y_r over matched repetitions.

    Negative values favour x. An interval entirely below zero supports a lower mean for x under this
    design; an interval containing zero is INCONCLUSIVE, not evidence that the arms are equal.
    """
    x, y = list(map(float, x)), list(map(float, y))
    if len(x) != len(y):
        raise ValueError(f"paired arms need matched repetitions: {len(x)} vs {len(y)}")
    if len(x) < 2:
        raise ValueError("a paired interval needs at least two repetitions")
    d = [xi - yi for xi, yi in zip(x, y)]
    n = len(d)
    mean, sd = statistics.fmean(d), statistics.stdev(d)
    half = _tcrit(n - 1, conf) * sd / math.sqrt(n)
    lo, hi = mean - half, mean + half
    return {"n": n, "mean_diff": mean, "sd_diff": sd, "lo": lo, "hi": hi, "conf": conf,
            "verdict": ("x faster" if hi < 0 else "y faster" if lo > 0 else "inconclusive")}


def paired_ratio(x, y, conf: float = 0.95) -> dict:
    """Geometric-mean ratio y/x (a speedup of x over y) with a t interval on the log differences.

    This is the paired form of the geometric-mean speedup we quote: the interval is exact for the
    geometric mean, and excludes 1.0 exactly when the paired log difference excludes 0.
    """
    if any(v <= 0 for v in list(x) + list(y)):
        raise ValueError("ratios need strictly positive times")
    r = paired_interval([math.log(v) for v in y], [math.log(v) for v in x], conf)
    return {"n": r["n"], "geo_ratio": math.exp(r["mean_diff"]), "lo": math.exp(r["lo"]),
            "hi": math.exp(r["hi"]), "conf": conf,
            "verdict": ("x faster" if r["lo"] > 0 else "y faster" if r["hi"] < 0 else "inconclusive")}
