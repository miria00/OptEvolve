"""Paired comparison: the pairing is what carries the signal, and overlapping per-arm ranges are not a test."""
import math

import pytest


def test_overlapping_arm_ranges_can_still_be_a_clear_paired_difference():
    """The exact point: per-arm ranges overlap almost entirely, yet every matched pair agrees."""
    from atlas.bench.paired import paired_interval
    x = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
    y = [11.0, 21.0, 31.0, 41.0, 51.0, 61.0]
    assert min(x) < min(y) < max(x) and min(y) < max(x) < max(y)      # the ranges overlap
    r = paired_interval(x, y)
    assert r["hi"] < 0 and r["verdict"] == "x faster"
    assert r["mean_diff"] == pytest.approx(-1.0) and r["sd_diff"] == pytest.approx(0.0, abs=1e-12)


def test_interval_matches_the_textbook_formula():
    from scipy import stats
    from atlas.bench.paired import paired_interval
    x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    y = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    r = paired_interval(x, y)
    mean, sd, n = 3.5, stats.tstd([1, 2, 3, 4, 5, 6]), 6
    half = float(stats.t.ppf(0.975, n - 1)) * sd / math.sqrt(n)
    assert r["n"] == 6 and r["mean_diff"] == pytest.approx(mean)
    assert r["lo"] == pytest.approx(mean - half) and r["hi"] == pytest.approx(mean + half)
    assert float(stats.t.ppf(0.975, 5)) == pytest.approx(2.5706, abs=1e-4)


def test_noisy_but_consistent_pairs_beat_a_wide_spread():
    """Run-to-run spread is large, the per-pair difference is small and consistent: still significant."""
    from atlas.bench.paired import paired_interval, paired_ratio
    y = [0.90, 1.40, 0.70, 1.90, 1.10, 1.60]
    x = [v * 0.80 for v in y]
    assert paired_interval(x, y)["hi"] < 0
    r = paired_ratio(x, y)
    assert r["geo_ratio"] == pytest.approx(1.25) and r["lo"] > 1.0 and r["verdict"] == "x faster"


def test_inconclusive_is_not_equality_and_mismatched_reps_are_refused():
    from atlas.bench.paired import paired_interval
    r = paired_interval([1.0, 2.0, 3.0, 4.0], [1.1, 1.9, 3.2, 3.8])
    assert r["lo"] < 0 < r["hi"] and r["verdict"] == "inconclusive"
    with pytest.raises(ValueError):
        paired_interval([1.0, 2.0], [1.0])
    with pytest.raises(ValueError):
        paired_interval([1.0], [1.0])
