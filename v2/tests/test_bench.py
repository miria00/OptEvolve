"""Tests for the benchmark/data wing (atlas.bench) on the REAL OCT cache."""
import json

import numpy as np
import pytest

from atlas.bench.kermany import load_patches
from atlas.bench.metrics import sgm, speedup_vs_baseline, time_to_tol
from atlas.bench.runner import DEFAULT_PDHG, run_cell, save_card
from atlas.bench.tv_cell import build_instances, oracle


# ---------------------------------------------------------------------------
# Kermany loader (real local cache)
# ---------------------------------------------------------------------------


def test_load_patches_shapes_dtype_range():
    patches = load_patches(4, size=64, split="test", seed=123)
    assert patches.shape == (4, 64, 64)
    assert patches.dtype == np.float64
    assert patches.min() >= 0.0 and patches.max() <= 1.0
    # real scans, not blanks: every patch has structure
    assert all(p.std() > 1e-3 for p in patches)


def test_load_patches_deterministic_by_seed():
    a = load_patches(3, size=32, seed=7)
    b = load_patches(3, size=32, seed=7)
    c = load_patches(3, size=32, seed=8)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_load_patches_random_crop_seeded():
    a = load_patches(2, size=32, seed=5, crop="random")
    b = load_patches(2, size=32, seed=5, crop="random")
    np.testing.assert_array_equal(a, b)


def test_load_patches_meta():
    patches, meta = load_patches(2, size=32, seed=0, return_meta=True)
    assert len(meta) == 2
    assert meta[0]["label_name"] in {"CNV", "DME", "DRUSEN", "NORMAL"}
    assert meta[0]["orig_height"] >= 32 and meta[0]["orig_width"] >= 32


# ---------------------------------------------------------------------------
# Oracle vs CVXPY on a real 32x32 patch
# ---------------------------------------------------------------------------


def test_oracle_matches_cvxpy_on_32x32(tmp_path):
    import cvxpy as cp

    lam = 0.1
    [problem] = build_instances(1, size=32, lam=lam, seed=3)
    ref = oracle(problem, cache_dir=str(tmp_path))
    assert ref["rel_gap"] <= 1e-9

    y = np.asarray(problem.data["y"])
    x = cp.Variable(y.shape)
    tv_aniso = cp.sum(cp.abs(x[1:, :] - x[:-1, :])) + cp.sum(cp.abs(x[:, 1:] - x[:, :-1]))
    obj = cp.Minimize(0.5 * cp.sum_squares(x - y) + lam * tv_aniso)
    cvx = cp.Problem(obj)
    cvx.solve(solver=cp.CLARABEL)
    assert cvx.status == cp.OPTIMAL
    assert abs(ref["P_star"] - cvx.value) <= 1e-5

    # cache round-trip: second call hits the cache, identical values
    ref2 = oracle(problem, cache_dir=str(tmp_path))
    assert ref2["cache_hit"] is True
    assert ref2["P_star"] == ref["P_star"]
    np.testing.assert_array_equal(ref2["x_star"], ref["x_star"])


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def test_time_to_tol_from_history():
    hist = np.array([1e-1, 1e-3, 1e-5, 1e-7])
    t, it = time_to_tol(hist, 1e-4, iters=400, wall_time_s=4.0, sample_every=100)
    assert it == 300 and t == pytest.approx(3.0)
    t_inf, it_none = time_to_tol(hist, 1e-9, iters=400, wall_time_s=4.0, sample_every=100)
    assert t_inf == float("inf") and it_none is None


def test_sgm_and_speedup():
    assert sgm([5.0, 5.0, 5.0]) == pytest.approx(5.0)
    assert sgm([1.0, float("inf")]) == float("inf")
    # sgm of a constant list is the constant, so the speedup is the ratio
    assert speedup_vs_baseline([1.0, 1.0], [3.0, 3.0]) == pytest.approx(3.0)
    # mixed lists: SGM-10 computed explicitly
    expected = (np.exp(np.mean(np.log(np.array([2.0, 8.0]) + 10.0))) - 10.0) / (
        np.exp(np.mean(np.log(np.array([1.0, 4.0]) + 10.0))) - 10.0
    )
    assert speedup_vs_baseline([1.0, 4.0], [2.0, 8.0]) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Runner: well-formed card on 2 tiny real instances
# ---------------------------------------------------------------------------


def test_run_cell_card_well_formed(tmp_path):
    instances = build_instances(2, size=32, lam=0.1, seed=11)
    card = run_cell(DEFAULT_PDHG, instances, tol=1e-4)

    assert card["n_genomes"] == 1 and card["n_instances"] == 2
    assert len(card["rows"]) == 2
    for row in card["rows"]:
        assert row["scheme"] == "pdhg"
        assert row["params"]["t"] == DEFAULT_PDHG.t
        assert row["converged"] is True
        assert row["rel_gap_achieved"] <= 1e-4
        assert row["iters"] > 0 and row["wall_time_s"] > 0
        cert = row["cert"]
        assert cert["satisfied"] is True and cert["scheme"] == "PDHG"
    agg = card["aggregates"][0]
    assert agg["failure_rate"] == 0.0
    assert np.isfinite(agg["sgm10_time_s"]) and agg["sgm10_time_s"] > 0
    assert card["hardware"]["jax_backend"] == "cpu"

    path = save_card(card, str(tmp_path / "card.json"))
    with open(path) as fh:
        loaded = json.load(fh)
    assert loaded["cell"] == card["cell"]
    assert len(loaded["rows"]) == 2


def test_run_cell_records_typecheck_failure_as_row():
    instances = build_instances(1, size=32, lam=0.1, seed=11)
    # t*s*||L||^2 = 8 > 1: violates the PDHG side condition
    card = run_cell({"scheme": "pdhg", "params": {"t": 1.0, "s": 1.0}}, instances)
    row = card["rows"][0]
    assert row["converged"] is False
    assert "TypeCheckError" in row["error"]
    assert card["aggregates"][0]["failure_rate"] == 1.0
    assert card["aggregates"][0]["sgm10_time_s"] == float("inf")


def test_run_cell_runs_rho_plus_halpern():
    """relax-then-anchor is admitted as of 2026-09-05, so this genome must
    RUN rather than being recorded as a construction failure. (A genuine
    construction failure is still recorded as a row, not a crash: see
    test_run_cell_records_construction_failure_as_row.)"""
    instances = build_instances(1, size=32, lam=0.1, seed=11)
    card = run_cell(
        {"scheme": "drs", "params": {"step": 1.0, "rho": 1.5, "halpern": True}},
        instances,
    )
    row = card["rows"][0]
    assert row.get("error") is None, row.get("error")


def test_run_cell_records_construction_failure_as_row():
    """A real side-condition violation is still a row, not a crash."""
    instances = build_instances(1, size=32, lam=0.1, seed=11)
    card = run_cell(
        {"scheme": "pdhg", "params": {"t": 0.5, "s": 0.5}}, instances,
    )
    row = card["rows"][0]
    assert row["converged"] is False
    assert "TypeCheckError" in row["error"]
    assert card["aggregates"][0]["failure_rate"] == 1.0


def test_run_cell_records_drs_inf_step_as_row():
    """Regression: DRS step=inf used to pass the Level-1 gate and run,
    emitting a garbage row with a satisfied cert."""
    instances = build_instances(1, size=32, lam=0.1, seed=11)
    card = run_cell({"scheme": "drs", "params": {"step": float("inf")}}, instances)
    row = card["rows"][0]
    assert row["converged"] is False
    assert "TypeCheckError" in row["error"]


def test_save_card_emits_strict_rfc8259_json(tmp_path):
    """Regression: json.dump of inf aggregates used to emit bare Infinity
    tokens that strict parsers (JSON.parse etc.) reject."""
    instances = build_instances(1, size=32, lam=0.1, seed=11)
    card = run_cell({"scheme": "pdhg", "params": {"t": 1.0, "s": 1.0}}, instances)
    assert card["aggregates"][0]["sgm10_time_s"] == float("inf")  # in-memory
    path = save_card(card, str(tmp_path / "card.json"))
    with open(path) as fh:
        text = fh.read()

    def _reject(token):  # called for Infinity/-Infinity/NaN only
        raise AssertionError(f"non-strict JSON token {token!r} in saved card")

    loaded = json.loads(text, parse_constant=_reject)
    assert loaded["aggregates"][0]["sgm10_time_s"] == "inf"  # stringified
