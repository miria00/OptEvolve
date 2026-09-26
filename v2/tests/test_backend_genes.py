"""Integration: the backend gene block (atlas/backend/CONTRACT.md).

Stage-0 validates the block STRUCTURALLY (it never enters the math
typechecks); the solve path routes a present block through the
atlas.backend precision policies; an absent block is the exact P0 km_run
path. Certificates are always f64.
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pytest

from atlas.genome import (
    Backend,
    Genome,
    GenomeParams,
    content_hash,
    genome_from_dict,
    genome_to_dict,
    validate,
)
from atlas.ir.spec import tv_denoise_problem
from atlas.schemes.compile import solve_genome
from atlas.typing_ import TypeCheckError
from atlas.wrappers.km import Budget

_BASE = {"scheme": "condat_vu", "params": {"tau": 0.5, "sigma": 0.186}}


@pytest.fixture(scope="module")
def tv32():
    y = np.random.default_rng(0).uniform(0.0, 1.0, (32, 32))
    return tv_denoise_problem(y, lam=0.1)


def _g(backend=None, **extra):
    d = dict(_BASE, **extra)
    if backend is not None:
        d["backend"] = backend
    return genome_from_dict(d)


# ---------------------------------------------------------------------------
# Stage-0: structural validation, math checks untouched
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "backend",
    [
        {"precision": "f64", "K": 100},
        {"precision": "f32_cert64", "K": 1},
        {"precision": "f32_cert64", "K": 10000, "batch": True},
        {"precision": "schedule", "K": 50, "theta": 0.01},
    ],
)
def test_valid_backend_blocks_pass(backend):
    v = validate(_g(backend))
    assert v.ok, v.error
    # the cert is the SAME as without the block: backend never enters math
    v0 = validate(_g())
    assert v.cert.to_dict() == v0.cert.to_dict()


@pytest.mark.parametrize(
    "backend,frag",
    [
        ({"precision": "bf16"}, "cancellation"),
        ({"precision": "bfloat16"}, "cancellation"),
        ({"precision": "f16"}, "cancellation"),
        ({"precision": "float16"}, "cancellation"),
        ({"precision": "half"}, "cancellation"),
        ({"precision": "f32"}, "must be one of"),
        ({"precision": "f64", "K": 0}, "backend.K"),
        ({"precision": "f64", "K": 10001}, "backend.K"),
        ({"precision": "schedule", "K": 10}, "backend.theta"),
        ({"precision": "schedule", "K": 10, "theta": 0.0}, "backend.theta"),
        ({"precision": "schedule", "K": 10, "theta": 1.0}, "backend.theta"),
        ({"precision": "f64", "K": 10, "theta": 0.5}, "only to precision"),
    ],
)
def test_invalid_backend_blocks_rejected(backend, frag):
    v = validate(_g(backend))
    assert not v.ok
    assert frag in v.error


def test_switch_genome_takes_only_f64_backend():
    d = {
        "scheme": "mix",
        "mix": {
            "kind": "switch",
            "window": 25,
            "aggressive": {"scheme": "pdhg", "params": {"tau": 5.0, "sigma": 5.0}},
            "fallback": {"scheme": "pdhg", "params": {"tau": 0.5, "sigma": 0.24}},
        },
    }
    assert validate(genome_from_dict(d)).ok
    assert validate(
        genome_from_dict({**d, "backend": {"precision": "f64", "K": 100}})
    ).ok
    v = validate(
        genome_from_dict({**d, "backend": {"precision": "f32_cert64", "K": 100}})
    )
    assert not v.ok and "f64 backend" in v.error


def test_backend_fuzz_never_crashes_the_gate():
    """CONTRACT TODO 1: fuzz the gate over the backend block; validate()
    must return ok/error, never raise, and never accept sub-f32 names."""
    rng = np.random.default_rng(42)
    precisions = [
        "f64", "f32_cert64", "schedule", "bf16", "bfloat16", "f16",
        "float16", "half", "f32", "float64", "", "F64",
    ]
    n_ok = n_rej = 0
    for _ in range(2000):
        b = Backend(
            precision=str(rng.choice(precisions)),
            K=int(rng.integers(-5, 20000)),
            theta=(
                None
                if rng.uniform() < 0.3
                else float(rng.choice([0.0, 1.0, 1e-3, 0.5, np.inf, -0.1]))
            ),
            batch=bool(rng.uniform() < 0.5),
        )
        g = Genome(
            scheme="condat_vu",
            params=GenomeParams(tau=0.5, sigma=0.186),
            backend=b,
        )
        v = validate(g)
        if v.ok:
            n_ok += 1
            assert b.precision in ("f64", "f32_cert64", "schedule")
            assert 1 <= b.K <= 10000
        else:
            n_rej += 1
    assert n_ok > 0 and n_rej > 0


# ---------------------------------------------------------------------------
# Canonical form / hashing
# ---------------------------------------------------------------------------


def test_absent_block_hashes_differently_from_explicit_f64():
    g0 = _g()
    g1 = _g({"precision": "f64", "K": 100})
    assert content_hash(g0) != content_hash(g1)


def test_backend_roundtrips_through_dict():
    g = _g({"precision": "schedule", "K": 50, "theta": 0.01, "batch": True})
    g2 = genome_from_dict(genome_to_dict(g))
    assert content_hash(g) == content_hash(g2)
    assert g2.backend == g.backend


def test_theta_nulled_in_canonical_for_non_schedule():
    # theta is only meaningful for schedule; f64/f32_cert64 with theta are
    # REJECTED by the gate, so canonicalization never sees them valid; the
    # canonical dict still nulls it defensively.
    from atlas.genome import canonical_dict

    g = _g({"precision": "f32_cert64", "K": 100})
    assert canonical_dict(g)["backend"]["theta"] is None


# ---------------------------------------------------------------------------
# Solve-path wiring
# ---------------------------------------------------------------------------

_BUDGET = Budget(max_iters=20000, tol=1e-4, sample_every=100)


def test_f32_cert64_reaches_tol_with_f64_certificate(tv32):
    r = solve_genome(tv32, _g({"precision": "f32_cert64", "K": 50}), _BUDGET)
    assert float(r.rel_gap_history[-1]) <= 1e-4
    bk = r.extra["backend"]
    assert bk["iterate_dtypes"] == ["float32"]
    assert bk["certificate_dtype"] == "float64"
    assert r.x.dtype == np.float64  # final state promoted


def test_schedule_switches_and_reaches_tol(tv32):
    r = solve_genome(
        tv32, _g({"precision": "schedule", "K": 25, "theta": 0.01}), _BUDGET
    )
    assert float(r.rel_gap_history[-1]) <= 1e-4
    bk = r.extra["backend"]
    assert bk["iterate_dtypes"][0] == "float32"


def test_explicit_f64_matches_absent_block_solution(tv32):
    r0 = solve_genome(tv32, _g(), _BUDGET)
    r1 = solve_genome(tv32, _g({"precision": "f64", "K": 100}), _BUDGET)
    assert float(r1.rel_gap_history[-1]) <= 1e-4
    # Same operator T, but the policy certifies every K iterations while the
    # km path checks every iteration, so the two stop at DIFFERENT iterates
    # inside the tol ball. f is 1-strongly convex, so ||x - x*||^2 <= 2*gap;
    # both iterates lie within that radius of the same optimum.
    from atlas.certify.residuals import tv_primal

    P0 = float(tv_primal(r0.x, tv32.data["y"], tv32.data["lam"], tv32.L))
    gap_abs = 1e-4 * (abs(P0) + 1.0)
    radius = np.sqrt(2.0 * gap_abs)
    assert np.linalg.norm(r1.x - r0.x) <= 2.0 * radius
    assert r0.extra is None  # absent block = exact P0 path, no policy record


def test_wrappers_combine_inside_step_under_policy(tv32):
    g = genome_from_dict(
        {
            "scheme": "condat_vu",
            "params": {"tau": 0.5, "sigma": 0.186},
            "wrapper": {"halpern": True, "restart": "fixed_k", "restart_k": 50},
            "backend": {"precision": "f32_cert64", "K": 50},
        }
    )
    assert validate(g).ok
    r = solve_genome(tv32, g, _BUDGET)
    assert float(r.rel_gap_history[-1]) <= 1e-4


def test_schedule_theta_must_exceed_budget_tol(tv32):
    g = _g({"precision": "schedule", "K": 25, "theta": 1e-3})
    tight = Budget(max_iters=100, tol=1e-3, sample_every=10)
    with pytest.raises(TypeCheckError):
        solve_genome(tv32, g, tight)
