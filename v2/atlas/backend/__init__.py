"""atlas.backend: hardware detection, precision policies, per-atom kernel
checks (the bounded Hawkeye scope; see CONTRACT.md in this directory).

Invariant: backend genes change how T is computed, never what T is.
Certificates are evaluated at exact float64, always.
"""
from atlas.backend.batched import (
    BatchedSolveResult,
    batched_compile_count,
    batched_solve,
    clear_batched_cache,
)
from atlas.backend.device import detect_device
from atlas.backend.precision import (
    POLICIES,
    F32Cert64Policy,
    F64Policy,
    PolicyResult,
    PrecisionError,
    SchedulePolicy,
    certified_gap,
    get_policy,
    promote64,
)
from atlas.backend.unit_tests import (
    REGISTRY,
    CheckReport,
    check,
    check_linop,
    default_impls,
    run_all,
    run_check,
)

__all__ = [
    "BatchedSolveResult",
    "batched_solve",
    "batched_compile_count",
    "clear_batched_cache",
    "detect_device",
    "POLICIES",
    "F64Policy",
    "F32Cert64Policy",
    "SchedulePolicy",
    "PolicyResult",
    "PrecisionError",
    "certified_gap",
    "get_policy",
    "promote64",
    "REGISTRY",
    "CheckReport",
    "check",
    "check_linop",
    "default_impls",
    "run_all",
    "run_check",
]
