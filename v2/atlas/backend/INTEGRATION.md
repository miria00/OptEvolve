# Batched vmap backend: integration contract

Owner: batched-backend lane. Audience: the integrator wiring
`atlas/schemes/compile.py::solve_genome` (and bench runners) to the
batched path. This file is the contract; the implementation is
`atlas/backend/batched.py`. Nothing in `atlas/genome` or
`atlas/schemes/compile.py` was touched by this lane.

## Entry point

```python
from atlas.backend import batched_solve, BatchedSolveResult

res = batched_solve(problems, params_or_genome, budget,
                    precision_policy=None, K=None)
```

- `problems`: list of `CompositeProblem`s, ALL the same kind and shapes.
  Supported kinds: `tv_denoise` (same `(H, W)`; `lam` may differ per
  lane) and `lp_standard` (same `(m, n)`; netflow incidence instances
  additionally need the same arc and node counts, dense instances the
  same `A` shape; mixed LinOp families are refused). Any mismatch raises
  `ValueError` mentioning "same-shape contract".
- `params_or_genome`: either `PDHGParams` / `CondatVuParams`, or a
  Genome, read DUCK-TYPED via the current public attribute names
  (`scheme`, `params.tau/.sigma/.rho`, `wrapper.halpern/.restart_k`,
  `mix`, `backend.precision/.K`). If the genome schema renames these,
  `atlas/backend/batched.py::_extract_plan` is the ONE function to
  update. Mix genomes (avg and switch_k alike) raise
  `NotImplementedError` (switch_k's prefix handoff and finiteness checks
  are host-side per-lane control flow); metric (rescale) genomes raise
  `NotImplementedError` (the Ruiz/Pock-Chambolle diagonals are
  per-instance data with per-lane norm bounds); schemes other than
  `pdhg` / `condat_vu` raise `NotImplementedError`. All refusals are
  loud; the caller falls back to the sequential `solve_genome` path.
- `budget`: `atlas.wrappers.km.Budget`. `sample_every` is ignored; the
  certificate cadence is `K`. Like the precision-policy path, the last
  chunk may overshoot `max_iters` by up to `K - 1` iterations.
- `precision_policy`: `"f64"` | `"f32_cert64"` | `None` (None: the
  genome's backend gene if present, else `"f64"`). `"schedule"` raises
  `NotImplementedError` (its f32 to f64 switch is a per-lane event with
  no sound lockstep analogue yet); sub-f32 names raise `PrecisionError`.
- `K`: certificate cadence in iterations (None: genome `backend.K`,
  else 100).

## Suggested solve_genome routing (integrator's side)

A genome with `backend.batch == true` evaluated on a list of same-shape
instances should route to `batched_solve(instances, genome, budget)`
instead of a per-instance `solve_genome` loop, PROVIDED the genome is
non-mix, its scheme is in `atlas.backend.batched.SUPPORTED_SCHEMES`, and
its precision is in `SUPPORTED_PRECISIONS`. On any of the refusals above
fall back to the sequential path and record the reason; the refusals are
loud exceptions, never silent fallbacks inside `batched_solve` itself.

## Semantics guarantees (what the tests pin)

- Per-lane arithmetic is the sequential path's, bit-for-bit: the lane
  step constructs the SAME atom objects (`QuadraticDataFidelity`,
  `ScaledL1`, `LinearCost`, `NonnegOrthant`, `AffineEqualitySet`) and
  calls them in the order of `base_schemes`' `T` closures; wrapper genes
  (rho relaxation, Halpern anchor + fixed-k restart) use the identical
  update algebra with a per-lane anchor and counter. Equivalence is
  enforced at 1e-10 (f64, fixed iteration count) in
  `tests/test_batched_backend.py`, and the vmapped atom kernels each pass
  the Hawkeye registry checks (`batched_*` entries in
  `atlas/backend/unit_tests.py`).
- Per-instance stopping is by MASKING: the batch iterates in lockstep
  and a converged lane is frozen with `jnp.where` on a done-mask.
  Frozen lanes still burn FLOPs until the last lane finishes or the cap;
  that is the standard lockstep tradeoff and is deliberate (compacting
  the batch would recompile per survivor set).
- Certificates: the per-instance duality gap is evaluated on the batch
  at exact float64 every `K` iterations (iterates promoted; a separate
  f64 copy of the problem data is used). A lane terminates ONLY on its
  own f64 gap. Level-1 construction certificate: checked once per solve
  against the WORST operator norm in the batch, so acceptance covers
  every lane; violations raise `TypeCheckError` (no clamping).
- `iters[i]` has cadence-`K` granularity (first check at which lane i's
  gap was <= tol); `batch_iters` is what every lane physically executed.

## Timing convention (the factorial's backend-arm contract)

- `batch_wall_s`: ONE wall clock for the whole batch, covering compiled
  chunk execution plus the f64 certificate evaluations, EXCLUDING AOT
  compile (`compile_time_s` is recorded separately; the chunk and gap
  executables are cached across solves keyed by scheme/wrapper statics,
  K, and abstract shapes/dtypes, so repeat solves pay zero compile).
- `per_instance_effective_s = batch_wall_s / n_instances`: a DERIVED
  convenience label, not a measurement. The hardware factorial compares
  backends on `batch_wall_s` at matched instance sets; report both, never
  mix them in one column.

## Cache / test hooks

`atlas.backend.batched_compile_count()` and `clear_batched_cache()`
mirror the km-path hooks for compile-count assertions.
