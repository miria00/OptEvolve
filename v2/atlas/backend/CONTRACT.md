# Backend gene contract (atlas.backend, bounded Hawkeye scope)

CURRENT UPDATE (2026-09-06): explicit f64/f32 policies use a compiled device
loop with K-step certificate cadence and an exact iteration cap. Schedule
policies reserve a bounded f64 refinement phase. Relaxation precedes the
Halpern anchor on both paths. The absent backend still checks every iteration;
it is not equivalent to explicit K=100. See `../../docs/HARDWARE_WORKFLOW.md`
for measured hardware context, profiling, and the diagnostic protocol.
The generated LP CSR update templates now use the `lp_csr_update` registry
check (eight independent NumPy-reference trials), followed by per-instance
reference-map admission. Non-finite outputs fail admission. `backend.kernel`
selects `jax` (default), `csr_thread128`, `csr_warp128`, or `csr_warp256`.
The CUDA alternatives currently require standard LP PDHG and f64. Default
`jax` is omitted from serialization to preserve historical genome hashes.
The historical integration notes below predate this implementation.

INTEGRATION STATUS (2026-08-26, integrator): TODOs 1-6 executed. The
`backend` block is in atlas/genome/schema.py (structural Stage-0 rules,
canonical JSON + hash, absent != explicit f64; switch genomes accept only
f64), the solve path routes present blocks through get_policy via
atlas.schemes.base_schemes._run_built_policy (wrappers combine inside
step_factory; absent block = exact P0 km_run path), RandomBackend draws
backend genes (p_backend), the LLM mutation menu documents them and
LLMBackend takes a `hardware=detect_device()` context record, and the
run_cell card carries a device record (full probe opt-in via
ATLAS_CARD_DEVICE=1). Tests: tests/test_backend_genes.py. TODO 7 (Stage-0.5
kernel gate) remains a hook only: no alternative kernel implementations are
registered in this build, run_check is the mandatory gate when one lands.
TODO 8 (batch) validates + records, no-op as specified. Policy-path wall
times include compile (SolveResult.extra["wall_includes_compile"]).

Status: backend package built and tested standalone. This document is the
integration contract for the agent that owns atlas/genome, atlas/schemes and
atlas/evolve, which are being rewritten in parallel and are deliberately NOT
imported by anything in atlas/backend except the stable P0 modules
(atlas.ir, atlas.operators, atlas.certify).

## 1. Genome schema extension

The genome gains one optional top-level block:

```json
{
  "backend": {
    "precision": "f64" | "f32_cert64" | "schedule",
    "K": 100,
    "theta": 0.001,
    "batch": false
  }
}
```

Field semantics and Stage-0 validation rules (validate() rejects with the
violated condition string, never clamps, matching the existing gate style):

| field | meaning | Stage-0 rule |
|---|---|---|
| precision | which execution policy compiles the solve | must be one of `f64`, `f32_cert64`, `schedule`; `bf16`/`f16` in any spelling is rejected with the cancellation explanation (see section 3); unknown strings rejected |
| K | certificate cadence: the f64 duality gap is evaluated every K iterations | integer, `1 <= K <= 10000`; finite |
| theta | schedule only: switch f32 -> f64 when the f64 gap `<= theta` (a plateau of the f64 gap also switches) | float, `0 < theta < 1`, finite; must be `> tol` of the budget or the schedule degenerates; for `f64` and `f32_cert64` the field must be absent or null |
| batch | reserved for the vmap population backend (P4) | bool; `true` must validate but is currently a no-op (record it in the card as `"batch": "declared, inactive"`) |

When the block is absent the legacy f64 path checks every iteration. An
explicit block defaults to K=100 and therefore has a different stopping grid.

Canonical JSON / content hash: the backend block participates in canonical
serialization like every other gene (sorted keys, absent-vs-default NOT
normalized away, so a genome that says f64 explicitly hashes differently
from one that omits the block; keep the existing convention, whichever the
rewrite chooses, consistent between validate() and hash).

## 2. Runtime contract (what the backend provides)

`atlas.backend.precision.get_policy(name, **params)` returns a policy object
with

```python
policy.run(step_factory, init_factory, gap64, tol, max_iters) -> PolicyResult
```

- `step_factory(dtype) -> step(state, k) -> state`: the scheme's fixed-point
  map T with ALL constants (y, lam, t, s, rho, ...) cast to `dtype`. This is
  the one thing the scheme rewrite must expose per scheme. The P0 solve_*
  functions already build exactly this closure internally; factoring it out
  is the only structural ask.
- `init_factory(dtype) -> state0`: initial iterate pytree in `dtype`.
- `gap64(state) -> scalar`: the rigorous duality gap on an f64 state, e.g.
  `atlas.certify.residuals.rel_gap` closed over f64 problem data. The policy
  promotes the iterate to f64 before calling it and raises `PrecisionError`
  if the returned scalar is not float64.
- `PolicyResult` carries: final f64 state, iters, converged, certified_gap
  (np.float64), gap_history and gap_iters (per certificate evaluation),
  switch_iter (schedule only), iterate_dtypes, certificate_dtype ("float64",
  always).

Termination is decided ONLY by the f64 certified gap. Wall-clock timing of a
policy run should follow the km_run convention: the chunk runners are jitted
once per (phase, dtype) and the first chunk pays compile; if time-to-tol
feeds fitness, warm up one chunk per dtype before the timer.

## 3. The invariant (paper section 3.6 wording)

Backend genes select an execution policy for the same mathematical operator.
Changing dtype changes numerical errors, so exact-arithmetic convergence
guarantees do not transfer verbatim to mixed precision. Construction
assumptions, implementation tests, and runtime numerical checks are separate
obligations. Every
kernel variant must pass the five per-atom Hawkeye checks
(`atlas.backend.unit_tests.run_all`) against the float64 numpy references at
its declared tolerance before it may appear in any phenotype. Certificates
are NEVER evaluated below float64: the duality gap is a difference of
near-equal numbers, bf16 cannot represent a 1e-4 relative gap at all, and
f32 summation noise at n around 1e6 is the size of the gap itself; the
backend enforces this mechanically (`certified_gap` raises on any non-f64
certificate value, and `get_policy` refuses bf16/f16 by name).

## 4. Integration TODO list (integrator agent)

1. Schema: add the `backend` block to genome/schema.py validate() with the
   table-1 rules; include it in canonical JSON + content hash; extend the
   20k fuzz to cover it (invalid precision strings, bf16 spellings, K=0,
   theta=0, theta>=1 all rejected).
2. Schemes: expose per-scheme `step_factory(dtype)` and `init_factory(dtype)`
   (factor the existing closure construction; constants cast to dtype) and
   an f64 `gap64` closure; route solve() through
   `get_policy(genome.backend.precision, K=..., theta=...).run(...)` when the
   block is present, keeping the current km_run path bit-identical for the
   default f64 block.
3. Wrappers: over-relaxation and Halpern combine INSIDE step_factory (they
   are part of what T is, not how it is computed), so their typechecks stay
   exactly where they are.
4. Benchmark card: add `"backend": {"precision", "K", "theta",
   "iterate_dtypes", "switch_iter", "certificate_dtype"}` and the
   `detect_device()` record under `"hardware"`.
5. Evolve: mutation operators may flip precision within the enum and perturb
   K (log-scale int) and theta (log-scale in (tol, 1)); the Stage-0 gate
   already rejects invalid values, so no clamping in the mutator.
6. Context Builder: include the `detect_device()` record (kind, memory_gb,
   tflops_f32, tflops_f64 and their ratio) in the LLM prompt; without it the
   cross-hardware divergence claim is only about random search.
7. Kernel gate (Stage-0.5): any NEW kernel implementation registered for an
   atom must ship a `run_check(name, impl, tol)` pass before entering the
   population; wire this next to the Stage-0 typecheck so an uncertified
   kernel is a certified rejection, not a crash.
8. batch=true: reserved; validate, record, no-op until the P4 vmap backend
   lands.
