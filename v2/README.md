# Optimization Atlas v2

Hardware-aware execution and search are documented in
[docs/HARDWARE_WORKFLOW.md](docs/HARDWARE_WORKFLOW.md). The new versioned
diagnostic driver measures certificate cadence, precision policies,
joint solver/generated-CUDA-template choices, and hardware-context proposer controls.
Recorded implementation results are in
[results/hardware_codex/REPORT.md](results/hardware_codex/REPORT.md).
The paper direction and decisive controls are in
[docs/ICLR_HARDWARE_DIRECTION.md](docs/ICLR_HARDWARE_DIRECTION.md).

Typed operator-splitting solver framework in JAX (P0 core): four certified
schemes (FBS, DRS, PDHG, Condat-Vu) on TV denoising of real Kermany2017
clinical OCT B-scans, plus an evolution loop (LLM-guided vs random control)
over a typed solver genome. Design details: `DESIGN.md`; project plan:
`papers/optimization_atlas_v2_plan.md`.

## Environment

- CPU-only JAX with float64, enforced at `import atlas` (the GPU is owned by
  a vLLM server). Entrypoints still set `JAX_PLATFORMS=cpu` defensively.
- Python (CANONICAL for all solver work since 2026-09-03):
  `/home/miria/jaxenv2/bin/python` (jax 0.11.1, matches mazeppa-1 exactly;
  see `results/sprint/phase1/env_parity.json`). GPU runs from jaxenv2 must
  use `env -u LD_LIBRARY_PATH` (the shell's CUDA-12.4 path shadows the pip
  nvidia wheels and silently falls back to CPU); CPU runs are unaffected.
- `/home/miria/jaxenv` (jax 0.9.2) is LEGACY: it hosts the vLLM server and
  its packages must not be touched. It still passes the suite and remains a
  secondary check until deprecated; do not start new solver runs in it.
- LLM endpoint: OpenAI-compatible vLLM at `http://localhost:8000/v1`
  (Qwen/Qwen2.5-Coder-1.5B-Instruct).

## Tests

```
/home/miria/jaxenv2/bin/python -m pytest            # from the v2 root
```

The one `integration`-marked test needs the live vLLM endpoint and skips
otherwise.

## Demo / results

```
/home/miria/jaxenv2/bin/python scripts/run_tv_cell.py \
    --size 64 --n-train 2 --n-holdout 1 --generations 2 --pop-size 2 \
    --seed 0 --evolve-seed 42
```

writes `results/baseline_card.json`, `results/evolution_history.json`,
`results/comparison.md`, `results/evolution_curves.png`. Current numbers
(toy budget above) are in `results/comparison.md`; at this budget the
default schemes beat both evolved champions; the demo demonstrates the
pipeline (certified construction, strict-time fitness, held-out split,
flat-fitness detection), not an evolution win.

## Measurement methodology

- `wall_time` / time-to-tol cover ONLY the compiled solver loop: the KM loop
  is AOT-compiled (`jax.jit(...).lower(...).compile()`) before the timer
  starts, so JIT trace/compile time never enters fitness or reported times.
- Fitness is the strict SGM-10 of time-to-tol (non-converged = inf); the
  PAR-1 aggregate is auxiliary only.
- The evolution loop records per-candidate fitnesses per generation and
  flags flat/near-flat fitness landscapes (`flat_fitness_warning`) in the
  history and in `comparison.md`.
- All results JSON is strict RFC 8259 (non-finite floats are stringified as
  `"inf"`/`"-inf"`/`"nan"`).

## Reproducibility

- The random-control arm is fully deterministic under `--evolve-seed`.
- LLM proposals carry a deterministic per-request `seed` (vLLM honors it),
  so identical runs replay identical proposals *given identical prompts*.
  Residual nondeterminism remains because later-generation prompts embed
  measured wall-clock fitnesses, which carry timing noise; ordering claims
  between configurations within a few percent of each other (e.g. default
  pdhg vs condat_vu) are inside that noise and should not be read as
  rankings.
