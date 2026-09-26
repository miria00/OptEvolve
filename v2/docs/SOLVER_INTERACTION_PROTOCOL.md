**Development experiment: solver and execution-policy interaction**

The first execution milestone of the ICLR attack plan tests whether useful
interaction exists before allocating a larger search budget. The implemented
entry point is `scripts/solver_interaction.py`. Output directories must be new;
the script writes its machine-readable protocol and candidate order before
loading data or timing any solve.

The development-v1 runs use the preserved `source_v1.tar.gz` snapshot in
`results/attack_plan_20260906/`. The current runner is development-v2, adding
patient-group checks before measurements. Numerical candidate definitions,
tolerances, budgets and selection rules are unchanged. Each result must retain
its own source version; do not relabel v1 measurements as a v2 execution.

| Choice | Frozen diagnostic setting |
|---|---|
| Data | Twelve 256x256 center crops, deterministic image sampling from the cached Kermany dataset train split, seed 20260907 |
| Partitions | Six training and six development-validation images; no LP or final holdout data |
| Objective | Anisotropic TV, lambda 0.1 |
| Algorithms | PDHG and Condat-Vu, each at (tau, sigma, rho) = (0.05, 1, 1.9) and (0.25, 0.25, 1.5) |
| Structural control | A direct numerical map comparison must show the two schemes differ before timing |
| Policies | f64, f32 with f64 checks, and f32-to-f64 refinement at threshold 0.01; each at K = 16, 64, 256 |
| Baseline | Previously banked Condat-Vu, f64, K = 64; an improved execution baseline, not a claim of globally optimal expert tuning |
| Tolerances | 1e-4 and 1e-6 |
| Budget | Exactly at most 5,000 iterations per solve, including the final truncated block; no wall-time deadline |
| Timing | One complete cold/warmup solve followed by three warm repeats; synchronized numerical stopping checks included |
| Additional checks | Independent NumPy original-problem gap outside the timer, compared with the reported gap |
| Selection | Maximize train solve count, then minimize train SGM-1 with fixed 120-second non-solve penalty, then lexical tie break |
| Order | Fixed randomized candidate order, seed 718; reverse candidate order for validation |
| Choices compared | Fixed, algorithm-only, implementation-only, algorithm-then-implementation, reverse order, joint |
| Inference | Finite-catalog interaction diagnostic; not fresh-evaluation-matched evolutionary runs and not evidence of LLM superiority |

The code makes selections before evaluating the validation fold at each
tolerance. All validation catalog measurements are retained for diagnosis;
that fold is development data and must not later be described as the untouched
final evaluation. Failures remain visible in every comparison. A penalized SGM
is not the actual mean runtime of successful solves.

The first protocol incorrectly stated that patient IDs were unavailable.
Encoded image storage paths contain a class-patient-scan token, although the old
loader returned only image/label metadata. A separate audit found twelve distinct
filename-derived patient groups in the v1 sample and zero inter-fold overlap.
`patient_group_audit.json` records this correction without changing the original
protocol or any input. The v2 loader records hashed groups; the runner rejects
missing or overlapping groups before profiling. Sampling remains uniform over
images, not patients. At larger scale, report group counts and use grouped
selection/inference rather than treating every image as an independent patient.

V1 device runs are exploratory workstation measurements. CPU execution overlaps
light development activity and a draft PDF build; GPUs retain their existing
desktop/server allocations. Do not use these timings as the final hardware
comparison. A final campaign needs a quiet resource allocation, declared reuse
assumptions, repeated search seeds, stronger automatic tuning baselines and
reporting of compile/preparation/search costs.

Run the current protocol from the v2 root, with a fresh destination:

```bash
env -u LD_LIBRARY_PATH JAX_PLATFORMS=cpu \
  /home/miria/jaxenv2/bin/python scripts/solver_interaction.py \
  --out results/attack_plan_20260906/new_development_v2
```

For GPU runs, explicitly select `JAX_PLATFORMS=cuda` and use the solver
environment with a compatible JAX version. The experiment does not make LLM
requests. Raw source/input hashes and GPU allocation are recorded in each run.
