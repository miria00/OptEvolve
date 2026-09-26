# Hardware-aware OptEvolve

The current submission work starts with the
[solver interaction protocol](SOLVER_INTERACTION_PROTOCOL.md) and
[claim ledger](ICLR_CLAIM_LEDGER.md). The new
`scripts/solver_interaction.py` diagnostic compares distinct TV updates against
compiled f64 execution at K=64, with patient-group checks before timing. The
original hardware campaign below remains reproducible as a separate diagnostic.

Hardware awareness is a required paper contribution. The implemented pilot
searches algorithm and execution-policy combinations using measured operator
costs on a declared device. It follows Hawkeye's executable-example approach:
an optimization strategy is accompanied by correctness obligations, a
measurement, and conditions under which it may lose.

## Implemented

- `atlas/backend/hardware.py`: measured f64/f32 base-step costs, original
  matvec/adjoint costs, f64 certificate cost, optimized HLO artifacts, and
  compact context under `none`, `identity`, and `measured` conditions.
- Explicit `Backend(precision="f64"|"f32_cert64", K=...)` uses one compiled
  device loop. Numerical checks occur every K iterations and at the exact
  iteration cap. Histories record the actual check indices and count.
- `schedule` executes a bounded f32 prefix and reserves at least one check
  block for f64 refinement. Threshold or plateau can switch earlier. A
  prefix that already passes the f64 numerical criterion can terminate.
- Relaxation is applied before the Halpern anchor on both execution paths.
  Cache keys include relaxation; certificate data remain dynamic across
  same-shape problems.
- LP dense and BCOO operators preserve the iterate dtype. The original f64
  data and original coordinates are used for numerical certification.
- The generic LLM backend accepts operator profiles and records token use.
  Its fallback seed is offset from the random-control stream even after
  `reseed()`. Historical experiments are unchanged.
- `scripts/hardware_campaign.py`: versioned profiling, a finite-catalog
  algorithm-by-policy comparison, train-only selection, transfer-ready
  measurements, and an optional budget-matched proposer pilot. Existing
  output directories are refused and results checkpoint after each row.
- `scripts/profile_solver_kernels.py`: isolated Nsight capture after
  compilation, with named NVTX ranges for the solver, step block, matvecs,
  and certificate. Enable CUDA graph node tracing to count actual kernels.
- `atlas/backend/cuda_lp.py` generates three FP64 CUDA/FFI template
  specializations: one thread per CSR row, or one warp per row with 128 or
  256 threads per block. Each fuses a matvec and its PDHG vector update.
  Two ordered launches preserve the primal-to-dual dependency. CUDA graph
  capture is supported; the compiler can add a loop-counter kernel. Compiled sources, build commands, and logs are
  content-addressed under `~/.cache/optevolve/cuda_lp/`.
- `Backend(kernel="csr_thread128"|"csr_warp128"|"csr_warp256", K=...)`
  enters the genome hash and production compiler. The current templates
  accept standard LP PDHG in f64 only. The mathematical construction gate
  precedes compilation; four reference states must agree numerically before
  fitness evaluation. Wrappers and original-coordinate checks are retained.

## Run a diagnostic

From the v2 directory, with the canonical local environment:

```bash
env -u LD_LIBRARY_PATH JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
  /home/miria/jaxenv2/bin/python scripts/hardware_campaign.py \
  --cell oct --n-train 3 --n-holdout 3 --repeats 3 --max-iters 20000 \
  --out results/hardware/my_oct_run
```

Use `JAX_PLATFORMS=cpu` for the CPU. On mazeppa-1 use its `jaxenv` interpreter.
Never compare devices with simultaneous LLM inference on the timed GPU.
The driver includes compilation/preparation separately from warm solver time.
Its 120-second non-solve penalty is a scoring convention, not a wall-time
interrupt. Its numerical execution cap is `--max-iters`.

For the LP diagnostic, only names from the frozen calibration partition are
admitted. The supplied file hashes are checked before loading.

```bash
env -u LD_LIBRARY_PATH JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
  /home/miria/jaxenv2/bin/python scripts/hardware_campaign.py \
  --cell lp --lp-names air05,trento1,neos8 --n-train 2 --n-holdout 1 \
  --max-iters 20000 --out results/hardware/my_lp_run
```

The within-calibration train/holdout split is exploratory. It is not the
frozen final paper holdout and does not revise the failed calibration.

## Proposer controls

```bash
JAX_PLATFORMS=cpu /home/miria/jaxenv2/bin/python scripts/hardware_campaign.py \
  --cell oct --pilot-only --pilot-evals 8 --n-seeds 2 --tolerances 0.0001 \
  --out results/hardware/my_context_pilot
```

The five arms are independent random proposals, typed population mutation,
and LLM mutation with no hardware context, device identity, or measured
operator context. Each arm gets the same baseline and a target number of
fresh candidate evaluations. Duplicate proposals do not consume that budget.
The LLM and fallback streams are separately recorded. Scope rejections,
fallbacks, LLM calls, tokens, proposal transcripts, and actual budget
completion are saved. A budget-incomplete run is not a matched-budget result.
The small pilot is an execution check and preliminary measurement; it cannot
establish a general LLM or context advantage.

The existing TV driver can also receive the profile:

```bash
/home/miria/jaxenv2/bin/python scripts/run_tv_cell.py \
  --hardware-profile results/hardware/my_oct_run/profile.json \
  --hardware-context measured
```

## Actual GPU kernels

```bash
env -u LD_LIBRARY_PATH JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
  nsys profile --trace=cuda,nvtx --cuda-graph-trace=node --sample=none \
  --cpuctxsw=none --capture-range=cudaProfilerApi --capture-range-end=stop \
  --output=results/hardware/my_trace \
  /home/miria/jaxenv2/bin/python scripts/profile_solver_kernels.py \
  --cell lp --lp-names trento1 --steps 200 --repeats 3 \
  --out results/hardware/my_trace_metadata
```

Use `nsys stats --report nvtx_gpu_proj_sum,cuda_gpu_kern_sum --format csv`
on the report. Profiling changes timings; use uninstrumented campaign
results for speedup claims. HLO fusion instructions are not a kernel count.

## Assurance and claim boundaries

Construction certificates describe the mathematical operator under their
stated assumptions. Implementation tests check numerical behavior. Runtime
f64 checks validate the declared stopping criterion; float64 is not exact
arithmetic, an interval bound, or a proof of mixed-precision convergence.
The LP runner preserves original-coordinate checks after Ruiz+PC scaling.

The generated kernel family is a bounded template search. It does not yet
generate arbitrary CUDA programs using an LLM, expose a sparse-format gene,
or extend batching to metric/relaxation-plus-Halpern genomes.
Pure f32 is an empirical policy and can fail to meet tight tolerances.

Same-algorithm backend gains, algorithm gains, and joint-selection gains
must be reported separately. A different genome string is not evidence of
hardware specialization. If the same policy wins on both GPUs, report it.

## Joint algorithm and generated-kernel diagnostic

Pass `--generated-kernels` to the LP campaign. A bounded example is:

```bash
env -u LD_LIBRARY_PATH JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
  /home/miria/jaxenv2/bin/python scripts/hardware_campaign.py \
  --cell lp --lp-names comp07-2idx,neos-1171737,neos-1171448,neos-4300652-rahue \
  --n-train 2 --n-holdout 2 --algorithm-filter frozen_B0,expert_relaxation \
  --generated-kernels \
  --policy-filter legacy,f64_K50,f64_K200,csr_thread128_K50,csr_thread128_K200,csr_warp128_K50,csr_warp128_K200,csr_warp256_K50,csr_warp256_K200 \
  --tolerances 0.000001 --repeats 3 --max-iters 20000 \
  --out results/hardware/my_joint_run
```

This deliberately uses previously solvable calibration instances to test
kernel speed, so its solve rate is not representative. The driver records
algorithm-only, backend-only, sequential, and joint catalog selections.
Default kernel `jax` preserves old genome serialization and hashes. CUDA
tests are opt-in with `OPTEVOLVE_TEST_GPU=1`; the default suite stays on CPU.

Reference: Hawkeye, Hardware-Aware GPU Kernel Optimization with Minimal
Supervision, local paper at `../../papers/62_Hawkeye_Hardware_Aware_GPU_.pdf`;
conference link https://icml.cc/virtual/2026/82754.
