# INTEGRATION.md

Handover for the coauthor. Written 10 September 2026 against the drafts in
`/home/miria/OptEvolve/v2/docs/iclr_draft_20260910/` and the manuscript in
`/home/miria/OptimizationAtlas/v2/OptAtlas_tex_2026/`.

Deadlines: ICLR 2027 abstract 18 September 2026, paper 25 September 2026.

Style contract in force for every drafted file and for this one: no em-dash
characters, no italics, plain declarative prose.

Provenance rule in force: every number printed in the paper must be traceable
to a file under `/home/miria/OptEvolve/v2/results/`. Where a number has no
artifact, this document says so and the paper must say UNSUPPORTED or drop it.

---

## 0. Read this first: three paths point at one directory

`/home/miria/OptimizationAtlas` is a symbolic link to `/home/miria/OptEvolve`,
and `/home/miria/OptEvolve/v2/OptAtlas_tex_2026` is a symbolic link to
`/home/miria/OptEvolve/v2/OptEvolve_tex_2026`. All three of the following
resolve to the same inode:

| Path | Resolves to |
|---|---|
| `/home/miria/OptimizationAtlas/v2/OptAtlas_tex_2026/` | the manuscript tree |
| `/home/miria/OptEvolve/v2/OptAtlas_tex_2026/` | the same tree |
| `/home/miria/OptEvolve/v2/OptEvolve_tex_2026/` | the same tree |

Verified: `1-introduction.tex` under the first and third paths is device 66306
inode 97158203 in both cases.

Consequence. Writing to `v2/OptEvolve_tex_2026/` is writing to the protected
manuscript. Nothing in this draft round has touched any of the three paths, and
nothing should until you have reviewed the proposals below. The drafts live
only in `v2/docs/iclr_draft_20260910/`.

---

## 1. Drafted files and what each one replaces

Five section bodies are drafted. Each is a section body only: no preamble, no
`\begin{document}`.

| Drafted file | Lines | Replaces in the manuscript | Old size | Kind of change |
|---|---|---|---|---|
| `1-introduction.tex` | 209 | `1-introduction.tex` | 2911 B | Complete rewrite. New lead is the width-against-tolerance separation, then the factorisation, then four contribution paragraphs and a "what this paper does not show" subsection. Adds `tab:intro-e2e`. |
| `2-related-work.tex` | 213 | `2-related-work.tex` | 3050 B | Complete rewrite. Old version positions against FunSearch, AlphaEvolve, ReEvo, GEPA, AutoOPT and Hawkeye only. New version positions against the GPU first-order LP line, the reduced-width line, cadence engineering, portfolios, AutoML nulls and autotuning. Adds `tab:related`. |
| `3-methods.tex` | 280 | `3-methods.tex` | 7612 B | Complete rewrite. Keeps the recipe, gate and assurance-level material, drops the search-loop and blending prose, adds the execution-structure gene, the fusion kernel geometry and numerical admission. Adds `tab:recipe`, `tab:gate`, `tab:tiles`. |
| `5-experiments.tex` | 485 | `5-experiments.tex` (7779 B) and `generated/development_interaction.tex` | 7779 B + generated table | Complete rewrite. The generated interaction table and its 1.47 and 1.57 numbers are not carried forward. Adds `tab:Nvectors`, `tab:periter`, `tab:loopmech`, `tab:fusion`, `tab:ushape`, `tab:regret`, `tab:roofline`. |
| `6-conclusion.tex` | 182 | `6-conclusion.tex` | 691 B | Complete rewrite and large expansion. Emits two top-level sections, Discussion and Conclusion, from one input file. Adds `tab:leak` and `tab:falsify`. |

### 1.1 Not drafted, still required

| Manuscript object | Status | What is needed |
|---|---|---|
| Abstract, inside `submission.tex` lines 100 to 118 | Not drafted | Must be rewritten from scratch. The current abstract carries the misattribution described in section 2 below and three cut-list items. |
| Title, `submission.tex` line 45 | Not drafted | "OptEvolve: Hardware-Aware Search for Certified Optimization Solvers" leads with kernels and search. The governing plan asks for a title that leads with the factorisation, because KernelBench-Verified (arXiv:2607.16241, June 2026, unpublished) is a live scoop risk on the kernel framing. |
| `4-theory.tex` | Not drafted | Kept as is or lightly edited. It has five dangling cross-references and two discovery-language problems. See section 2, items W12 and W13. |
| `appendix-theory.tex`, `appendix-blending.tex`, `appendix-instantiation.tex` | Not drafted | Reviewed only for cross-references. `appendix-theory.tex` refers to `sec:theory`, which survives. |
| `refs.bib` | Not drafted | 20 citation keys used by the drafted `2-related-work.tex` do not exist yet. Listed in section 4.3. |
| `submission.tex` body | Not drafted | The `\input` order, the reproducibility, ethics and AI-use statements, and the removal of `\input{generated/development_interaction}` from `5-experiments.tex`. Ready-to-paste statements are in section 5 of this document. |

### 1.2 Label breakage introduced by the swap

The drafted `3-methods.tex` and `5-experiments.tex` do not define the labels the
retained `4-theory.tex` depends on. Five references break at build time:

| Broken reference | Where it is used | Old definition site | New status |
|---|---|---|---|
| `sec:gate` | `4-theory.tex:5` | old `3-methods.tex:63` | Not defined by the new methods draft |
| `sec:results` | `4-theory.tex:61`, `4-theory.tex:177` | old `5-experiments.tex:24` | Not defined by the new experiments draft |
| `sec:methods` | `4-theory.tex:148`, `4-theory.tex:204` | old `3-methods.tex:2` | Not defined |
| `sec:mixing` | `4-theory.tex:206` | old `3-methods.tex:40` | Not defined |
| `sec:hardware` | old `1-introduction.tex:37` | old `3-methods.tex:86` | Reference removed with the old introduction |

The new experiments draft defines `sec:exp` with subsection labels
`sec:exp:factorisation`, `sec:exp:execution`, `sec:exp:fusion`,
`sec:exp:method`, `sec:exp:omega`, `sec:exp:negative` and
`sec:exp:conditions`. The new methods draft defines no section label at all.
Fix by adding `\label{sec:methods}`, `\label{sec:gate}` and
`\label{sec:mixing}` to the new methods draft and pointing `sec:results` at
`sec:exp:method`, or by editing the four `4-theory.tex` lines.

---

## 2. Claims in the current manuscript that are now known wrong

Severity key. FATAL means a reviewer who reads the artifacts would call the
claim incorrect. SCOPE means the claim is arithmetically right and the scope is
wrong. STALE means the claim was right when written and the corpus has moved.

### W1. FATAL. The headline credits precision for an effect that is loop structure

Where: `submission.tex:104` to `submission.tex:109` and
`5-experiments.tex:114` to `5-experiments.tex:118`.

As written: "execution-policy selection reduces warm-time scores by factors of
1.47 and 1.57 on an RTX 4090 and RTX 5090 ... Both GPUs select
float32-to-float64 refinement with a longer check interval", and in the
experiments section "Both choose the banked Condat-Vu solver with
float32-to-float64 refinement and K=256".

Why it is wrong. `atlas/schemes/base_schemes.py:813` dispatches the compiled
loop structure off the precision gene. `f64` and `f32_cert64` always ran
`run_device_policy`, which is `jax.lax.fori_loop(k, end, ...)` with a traced
bound inside a `while_loop` and which XLA cannot unroll. `schedule` always ran
the host-phase chunk runner, `jax.lax.fori_loop(0, K, ...)` with a static
Python K, which XLA unrolls and fuses across iterations. The precision gene and
the execution-structure choice are one gene, and no cell of the shipped catalog
separates them. Selecting `schedule` selected the chunked loop, and the paper
attributed the resulting gain to arithmetic width.

The corrected decomposition, TV at n=256 on the RTX 4090, microseconds per
iteration, mean over the six OCT scans:

| Execution | Width | K=16 | K=64 | K=256 |
|---|---|---|---|---|
| Device loop | float64 | 22.54 | 20.39 | 19.78 |
| Device loop | float32 iterates | 22.99 | 20.41 | 19.78 |
| Host phases | float64 | 27.18 | 16.19 | 13.29 |
| Host phases | float32 throughout | 28.00 | 12.23 | 8.43 |

Isolated at K=256: arithmetic width buys 1.000x inside the device loop and
1.576x once the loop is chunked. Execution structure buys 1.488x at float64 and
2.345x at float32, and reverses to 0.829x at K=16.

Artifacts. `results/tv_ffi_20260906/DECOMPOSITION_REPORT.md` sections 1 and 2,
computed from `results/tv_ffi_20260906/cadence_4090_v1/factorial.json`,
`f32row_4090/factorial.json`, `f64host_4090/factorial.json` and
`px999999_4090/factorial.json`.

Correction in the drafts: `5-experiments.tex` subsection `sec:exp:execution`
carries the table above verbatim, and `1-introduction.tex` opens by separating
arithmetic width from termination tolerance before any number is printed.

### W2. FATAL. The manuscript never had a gene for the thing that carried the gain

Where: `3-methods.tex:88` to `3-methods.tex:91`, "The execution policy selects
float64 iteration, float32 iteration with float64 checking, or a finite float32
phase followed by float64 refinement."

Why it is wrong. That sentence describes a three-valued precision enum. It does
not say that the same enum silently selects the compiled loop structure, which
is the variable that carries the measured effect. Enumerating the pairs over all
fifteen factorial artifacts under `results/tv_ffi_20260906/` and
`results/method_axis_20260909/` gives 2208 rows of (f64, device_loop), 144 of
(f32_cert64, device_loop) and 1680 of (schedule, host_precision_phases), with no
cross-pairing anywhere.

Correction: promote execution structure to a gene of its own and report it. The
drafted `3-methods.tex` does this and states the confound explicitly. The
consequence is a code change, not only an analysis change, and the drafted
`5-experiments.tex` records that the best configuration measured, XLA float32
with host phases at K=256 at 0.00576 penalised SGM-1 seconds against the
baseline 0.01216, is not expressible in the shipped genome and was produced by
patching the policy registry.

Artifacts. `results/tv_ffi_20260906/DECOMPOSITION_REPORT.md` sections 1 and 6,
`results/tv_ffi_20260906/px999999_4090/factorial.json`,
`results/tv_ffi_20260906/cadence_4090_v1/factorial.json`.

### W3. SCOPE. "Generated CUDA kernels are slower than XLA" is device specific

Where: `submission.tex:112` to `submission.tex:114`, and
`5-experiments.tex:67` to `5-experiments.tex:75`.

As written: "A separate LP diagnostic finds that the tested generated CUDA
kernels are slower than XLA despite fewer launches", and "The thread-per-row
CUDA variant at K=50 is about 25 percent slower than XLA on the two held-out LP
instances on the RTX 4090, and about 50 percent slower on the RTX 5090."

Why it is wrong as a general statement. Holding execution structure fixed, the
`tv_fused128` custom kernel is at parity with XLA on the RTX 4090: 19.21 against
19.76 microseconds per iteration in the device loop at K=256, an edge of 1.03x,
and 13.27 against 12.90 in host phases, a loss of 0.97x. The same kernel gains
1.45x from chunked execution exactly as XLA does. The kernel loss is
5090-specific: there `tv_fused128_K256` costs 29.49 microseconds against XLA's
21.13, a 1.40x loss.

Two separate corrections are required. First, the sentence must not be stated as
a property of generated kernels in general. Second, the manuscript's sentence is
about LP CSR kernels while the correcting measurement is the TV kernel, so the
LP CSR claim itself has no 2026-09-09 or 2026-09-10 re-measurement. It must be
restricted to the device and family it was measured on, or re-measured, or cut.

Artifacts. `results/tv_ffi_20260906/DECOMPOSITION_REPORT.md` section 4 and
`results/tv_ffi_20260906/kernelhost_4090/factorial.json`.

### W4. SCOPE. Every RTX 5090 number is now out of scope

Where: `submission.tex:106`, `5-experiments.tex:32` to `5-experiments.tex:39`
(table rows), `5-experiments.tex:68`, `5-experiments.tex:116`,
`5-experiments.tex:126` to `5-experiments.tex:129`.

The paper's device scope is now two architectures, the RTX 4090 (sm_89) and the
RTX PRO 6000 Blackwell Max-Q (sm_120). The 5090 is explicitly out of scope and
is not present on the measuring machine.

Affected numbers, all of which must be removed or re-measured: 1.57x, 4.37x,
4.03x, 2.31x, the "about 50 percent slower" kernel figure, the 14.66 to 9.33 ms
selection, and the 0.75 percent joint improvement.

Artifacts. `results/kernel_fusion_20260909/REPORT.md` lines 3 to 5, and
`results/tv_ffi_20260906/DECOMPOSITION_REPORT.md` caveats block.

### W5. STALE. The legacy denominator is not a competent baseline

Where: `5-experiments.tex` table `tab:tv-results`, the "legacy/selected" column,
values 4.60x, 2.91x, 5.20x, 4.55x, 4.37x, 4.03x, 2.25x and 2.31x.

Those ratios compare against legacy execution. Against the compiled float64
K=64 baseline the same catalog selection is 1.540x, and the whole table rests on
3 training and 3 held-out 128 by 128 patches. The corpus now carries six
256 pixel scans per size and four independent factorial artifacts. The table
should be dropped rather than corrected.

Artifacts. `results/hardware_codex/REPORT.md` selection table,
`results/tv_ffi_20260906/DECOMPOSITION_REPORT.md` section 6.

### W6. STALE. "Native TV kernels are absent from this catalog"

Where: `5-experiments.tex:131`.

False as of 9 September 2026. The corpus contains `tv_fused128`, `tv_recip256`
and the temporal kernel `results/kernel_fusion_20260909/tv_temporal.cu`, all
measured end to end at four sizes on two devices.

Artifacts. `results/tv_ffi_20260906/kernelhost_4090/factorial.json`,
`results/kernel_fusion_20260909/e2e/` and `e2e_bw/`.

### W7. FATAL BY OMISSION. Related work does not position against the nearest lines

Where: all of `2-related-work.tex`.

The section as written positions against FunSearch, AlphaEvolve, ReEvo, GEPA,
AutoOPT, Lyapunov proof discovery, splitting theory, PDLP, OSQP and Hawkeye. It
does not mention cuPDLP, cuPDLPx, MPAX, HPR-LP, PDOT, cuOpt, CuClarabel, cuOSQP,
Carson and Higham, Ansor, clSpMV, KernelFoundry, Pushak and Hoos, SATzilla or
ASlib. Every one of those is nearer to the current claims than any of the cited
works, and four of them publish results the paper must not claim as new. The
drafted replacement positions against all of them.

### W8. WRONG FRAMING. The research question is answered and the paper still asks it

Where: `1-introduction.tex:27` to `1-introduction.tex:31`, and
`6-conclusion.tex:8` to `6-conclusion.tex:12`.

As written: "does hardware-aware selection improve time to a checked solution
after strong algorithm and implementation tuning, and when must these choices be
made jointly?" and "The central experimental question is whether joint selection
improves time to a checked solution beyond competent algorithm and
implementation tuning."

That question now has a measured answer, and it is a null with a published
precedent (Pushak and Hoos, ACM TELO 2(3) Article 10, 2022). Keeping it as the
central question makes the paper's own headline result a failure. The
factorisation explains the null: when one factor is set by cadence and the other
by execution, per-axis search recovers the joint optimum.

Artifacts. `results/hardware_codex/REPORT.md` selection tables, all four
selection strategies at 0.392627 penalised holdout seconds on the 4090 and
0.379070 on the 5090.

### W9. INCOMPLETE. The float32 accuracy result needs a scheme qualifier

Where: `submission.tex:111` to `submission.tex:112`, "pure float32 iterations can
fail despite float64 checking", and `5-experiments.tex:130`, "Pure float32
Condat-Vu solves none of the six at this tolerance."

The Condat-Vu figure is right. The statement is scheme dependent: at 1e-6 with
float32 iterates and a float64 certificate, `condat_vu_banked` solves 0 of 6 and
`pdhg_banked` solves 1 of 6, on both architectures. Write both.

Artifacts. `results/tv_ffi_20260906/f32row_4090/factorial.json`,
`results/bw_20260909/sz256_bw/factorial.json`.

### W10. INCOMPLETE. The gate is described but never shown to be family dependent

Where: `3-methods.tex:65` to `3-methods.tex:71`.

The construction gate is described correctly and is never measured. The corpus
now shows it is an exact function of the gate product alone, admitting 351 of
351 settings at each of theta in {0.1, 0.3, 0.6, 0.9, 0.99} and 0 of 351 at
theta = 1.02 over 2106 runs, and that the admissible region is family dependent:
the balanced preset that is legal and merely slow on TV is rejected before any
arithmetic on all 24 balanced min-cost flow rows, at gate left-hand sides
1.02267, 1.0304, 1.03577 and 1.03781 at grid sizes 12, 16, 24 and 32.

Artifacts. `results/structure_20260909/structure_summary_4090.json`,
`results/netflow_20260909/grid_4090.json` (48 rows, 24 carrying an `error` key,
values verified 10 September 2026).

### W11. INCOMPLETE. The iteration-count law is absent from the manuscript

The single most defensible measured claim in the corpus does not appear in the
manuscript at all: the iteration count is a function of certificate cadence and
tolerance alone. On TV at n=256, tolerance 1e-4, the per-instance vectors are
identical element by element across two architectures, both arithmetic widths,
both execution structures and every kernel: 584.0, 597.3 and 682.7 at K=16, 64
and 256. On min-cost flow all 24 static-against-traced loop pairs and all 12
cross-scheme pairs have exactly equal counts.

One qualifier the drafts enforce and the coauthor must not drop: scheme identity
is exactly inert at K=64 and K=256 and splits at K=16, where `condat_vu_banked`
gives 584.0 and `pdhg_banked` gives 578.7 on both devices. Never write
"identical across both schemes" without the cadence qualifier.

Artifacts. `results/tv_ffi_20260906/cadence_4090_v1/factorial.json`,
`results/bw_20260909/sz256_bw/factorial.json`,
`results/netflow_20260909/grid_4090.json`.

### W12. FATAL. Discovery language in the theory section

Where: `4-theory.tex:61`, "The champion of Section~\ref{sec:results}";
`4-theory.tex:177`, "The discovered policy of Section~\ref{sec:results}";
`4-theory.tex:180`, "the evolved"; `4-theory.tex:183`, "the champion".

The banked policy (tau, sigma, rho) = (0.05, 1.0, 1.9) is a pre-existing catalog
entry, not a search output. The artifact says so in terms: "The algorithm
catalog contains existing textbook and banked policies for OCT, and frozen B0
plus expert wrapper settings for LP. Selection from this catalog is not
autonomous algorithm discovery."

Replace "champion", "discovered policy" and "evolved" with "the banked setting"
throughout. Four lines.

Artifact. `results/hardware_codex/REPORT.md` line 11.

### W13. BUILD. `4-theory.tex` cross-references break

See section 1.2 above. Five dangling references.

### W14. STALE. Two problem families are now three

Where: `5-experiments.tex:136`, "A second mathematical problem family remains
necessary for the intended breadth of claims."

The corpus now carries min-cost flow on grid and regular topologies and optimal
transport in addition to TV. The sentence should be deleted, and the remaining
gap named precisely instead: there is no capacitated MCF run anywhere in the
corpus.

Artifacts. `results/netflow_20260909/grid_4090.json`,
`results/structure_20260909/`, `results/finish_20260910/omega_ot_4090.json`.

---

## 3. Cut list

Everything in this section must not appear in the submitted paper in any form,
including in an appendix, a footnote or a rebuttal.

### C1. Any LLM sample-efficiency or proposal advantage

Appears at `submission.tex:115`, `1-introduction.tex:23` to `1-introduction.tex:31`,
`1-introduction.tex:45`, `5-experiments.tex:45`, `5-experiments.tex:78` to
`5-experiments.tex:84`, `5-experiments.tex:94`, `6-conclusion.tex:8`.

The arms were not independent. All 336 of 336 fallback-sourced candidates in the
model-guided arm are byte-identical to candidates of the matched random run, and
the budgets were unequal at 76 to 136 fresh solver evaluations against 160 per
random run. The current manuscript already hedges rather than claims, and even
the hedged framing must go, because it keeps an unsupported comparison as the
paper's organising question. The drafted `1-introduction.tex` states the null in
a "what this paper does not show" subsection and moves on.

Note for the coauthor: the figure is 336 of 336, from
`results/sprint/phase2/analysis/gate1_verdict.md` lines 101 to 102 and
`gate1_numbers.json`. An "8 of 8 byte-identical draws" figure circulating in
planning notes is not the artifact figure. Do not paste it.

### C2. Any 1.41x mixed-precision headline, and its manuscript instantiations

The 1.41x figure sits below the published float32 against float64 envelope for a
GPU first-order solver (cuOSQP, JPDC 144:55-67, 2020, section 7.4), which also
supplies the memory-bound explanation. A paper whose headline is 1.41x from
mixed precision is answering a question that is already answered.

In the manuscript this appears as 1.47x and 1.57x at `submission.tex:105` and
`5-experiments.tex:115` to `5-experiments.tex:116`, and as the whole of
`generated/development_interaction.tex`. Cut the headline. The component
measurements survive only inside the decomposition of W1, where the point is
that width buys 1.000x in one loop structure and 1.576x in another, and where
the contribution is the mechanism rather than the multiplier.

Also cut `\input{generated/development_interaction}` at `5-experiments.tex:112`.

### C3. Any joint-search advantage over sequential per-axis search

Appears at `submission.tex:109` to `submission.tex:110`, `1-introduction.tex:29`
to `1-introduction.tex:31`, `5-experiments.tex:57` to `5-experiments.tex:66`,
`5-experiments.tex:126` to `5-experiments.tex:129`, `6-conclusion.tex:8` to
`6-conclusion.tex:12`.

Joint tied sequential. On LP, backend-only, sequential, reverse-sequential and
joint selection all choose `frozen_B0` with `f64_K50` at 0.392627 penalised
holdout seconds. The 0.75 percent RTX 5090 joint improvement must go with the
5090. The null itself may be reported, and must be reported as reproducing
Pushak and Hoos rather than as a finding, with the factorisation as the
explanation.

### C4. Any accelerator-portability or hardware-specialization claim

Two GPU architectures were measured. Both GPUs in the old catalog selected the
same policy, so that catalog cannot support specialization either. The
tile-selection result is a per-device decision on two devices and must be stated
as exactly that.

### C5. Novelty claims the surveys have already refuted

Do not claim first use of reduced-precision iterates with full-precision
certification (cuOpt ships `CUOPT_PDLP_PRECISION`, merged 7 March 2026;
CuClarabel arXiv:2412.19027; Liang and Yang arXiv:2605.31425). Do not claim
original-coordinate certification as a contribution (OR-Tools, cuPDLP.jl,
cuPDLPx; OSQP has `scaled_termination`). Do not claim discovery of the float32
accuracy ceiling; it is a theorem (Carson and Higham, SISC 40(2):A817-A847,
2018; Brisebarre et al. arXiv:2607.04828). Do not claim certificate cadence as
new; OR-Tools defaults `termination_check_frequency` to 64 with an explicit
"(involves extra work)" comment and the field hard-codes 5, 25, 30, 40, 64, 100,
150, 200. Do not claim per-device policy selection as new (KernelFoundry ICML
2026, Ansor OSDI 2020, clSpMV ICS 2012; OSQP already ships float32 on CUDA and
float64 on CPU). Do not restore any "first feedback-versus-resampling ablation"
claim.

### C6. Accept tiers for cited ICLR papers

Do not write poster, spotlight or oral for any cited ICLR paper anywhere.

### C7. The word "precision" used to mean tolerance

Define arithmetic width against termination tolerance in the first paragraph and
never let them share a word. Kempke and Koch arXiv:2503.10344 is titled
"low-precision" and means loose tolerance throughout, and reports a 1.5x speedup
from it, which is close enough to our numbers to be confused in a reader's
memory. The drafted `1-introduction.tex` opens with this separation and the
drafted `2-related-work.tex` repeats it.

### C8. The Theta(1/K) contraction claim for capacitated MCF

The corpus contains no capacitated min-cost-flow run. The contraction is a
closed-form property of the construction with no accompanying measurement. The
drafted `5-experiments.tex` deliberately omits it and states the absence. Keep
it out, or run it. Mark UNSUPPORTED if it must be mentioned.

### C9. The RTX 5090

See W4.

---

## 4. What still has to be run or written before submission

### 4.1 Measurements, in priority order

**R1. The frozen final holdout, which has never been evaluated.** This is the
single largest outstanding obligation and it is stated as unevaluated in two
artifacts: `results/hardware_codex/REPORT.md` line 7, "The original
pre-registration and final holdout remain untouched", and
`results/attack_plan_20260906/REPORT.md` line 120, "Final holdouts remain
unevaluated." Every measurement in the corpus is exploratory. The runner is
`scripts/eval_holdout.py`, which evaluates two distances reported separately, a
within-distribution set of 10 fresh netflow instances at cell seed 7007 and an
out-of-distribution set from Mittelmann and MIPLIB, at tolerances 1e-4 and 1e-6
with 60 and 120 second caps. The driver is `scripts/launch_local_holdout.sh`.
Warning: that driver begins with `cd /home/miria/OptimizationAtlas/v2`, which is
the aliased path from section 0. Decide before running whether the holdout is
the Phase 2 netflow holdout or the LP and OCT pre-registration holdout, because
the two are different objects and the manuscript refers to the second.

**R2. Score the wide-grid omega sweeps that have already landed and are
unanalysed.** Four artifacts exist that no drafted section cites:
`results/finish_20260910/omega_wide_bw.json` (1404 rows, ratio grid extended
down to 1e-4 on the same four cells as `omega_4090.json`),
`omega_ot_wide.json` (702 rows, optimal transport, wide grid),
`omega_holdout.json` (702 rows, two cells not used in fitting: `regular:256:2`
and `grid:32`), and `omega_seeds_4090.json` (1296 rows, 8 seeds). Recomputed
from those rows on 10 September 2026, with oracle equal to the per-instance
minimum iteration count over the ratio grid, a policy scored only where it
converged, and the geometric mean of N over oracle:

| Sweep | Instances | Oracle at grid floor | Best constant ratio | Rule r = C / omega0^2 | Regret-optimal exponent | Naive log-log slope |
|---|---|---|---|---|---|---|
| Narrow grid, 4090 and Blackwell pooled, as drafted | 209 | 91 (43.5 percent) | 1.827 geomean, 23.03 worst | 1.187 geomean, 3.87 worst at C = 0.251 | 2.15 | -0.931 |
| Wide grid, Blackwell, same 4 cells | 108 | 26 (24.1 percent) | 2.457 geomean, 20.57 worst | 1.188 geomean, 2.12 worst at C = 1.122 | 2.00 | -1.531 |
| Wide grid restricted back to the narrow ratios, same cells and device | 104 | 46 (44.2 percent) | 1.851 geomean, 23.03 worst | 1.135 geomean, 2.12 worst at C = 1.122 | 1.95 | -0.988 |
| Wide grid, optimal transport | 54 | 0 (0.0 percent) | 1.428 geomean, 13.19 worst | 1.026 geomean, 1.15 worst at C = 31.62 | 2.00 | -1.213 |
| Wide grid, two held-out cells | 54 | 16 (29.6 percent) | 2.284 geomean, 12.19 worst | 1.093 geomean, 1.99 worst at C = 0.126 | 1.85 | -1.469 |

Three things follow, and all three strengthen the paper. First, the censoring
explanation is now demonstrated rather than argued: on identical cells and the
identical device, widening the ratio grid moves the naive slope from -0.988 to
-1.531 and halves the floor censoring from 44.2 percent to 24.1 percent, while
the regret-optimal exponent moves from 1.95 to 2.00 and the fitted constant does
not move at all. Second, the exponent is exactly 2.00 on two independent wide
grids, netflow and optimal transport. Third, the rule holds on two cells that
were not used to fit it, at 1.093 geomean regret against 2.284 for the best
constant. Caveat that must travel with this: the constant C is not universal. It
is 1.122 on the netflow wide cells, 0.126 on the held-out netflow cells and
31.62 on optimal transport, where omega0 itself spans only 0.4464 to 0.7928. The
exponent transfers across families and the constant does not.

Source rows: `results/finish_20260910/omega_4090.json`, `omega_bw.json`,
`omega_wide_bw.json`, `omega_ot_wide.json`, `omega_holdout.json`. No scoring
script is committed; see R7.

**R3. CLEARED for the 4090; still open for the second architecture.** The
admission output is now archived at
`results/artifacts_20260910/kernel_admission_4090.log`, 106 lines, 102 printed
cases, worst absolute deviation 4.649e-15 at rho = 1.9, PASS. Because
`validate.py` prints a case only when it deviates by more than 1e-12 or is not
bitwise equal, the absence of every rho = 1 case from that log is itself the
rho = 1 bitwise result, so both numbers are now backed by a stored table rather
than by prose. What remains open is the second architecture: no Blackwell
admission run is archived, so `3-methods.tex` no longer prints the sm_120
figure that `results/kernel_fusion_20260909/REPORT.md` line 131 reports as
prose. Run it there and commit the log if the paper wants that sentence back.

**R4. The missing loop-structure control.** The drafted `5-experiments.tex`
writes "It does not shrink as the outer count falls to one, so it is not a lost
unrolling opportunity." That inference does not follow from the design of
`results/finish_20260910/loop_mechanism.json`. In every traced column the inner
trip count is traced, so cross-iteration fusion is unavailable in every traced
column, and a lost-fusion mechanism predicts exactly the flat per-iteration
penalty that was measured. What the artifact does establish, and what the
sentence should say, is that the penalty is neither a per-outer-trip overhead
nor a per-certificate-check cost. The static-side cost is also nearly flat in
unroll depth, 8714 nanoseconds per iteration at K=16 against 8175 at K=512 at
n=128, a 6.6 percent spread. The discriminating control that is missing is a
static inner bound of K=1, which has unroll depth 1 with a static trip count. If
that cell is as slow as the traced cells, the mechanism is unrolling. If it is
as fast as the other static cells, the mechanism is the traced bound itself.
This is one cheap run and it decides how strongly the mechanism can be stated.

**R5. Re-measure the headline cells on a clear device.** A vLLM server held
15427 MiB of 24564 MiB resident on the timed 4090 throughout, at 0 percent
utilisation. The reference cell is stable to 0.66 percent across 12 independent
processes, so relative comparisons hold, but the reported figures should be
re-taken with the device clear.

**R6. Settle the timing protocol.** All fusion numbers in the drafts are median
of 3. A median-of-9 re-measurement exists at
`results/finish_20260910/fusion_r9_sz256.json` and `fusion_r9_sz496.json` and
gives larger values, 1.940x at n=256 and 1.428x at n=496 against the drafted
1.867x and 1.356x. The drafted figures are therefore the conservative ones.
Choose one protocol, state it, and never mix the two.

### 4.2 Analysis artifacts that must be committed

**R7. A committed scoring script for the omega regret analysis.**
`scripts/diagnostics/` currently holds `loop_mechanism.py`, `omega_sweep.py`,
`roofline_probe.py`, `score_roofline.py` and `sharpness_probe.py`. There is no
`score_omega.py`. Every regret number in the drafted introduction, experiments
and conclusion was recomputed by hand from the raw rows. Until a script and its
output are committed under `results/`, those numbers sit outside the provenance
rule. This is exactly the objection recorded in the drafted
`2-related-work.tex` header, which forbids the omega figures in that section.

**R8. A committed fANOVA script and its output.** The variance percentages
(algorithm at 0.00 percent and 0.34 percent, cadence main effect at 39.44
percent and 60.98 percent, width by cadence at 39.73 percent and 35.51 percent,
cadence on log per-iteration cost at 47.43 percent and 51.19 percent) exist only
as prose tables in `results/tv_ffi_20260906/DECOMPOSITION_REPORT.md` sections 2
and 4. A grep for fANOVA across `scripts/` and `results/` returns that file and
nothing else. Regenerate and persist before submission, and cite Hutter, Hoos
and Leyton-Brown, ICML 2014, PMLR 32(1):754-762.

### 4.3 Writing and build work

**R9. Twenty missing bibliography keys.** The drafted `2-related-work.tex` cites
these and `refs.bib` does not contain them: `Lu2026cuPDLP`, `Lu2026cuPDLPx`,
`Lu2026MPAX`, `PDOT2024`, `cuOpt2026`, `CarsonHigham2018`, `Brisebarre2026`,
`CuClarabel2024`, `LiangYang2026`, `cuOSQP2020`, `KempkeKoch2025`,
`PushakHoos2022`, `Hutter2014fANOVA`, `Xu2008SATzilla`, `BischlASlib2016`,
`Zheng2020Ansor`, `KernelFoundry2026`, `Su2012clSpMV`,
`KernelBenchVerified2026`, `Wolf1991TemporalBlocking`. Present already:
`Applegate2021Practical`, `Lu2024HPRLP`, `ORToolsPDLP2026`, `Gleixner2021MIPLIB`,
`Stellato2020OSQP`, `Goulart2024Clarabel`.

**R10. A citation-verification pass.** The literature verdicts the drafts
position against come from a four-lane survey recorded as prose in
`docs/ICLR_SUBMISSION_ATTACK_PLAN.md`. That survey has no artifact under
`results/`, so it sits outside the provenance rule that governs every other
claim in the paper. Verify each verdict against its primary source. The one
third-party numeric claim printed in the drafts is the OR-Tools default of 64;
confirm it against the source file.

**R11. Resolve the internal contradiction about the omega numbers.** The drafted
`2-related-work.tex` header states that the omega regret figures "are forbidden
until scripts/diagnostics scoring is committed under results/". The drafted
`1-introduction.tex`, `5-experiments.tex` and `6-conclusion.tex` all print them.
R7 resolves this. Until it is resolved the four files disagree with each other.

**R12. Reconcile three small numeric disagreements before compiling.**

| Quantity | Drafted `5-experiments.tex` | Drafted `6-conclusion.tex` | Recomputed 10 Sep 2026 |
|---|---|---|---|
| Instances with oracle ratio at the grid ceiling | 11 of 209 | 10 | 10 of 209 |
| Rule regret, geomean-optimal constant | 1.181 at C = 0.263 | 1.181 | 1.187 at C = 0.251 |
| Rule regret, worst-case-optimal constant | 1.190 at C = 0.0369, worst 3.18 | not printed | 1.192 at C = 0.0398, worst 3.18 |

The differences in C come from the resolution of the constant grid and from how
a predicted ratio is snapped to the measured grid. Fix the snapping convention
in the committed script of R7 and regenerate all three files from one source.

Separately, planning notes circulating a 1.175x geomean and a 1.56x geomean
improvement do not match either the drafts or the artifacts. The artifact-backed
pair is a best constant ratio at 1.827 geomean and 23.03 worst against the rule
at about 1.19 geomean and 3.18 worst. Use the recomputed values.

**R13. Remove the duplicated table.** `tab:intro-e2e` in the drafted
introduction and `tab:fusion` in the drafted experiments share the baseline and
best-speedup rows. Keep one. The drafted experiments file carries a note saying
so.

**R14. Rewrite the abstract and decide the title.** Neither is drafted. The
abstract must lead with the factorisation, must separate arithmetic width from
termination tolerance before any number, must not print 1.47x or 1.57x, must not
mention the 5090, and must state that the frozen holdout is unevaluated.

---

## 5. Ready-to-paste statements

These replace the three starred sections in `submission.tex` lines 121 to 145.
Plain LaTeX, no em-dash characters, no italics.

### 5.1 Reproducibility statement

```latex
\section*{Reproducibility Statement}
Every measurement reported here is exploratory. The frozen final holdout and the
original pre-registration remain unevaluated, and no number in this paper was
selected against them. All results come from two GPUs, an RTX 4090 (sm\_89) and
an RTX PRO 6000 Blackwell Max-Q (sm\_120), and from exhaustive enumeration of
finite catalogs rather than from budget-matched search runs.

Artifacts. Each recorded run serialises its complete recipe, the input hashes,
the device and compiler versions, the selection partition, the per-instance
iteration count, the wall time to a certified gap, and the certificate value in
float64 in original coordinates. Iteration counts are reported per instance and
not only as means, because the invariance claims of Section~\ref{sec:exp} are
element-by-element identities rather than agreements in mean.

Protocol. Unless stated otherwise, timings are the median of three repeats after
one discarded warm-up, tolerance is a relative duality gap of $10^{-4}$
evaluated in float64 in original coordinates, and the end-to-end sweeps use six
retinal OCT scans per size. The montage sizes at 992 pixels and above use four
images per arm and are built by tiling several distinct scans, because a raw
scan caps an honest crop near 496 pixels. The scaling sweeps use three seeds per
cell. A median-of-nine re-measurement of two fusion cells returns larger
speedups than the median-of-three figures we report, so the reported figures are
the conservative ones.

Measurement conditions. Neither machine was exclusive during timing. A resident
inference server held 15427\,MiB of 24564\,MiB on the timed RTX 4090 at zero
percent utilisation. A reference cell re-measured in twelve independent
processes spans 20.297 to 20.432 microseconds per iteration, a spread of 0.66
percent, which is far below every effect we report. The Blackwell box is shared;
the fusion microbenchmarks used one idle GPU but not exclusively, and all four
GPUs were idle for the cross-architecture end-to-end runs.

Known gaps in the artifact record. The step-ratio regret scoring was computed
from the stored raw rows without a committed scoring script; there is no
`score_omega.py` under `scripts/diagnostics/`. The dense gate grid records
step-size pairs with a null iteration count for rejected rows, because rejected recipes are never run,
so the construction gate is verified as sufficient and not as necessary.
```

### 5.2 Ethics statement

```latex
\section*{Ethics Statement}
The imaging experiments solve a convex denoising objective on an existing
retinal OCT dataset and involve no human subjects, no new data collection and no
personally identifying information. Reaching an optimization tolerance is a
statement about the optimization problem and not about the image. It does not
establish preservation of pathology, diagnostic adequacy or suitability for any
clinical decision, and we report no clinical efficacy result of any kind.
Dataset attribution and the applicable redistribution terms accompany the
submission artifact, and the released code loads the data through its documented
source rather than redistributing it.

The optimization families we study are standard convex benchmarks with no
foreseeable dual-use concern. The measurements consume GPU time on two
workstation-class devices; we report wall-clock cost per configuration so that
the compute footprint of the reported sweeps can be assessed directly.
```

### 5.3 AI use statement

```latex
\section*{AI Use Statement}
Language models appear in this work in two distinct roles, and we separate them.

As an object of study, a language model is one of the proposal mechanisms in the
search loop. The local proposal pilot uses Qwen2.5-Coder-1.5B-Instruct. We claim
no advantage for that mechanism, and we state why: the model-guided and random
arms were not independent, because all 336 of 336 fallback-sourced candidates in
the model-guided arm are byte-identical to candidates of the matched random run,
and the arms had unequal budgets of 76 to 136 fresh solver evaluations against
160 per random run. That comparison is reported as uninterpretable rather than
as a null result, and no sample-efficiency claim rests on it.

As a research tool, language models assisted the workflow, including literature
discovery, experiment planning, code implementation, debugging, manuscript
drafting and the interpretation of results. Coding assistants, including Claude
and Codex, contributed to implementation and to writing. Every mathematical
statement, citation, artifact path and reported measurement was verified by the
authors against the stored run records, and the authors take full
responsibility for the content. A complete model and version inventory
accompanies the submission artifact.
```

---

## 6. One-page summary of what changed

The paper's claim moves from "hardware-aware selection of a precision policy is
worth 1.47x" to "time to a certified solution factorises, and the certificate
cadence is the only gene in both factors". That move is forced by one
measurement: the precision gene and the execution-structure gene were the same
gene, so the old headline credited arithmetic width for an effect that belongs
to the compiled loop structure. Width buys 1.000x inside the traced device loop
and 1.576x once the loop is chunked, at the same device, size and cadence.

What survives from the old manuscript: the typed recipe representation, the
symbolic construction gate, the original-coordinate float64 certification, the
three assurance levels, the float32 accuracy ceiling at 1e-6, and both nulls.

What is new and load bearing: the exact iteration-count law across two
architectures, both widths, both execution structures and every kernel; the
settled loop-structure mechanism; the temporally fused kernel with its U-shaped
gain and its per-device optimal tile; the family-dependent admissible region;
the step-ratio rule at exponent 2; and two negative results that make the case
for search rather than prediction.

What must not be claimed: an LLM advantage, a joint-search advantage, a
1.41x-class mixed-precision headline, accelerator portability, or novelty in
reduced-precision iterates, original-coordinate certification, certificate
cadence or per-device policy selection.
