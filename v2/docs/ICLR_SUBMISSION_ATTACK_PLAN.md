**OptEvolve: ICLR submission assessment and attack plan, revised 6 September 2026**

This revision supersedes the morning assessment. The earlier version is kept at
`.attack_plan_20260906_pre_decomposition.bak`. Two things changed today: four
literature lanes returned with a specific novelty verdict, and a measurement
campaign found that the paper's central axis was confounded. The measurements
are in [the decomposition report](../results/tv_ffi_20260906/DECOMPOSITION_REPORT.md).

My assessment is unchanged in verdict and changed in reason. The submission
would still likely be rejected in its current manuscript state, but no longer
because the evidence is thin. It is because the manuscript currently attributes
its headline number to the wrong cause. Fixing that attribution is now the
highest-value work available before September 18, and it converts a set of
unremarkable speedups into a mechanism.

**What changed: the precision axis was never the precision axis**

`atlas/schemes/base_schemes.py:813` dispatches the execution path on the
precision gene. Float64 and float32 both run a device loop whose inner
`fori_loop` has traced bounds, which XLA cannot unroll. Only the `schedule`
policy runs the chunked path whose `fori_loop` has a static bound, which XLA
unrolls and fuses. Precision and execution structure are therefore the same
gene, and no cell in the published catalog separates them.

Separating them by hand gives the result the paper needs:

| Effect, per-iteration cost, RTX 4090, tolerance 1e-4 | K=16 | K=64 | K=256 |
|---|---|---|---|
| float32 over float64, inside the fused device loop | 0.98x | 1.00x | 1.00x |
| float32 over float64, inside chunked host phases | 0.97x | 1.32x | 1.58x |
| chunked over fused, holding float64 | 0.83x | 1.26x | 1.49x |
| chunked over fused, holding float32 | 0.82x | 1.67x | 2.35x |

Float32 buys exactly nothing at every cadence and both tolerances when the loop
is not unrollable, and buys 1.58x when it is. The iteration count N is exactly
584/597/683 at K=16/64/256 across both schemes, both precisions, both execution
structures, both custom kernels and both GPUs, because the stopping test is only
evaluated at multiples of K. So cadence sets N and everything else sets the
per-iteration cost, and the two execution knobs interact strongly while the
mathematical knob does not participate at all.

This is worth stating plainly because it inverts the manuscript's framing. The
abstract currently credits "float32-to-float64 refinement with a longer check
interval". The refinement is doing a minority of the work; the loop structure
is doing the majority; and the paper never had a gene for loop structure.

**Why this rescues the paper rather than damaging it**

The surveys came back with a hard problem. Every individual number OptEvolve
holds is either published or below a published envelope:

- cuOSQP, JPDC 144:55-67 (2020), section 7.4, already published a sub-2x
  float32-over-float64 envelope for a GPU first-order solver, with the correct
  explanation that the kernels are memory-bound. Our 1.41x sits inside it.
- NVIDIA cuOpt shipped `CUOPT_PDLP_PRECISION` with single, mixed and double
  modes in PR #910, merged 7 March 2026, and documents that float32 "may require
  more iterations to converge".
- Certificate cadence is long-standing solver engineering. OR-Tools defaults
  `termination_check_frequency` to 64 with an explicit "(involves extra work)"
  comment; implementations hard-code 5, 25, 30, 40, 64, 100, 150 and 200.
- Per-device policy selection is established by KernelFoundry (ICML 2026),
  Ansor (OSDI 2020), SparseDitto and clSpMV (ICS 2012). OSQP already ships
  float32 on CUDA and float64 on CPU.
- The precision ceiling itself is a theorem: Carson and Higham, SISC
  40(2):A817-A847 (2018), and Brisebarre, Carrino, Mary and Riccietti,
  arXiv:2607.04828 (July 2026), for optimization specifically.
- Pushak and Hoos, ACM TELO 2(3) Article 10 (2022), already published the
  finding that per-hyperparameter sequential optimization ties with joint
  optimization across AutoML landscapes. Our null has a precedent.

Against that, a paper that reports 1.41x from mixed precision is dead on
arrival. But a paper that explains why the published envelope ranges from 0.62x
to 2.35x across the literature, and shows the explanation is loop structure
rather than arithmetic width, is answering a question the field has left open.
cuOSQP measured the envelope and attributed it to memory-boundness without
testing the attribution. We can test it: in the non-unrollable structure the
iteration is not memory-bound and float32 gives 1.00x; in the unrollable one it
is and float32 gives 1.58x. That is a mechanism for a published-but-unexplained
spread, which is exactly what the ICLR guide means by contributing new knowledge
without requiring state of the art.

The claim also connects to how the field actually writes its solvers. cuPDLPx
captures exactly `termination_evaluation_frequency` iterations as a replayed
CUDA graph, and HPR-LP maintains two kernel variants specifically to avoid
paying for the check. Those are the same design decision we are measuring:
the cadence constant and the loop structure are chosen together, and the
precision benefit is contingent on that joint choice. No paper says so.

**The revised paper**

Working question: for a declared problem distribution, tolerance and device,
which axes of an executable solver recipe actually carry performance, and why?

Claimed structure, in decreasing order of confidence:

1. The iteration count is a function of the certificate cadence alone. Exact,
   measured across 12 configurations and two devices, with a one-line mechanism.
2. The mathematical axis is inert on this cell. Algorithm accounts for 0.00% to
   0.34% of log-time variance, and every interaction involving it is at or below
   0.23%. The kernel axis is likewise at parity on the 4090.
3. Arithmetic width is not a per-iteration speed multiplier. It is contingent on
   loop structure, with a measured 1.00x against 1.58x split, and it is
   separately subject to a hard accuracy ceiling: pure float32 solves 0 of 6 at
   1e-6 while every float64 and mixed configuration solves 5 of 6.
4. Therefore the axes the field searches over are the inert ones, and the axes
   nobody exposes as searchable carry the variance.

Point 3's second half is the single most defensible novel measurement we hold.
The surveys confirmed that no GPU first-order LP paper publishes any float32
convergence comparison at tight tolerance, that cuOpt's own source casts
tolerances to float with no clamp or warning, and that the field's 2025 survey
(Lu and Yang, arXiv:2506.02174) contains no floating-point precision discussion
at all. The gap between cuOpt's documented "converges slower" and our measured
"does not converge" is real and unclaimed.

Title: keep hardware explicit and drop any suggestion of joint discovery.
"OptEvolve: Which Axes of a Certified Solver Recipe Actually Matter" is closer
to the evidence than the current title. Decide at the abstract deadline.

**Sentences that cannot be written, from the survey**

Do not claim first use of reduced-precision iterates with full-precision
certification (cuOpt ships it; CuClarabel and Liang and Yang arXiv:2605.31425
do it). Do not claim original-coordinate certification as a contribution
(OR-Tools, cuPDLP.jl and cuPDLPx all state it; OSQP has `scaled_termination`).
Do not claim discovery of the float32 accuracy ceiling, of certificate overhead,
or of precision search (Precimonious, HiFPTuner, AMPT-GA, VaPr, and Carson and
Chen arXiv:2601.00728 from January 2026). Do not claim per-device policy
selection as novel. Do not lean on 1.41x. Do not say "the winning policy is
hardware-specific" while both GPUs choose the same one. Do not restore the
"first feedback-versus-resampling ablation" claim; Verma arXiv:2607.26117
matches on model family, temperature, test, arms and mechanism. Do not state
an accept tier for any cited ICLR paper; OpenReview is bot-walled and the
`poster_` URL slug is provably uninformative.

Define arithmetic width against termination tolerance in the first paragraph and
never let them share the word "precision". Kempke and Koch, arXiv:2503.10344,
is titled "low-precision" and means loose tolerance throughout, and it reports
a 1.5x speedup from that, uncomfortably close to our number in a reader's memory.

**Priority 0, immediate, September 6 to 8**

Make execution structure a first-class gene. The best measured configuration,
float32 iterates with chunked execution at K=256, is 2.11x over the compiled
float64 K=64 baseline against the catalog selection's 1.54x, and the current
genome cannot express it: it cannot say "float32 with chunked execution" without
also saying "switch to float64 eventually". This is a schema and dispatch change
in `atlas/genome/schema.py` and `atlas/schemes/base_schemes.py`, plus the
construction gate in `atlas/backend/cuda_tv.py` which currently reads
`backend.precision == "f64"` as a proxy for float64 iterates. Until this lands,
every catalog result silently varies two things at once.

Then correct the manuscript's causal attribution everywhere it appears: abstract,
introduction, experiments and conclusion all currently credit precision for an
effect that is mostly loop structure. Keep the claim ledger discipline from the
previous plan: every number traced to an artifact, stale generation comparison
removed, related work rewritten around the v2 system.

Re-measure the headline table with the GPU clear. vLLM held 15.4 GB on the timed
4090 throughout today's campaign at 0% utilization, and the reference cell is
stable to 0.66% across 12 processes, so the relative structure is sound, but the
published absolute numbers should not carry that caveat.

**Priority 1, the experiment that carries the paper, September 8 to 12**

The decomposition currently rests on one problem family, one device, one problem
size and six images. That is the binding weakness, not the analysis. In order:

Replicate the execution-structure effect on the 5090. It is not attached to this
machine; arrange access. The 5090 catalog already shows the same N invariance
and a 1.85x host-over-device ratio at K=256, but it has no float32 host cell.
This also settles the one live hardware-specialization candidate: `tv_fused128`
is 1.03x ahead of XLA on the 4090 and 1.40x behind it on the 5090, which if it
survives re-measurement is the first genuine cross-device reversal the project
has produced. Treat it as a candidate, not a result, until re-measured.

Extend the cadence grid. Three points cannot distinguish a cadence optimum from
a monotone trend, and cadence is now the largest single variance term. Run
K in {8, 16, 32, 64, 128, 256, 512, 1024} and report N(K) and t(K) separately.
N(K) should be the crossing iteration rounded up to a multiple of K, which is a
falsifiable prediction with no free parameters, and t(K) should decompose into
a certificate term proportional to 1/K and a fusion term. Fitting both and
predicting the optimum on held-out images is the causal test the previous plan
asked for and never got.

Add problem sizes. Every measurement is at 256x256. The memory-bound argument
predicts the float32 advantage grows with problem size and the fusion advantage
shrinks once each chunk saturates the device. Both are testable in an afternoon
and both are the kind of prediction reviewers reward.

Carry the decomposition to LP. LP is where the certificate is expensive relative
to the iteration and where the cited literature lives. The cadence constants
scattered across PDLP, cuPDLP, HPR-LP and cuPDLPx are LP constants, so the
strongest version of the cadence argument is an LP figure showing where each
published constant lands on our measured curve. Decompose our own certificate
cost first: D-PDLP's model gives roughly 1.5x an iteration and we report 2.05x
to 2.93x, so account for the excess before printing it.

**Priority 2, main evidence, September 12 to 17**

The matched-budget search comparison from the previous plan still stands, with
one change: the search now has a reason to exist. If execution structure and
cadence carry the variance and the algorithm does not, then the interesting
search question is whether a search that can reach the execution axes beats one
that cannot, under equal budget. Run the arms from the previous plan, with the
execution-structure gene included, on development data.

Hold the sample-size discipline: at least tens of independent held-out units,
grouped by patient where identifiers permit, with solve fractions and censored
time-to-certificate reported rather than successful cases only. Ten search seeds
per main arm. Predeclare the primary comparison and the sequential budget split.

Cut or re-run the LLM-versus-random claim. The arms shared a proposal stream
with eight of eight byte-identical draws. It is the project's largest integrity
exposure and it is not needed by the revised paper. Cutting it is the default.

**Priority 3, explanation and assurance, September 15 to 20**

Four figures: the N(K) invariance with the rounding prediction overlaid; the
execution x precision x cadence surface with the fANOVA decomposition; the
cost decomposition showing why fewer launches can still lose; and the accuracy
ceiling with solve fractions at 1e-4, 1e-6 and 1e-8.

Keep the three assurance levels distinct and stated: exact-arithmetic convergence
under stated assumptions, empirical reference-equivalence admission for compiled
updates, and numerical termination checks on the original problem. Float64
evaluation is not a floating-point proof. A rejected sufficient condition is not
a divergence proof.

Report the reuse assumption. Warm solves here are milliseconds; a break-even
count in the thousands should be stated, not hidden.

**Calendar**

| Date | Deliverable |
|---|---|
| September 6 to 8 | Execution-structure gene landed; manuscript attribution corrected; headline re-measured on a clear GPU |
| September 8 to 12 | 5090 replication, cadence grid, problem sizes, LP decomposition |
| September 12 to 17 | Matched-budget search comparison including the execution axes |
| September 18, 23:59 AoE | Abstract reflecting the decomposition, not the old speedup framing |
| September 19 to 22 | One frozen final evaluation; main tables with uncertainty and failures |
| September 23 to 24 | Independent claim, source and math review; clean-environment reproduction; anonymous PDF |
| September 25, 23:59 AoE | Submission, targeting a day early |

Nine pages of main text. Budget roughly one page motivation, half a page related
work, one page method, one page assurance, four and a half pages of evidence,
one page interpretation and limitations. The decisive comparison and the
assumptions belong in the main text; reviewers need not read the appendix.

**The standing decision rule**

Proceed with the decomposition as the headline. It is the only claim we hold
that is not individually pre-empted, it has a mechanism, it makes falsifiable
predictions, and it explains a spread the cited literature reports without
explaining. Do not restore a joint-discovery headline unless a budget-matched
gain over the strongest sequential control survives held-out evaluation.

The honest risk remains that the whole result reads as an artifact of one JAX
lowering decision on one problem family. The defense is breadth: a second device,
a second problem size, LP, and a cadence curve with a fitted predictive model.
That breadth is achievable in the twelve days remaining, and it is a better use
of them than any further search engineering.

**Addendum, 9 September 2026: the three axes, the endgame, and the family queue**

The user fixed the paper's structure. Three axes, and a status report means the
four-row table (three axes plus endgame) with a "what is missing" column.

1. Solving algorithm: scheme, parameters, wrappers.
2. Input problem family, across multiple datasets and scales, characterised by
   condition number and structure.
3. Hardware speedup: kernel fusion and iteration fusion.

**Endgame:** given a specific problem or benchmark with a specific condition
number and structure, automatically discover a recipe that beats the default
baselines. Conditioning on structure is the goal; one global winner is not.

**What landed on 9 September.** Netflow (grid min-cost flow) is a second
family. The execution-structure gain replicates at 1.386x mean over 24 runs,
iteration count is identical across loop structures in 24 of 24 runs, and
scheme identity is inert again in the affine-smooth-term case where it was
predicted to collapse.

The most useful result is that the "balanced" parameters are **inadmissible**
on netflow rather than merely slow: the construction gate rejects them
symbolically at `t*s*||L||^2 = 1.02267 > 1`, in 24 of 48 configurations. The
same parameters are legal but 4.31x slower on TV. So the admissible region of
the method axis is problem-family dependent and the typed gate detects it
before any execution. That is structure-conditioned selection arriving for
free, and it is the bridge from the current catalog to the endgame.

**Family queue, ranked.** Assets already present: `atlas/bench/netflow.py`,
`data/lp_suites/` (MIPLIB and Mittelmann, 823 MB), and `lasso_box_problem`.

| Family | Why it earns a slot | Structure knob |
|---|---|---|
| Optimal transport | PDOT (arXiv 2407.19689) is a published restarted-PDHG GPU solver, so this is the family where we can be measured against someone else's baseline rather than only our own default | regularisation sets the condition number; the constraint matrix is never formed |
| Multi-commodity flow | extends netflow directly; coupling constraints change the operator norm and therefore the admissible parameter region, which is the effect already measured | commodities and topology |
| Basis pursuit / Dantzig selector | the canonical composite family; conditioning sweeps continuously | sensing-matrix coherence |
| TV computed tomography | same prox, Radon operator rather than a gradient stencil, so temporal fusion should not apply; an informative negative for the fusion claim | angles and detectors |
| SDP relaxation / Lovasz theta | far worse conditioned; the prox is a PSD projection and per-iteration cost is an eigendecomposition rather than a matvec, testing whether the factorisation survives a different cost model | graph size and density |
| Model predictive control | the small-problem many-solves regime where the measured 2.63x at n=128 matters most and search cost amortises fastest | horizon and state dimension |

Optimal transport and multi-commodity flow are the two to run next: the first
buys an external baseline, the second buys a principled operator-norm knob on
code that already runs.
