# Executable discovery recovery, 26 September 2026

The current submission demonstrates strong manually constructed solvers and some
successful generated operators. It does not yet demonstrate that the outer loop
constructs the strongest complete solvers. A successful experiment must retain
the proposed programs, their parents, all failures, evaluation costs, and a frozen
test set. A recipe description or selection of a supplied winner is insufficient.

## What to train and what to execute

The user supplies a problem class, a target certificate and hardware. The proposer
receives standard solver references and a language of transformations. It writes
an executable solver, observes actual execution, and edits its formulation,
algorithm, carried state, work restriction, handoff, numerical precision, device,
and compiled work width. Standard algorithms are legitimate teaching material;
our completed winning implementations are evaluation targets, not hidden answers
in the training prompts.

The important separation is between learning the search policy and learning the
numerical solution. Train the policy to make useful solver changes across problem
classes. The numerical solver still computes and certifies each instance. Do not
replace the certificate with a learned judge or reward attractive explanations.

## Implemented first experiment

`scripts/discovery/executable/` provides an executable eight-hook program interface,
a projected-gradient seed, a fixed independent float64 acceptance test, and a
fleet runner. The first problem family is batched simplex-constrained least
squares, including the paper's Cuprite spectral-unmixing design. The space contains
general numerical code, not a list of hand-built solution implementations.

The four arms, with two initial independent seeds and twelve fresh programs per
seed, are:

| Arm | Standard method references | Four task-specific cards | Execution history |
|---|---|---|---|
| full_cards | yes | yes | yes |
| withheld_cards | yes | no | yes |
| no_references | no | no | yes |
| resample | yes | no | no |

The first contrast measures the value of task-specific cards. The second measures
standard-method teaching. `withheld_cards` versus `resample` isolates execution
feedback and selection. These are small pilots, not powered significance tests.
This is an HSI ablation, **not the manuscript's lasso D3–D6 ablation**. A random
proposer over an identical typed grammar remains a separate required control.

Each candidate must pass three small numerical admission cases, including a
rank-deficient and a zero design, before timed search cases. A gate rejection uses
one candidate evaluation and is scored as failure; format errors and duplicates
are separately recorded with a hard proposal-call cap. Both fresh candidate
counts and actual execution/model time must be reported because passing programs
use more expensive evaluations than rejected ones.

Fitness is solved count followed by geometric mean time, with unsolved cases
charged the entire fixed cap. Partial certificate progress breaks failure ties
only in search; it is never counted as success. Time includes algorithm setup,
JIT compilation, execution, transfers, and external certificate checks. JAX
runtime initialization is excluded from that clock and included in a separately
recorded process wall time. These measurements must not be described as the
paper's fully cold first-visit result without the startup cost.

The search gets only training cases. Pixel IDs and hashes were frozen before model
calls. Validation and test inputs are initially absent from search workers. Pixel
holdouts share a library, so this is within-library transfer, supplemented by
synthetic shape/conditioning shifts, not unseen real-dataset generalization.

The hidden audit solver ports the existing shared-inverse ADMM implementation
into the same harness. It is never a parent and its source is never in a model
prompt. It establishes headroom and a rediscovery target. The full paper solver,
strong external specialists, and true cold timing remain necessary final comparators.

## Scope of the gate and hardware

The new general-code interface is **not** the paper's convergence-certified genome
grammar. Its API/AST guard and finite numerical tests do not prove convergence or
equivalence of arbitrary generated operators. A synthesized candidate needs a
separate mapping into the existing constructive certificate rules before it can
support that stronger claim. The AST restriction is not a security sandbox.

All eight hooks execute. Device choice sets the actual JAX device context, the
precision hook selects internal arithmetic while the external judge stays fixed,
and width controls the iterations per launch/check and available loop unrolling.
The LLM writes its compiled JAX loops. Native CUDA/FFI source synthesis and the
paper's exact lasso insert/delete and equiangular-screening mutations are not yet
part of this first family adapter. Empty restriction/handoff hooks are legitimate
baseline choices, not evidence that those transformations have been discovered.

## Learning after there is an execution signal

1. Keep the frozen 27B proposer as a control. Store siblings generated from the
   exact same prompt. Remeasure candidate pairs before constructing performance
   preferences; do not train on differences within timing noise.
2. Use accepted traces for a small supervised warm start if API compliance is the
   bottleneck. Avoid making all negative examples mere JSON errors: that only
   teaches formatting, which was the failure of the prior repair study.
3. Train a LoRA mutation policy with execution-ranked preference pairs. Include
   different successful lineages and old examples. Evaluate the trained and frozen
   policies at equal fresh-program and separately equal wall-time/token budgets.
   Keep task families and withheld cards out of both training and retrieval.
4. Only after diverse successful trajectories exist, consider multi-turn policy
   optimization with delayed reward for the best certified descendant. A useful
   reward is a capped negative log time-to-certificate with a fixed failure
   penalty. Construction and feasibility diagnostics can supply auxiliary
   signals, but cannot outweigh the actual solution objective.

Train on executable actions or code, not unchecked generated explanations. The
first valid programs already demonstrate why: some have incorrect mathematical
claims in their rationale while their actual executable update passes numerical
checks. Those explanations must not become proof labels or reasoning supervision.

The optional `typed_space.py` / `typed_campaign.py` path now supplies a compact
composition grammar using the existing FBS, DRS and relaxation construction rules.
It includes factorization policies, pixel restriction, finite FBS-to-DRS handoff,
precision, device, chunk width and loop unrolling. Canonicalization removes inactive
parameters, and LLM and random proposers use identical parameter enumerations.
Tests compare all three factor policies to an independently computed DRS update.
This is construction from standard methods, not invention of a new base algorithm.
The certificates for base maps do not automatically prove the additional retirement
and handoff adapter behavior. Full output acceptance remains independent.

The first free-code pilot completed 96 fresh candidates. Three generated programs
solved both search cases; none established the requested discovery advantage.
Function-level patches and exact failed-source repair are tracked as a separate
experiment. One early withheld-card patch produced FISTA and reached the hard
search case in 5.67 seconds, against a 4.17-second hidden-reference pilot. These
are exploratory single measurements and require confirmation.

Audit note: the first pilot used a common remote result path for identical seed
baselines on the same node. Candidate paths were distinct. Baseline measurements
from that pilot must not be used for precise paired speed claims. Later runs use
per-chain/per-device paths and verify the executed source hash before consuming a
result. Source snapshots and worker bundles preserve the later run versions.

RL cannot solve an empty action space, an uninformative fitness cell or a wrong
evaluator. Training a policy on the old keyword-scored recipes would reinforce the
wrong task. This first experiment intentionally determines whether there is
useful executed experience to train on. No weight training is claimed by merely
implementing the outer loop or exporting its data.

## Adversarial training means counterexample generation

### Solver league implemented after the user's clarification

`league_campaign.py` implements a second, complementary form of adversarial
search: competitors are executable solver programs. Off-the-shelf baselines set
the initial target. A challenger that wins a repeated contest becomes the next
incumbent. The next generation competes against that incumbent, the fixed initial
anchor, and a rotating archived opponent. Baseline source stays outside the
proposer's context; standard teaching references are allowed.

The initial league pilot resumes from the already LLM-generated active-set
program. It is a continuation experiment, not another independent rediscovery
from projected gradient. Its first opponent is the hidden ADMM audit port. The
program used as its seed is frozen before the continuation. Final evaluation of
the earlier seed remains a separate experiment.

Promotion requires every repeated output to pass the fixed independent judge,
a lower paired-bootstrap timing bound above 1.05x against the current champion,
and no per-case geometric-mean slowdown greater than 1.20x. Three paired repeats
are the pilot default. These intervals assess repeatability on the search cases;
they do not measure generalization or replace independent search seeds. Original
baselines remain on the scoreboard, so improvement is anchored rather than
defined solely by a moving opponent. Keeping older champions is a regression
check; it does not assert game-theoretic convergence.

Termination is explicit: a fresh-candidate budget, wall-time budget checked
between contests, a target speedup against the fixed anchor if specified, or
eight evaluated challengers without a promotion in this pilot. Syntax failures
and duplicates also have a bounded proposal-call budget. Exact same-prompt
siblings, executable programs, measurements and promotion decisions are saved.
Proposer sampling, parent choice, and paired measurement order have separate RNG
streams. The controller has an end-to-end mocked test that verifies successive
promotions, a fixed sibling prompt, and budget termination.

This implements the user's baseline-to-champion-to-next-champion idea. Model
weights remain frozen. Training on the resulting executed contests is a later
experiment; a league by itself is not RL. Repeatedly choosing better descendants
also does not guarantee the globally fastest algorithm. A diverse archive and
occasional alternative parents are needed to escape a single lineage.

The small first league stopped after one promotion against ADMM and then eight
challengers without another promotion. That first promotion inherited the earlier
LLM active-set method; it does not establish additional improvement over that
method. Larger-batch continuations use four training cases, including new 4,096-
and 16,384-pixel batches excluded from the original validation/test allocation.
Because the original test has now been inspected, `make_final_data.py` seals new
6,144- and 10,000-pixel batches plus a new synthetic case for these continuations.
`finish_recovery.py` records the pooled training-only selection rule, waits for
the designated leagues, freezes the chosen code, then runs the final tests.

Two additional treatments are explicitly exploratory. The executable-reference
arm adds tested, ordinary NumPy ADMM/DRS examples and a stronger hidden schedule
opponent. It changes both teaching and opponent coverage, so it is not an isolated
causal test of example teaching. The later typed hybrid space adds a generic
host-inverse action: compute the shifted inverse in host float64, transfer it to
the selected device and run the selected standard operator. This human-authored
primitive expands the action space in response to measured setup cost. It is not
a new LLM-discovered operator. A random proposer over the same expanded support
runs separately; results from the earlier, smaller grammar are not pooled with it.

`audit_paper_schedule.py` is a hidden, single-target adaptation of the paper's
`hsi/ours_v5.py` schedule, including device prechecks and pixel retirement. In one
A100 measurement it reached the 16,384-pixel training target in 29.6 seconds,
where the simpler ADMM audit port missed 60 seconds. It is a stronger comparator,
not an exact reproduction of the published multi-target H100 timing. Its source
is not given to the proposer and it is never an LLM seed.

### Optional problem adversary

Use an adversary to propose valid problem instances that separate the candidate
from a reference or expose numerical failure: condition number, rank deficiency,
support changes, nearly active constraints, batch size and hardware scale. Admit
only instances in the declared mathematical class. Maintain a mixture of ordinary
instances and informative failures; reward learning progress or calibrated regret,
not impossibility or a hacked certificate. Put discovered failures into a training
replay buffer. Never adapt the final test set or allow the adversary to edit the
judge. This is a later curriculum arm, distinct from the fixed-distribution pilot.

## Relevant primary sources

- [EvoTune](https://arxiv.org/abs/2504.05108) couples evolutionary exploration to
  preference optimization on evaluated programs, including diversity preservation.
  It is the closest direct precedent for training this mutation proposer.
- [RLEF](https://arxiv.org/abs/2410.02089) trains code models to use multi-turn
  execution feedback. Its coding results motivate the repair-policy experiment;
  they do not establish numerical solver discovery.
- [ShinkaEvolve](https://arxiv.org/abs/2509.19349) motivates archive selection,
  novelty rejection and diverse lineages before expensive evaluations.
- [SimpleTES](https://github.com/wq-will/SimpleTES) makes trajectories, local
  selection and execution-driven refinement explicit. Its released lasso solver
  is an external comparison, not a seed for a withheld-lasso experiment.
- [AlphaEvolve](https://arxiv.org/abs/2506.13131) demonstrates direct program edits
  evaluated by executable objectives across scientific and infrastructure tasks.
- [Absolute Zero](https://arxiv.org/abs/2505.03335) motivates executor-grounded
  self-play curricula. Its reasoning-task results do not guarantee that an
  adversarial solver curriculum will work here.
- [Hawkeye](https://icml.cc/virtual/2026/82754), read from the local paper
  `papers/62_Hawkeye_Hardware_Aware_GPU_.pdf`, pairs architecture-specific
  executable teaching examples with profiling metrics. Its useful lesson here
  is to verify that a proposed hardware change affected the intended cost. The
  new worker records restriction, handoff, output and certificate time separately
  and counts state-shape changes. This is stage profiling, not a hardware-counter
  implementation of Hawkeye or an automatic native-kernel generator.
- Standard teaching references: [Proximal Algorithms](https://web.stanford.edu/~boyd/papers/prox_algs.html)
  and [ADMM](https://web.stanford.edu/~boyd/papers/admm_distr_stats.html).

## Fleet and next decision

The authorized fleet is four A100s on node-0 and two on node-1. The H100 remains
the 27B proposer. A fresh local tunnel on port 8011 was opened because the older
8001 tunnel timed out. The local 4090 and its resident model are untouched.

Advance only on measured results. The desired result is that a reference-taught,
withheld-card LLM independently writes the important solver changes, reaches the
manual target on unseen instances, and improves over independent resampling under
matched budgets. Finding a working solver without that final comparison is useful
engineering progress, but does not establish that the evolutionary feedback caused
the discovery. No outcome, including a positive one, should be promised in advance.

## Revised success criterion: method recovery, not beating the reference

The user clarified that the target is automatic recovery of a method in the
paper. A new speed record is unnecessary. Earlier league promotions required a
repeatable speed improvement, so their zero-promotion counts do not directly
answer this revised question. Preserve those results as exploratory searches.

The prospective `recovery_v1` assay starts from the ordinary FBS design. It adds
standard ADMM as an executable algorithm atom and device-screened certification
as an execution policy to the previously restricted FBS/DRS compiler. These are
human-supplied standard primitives, equally available to the LLM and random
control. They make the HSI reference composition reachable; choosing them is
composition, not synthesizing a missing implementation or inventing ADMM.

The private target audit requires the compiled source hash to match its admitted
specification, actual ADMM dispatches, shared factor reuse, the known simplex
projection, observed GPU arrays, observed removal of active columns, and more
device screens than external certificate calls. All original columns still pass
the independent float64 certificate. The audit is not sent to the proposer.
Dispatch counts are Python calls into compiled functions, not physical CUDA
kernel counts. The ADMM adapter has independent numerical admission and
differential tests, but does not yet have an Atlas end-to-end proof certificate.

Four arms (`withheld_cards`, `full_cards`, `random`, `resample`) each receive two
seeds and eight fresh candidates, with identical action support. Every LLM arm
gets standard equations and executable textbook teaching examples. Only the
full-cards arm gets the four task-specific hints. Admission costs count toward
the fresh-candidate budget. Duplicates and format failures are recorded separately.
The starting FBS source is saved, not rebenchmarked in this new assay.

Search cases are the previously frozen Cuprite-4096 and Cuprite-16384 training
subsets plus the synthetic-96 case. Each has a 120-second algorithm budget.
Changing this cap from earlier 60-second pilots is prospective and explicit.
Successful recovery does not require beating a hidden reference runtime.

The first training recovery per LLM arm, selected chronologically across its two
seeds, is frozen for three repetitions on each of the three sealed `data_final`
cases. All nine must pass, together with the execution audit. This is an
exploratory first-success selection rule, not a powered comparison of discovery
rates. Test evidence never returns to search. Search still completes its fixed
fresh-candidate budget. The selected program must subsequently be checked at the
paper's full batch size and target ladder to support that exact performance
claim; this assay tests single-target method composition on held-out subsets.

Artifacts: `results/executable_discovery_20260926/recovery_v1/` and
`recovery_v1_confirmation/`. Controller: `recovery_campaign.py`; private auditor:
`recovery_space.py`; automatic freeze-and-confirm monitor: `watch_recovery.py`.
The earlier pooled speed-based final selector was stopped before launching any
sealed final evaluation so these cases remained available for this protocol.

### Semantic equivalence correction

The existing DRS atom already expresses the mathematical ADMM iteration for this
identity split, using a different carried state. With feasible initialization,
ADMM's `v = z + u` is the DRS carried variable and its feasible `z` is `P(v)`.
A differential check over 300 transitions, three proximal steps and four
relaxations found a maximum discrepancy of 2.14e-14. This corrects the earlier
suggestion that selecting DRS necessarily selects a different algorithmic core.
Adding a direct ADMM spelling was not necessary for mathematical expressivity;
adding device-screened certification was necessary for the full scheduled method.

The original literal-ADMM audit and all its results remain intact. A secondary
semantic audit (`recovery_equivalence.py`) recognizes this explicit state mapping,
while preserving every execution and certificate requirement. Its protocol is
stored in `recovery_v1_equivalence_confirmation/protocol.json`; it was added before
any candidate passed that audit or any sealed confirmation began. Both the literal
and semantic conclusions must be reported, rather than silently changing old hits.

Unambiguous SSH banner failures precede remote command execution. One such failure
for full-cards seed 0 proposal 3 is being explicitly remeasured and retained under
`infrastructure_rechecks/`. This does not replace the original search score, feed
additional evidence back to the proposer, or erase the extra evaluation cost.
