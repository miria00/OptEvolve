**Paper status and end goal, 6 September 2026**

The governing [attack plan](ICLR_SUBMISSION_ATTACK_PLAN.md) remains in force.
This document distinguishes measured findings from the evidence the intended
submission still needs. The current assessment remains likely reject on research
evidence, despite a substantial implementation and a more accurate manuscript.
This is a research judgment, not an estimated acceptance probability.

The working title is **OptEvolve: Hardware-Aware Search for Certified Optimization
Solvers**. Keep hardware explicit. Reserve a stronger title such as
**OptEvolve: Joint Algorithm–Kernel Search for Certified Optimization** for an
experiment showing a meaningful benefit from the coupling. Do not put LLM
discovery in the title without demonstrating what the LLM adds.

**The paper's question**

For a declared problem distribution, hardware target and accuracy requirement,
when does optimizing the mathematical solver together with its implementation
reduce time to a checked solution beyond strong independent or sequential tuning?

An executable recipe includes the splitting and its parameters, supported
wrappers, kernel variant, precision policy and check cadence. Its evaluation
must include solve failures and numerical stopping checks, while exposing
compilation, preparation and search costs. There are three separate assurance
levels: mathematical convergence under stated assumptions; numerical admission
of an implementation against reference behavior; and a stopping check on the
original problem. None implies formal floating-point correctness by itself.

**Where we are**

| Component | Evidence now | What it establishes |
|---|---|---|
| Manuscript | Reconciled to v2; anonymous draft has eight main-text pages | A reviewable current draft, not submission readiness |
| Construction and checking | Typed gate, scoped proofs, independent numerical checks and regression tests | A foundation for controlled search; remaining proof obligations are explicit |
| Hardware-policy tuning | Six-validation-image TV diagnostic: 1.47x on 4090 and 1.57x on 5090 over compiled f64/K64 at 1e-4 | A development execution-policy gain beyond the legacy denominator |
| Hardware conditioning | CPU retains f64/K64; both GPUs choose refinement/K256 | A preliminary CPU/GPU difference; no GPU-to-GPU specialization at that tolerance |
| Joint selection | Joint equals implementation-first sequential selection on both GPUs at 1e-4 | No demonstrated benefit over both sequential orders |
| Tight tolerance | Selected policies solve 5/6 at 1e-6 under 5,000 iterations; pure-f32 CV solves 0/6 | Accuracy and failure handling materially affect the comparison |
| Native kernels | LP variants lose to XLA; TV FFI variants now implement both distinct schemes and pass admission/trajectory checks | The same-cell algorithm/kernel implementation gap is being closed; speed and search gains require measurement |
| LLM guidance | Corrected pilot is too small and fallback-heavy; old coupled-RNG comparison excluded | No demonstrated LLM advantage |

The previous milestone is documented in
[its execution report](../results/attack_plan_20260906/REPORT.md).
The current TV kernel measurements and code changes are documented in
[the kernel report](../results/tv_ffi_20260906/REPORT.md).

**Contributions to aim for**

1. A precisely specified, construction-gated search over executable solver
   recipes. The contribution is the supported contract and what it enables,
   not priority for verifier feedback, evolutionary search or certificate cadence.
2. A demonstrated algorithm/implementation interaction. Show that joint
   selection improves the declared held-out objective beyond both sequential
   orders under matched budgets, or establish a narrower predictive result
   about when hardware-aware execution choices help and when they do not.
3. A causal account of those outcomes. Measure updates, checking, precision
   transitions, detection delay and generated-kernel behavior. Predict an effect
   from development measurements, then test it on reserved data or hardware.

A reusable, auditable artifact supports those contributions. A working artifact
and several standard engineering speedups are not automatically a sufficient
research contribution. The repair-feedback negative can remain a scoped
secondary result or appendix; it should not compete with the main hardware story.

**Final comparisons**

| Comparison | Required controls |
|---|---|
| Algorithm versus implementation versus joint | Strong fixed recipe; algorithm-only; implementation-only; algorithm-then-implementation; implementation-then-algorithm; joint |
| Search controller | Independent random search, typed population mutation and a competent automatic configurator over the same eligible space |
| LLM, if claimed | LLM with absent, identity-only and measured hardware context; identical execution budgets, explicit invalid/duplicate/fallback accounting |
| Native kernel contribution | Same algorithm, precision and cadence with XLA versus each admitted FFI variant; include the best eligible XLA precision/cadence policy |
| Numerical/execution mechanism | f64 versus f32 with f64 checks versus refinement; multiple cadences including 64; fused versus reference execution; return accuracy and failures |
| Practical solver context | Tuned PDHG and Condat-Vu for TV; production PDLP and HiGHS if making practical LP performance claims; relevant established solvers for any new family |

Match starting information, data splits and eligible grammar. Report budget
equality in fresh executed evaluations and elapsed search time separately.
Predeclare the sequential budget split. A complete catalog compares achievable
selections but does not establish search efficiency. Targets such as ten search
seeds are design choices, not ICLR acceptance rules.

**Final benchmark package**

- TV/OCT remains the main working cell. Use patient-group-disjoint selection,
  validation and final evaluation, retain failures, and include multiple image
  sizes and tolerances. Aim for at least 50 independent held-out patients if
  available; repeated timings do not increase this count. Previously inspected
  public test images must not be relabeled as an untouched final test.
- LP remains a certification and sparse-hardware benchmark, with a declared
  representative suite and production-solver context if performance is claimed.
  Preserve the failed calibration and its frozen final partition. Scheme
  relabeling on standard LP does not establish structural discovery because
  the smooth term is affine and the current PDHG/CV maps coincide there.
- For a broad structural-search claim, validate a second meaningful composite
  family before scaling. Sparse least squares with l1 regularization is a
  candidate, but its IR, stopping criterion, solver maps and strong baseline
  must be implemented and checked first. One dataset with many parameter
  settings is not many independent datasets. If that work does not fit the
  deadline, narrow the claim to the families actually validated.
- Measure CPU and both available GPUs, keep resource conditions explicit,
  and test transfer in both directions. Same-family GPUs are a limited hardware
  sample; do not claim general accelerator portability from two GeForce cards.
- Report solve fraction, time-to-criterion distributions with censoring,
  paired uncertainty at the independent problem/group level, cold and warm
  performance, compilation/preparation/search/model cost, and the reuse count
  needed to amortize optimization. Keep the final evaluation untouched until
  recipe selection and analysis decisions are frozen.

**Remaining attack plan**

| Stage | Status | Remaining result |
|---|---|---|
| P0: claims and protocol | Substantially implemented | Complete the inherited bibliography/proof audit, final timing protocol and strong automatic tuning baseline |
| P1: interaction and native TV kernels | Catalog diagnostic complete; native TV implementation and comparison in progress | Decide whether there is practical, repeatable interaction after competent tuning; current catalog does not establish it |
| P2: main evidence | Outstanding | Matched-budget search runs, repeated seeds, larger independent problem groups and validated second-family scope |
| P3: explanation | Initial TV/LP traces and numerical failures available | A predicted ranking or cost change that survives an independent test, with component ablations |
| Final evaluation | Outstanding | Freeze choices, evaluate final holdout once, report failures and uncertainty |
| Submission package | Outstanding | Final narrative/figures, reproducible anonymous artifact, completed AI-use inventory and independent claim/source review |

The practical decision is due by September 10: if joint search has useful
headroom, scale the matched controls. If it does not, pursue a narrower
hardware/certificate result only if it yields new, supported knowledge.
Do not manufacture a joint-discovery claim from different genome strings,
small timing fluctuations, or improvements over an avoidably weak baseline.

ICLR's reviewer guide asks whether a submission adds sufficient value and new
knowledge, and evaluates correctness, experimental rigor, reproducibility and
novel findings. SOTA alone is not the definition of a contribution.
[Official reviewer guide](https://iclr.cc/Conferences/2027/ReviewerGuidelines).
The abstract is due September 18 and the paper September 25, 2026, both at
23:59 AoE. [Official call for papers](https://iclr.cc/Conferences/2027/CallForPapers).
