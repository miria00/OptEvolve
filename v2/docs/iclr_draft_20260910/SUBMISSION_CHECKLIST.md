# ICLR 2027 submission checklist and day-by-day plan

Written 10 September 2026. Covers the OptEvolve submission drafted in
`/home/miria/OptEvolve/v2/docs/iclr_draft_20260910/`.

Scope rules this document inherits and does not relax:

1. Nothing under `/home/miria/OptimizationAtlas/v2/OptAtlas_tex_2026` is
   edited. Changes to that manuscript are proposals for a coauthor.
2. Nothing under `/home/miria/OptEvolve/v2/atlas/` is modified.
3. No git write commands.
4. Every number that reaches the PDF is traceable to a file under
   `/home/miria/OptEvolve/v2/results/`. A number without a file is written as
   UNSUPPORTED and cut.
5. Style: no em-dash characters, no italics, plain declarative prose.

Source for every rule in Part A: `https://iclr.cc/Conferences/2027/AuthorGuidelines`,
`https://iclr.cc/Conferences/2027/CallForPapers`,
`https://iclr.cc/Conferences/2027/AIPolicyForAuthors`, and the local copy of the
template at `/home/miria/OptEvolve/v2/OptEvolve_tex_2026/iclr2027/iclr2027_conference.tex`.
All three web pages were fetched on 10 September 2026.

---

## Part A. The ICLR 2027 rules

### A.1 Deadline structure: two deadlines, one week apart

| Event | Date | Time |
|---|---|---|
| Abstract deadline | 18 September 2026 | 11:59 PM Anywhere on Earth |
| Full paper deadline | 25 September 2026 | 11:59 PM Anywhere on Earth |
| Reviews released | 5 November 2026 | |
| Author and reviewer discussion | 5 to 18 November 2026 | |
| Reviewer and AC discussion | 19 November to 16 December 2026 | |
| Final decisions | 16 December 2026 | |

The guidelines state the deadlines are final with no accommodations. Anywhere on
Earth is UTC minus 12, so 11:59 PM AoE on 18 September is 11:59 AM Pacific on
19 September. Do not plan against that extra half day. Treat the working
deadline as 18 September 23:59 local.

What the abstract deadline actually binds:

- The title and abstract must be entered in OpenReview by 18 September.
- No new authors may be added after the abstract deadline. Author order may
  still change up to the paper deadline.
- The abstract text can be revised up to the paper deadline, but the fact of a
  submission cannot be created after 18 September. A paper with no abstract
  registered on 18 September cannot be submitted on 25 September.

Action: register title, abstract, authors and subject area on OpenReview on
16 September, two days early, so a portal failure is survivable.

### A.2 Page limit

- Main text at initial submission: 9 pages or fewer. The guidelines call this a
  strict upper limit; the template says "There will be a strict upper limit of
  9 pages for the main text of the initial submission, with unlimited additional
  pages for citations. This limit will be expanded to 10 pages for
  rebuttal/camera ready."
- Overage is desk rejected.
- References and bibliography: unlimited, do not count.
- Appendix: unlimited, does not count, and reviewers are not required to read it.
- The AI use statement, the ethics statement and the reproducibility statement
  each sit outside the page limit.

Budget for our 9 pages, revised from the attack plan to match what is now
measured:

| Block | Pages |
|---|---|
| Introduction, including the width against tolerance definition and the contributions list | 1.25 |
| Related work and positioning | 0.75 |
| Method: the recipe, the construction gate, the execution layer | 1.25 |
| The factorisation and its evidence (iteration invariance, cost invariance) | 2.0 |
| Execution structure, fusion, the U shape, the wrong-tile negative | 1.75 |
| The omega0 rule and the two negative predictors | 1.5 |
| Limitations and discussion | 0.5 |

That totals 9.0 with no slack. The current draft is 1415 lines of TeX across six
files and has never been compiled. Compiling it early is the only way to learn
the real number. See D.10.

### A.3 Anonymity

Double blind. Reviewers cannot see author names and authors cannot see reviewer
names. The guidelines are explicit that "Any paper where author identity is
revealed in either the main text or the supplementary material will be desk
rejected."

Specific exposure surfaces for this project, all of which must be swept:

- Absolute paths. The draft sections and the results artifacts are full of
  `/home/miria/OptEvolve/v2/...`. The username `miria` is in every one. None of
  these may appear in the PDF or in the supplementary archive.
- The dataset cache path `data/kermany2017_oct/zacharielegault___kermany2017-oct/`
  appears in run logs. Cite the Kermany OCT dataset by publication, not by cache
  path.
- The `gh` remote, any repository URL, and the `Co-Authored-By` trailer used in
  commits. If the supplementary archive is built from a git checkout, strip
  `.git` entirely.
- Machine identifiers. The reports name "the Blackwell box is shared and all
  four GPUs held other users' jobs" and "the 4090 carried a resident vLLM
  server". Keep the substance of both caveats because they are honest
  measurement conditions, but write them without naming a lab, a cluster or a
  person.
- Acknowledgments and funding: omit from the submission version entirely.
  The template notes acknowledgments go in only for the final copy.
- PDF metadata. Check the Author, Creator and Producer fields after building.
- arXiv. Posting to arXiv during review is allowed by the dual submission
  policy. If a preprint exists, the submission must cite it in the third person
  and must not say "our earlier paper".

Action: a scripted anonymity sweep, run on the built PDF, that greps for
`miria`, `OptEvolve`, `OptimizationAtlas`, `/home/`, `github`, the coauthor
names, and any institution string. Run it twice: once on the extracted PDF text,
once on the supplementary archive listing.

### A.4 The four statements

The template at `iclr2027/iclr2027_conference.tex` lines 397 to 444 gives the
exact wording for three of these.

**AI use statement. REQUIRED.** Does not count toward the page limit. Should not
exceed 1 page. The AI policy distinguishes required disclosure from recommended
disclosure.

Required disclosure covers, in the policy's own list: generating synthetic
datasets, helping develop theoretical models or conceptual frameworks,
formulating mathematical claims, writing assistance for proofs, hypothesis
refinement, methodology design, implementation, translation, dataset processing,
data analysis, and result interpretation.

Recommended disclosure covers: survey design, figure creation, code editing,
literature analysis, brainstorming, structure suggestions, readability
improvements.

The policy states "Ultimately the paper's authors are responsible for the
contents of their submissions" and that substantial falsehood, plagiarism or
misrepresentation involving an LLM is a Code of Ethics violation that may lead
to desk rejection.

This project is unusually exposed here, and the honest statement is long. What
must be disclosed, based on how the work was actually done:

- Implementation. Large parts of `atlas/`, the CUDA kernel
  `results/kernel_fusion_20260909/tv_temporal.cu`, the diagnostic scripts under
  `scripts/diagnostics/`, and the benchmark families under `atlas/bench/` were
  written with LLM assistance. Required disclosure.
- Data analysis and result interpretation. The fANOVA decomposition, the regret
  analysis, the Simpson's paradox diagnosis in
  `results/diagnostics_20260909/REPORT.md`, and the causal reattribution from
  precision to loop structure were all produced with LLM assistance. Required
  disclosure.
- Literature analysis. The four-lane survey recorded in
  `docs/ICLR_SUBMISSION_ATTACK_PLAN.md` was LLM-assisted. Recommended
  disclosure, and it interacts with A.8 below: those citations have not all been
  checked against primary sources yet.
- Writing. The draft sections in this directory were LLM-drafted against a
  provenance contract. Recommended disclosure.

The statement must also say what verification was done. The strongest true
sentences available:

- Every numeric claim in the paper carries a source comment naming a file under
  `results/`, and the experiment sections state that all figures were recomputed
  from raw JSON rows on 10 September 2026.
- The fused kernel was admitted against a sequential reference: bitwise
  identical at rho equal to 1, and a worst deviation of 4.6e-15 at rho equal to
  1.9 over eight fused iterations, on both sm_89 and sm_120.
  Source: `results/kernel_fusion_20260909/REPORT.md` sections 2 and 5.
- The repository carries 386 or more tests. That number is from the skill
  ledger and not from a file under `results/`, so either find a test-run
  artifact or drop the count and say "a test suite" without a number.

Do not write anything in this statement that claims an LLM contributed a
research idea that a human did not verify. Do not claim the LLM comparison arms
as a result anywhere in the paper. See F.1.

**Ethics statement. RECOMMENDED.** At most 1 page, placed at the end of the main
text before references, and outside the page limit. This paper has one real
ethics surface: it uses the Kermany retinal OCT dataset, which is human subject
imaging. State the dataset provenance, its public release terms, that no
patient identifiers are used, and that no clinical claim is made. One short
paragraph is enough. Include it.

**Reproducibility statement. RECOMMENDED.** Paragraph length, at the end of the
main text before references, outside the page limit. The template is explicit
that this paragraph should reference the parts of the paper, appendix and
supplementary material that support reproduction rather than contain the details
itself.

Mandatory content for this submission, in addition to the pointers:

- The frozen final holdout defined by `scripts/make_cell_split.py` has never
  been evaluated. Say so in this statement, in one plain sentence, and say that
  no claim in the paper depends on it.
- The measurement conditions: the timed 4090 carried a resident inference
  server at 0 percent utilization during the 6 September campaign, and the
  Blackwell machine is shared and the runs were not exclusive. Both are already
  written down in `results/tv_ffi_20260906/DECOMPOSITION_REPORT.md` and
  `results/kernel_fusion_20260909/REPORT.md`. Carry them forward.
- The reference cell stability figure: `f64_K64` was re-measured in 12
  independent processes and spans 20.297 to 20.432 microseconds per iteration,
  a 0.66 percent spread.
  Source: `results/tv_ffi_20260906/DECOMPOSITION_REPORT.md`.
- The seed-noise floor for the omega0 experiment, so a reader can tell which
  regret improvements are real.
  Source: `results/finish_20260910/omega_seeds_4090.json`.

**Author contributions. OPTIONAL.** Skip for the submission version. It is an
anonymity risk with no upside.

### A.5 Supplementary material

- Authors may submit a single file that is the paper followed by supplementary
  text. Source code may be uploaded as part of the supplementary material.
- Reviewers are not required to review supplementary material.
- Everything in it is bound by the anonymity rule.

Consequence for us: nothing load bearing goes in the supplement. The decisive
comparison, the assumptions, and the negative results all belong in the 9 pages.

Plan for the supplement:

1. An anonymized source archive containing `atlas/`, `scripts/diagnostics/`,
   `results/kernel_fusion_20260909/tv_temporal.cu`, and the exact run commands.
   Build it with `.git` removed and paths rewritten.
2. The raw JSON artifacts the paper cites, or a manifest with SHA-256 hashes if
   the archive would be too large. `results/finish_20260910/` alone is 2.3 MB,
   so the raw JSON is affordable.
3. The frozen analysis script from D.3, so the regret numbers are recomputable
   from the raw rows by one command.

### A.6 Style files and formatting

- Official style files: `https://media.iclr.cc/Conferences/ICLR2027/iclr-2027-style-files.zip`.
- A local copy exists at `/home/miria/OptEvolve/v2/OptEvolve_tex_2026/iclr2027/`
  containing `iclr2027_conference.sty`, `.bst`, `.bib`, `fancyhdr.sty`,
  `natbib.sty`, `math_commands.tex` and the template `.tex`. The `.sty` and the
  template `.tex` are dated 28 July 2026; the rest are dated 25 June 2025.
- Verify the local copy against the published zip before the abstract deadline.
  A style file that has been silently revised is a page-count risk.
- The submission build must not define `\iclrfinalcopy`. That macro switches on
  the non-anonymous final layout.

### A.7 OpenReview and authorship

- All authors need OpenReview profiles.
- No new authors after 18 September. Order may change until 25 September.
- Maximum 20 co-authorships per person.
- At least one author must be registered to review at least 3 papers, and must
  hold a qualifying publication at the specified venues.
- An author on 3 or more submissions must review 6 or more papers.

Action: confirm the reviewing obligation is met and assigned before 16
September. This is an administrative desk-rejection route that has nothing to
do with the science and is easy to miss.

### A.8 Dual submission and citation integrity

Submissions substantially similar to previously published or accepted work are
prohibited. Papers on non-peer-reviewed sites such as arXiv are permitted, and
posting to arXiv during review is permitted.

This is fine for us. The live risk is not dual submission but citation
accuracy. `2-related-work.tex` carries a comment block listing keys that do not
yet exist in any `refs.bib`:

`Lu2026cuPDLP`, `Lu2026cuPDLPx`, `Lu2026MPAX`, `PDOT2024`, `cuOpt2026`,
`CarsonHigham2018`, `Brisebarre2026`, `CuClarabel2024`, `LiangYang2026`,
`cuOSQP2020`, `KempkeKoch2025`, `PushakHoos2022`, `Hutter2014fANOVA`,
`Xu2008SATzilla`, `BischlASlib2016`, `Zheng2020Ansor`, `KernelFoundry2026`,
`Su2012clSpMV`, `KernelBenchVerified2026`, `Wolf1991TemporalBlocking`.

Twenty keys. The only `refs.bib` in the tree is
`/home/miria/OptEvolve/v2/OptEvolve_tex_2026/refs.bib`, which holds
`Applegate2021Practical`, `Lu2024HPRLP`, `ORToolsPDLP2026`, `Gleixner2021MIPLIB`,
`Stellato2020OSQP` and `Goulart2024Clarabel` among others. Each of the twenty
must be created and checked against a primary source. The survey that produced
them is prose in this repository with no artifact under `results/`, so it sits
outside the claim-ledger discipline and needs its own verification pass. See D.9.

---

## Part B. Compliance status

| Requirement | Status | Owner action |
|---|---|---|
| Abstract registered by 18 Sep AoE | NOT DONE | Register 16 Sep |
| Paper submitted by 25 Sep AoE | NOT DONE | Target 24 Sep |
| 9 page main text | UNKNOWN, never compiled | D.10, compile 11 Sep |
| Double blind, no identity in text | AT RISK, paths everywhere | D.11 sweep |
| Double blind, no identity in supplement | AT RISK, archive not built | D.11 |
| AI use statement present | NOT WRITTEN | D.12 |
| Ethics statement present | NOT WRITTEN | D.12 |
| Reproducibility statement present | NOT WRITTEN | D.12 |
| Supplementary archive | NOT BUILT | D.13 |
| Style files current | UNVERIFIED, local copy dated 28 Jul 2026 | A.6 |
| Bibliography complete and checked | 20 KEYS MISSING | D.9 |
| Every number traced to `results/` | MOSTLY, three exceptions | C.3 |
| OpenReview profiles and review obligation | UNVERIFIED | A.7 |
| Frozen holdout disclosed as unevaluated | PLANNED, not written | D.12 |

---

## Part C. Evidence ledger

### C.1 Claims that are fully supported today

| Claim | Artifact |
|---|---|
| Iteration count is 584 / 597 / 683 at K=16/64/256, identical across schemes, precisions, execution structures and kernels | `results/tv_ffi_20260906/DECOMPOSITION_REPORT.md` section 3, `cadence_4090_v1/factorial.json` |
| The same iteration counts hold on sm_120 | `results/kernel_fusion_20260909/REPORT.md` section 3, `results/bw_20260909/` |
| Iteration count identical across loop structures on netflow, 24 of 24 | `results/netflow_20260909/grid_4090.log` and `grid_4090.json` |
| fANOVA: execution 15.98 percent, precision 2.90 percent, cadence 47.43 percent, execution by cadence 25.96 percent at 1e-4 | `results/tv_ffi_20260906/DECOMPOSITION_REPORT.md` section 2 |
| Algorithm identity is 0.00 percent at 1e-4 and 0.34 percent at 1e-6 of log time variance | same, section 4 |
| Float32 buys 1.00x in the device loop and 1.58x in host phases at K=256 | same, section 2 |
| Pure float32 solves 0 of 6 at 1e-6 and runs to the 5000 iteration cap | same, section 3 |
| The traced-bound loop penalty is per iteration and flat in outer trips: 11972 ns/iter at 32 trips, 10769 ns/iter at 1 trip, n=128 | `results/finish_20260910/loop_mechanism.json`, rows n=128 K=16 and K=512, traced minus static at cert false |
| The certificate term in that experiment is noise, from -240 to +845 ns across all 48 cells | same file, cert true minus cert false within each cell |
| Fusion numerical admission: bitwise identical at rho=1, 4.649e-15 worst at rho=1.9, on both devices | `results/kernel_fusion_20260909/REPORT.md` sections 2 and 5 |
| End to end 4090: 2.629 / 1.867 / 1.634 / 1.356 at n=128/256/384/496 | `results/kernel_fusion_20260909/e2e/full_4090_sz{128,256,384,496}.json` |
| End to end Blackwell: 1.732 / 1.450 / 1.625 / 1.149 | `results/kernel_fusion_20260909/e2e_bw/full_bw_sz*.json` |
| The wrong tile is worse than not fusing: 0.743x at n=496 | `results/kernel_fusion_20260909/e2e/full_4090_sz496.json` |
| Fusion is U shaped: 1.571x at 128 px, 1.060x at 992 px, 1.410x at 1984 px | `results/kernel_fusion_20260909/large/large_4090_m{2,3,4}.json` and REPORT section 7 |
| The static loop benefit reverses past about 1000 px: 0.989x at 1488 px | same |
| Step sizes are worth 4.31x on N and decide feasibility at 1e-6, balanced solves 0 of 6 | `results/method_axis_20260909/factorial.json`, REPORT section 8 |
| The gate `t*s*||L||^2 < 1` rejects the balanced parameters on netflow at 1.023 | `results/structure_20260909/structure_summary_4090.json`, field `raw_t025_s025_gate_lhs` |
| The gate admits or rejects as a function of theta only | same file, `gate_is_function_of_theta_only` |
| OT operator norm is 1.02 times sqrt(m+n) exactly | `results/finish_20260910/omega_ot_4090.json`, `Lnorm` 8.16 at `ot:32:32` against sqrt(64) equal to 8 |
| Roofline picks the best fusion tile 1 of 7 times, best model mean Spearman 0.567 | `results/artifacts_20260910/roofline_scores_4090.log`, `results/diagnostics_20260909/REPORT.md` lines 26-29, `roofline_4090.json` |
| Sharpness pooled r = +0.943 is Simpson's paradox, within netflow -0.539, within regular -0.842 | same, section 2, `sharpness_full_4090.json` |
| A leakage-free sharpness estimate needs more than half the solve to reach r = 0.620 | same |
| omega0 natively spans 6.969 to 9.125 across all cells before rescaling | `results/structure_20260909/structure_summary_4090.json` per-cell `omega0`, and the alpha equal to 1 rows of `results/finish_20260910/omega_{4090,bw}.json` |
| The omega0 sweep covers 8 cells, 1944 rows, 1510 converged, 209 instances with at least one converged run | `results/finish_20260910/omega_4090.json` plus `omega_bw.json`, recomputed 10 Sep |

### C.2 Claims that are supported but need a stated repeat count

The end to end fusion headline was re-measured at 9 repeats on two sizes and the
numbers move:

| n | repeats | best arm | speedup over traced baseline | source |
|---|---|---|---|---|
| 256 | 3 | tf24x4 | 1.867x | `kernel_fusion_20260909/e2e/full_4090_sz256.json` |
| 256 | 9 | tf24x4 | 1.940x | `finish_20260910/fusion_r9_sz256.json` |
| 496 | 3 | tf32x4 | 1.356x | `kernel_fusion_20260909/e2e/full_4090_sz496.json` |
| 496 | 9 | tf32x4 | 1.428x | `finish_20260910/fusion_r9_sz496.json` |

The argmax tile is unchanged at both sizes, which is the load-bearing claim. The
magnitude moves by 3.9 and 5.3 percent. The draft currently prints the 3-repeat
values in `1-introduction.tex` line 111 and `5-experiments.tex` line 201, and
`1-introduction.tex` line 125 already carries a comment noting the discrepancy.
Pick one repeat count, print it everywhere, and say in the caption which it is.
The recommendation is 9 repeats, since the 9-repeat run is the later and more
careful measurement, which means n=128 and n=384 need the same treatment. See D.7.

### C.3 Claims currently NOT traceable to a file under `results/`

These are the three that must be fixed or cut.

**C.3.1 The omega0 regret headline.** The figures 1.827x geomean and 23.03x
worst for a global constant ratio, 1.175x and 3.18x for the PDLP rule, and the
best-fit exponent of 2.00 appear in no file under `results/`. A grep for
`regret`, `1.827`, `23.03` across `results/` returns nothing. They were computed
ad hoc. `scripts/diagnostics/omega_sweep.py` only sweeps and writes rows; it
computes no regret.

This matters beyond bookkeeping. Recomputing from
`omega_4090.json` plus `omega_bw.json` on 10 September, with instances defined as
(cell, seed, alpha), an oracle taken as the minimum iteration count over the
ratio grid, and the constant C chosen to minimise geometric mean regret, gives
1.187x geomean and 3.867x worst for the PDLP rule over 209 instances. That is
close to but not equal to 1.175x and 3.18x. The global-constant row cannot be
computed at all under that convention, because no single ratio on the grid
converges on every instance, so the 1.827x and 23.03x pair must be using a
different convention for non-convergence. `1-introduction.tex` line 168 already
records that charging the 60000 cap instead gives 1.851 and 26.04, which
confirms the convention is doing real work.

The abstract, the introduction and the conclusion currently print three
different pairs: 1.181/3.87, 1.190/3.18 and 1.175/3.18. This is the single
largest correctness exposure in the draft. Fix by writing one frozen analysis
script. See D.3.

**C.3.2 The MCF gate contraction.** The claim that the multi-commodity flow
admissible region contracts as Theta(1/K) is derived in
`atlas/bench/mcf.py` lines 105 to 143, which is source and not a result. There
is no MCF artifact anywhere under `results/`. Either produce one or drop the
claim. See D.5.

**C.3.3 The test count.** The figure of 386 or more tests comes from the project
skill ledger and not from a file under `results/`. Either capture a test run into
`results/` or write "a test suite" with no number.

### C.4 A design flaw in the omega0 holdout

`results/finish_20260910/omega_holdout.json` sweeps cells `regular:256:2` and
`grid:32`. But `grid:32` is one of the four cells in
`results/finish_20260910/omega_bw.json`, which is half the 8-cell set the rule
was fitted on. Only `regular:256:2` is genuinely untouched.

Two honest options. Either re-run the holdout on two cells that appear in no
fitting file, which costs about 6 minutes, or report a one-cell holdout and say
so. Do not describe the current file as a two-cell holdout. See D.4.

### C.5 What the wide ratio grid already settled, and what it did not

The brief describes a wider grid as "running". It has landed:

- `omega_wide_bw.json`, 1404 rows, ratios from 1e-4 to 100, four cells.
- `omega_ot_wide.json`, 702 rows, same grid, two OT cells.
- `omega_holdout.json`, 702 rows, same grid, two cells.

Recomputed on 10 September, the exponent that minimises geometric mean regret is
2.1 on the narrow 8-cell set, 1.9 on the wide Blackwell set, 1.9 on OT wide and
1.8 on the holdout. Four independent datasets land between 1.8 and 2.1, which
supports the exponent 2 claim.

The caveat that must be stated is different from the one in the brief. On the
narrow grid the regret curve is nearly flat in the exponent: 1.193 at p=1.3,
1.183 at p=1.7, 1.183 at p=2.1, 1.193 at p=2.9. The narrow grid does not
discriminate p=2 from p=1.5. The wide grid does: 1.268 at p=1.1 against 1.193 at
p=1.9. So the correct sentence is that the exponent is identified by the wide
grid and not by the narrow one, and the paper should report a profile-regret
interval over the exponent rather than a point estimate of 2.00.

The naive log-log slope remains far from 2 even on the wide grid: -0.931 narrow,
-1.531 wide, -1.213 on OT wide, -1.469 on the holdout. Keep the censoring
explanation. Do not claim the slope fit recovers the exponent.

The 4090 half of the headline set was never re-run on the wide grid.
`omega_4090.json` still uses ratios from 0.01 to 100, and at omega0 equal to
776.456 only the two smallest ratios converge, so r star is pinned at the grid
floor. See D.2.

---

## Part D. What is missing, ranked, with cost on one GPU

Cost estimates are grounded in measured runtimes, not guesses. The calibration
points:

- The full omega0 sweep `omega_wide_bw.json` is 1404 rows and 536.0 seconds of
  summed solver execution, which was about 14 minutes of wall clock from the log.
- `omega_4090.json` is 972 rows and 347.6 seconds of execution.
- One end to end fusion sweep at one size, 9 arms by 6 images by 3 repeats, took
  15 to 21 seconds of wall clock, from the file timestamps 04:50:39 through
  04:51:33 for four sizes.
- The three montage sweeps took 53 and 77 seconds each, from timestamps
  13:08:37 through 13:10:47.
- The condition-number probe in `structure_20260909` did 2106 runs in 987.8
  seconds of GPU execution.

The conclusion from those numbers is important for the plan: the GPU is not the
bottleneck. Every missing experiment on this list totals under two hours of
compute. The bottleneck is prose, provenance, figures and the bibliography.

### D.1 Do NOT run: the frozen final LP holdout

`scripts/eval_holdout.py` and `scripts/make_cell_split.py` define a
pre-registered three-way split with a FINAL set that has never been evaluated.

Do not run it. It belongs to the search-centric paper that is being abandoned.
Running it would create a search result the current paper does not need and
cannot afford the pages to defend, and the brief is explicit that no
joint-search advantage may be claimed.

Instead, disclose it. One sentence in the reproducibility statement: the split
exists, it was frozen before any measurement, it has never been evaluated, and
no claim in this paper depends on it.

For the record, if it were run: 10 within-distribution netflow instances plus 48
OOD small plus 19 OOD medium is 77 instances, times 2 tolerances, times the
number of configs, with 60 and 120 second caps, on CPU with 20 workers. At 4
configs that is 616 tasks and an upper bound of 616 times 120 divided by 20,
about 62 minutes, with the realistic figure well under that because most tasks
do not hit the cap. Cheap, and still the wrong thing to do.

### D.2 Wide ratio grid on the four 4090 cells. 20 minutes.

Command shape, matching `omega_wide_bw.json`:

```
python scripts/diagnostics/omega_sweep.py \
  --cells grid:16,grid:24,regular:256:4,regular:512:4 \
  --alphas 0.01,0.03,0.1,0.3,1,3,10,30,100 \
  --ratios 0.0001,0.0003,0.001,0.003,0.01,0.03,0.1,0.3,1,3,10,30,100 \
  --theta 0.99 --seeds 3 --K 64 --tol 1e-4 --max-iters 60000 \
  --out results/finish_20260910/omega_wide_4090.json
```

1404 rows. Budget 20 minutes of wall clock on a clear 4090.

Why it is needed: the headline 8-cell number pools a censored 4090 half with an
uncensored Blackwell half. Once this lands, recompute the headline on the pooled
wide grid and report that as the primary number, with the narrow grid as a
robustness row.

### D.3 A frozen regret analysis script. No GPU. Half a day of work.

This is the highest priority item on the list, higher than any experiment.

Write `scripts/diagnostics/omega_regret.py` that reads a list of sweep JSON files
and emits a single summary JSON into `results/`. It must fix, in code and in the
output, every convention that is currently implicit:

1. What an instance is. Proposal: the triple (cell, seed, alpha).
2. What the oracle is. Proposal: minimum iteration count over the ratio grid,
   restricted to converged runs.
3. How non-convergence is charged. This is the convention that produces the
   1.827 against 1.851 difference. Proposal: charge the cap, since that is what a
   user actually pays, and report the alternative in a footnote.
4. How the constant C is selected. Proposal: the C that minimises geometric mean
   regret over the fitting set, reported to four significant figures, and snapped
   to the ratio grid at evaluation time because that is what a user can select.
5. Whether the exponent is fixed at 2 or fitted. Proposal: report both, with a
   profile-regret curve over the exponent so the flatness noted in C.5 is visible
   rather than hidden.
6. Instance counts, converged counts, and the seed-noise floor from
   `omega_seeds_4090.json` printed alongside every regret figure.

Then regenerate every regret number in the abstract, the introduction, the
experiments section and the conclusion from that one file. Right now those four
places print three different pairs.

### D.4 Re-run the omega0 holdout on two untouched cells. 6 minutes.

`grid:32` is contaminated by `omega_bw.json`. Pick two cells appearing in no
fitting file, for example `regular:256:2` and `regular:768:3`, and re-run at the
wide ratio grid. 702 rows, about 5 to 6 minutes based on the 320.9 seconds of
execution the current holdout file took.

If the schedule slips, the fallback is free: report the one untouched cell and
say the second was contaminated.

### D.5 An MCF artifact, or cut the MCF claim. 40 minutes, or zero.

The Theta(1/K) contraction is a closed form in `atlas/bench/mcf.py`, so the gate
half is O(1) to evaluate. A minimal artifact is a script that, for a few
topologies and K in {1, 2, 4, 8, 16, 32}, records the closed-form operator norm,
the maximum admissible t times s, and the largest K admissible at a fixed step
pair, and writes it to `results/mcf_20260910/gate_4090.json`. That part costs
minutes and no GPU.

Adding solve confirmation, so the claim is not purely symbolic, costs about 30
to 40 minutes on one GPU at netflow-comparable sizes.

Decision rule: MCF is the fourth family and the paper already has three. If
Sep 12 arrives and D.2 through D.4 are not finished, cut MCF from the paper and
keep OT, TV and netflow. Do not print a Theta(1/K) claim sourced to a code
comment.

### D.6 The full cadence grid with the rounding prediction. 15 minutes.

The attack plan asked for K in {8, 16, 32, 64, 128, 256, 512, 1024} with N(K)
reported as the crossing iteration rounded up to the next multiple of K, which is
a falsifiable prediction with no free parameters. What exists is K in {16, 64,
256} on TV and K in {64, 128, 256, 512} on the fusion cadence sweep.

This is the cheapest strong figure available. One TV cell, 6 images, 8 cadences,
2 execution structures, 3 repeats is comparable to two of the end to end fusion
sweeps, so 15 minutes. The payoff is a figure where a predicted curve with zero
fitted parameters overlays measured points.

If it is run, hold out two of the six images from any parameter choice and show
the prediction on those. That converts a description into a test.

### D.7 Nine-repeat fusion at n=128 and n=384. 5 minutes.

`fusion_r9_sz256.json` and `fusion_r9_sz496.json` exist. The two remaining sizes
do not. At 9 repeats the 15 to 21 second sweeps become 45 to 63 seconds each, so
both sizes cost about 2 minutes plus load time.

Without this the headline table mixes 3-repeat and 9-repeat rows.

### D.8 Uncertainty on every headline. No GPU beyond D.7.

Every table in the draft prints point estimates. At minimum, print the spread
across repeats for the fusion table and the spread across the 6 images for the
iteration counts. The existing artifacts already contain the per-instance and
per-repeat values, so this is an analysis change and not a measurement.

### D.9 The bibliography. No GPU. One full day.

Twenty keys, listed in A.8, do not exist. Each needs a real entry checked against
a primary source: authors, title, venue, year, volume, pages, DOI or arXiv ID.

Specific traps already identified and to be preserved as comments in the bib:

- `KempkeKoch2025`, arXiv:2503.10344, is titled with "low-precision" and means
  loose tolerance. It reports a 1.5x speedup from that. Annotate the entry so no
  future edit confuses it with arithmetic width.
- `KernelBenchVerified2026`, arXiv:2607.16241, June 2026, is unpublished and is a
  live scoop risk. `2-related-work.tex` already flags this as a coauthor
  decision.
- No accept tier may be stated for any cited ICLR paper. OpenReview is bot-walled
  and the URL slug is uninformative.
- The OR-Tools default of 64 for `termination_check_frequency` is the one
  third-party numeric claim printed in the related work section. Confirm it
  against the source file.

Assign this to a coauthor. It is the largest block of non-GPU work remaining and
it does not depend on any experiment.

### D.10 Compile the paper. No GPU. Do it on 11 September.

Six section files exist: `0-abstract.tex`, `1-introduction.tex`,
`2-related-work.tex`, `3-methods.tex`, `5-experiments.tex`, `6-conclusion.tex`.
There is no section 4, no `main.tex`, no `refs.bib` for this draft, and no
figures directory. Nothing has ever been compiled.

The gap at section 4 is deliberate in the old manuscript, where `4-theory.tex`
held the theory. Decide now whether the new paper has an assurance section or
whether the three assurance levels fold into the method section. The attack plan
budgets one page for assurance.

Until it compiles there is no page count, which means A.2 is unverifiable and the
desk-rejection risk is unquantified. This is the second highest priority item
after D.3.

### D.11 The anonymity sweep. No GPU. Two hours.

Write a script that extracts text from the built PDF and greps for: `miria`,
`/home/`, `OptEvolve`, `OptimizationAtlas`, `github`, `gitlab`, `zacharielegault`,
each coauthor surname, and each institution string. Run it on the PDF and on a
listing of the supplementary archive. It must exit non-zero on any hit.

Run it as the last step before every upload, not once.

### D.12 The three statements. No GPU. Three hours.

Draft the AI use statement per A.4, the ethics statement, and the reproducibility
statement. These are outside the page limit, so they cost nothing but time, and
the AI use statement is required and its absence is a compliance failure.

### D.13 The supplementary archive. No GPU. Three hours.

Per A.5. Build it, strip `.git`, rewrite paths, verify with D.11.

### D.14 Not planned, and the reason recorded

**LP decomposition.** The attack plan made this Priority 1 and it never landed.
`data/lp_suites/` holds MIPLIB and Mittelmann at 823 MB. A cadence by execution
grid on 20 LP instances at a 60 second cap is roughly 20 times 8 times 2 times 60
seconds, about 5.3 GPU-hours as an upper bound. That is affordable in wall clock
but not in pages or in the risk it adds two weeks out. Cut it, and state in the
limitations that the cadence result is measured on TV, netflow and OT and not on
the LP suites where the cited cadence constants originate. That is a real
limitation and it should be written plainly rather than hidden.

**5090 replication.** Out of scope per `results/kernel_fusion_20260909/REPORT.md`.
Two architectures is what we have. Do not make a portability claim.

---

## Part E. Day by day, 10 to 25 September

The GPU work totals under two hours. The plan therefore front-loads compute so it
is out of the way, and spends the remaining fourteen days on the things that
actually decide the outcome: a compiled page count, a frozen analysis, a real
bibliography, and an anonymous PDF.

### Thursday 10 September. Freeze the analysis, learn the page count.

- Start D.2, the wide ratio grid on the 4090 cells, in the background. 20 minutes.
- Start D.7, nine-repeat fusion at n=128 and n=384. 5 minutes.
- While those run: write `scripts/diagnostics/omega_regret.py` (D.3). Fix all six
  conventions in code. Emit one summary JSON into `results/`.
- Assemble `main.tex` from the six existing section files and compile (D.10).
  Report the page count tonight. This is the number that determines how much of
  the next week is cutting rather than writing.
- Decide the section 4 question: separate assurance section, or fold into method.

Exit condition for the day: a page count exists, and the regret script exists.

### Friday 11 September. Reconcile every number.

- Run D.3 over the pooled wide grid including the new 4090 file. Regenerate the
  regret headline.
- Propagate the single regret pair into `0-abstract.tex`, `1-introduction.tex`,
  `5-experiments.tex` and `6-conclusion.tex`. Right now those print three
  different pairs. After today they print one.
- Propagate the repeat-count decision from C.2 into every fusion table.
- Run D.4, the clean two-cell omega0 holdout. 6 minutes.
- Add the profile-regret curve over the exponent from C.5, so the flatness on the
  narrow grid is shown rather than asserted.

Exit condition: no number appears in two places with two values.

### Saturday 12 September. Decide MCF, run the cadence grid.

- Run D.6, the eight-point cadence grid with two held-out images. 15 minutes.
  This is the best figure-per-minute item remaining.
- Decide MCF (D.5). If in, run the gate artifact plus solves, 40 minutes. If out,
  delete every MCF sentence from the draft today rather than at the deadline.
- Begin D.9, the bibliography. Twenty keys. Assign to a coauthor if one is
  available; otherwise this becomes the dominant task of the next four days.

Exit condition: all GPU work for the paper is complete. Nothing after today
should require a GPU except a re-measurement forced by a reviewer-facing error.

### Sunday 13 September. Figures.

Four figures, drawn from artifacts that already exist:

1. N(K) with the rounding prediction overlaid, from D.6. Zero fitted parameters.
2. The execution by precision by cadence surface with the fANOVA decomposition,
   from `results/tv_ffi_20260906/`.
3. Fusion speedup against problem size on both devices, showing the U shape and
   the static-loop reversal, from `kernel_fusion_20260909/large/` and `bw_day2/`.
4. Regret against a per-instance oracle, constant rule against the omega0 rule,
   from the D.3 summary.

A fifth if pages allow: the two negative predictors from
`diagnostics_20260909/REPORT.md`, since the negatives are the argument for search.

### Monday 14 September. Cut to nine pages.

Whatever the Thursday compile said, this is the day it becomes nine. Cutting
order, least valuable first: MCF if still present, the fifth figure, the
cross-architecture cadence subsection, the montage size table beyond three rows.
Move nothing decisive to the appendix, because reviewers are not required to read
it.

### Tuesday 15 September. Statements and internal read.

- Write all three statements (D.12).
- First full internal read by someone who has not written any of it. Their brief:
  find every sentence that asserts causation without a mechanism, and every
  number without a source comment.

### Wednesday 16 September. Register on OpenReview. Two days early.

- Confirm all authors have profiles and the reviewing obligation of A.7 is met
  and assigned.
- Enter title, abstract, authors, subject area.
- Confirm the local style files match the published zip (A.6).
- Title decision. `TITLE_OPTIONS.md` exists in the draft directory; settle it
  today, because the title is bound at the abstract deadline in practice even
  though it can technically be edited later.

### Thursday 17 September. Buffer.

Held empty on purpose. Something will have slipped. If nothing has, use it on
the bibliography.

### Friday 18 September. Abstract deadline, 11:59 PM AoE.

Confirm the submission is registered. Do not treat AoE as extra time.

### Saturday 19 and Sunday 20 September. The evidence sections.

The two days of concentrated writing on sections 3 and 5, now that the numbers
are frozen and the page budget is known. Every table gets uncertainty per D.8.
Every negative result gets stated as a result rather than as a caveat.

### Monday 21 September. Claim audit.

Walk `docs/ICLR_CLAIM_LEDGER.md` and Part F of this document line by line against
the compiled PDF. One person, one pass, no writing. The output is a list of
sentences to delete.

### Tuesday 22 September. Supplementary archive and code drop.

D.13. Build it, anonymize it, verify it with the D.11 script, and check that the
frozen regret script reproduces the paper's regret table from the raw rows by one
command.

### Wednesday 23 September. Independent review.

Three passes, ideally three people:

1. Every number against its cited artifact. Not the report, the JSON.
2. Every citation against a primary source.
3. A clean-environment reproduction of one headline table from the supplementary
   archive.

### Thursday 24 September. Submit.

Build the final PDF. Run the anonymity sweep on the PDF text and on the archive
listing. Check PDF metadata. Confirm nine pages. Upload. Submit a day early so
that a portal problem is recoverable.

### Friday 25 September. Paper deadline, 11:59 PM AoE.

Reserved for a portal failure only. Nothing is written today.

---

## Part F. Sentences that cannot be written

Grep the compiled PDF text for each of these on 21 September. This list comes
from the four-lane survey in `docs/ICLR_SUBMISSION_ATTACK_PLAN.md` and is
reproduced here so the audit is mechanical.

### F.1 Cut entirely

- Any LLM advantage claim. The arms shared a proposal stream with 8 of 8
  byte-identical draws. This is the project's largest integrity exposure and the
  revised paper does not need it.
- Any joint-search advantage over sequential search. Joint tied sequential, and
  Pushak and Hoos, ACM TELO 2(3) Article 10, 2022, published that null in AutoML
  already. The factorisation explains it, which is a better outcome than hiding
  it.

### F.2 Cannot claim as novel

- Reduced-precision iterates with full-precision certification. NVIDIA cuOpt
  ships `CUOPT_PDLP_PRECISION`, merged 7 March 2026. CuClarabel is
  arXiv:2412.19027. Liang and Yang is arXiv:2605.31425.
- Certification in original coordinates. OR-Tools, cuPDLP.jl and cuPDLPx all
  state it; OSQP has `scaled_termination`.
- The float32 accuracy ceiling. It is a theorem: Carson and Higham, SISC
  40(2):A817 to A847, 2018; Brisebarre and coauthors, arXiv:2607.04828.
- Certificate cadence as an idea. OR-Tools defaults
  `termination_check_frequency` to 64 with an explicit comment that it involves
  extra work, and the field hard-codes 5, 25, 30, 40, 64, 100, 150, 200.
- Per-device policy selection. KernelFoundry ICML 2026, Ansor OSDI 2020, clSpMV
  ICS 2012. OSQP already ships float32 on CUDA and float64 on CPU.

### F.3 Numbers and phrasings to avoid

- Do not lean on any 1.41x figure. It sits below the published fp32 over fp64
  envelope in cuOSQP, JPDC 144:55 to 67, 2020, section 7.4, which also supplies
  the memory-bound explanation.
- Do not state an accept tier, poster, spotlight or oral, for any cited ICLR
  paper.
- Do not claim general accelerator portability. Two GPU architectures.
- Do not say the winning policy is hardware-specific in any place where both
  devices choose the same one. The honest version is in
  `kernel_fusion_20260909/REPORT.md` section 5: the devices agree on the optimal
  tile at n=128 and n=256 and disagree at n=384 and n=496.
- Do not print a Theta(1/K) MCF claim unless D.5 produces an artifact.
- Do not print a test count unless it comes from a file under `results/`.

### F.4 The terminology trap

Arithmetic width is the floating-point format of the iterates. Termination
tolerance is the threshold the certificate must meet. Define both in the first
paragraph and never let them share the word precision. Kempke and Koch,
arXiv:2503.10344, is titled "low-precision", means loose tolerance throughout,
and reports a 1.5x speedup from it, which is close enough to our numbers to
mislead a reader working from memory. `1-introduction.tex` already opens with
this separation. Keep it there.

### F.5 The scoop to manage

KernelBench-Verified, arXiv:2607.16241, June 2026, unpublished. Lead with the
factorisation, not with the kernel. The kernel is evidence for the factorisation,
not the contribution. `2-related-work.tex` carries a flagged coauthor decision on
whether to cite it at all.

---

## Part G. Final pre-flight, 24 September

Do not upload until every line is checked.

- [ ] Main text is 9 pages or fewer, measured on the built PDF, not estimated.
- [ ] References start on page 10 or later and are unlimited.
- [ ] AI use statement present, after the main text, at most 1 page.
- [ ] Ethics statement present, at most 1 page, covering the OCT dataset.
- [ ] Reproducibility statement present, one paragraph, and it states in plain
      words that the frozen final holdout has never been evaluated.
- [ ] `\iclrfinalcopy` is not defined anywhere in the build.
- [ ] No acknowledgments, no funding, no author contributions.
- [ ] Anonymity sweep exits clean on the PDF text.
- [ ] Anonymity sweep exits clean on the supplementary archive listing.
- [ ] PDF metadata Author, Creator and Producer fields carry no identity.
- [ ] Every number in the PDF appears in the D.3 summary file or in another named
      artifact under `results/`.
- [ ] The regret pair is identical in the abstract, the introduction, the
      experiments section and the conclusion.
- [ ] The fusion table states its repeat count and uses one repeat count
      throughout.
- [ ] Every bibliography entry has been checked against a primary source.
- [ ] No accept tier is stated for any cited ICLR paper.
- [ ] No em-dash characters. `grep -c "\xe2\x80\x94"` returns 0 on every source
      file and on the extracted PDF text.
- [ ] No italics used for emphasis.
- [ ] No LLM advantage claim and no joint-search advantage claim appears
      anywhere, including the appendix.
- [ ] The supplementary archive reproduces one headline table by a single
      command in a clean environment.
- [ ] Submitted, with a confirmation screenshot, before 24 September ends.
