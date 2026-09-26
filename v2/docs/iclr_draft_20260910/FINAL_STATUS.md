# FINAL_STATUS: whole paper pass, 10 September 2026

Scope: all six section files in `/home/miria/OptEvolve/v2/docs/iclr_draft_20260910`,
read in order 0-abstract, 1-introduction, 2-related-work, 3-methods, 5-experiments,
6-conclusion. Every number below was recomputed on this pass by opening the named
file under `/home/miria/OptEvolve/v2/results/`. Where a value did not reproduce it is
reported with the value that did.

Style contract on this file: zero U+2014, zero U+2013, zero triple hyphen, no italic
markup, pure ASCII.

## 0. The single most important finding first

Four of the six files (abstract, introduction, experiments, conclusion) went through
a correction pass. **Two did not: `2-related-work.tex` and `3-methods.tex`.** Those two
still carry, verbatim, three of the six original refuted claims from
`FACTCHECK_DEFECTS.md`, including the two that were quoted in the brief as
representative defects. This is a mechanical omission, not a research problem, and it
is the cheapest blocking item on the list.

## 1. Section table

"Numbers stated" is the count of numeric tokens in printed body text and tables,
after stripping comments and LaTeX layout arguments. "Traced" means the value was
recomputed from an opened file under `results/` and agreed to the digits printed.
"Unresolved" counts distinct printed values that failed recomputation, are attached to
a scope under which they do not reproduce, or have no artifact under `results/` at all.

| Section | Words | Numbers stated | Traced to an artifact | Unresolved |
|-|-|-|-|-|
| 0-abstract | 180 | 22 | 19 | 3 |
| 1-introduction | 1947 | 351 | 335 | 16 |
| 2-related-work | 956 | 83 | 74 | 9 |
| 3-methods | 1363 | 166 | 159 | 7 |
| 5-experiments | 4717 | 942 | 931 | 11 |
| 6-conclusion | 2809 | 459 | 441 | 18 |
| **total** | **11972** | **2023** | **1959** | **64** |

Supporting mechanical facts:

* Style: `file -bi` reports `us-ascii` for all six files. Zero U+2014, zero U+2013,
  zero occurrences of the triple hyphen, and zero italic or emphasis macros of any
  spelling. Clean.
* Provenance: 213 source comments across the six files, naming 100 distinct paths
  under `results/`. Exactly one named path does not exist:
  `results/artifacts_20260910/roofline_scores_4090.log`, which the draft itself flags
  as an outstanding coauthor action at `5-experiments.tex:745`.
* Citations: 26 keys are cited, all of them from `2-related-work.tex`. Only 6 exist in
  `OptEvolve_tex_2026/refs.bib` (Applegate2021Practical, Gleixner2021MIPLIB,
  Goulart2024Clarabel, Lu2024HPRLP, ORToolsPDLP2026, Stellato2020OSQP). **20 keys are
  undefined.** The paper does not compile with resolved references today.
* Forbidden claims: the four prohibitions are honoured with explicit printed
  disclaimers in sections 0, 1, 5 and 6. Section 2 prints no disclaimer and contains
  one sentence that reads as a general machine-independence claim (see A3 below).

## 2. Claims that appear in two sections with different numbers

This was the hunt the brief asked for, and it is where the damage is. Twenty-three
conflicts survived the piecewise pass. They are grouped by what a reviewer would do
with them.

### 2A. The paper reports three different fitted rules and four different pooled numbers

This is the worst cluster, because the omega0 rule is the headline result and every
section quotes it differently.

**C1. The rule itself.** All four sections say the rule was fitted on the netflow
sweep alone. They give three different rules:

| where | rule as printed | geomean regret on the fit set (recomputed) |
|-|-|-|
| `0-abstract.tex` ledger, lines 55 to 57 | C = 1.10, p = 2.00 | 1.1881 |
| `1-introduction.tex:249` and Table 2 caption | C = 1.16, p = 1.8 | 1.1981 |
| `5-experiments.tex:589`, `6-conclusion.tex:129` | C = 0.977, p = 2 | 1.1858 |

The joint argmin on `results/finish_20260910/omega_wide_bw.json`, minimising geometric
mean regret with C searched on a 0.01 dex log grid, is p = 2.00, C = 0.9772, regret
1.18583. The experiments and conclusion value is the fit. The introduction value is
1.03 percent worse than the fit on the set it claims to have been fitted on.

**C2. Pooled fixed-ratio baseline over the 162 held-out instances.** Four numbers for
one sentence-level claim: `1.82` (abstract), `1.825` (intro Table 2), `2.504`
(experiments Table 5), `2.784` (conclusion Table 2). All four recompute, under four
different protocols:

* 1.8249 is the geometric mean of three per-family best fixed ratios, each scored only
  where that ratio converged, so on 54 + 44 + 45 = 143 instances, not 162.
* 2.5039 is one shared ratio (r = 0.1) across all three families, scored where it
  converged (130 of 162).
* 2.7843 is one shared ratio (r = 3) with capped runs charged at 60000 iterations, on
  the 11 point ratio grid common to the three sweeps.
* 2.8204 is the same thing on the full per family grids.

Nothing in the printed text of the abstract or introduction says the 1.82 figure is a
per family aggregate scored on 143 instances.

**C3. Pooled rule regret.** `1.12` (abstract), `1.125` (intro), `1.131` (experiments),
`1.111` (conclusion). Recomputed: 1.1241 at C = 1.10 p = 2; 1.1252 at C = 1.16 p = 1.8;
1.1311 at C = 0.977 p = 2 on the full grids; 1.1105 at C = 0.977 p = 2 restricted to
the 11 point common grid. Every one reproduces; they are four different rules or grids
presented as one result.

**C4. Fixed-arm coverage.** Experiments Table 5 gives 87 / 54 / 44 / 45 and 130 of 162.
Conclusion Table 2 gives 103 / 54 / 49 / 47 and 131 of 162. The experiments column is
the coverage of the best-regret ratio; the conclusion column is the coverage of the
best-coverage ratio, which is a different ratio on four of the five rows. Only the
conclusion caption discloses the switch, and it says "three of the five rows" when the
count is four.

**C5. The same protocol name, two answers.** `6-conclusion.tex:141` introduces "the
alternative protocol in which each policy is scored only on the instances it
certifies" and gives 1.927 on 32 of 54 for multi-commodity flow and 1.897 on 28 of 54
for the held-out topologies. `5-experiments.tex` gives, under the same description,
2.185 on 44 of 54 and 1.947 on 45 of 54. Both recompute. The conclusion drops the 80
percent coverage floor for two rows and keeps it for the other two.

**C6. No-floor pooled fixed ratio.** `1.752` on 108 of 162 (experiments line 651)
versus `1.732` on 108 of 162 (conclusion line 144). Both are r = 30; the difference is
the full grid versus the 11 point common grid, stated in neither sentence.

### 2B. The exponent is characterised four different ways

**C7.** `0-abstract.tex`: "p is not identified". `1-introduction.tex:295`: "The
exponent is approximately 1.8". `5-experiments.tex:707`: "This grid localises the
exponent to roughly 1.8 to 2.3 and no further". `6-conclusion.tex:247`: "bracket the
exponent between about 1.7 and 2.1". Four verdicts, one quantity.

Recomputed on all five grids, with C refitted at each exponent: the regret-minimising
exponent is 2.10, 2.00, 1.90, 2.00, 2.00 and the log-log slope of the argmin is
-0.9365, -1.5412, -1.5781, -1.6834, -1.7380 under the geometric tie rule. The value
"approximately 1.8" is not the argmin on any of the five grids. The introduction's own
Table 2 rule uses p = 1.8 and the experiments and conclusion tables use p = 2.

**C8. Regret at p = 2.0 on the ten-decade grid.** `1.066` (intro line 304), `1.0618`
(abstract ledger), `1.0574` (experiments comment at line 734). Recomputed on
`omega_ultrawide_bw.json`: 1.06177 with C on a 0.01 dex grid, 1.05739 on a 0.001 dex
grid. The introduction's 1.066 is not reachable at either resolution. p = 1.8 gives
1.0702 under both and is quoted correctly everywhere.

**C9. Censoring counts.** Intro and experiments: 102 of 209, 33 of 108, 46 of 144, 15
of 80, 14 of 88. Conclusion: 101 of 209 (91 floor plus 10 ceiling), 32 of 108, 46 of
144, 14 of 80, 14 of 88. Recomputed, counting an instance as censored if any tied
argmin sits at a grid endpoint: 102 (91 floor, 11 ceiling), 33, 46, 15, 14. Counting
the geometric mean of the tied argmins instead: 100, 31, 45, 14, 14. The conclusion's
101 and 32 match neither convention.

**C10. Log-log slopes.** Intro and experiments print -0.94, -1.54, -1.58, -1.68,
-1.74. Conclusion prints -0.93, -1.53, -1.57, -1.68, -1.73. Both reproduce: the first
set is the geometric tie average (-0.9365, -1.5412, -1.5781, -1.6834, -1.7380), the
second is ties broken to the smallest ratio (-0.9309, -1.5307, -1.5716, -1.6762,
-1.7324). Four of the five printed values differ between sections and neither section
names its tie convention in the printed text.

**C11. The one percent band on the narrow grid.** Experiments line 705: "every
exponent from 1.4 to 2.6 is within 1 percent of the optimum". Conclusion line 246:
"1.15 to 2.90 on the narrow pooled grid". Recomputed, the within-one-percent set is
`{1.15} union [1.4, 2.9]` at 0.01 dex in C and `{1.15, 1.2} union [1.4, 2.9]` at 0.001
dex. It is not an interval, so the conclusion's phrasing is wrong in form as well as in
endpoints, and the two sections disagree about both endpoints.

### 2C. Direct contradictions of fact between sections

**C12. Is cadence the only shared gene?** `1-introduction.tex:79`: "Cadence is the
only gene appearing in both factors, so it is the sole coupling point of the two",
reinforced by the paragraph heading at line 85 and by `2-related-work.tex:100` and its
Table 1 row. Against this, `6-conclusion.tex:423`: "It is not the only gene in both.
Size appears in both factors as well, and precision changes N at some sizes". And
`0-abstract.tex` asserts a stronger thing than either: "the two depend on disjoint
parts of a solver recipe", which asserts no shared gene at all. Three positions, one
claim, and it is the paper's thesis sentence.

The data side with the conclusion. `results/artifacts_20260910/fanova.json` shows
cadence moving the per-iteration cost as well as the iteration count: host phases at
float64 run 27.18, 16.19 and 13.29 microseconds at K = 16, 64 and 256, a 2.05 factor,
and host phases at float32 run 28.00, 12.23 and 8.43, a 3.32 factor.

**C13. Mean iteration count on TV at K = 256 and tolerance 1e-4.** This is the most
serious single conflict in the paper. The catalog factorials give 682.667 at n = 256,
and that number is the load-bearing constant of the abstract, the introduction, section
2, section 5 and section 6. The fusion end-to-end sweep, quoted at
`1-introduction.tex:88` and again in the conclusion, gives 640.0 at the same size,
cadence and tolerance. Recomputed at all four sizes:

| n | catalog factorial, K = 256 | fusion end-to-end sweep, K = 256 |
|-|-|-|
| 128 | 682.667 | 554.667 |
| 256 | 682.667 | 640.000 |
| 384 | 853.333 | 810.667 |
| 496 | 1152.000 | 1024.000 |

Sources: `tv_size_20260909/sz{128,256,384,496}_4090/factorial.json` against
`kernel_fusion_20260909/e2e/full_4090_sz{128,256,384,496}.json`. Both harnesses run
the same genome, condat_vu with tau 0.05, sigma 1.0, rho 1.9, six OCT images at the
same size, tolerance 1e-4, K = 256. The catalog artifacts archive an instance manifest
(`data.json`, first instance index 14958, identical `input_sha256` across the catalog
runs). **The fusion sweep archives no instance manifest at all**, so the difference
cannot be attributed to a different instance population from any file under `results/`.
As the archive stands, the paper prints two different values of N for the same stated
configuration and asserts that N is a function of cadence and tolerance for a fixed
instance.

**C14. Per-iteration cost of the two schemes at f64_K256 in `method_axis`.**
`5-experiments.tex:104` prints "19.76 against 19.76 microseconds" and cites
`method_axis_20260909/factorial.json`. `6-conclusion.tex` Table 3 prints 19.798 and
19.683 for the same two cells of the same file. Recomputed from that file: 19.7976
(Condat-Vu) and 19.6826 (PDHG) as the mean of per-instance solver_s over iters, or
19.7834 and 19.6834 aggregated. The pair 19.7628 and 19.7619 exists, but in
`tv_ffi_20260906/kernelhost_4090/factorial.json`, whose own penalised SGM-1 ratio is
1.00093 and not the 0.99494 printed in the same sentence.

**C15. The gate product.** `3-methods.tex:130`: "Across all 2673 rows of the three
sweeps the gate product is exactly 0.990000000 and the ratio is exactly r".
`5-experiments.tex:487`: "equals 0.99 to within 3.4e-16 in all 1944 rows and exactly in
774 of them". Recomputed over `omega_4090.json`, `omega_bw.json` and
`omega_ot_4090.json`: 2673 rows, the product is exactly 0.99 in 882 of them, tau/sigma
is exactly r in 1575 of them, and the maximum deviation is 3.331e-16. Section 3 is
wrong; section 5 is right except that 3.331e-16 rounds to 3.3e-16, which is what
`1-introduction.tex:240` prints. So the intro and the experiments section also disagree
in the second digit of the same quantity.

**C16. The device of the admission worst case.** `3-methods.tex:265`: worst deviation
"4.6e-15 ... on the 4090, with 4.649e-15 on sm_120". The archived artifact
`results/artifacts_20260910/kernel_admission_4090.header` records `device: RTX 4090
(sm_89)` and the log's final lines read "worst absolute deviation across all cases:
4.649e-15" and "PASS". Sections 1, 5 and 6 all attribute 4.649e-15 to the 4090. No
Blackwell admission log exists under `results/`.

**C17. Is the admission log archived?** `3-methods.tex:267`: "The admission script is
stored and rerunnable but its output log is not archived, and the reproducibility
statement records this". `5-experiments.tex:266`: "The archived admission run".
`6-conclusion.tex:455`: "The numerical admission run for the temporal fusion kernel is
archived with the other artifacts". The brief is explicit that any sentence of the
first kind is now false and must go. It is still there, and `INTEGRATION.md:661` and
`TITLE_OPTIONS.md:55` repeat the same stale premise.

**C18. Bitwise identity.** `3-methods.tex:264` claims "bitwise identity at rho = 1".
`5-experiments.tex:271`: "No case is bitwise identical at this rho".
`6-conclusion.tex:460`: "No case is bitwise identical". The archived log contains 102
rows, every one at rho 1.9 with the bitwise column reading "no". There is no rho = 1
row anywhere under `results/`, so section 3's claim is sourced only to REPORT.md prose.

**C19. The 584 / 597.3 / 682.7 triple under a hand written kernel.**
`2-related-work.tex:56` still says the triple "recurs under both arithmetic widths,
both execution structures, XLA and a hand written kernel, and a second GPU
architecture". `tv_ffi_20260906/kernelhost_4090/factorial.json` holds only K = 64 and
K = 256 policies; there is no cadence 16 row for the custom kernel anywhere under
`results/`. The abstract's own ledger records this as deleted defect (6).

**C20. The float32 tolerance ceiling.** `2-related-work.tex:82`: Condat-Vu with float32
iterates "solves 0 of 6 instances at 1e-6 at all three cadences and PDHG solves 1 of 6,
on both architectures". Recomputed: on the 4090 (`f32row_4090`) this holds at all three
cadences. On the Blackwell the float32 arm exists at K = 64 only, and across sizes it
gives 1 of 6 and 3 of 6 at n = 128, 0 of 6 and 1 of 6 at n = 256, and 0 of 6 and 0 of 6
at n = 496. The introduction states the same result correctly, scoped to n = 256.

**C21. What float32 buys in a chunked loop.** `2-related-work.tex:78` and its Table 1
row: 1.569x. `5-experiments.tex:163`: 1.576x. Both reproduce from the same four
artifacts: 0.00903912 / 0.00576135 = 1.5689 on penalised SGM-1 seconds, and 13.2907 /
8.4316 = 1.5763 on mean microseconds per iteration. The iteration count is 682.667 in
both cells, so a reader is entitled to expect one number.

**C22. Cadence's share of the variance.** 47.43 percent in the introduction, the
experiments section and the conclusion; 39.44 and 60.98 percent in
`2-related-work.tex:98` and its Table 1 row. Both are in `fanova.json` and both are
correct, but they are shares of different responses (log per-iteration cost versus log
penalised SGM-1) over different factor sets, and section 2 never names its response.

**C23. Two smaller ones.** The worst-tile penalty at n = 496 is 0.778x in the abstract
(median of 9, `fusion_r9_sz496.log`) and 0.743x in the introduction, section 2 and the
experiments table (median of 3), while the introduction's own protocol note at line 189
says "Never mix the two protocols". And the fusion timing drift is "4 to 8 percent" in
the introduction, "4.0 and 7.8 percent" in the experiments section, and "4 to 5
percent" in the conclusion falsifier table; recomputed it is 3.9 and 5.3 percent on the
best-tile speedup and 4.0 and 7.8 percent on the static-over-traced gain.

## 3. What a reviewer can still attack, ranked

Severity is judged by what a reviewer can do with the finding, not by the size of the
arithmetic error.

### Tier 1: attacks that reach the thesis

**A1. N is not reproducible across the paper's own two harnesses.** C13 above. The
abstract's first sentence and section 5's opening both assert that iteration count is
set by cadence and tolerance for a fixed instance, and the paper prints 682.667 and
640.0 for the same size, cadence, tolerance, scheme and step sizes, from two artifacts
it cites two paragraphs apart. Because no instance manifest is archived for the fusion
sweep, the paper cannot answer the question from its own archive. Everything else on
this list is smaller than this.

**A2. The headline rule is not one rule.** C1 through C6. A reviewer who recomputes
Table 2 of the introduction against Table 5 of the experiments section finds different
exponents, different constants, different baselines and different pooled numbers, all
described as the same fit on the same data. The result is real and it survives
correction, but as printed it looks like protocol shopping.

**A3. Section 2 asserts machine independence from two machines.**
`2-related-work.tex:57`: "Iteration count is a function of the configuration, not of
the machine", and the Table 1 row "Iteration count is invariant to device, width,
execution structure and kernel". Sections 0, 1, 5 and 6 all carry an explicit
disclaimer that two architectures are not a portability claim. Section 2 carries no
disclaimer in printed text; its scope guards live in comments, which do not compile.

**A4. Three refuted claims from the original defect list are still in the draft.**
`3-methods.tex:95` still reads "1.020000 at the grid cells and lies in [1.0192, 1.0200]
across all thirteen cells". Recomputed from `structure_summary_4090.json`, the four
grid ratios are 1.020000000, 1.019999895, 1.019912374 and 1.019959412, and the minimum
over the thirteen cells is 1.019185616, which is below the stated lower endpoint. The
interval excludes its own minimum, exactly as the defect list said. C19 and C17 are the
other two survivors.

**A5. Twenty of twenty-six citations are undefined.** The paper cannot be compiled with
a reference list today, and section 2's own header comment says the underlying survey
"is prose in this repo and has no artifact under results/", so its literature verdicts
have had no verification pass.

### Tier 2: attacks that cost a result

**A6. The exponent has four public verdicts.** C7. A reviewer reading the abstract
("not identified"), the introduction ("approximately 1.8") and the conclusion
("between about 1.7 and 2.1") in one sitting will conclude the authors do not agree
with each other. The defensible version is the conclusion's, because the regret
estimator sits at 2.00 on four of five grids and the slope estimator has not reached
it.

**A7. Three of the four rows of the roofline table have no artifact.**
`5-experiments.tex` Table 7 prints 0.363, 0.282 and 0.567 with pick rates. Only the
first row (0.355, 1 of 7) is recomputable from `roofline_4090.json`, which stores a
single `pred_s_per_pixel_iter` per tile; I reproduced that row exactly (-0.943, -0.143,
+0.143, +0.943, +0.829, +0.829, +0.829, mean 0.3551, 1 pick of 7, both models choosing
32x4 at every size). The other three rows exist only as a markdown table in
`diagnostics_20260909/REPORT.md` plus uncommitted stdout from
`scripts/diagnostics/score_roofline.py`, which lives outside `results/`. The conclusion
compounds this by citing 0.567 to sources that cannot carry it.

**A8. Two superlatives are false for the artifacts they cite.**
`5-experiments.tex:251` and `:254`: the lowest penalised SGM-1 in
`px999999_4090/factorial.json` is pdhg_banked schedule_K256 at 0.00570206 s, below the
quoted 0.00576, and the lowest catalog cell in `cadence_4090_v1/factorial.json` is
pdhg_banked schedule_K256 at 0.00775176 s, giving 1.568x rather than the quoted 0.00790
s and 1.540x. The 2.110x ratio itself is clean because both halves are Condat-Vu; only
the word "best" needs the scheme attached.

**A9. Six conclusion values do not recompute.** C9 and C10 above, plus: "8 cells" for
the transfer sweeps when the four files span 10 distinct cells (grid:16, grid:24,
regular:256:4, regular:512:4, ot:32:32, ot:64:64, mcf:16:2, mcf:16:4, grid:32,
regular:256:2); "All eight sweeps ran at K=256" attached to the 72 instance claim when
14 sweeps supply those 72 instances (8 files of 6 plus 6 files of 4); "a different ratio
on three of the five rows" when it is four; "Two of the four subgroups therefore do not
invert" when only TV carries the pooled sign, and when the four subgroups are not
disjoint because netflow with n = 13 is the union of grid with n = 7 and regular with
n = 6.

**A10. "Disjoint instance sets" is false.** `6-conclusion.tex:267`.
`omega_seeds_4090.json` covers grid:16 and regular:256:4 at seeds 0 to 7;
`omega_wide_bw.json` covers those two cells among four at seeds 0 to 2. They share 2
cells times 3 seeds times 9 alphas = 54 instances, and I confirmed identical omega0 on a
shared instance (grid:16, seed 0, alpha 1: 7.389690541121416 in both).

**A11. No omega file records a device.** `6-conclusion.tex:266` says the transfer
results cover "2 devices". None of the eleven `omega_*.json` files carries a device
field in its args block; the attribution rests on file naming and on
`q3_4090.log` / `q4_4090.log`. The optimal transport and multi-commodity flow sweeps
that carry the abstract's headline transfer result are single-run and single-machine.

**A12. Two monotonicity claims are false.** `6-conclusion.tex:232`: "the slope
magnitude rises monotonically as the fraction of optima on a grid edge falls". The
slope magnitudes are monotone as listed but the edge fractions are 0.483, 0.296, 0.319,
0.175, 0.159, which invert between the second and third grids. `6-conclusion.tex:247`:
"approach each other from opposite directions as censoring falls". Ordered by falling
edge fraction the regret estimator reads 2.10, 1.90, 2.00, 2.00, 2.00; it moves once
and then stops.

**A13. A stale promise about a run that has finished.** `5-experiments.tex:709`: "A
finer ratio grid is still running and can only narrow that interval", and
`6-conclusion.tex:249`: "A finer grid is still running". `omega_fine_4090.json` exists,
holds 1520 rows over 19 ratios, and is already the fourth row of the exponent table in
the same section. `q5.log` reads "[q5] FINE DONE". A grid that has not been scored also
cannot be promised to narrow anything; the exponent has moved three times already.

### Tier 3: attacks that cost a sentence

**A14.** `0-abstract.tex`: 2.6x is the eight-arm spread at 128 pixels (2.6294) in a
sentence whose established context is 256 pixels, where the spread is 1.867x on the
4090 and 1.450x on the Blackwell; over all fourteen archived eight-arm files the range
is 1.4418 to 2.6294. 0.778x needs "at 496 pixels": the same tile at 256 pixels is
1.366x in the same nine-repeat protocol.

**A15.** `0-abstract.tex`: "Step sizes are worth 4.3x" mislabels the variable. The two
presets in `method_axis_20260909/factorial.json` are banked (tau 0.05, sigma 1.0, rho
1.9) and balanced (tau 0.25, sigma 0.25, rho 1.5); rho differs, so 4.3125 is a preset
effect. `1-introduction.tex:223` states this caveat correctly in a comment and
`1-introduction.tex:217` says "Step sizes and relaxation together", so only the
abstract needs the word.

**A16.** `1-introduction.tex:28`: "moved it by at most 1.9 percent" is true of the
six-instance means (586.667 / 576.000 = 1.85 percent) and false per instance, where the
move reaches 5.3 percent (`bw_20260909/sz128_bw`, K = 64: 1216 in float64 against 1280
in float32 on instance 0) and 4.5 percent at n = 496.

**A17.** `1-introduction.tex:102`: the traced-versus-static geometric means 2.03x
(spanning 1.94 to 2.13) and 2.34x (spanning 2.30 to 2.37) reproduce exactly from
`loop_mechanism.json` in the certificate-off cells only. With the certificate on,
n = 128 is 2.28x spanning 2.24 to 2.31. The sentence names neither setting.

**A18.** `1-introduction.tex:310`: "the alternative tie rules move each slope by at most
0.01". On the six-decade grid the geometric tie rule gives -1.5412 and the extreme tie
rules give -1.5307 and -1.5516, a move of 0.0105. The experiments comment's bound of
0.015 holds; the introduction's 0.01 does not.

**A19.** `1-introduction.tex:130`: "Header records device sm_89 only" is attached to a
citation of `kernel_admission_4090.log`, which has no device header. The sm_89 line is
in `kernel_admission_4090.header` in the same directory.

**A20.** `6-conclusion.tex:345` sources the OR-Tools default of 64 to
`tv_ffi_20260906/DECOMPOSITION_REPORT.md section 3`. That section is headed "Iteration
count depends only on the certificate cadence" and contains no OR-Tools content, and
the string `termination_check_frequency` appears nowhere under `results/`. The claim is
a literature fact and is cited correctly at `2-related-work.tex:89`; only the results
pointer is wrong.

**A21.** `2-related-work.tex:126` gives the AutoML null as "at worst 1.0305x on the
4090", from `attack_plan_20260906/analysis/REPORT.md`, while `6-conclusion.tex:285`
gives 1.020x for the same phenomenon from `method_axis_20260909`. Two campaigns, two
numbers, neither cross-referenced.

**A22.** `1-introduction.tex:231` describes the native omega0 spread inside min-cost
flow as 6.969 to 9.125 (24 groups, `omega_4090` plus `omega_bw`), while
`5-experiments.tex:668` describes it as 6.97 to 8.25, a factor of 1.18
(`omega_wide_bw`). Both reproduce; the descriptive claim is the same.

**A23.** `5-experiments.tex:60`: "The same vectors appear in four separate factorial
artifacts" is false at K = 16, where `f32row_4090` gives PDHG 1008, 512, 608, 416, 320,
640 against `cadence_4090_v1`'s 1008, 496, 608, 400, 320, 640. The next sentence states
the exception correctly, so the paragraph is honest and only the lead sentence needs the
K >= 64 scope it currently borrows.

### What is not attackable

For balance, the following were rechecked from raw rows on this pass and are exact.
Every element of Table 1 in the introduction and the fusion table in section 5, both
devices, all four sizes, best and worst tiles, baselines and speedups. The whole U shape
table at seven sizes on both devices and both static-over-traced rows. The loop
mechanism table, all twelve penalties and ratios, the 10.0, 10.9 and 6.2 percent drops
and the 24 certificate differences spanning -239.90 to +844.56 ns. The whole of
`fanova.json`, rebuilt from the four raw factorials. The regret table on the narrow grid
including 1.827 / 23.03 / 208, 1.181 / 3.87 and 1.190 / 3.18 and every derived ratio
(1.548, 5.96, 1.536, 7.25, 2.07, 1.851, 26.04). Every cell of the transfer table in
section 5 and of the omega table in section 6. All eight grouping invariance counts
(169 of 169, 83 of 83, 25 of 25, 71 of 72, 230 of 237, 437 of 601, 12 of 120, 71 of
120) and the 1944 / 1510 / 216 / 209 totals. The gate block (2106 runs, 1755 admitted,
351 rejected, 257 capped, 987.8 GPU seconds, 351 of 351 at five thetas and 0 of 351 at
1.02, the dense grid at 1573 runs, 1220 admitted, 353 rejected, 121 step pairs, 13
cells, PDHG only, all solve fields null). The netflow block (24 executed rows, 24 equal
iteration counts, 22 certified, geometric means 1.3946 and 1.3850, 12 pairs, 11
certified, the 20000 cap at gap 2.858e-4, and the four gate strings). All sharpness
correlations (+0.9426 pooled over 17, -0.5387, -0.8423, -0.1197, +0.9825, raw pooled
+0.7242, means 92.1 / 10230 and 43.3 / 624, median solve 9600). The optimal transport
closed form to between 1.081e-14 and 3.186e-14 relative error with admissible product
ratios of exactly 2.000000 and 1.500000. The admission log (106 lines, 102 rows, 17
variants, rho 1.9 throughout, bitwise no in every row, worst 4.649e-15, PASS). The
holdout audit (32 data.json files, 23 carrying `final_holdout_loaded`, all false, 9
predating the field). The nine-repeat reruns, both logs, every row.

## 4. Go or no-go

**NO-GO on the current files.** The blocking set is finite, is dominated by editing
rather than by measurement, and is achievable well before the 18 September abstract
deadline. Nothing on the list requires a new result.

### Blocking items, in the order they should be cleared

**B1. Reconcile 682.667 against 640.0, or scope both.** (C13, A1.) Either archive the
fusion sweep's instance manifest and state in section 5 that the fusion population is a
different draw of six scans from the catalog population, or re-run one fusion cell on
the catalog instances and show the counts agree. Until one of these is done, the
paper's central invariance claim is contradicted by two of its own tables. This is the
only blocking item with a measurement component, and it is small.

**B2. Run the correction pass over `2-related-work.tex` and `3-methods.tex`.** (A4,
C15 to C21.) Specifically: delete "XLA and a hand written kernel" from the triple
sentence; scope the 0 of 6 and 1 of 6 float32 result to n = 256 and name the cadences
actually measured on each device; replace "1.020000 at the grid cells and lies in
[1.0192, 1.0200] across all thirteen cells" with the four grid values and the true range
1.0192 to 1.0200 open at the bottom, or simply "between 1.0191856 and 1.0200000";
replace "the gate product is exactly 0.990000000" with "equals 0.99 to within 3.3e-16 in
all 2673 rows and exactly in 882 of them"; delete "with 4.649e-15 on sm_120" or attach a
Blackwell admission log; delete the sentence saying the output log is not archived, and
the same sentence in `INTEGRATION.md:661` and `TITLE_OPTIONS.md:55`; delete or scope
"bitwise identity at rho = 1"; add a printed scope sentence to section 2 replacing
"not of the machine".

**B3. Pick one rule and one protocol for the omega0 result, and use it in all four
sections.** (C1 to C6, A2.) The recommendation is C = 0.977 at p = 2, because it is the
argmin on the fit set and it is already what sections 5 and 6 print. Rebuild the
introduction's Table 2 from it, restate the abstract's pair as either "2.17x to 1.12x
with capped runs charged at the cap" or "1.82x to 1.12x, with the fixed baseline scored
only where it converged, on 143 of 162", and make the two tables in sections 5 and 6
name their protocols in the same words.

**B4. Pick one verdict on the exponent.** (C7 to C11, A6.) The conclusion's wording is
the defensible one. Delete "approximately 1.8" from the introduction, fix 1.066 to
1.057 or 1.062 and say which C resolution was used, harmonise the censoring counts to
102 / 33 / 46 / 15 / 14 and the slopes to one tie convention with the convention named
in the caption, and replace the narrow-grid band with the true set or drop it.

**B5. Create the 20 missing bibliography entries and verify each against its primary
source.** (A5.) Section 2's own header comment already lists them.

**B6. Persist `roofline_scores_4090.log` under `results/artifacts_20260910/`, or delete
the three unsourced rows of Table 7.** (A7.) This costs no GPU time; the script is
committed and scores an existing JSON.

**B7. Clear the Tier 2 conclusion arithmetic**: 8 cells to 10, three of five to four of
five, eight sweeps to fourteen, Two of four to One of four, delete "disjoint instance
sets", delete "monotonically", delete "and 2 devices" or say the attribution is by run
log, drop the DECOMPOSITION_REPORT pointer for OR-Tools, and add
`diagnostics_20260909/REPORT.md` as the source for 0.567 if that row survives B6.

**B8. Delete both "still running" sentences.** (A13.) The fine grid has run and is in
the table.

Tier 3 items (A14 to A23) are one-clause fixes and are not blocking, but they are cheap
and every one of them is a sentence a reviewer can quote.

## 5. Remaining experiments, with GPU hours

All estimates are grounded in the measured cost of the same kind of run already in the
archive. A full factorial campaign at one size costs 31 to 184 GPU seconds
(`campaign_s` in the seven `tv_size` and `bw_20260909` factorials). A complete omega
sweep of 650 to 1730 rows costs 256 to 669 GPU seconds of solve time (`exec_s` summed
per file); the eleven archived sweeps total 1.24 GPU hours. Wall clock runs roughly two
to three times the solve time because of build and load. Estimates below are wall clock
on one GPU.

| # | Experiment | Why it is needed | GPU hours |
|-|-|-|-|
| E1 | Re-run one fusion end-to-end cell (n = 256, K = 256, 8 arms, 3 repeats) on the catalog instance draw, and archive a `data.json` manifest for the fusion harness at all four sizes on both devices | Blocking item B1. Resolves the 682.667 against 640.0 contradiction, which is the paper's most exposed claim | 0.2 |
| E2 | Blackwell float32 (`f32_cert64`) rows at K = 16 and K = 256, n = 256, both schemes, both tolerances | Lets section 2's "at all three cadences ... on both architectures" be either supported or correctly narrowed; today only K = 64 exists on sm_120 | 0.1 |
| E3 | Custom kernel (`kernelhost`) at cadence 16, n = 256, both schemes | Lets the 584.0 / 597.3 / 682.7 triple legitimately claim kernel invariance, or confirms the deletion in B2 | 0.1 |
| E4 | Temporal fusion admission run on the Blackwell, 17 variants times 3 shapes times 2 proximal settings, plus a rho = 1 pass on both devices | Supports the "4.649e-15 on sm_120" sentence and the "bitwise identity at rho = 1" sentence, or confirms both deletions | 0.1 |
| E5 | Blackwell factorial at n = 384 | The two-device iteration-identity claim currently covers n = 128, 256 and 496 only, while the fusion sweeps cover n = 384 on both devices | 0.1 |
| E6 | Replicate `omega_ot_wide`, `omega_mcf` and `omega_holdout` on the second device, and add a `device` field to the sweep writer | The transfer result, which is the abstract's headline, is single-machine today and no omega file records a device (A11) | 1.0 |
| E7 | The factor-1.6 ratio grid on the ultrawide range: about 37 ratios times 11 alphas times 4 seeds times 2 cells | Only if the paper wants to tighten the exponent past the current bracket. Not blocking; B4 makes the paper correct without it | 1.0 |
| E8 | Adaptive-schedule leak at a second size (n = 128) and a second family (netflow), one factorial each | The conclusion's own falsifier table says this falsifier "is not yet evaluable". Turning it evaluable is the single cheapest new result in the paper | 0.3 |
| E9 | Re-run `scripts/diagnostics/score_roofline.py` and persist stdout | Blocking item B6. CPU only, scores an existing JSON | 0.0 |
| | **Total, blocking only (E1 to E5, E9)** | | **0.6** |
| | **Total, blocking plus scope repairs (add E6, E8)** | | **1.9** |
| | **Total including the optional exponent grid (add E7)** | | **2.9** |

The whole remaining measurement programme is under three GPU hours. The submission
risk is entirely in the editing, not in the compute.
