# SIGNOFF: second whole-paper pass, 10 September 2026

Scope: `0-abstract.tex`, `1-introduction.tex`, `2-related-work.tex`, `3-methods.tex`,
`5-experiments.tex`, `6-conclusion.tex`, `refs-additions.bib`, plus the three
bookkeeping files (`INTEGRATION.md`, `TITLE_OPTIONS.md`, `SUBMISSION_CHECKLIST.md`)
where the section files point at them.

Method: every number reported below was recomputed on this pass by opening the named
file under `/home/miria/OptEvolve/v2/results/` and rerunning the arithmetic. I did not
trust the four clearing reports; I re-derived their claims. Scripts are in the session
scratchpad (`omega.py`, `alt.py`, `exp.py`, `t4.py`, `fx.py`). Nothing outside this
directory was written and no git operation was run.

Style contract on this file and on all six section files plus the .bib: zero U+2014,
zero U+2013, zero triple hyphen, zero italic or emphasis macros of any
spelling, pure ASCII. Verified by byte scan after writing.

## 0. Verdict in one paragraph

The four clearing passes did real work. B4, B5, B6, B7 and B8 are genuinely finished and
I could not break them. B3 is finished except for one number. But **B1 was cleared as an
artifact and never propagated into the sections that carry the defect**, and **B2 left
the refuted claim that its own verifier flagged sitting in the file B2 owns**. The
piecewise pass also did what piecewise passes do: it left 18 places where one claim
carries two numbers, six of them blocking. The paper is closer than it was and it is
still **NO-GO**. Every remaining blocker is editing. There is no measurement on the
critical path.

## 1. Per-item table

| Item | Verdict | Evidence |
|-|-|-|
| **B1** Reconcile 682.667 against 640.0 | **HALF CLEARED. The measurement half is done; the editing half is not.** | The artifact exists and reproduces. `results/artifacts_20260910/fusion_on_catalog_draw.json` runs the fusion harness at `data_seed 20260907, draw_pool 12, n 6` and returns per-instance iters `[1024, 512, 768, 512, 512, 768]`, mean 682.667, byte-identical to `tv_ffi_20260906/cadence_4090_v1/factorial.json`. `A1_RESOLUTION.md` records the disjoint draws. **But the sections were not updated.** Grep of all six .tex files for `data seed`, `seed 0` or `A1_RESOLUTION` returns hits in exactly two files: `2-related-work.tex` (lines 54, 59, 178, catalog fold) and `3-methods.tex` (lines 294, 296, 300, fusion fold). The brief's own rule, "EVERY table reporting an iteration count must now name its data seed and draw size", is satisfied in neither `0-abstract.tex`, `1-introduction.tex`, `5-experiments.tex` nor `6-conclusion.tex`. See conflicts X1 and X2. |
| **B2** Correction pass over `2-related-work.tex` and `3-methods.tex` | **NOT CLEARED.** Seven of eight sub-items done, one done on an unsourceable premise, one refuted claim survives, two new defects introduced. | Done and verified: (a) "XLA and a hand written kernel" deleted, and `kernelhost_4090` plus `catalog_4090` do hold K=64 and K=256 only, both returning 597.3333 and 682.6667; (b) the float32 0-of-6 / 1-of-6 result is scoped to n=256 with the per-device cadences named, which matches `f32row_4090` (0/6 and 1/6 at K=16, 64, 256) and `bw_20260909/sz256_bw` (K=64 only, 0/6 and 1/6); (c) the four grid ratios 1.0200000, 1.0199999, 1.0199124, 1.0199594 and the true range 1.0191856 to 1.0200000 reproduce cell by cell from `structure_summary_4090.json`; (e) the sm_120 admission figure is gone and `find` over `results/` returns only 4090 admission artifacts; (f) no "log is not archived" sentence survives anywhere outside the two review ledgers; (h) the printed scope sentence at lines 69 to 72 reproduces (`sz128_bw` 576.0 against 586.6667, `sz496_bw` 1066.6667 against 1077.3333). **Not done:** `2-related-work.tex:130-131` still prints "Scheme identity is inert to the last iteration here" and Table 1 line 203 repeats "Scheme identity is inert". Both are refuted by files I opened. See blocker BL2. **New defects:** the ratio deviation at `3-methods.tex:143` and the rho=1 inference at `3-methods.tex:285-288`. See BL8 and BL9. |
| **B3** One rule, one protocol, all four sections | **CLEARED except one number.** | The rule is `r = 0.977 / omega_0^2` at p = 2 in all four sections and nowhere else; grep for `C = ` returns only that value plus Table 4's two explicitly in-sample constants. I refit independently on `omega_wide_bw.json` (108 instances, 1404 rows), p on 0.05 and C on 0.01 dex: argmin p = 2.00, C = 0.9772, geomean regret 1.185835. Every cell of the three transfer tables reproduces exactly: fixed 2.7790 / 1.4279 / 2.8395 / 2.5136 / 2.1681, rule 1.1858 / 1.1786 / 1.0318 / 1.1899 / 1.1311, rule worst 2.1250 / 1.5625 / 1.2623 / 1.8252 / 1.8252, certified 108 / 54 / 54 / 54 / 162 against 100 / 54 / 48 / 44 / 146, omega ranges to four decimals. Table 4 reproduces: r = 0.01 gives 1.85083 and 26.04167 on 208 of 209; C = 0.263 gives 1.18084 and 3.86667 on 209; C = 0.037 gives 1.18998 and 3.17647 on 209, with the worst case flat over C in [0.0292, 0.0415], which justifies two digits. All derived gains check (1.5674, 6.7349, 1.5553, 8.1984, 2.0725, 2.34, 1.21, 2.75, 2.11, 1.92). **The one failure** is the alternative-protocol pooled figure. See BL7. |
| **B4** One verdict on the exponent | **CLEARED.** | Recomputed all five grids under the two stated conventions. Censoring 102 of 209 (91 floor, 11 ceiling), 33 of 108, 46 of 144, 15 of 80, 14 of 88. Slopes with tied argmins averaged geometrically: -0.9365, -1.5412, -1.5781, -1.6834, -1.7380; ties to smallest -0.9309 to -1.7324, ties to largest -0.9421 to -1.7437, so the caption's "at most 0.011" holds (max move 0.0105). Regret exponents 2.10, 2.00, 1.90, 2.00, 2.00 at regrets 1.1805, 1.1858, 1.1571, 1.1685, 1.0618. Floor-excluded narrow-grid slope -1.1368 over 118, matching the printed -1.14. The intro and the conclusion now agree digit for digit, which they did not at the last pass. "Approximately 1.8" survives only inside the abstract's ledger comment forbidding it. The finer grid is `omega_fine_4090.json`, 19 ratios from 1e-5 to 300 at a geometric step of 2.60, 80 instances, and is row 4 of the table. |
| **B5** Twenty missing bibliography keys | **CLEARED.** | I re-extracted every cite-like macro from all six .tex files: 26 distinct keys. `refs-additions.bib` defines exactly 20; `OptEvolve_tex_2026/refs.bib` (56 entries, mtime 2026-09-06, unwritten) supplies the other 6. Set arithmetic after a merge: zero undefined, zero orphans, zero collisions. One error in the new file: its header says the 26 keys are "all of them in `2-related-work.tex`", and `Applegate2021Practical` is also cited at `5-experiments.tex:598`. Downstream pointers are stale, not wrong: `2-related-work.tex:13-21` still instructs a future pass to create the keys, `INTEGRATION.md:62` still says they "do not exist yet", and `SUBMISSION_CHECKLIST.md` D.9 still budgets a day for them. |
| **B6** Persist `roofline_scores_4090.log` | **CLEARED.** | `results/artifacts_20260910/roofline_scores_4090.log` exists, 26 lines. Every printed figure is in it: mean Spearman 0.355 / 0.363 / 0.282 / 0.567, pick rates 1/7, 0/7, 0/7, 1/7, fitted launch cost 1.00e-05 s, and per-size M1 correlations -0.943, -0.143, +0.143, +0.943, +0.829, +0.829, +0.829. Table 7 survives intact; `5-experiments.tex:817-824` cites the log and `diagnostics_20260909/REPORT.md` lines 26 to 29. |
| **B7** Tier 2 conclusion arithmetic | **CLEARED.** | 10 distinct cells across the four transfer files, verified by enumeration (grid:16, grid:24, regular:256:4, regular:512:4, ot:32:32, ot:64:64, mcf:16:2, mcf:16:4, regular:256:2, grid:32); `6-conclusion.tex:304` says 10 and the surviving "8 cells" at `1-introduction.tex:243` and `5-experiments.tex:484` are the narrow sweep, which is correct. Fourteen sweeps at both `6-conclusion.tex:11` and `:363`; I enumerated 14 files, 8 arms each, K=256 in all, 72 instances, zero iteration mismatches, zero certification failures. "One of the four subgroups" at `:406` with the non-disjointness stated; recomputed pooled +0.943 (unlogged +0.724), netflow -0.539 (13), regular -0.842 (6), grid -0.120 (7), TV +0.983 (4), and netflow is exactly grid plus regular. "Disjoint instance sets" replaced by the true 54-of-144 overlap. "Monotonically" gone, and the non-monotonicity is real (censored fractions 0.488, 0.306, 0.319, 0.188, 0.159 invert between grids 2 and 3). "2 devices" gone; no `omega_*.json` carries a device field. The OR-Tools pointer now cites the survey record and flags the fact as external; `DECOMPOSITION_REPORT.md` indeed contains no OR-Tools string. The "three of five rows" clause no longer exists because B3 deleted the caption that carried it. |
| **B8** Delete the "still running" sentences | **CLEARED.** | Grep for "still running", "not archived", "not yet archived", "unarchived" and "COAUTHOR ACTION" over every .tex and .md in this directory returns hits only in `FINAL_STATUS.md` and `FACTCHECK_DEFECTS.md`, which are the review ledgers and should keep describing the defects. |

## 2. Claims that still appear in two places with different numbers

This is the hunt the brief asked for. The previous pass left 23. This pass closed most of
them and left 18, of which six are blocking. Numbering is fresh; the FINAL_STATUS letter
is given where one applies.

### 2A. Blocking conflicts

**X1. Mean iteration count on TV at K=256 and tolerance 1e-4. The B1 defect, unchanged in
the sections.** (FINAL_STATUS C13 and A1.) The catalog artifacts give 682.667 at n=128,
256 and 384-adjacent sizes; the fusion harness gives 554.667, 640.000, 810.667 and
1024.000 at n=128, 256, 384 and 496. Both reproduce; I recomputed each. The paper prints
both and discloses the difference in exactly one place.

| where | printed | population | population named in printed text? |
|-|-|-|-|
| `0-abstract.tex:136-137` | 682.7 at K=256 | catalog, seed 20260907, 6 of 12 | no |
| `1-introduction.tex:18-19` | 682.7 at K=256 | catalog | no |
| `1-introduction.tex:87-90` | 554.7, 640.0, 810.7, 1024.0 | fusion, seed 0, 6 drawn | no |
| `2-related-work.tex:60` | 682.7 | catalog | **yes**, lines 54 and 59 |
| `5-experiments.tex` Table 1 | 682.7 | catalog | no |
| `3-methods.tex:297-301` | (iteration identity only) | fusion | **yes**, line 300 |
| `6-conclusion.tex:17-22` | 682.667 at n=128 and n=256, 1152.000 at n=496 | catalog | no |
| `6-conclusion.tex:434-436` | 640 at n=256, 1024 at n=496 | fusion | no |

**X2. The conclusion contradicts itself inside one file.** `6-conclusion.tex:19` prints
"1066.667 and 1152.000 at n=496 ... for K=64 and K=256", and `6-conclusion.tex:435`
prints "1024 at n=496" for the same size, cadence and tolerance, 416 lines apart, with
no population named at either site. A reviewer who reads the discussion and the falsifier
table in one sitting sees the paper's own invariance claim broken by the paper.

**X3. What float32 buys once the loop is chunked.** (C21, unfixed.) `2-related-work.tex:93`
and Table 1 line 194 print **1.569x**. `5-experiments.tex:162` prints **1.576x**, and so
do `TITLE_OPTIONS.md:42` and `INTEGRATION.md` at lines 126, 385 and 726. Both reproduce
from the same four artifacts, at the same device, size, cadence and iteration count
(682.667 in all four cells): 0.00904198 / 0.00576269 = 1.5691 is the ratio of mean solver
seconds, and 13.290695 / 8.431614 = 1.5763 is the ratio of the mean per-instance
microseconds per iteration recorded in `fanova.json`. Section 5's own Table 3 prints
13.29 and 8.43, so section 2 is the outlier against five other printings and against the
table a reader will divide.

**X4. The worst temporal tile at n=496.** (C23, unfixed.) `0-abstract.tex:141` prints
**0.778x**, which is the nine-repeat value (`fusion_r9_sz496.json`, tf8x4 at 0.7784
against the traced baseline). `1-introduction.tex:133`, `2-related-work.tex:161`,
`2-related-work.tex:211` and `5-experiments.tex:303` print **0.743x**, the three-repeat
value (0.7433). `1-introduction.tex:189` states the paper's own rule: "Never mix the two
protocols." The abstract mixes them, and it mixes them inside one sentence pair, because
its 2.6x is the three-repeat 2.6294 at n=128. Neither abstract figure names its size.

**X5. Per-iteration cost of the two schemes at f64_K256 under XLA.** (C14, unfixed.)
`5-experiments.tex:104` prints "penalised SGM-1 times in ratio 0.99494 and per-iteration
costs of 19.76 against 19.76 microseconds" and sources the sentence to
`method_axis_20260909/factorial.json`. Recomputed from that file: the SGM-1 ratio is
0.994940, which is right, and the per-iteration costs are **19.798 and 19.683**, not
19.76 and 19.76. The pair 19.763 and 19.762 exists in
`tv_ffi_20260906/kernelhost_4090/factorial.json`, whose own SGM-1 ratio is 1.00093.
`6-conclusion.tex:104-105` prints 19.798 and 19.683 for the same two named cells, and
`5-experiments.tex` Table 3 prints 19.78 for the same cell from a third artifact. The
paper therefore prints the per-iteration cost of TV, n=256, Condat-Vu banked, f64, K=256,
device loop, on the 4090 as 19.76, 19.78 and 19.798, and it happens that the value chosen
to support "cost per iteration is invariant to scheme identity under XLA" is the one that
makes the invariance tightest (ratio 1.0000 rather than 1.0058).

**X6. Is scheme identity inert?** `2-related-work.tex:130-131`: "Scheme identity is inert
to the last iteration here, so a portfolio over scheme names has nothing to win", and
Table 1 line 203: "Scheme identity is inert." Against this: `0-abstract.tex:141-143`
("inert under XLA but costs 1.30x on a hand written kernel"), `5-experiments.tex:101-123`
(both the N split and the t split, stated correctly) and `6-conclusion.tex:336-343`. The
data side with the abstract and sections 5 and 6. `cadence_4090_v1/factorial.json` at
K=16, n=256, tol 1e-4: condat_vu_banked 584.0 `[1008,512,608,416,320,640]` against
pdhg_banked 578.6667 `[1008,496,608,400,320,640]`, and the split reproduces on the
Blackwell. `method_axis_20260909/factorial.json` at 1e-6 and K=64: 2860.0 against 2870.667.
On cost, `kernelhost_4090` under the custom kernel: 19.207 against 25.021 microseconds per
iteration at K=256, a factor 1.303, and 1.264, 1.179 and 1.197 in the other three
custom-kernel cells. The file's own scope guard (5) records the K=16 counterexample in a
comment, which does not compile.

**X7. Are the two factors disjoint?** `0-abstract.tex:133`, the paper's opening sentence:
"the two depend on disjoint parts of a solver recipe." `1-introduction.tex:79-80` and
`2-related-work.tex:120`: cadence is the one gene in both, so they are not disjoint.
`6-conclusion.tex:479-483`: "It is not the only gene in both. Size appears in both factors
as well, and precision changes N at some sizes." Three positions on the thesis sentence.
The archive sides with the conclusion: `fanova.json` gives cadence 47.43 percent of the
variance in log per-iteration cost, so cadence is in both factors; `bw_20260909` gives
586.667 against 576.000 at n=128 and 1077.333 against 1066.667 at n=496, so precision is
in both.

### 2B. Non-blocking conflicts, one clause each

**X8. The alternative-protocol pooled baseline.** `5-experiments.tex:699` and
`6-conclusion.tex:149` both print, in identical words, "a pooled 1.825x on 143 of the 162
held-out instances". 1.8249 is the unweighted geometric mean of the three per-family
values (1.4279 on 54, 2.1854 on 44, 1.9475 on 45). The geometric mean over those 143
instances is **1.7947**. The two sections agree with each other and both disagree with the
pooling rule their own captions define. This is listed again as blocker BL7 because it
inflates the baseline the rule is measured against.

**X9. The gate product deviation.** `1-introduction.tex:240` prints 3.3e-16;
`3-methods.tex:142` and `5-experiments.tex:488` print 3.4e-16. The recomputed maximum is
3.3306690738754696e-16 over both the 1944-row and the 2673-row sets, so 3.3e-16 is right
and 3.4e-16 is rounded the wrong way in two files.

**X10. The AutoML null.** `2-related-work.tex:146` and Table 1 line 207 give "1.0000x on
CPU at both tolerances and at worst 1.0305x on the 4090", from
`attack_plan_20260906/analysis/summary.json`; I confirm sequential over joint is exactly
1.000000 on CPU at both tolerances and 1.030456 on the 4090 at 1e-4.
`6-conclusion.tex:329` gives "a 1.020x gap" from `method_axis_20260909` (0.0077793 against
0.0076265, ratio 1.02003). `1-introduction.tex:363-367` gives an exact tie on the LP
catalog at 0.392627 s. Three quantifications of one null, on the same device, none
cross-referenced.

**X11. Fusion timing drift between protocols.** `1-introduction.tex:380-381` says "4 to
8 percent"; `5-experiments.tex:925-927` says 4.0 and 7.8 percent; `6-conclusion.tex:436`
says "the timing ratios move 4 to 5 percent". Recomputed: the best-tile speedup moves
+3.86 and +5.33 percent, the static-over-traced gain +3.99 and +7.82 percent, the baseline
+1.97 and +1.07 percent. The conclusion's interval is wrong at both ends.

**X12. The netflow gate value.** `2-related-work.tex:141` prints only "1.02267 on
netflow", which is the grid-size-12 value. `3-methods.tex:92-93`, `5-experiments.tex:411`
and `6-conclusion.tex:494` all print the full set 1.02267, 1.03040, 1.03577, 1.03781.
Section 2 should name the size or the range.

**X13. Adjacent speedups against undeclared and different denominators.**
`2-related-work.tex:160-161` gives 1.356x and 0.743x at n=496, which reproduce only
against the traced device loop; `2-related-work.tex:165` gives the U-shape 1.571x, 1.060x,
1.410x, which reproduce only against the chunked static loop. Against the chunked loop the
n=496 best is 1.2398; against the traced loop the 128 px best is 2.6294. Neither sentence
names its baseline.

**X14. The rule-worst column.** `5-experiments.tex:658-663` prints 2.12 / 1.56 / 1.26 /
1.83 / 1.83; `1-introduction.tex:282-287` and `6-conclusion.tex:185-190` print 2.125 /
1.562 / 1.262 / 1.825 / 1.825. Same table, same protocol, two precisions.

**X15. Native omega0 inside netflow.** 6.969 to 9.125 at `1-introduction.tex:231` and
`5-experiments.tex:471` (24 groups of the narrow sweep); 6.97 to 8.25 at
`5-experiments.tex:716` (12 problems of the wide sweep). Both reproduce; the descriptive
claim is the same claim.

**X16. The optimal transport norm recovery.** `1-introduction.tex:213-214` says "within
4e-14 relative"; `5-experiments.tex:442-443` says 1.1e-14 to 3.2e-14. Recomputed:
2.57e-14, 3.19e-14, 1.08e-14. The intro's bound is loose rather than wrong.

**X17. Are the 2106 gate evaluations runs or settings?** `1-introduction.tex:196-197`:
"2106 counts settings evaluated, not solves." `5-experiments.tex:423`: "over 2106 runs".
`5-experiments.tex:899`: "1755 of 2106 settings". `6-conclusion.tex:352`: "1755 of 2106
runs were admitted". Only 1755 of the 2106 were executed, so the introduction's noun is
the correct one and section 5 uses both nouns within one subsection.

**X18. The RTX 5090.** `1-introduction.tex:367` says the 5090 "appears nowhere else in
this paper" in the same sentence that prints a 5090 number (0.379070);
`6-conclusion.tex:217-218` says its rows "are not used here". Compatible but jarring.

## 3. Fresh go or no-go

**NO-GO.**

Nine blockers. All nine are editing. None requires a GPU. Ordered by what a reviewer does
with them.

**BL1. Propagate B1 into the four sections that carry the defect.** X1 and X2. The abstract,
the introduction, section 5's Table 1 caption and the conclusion all print iteration counts
with no population. The fix is four caption sentences and one clause in the introduction's
contribution (a): the catalog fold is six scans at data seed 20260907 drawn from a pool of
twelve, the fusion fold is six scans at seed 0, the overlap is 0 of 6, and
`fusion_on_catalog_draw.json` shows the two agree at 682.667 when run on the same draw.
Until this is done the paper prints 640.0 and 682.667, and 1024.0 and 1152.000, for what
its own prose calls the same configuration.

**BL2. Delete or scope "Scheme identity is inert" in `2-related-work.tex:130-131` and in
Table 1 row 203.** X6. This is refuted by two files I opened and contradicted by three
other sections including the abstract. It is a two-word fix and it is the sentence a
portfolio-methods reviewer will read first.

**BL3. Fix the abstract's opening sentence.** X7. "The two depend on disjoint parts of a
solver recipe" is refuted by `fanova.json` and by the conclusion's own paragraph. The
paper's actual finding is stronger and honest: the two factors share exactly one gene in
the recipe, cadence, plus size and, at some sizes, precision.

**BL4. Pick one convention for the float32 chunked-loop gain.** X3. 1.576x is the value
consistent with Table 3, with `fanova.json` and with three other documents; 1.569x appears
twice in section 2 and nowhere else.

**BL5. Pick one repeat protocol for the abstract's fusion figures.** X4. Either print
0.743x at n=496 with the three-repeat protocol used everywhere else, or print the whole
nine-repeat pair and say so. Both figures also need their sizes.

**BL6. Repair `5-experiments.tex:104`.** X5. The sentence takes its SGM-1 ratio from
`method_axis` and its per-iteration costs from `kernelhost`, and the numbers it prints are
contradicted by the conclusion's own table from the artifact it cites.

**BL7. Fix the 1.825 alternative-protocol figure in both sections.** X8. Print 1.795 as the
instance pool, matching the main-protocol 2.168 row, or say in the prose that this one row
is a per-family average. As written it violates the pooling rule stated three captions
away, and the error runs in the direction that flatters the rule.

**BL8. `3-methods.tex:143`.** "The recomputed ratio agrees with r to within 2.4e-16" reads
as an absolute deviation, in parallel with the product clause immediately before it, which
is absolute. Recomputed over all 2673 rows the maximum absolute deviation of tau/sigma from
r is **1.421e-14** (at r = 100, cell grid:24 in `omega_4090.json`); 2.368e-16 is the maximum
**relative** deviation. Exactly 1575 rows are exact, as printed. Say "a relative 2.4e-16",
or print 1.5e-14 absolute.

**BL9. `3-methods.tex:285-288`.** The sentence "The script prints a case only when it is not
bitwise equal to the sequential reference ... so at rho=1 the fused launch reproduced the
reference bit for bit in all 102 cases" rests on `results/kernel_fusion_20260909/validate.py`.
That script cannot have produced `kernel_admission_4090.log`, and I verified this two ways.
Its line 23 reads `for (B, T) in VARIANTS:` while `build.py:9` defines VARIANTS as 3-tuples
beginning `(8,4,64)`, so it raises ValueError on the first iteration. Its line 42 formats the
label as `f"tv_tf{B}x{T:<6d}"`, which yields `tv_tf8x4     `, while every row of the archived
log is labelled `tv_tf8x4t64`, `tv_tf16x2t128`, `tv_tf40x2t256`, which is `build.py`'s
`f"tv_tf{B}x{T}t{TH}"` convention. A filesystem-wide `find` returns exactly one `validate.py`.
So the print predicate that would license the argument from absence belongs to a script that
is not stored. Under the hard rule, delete the sentence, or archive the generating script.
Everything the log itself asserts does reproduce: 106 lines, 102 scored cases, 17 variants
matching `build.py`, all at rho 1.9, bitwise "no" in every row, worst 4.649e-15 at
`tv_tf8x8t64` and `tv_tf16x8t128` on 100x130, PASS.

Not blocking but they should go in the same pass, because each is a sentence a reviewer can
quote: X9 through X18 above, plus the four surviving Tier 3 items from the last pass that
nobody owned. Those four are: the two false superlatives at `5-experiments.tex:251-255` (the
lowest penalised SGM-1 in `px999999_4090` is pdhg_banked schedule_K256 at 0.00570206 s, not
the quoted 0.00576, and the lowest catalog cell in `cadence_4090_v1` is pdhg_banked
schedule_K256 at 0.00775176 s giving 1.568x, not the quoted 0.00790 s and 1.540x, so only
the word "best" needs the scheme attached); `1-introduction.tex:28-29`, where "moved it by at
most 1.9 percent" is true of the six-instance means and false per instance, where
`bw_20260909/sz128_bw` moves 1216 to 1280 on instance 0, a 5.3 percent move;
`1-introduction.tex:102-104`, where the 2.03x and 2.34x traced penalties are the
certificate-off cells only and the certificate-on values are 2.027x and 2.278x, with the
setting named in neither; and `0-abstract.tex:143`, where "Step sizes are worth 4.3x"
mislabels a preset that also changes rho, a caveat the introduction states correctly.

Three stale pointers in the bookkeeping files should be corrected so the next pass does not
redo finished work: `2-related-work.tex:13-21` still tells a reader the 20 keys "must be
created"; `INTEGRATION.md:62` still says they "do not exist yet"; `SUBMISSION_CHECKLIST.md`
D.9 still budgets a day for them.

## 4. What a reviewer can still attack, ranked, with honest severity

**Tier 1, reaches the thesis.**

**R1. The paper prints two iteration counts for one configuration and discloses it in a
comment.** Severity: fatal if unfixed, trivial to fix. This is BL1. The abstract's first
substantive sentence asserts that cadence and tolerance set the iteration count for a fixed
instance; the introduction and the conclusion then print 640.0 and 682.667, and 1024.0 and
1152.000, without saying the instances differ. The evidence to answer it exists and is
strong. Nothing else on this list is as cheap or as damaging.

**R2. Two GPU architectures, one problem family, one size, one tolerance.** Severity: real
and unfixable before the deadline; already disclosed. The whole invariance result lives at
n=256 with K at least 64 on two consumer-class NVIDIA parts, and the width half of it fails
one size away in both directions. Sections 0, 1, 2, 5 and 6 all now carry the scope in
printed text, which is the correct response, but a reviewer is entitled to say the headline
is a claim about one cell of a five-dimensional design. This is a limitations-section
argument, not a defect.

**R3. The transfer result is a rescaling experiment on one machine.** Severity: high; costs
the strength of the headline, not its correctness. Within each set, variation in omega0 is
variation in alpha, and `c -> alpha c` is exactly `(tau, sigma) -> (alpha tau, sigma/alpha)`.
The native spread inside netflow is 6.97 to 8.25, a factor of 1.18, against a swept range of
0.0697 to 825. `5-experiments.tex:710-721` states this plainly, which is the right move. What
remains attackable is that the part which is not a rescaling, the across-family comparison, is
three families at three native omega0 clusters, that no omega file records a device, and that
one constant does not fit them (0.977, 31.62, 1.380, 0.275, a span of 115). The rule is real;
the claim it supports is narrower than "a rule read off a pre-solve quantity".

**R4. Cost-per-iteration invariance to scheme identity is claimed twice with the artifact that
best supports it.** Severity: medium-high, because it is the kind of thing a determined
reviewer finds and then distrusts the rest. BL2 and BL6. The claim is defensible when stated
as section 5 states it in the next sentence, and indefensible in the form section 2 prints.

**Tier 2, costs a result.**

**R5. Every fusion figure is unstable in its third digit and the paper reports three digits.**
Severity: medium; disclosed. The nine-repeat rerun moves the static-over-traced gain by 7.8
percent at n=496 and the best-tile speedup by 5.3 percent, against a process-to-process
dispersion figure of 0.66 percent that is prose-only. `5-experiments.tex:928-930` says so
explicitly, which largely defuses it, but the abstract then quotes 0.778x to three digits
from the other protocol.

**R6. The gate is verified sufficient, never necessary, and 257 admitted runs still hit the
cap.** Severity: medium; disclosed in both section 5 and the conclusion. A reviewer can point
out that a feasibility test which rejects 351 of 2106 settings without ever running them is a
weaker object than the "one structure-dependent selector that does work" framing.

**R7. The fANOVA has one observation per cell, no residual term, and cadence carries more
levels than its competitors.** Severity: medium; the conclusion's falsifier table already
concedes all three. The 47.43 percent share is an exact decomposition of a saturated design,
not an estimate with an error bar.

**R8. One falsifier is declared not evaluable.** Severity: medium. The adaptive-schedule leak
is measured at one cadence, one size, one family and one tolerance, and the falsifier table
says so. That is honest and it is also an invitation.

**Tier 3, costs a sentence.**

**R9.** The rho=1 admission half cannot be recovered from `results/`, and `validate.py` as
committed does not run (BL9). A reproducibility-minded reviewer who clones and runs it gets a
ValueError.

**R10.** The best-configuration claim at `5-experiments.tex:251-255` names the wrong cells as
best in two artifacts.

**R11.** Sample sizes: six OCT images per size, four montages above 496 px, three timing
repeats, three seeds per cell in four of the six omega sweeps. Stated at
`6-conclusion.tex:359-365`.

**R12.** The TV subgroup of the sharpness analysis is a four-point correlation spanning two
image sizes, and the paper says so.

**What is not attackable.** Recomputed from raw rows on this pass and exact: the whole fusion
table on both devices at all four sizes, best and worst tiles, baselines and speedups; the
entire U-shape table at seven sizes on both devices including both static-over-traced rows and
the per-device tile choices at all seven sizes; the loop-mechanism table, all twelve penalties
and ratios, the 10.0, 10.9 and 6.2 percent falls, the 24 certificate differences spanning
-239.90 to +844.56 ns and the 845 / 697 / -240 ns triple; every share in `fanova.json`
including the four twelve-cell per-iteration vectors; the whole of Table 4 on the narrow grid
and every derived ratio; every cell of the transfer tables in sections 1, 5 and 6; the whole
exponent table under both stated conventions and the -1.14 floor-excluded slope; the gate block
(2106, 1755, 351, 257, 987.8 GPU seconds, 351 of 351 at five thetas and 0 of 351 at 1.02) and
`tab:gate` cell by cell, with `ts_max_admissible * norm_bound_gate^2` exactly 1.0 in float64 in
all thirteen cells; the netflow block (24 executed, 24 equal counts, 22 certified, geometric
means 1.3946 and 1.3850 spanning 1.2929 to 1.4918 and 1.2765 to 1.4918, 12 pairs, 11 certified,
the 20000 cap at gap 2.858e-4, all four gate strings verbatim); the tile-geometry table from
`5(B+2T)^2*8` and `S^2/B^2`; the optimal transport closed form to 1.08e-14, 2.57e-14 and
3.19e-14 relative with admissible-product ratios exactly 2.000 and 1.500; the 72-of-72 fusion
identity across all 14 files with zero mismatches and zero certification failures; the roofline
log; the admission log; the 336-of-336 RNG coupling; the joint-selection tables at 0.392627 and
0.379070; and the holdout audit.

## 5. Remaining work before 18 and 25 September

### 5.1 Editing only, no GPU

| Task | Files | Cost |
|-|-|-|
| BL1: name the population at every printed iteration count | `0-abstract`, `1-introduction`, `5-experiments` Table 1 caption, `6-conclusion` (two sites) | 2 h |
| BL2, BL3: delete the two refuted thesis-level sentences and restate the coupling claim | `2-related-work` (prose plus Table 1 row), `0-abstract` | 1 h |
| BL4, BL5, BL6, BL9: one number, one protocol, one artifact, one deletion | `2-related-work`, `0-abstract`, `5-experiments`, `3-methods` | 1 h |
| BL7, BL8: the pooled 1.825 and the ratio deviation | `5-experiments`, `6-conclusion`, `3-methods` | 1 h |
| X9 to X18, plus the four unowned Tier 3 items | all six | 2 h |
| Update the three stale pointers | `2-related-work` header, `INTEGRATION.md`, `SUBMISSION_CHECKLIST.md` | 0.5 h |
| Fix `refs-additions.bib` header ("all but one of them"), retarget the merge path to `OptEvolve_tex_2026/refs.bib`, add `pages={1--30}` to PushakHoos2022 | `refs-additions.bib` | 0.5 h |
| Merge `refs-additions.bib` into `OptEvolve_tex_2026/refs.bib` and compile once | outside this directory, coauthor action | 1 h |
| Write `4-theory.tex` links, the abstract inside `submission.tex`, the title, and the reproducibility, ethics and AI-use statements | `submission.tex`, per `INTEGRATION.md` section 1.1 | 4 h |

Roughly **13 hours of editing**, of which about **6 hours clear the nine blockers**. This
fits inside the 18 September abstract deadline with room, and the abstract itself needs
only BL1, BL3, BL5 and the size labels.

### 5.2 Needs GPU time

Nothing on this list is blocking. The paper is correct without any of it once section 4's
edits are made; every item here widens a scope the paper currently states honestly.

| # | Run | What it buys | GPU hours |
|-|-|-|-|
| G1 | Re-run the fusion harness with manifest printing at n = 128, 384 and 496 on both devices, or simply archive a `data.json` for those six cells | Makes the seed-0 population an archived fact at all four sizes rather than an inference from the harness source; n = 256 is already covered by `fusion_on_catalog_draw.json` | 0.2 |
| G2 | Temporal-fusion admission at rho = 1, on the 4090, with the generating script committed under `results/` | Restores `3-methods.tex:285-288` instead of deleting it (BL9). Deleting the sentence costs nothing, so this is optional | 0.1 |
| G3 | Blackwell admission run, 17 variants x 3 shapes x 2 proximal settings | Lets the admission claim cover both architectures instead of the 4090 only | 0.1 |
| G4 | Blackwell `f32_cert64` at K = 16 and K = 256, n = 256, both schemes, both tolerances | Widens the width-invariance claim on sm_120 from one cadence to three | 0.1 |
| G5 | Custom kernel at K = 16, n = 256, both schemes | Lets the 584.0 / 597.3 / 682.7 triple claim kernel invariance rather than the current K >= 64 scope | 0.1 |
| G6 | Blackwell factorial at n = 384 | The two-device identity currently covers 128, 256 and 496 in the catalog artifacts | 0.1 |
| G7 | Replicate `omega_ot_wide`, `omega_mcf` and `omega_holdout` on the second device, and add a `device` field to the sweep writer | The headline transfer result is single-machine today, and no omega file records a device. This is the largest scope gap the paper does not currently overclaim on | 1.0 |
| G8 | Adaptive-schedule leak at n = 128 and on netflow, one factorial each | Turns the falsifier the conclusion calls "not yet evaluable" into an evaluated one. The single cheapest new result available | 0.3 |
| G9 | A wider or finer ratio grid to close the 1.7 to 2.1 exponent bracket | Optional. B4 makes the paper correct without it, and four grids have already failed to move the regret estimator off 2.00 | 1.0 |
| | **Total, everything** | | **3.0** |

The submission risk is entirely in the editing. Three GPU hours would buy scope, and zero
GPU hours are needed to be correct.

### 5.3 Recommended order

1. BL1 through BL9, in that order. Six hours. Re-run a mechanical consistency grep afterwards.
2. G8 and G7 in the background on the two boxes while the editing proceeds. They are the only
   runs that change what the paper can claim rather than how carefully it says it.
3. X9 through X18 and the four Tier 3 leftovers.
4. Merge the bibliography and compile. Nothing in this directory has ever been compiled;
   there is no `main.tex` for this draft and no TeX toolchain on this machine, so the
   bibliography has been validated structurally and never by bibtex. Budget for the first
   compile finding cross-reference breakage.
5. Only then decide on G2 through G6, all of which are alternatives to a deletion.

## 6. What must never be claimed, rechecked

All four prohibitions hold in the current files.

* **No LLM advantage.** `1-introduction.tex:358-362` and `6-conclusion.tex:317-321` both
  state the RNG coupling; 336 of 336 fallback-sourced candidates byte-identical, which I
  confirmed sums as 74 + 99 + 48 + 68 + 47 in `gate1_numbers.json`.
* **No joint-search advantage over sequential.** Stated as a null in three places. The one
  hazard is X10, where the null's size is quantified differently in sections 2 and 6.
* **No general accelerator portability.** Printed disclaimers in sections 0, 1, 5 and 6, and
  now a printed scope sentence in section 2 as well (lines 69 to 72), which was the gap the
  last pass found.
* **The frozen final holdout has never been evaluated.** Declared in
  `1-introduction.tex:384`, `5-experiments.tex:935-938` and `6-conclusion.tex:232-239`, and
  the audit reproduces: `final_holdout_loaded` is false in every `data.json` under
  `results/` that carries the field.

## 7. Style contract on this file

Zero U+2014, zero U+2013, zero triple hyphen, zero italic or emphasis macros
of any spelling, pure ASCII.
