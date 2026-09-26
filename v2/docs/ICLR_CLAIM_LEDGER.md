**OptEvolve manuscript claim ledger, 6 September 2026**

The user authorized the submission attack plan and requested execution. The
initial manuscript was preserved in
`results/attack_plan_20260906/manuscript_before.tar.gz` before edits. The source
hash and authorization scope are in `start.json` beside that archive.

This pass aligns the local draft with current evidence. It is not a completed
mathematical audit or the final submission evaluation. No publication, GitHub
push, or coauthor message was performed.

| Claim or issue | Source | Action / permitted interpretation |
|---|---|---|
| LLM reaches champion in generation 4 versus random in generation 6 | `results/sprint/phase2/PHASE2_REPORT.md`, RNG finding V1 | Removed the positive efficiency claim and the old discovery-card figure. Historical coupled arms cannot establish comparative efficiency. |
| First correctness-closed LLM evolution / verifier-feedback mechanism | Existing manuscript, updated related-work survey | Removed unqualified priority language. Added primary-source citations for ReEvo and AutoOPT; positioning must be justified by a precise contract and experiments. |
| Four certificate levels, SRG/PEP, five transformations, public leaderboard | Existing introduction/methods versus implemented source | Replaced present-tense contribution claims with implemented base/relaxation grammar, execution policies and versioned evaluation. |
| v2 mutates simplex/IPM families using an OpenEvolve/CLARABEL oracle | Old related-work section | Rewritten around typed splittings, the v2 population loop and original-coordinate numerical checks. |
| 285/285 over 2,002 evaluations | `results/sprint/phase2/PHASE2_REPORT.md`, strict-integrity audit | Retained as successful numerical checks over evaluation rows; not 2,002 independent instances or a proof of absence of all numerical errors. |
| 1,340/1,340 OOD rejections prove transfer failure | Same report, corrected OOD counts | Retained only as a sufficient-condition gate stress test. Rejection does not prove divergence or non-transfer. |
| TV 2.91–5.20x; LP approximately 2.25–2.31x | `results/hardware_codex/REPORT.md` and raw campaign artifacts | Added with small sample sizes and explicit same-algorithm legacy denominator. Not superiority over a fully tuned baseline. |
| No reduced-precision policy loses a certificate | `results/hardware_codex/REPORT.md`, pure-f32 failures at 1e-6 | Corrected: lower precision can fail to attain tolerance; every returned success still requires the numerical check. |
| Generated kernels provide speedup | Same report, joint LP catalogs and Nsight evidence | Replaced with measured negative: generated variants lose to XLA; fewer launches do not ensure lower solve time. |
| Joint beats sequential | Same report, joint LP selection | Explicit tie. Full-catalog selection is not a budget-matched evolutionary experiment. |
| Device identifier implies hardware specialization | Same report, cross-device transfer table | Both GPUs choose the same recipes; CPU/GPU differences are preliminary and reverse transfer is not uniformly penalized. |
| Repair feedback is generally harmful or the first resampling ablation | `results/sprint/phase3/repair_ablation.json` | Retained 52/42/36 percent within two retries as a paired Qwen2.5-Coder-1.5B result; removed universal/priority implications. |
| Runtime reports the theoretical H-norm step residual | `atlas/schemes/base_schemes.py`, actual TV/LP metrics | Corrected theory and appendix prose: the theorem's residual and numerical stopping statistics are different. |
| Unrestarted Halpern theorem automatically covers fixed-period resets | Main theory and generated instantiation prose | Distinguished the maps and removed automatic projection/rate transfer to restarted trajectories. The theorem itself remains scoped to its assumptions. |
| Nonfinite switch prefix always causes a failed solve | `atlas/wrappers/switch_k.py` | Corrected appendix prose to describe recovery from the last finite state. |
| Tests prove numerical compiler correctness | Hardware methods and generated instantiation appendix | Reworded as numerical evidence, with exact-arithmetic and correct-atom assumptions separate. Updated generator prose to preserve this distinction on regeneration. |
| Cadence as a new mechanism | [PDLP solver source](https://github.com/google/or-tools/blob/stable/ortools/pdlp/solvers.proto) | Added conventional configurable cadence to related work and the strong execution baseline. |
| Patient IDs unavailable in cached OCT data | Development-v1 protocol versus encoded dataset image paths | Correction sidecars record the filename-derived group audit: six training and six validation groups, zero overlap. New runner revision rejects unknown/overlapping groups before profiling/timing. Old protocol and measurements are preserved. |
| 36-recipe TV factorial: 1.47x on RTX 4090 and 1.57x on RTX 5090 | `results/attack_plan_20260906/analysis/REPORT.md`, three complete raw catalogs | Six training and six validation images at 256x256, tolerance 1e-4. Denominator is compiled banked Condat-Vu, f64, K=64. Both GPUs select refinement at K=256; CPU retains its baseline. Workstation diagnostic, not a production-solver or search-efficiency comparison. |
| New TV factorial proves joint search is better | Same catalogs and training-only selection reconstruction | Joint ties implementation-first sequential selection on both GPUs at 1e-4. The 5090 tight-tolerance score difference is about 0.75 percent with one non-solve in each selection; it does not establish a robust advantage. |

**Primary citations added or corrected**

ReEvo metadata and mechanism were checked against
[arXiv:2402.01145](https://arxiv.org/abs/2402.01145).
AutoOPT's staged design and Lean verification were checked against
[arXiv:2608.07407](https://arxiv.org/abs/2608.07407).
The adjacent Lyapunov/proximal-discovery work was checked against
[arXiv:2606.26077](https://arxiv.org/abs/2606.26077).
The Hawkeye entry now points to its
[OpenReview record](https://openreview.net/forum?id=e3pxJbBRBk), replacing a
nonstandard alphaXiv identifier. The full inherited theory bibliography still
needs an entry-by-entry audit; this pass does not certify every historical entry.

**Remaining obligations**

The theorem's supplied norm bound must be valid; an empirical power-iteration
estimate alone is not a proved upper bound. Transformation/metric transport,
inexact atoms, floating-point error, and restarted wrapper guarantees require
their own scoped arguments. The current appendix reference checks do not close
those proof obligations.

The empirical contribution still requires strong matched-budget controls,
independent problem units, cost accounting and a frozen final evaluation.
The manuscript labels the present numbers as diagnostics. Its evidence-required
paragraph and disclosure-inventory notes identify unfinished work; they are not
submission-ready placeholders to leave in the final PDF.
