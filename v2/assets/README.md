# assets/ — plotting material and generated figures

Convention: ALL plotting code and ALL generated figures live here.

**Current figure status, September 6, 2026:** the old `discovery_card.png`
contains the coupled-RNG LLM/random comparison and has been removed from the
active manuscript. It must not be reused as sample-efficiency evidence.
The current development figures are generated with the versioned report by
`scripts/summarize_solver_interaction.py`; retain their source-run provenance
when exporting a paper figure here. The historical headline plan below is
superseded by the [submission attack plan](../docs/ICLR_SUBMISSION_ATTACK_PLAN.md).

- `*.py` — plotting scripts (run from V2 root with /home/miria/jaxenv/bin/python)
- `*.png` / `*.pdf` — generated figures; suffix indicates the run
  (`_toy64` = 64x64 2gen x 2pop smoke run; unsuffixed/`_full` = real-scale runs)
- Numeric inputs come from `../results/*.json` (cards, evolution history) —
  results/ holds data, assets/ holds visuals.

Planned headline figure: `discovery_card.png` (4 panels: evolution trajectory
LLM-vs-random-vs-default; held-out duality-gap-vs-walltime traces;
per-instance held-out comparison; OCT patch before/after TV denoising —
caption carries the champion genome WITH its construction certificate).
