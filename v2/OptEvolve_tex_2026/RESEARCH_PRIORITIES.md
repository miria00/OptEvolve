# OptEvolve: research priorities for ICLR 2027

**Recommendation:** pilot sparse logistic learning first; expand to multitask learning only if the mechanism shows headroom. These are research bets, not established SOTA opportunities.

## 1. Two benchmark additions

### 1. Elastic-net logistic regression

- **Workload:** sparse classifiers and regularization paths, not just isolated cold solves. Target correlated features and statistically useful sparsity.
- **Datasets:** [Tabula Sapiens](https://registry.opendata.aws/tabula-sapiens/) processed expression data for cell-type classification (v2: over 1.1M cells, 24 donors); [RCV1](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#rcv1.binary) for sparse controls; [epsilon](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#epsilon) for dense GPU controls. Pin versions and preprocessing; separate donors for predictive evaluation and instance families for solver-search evaluation. Preserve RCV1's documented train/test split.
- **Comparables:** [cuML LogisticRegression](https://docs.nvidia.com/cuml/latest/api/generated/cuml.linear_model.LogisticRegression/) already supports GPU elastic-net logistic via quasi-Newton/OWL-QN; [skglm](https://jmlr.org/papers/v26/24-0008.html) supplies modern working-set/proximal-Newton machinery; [glmnet](https://glmnet.stanford.edu/articles/glmnet.html) is the essential warm-started path baseline.
- **Risk:** GPU versus CPU alone is not algorithmic discovery. Screening and curvature are already established competitors' strengths.

### 2. Multitask sparse logistic regression for molecular assays

- **Workload:** fixed molecular fingerprints or frozen embeddings, with independent elastic-net heads or a coupled group/sparse-group penalty for shared feature selection. Only the latter is a distinct structured optimization problem.
- **Datasets:** [OGBG-MolPCBA](https://ogb.stanford.edu/docs/graphprop/) (437,929 molecules, 128 assays); smaller OGB MoleculeNet datasets such as Tox21 for transfer checks. Preserve scaffold splits and mask missing labels.
- **Comparables:** batched cuML and per-task glmnet/skglm for independent heads; tuned GPU proximal gradient/FISTA and a compatible block-coordinate/proximal-Newton solver for the **same coupled objective**. Verify library support before choosing the coupled baseline.
- **Risk:** shared sparsity needs statistical justification; batching may explain all gains. Faster fitting is not a claim of beating graph models on predictive accuracy.

## 2. Methodology additions, in exploration order

| Addition | Cheapest informative test | Ramifications |
|---|---|---|
| **Fixed diagonal preconditioning, then bounded adaptation** | Compare scalar steps, fixed diagonal scaling, and diagonal updates during a finite initial phase followed by freezing. Count setup/refresh cost. | Helps LPs/quadratics too. Positive definiteness alone is insufficient: certify final step bounds and valid state transfer. Indefinite adaptation needs additional theory; dense metrics can destroy cheap proximal operations. |
| **Certificate-driven structure reduction** | Use safe screening to eliminate provably zero coordinates, then compare reduced first-order and proximal-Newton solves against skglm. | Extend the grammar with reduction witnesses, reduced-problem bounds, and a lift back to the original certificate. Stable observed support is not proof; surviving coordinates may remain nonsmooth. |
| **Switching only where it has demonstrated leverage** | Branch from actual saved states at several switch times, including conversion costs. | If the best tested switch barely helps, stop tuning triggers. If it helps substantially, search the controller. Safeguarded residual quasi-Newton is a later option, not another arbitrary switch rule. |

Foundations: [Gap Safe screening](https://jmlr.org/papers/v18/16-577.html), [variable-metric forward–backward splitting](https://doi.org/10.1080/02331934.2012.733883), [SuperMann](https://arxiv.org/abs/1609.06955).

## 3. Success, framing, and decisive evidence

**Success:** a repeatable, practically meaningful time-to-certificate improvement over strong CPU **and** GPU comparables on held-out instance families, explained by a structural mechanism. A roughly 2× gain is a useful internal target, not a venue requirement. If performance only ties, quantify reduced tuning/search effort or better robustness rather than claiming superior solvers.

**Framing:** “OptEvolve discovers compact, transferable solver compositions whose certificates enable both safe execution and computational simplification.” Distinguish ingredient rediscovery from a useful new composition, and convergence guarantees from finite-precision implementation correctness.

**Essential experiments:**

- Matched objectives, preprocessing, intercepts, accuracy, and hardware accounting; cold solves versus paths; device-resident versus transfer-inclusive timings.
- Ablate scaling, adaptation, reduction, local solving, and batching separately; include hand-composed controls.
- Compare LLM search with random search and an appropriate numerical tuner at matched evaluation budgets; report search cost and amortization.
- Freeze final benchmarks; repeat seeds, show dispersion/failures, and test transfer across datasets, regularization regimes, and sizes. Report predictive utility separately from optimization accuracy.

### Closest work and useful publication models

- **[Optimization Algorithm Design via Electric Circuits, NeurIPS 2024](https://web.stanford.edu/~boyd/papers/optimization_via_circuits.html):** close automated, convergence-preserving algorithm-design precedent. Differentiate through typed composition, empirical hardware-conditioned search, and certificate-driven reduction, if demonstrated.
- **[A Generalization Result for Convergence in Learning-to-Optimize, ICML 2025](https://proceedings.mlr.press/v267/sucker25a.html):** distinguish deterministic guarantees for admitted genomes from probabilistic generalization guarantees. “Learning with convergence guarantees” alone is not novel.
- **[Greedy Learning to Optimize with Convergence Guarantees](https://arxiv.org/abs/2406.00260):** close learned-preconditioning work; preprint status verified, not main-conference acceptance. Differentiate structural solver synthesis from learning update parameters.
- **[The Polar Express, ICLR 2026](https://iclr.cc/virtual/2026/poster/10006553):** model for connecting a numerical mechanism, hardware-aware design, theory, and an important downstream workload.
- **[Deconstructing What Makes a Good Optimizer, ICLR 2025](https://iclr.cc/virtual/2025/poster/27658):** model for turning near-ties into useful mechanism analysis. A universal winner is not necessary; explanatory evidence is.
