# OT and QP benchmark shortlist

## Optimal transport

| Workload | Brief description | Recommendation |
|---|---|---|
| **MOSTA / [HiRef](https://github.com/raphael-group/HiRef)** | Mouse-embryo expression point clouds; stages range from thousands to >100k cells. Public data link and preprocessing script supplied. | **First priority:** consecutive-stage pairs, subsamples → full pairs. Freeze features, costs and masses; label balanced unregularized OT as the chosen model, not a reproduction of developmental dynamics. |
| **CIFAR-10 feature OT / [cuRegOT recipe](https://arxiv.org/abs/2605.08793)** | Class-to-class ResNet-18 features; 5k × 5k supports. | **Quick pilot:** multiple class pairs with fixed checkpoint/preprocessing. Reuse data, but explicitly replace cuRegOT’s entropic objective with unregularized OT. |
| **[Stanford Dragon](https://graphics.stanford.edu/data/3Dscanrep/)** | Public 3-D geometry; reconstructed mesh has 566,098 vertices. | **Structural stress test:** weighted geometric OT with size ladders; prioritize only with scalable coupling storage. Research-use restrictions apply. |
| **[Waddington-OT](https://github.com/broadinstitute/wot)** | Single-cell reprogramming time course; original model includes growth/death, relaxed marginals and entropy. | **Defer:** faithful formulation expands scope; a balanced LP is only a surrogate. |
| **[DOTmark scale control](https://arxiv.org/abs/1610.03368)** | Existing resolutions extend through 512 × 512 pixels. | Retain and increase resolution. At 512² supports per side, a dense FP64 coupling alone needs **512 GiB**; costs/workspace are extra. |

## Convex quadratic programming

| Workload | Brief description | Recommendation |
|---|---|---|
| **[QAPLIB convex relaxations](https://arxiv.org/html/2507.02470v1#S4.SS5)** | HPR-QP’s 36-instance suite: 2,500–65,536 variables with structured Hessians; construction documented in [code](https://github.com/PolyU-IOR/HPR-QP). | **First priority:** reproduce the convex relaxations, not original discrete/nonconvex QAPs. Preserve operator structure; compare against HPR-QP. |
| **[OSQP application generators](https://github.com/osqp/osqp_benchmarks)** | Constrained control, portfolio, SVM, and SuiteSparse-derived regression workloads. | **Second priority:** scale control horizons and regression matrices; include cold solves and warm-start sequences. Generated workloads, not a fixed downloaded corpus. |
| **[Synthetic QAP relaxations](https://arxiv.org/html/2507.02470v1#S4.SS5)** | HPR-QP specifies operator-based instances through ~67M variables. | Optional extreme-scale test; label synthetic and separate representation benefits from algorithmic gains. |
| **[QPLIB](https://qplib.zib.de/)** | Downloadable mixed quadratic-program collection, CC-BY 4.0. | Supplementary coverage only: filter continuous, convex, linearly constrained instances. Retain Maros–Mészáros as a reference. |

## Solver baselines

- **Unregularized CUDA OT: [PDOT](https://arxiv.org/abs/2407.19689).** Restarted PDHG implemented with CUDA.jl; solves balanced OT without entropy, to numerical tolerance. **Include if obtainable:** the paper’s [code URL](https://github.com/jinwen-yang/PDOT.jl) returned 404 when checked; request code from authors. A relevant research baseline, not a verified industry standard.
- **Other OT:** retain POT network simplex; use cuRegOT, OTT-JAX, GeomLoss and HiRef only in explicitly regularized/approximate comparisons.
- **QP:** add [HPR-QP](https://github.com/PolyU-IOR/HPR-QP), [PDQP](https://github.com/jinwen-yang/PDQP.jl), [OSQP CUDA](https://osqp.org/docs/backends/index.html), and [CuClarabel](https://github.com/oxfordcontrol/Clarabel.jl/tree/CuClarabel). HPR-QP recommends its [CUDA/C implementation](https://github.com/PolyU-IOR/HPR-QP-C) for benchmarking; check structured-workload support.
