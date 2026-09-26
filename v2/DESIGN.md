# Optimization Atlas v2 — P0 core design

**What exists (this build).** A typed operator-splitting solver core in JAX
(CPU, float64), package `optimization-atlas`, import name `atlas`. The IR is
the E1 composite form `min_x f(x) + g(x) + h(Lx)` with optional blocks
(`atlas/ir/spec.py`); the canonical P0 instance is anisotropic TV denoising
`min_x (1/2)||x-y||^2 + lam*||Dx||_1` with `D` = 2D forward differences,
Neumann boundary, `||D||^2 <= 8`. Atoms carry `PropSet` tags — Lipschitz,
cocoercivity (`beta = 1/L`, Baillon–Haddad, LSCOMO §2.2.1), averagedness
(proxes are 1/2-averaged, §2.5), prox-friendliness — in `atlas/operators/`
(`base.py`, `prox.py`, `linear.py`; conjugate proxes via the Moreau identity).
Four schemes live in `atlas/schemes/base_schemes.py`, each a jitted
`lax.while_loop`: **FBS** (§2.7, `a in (0, 2/L)`, averagedness `2/(4-aL)`),
**DRS** (§2.7.1, any `a>0`, 1/2-averaged), **PDHG** (§3.3 as variable-metric
PPM, `t·s·||L||² < 1`), **Condat–Vu** (§3.3 eq. 3.13,
`t·L_f/2 + t·s·||L||² < 1`). For the TV instance, FBS and DRS run on the dual
`min_{||u||_inf<=lam} (1/2)||Dᵀu||² − <y, Dᵀu>` (primal recovered exactly as
`x = y − Dᵀu`; DRS's quadratic resolvent is a CG linear solve); PDHG and
Condat–Vu run on the primal saddle form. Wrappers (`atlas/wrappers/km.py`):
plain KM, over-relaxation `rho in (0,2)` with the averagedness combination
rule `alpha' = rho·alpha < 1`, and the Halpern/OHM anchor (§12.2,
nonexpansive maps).

**Certificates (mini 4-level stack).** Level 1 (construction,
`atlas/typing_/rules.py`): every solve first runs `typecheck_*`, which
validates the side condition and returns a `ConstructionCert` {scheme,
params, condition_str, satisfied, averagedness, LSCOMO citation} or raises
`TypeCheckError` — mandatory, enforced inside `solve()`. Level 3 (runtime,
`atlas/certify/residuals.py`): fixed-point residual plus the **rigorous TV
duality gap** — dual point projected to `||u||_inf <= lam`, dual value
`G(u) = <y, Dᵀu> − (1/2)||Dᵀu||²`, so `gap = P(x) − G(u) >= 0` always;
termination on `rel_gap = gap/(|P|+1) <= 1e-4` (tests verify `1e-6`).
Level 4 (empirical): `benchmark_card()` emits per-instance JSON with
time-to-tol, iterations, residuals, and a hardware note. (Level 2 —
rate certificates — is post-P0.)

**P0 scope cuts.** No SRG/PEP machinery (plan `certify/srg.py`, `pep.py` are
P3); no derivation-tree engine, genome schema, or mutation moves; no
`SearchBackend` / SkyDiscover integration (sibling task — `evolve/backend.py`
in the plan); no CVXPY/NL front-ends, metrics/Ruiz, registry, CLI, or CI.
Schemes are instantiated for the canonical TV instance rather than compiled
from arbitrary specs (`compile/jaxc.py` genome→step compilation is the P0→P1
bridge that generalizes this). Certificates cover levels 1/3/4 only.

**Map to plan §3.2 paths.** `atlas/ir/spec.py` → `ir/spec.py` [P0];
`atlas/operators/{base,prox,linear}.py` → `operators/*` [P0];
`atlas/typing_/rules.py` → `typing_/rules.py` + a slice of `check.py`/
`certs.py` [P1, pulled forward for the mandatory Level-1 gate];
`atlas/schemes/base_schemes.py` → `schemes/base_schemes.py` [P0] (PDHG/CV
golden forms anticipate `schemes/zoo.py` [P1]); `atlas/wrappers/km.py` →
`wrappers/km.py` [P0] + the anchor slice of `halpern.py` [P1];
`atlas/certify/residuals.py` → `certify/residuals.py` [P0] + a slice of
`card.py` [P3]. Constraint honored throughout: JAX pinned to CPU
(`JAX_PLATFORMS=cpu` before import) with x64 — the GPU belongs to the vLLM
server and contention silently hangs evaluations.

**Quickstart (runs as-is):**

```python
import os; os.environ["JAX_PLATFORMS"] = "cpu"
import jax, numpy as np, atlas; jax.config.update("jax_enable_x64", True)
problem = atlas.tv_denoise_problem(np.random.default_rng(0).random((64, 64)), lam=0.1)
result = atlas.solve_pdhg(problem, atlas.PDHGParams(t=0.5, s=0.99 / 4.0), atlas.Budget(max_iters=100_000, tol=1e-4))
print(result.cert, "\nrel_gap:", result.rel_gap_history[-1], "iters:", result.iters)
```
