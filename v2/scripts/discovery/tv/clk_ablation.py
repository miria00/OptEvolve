# LD_LIBRARY_PATH TRAP, the robust form.  Clearing this variable from inside
# Python is too late: the dynamic linker has already resolved against the
# system CUDA and JAX reports "Backend 'cuda' is not in the list of known
# backends", silently leaving only CPU.  The only reliable fix is to re-exec
# the interpreter with a clean environment before anything is imported, which
# is what this does.  Every entry point that touches the GPU needs it, because
# a nohup'd run inherits whatever the launching shell had.
import os as _os, sys as _sys
# atlas/__init__.py pins JAX_PLATFORMS=cpu on import, by design, so that
# importing the genome library never grabs a GPU.  An entry point that DOES
# want the GPU must therefore set it BEFORE importing atlas, which that file
# says in as many words.  Getting this wrong produces "no platforms that are
# instances of gpu", which is the identical symptom to the LD_LIBRARY_PATH
# trap and cost four wrong diagnoses; LD_LIBRARY_PATH is cleared here too,
# since both must hold and both are inherited through nohup.
_os.environ["JAX_PLATFORMS"] = "cuda,cpu"
if _os.environ.get("LD_LIBRARY_PATH") and _os.path.isfile(_sys.argv[0] or ""):
    _env = {k: v for k, v in _os.environ.items() if k != "LD_LIBRARY_PATH"}
    _os.execve(_sys.executable, [_sys.executable] + _sys.argv, _env)

"""Matched-budget ablation of the search axes on the TV cell.

Every configuration spends the SAME number of evaluations, C * L * K, so
this compares how a budget is SPLIT, not how big it is.  The single-
trajectory loop with global elitism is included as the incumbent.
"""
import json, math, sys, os
sys.path.insert(0, "/home/miria/OptEvolve/v2")
sys.path.insert(0, "/home/miria/OptEvolve/v2/scripts/discovery/tv")
from atlas.evolve.loop import evolve, evolve_clk
from atlas.evolve.random_backend import RandomBackend
from tv_fitness import TVCell

NPY = "/home/miria/data/oct/oct_128_noisy.npy"
BUDGET, L = 48, 6
SEEDS = (11, 23, 37)
rows = []

for seed in SEEDS:
    cell = TVCell(NPY, budget_s=1.5, max_iter=30000)
    h = evolve(RandomBackend(seed=seed), cell, generations=L,
               pop_size=BUDGET // L, seed=seed)
    rows.append(dict(config="incumbent C=1 global-elitism", C=1, K=BUDGET // L,
                     seed=seed, best=h["best_fitness"], evals=h["n_unique_evaluated"]))
    for C in (1, 2, 4, 8):
        K = BUDGET // (L * C)
        if K < 1:
            continue
        cell = TVCell(NPY, budget_s=1.5, max_iter=30000)
        h = evolve_clk(lambda c, s=seed: RandomBackend(seed=s + 1009 * c),
                       cell, C=C, L=L, K=K, seed=seed)
        rows.append(dict(config=f"clk C={C} K={K}", C=C, K=K, seed=seed,
                         best=h["best_fitness"], evals=h["n_unique_evaluated"]))
    print("seed %d done" % seed, flush=True)

agg = {}
for r in rows:
    agg.setdefault(r["config"], []).append(r["best"])
print("\n%-32s %10s %10s %10s" % ("configuration", "mean t(s)", "best t(s)", "fails"))
for k, v in agg.items():
    fin = [-x for x in v if math.isfinite(x)]
    print("%-32s %10s %10s %10d"
          % (k, ("%.4f" % (sum(fin) / len(fin))) if fin else "-",
             ("%.4f" % min(fin)) if fin else "-", len(v) - len(fin)))
json.dump(rows, open("/home/miria/OptEvolve/v2/results/rl_20260924/clk_ablation.json", "w"), indent=1)
