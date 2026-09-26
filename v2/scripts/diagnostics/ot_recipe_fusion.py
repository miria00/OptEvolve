"""Why do composed PDHG recipes on transport differ ~13x in time per evaluation (Blackwell creation
loop, 2026-09-11) when their arithmetic per step is nearly the same? Run every admitted recipe for a
fixed number of evaluations (tol 0, so no early exit), with and without an optimization barrier after
the base map, and count what XLA compiled into the chunk (fusions, reductions). Iterates are identical
with and without the barrier; only execution changes."""
import argparse, json, os, re, sys, time
from dataclasses import replace
import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
sys.path.insert(0, ".")
from atlas.bench import dotmark
from atlas.evolve import kb_composer as kc
from atlas.typing_.rules import TypeCheckError

ap = argparse.ArgumentParser()
ap.add_argument("--instance", default="CauchyDensity:32:1001-1002"); ap.add_argument("--kappa", type=float, default=30.0)
ap.add_argument("--evals", type=int, default=3200); ap.add_argument("--out", required=True); ap.add_argument("--hlo-dir", default=None)
a = ap.parse_args()
assert jax.devices()[0].platform == "gpu", f"not on a GPU: {jax.devices()}"
cls, res, pr = a.instance.split(":"); i1, i2 = map(int, pr.split("-"))
prob = dotmark.composite(cls, int(res), i1, i2, operator="reduction")
ratio = float(np.linalg.norm(np.asarray(prob.data["c"])) / np.linalg.norm(np.asarray(prob.data["b"]))) / a.kappa
if a.hlo_dir:
    os.makedirs(a.hlo_dir, exist_ok=True)
captured = {}
_jit = jax.jit


class _Spy:                                   # records the optimized HLO of what run_recipe compiles
    def __init__(self, j): self.j = j
    def __call__(self, *x, **k): return self.j(*x, **k)
    def lower(self, *x, **k):
        low = self.j.lower(*x, **k)
        class _L:
            def compile(_s):
                c = low.compile(); captured["hlo"] = c.as_text(); return c
        return _L()


def hlo_stats(txt):
    return {"fusions": len(re.findall(r"= [^=\n]*\bfusion\(", txt)), "reduces": len(re.findall(r"\breduce\(", txt)),
            "while": len(re.findall(r"\bwhile\(", txt)), "bytes": len(txt)}


rows = []
print(f"device {jax.devices()[0]}; ot {a.instance}; omega {ratio:.4g}; {a.evals} evaluations per run", flush=True)
for r in kc.enumerate_recipes(bases=("pdhg",)):
    r = replace(r, ratio=ratio)
    try:
        kc.typecheck_recipe(r, prob)
    except TypeCheckError:
        continue
    for barrier in (False, True):
        jax.jit = lambda f, *x, **k: _Spy(_jit(f, *x, **k))
        try:
            out = kc.run_recipe(prob, r, tol=0.0, max_evals=a.evals, barrier=barrier)
        finally:
            jax.jit = _jit
        st = hlo_stats(captured.get("hlo", ""))
        row = {"recipe": r.name(), "barrier": barrier, "evals": int(out["evals"]), "wall_s": float(out["wall_s"]),
               "us_per_eval": 1e6 * float(out["wall_s"]) / max(int(out["evals"]), 1), "gap": float(out["gap"]), **st}
        rows.append(row)
        if a.hlo_dir:
            open(os.path.join(a.hlo_dir, f"{len(rows):02d}_{'bar' if barrier else 'nobar'}.hlo.txt"), "w").write(captured.get("hlo", ""))
        print(f"  {r.name():72s} barrier={barrier!s:5s} {row['us_per_eval']:8.1f} us/eval  gap {row['gap']:.1e}  "
              f"fusions {st['fusions']:3d} reduces {st['reduces']:3d}", flush=True)
        jax.clear_caches()
json.dump({"instance": a.instance, "omega": ratio, "rows": rows}, open(a.out, "w"), indent=1)
print("OT_FUSION_DONE")
