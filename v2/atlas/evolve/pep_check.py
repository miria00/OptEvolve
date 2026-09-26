"""Class-level soundness and ranking of composed recipes by performance estimation.

For a recipe with fixed coefficients (relaxation, Halpern anchor) on an alpha-averaged base,
the worst-case fixed-point residual over the whole operator class is computed with PEPit, in
its isolated environment, through a subprocess. This is the automatic check for a transfer:
before anything runs on an instance, it certifies that the composition contracts in the worst
case (and at what rate) or shows that it does not. Restarts and Anderson acceleration have
iterate-dependent coefficients and are not fixed-coefficient PEPs; they are reported as not
expressible and left to the certified runs.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

PEPENV = os.path.expanduser("~/pepenv/bin/python")
WORKER = str(Path(__file__).resolve().parents[2] / "scripts" / "discovery" / "pep_worker.py")


def available() -> bool:
    return os.path.exists(PEPENV)


def worst_cases(specs: list[dict]) -> list:
    pr = subprocess.run([PEPENV, WORKER], input=json.dumps(specs), capture_output=True, text=True, timeout=600)
    if pr.returncode:
        raise RuntimeError(pr.stderr[-500:])
    return json.loads(pr.stdout.strip().splitlines()[-1])


def recipe_spec(recipe, alpha_base: float, N: int):
    """The PEP of a recipe, or None when its coefficients are iterate dependent."""
    if recipe.base == "fista" or recipe.restart or recipe.anderson:
        return None
    rho_abs = recipe.rho / alpha_base if recipe.rho is not None else 1.0
    return {"alpha": float(alpha_base), "rho": float(rho_abs), "anchor": bool(recipe.anchor), "N": int(N)}


def pep_rank(recipes, alpha_base: float, Ns=(10, 20)) -> list[dict]:
    """Worst-case residual at each horizon and a verdict per recipe."""
    specs, index = [], []
    for i, r in enumerate(recipes):
        for N in Ns:
            sp = recipe_spec(r, alpha_base, N)
            if sp is not None:
                specs.append(sp); index.append((i, N))
    vals = worst_cases(specs) if specs else []
    table = {}
    for (i, N), v in zip(index, vals):
        table.setdefault(i, {})[N] = v
    out = []
    for i, r in enumerate(recipes):
        taus = table.get(i)
        if taus is None:
            out.append({"recipe": r.name(), "verdict": "not expressible as a fixed-coefficient PEP", "tau": None})
            continue
        t0, t1 = taus[Ns[0]], taus[Ns[-1]]
        if t0 is None or t1 is None:
            verdict = "PEP failed (expansive or unbounded)"
        elif t1 < 0.9 * t0:
            verdict = f"worst-case convergent: tau {t0:.3g} -> {t1:.3g} from N={Ns[0]} to N={Ns[-1]}"
        else:
            verdict = f"NO worst-case convergence: tau {t0:.3g} -> {t1:.3g}"
        out.append({"recipe": r.name(), "tau": taus, "verdict": verdict})
    return out
