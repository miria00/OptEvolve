"""KB-driven composer: new first-order recipes built from knowledge-base ingredients.

This automates the route by which restarted Halpern PDHG was found: take a base
operator whose certificate states a property (averaged with constant alpha, in its
own metric), apply classical operator ideas from the knowledge base whose
preconditions that property meets (Krasnoselskii-Mann relaxation, Halpern anchoring,
fixed and residual-triggered restarts, reflection under the anchor, safeguarded
Anderson acceleration), let the typed gates decide admissibility, run every admitted
recipe to the certified certificate of its problem, and report novelty against the
knowledge base's practice layers.

Generic over base operators: forward-backward, FISTA (momentum; no wrappers), PDHG,
Condat-Vu, Douglas-Rachford and Davis-Yin, on any problem their builders accept. The
operator state (an array or a tuple such as (x, u)) is flattened to one vector, so every
wrapper applies unchanged. Cost is counted in evaluations of the base map T; Anderson
evaluates the fallback step only when its safeguard rejects (lax.cond), so the count
equals the work done.
"""
from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

from atlas.typing_ import TypeCheckError

BASES = ("fbs", "fista", "pdhg", "condat_vu", "drs", "dys")


@dataclass(frozen=True)
class Recipe:
    base: str = "fbs"
    step_frac: float = 1.0            # fbs/fista/dys: a = step_frac / L; drs: a = step_frac;
                                      # pdhg/condat_vu: fill theta of the gate boundary
    ratio: Optional[float] = None     # pdhg/condat_vu primal weight omega = sqrt(s/t); None = 1/(t* ||L||) default
    rho: Optional[float] = None       # relaxation relative to its maximum: rho = rho_frac / alpha
    anchor: bool = False              # Halpern anchoring
    restart: Optional[tuple] = None   # ("fixed", k) | ("adaptive", beta), anchor only
    anderson: int = 0                 # Anderson memory (0 = off), no anchor

    def name(self) -> str:
        if self.base in ("fbs", "fista", "dys"):
            head = f"{self.base}(a={self.step_frac:g}/L)"
        elif self.base == "drs":
            head = f"drs(a={self.step_frac:g})"
        else:
            head = f"{self.base}(fill={self.step_frac:g}" + (f",omega={self.ratio:g}" if self.ratio else "") + ")"
        parts = [head]
        if self.rho is not None:
            parts.append(f"relax({self.rho:g}/alpha)")
        if self.anchor:
            parts.append("halpern")
        if self.restart:
            parts.append(f"restart[{self.restart[0]}={self.restart[1]:g}]")
        if self.anderson:
            parts.append(f"anderson(m={self.anderson})")
        return " + ".join(parts)

    def ingredients(self) -> list[str]:
        ids = ["fista"] if self.base == "fista" else []
        if self.rho is not None:
            ids.append("krasnoselskii_mann")
        if self.anchor:
            ids.append("halpern")
        if self.restart:
            ids.append("restart" if self.restart[0] == "fixed" else "adaptive_restart")
        if self.anderson:
            ids.append("anderson")
        return ids


def base_params(problem, r: Recipe):
    """The base scheme's parameters, on the boundary of its gate where one exists."""
    from atlas.backend.adaptive import boundary_steps
    from atlas.schemes.base_schemes import (CondatVuParams, DRSParams, DYSParams, FBSParams,
                                            FISTAParams, PDHGParams)
    if r.base in ("fbs", "fista", "dys"):
        L = float(problem.f.props.lipschitz)
        a = r.step_frac / L
        return {"fbs": FBSParams, "fista": FISTAParams, "dys": DYSParams}[r.base](step=a)
    if r.base == "drs":
        return DRSParams(step=r.step_frac)
    nb = float(problem.L.norm_bound)
    Lf = float(getattr(getattr(problem.f, "props", None), "lipschitz", 0.0) or 0.0) if r.base == "condat_vu" else 0.0
    # default primal weight: the head-to-head's point for Condat-Vu (t = 1/L_f, s on the boundary,
    # omega = 0.7 L_f / ||L||); 1 for PDHG unless the recipe carries a calibrated ratio
    omega = r.ratio if r.ratio else (0.7 * Lf / nb if Lf > 0 else 1.0)
    t, s = boundary_steps(omega, nb, Lf, r.step_frac)
    return (PDHGParams if r.base == "pdhg" else CondatVuParams)(t=t, s=s)


def typecheck_recipe(r: Recipe, problem):
    """Build the base operator (its builder runs the base gate) and apply the knowledge
    base's gate rules for each ingredient. Returns (built, cert, rho_abs)."""
    from atlas.schemes.base_schemes import build_scheme
    from atlas.typing_.rules import (typecheck_adaptive_restart, typecheck_anderson, typecheck_halpern,
                                     typecheck_relaxation, typecheck_restart)
    if r.base not in BASES:
        raise TypeCheckError(f"unknown base {r.base!r}")
    if r.base == "fista" and (r.rho is not None or r.anchor or r.restart or r.anderson):
        raise TypeCheckError("FISTA momentum is not an averaged map: no relaxation, anchor, restart or Anderson")
    if r.base in ("pdhg", "condat_vu") and not problem.has_linear_map:
        raise TypeCheckError(f"{r.base} needs an h(Lx) block; this problem has no linear map")
    built = build_scheme(problem, r.base, base_params(problem, r), check=True)
    cert = built.cert
    rho_abs = None
    if r.rho is not None:
        rho_abs = r.rho / cert.averagedness
        cert = typecheck_relaxation(cert, rho_abs, with_anchor=r.anchor)
    if r.anchor:
        cert = typecheck_halpern(cert)
    if r.restart:
        cert = (typecheck_restart(cert, int(r.restart[1])) if r.restart[0] == "fixed"
                else typecheck_adaptive_restart(cert, float(r.restart[1])))
    elif not r.anchor and r.restart:
        raise TypeCheckError("restart needs the Halpern anchor")
    if r.anderson:
        if r.anchor:
            raise TypeCheckError("Anderson and the Halpern anchor are both outer extrapolations; not composed here")
        cert = typecheck_anderson(cert, int(r.anderson))
    return built, cert, rho_abs


def enumerate_recipes(bases=("fbs", "fista"), step_fracs=None):
    """The composition space for the given bases: every ingredient combination, admissible or not."""
    out = []
    for base in bases:
        if base == "fista":
            out.append(Recipe(base="fista"))
            continue
        fracs = step_fracs or {"fbs": (1.0, 1.9), "dys": (1.0, 1.9), "drs": (1.0,),
                               "pdhg": (0.98,), "condat_vu": (0.98,)}[base]
        for sf, rho, anchor, restart, aa in itertools.product(
                fracs, (None, 0.99, 1.0, 1.2), (False, True),
                (None, ("fixed", 64), ("fixed", 512), ("adaptive", 0.2), ("adaptive", 0.5)), (0, 5)):
            if restart and not anchor:
                continue
            out.append(Recipe(base, sf, None, rho, anchor, restart, aa))
    return out


def run_recipe(problem, r: Recipe, *, tol: float, max_evals: int, K: int = 64, aa_D: float = 1e6, aa_eps: float = 1e-6,
               barrier: bool = False):
    """Run a composed recipe to the certificate of `problem`'s gap kind. `barrier` (execution gene, no
    effect on the iterates) places an optimization barrier after each base-map application, so XLA cannot
    fuse the operator into the wrapper's arithmetic."""
    built, cert, rho_abs = typecheck_recipe(r, problem)
    T, metric = built.T, built.metric
    dyn = jax.tree_util.tree_map(jnp.asarray, built.dyn)
    v0 = jax.tree_util.tree_map(jnp.asarray, built.state0)
    rho = 1.0 if rho_abs is None else float(rho_abs)
    # The state stays a pytree. An earlier version flattened it to one vector (ravel_pytree) so every
    # wrapper was plain vector algebra; the unravel -> T -> ravel round trip inside the loop made PDHG on
    # a 1024 x 1024 transport plan 35x slower per step (1988 vs 56.5 us, identical iterates;
    # scripts/diagnostics/ot_flatten_probe.py). The problem data `d` is an argument of the chunk, never a
    # closure, so large arrays are not baked into the executable as constants.
    tm, leaves = jax.tree_util.tree_map, jax.tree_util.tree_leaves

    def tvdot(a, b):
        return sum(jnp.vdot(x, y) for x, y in zip(leaves(a), leaves(b)))

    def tnorm(a):
        return jnp.sqrt(tvdot(a, a))

    def twhere(c, a, b):
        return tm(lambda x, y: jnp.where(c, x, y), a, b)

    def F(v, d):                                  # relaxed base map on the pytree state
        Tv = T(v, 0, d)
        if barrier:
            Tv = jax.lax.optimization_barrier(Tv)
        return Tv if rho == 1.0 else tm(lambda x, t: x + rho * (t - x), v, Tv)

    if r.base == "fista" or (not r.anchor and not r.anderson):
        state = {"v": v0, "evals": jnp.asarray(0)}

        def body(s, d):
            return {"v": F(s["v"], d), "evals": s["evals"] + 1}
    elif r.anderson:
        m = int(r.anderson)
        g0 = tm(jnp.subtract, F(v0, dyn), v0)
        hist = lambda: tm(lambda x: jnp.zeros((m,) + x.shape, x.dtype), v0)
        state = {"v": v0, "g": g0, "DX": hist(), "DG": hist(), "nh": jnp.asarray(0), "naa": jnp.asarray(0.0),
                 "g0n": tnorm(g0), "evals": jnp.asarray(1)}

        def body(s, d):
            v, g, DX, DG = s["v"], s["g"], s["DX"], s["DG"]
            rows = lambda H: H.reshape(m, -1)
            # only the filled history rows take part: unfilled rows get a huge diagonal, so their
            # coefficients vanish (without this the early system is singular and the step is NaN)
            valid = jnp.arange(m) < s["nh"]
            gram = sum(rows(H) @ rows(H).T for H in leaves(DG))
            rhs = sum(rows(H) @ x.reshape(-1) for H, x in zip(leaves(DG), leaves(g)))
            ridge = 1e-10 * (jnp.trace(gram) + 1e-300)
            gam = jnp.linalg.solve(gram + jnp.diag(jnp.where(valid, ridge, 1e300)), rhs)
            gam = jnp.where(valid, gam, 0.0)
            # at the fixed point to machine precision the history vanishes, the ridge (relative to the
            # trace) underflows and the solve can return inf or NaN; gam = 0 is then exactly the plain
            # step, at its usual single evaluation, instead of a spurious safeguard rejection
            gam = jnp.where(jnp.all(jnp.isfinite(gam)), gam, 0.0)
            v_aa = tm(lambda x, gg, hx, hg: x + gg - jnp.tensordot(gam, hx + hg, axes=1), v, g, DX, DG)
            g_aa = tm(jnp.subtract, F(v_aa, d), v_aa)
            ok = tnorm(g_aa) <= aa_D * s["g0n"] * (s["naa"] + 1.0) ** (-(1.0 + aa_eps))

            def plain(_):
                vp = tm(jnp.add, v, g)
                return vp, tm(jnp.subtract, F(vp, d), vp), jnp.asarray(1)
            v_new, g_new, extra = jax.lax.cond(ok, lambda _: (v_aa, g_aa, jnp.asarray(0)), plain, None)
            idx = s["nh"] % m
            return {"v": v_new, "g": g_new,
                    "DX": tm(lambda H, a_, b_: H.at[idx].set(a_ - b_), DX, v_new, v),
                    "DG": tm(lambda H, a_, b_: H.at[idx].set(a_ - b_), DG, g_new, g),
                    "nh": s["nh"] + 1, "naa": s["naa"] + ok.astype(jnp.float64), "g0n": s["g0n"],
                    "evals": s["evals"] + 1 + extra}
    else:
        kind, par = (r.restart or (None, None))
        state = {"v": v0, "anchor": v0, "j": jnp.asarray(0.0), "ref": jnp.asarray(jnp.inf),
                 "nrestart": jnp.asarray(0.0), "evals": jnp.asarray(0)}

        def body(s, d):
            v, anc, j = s["v"], s["anchor"], s["j"]
            Fv = F(v, d)
            lam = 1.0 / (j + 2.0)
            vn = tm(lambda a_, f_: lam * a_ + (1.0 - lam) * f_, anc, Fv)
            jn = j + 1.0
            ref = s["ref"]
            if kind == "fixed":
                do = jn >= par
            elif kind == "adaptive":
                res = tnorm(tm(jnp.subtract, Fv, v))
                ref = jnp.where(jnp.isinf(ref), res, ref)
                do = res <= par * ref
                ref = jnp.where(do, res, ref)
            else:
                do = jnp.asarray(False)
            return {"v": vn, "anchor": twhere(do, vn, anc), "j": jnp.where(do, 0.0, jn), "ref": ref,
                    "nrestart": s["nrestart"] + do.astype(jnp.float64), "evals": s["evals"] + 1}

    def chunk(s, d):
        s = jax.lax.fori_loop(0, K, lambda _, st: body(st, d), s)
        return s, metric(s["v"], d)

    c0 = time.perf_counter()
    exe = jax.jit(chunk).lower(state, dyn).compile()
    compile_s = time.perf_counter() - c0
    evals, gap = 0, float("inf")
    hist = []                                     # (evals, gap) per chunk: the decay the diagnosis reads
    w0 = time.perf_counter()
    while evals < max_evals:
        state, gap = exe(state, dyn)
        gap = float(gap)
        evals = int(state["evals"])
        hist.append((evals, gap))
        if gap <= tol:
            break
    jax.block_until_ready(state)
    wall = time.perf_counter() - w0
    x, u = built.extract(state["v"], dyn)
    return {"recipe": r.name(), "base": r.base, "ingredients": r.ingredients(), "evals": evals, "wall_s": wall,
            "compile_s": compile_s, "gap": gap, "converged": gap <= tol, "x": np.asarray(x),
            "u": None if u is None else np.asarray(u), "cert": cert.condition_str,
            "aa_accepted": float(state["naa"]) if r.anderson else None,
            "restarts": int(state["nrestart"]) if "nrestart" in state else 0,
            "averagedness": float(cert.averagedness), "gap_history": hist}


def novelty(r: Recipe, layer: str = "glm") -> dict:
    """Ingredient-level status from the knowledge base for the target practice layer. A
    combination is only a candidate: its novelty needs a literature check before any claim."""
    import atlas.kb as kb
    st = {i: kb.by_id(i)["status"].get(layer, {}).get("mined", "unchecked") for i in r.ingredients()}
    unused = [i for i, v in st.items() if v is False]
    unchecked = [i for i, v in st.items() if v == "unchecked"]
    return {"ingredient_status": st,
            "verdict": ("uses only ingredients already in practice" if not unused and not unchecked
                        else "CANDIDATE: " + ", ".join(unused + unchecked) + " not recorded in " + layer
                        + " practice; literature check required")}
