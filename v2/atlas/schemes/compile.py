"""Genome -> executable phenotype: trusted compilation of typed genomes.

`solve_genome(problem, genome, budget)` runs any Stage-0-shaped genome -
a base scheme, an "avg" blend (compiled iteration lambda*T_a +
(1-lambda)*T_b on the SAME problem, certified by the convex-combination
averagedness rule), a "switch_k" finite prefix (K aggressive iterations,
then the fully certified tail from wherever the prefix left off, with a
non-finite handoff guard), the DEPRECATED residual-window "switch"
safeguard (aggressive step accepted only on certified gap decrease below
the best of the last `window` certified gaps, else the certified
fallback step; kept runnable, absent from the proposal menus), or any of
these with the METRIC gene (certified problem rescaling, atlas.schemes.
rescale; base LP genomes only in this build) - through the cached
dynamic-parameter kernels.

Every path re-runs the MANDATORY Level-1 typechecks against the actual
problem's operator constants (not the schema's canonical constants), so a
genome that slipped past a mismatched gate still cannot run uncertified;
a metric genome is typechecked against the RECOMPUTED norm bound of the
rescaled operator. The exceptions are the switch genome's aggressive
branch (exempt, gated at runtime by the Level-3 certificate) and the
switch_k genome's aggressive prefix (exempt, harmless because the
certified tail converges from any finite start; the runtime guard
supplies the finiteness).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from atlas.genome import Genome, SubScheme
from atlas.schemes.base_schemes import (
    BuiltScheme,
    CondatVuParams,
    DRSParams,
    DYSParams,
    FBSParams,
    PDHGParams,
    SolveResult,
    _wrap_cert,
    build_scheme,
    run_built,
)
from atlas.certify.residuals import lp_rel_gap, qp_rel_kkt
from atlas.schemes.rescale import rescale_lp_problem, rescale_qp_problem
from atlas.typing_ import (
    TypeCheckError,
    typecheck_avg,
    typecheck_metric_rescale,
    typecheck_switch,
    typecheck_switch_k,
)
from atlas.wrappers.km import Budget, km_run
from atlas.wrappers.switch import switch_run
from atlas.wrappers.switch_k import switch_k_prefix


@dataclass(frozen=True)
class _WrapOnly:
    """Wrapper genes for run_built when the scheme params live elsewhere."""

    rho: Optional[float] = None
    halpern: bool = False
    restart_k: Optional[int] = None
    outer: Optional[str] = None


def _params_for(scheme: str, tau: float, sigma, *, rho=None, halpern=False,
                restart_k=None, outer=None):
    if scheme == "fbs":
        return FBSParams(step=tau, rho=rho, halpern=halpern, restart_k=restart_k, outer=outer)
    if scheme == "drs":
        return DRSParams(step=tau, rho=rho, halpern=halpern, restart_k=restart_k, outer=outer)
    if scheme == "pdhg":
        return PDHGParams(t=tau, s=sigma, rho=rho, halpern=halpern, restart_k=restart_k, outer=outer)
    if scheme == "condat_vu":
        return CondatVuParams(
            t=tau, s=sigma, rho=rho, halpern=halpern, restart_k=restart_k,
            outer=outer,
        )
    if scheme == "dys":
        return DYSParams(step=tau, rho=rho, halpern=halpern, restart_k=restart_k, outer=outer)
    raise TypeCheckError(f"unknown scheme {scheme!r}")


def genome_solver_params(g: Genome):
    """Map a BASE genome onto (scheme, params dataclass). Mix genomes have
    no single params dataclass; use solve_genome directly."""
    if g.scheme == "mix":
        raise ValueError("mix genomes have no single (scheme, params) phenotype")
    rk = g.wrapper.restart_k if g.wrapper.restart == "fixed_k" else None
    return g.scheme, _params_for(
        g.scheme,
        g.params.tau,
        g.params.sigma,
        rho=g.params.rho,
        halpern=g.wrapper.halpern,
        restart_k=rk,
        outer=g.wrapper.kind,
    )


def genome_cap_scheme(g: Genome) -> str:
    """Which base scheme's iteration cap a genome should be budgeted under
    (drs iterations carry an inner CG solve and are ~10x the cost)."""
    if g.scheme != "mix":
        return g.scheme
    if g.mix.kind == "avg":
        subs = (g.mix.a, g.mix.b)
    elif g.mix.kind == "switch_k":
        subs = (g.mix.aggressive, g.mix.tail)
    else:
        subs = (g.mix.aggressive, g.mix.fallback)
    names = [s.scheme for s in subs if s is not None]
    return "drs" if "drs" in names else names[0]


def _build_sub(problem, s: SubScheme, check: bool) -> BuiltScheme:
    return build_scheme(
        problem, s.scheme, _params_for(s.scheme, s.tau, s.sigma), check=check
    )


def _check_same_state_space(a: BuiltScheme, b: BuiltScheme, what: str):
    ta = jax.tree_util.tree_structure(a.state0)
    tb = jax.tree_util.tree_structure(b.state0)
    if ta != tb:
        raise TypeCheckError(
            f"{what} needs a shared state space; got {a.scheme} "
            f"({ta}) vs {b.scheme} ({tb})"
        )


def _solve_avg(problem, g: Genome, budget: Budget) -> SolveResult:
    m = g.mix
    if _fam(m.a.scheme) != _fam(m.b.scheme):
        raise TypeCheckError(
            "avg blend parents must share a state space (same scheme or "
            "the pdhg/condat_vu pair)"
        )
    if _fam(m.a.scheme) == "saddle" and (
        m.a.tau != m.b.tau or m.a.sigma != m.b.sigma
    ):
        raise TypeCheckError(
            "avg blend of primal-dual schemes requires identical (tau, sigma) "
            "(common averagedness metric)"
        )
    if _fam(m.a.scheme) in ("drs", "dys") and m.a.tau != m.b.tau:
        raise TypeCheckError(
            "avg blend of drs/dys parents requires identical tau: the "
            "drs/dys fixed-point encoding depends on the resolvent step, "
            "so different-tau parents share no common fixed point"
        )
    built_a = _build_sub(problem, m.a, check=True)
    built_b = _build_sub(problem, m.b, check=True)
    _check_same_state_space(built_a, built_b, "avg blend")
    cert = typecheck_avg(built_a.cert, built_b.cert, m.lam)
    rk = g.wrapper.restart_k if g.wrapper.restart == "fixed_k" else None
    cert = _wrap_cert(cert, g.params.rho, g.wrapper.halpern, rk)

    def T(state, k, dyn):
        Ta = built_a.T(state, k, dyn["a"])
        Tb = built_b.T(state, k, dyn["b"])
        lam_m = dyn["lam_mix"]
        return jax.tree_util.tree_map(
            lambda p, q: lam_m * p + (1.0 - lam_m) * q, Ta, Tb
        )

    def metric(state, dyn):
        return built_a.metric(state, dyn["a"])

    def extract(state, dyn):
        return built_a.extract(state, dyn["a"])

    blend = BuiltScheme(
        scheme=f"avg({built_a.scheme},{built_b.scheme})",
        T=T,
        metric=metric,
        extract=extract,
        state0=built_a.state0,
        dyn={"a": built_a.dyn, "b": built_b.dyn, "lam_mix": m.lam},
        static_key=("avg", built_a.static_key, built_b.static_key),
        cert=cert,
        gap_kind=built_a.gap_kind,
    )
    wrap = _WrapOnly(rho=g.params.rho, halpern=g.wrapper.halpern, restart_k=rk)
    return run_built(blend, cert, wrap, budget, backend=g.backend)


def _solve_switch(problem, g: Genome, budget: Budget) -> SolveResult:
    m = g.mix
    if _fam(m.aggressive.scheme) != _fam(m.fallback.scheme):
        raise TypeCheckError(
            "switch branches must share a state space (same scheme or the "
            "pdhg/condat_vu pair)"
        )
    if g.backend is not None and g.backend.precision != "f64":
        raise TypeCheckError(
            "a switch genome supports only the f64 backend in this build"
        )
    built_fb = _build_sub(problem, m.fallback, check=True)  # MUST certify
    built_agg = _build_sub(problem, m.aggressive, check=False)  # exempt
    _check_same_state_space(built_agg, built_fb, "switch")
    if built_fb.gap_kind not in ("dual_gap", "lp_gap"):
        raise TypeCheckError(
            "switch requires a problem with a rigorous runtime duality-gap "
            "certificate (the monitor's accept test IS the certificate); "
            f"problem {problem.name!r} only offers a fixed-point residual"
        )
    cert = typecheck_switch(built_fb.cert, m.window)

    def T_agg(state, k, dyn):
        return built_agg.T(state, k, dyn["agg"])

    def T_fb(state, k, dyn):
        return built_fb.T(state, k, dyn["fb"])

    def metric(state, dyn):
        return built_fb.metric(state, dyn["fb"])

    dyn = {"agg": built_agg.dyn, "fb": built_fb.dyn, "tol": float(budget.tol)}
    state, iters, res_h, met_h, wall, n_acc, n_rej = switch_run(
        T_agg,
        T_fb,
        built_fb.state0,
        budget,
        metric,
        dyn,
        window=int(m.window),
        static_key=("switch", built_agg.static_key, built_fb.static_key),
    )
    x, u = built_fb.extract(state, dyn["fb"])
    return SolveResult(
        x=np.asarray(x),
        u=None if u is None else np.asarray(u),
        iters=iters,
        fp_residual_history=res_h,
        rel_gap_history=met_h,
        wall_time=wall,
        cert=cert,
        extra={
            "n_aggressive_accepted": n_acc,
            "n_fallback_steps": n_rej,
            "window": int(m.window),
        },
    )


def _solve_switch_k(problem, g: Genome, budget: Budget) -> SolveResult:
    """Finite-prefix switching: K aggressive iterations, then the FULLY
    certified tail from wherever the prefix left off.

    The tail's Level-1 typecheck is MANDATORY (its certified KM iteration
    is globally convergent from any start, which is why the finite prefix
    is harmless); the aggressive branch is built check=False (exempt by
    design, structurally validated by the Stage-0 gate). The prefix
    runtime (atlas.wrappers.switch_k) guards the handoff: a non-finite
    prefix state restarts the tail from the last finite state (the
    initial point if none), recorded in result.extra["switch_k"].

    Budget note: the tail runs under the full budget.max_iters cap (one
    cached kernel for every K instead of one per distinct K); reported
    iters = prefix + tail, so a budget-exhausted switch_k solve can
    exceed max_iters by at most K <= 5000 iterations.
    """
    m = g.mix
    if _fam(m.aggressive.scheme) != _fam(m.tail.scheme):
        raise TypeCheckError(
            "switch_k branches must share a state space (same scheme or "
            "the pdhg/condat_vu pair)"
        )
    if g.backend is not None and g.backend.precision != "f64":
        raise TypeCheckError(
            "a switch_k genome supports only the f64 backend in this build"
        )
    built_tail = _build_sub(problem, m.tail, check=True)  # MUST certify
    built_agg = _build_sub(problem, m.aggressive, check=False)  # exempt
    _check_same_state_space(built_agg, built_tail, "switch_k")
    agg = m.aggressive
    agg_desc = f"{agg.scheme}(tau={agg.tau:.6g}" + (
        f", sigma={agg.sigma:.6g})" if agg.sigma is not None else ")"
    )
    cert = typecheck_switch_k(built_tail.cert, int(m.K), agg_desc)

    handoff, prefix_iters, reset, reset_iter, handoff_iter, wall_prefix = (
        switch_k_prefix(
            built_agg.T,
            built_tail.state0,
            int(m.K),
            built_agg.dyn,
            static_key=("switch_k", built_agg.static_key),
        )
    )
    dyn = dict(built_tail.dyn)
    dyn["tol"] = float(budget.tol)
    dyn["rho"] = 1.0
    state, iters, res_h, gap_h, wall = km_run(
        built_tail.T,
        handoff,
        budget,
        built_tail.metric,
        dyn,
        mode="km",
        static_key=built_tail.static_key,
    )
    x, u = built_tail.extract(state, dyn)
    return SolveResult(
        x=np.asarray(x),
        u=None if u is None else np.asarray(u),
        iters=prefix_iters + iters,
        fp_residual_history=res_h,
        rel_gap_history=gap_h,
        wall_time=wall_prefix + wall,
        cert=cert,
        extra={
            "switch_k": {
                "K": int(m.K),
                "prefix_iters": prefix_iters,
                "tail_iters": iters,
                "nonfinite_reset": reset,
                "reset_iter": reset_iter if reset else None,
                "handoff_from_iter": handoff_iter if reset else prefix_iters,
            }
        },
    )


def _prepare_with_metric(problem, g: Genome, prescaled=None):
    """Metric gene: solve the equivalently RESCALED problem, terminate and
    certify on the ORIGINAL problem's f64 duality gap.

    See atlas.schemes.rescale for the derivation of the transforms
    (x = D2 x', u = D1 u'). The scheme is built (and Level-1 typechecked)
    on the rescaled problem, whose operator norm bound is recomputed by
    power iteration x1.02, so the existing side conditions are checked
    against the actual rescaled ||L'||. The BuiltScheme's metric and
    extract are then replaced with original-variable versions: the
    runtime certificate is lp_rel_gap of (D2 x', D1 u') against the
    ORIGINAL (c, A, b), and result.x/u are original-problem variables.
    """
    mt = g.metric
    if g.scheme == "mix":
        raise TypeCheckError(
            "the metric gene applies to BASE genomes only in this build"
        )
    if g.scheme not in ("pdhg", "condat_vu"):
        raise TypeCheckError(
            "the metric gene applies to the LP saddle schemes "
            f"(pdhg/condat_vu) in this build; got {g.scheme!r}"
        )
    # qp_standard is closed under the same diagonal rescaling (see
    # atlas.schemes.rescale, CONVEX QP); its certificate is qp_rel_kkt.
    is_qp = problem.name == "qp_standard"
    rescale = rescale_qp_problem if is_qp else rescale_lp_problem
    if prescaled is None:
        rescaled, d1, d2, nb = rescale(
            problem, mt.kind, iters=mt.iters, alpha=mt.alpha
        )
    else:
        # A caller that already paid for this exact rescale can hand it back instead of paying
        # again. rescale_qp_problem is deterministic in (problem, kind, iters, alpha) and its power
        # iteration is seeded, so reuse is EXACTLY equivalent, not an approximation. Without this,
        # a QP solve rescales twice (once in the caller to read Lf/nb for the steps, once here),
        # which was 4.6 s of the 5.268 s lasso n_var 50,000 result.
        if prescaled.get("kind") != mt.kind or int(prescaled.get("iters", -1)) != int(mt.iters or -1) \
                or prescaled.get("alpha") != mt.alpha:
            raise TypeCheckError(
                f"prescaled rescale was computed for kind={prescaled.get('kind')!r} "
                f"iters={prescaled.get('iters')!r} alpha={prescaled.get('alpha')!r}, but the metric "
                f"gene asks for kind={mt.kind!r} iters={mt.iters!r} alpha={mt.alpha!r}")
        rescaled, d1, d2, nb = prescaled["value"]
    scheme, params = genome_solver_params(g)
    built = build_scheme(rescaled, scheme, params, check=True)
    if mt.kind == "ruiz":
        param_desc = f"iters={int(mt.iters)}"
    elif mt.kind == "ruiz_pc":
        param_desc = f"ruiz_iters={int(mt.iters)},alpha={mt.alpha:g}"
    else:
        param_desc = f"alpha={mt.alpha:g}"
    # Record HOW the bound was obtained: for pock_chambolle the default path
    # uses the UNIVERSAL Lemma-2 constant, not an estimate, and the
    # certificate line must say so (a reader must be able to tell a theorem
    # from a power-iteration estimate).
    from atlas.schemes.rescale import PC_SAFETY

    provenance = (
        f"Pock-Chambolle 2011 Lemma 2 certified bound 1 "
        f"x{PC_SAFETY:.12g}, "
        "universal bound after constructing the instance-dependent "
        "diagonals, no power iteration"
        if mt.kind in ("pock_chambolle", "ruiz_pc")
        else "power iteration x1.02"
    )
    if is_qp:
        provenance += "; L_f = ||D2 P D2|| recomputed by power iteration x1.02"
    cert = typecheck_metric_rescale(
        built.cert, mt.kind, param_desc, nb, bound_provenance=provenance
    )
    cert = _wrap_cert(cert, params.rho, params.halpern, params.restart_k,
                      outer=getattr(params, "outer", None))

    L_orig = problem.L
    dyn = dict(built.dyn)
    dyn.update(
        {
            "metric_d1": jnp.asarray(np.asarray(d1)),
            "metric_d2": jnp.asarray(np.asarray(d2)),
        }
    )
    if is_qp:
        P_orig = problem.f.P
        dyn.update({"orig_q": problem.f.q, "orig_lo": problem.h.lo,
                    "orig_hi": problem.h.hi})
    else:
        dyn.update({"orig_c": jnp.asarray(problem.data["c"], dtype=jnp.float64),
                    "orig_b": jnp.asarray(problem.data["b"], dtype=jnp.float64)})

    def metric(state, dyn_):
        xs, us = state
        x = dyn_["metric_d2"] * xs
        u = dyn_["metric_d1"] * us
        if is_qp:
            return qp_rel_kkt(x, u, P_orig, dyn_["orig_q"], L_orig,
                              dyn_["orig_lo"], dyn_["orig_hi"])
        return lp_rel_gap(x, u, dyn_["orig_c"], L_orig, dyn_["orig_b"])

    def extract(state, dyn_):
        xs, us = state
        return dyn_["metric_d2"] * xs, dyn_["metric_d1"] * us

    wrapped = BuiltScheme(
        scheme=built.scheme + f"+metric[{mt.kind}]",
        T=built.T,
        metric=metric,
        extract=extract,
        state0=built.state0,
        dyn=dyn,
        static_key=("metric_rescale", built.static_key, L_orig),
        cert=cert,
        gap_kind=built.gap_kind,
        gram=built.gram,
    )
    if getattr(g.backend, "kernel", "jax") != "jax":
        from atlas.backend.native import install_kernel
        wrapped = install_kernel(problem, g, wrapped)
    metadata = {
        "kind": mt.kind,
        "iters": None if mt.iters is None else int(mt.iters),
        "alpha": None if mt.alpha is None else float(mt.alpha),
        "opnorm_rescaled": float(nb),
        "opnorm_original": float(problem.L.norm_bound),
    }
    return wrapped, cert, params, metadata


def _solve_with_metric(problem, g: Genome, budget: Budget) -> SolveResult:
    wrapped, cert, params, metadata = _prepare_with_metric(problem, g)
    res = run_built(wrapped, cert, params, budget, backend=g.backend)
    res.extra = {**(res.extra or {}), "metric": metadata}
    return res


def _fam(scheme: str) -> str:
    return "saddle" if scheme in ("pdhg", "condat_vu") else scheme


def solve_genome(problem, g: Genome, budget: Budget) -> SolveResult:
    """Run a genome on a CompositeProblem: the single entry point for
    evolution fitness evaluation. Level-1 typechecks run against the
    problem's actual operator constants; TypeCheckError = certified
    construction failure."""
    from atlas.backend.native import check_compatible
    check_compatible(problem, g)
    if g.metric is not None:
        return _solve_with_metric(problem, g, budget)
    if g.scheme == "mix":
        if g.mix is None:
            raise TypeCheckError("scheme 'mix' requires a mix block")
        if g.mix.kind == "avg":
            return _solve_avg(problem, g, budget)
        if g.mix.kind == "switch_k":
            return _solve_switch_k(problem, g, budget)
        return _solve_switch(problem, g, budget)
    scheme, params = genome_solver_params(g)
    built = build_scheme(problem, scheme, params, check=True)
    cert = _wrap_cert(built.cert, params.rho, params.halpern, params.restart_k,
                      outer=getattr(params, "outer", None))
    if getattr(g.backend, "kernel", "jax") != "jax":
        from atlas.backend.native import install_kernel
        built = install_kernel(problem, g, built)
    return run_built(built, cert, params, budget, backend=g.backend)
