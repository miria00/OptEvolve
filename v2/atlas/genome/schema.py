"""Typed genome v1 and the Stage-0 gate.

A genome is a small typed record - which base scheme (or certified
composition) to run and with which stepsize/wrapper genes:

    {"scheme": "fbs"|"drs"|"pdhg"|"condat_vu"|"dys"|"mix",
     "params": {"tau": float, "sigma": float|null, "alpha": float|null,
                "rho": float|null},
     "wrapper": {"halpern": bool, "restart": "none"|"fixed_k",
                 "restart_k": int|null},
     "mix": null
        | {"kind": "avg", "lambda": float in (0,1),
           "scheme_a": {"scheme": <base>, "params": {"tau":..,"sigma":..}},
           "scheme_b": {"scheme": <base>, "params": {"tau":..,"sigma":..}}}
        | {"kind": "switch", "window": int >= 1,
           "aggressive": {"scheme": <base>, "params": {...}},
           "fallback":  {"scheme": <base>, "params": {...}}}}

Gene semantics (mapped onto atlas.schemes):
- tau:   primary stepsize - FBS/DRS/DYS resolvent step `a` (FBS runs on
         the TV dual, smoothness L = ||D||^2 = 8), PDHG/Condat-Vu primal
         step `t`.
- sigma: dual stepsize `s`; meaningful for pdhg/condat_vu only.
- alpha: reserved acceleration/anchor-weight gene in [0, 1]; carried and
         hashed but NOT consumed by any solver yet (P1 momentum hook).
- rho:   over-relaxation in (0, 2) with rho * averagedness < 1 (KM
         combination rule); mutually exclusive with the Halpern anchor.
- wrapper.halpern: Halpern/OHM anchor (nonexpansive maps) [LSCOMO SS12.2].
- wrapper.restart: "fixed_k" resets the Halpern anchor to the current
         iterate every restart_k iterations (CONSUMED by the solvers;
         the P0-inert status is gone). Requires halpern=true - the anchor
         is the only restartable state in this build.
- mix "avg": compiled iteration is lambda*T_a + (1-lambda)*T_b on the
         SAME problem. CERT RULE (typecheck_avg): a convex combination of
         alpha_i-averaged maps IN A COMMON METRIC is
         (lambda*alpha_a + (1-lambda)*alpha_b)-averaged, and its fixed
         points are the COMMON fixed points [LSCOMO SS2.5;
         Bauschke-Combettes Prop 4.42/4.47]. The gate enforces the
         common-metric/common-fixed-point prerequisites structurally:
         parents must be the SAME scheme, with the extra rule that the
         fixed-point ENCODING must also coincide: fbs parents may take
         different stepsizes (the FBS fixed points are the minimizers for
         every valid step), but drs/dys parents must carry IDENTICAL tau
         (their auxiliary-variable fixed-point encoding z depends on the
         resolvent step, so different-tau parents share NO fixed point
         and the blend provably cannot reach a solution); pdhg/condat_vu
         pairs need IDENTICAL (tau, sigma) so the primal-dual metric M
         coincides AND the saddle fixed-point encoding matches. Wrappers
         (rho/halpern/restart) apply on top of the blended map through the
         combined averagedness.
- mix "switch_k": FINITE-PREFIX switching (REPLACES "switch" in the
         proposal menus). z_{k+1} = T_aggressive(z_k) for k < K, then
         T_tail thereafter; the tail simply starts from wherever the
         prefix left off. TYPING RULE (typecheck_switch_k): the TAIL must
         pass the FULL Stage-0 gate (its scheme is globally convergent
         from any start under its certified conditions), so any finite
         prefix cannot affect asymptotic convergence; the AGGRESSIVE
         block must be STRUCTURALLY well-formed (valid scheme name,
         finite positive params) but is EXEMPT from convergence side
         conditions ("finite prefix, exempt" on the cert line). Gate
         constraints: 1 <= K <= 5000; aggressive and tail share the same
         state space (same grouping rule as avg: saddle schemes together,
         primal schemes together); f64 backend only; no outer wrappers.
         Runtime guard: periodic finiteness checks in the prefix and at
         the k = K handoff; a non-finite state restarts the tail from the
         LAST FINITE state (the initial point if none), recorded in
         result.extra.
- metric: optional METRIC GENE, primary form {"kind": "ruiz",
         "iters": int in [0, 30]} (also {"kind": "pock_chambolle",
         "alpha": float in [0, 2]}). Implemented as PROBLEM RESCALING,
         not new typing rules: at compile time Ruiz equilibration (or the
         Pock-Chambolle diagonal rule) on the explicit entries of L
         yields diagonals D1, D2; the solver runs on the equivalent
         rescaled problem (L' = D1 L D2, transformed data; derivation in
         atlas.schemes.rescale) with the operator-norm bound of L'
         recomputed by power iteration x1.02 and fed to the EXISTING
         gates; termination is the ORIGINAL problem's f64 duality gap.
         Scope (this build): lp_netflow base genomes (pdhg/condat_vu)
         only; the TV cell is rejected because diagonal rescaling does
         not preserve the quadratic-fidelity + scaled-L1 atom structure.
- mix "switch": DEPRECATED residual-window switching (kept runnable and
         tested; REMOVED from both proposal menus in favor of
         "switch_k"). The FALLBACK must pass the full
         Stage-0 gate; the AGGRESSIVE branch is explicitly EXEMPT from
         side conditions (that is the design: a step is accepted only when
         the rigorous Level-3 gap certificate decreased below the best of the last `window`
         iterations, so the safeguard inherits the fallback's
         convergence). Aggressive and fallback must share a state space
         (same scheme, or both in {pdhg, condat_vu}). No outer wrappers on
         a switch genome (scope: the controller IS the wrapper).

STAGE-0 GATE. `validate(genome, problem_kind)` maps the genome onto the
Level-1 construction rules (atlas.typing_.rules) for the canonical
instance constants of `problem_kind` and returns a ValidationResult. A
genome whose side condition fails is REJECTED - ok=False with the
violated-condition string (LLM repair feedback). Never clamps.
problem_kind:
- "tv_denoise" (default): the canonical TV cell (||L||^2 = 8, L_f = 1).
  dys is structurally inapplicable here (h composed with L) and is
  rejected with that message.
- "three_composite": f smooth + g + h without L (e.g. lasso_box). Only
  dys applies; L_f = 1.

Dedup: `canonical_json` normalizes the genome (floats to 12 significant
digits, inapplicable genes nulled) and `content_hash` is its sha256.

This module is pure Python (no jax import), so it is safe in any process.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Optional

from atlas.typing_ import (
    ConstructionCert,
    TypeCheckError,
    typecheck_avg,
    typecheck_condat_vu,
    typecheck_drs,
    typecheck_dys,
    typecheck_fbs,
    typecheck_halpern,
    typecheck_pdhg,
    typecheck_relaxation,
    typecheck_restart,
    typecheck_switch,
    typecheck_switch_k,
)

SCHEMES = ("fbs", "drs", "pdhg", "condat_vu", "dys")
TV_SCHEMES = ("fbs", "drs", "pdhg", "condat_vu")  # dys needs no-L problems
# "switch" is DEPRECATED (validates and runs; absent from proposal menus).
MIX_KINDS = ("avg", "switch", "switch_k")
RESTART_KINDS = ("none", "fixed_k")
# Registered outer wrappers (atlas.wrappers.registry; each is a knowledge-base
# entry with a gate rule and an implementation). Kept in sync by
# tests/test_haugazeau.py::test_registry_matches_kb_and_schema.
WRAPPER_KINDS = ("haugazeau",)
PROBLEM_KINDS = ("tv_denoise", "three_composite", "lp_netflow")
MAX_SWITCH_WINDOW = 100_000
MAX_SWITCH_K = 5000  # finite-prefix length bound (switch_k)
# Metric gene (problem rescaling; see atlas.schemes.rescale).
METRIC_KINDS = ("ruiz", "pock_chambolle", "ruiz_pc")
MAX_RUIZ_ITERS = 30
MAX_PC_ALPHA = 2.0
# Problem kinds whose composite structure is CLOSED under diagonal
# rescaling (the rescaled problem is the same atom family, so every
# existing typing rule applies verbatim): the standard-form LP only.
METRIC_PROBLEM_KINDS = ("lp_netflow",)
# Backend gene (execution policy; see atlas/backend/CONTRACT.md). The gene
# changes HOW T is computed, never WHAT T is: Stage-0 validates it
# structurally and it never touches the math side conditions.
BACKEND_PRECISIONS = ("f64", "f32_cert64", "schedule")
# Placement gene values. None (absent) means the process default device.
BACKEND_DEVICES = ("cpu", "gpu")
_SUB_F32_SPELLINGS = ("bf16", "bfloat16", "f16", "float16", "half")
MAX_BACKEND_K = 10_000

# Canonical instance constants; these mirror what atlas.schemes derives
# from the problem objects.
OPNORM = math.sqrt(8.0)  # ||D|| bound (Grad2D.norm_bound)
L_SMOOTH_PRIMAL = 1.0  # f = (1/2)||x - y||^2 is 1-smooth
L_SMOOTH_DUAL = OPNORM**2  # FBS runs on the TV dual: L = ||D||^2 = 8
# lp_netflow: standard-form LP min c'x s.t. Ax=b, x>=0 on DEGREE-2 netflow
# incidence matrices (atlas.bench.netflow, gen_regular_flow(degree=2)).
# ||A||^2 = 2 * lambda_max(graph Laplacian) <= 4*degree = 8 for degree 2,
# and the solver-side norm_bound carries the power-iteration safety factor
# 1.02, so the gate constant is 1.02*sqrt(8).
#
# SCOPE (corrected 2026-08-26, sprint finding): gate acceptance implies
# instance-level typecheck acceptance ONLY for the degree-2 regular family
# (the mini-E-R protocol family). Other lp_cell families have LARGER
# operator norms (measured: regular degree-4 ~3.867, grid k=5 ~3.880, both
# > 1.02*sqrt(8) = 2.885); a genome gate-valid under OPNORM_LP can then
# fail the mandatory solve-time typecheck on those instances (a certified
# construction failure, not a soundness hole). For any other family pass
# the instance set's actual bound via validate(..., lp_opnorm=...), e.g.
# atlas.bench.lp_cell.gate_opnorm(instances). f = <c, x> is linear:
# L_f = 0 (Condat-Vu reduces to the PDHG condition).
OPNORM_LP = 1.02 * math.sqrt(8.0)
# Mirror of atlas.schemes.rescale.PC_CERTIFIED_NORM * PC_SAFETY, duplicated
# here so the genome package stays jax-free (rescale pulls jax). A test
# asserts the two never drift apart.
PC_GATE_OPNORM = 1.0 * (1.0 + 1e-9)
L_SMOOTH_LP = 0.0


class GenomeFormatError(Exception):
    """The candidate is not a structurally well-formed genome (pre-typecheck)."""


@dataclass(frozen=True)
class GenomeParams:
    tau: float
    sigma: Optional[float] = None
    alpha: Optional[float] = None
    rho: Optional[float] = None


@dataclass(frozen=True)
class Wrapper:
    halpern: bool = False
    restart: str = "none"  # "none" | "fixed_k"
    restart_k: Optional[int] = None
    # None = legacy wrapper genes above only. A registered kind (WRAPPER_KINDS)
    # is serialized and hashed ONLY when set, so every legacy hash is unchanged.
    kind: Optional[str] = None


@dataclass(frozen=True)
class SubScheme:
    """A bare (scheme, tau, sigma) parent inside a mix block."""

    scheme: str
    tau: float
    sigma: Optional[float] = None


@dataclass(frozen=True)
class Mix:
    kind: str  # "avg" | "switch" (deprecated) | "switch_k"
    lam: Optional[float] = None  # avg only
    a: Optional[SubScheme] = None  # avg parents
    b: Optional[SubScheme] = None
    aggressive: Optional[SubScheme] = None  # switch + switch_k branches
    fallback: Optional[SubScheme] = None  # switch only
    window: Optional[int] = None  # switch only
    tail: Optional[SubScheme] = None  # switch_k only (fully certified)
    K: Optional[int] = None  # switch_k prefix length, 1 <= K <= 5000


@dataclass(frozen=True)
class Metric:
    """Metric gene: certified problem rescaling (atlas.schemes.rescale).

    kind "ruiz": row/col infinity-norm equilibration, `iters` rounds in
    [0, 30] (0 = identity). kind "pock_chambolle": the Pock-Chambolle
    diagonal preconditioning rule with exponent `alpha` in [0, 2]
    (Pock & Chambolle 2011). kind "ruiz_pc": Ruiz(`iters`) THEN
    Pock-Chambolle(`alpha`), the PDLP pipeline and the frozen LP
    substrate; because PC is applied LAST its universal Lemma-2 bound
    holds for the composed operator, so any `iters` enters the same
    certificate pathway. Either way the gene changes the PROBLEM
    (equivalently rescaled), never the typing rules: the recomputed
    ||L'|| bound feeds the existing side conditions, and the runtime
    certificate is evaluated on the ORIGINAL problem's gap."""

    kind: str = "ruiz"
    iters: Optional[int] = None  # ruiz and ruiz_pc
    alpha: Optional[float] = None  # pock_chambolle and ruiz_pc


@dataclass(frozen=True)
class Backend:
    """Execution-policy gene: iterate precision + certificate cadence.

    precision: "f64" (baseline), "f32_cert64" (f32 iterates, f64
    certificates every K), "schedule" (f32 start, switch to f64 at
    certified gap <= theta or on plateau). Certificates are NEVER
    evaluated below f64 (atlas.backend.precision enforces mechanically).
    K: certificate cadence in iterations. theta: schedule switch
    threshold, required iff precision == "schedule". batch: reserved for
    the P4 vmap backend (validated, recorded, currently a no-op)."""

    precision: str = "f64"
    K: int = 100
    theta: Optional[float] = None
    batch: bool = False
    kernel: str = "jax"  # bounded generated LP/TV update template; f64 only
    # Placement gene. None = the process default device, which is how every
    # genome written before this gene existed behaves, so their serialisation
    # and content hashes are unchanged. "cpu" / "gpu" pin execution; the process
    # must hold that backend (JAX_PLATFORMS=cuda,cpu) and a missing backend is
    # an error, never a silent fallback.
    device: Optional[str] = None


@dataclass(frozen=True)
class Genome:
    scheme: str
    params: GenomeParams = field(default_factory=lambda: GenomeParams(tau=1.0))
    wrapper: Wrapper = field(default_factory=Wrapper)
    mix: Optional[Mix] = None
    backend: Optional[Backend] = None  # absent = the exact P0 f64 path
    metric: Optional[Metric] = None  # absent = no rescaling


@dataclass(frozen=True)
class ValidationResult:
    """Stage-0 gate outcome: ok + Level-1 cert, or the violated condition."""

    ok: bool
    cert: Optional[ConstructionCert] = None
    error: Optional[str] = None


def _family(scheme: str) -> str:
    """State-space family: schemes in the same family iterate on the same
    variable with the same fixed-point encoding (blend/switch precondition)."""
    return "saddle" if scheme in ("pdhg", "condat_vu") else scheme


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def _sub_to_dict(s: Optional[SubScheme]) -> Optional[dict]:
    if s is None:
        return None
    return {"scheme": s.scheme, "params": {"tau": s.tau, "sigma": s.sigma}}


def genome_to_dict(g: Genome) -> dict:
    d = {
        "scheme": g.scheme,
        "params": {
            "tau": g.params.tau,
            "sigma": g.params.sigma,
            "alpha": g.params.alpha,
            "rho": g.params.rho,
        },
        "wrapper": {
            "halpern": g.wrapper.halpern,
            "restart": g.wrapper.restart,
            "restart_k": g.wrapper.restart_k,
        },
        "mix": None,
        "backend": None,
        "metric": None,
    }
    if g.mix is not None:
        m = g.mix
        if m.kind == "avg":
            d["mix"] = {
                "kind": "avg",
                "lambda": m.lam,
                "scheme_a": _sub_to_dict(m.a),
                "scheme_b": _sub_to_dict(m.b),
            }
        elif m.kind == "switch_k":
            d["mix"] = {
                "kind": "switch_k",
                "K": m.K,
                "aggressive": _sub_to_dict(m.aggressive),
                "tail": _sub_to_dict(m.tail),
            }
        else:
            d["mix"] = {
                "kind": "switch",
                "window": m.window,
                "aggressive": _sub_to_dict(m.aggressive),
                "fallback": _sub_to_dict(m.fallback),
            }
    if g.wrapper.kind is not None:
        d["wrapper"]["kind"] = g.wrapper.kind
    if g.metric is not None:
        d["metric"] = {
            "kind": g.metric.kind,
            "iters": g.metric.iters,
            "alpha": g.metric.alpha,
        }
    if g.backend is not None:
        d["backend"] = {
            "precision": g.backend.precision,
            "K": g.backend.K,
            "theta": g.backend.theta,
            "batch": g.backend.batch,
        }
        if g.backend.kernel != "jax":
            d["backend"]["kernel"] = g.backend.kernel
        if g.backend.device is not None:
            d["backend"]["device"] = g.backend.device
    return d


def _num(v, name: str) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise GenomeFormatError(f"{name} must be a number, got {v!r}")
    return float(v)


def _opt_num(d: dict, key: str, name: str) -> Optional[float]:
    v = d.get(key)
    return None if v is None else _num(v, name)


def _sub_from_dict(d, where: str) -> SubScheme:
    if not isinstance(d, dict):
        raise GenomeFormatError(f"{where} must be an object with scheme/params")
    scheme = d.get("scheme")
    if not isinstance(scheme, str) or scheme.strip().lower() not in SCHEMES:
        raise GenomeFormatError(
            f"{where}.scheme must be one of {list(SCHEMES)}, got {scheme!r}"
        )
    scheme = scheme.strip().lower()
    p = d.get("params")
    if not isinstance(p, dict) or p.get("tau") is None:
        raise GenomeFormatError(f"{where}.params must be an object with 'tau'")
    for bad in ("rho", "halpern"):
        if p.get(bad) not in (None, False):
            raise GenomeFormatError(
                f"{where}.params carries only tau/sigma; rho and wrappers "
                "belong at the mix genome's top level"
            )
    return SubScheme(
        scheme=scheme,
        tau=_num(p["tau"], f"{where}.params.tau"),
        sigma=_opt_num(p, "sigma", f"{where}.params.sigma"),
    )


def _mix_from_dict(d) -> Mix:
    if not isinstance(d, dict):
        raise GenomeFormatError("'mix' must be an object")
    kind = d.get("kind")
    if not isinstance(kind, str) or kind.strip().lower() not in MIX_KINDS:
        raise GenomeFormatError(f"mix.kind must be one of {list(MIX_KINDS)}, got {kind!r}")
    kind = kind.strip().lower()
    if kind == "avg":
        lam = d.get("lambda", d.get("lam"))
        if lam is None:
            raise GenomeFormatError("mix.lambda is required for kind='avg'")
        return Mix(
            kind="avg",
            lam=_num(lam, "mix.lambda"),
            a=_sub_from_dict(d.get("scheme_a"), "mix.scheme_a"),
            b=_sub_from_dict(d.get("scheme_b"), "mix.scheme_b"),
        )
    if kind == "switch_k":
        K = d.get("K", d.get("k"))
        if isinstance(K, bool) or not isinstance(K, int):
            raise GenomeFormatError(
                f"mix.K must be a positive integer, got {K!r}"
            )
        return Mix(
            kind="switch_k",
            K=K,
            aggressive=_sub_from_dict(d.get("aggressive"), "mix.aggressive"),
            tail=_sub_from_dict(d.get("tail"), "mix.tail"),
        )
    window = d.get("window")
    if isinstance(window, bool) or not isinstance(window, int):
        raise GenomeFormatError(
            f"mix.window must be a positive integer, got {window!r}"
        )
    return Mix(
        kind="switch",
        window=window,
        aggressive=_sub_from_dict(d.get("aggressive"), "mix.aggressive"),
        fallback=_sub_from_dict(d.get("fallback"), "mix.fallback"),
    )


def _metric_from_dict(d) -> Metric:
    if not isinstance(d, dict):
        raise GenomeFormatError("'metric' must be an object or null")
    kind = d.get("kind")
    if not isinstance(kind, str) or kind.strip().lower() not in METRIC_KINDS:
        raise GenomeFormatError(
            f"metric.kind must be one of {list(METRIC_KINDS)}, got {kind!r}"
        )
    kind = kind.strip().lower()
    iters = d.get("iters")
    if iters is not None and (isinstance(iters, bool) or not isinstance(iters, int)):
        raise GenomeFormatError(f"metric.iters must be an integer, got {iters!r}")
    alpha = d.get("alpha")
    if alpha is not None:
        alpha = _num(alpha, "metric.alpha")
    return Metric(kind=kind, iters=iters, alpha=alpha)


def genome_from_dict(d) -> Genome:
    """Parse a genome from a plain dict (e.g. LLM JSON). Structural checks
    only; raises GenomeFormatError with an LLM-feedback-ready message.
    Range/side-condition checks are validate()'s job."""
    if not isinstance(d, dict):
        raise GenomeFormatError(f"genome must be a JSON object, got {type(d).__name__}")
    scheme = d.get("scheme")
    if not isinstance(scheme, str):
        raise GenomeFormatError(
            f"'scheme' must be one of {list(SCHEMES) + ['mix']}, got {scheme!r}"
        )
    scheme = scheme.strip().lower()
    is_mix = scheme == "mix" or d.get("mix") is not None
    p = d.get("params")
    if p is None and is_mix:
        p = {}
    if not isinstance(p, dict):
        raise GenomeFormatError("'params' must be an object with at least 'tau'")
    if p.get("tau") is None and not is_mix:
        raise GenomeFormatError("'params.tau' is required")
    params = GenomeParams(
        tau=_num(p["tau"], "params.tau") if p.get("tau") is not None else 1.0,
        sigma=_opt_num(p, "sigma", "params.sigma"),
        alpha=_opt_num(p, "alpha", "params.alpha"),
        rho=_opt_num(p, "rho", "params.rho"),
    )
    w = d.get("wrapper", {})
    if w is None:
        w = {}
    if not isinstance(w, dict):
        raise GenomeFormatError("'wrapper' must be an object")
    halpern = w.get("halpern", False)
    if not isinstance(halpern, bool):
        raise GenomeFormatError(f"wrapper.halpern must be true/false, got {halpern!r}")
    restart = w.get("restart", "none")
    if not isinstance(restart, str):
        raise GenomeFormatError(f"wrapper.restart must be a string, got {restart!r}")
    restart = restart.strip().lower()
    restart_k = w.get("restart_k")
    if restart_k is not None:
        if isinstance(restart_k, bool) or not isinstance(restart_k, int):
            raise GenomeFormatError(
                f"wrapper.restart_k must be a positive integer, got {restart_k!r}"
            )
    wkind = w.get("kind")
    if wkind is not None:
        if not isinstance(wkind, str):
            raise GenomeFormatError(f"wrapper.kind must be a string or null, got {wkind!r}")
        wkind = wkind.strip().lower()
    mix = None
    if is_mix:
        if d.get("mix") is None:
            raise GenomeFormatError("scheme 'mix' requires a 'mix' block")
        if scheme != "mix":
            raise GenomeFormatError(
                "a genome with a 'mix' block must set scheme='mix'"
            )
        mix = _mix_from_dict(d["mix"])
    backend = None
    bd = d.get("backend")
    if bd is not None:
        if not isinstance(bd, dict):
            raise GenomeFormatError("'backend' must be an object or null")
        precision = bd.get("precision", "f64")
        if not isinstance(precision, str):
            raise GenomeFormatError(
                f"backend.precision must be a string, got {precision!r}"
            )
        K = bd.get("K", 100)
        if isinstance(K, bool) or not isinstance(K, int):
            raise GenomeFormatError(
                f"backend.K must be an integer, got {K!r}"
            )
        theta = bd.get("theta")
        if theta is not None:
            theta = _num(theta, "backend.theta")
        batch = bd.get("batch", False)
        if not isinstance(batch, bool):
            raise GenomeFormatError(
                f"backend.batch must be true/false, got {batch!r}"
            )
        kernel = bd.get("kernel", "jax")
        if not isinstance(kernel, str):
            raise GenomeFormatError("backend.kernel must be a string")
        device = bd.get("device")
        if device is not None and not isinstance(device, str):
            raise GenomeFormatError("backend.device must be a string or null")
        backend = Backend(
            precision=precision.strip().lower(), K=K, theta=theta, batch=batch,
            kernel=kernel.strip().lower(),
            device=None if device is None else device.strip().lower())
    metric = None
    md = d.get("metric")
    if md is not None:
        metric = _metric_from_dict(md)
    return Genome(
        scheme=scheme,
        params=params,
        wrapper=Wrapper(halpern=halpern, restart=restart, restart_k=restart_k,
                        kind=wkind),
        mix=mix,
        backend=backend,
        metric=metric,
    )


# ---------------------------------------------------------------------------
# Canonical form + content hash (exact dedup)
# ---------------------------------------------------------------------------


def _round(v: Optional[float]) -> Optional[float]:
    return None if v is None else float(f"{float(v):.12g}")


def _sub_canonical(s: Optional[SubScheme]) -> Optional[dict]:
    if s is None:
        return None
    needs_sigma = s.scheme in ("pdhg", "condat_vu")
    return {
        "scheme": s.scheme,
        "params": {
            "tau": _round(s.tau),
            "sigma": _round(s.sigma) if needs_sigma else None,
        },
    }


def canonical_dict(g: Genome) -> dict:
    """Normalized genome: inapplicable genes nulled so behaviorally identical
    genomes hash identically (sigma is meaningless for fbs/drs/dys; tau and
    sigma are meaningless at the top level of a mix; restart_k is
    meaningless when restart == 'none')."""
    is_mix = g.scheme == "mix"
    needs_sigma = g.scheme in ("pdhg", "condat_vu")
    fixed_k = g.wrapper.restart == "fixed_k"
    d = {
        "scheme": g.scheme,
        "params": {
            "tau": None if is_mix else _round(g.params.tau),
            "sigma": _round(g.params.sigma) if needs_sigma else None,
            "alpha": _round(g.params.alpha),
            "rho": _round(g.params.rho),
        },
        "wrapper": {
            "halpern": bool(g.wrapper.halpern),
            "restart": g.wrapper.restart,
            "restart_k": (
                int(g.wrapper.restart_k)
                if (fixed_k and g.wrapper.restart_k is not None)
                else None
            ),
        },
        "mix": None,
        # Convention (CONTRACT.md section 1): an ABSENT backend block is NOT
        # normalized to the explicit f64 default; the two hash differently.
        "backend": None,
        "metric": None,
    }
    if g.mix is not None:
        m = g.mix
        if m.kind == "avg":
            d["mix"] = {
                "kind": "avg",
                "lambda": _round(m.lam),
                "scheme_a": _sub_canonical(m.a),
                "scheme_b": _sub_canonical(m.b),
            }
        elif m.kind == "switch_k":
            d["mix"] = {
                "kind": "switch_k",
                "K": None if m.K is None else int(m.K),
                "aggressive": _sub_canonical(m.aggressive),
                "tail": _sub_canonical(m.tail),
            }
        else:
            d["mix"] = {
                "kind": "switch",
                "window": None if m.window is None else int(m.window),
                "aggressive": _sub_canonical(m.aggressive),
                "fallback": _sub_canonical(m.fallback),
            }
    if g.wrapper.kind is not None:
        d["wrapper"]["kind"] = g.wrapper.kind
    if g.metric is not None:
        d["metric"] = {
            "kind": g.metric.kind,
            "iters": (
                int(g.metric.iters)
                if (g.metric.kind in ("ruiz", "ruiz_pc")
                    and g.metric.iters is not None)
                else None
            ),
            "alpha": (
                _round(g.metric.alpha)
                if g.metric.kind in ("pock_chambolle", "ruiz_pc") else None
            ),
        }
    if g.backend is not None:
        b = g.backend
        d["backend"] = {
            "precision": b.precision,
            "K": None if b.K is None else int(b.K),
            "theta": _round(b.theta) if b.precision == "schedule" else None,
            "batch": bool(b.batch),
        }
        if b.kernel != "jax":
            d["backend"]["kernel"] = b.kernel
        if b.device is not None:
            d["backend"]["device"] = b.device
    return d


def canonical_json(g: Genome) -> str:
    return json.dumps(canonical_dict(g), sort_keys=True, separators=(",", ":"))


def content_hash(g: Genome) -> str:
    return hashlib.sha256(canonical_json(g).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Stage-0 gate: validate = core Level-1 typecheck. Reject, never clamp.
# ---------------------------------------------------------------------------


def _sub_structural_error(s: SubScheme, where: str) -> Optional[str]:
    if s.scheme not in SCHEMES:
        return f"{where}.scheme must be one of {list(SCHEMES)}, got {s.scheme!r}"
    if not math.isfinite(s.tau) or s.tau <= 0:
        return f"{where}.params.tau must be a finite positive number, got {s.tau!r}"
    if s.scheme in ("pdhg", "condat_vu"):
        if s.sigma is None:
            return f"{where}.params.sigma is required for scheme {s.scheme!r}"
        if not math.isfinite(s.sigma) or s.sigma <= 0:
            return (
                f"{where}.params.sigma must be a finite positive number, "
                f"got {s.sigma!r}"
            )
    return None


def _structural_error(g: Genome) -> Optional[str]:
    is_mix = g.scheme == "mix"
    if not is_mix and g.scheme not in SCHEMES:
        return f"'scheme' must be one of {list(SCHEMES) + ['mix']}, got {g.scheme!r}"
    if is_mix and g.mix is None:
        return "scheme 'mix' requires a 'mix' block"
    if not is_mix and g.mix is not None:
        return "a genome with a 'mix' block must set scheme='mix'"
    if not is_mix:
        if not math.isfinite(g.params.tau):
            return f"params.tau must be finite, got {g.params.tau!r}"
        if g.scheme in ("pdhg", "condat_vu"):
            if g.params.sigma is None:
                return f"params.sigma is required for scheme {g.scheme!r}"
            if not math.isfinite(g.params.sigma):
                return f"params.sigma must be finite, got {g.params.sigma!r}"
    if g.params.alpha is not None and not (0.0 <= g.params.alpha <= 1.0):
        return f"params.alpha must be in [0, 1] or null, got {g.params.alpha!r}"
    if g.params.rho is not None and not math.isfinite(g.params.rho):
        return f"params.rho must be finite or null, got {g.params.rho!r}"
    if g.wrapper.restart not in RESTART_KINDS:
        return f"wrapper.restart must be one of {list(RESTART_KINDS)}, got {g.wrapper.restart!r}"
    if g.wrapper.restart == "fixed_k":
        k = g.wrapper.restart_k
        if k is None or k <= 0:
            return f"wrapper.restart_k must be a positive integer when restart='fixed_k', got {k!r}"
        if not g.wrapper.halpern:
            return (
                "wrapper.restart='fixed_k' requires wrapper.halpern=true: "
                "the Halpern anchor is the only restartable state in this "
                "build (plain KM steps are memoryless)"
            )
    if g.wrapper.kind is not None:
        if g.wrapper.kind not in WRAPPER_KINDS:
            return (f"wrapper.kind must be one of {list(WRAPPER_KINDS)} or absent, "
                    f"got {g.wrapper.kind!r}")
        if g.wrapper.halpern or g.wrapper.restart != "none":
            return ("wrapper.kind is an outer step of its own; it does not combine "
                    "with wrapper.halpern/restart in this build")
        if g.params.rho is not None:
            return "wrapper.kind replaces the relaxation step; params.rho must be null"
        if is_mix:
            return "wrapper.kind applies to base genomes only in this build"
    if is_mix:
        m = g.mix
        if m.kind not in MIX_KINDS:
            return f"mix.kind must be one of {list(MIX_KINDS)}, got {m.kind!r}"
        if m.kind == "avg":
            if m.a is None or m.b is None:
                return "mix kind 'avg' requires scheme_a and scheme_b"
            if m.lam is None or not math.isfinite(m.lam) or not (0.0 < m.lam < 1.0):
                return f"mix.lambda must be in the open interval (0, 1), got {m.lam!r}"
            for s, where in ((m.a, "mix.scheme_a"), (m.b, "mix.scheme_b")):
                err = _sub_structural_error(s, where)
                if err:
                    return err
            fa, fb = _family(m.a.scheme), _family(m.b.scheme)
            if fa != fb:
                return (
                    f"avg blend needs a shared state space: parents must be "
                    f"the same scheme or both in (pdhg, condat_vu); got "
                    f"{m.a.scheme!r} and {m.b.scheme!r}"
                )
            if fa == "saddle" and (m.a.tau != m.b.tau or m.a.sigma != m.b.sigma):
                return (
                    "avg blend of primal-dual schemes requires IDENTICAL "
                    "(tau, sigma) on both parents: averagedness is metric-"
                    "dependent (M = [[I/t, -L^T], [-L, I/s]]) and the convex-"
                    "combination rule needs a common metric"
                )
            if fa in ("drs", "dys") and m.a.tau != m.b.tau:
                return (
                    "avg blend of drs/dys parents requires IDENTICAL tau: "
                    "the drs/dys fixed-point encoding depends on the "
                    "resolvent step (the auxiliary iterate z encodes the "
                    "solution through prox with step tau), so parents with "
                    "different tau share NO common fixed point and the "
                    "blend cannot converge to a solution of either parent"
                )
        elif m.kind == "switch_k":
            if m.aggressive is None or m.tail is None:
                return "mix kind 'switch_k' requires aggressive and tail"
            if (
                m.K is None
                or isinstance(m.K, bool)
                or not isinstance(m.K, int)
                or not (1 <= m.K <= MAX_SWITCH_K)
            ):
                return (
                    f"mix.K must be an integer in [1, {MAX_SWITCH_K}], "
                    f"got {m.K!r}"
                )
            for s, where in (
                (m.aggressive, "mix.aggressive"),
                (m.tail, "mix.tail"),
            ):
                err = _sub_structural_error(s, where)
                if err:
                    return err
            if _family(m.aggressive.scheme) != _family(m.tail.scheme):
                return (
                    "switch_k needs a shared state space: aggressive and "
                    "tail must be the same scheme or both in "
                    f"(pdhg, condat_vu); got {m.aggressive.scheme!r} and "
                    f"{m.tail.scheme!r}"
                )
            if g.wrapper.halpern or g.wrapper.restart != "none" or g.params.rho is not None:
                return (
                    "a switch_k genome takes no outer wrappers (rho/halpern/"
                    "restart) in this build"
                )
        else:  # switch (deprecated residual-window form)
            if m.aggressive is None or m.fallback is None:
                return "mix kind 'switch' requires aggressive and fallback"
            if (
                m.window is None
                or not isinstance(m.window, int)
                or not (1 <= m.window <= MAX_SWITCH_WINDOW)
            ):
                return (
                    f"mix.window must be an integer in [1, {MAX_SWITCH_WINDOW}], "
                    f"got {m.window!r}"
                )
            for s, where in (
                (m.aggressive, "mix.aggressive"),
                (m.fallback, "mix.fallback"),
            ):
                err = _sub_structural_error(s, where)
                if err:
                    return err
            if _family(m.aggressive.scheme) != _family(m.fallback.scheme):
                return (
                    "switch needs a shared state space: aggressive and "
                    "fallback must be the same scheme or both in "
                    f"(pdhg, condat_vu); got {m.aggressive.scheme!r} and "
                    f"{m.fallback.scheme!r}"
                )
            if g.wrapper.halpern or g.wrapper.restart != "none" or g.params.rho is not None:
                return (
                    "a switch genome takes no outer wrappers (rho/halpern/"
                    "restart): the safeguard controller IS the wrapper"
                )
    # Backend gene: STRUCTURAL validation only; it never enters the math
    # typechecks (execution policy changes HOW T is computed, not WHAT T is).
    if g.backend is not None:
        b = g.backend
        if b.device is not None and b.device not in BACKEND_DEVICES:
            return (f"backend.device must be one of {list(BACKEND_DEVICES)} "
                    f"or null, got {b.device!r}")
        if b.kernel != "jax" and b.device == "cpu":
            return "generated CUDA kernels cannot run with backend.device='cpu'"
        if b.kernel not in ("jax", "csr_thread128", "csr_warp128", "csr_warp256", "tv_fused128", "tv_fused256", "tv_recip256"):
            return "backend.kernel must be jax, a csr_thread/warp variant, or tv_fused128/tv_fused256/tv_recip256"
        if b.kernel.startswith("csr_") and (g.scheme != "pdhg" or b.precision != "f64"):
            return "generated CSR kernels require pdhg and f64; LP compatibility is checked at compilation"
        if b.kernel.startswith("tv_") and (g.scheme not in ("pdhg", "condat_vu") or b.precision != "f64" or g.metric is not None):
            return "generated TV kernels require pdhg/condat_vu and f64 without metric transforms"
        if not isinstance(b.precision, str):
            return f"backend.precision must be a string, got {b.precision!r}"
        if b.precision in _SUB_F32_SPELLINGS:
            return (
                f"backend.precision {b.precision!r} is refused deliberately: "
                "the duality-gap certificate is a difference of near-equal "
                "numbers (catastrophic cancellation); bf16/f16 cannot "
                "represent a 1e-4 relative gap at all, and certificates are "
                "evaluated at exact f64 only. Valid precisions: "
                f"{list(BACKEND_PRECISIONS)}"
            )
        if b.precision not in BACKEND_PRECISIONS:
            return (
                f"backend.precision must be one of {list(BACKEND_PRECISIONS)}, "
                f"got {b.precision!r}"
            )
        if (
            isinstance(b.K, bool)
            or not isinstance(b.K, int)
            or not (1 <= b.K <= MAX_BACKEND_K)
        ):
            return (
                f"backend.K must be an integer in [1, {MAX_BACKEND_K}], "
                f"got {b.K!r}"
            )
        if b.precision == "schedule":
            if (
                b.theta is None
                or isinstance(b.theta, bool)
                or not isinstance(b.theta, (int, float))
                or not math.isfinite(b.theta)
                or not (0.0 < float(b.theta) < 1.0)
            ):
                return (
                    "backend.theta must be a finite number in the open "
                    "interval (0, 1) when precision='schedule' (it must also "
                    f"exceed the run's tolerance), got {b.theta!r}"
                )
        elif b.theta is not None:
            return (
                "backend.theta applies only to precision='schedule'; it must "
                f"be null for precision {b.precision!r}"
            )
        if not isinstance(b.batch, bool):
            return f"backend.batch must be true/false, got {b.batch!r}"
        if (
            g.scheme == "mix"
            and g.mix is not None
            and g.mix.kind in ("switch", "switch_k")
            and b.precision != "f64"
        ):
            return (
                f"a {g.mix.kind} genome supports only the f64 backend in "
                "this build: the switch runtimes evaluate their guards "
                "inside their own compiled f64 loops"
            )
    # Metric gene: structural + range checks (problem-kind applicability is
    # validate()'s job since it depends on problem_kind).
    if g.metric is not None:
        mt = g.metric
        if mt.kind not in METRIC_KINDS:
            return f"metric.kind must be one of {list(METRIC_KINDS)}, got {mt.kind!r}"
        if mt.kind == "ruiz":
            if (
                mt.iters is None
                or isinstance(mt.iters, bool)
                or not isinstance(mt.iters, int)
                or not (0 <= mt.iters <= MAX_RUIZ_ITERS)
            ):
                return (
                    f"metric.iters must be an integer in [0, {MAX_RUIZ_ITERS}] "
                    f"for kind='ruiz', got {mt.iters!r}"
                )
            if mt.alpha is not None:
                return ("metric.alpha applies only to kind='pock_chambolle' "
                        "or 'ruiz_pc'")
        else:  # pock_chambolle or ruiz_pc (both carry alpha)
            if (
                mt.alpha is None
                or not isinstance(mt.alpha, (int, float))
                or isinstance(mt.alpha, bool)
                or not math.isfinite(mt.alpha)
                or not (0.0 <= float(mt.alpha) <= MAX_PC_ALPHA)
            ):
                return (
                    f"metric.alpha must be a finite number in [0, "
                    f"{MAX_PC_ALPHA:g}] for kind={mt.kind!r}, got "
                    f"{mt.alpha!r}"
                )
            # "ruiz_pc" is Ruiz(iters) THEN Pock-Chambolle(alpha), so it is
            # the one kind that legitimately carries BOTH knobs. This rule
            # previously rejected every ruiz_pc genome, and nothing caught it
            # because calibrate_b0 builds genomes with genome_from_dict and
            # never calls validate(); the proposer and the assay both do.
            if mt.kind == "ruiz_pc":
                if (
                    mt.iters is None
                    or isinstance(mt.iters, bool)
                    or not isinstance(mt.iters, int)
                    or not (0 <= mt.iters <= MAX_RUIZ_ITERS)
                ):
                    return (
                        f"metric.iters must be an integer in [0, "
                        f"{MAX_RUIZ_ITERS}] for kind='ruiz_pc', got "
                        f"{mt.iters!r}"
                    )
            elif mt.iters is not None:
                return "metric.iters applies only to kind='ruiz' or 'ruiz_pc'"
        if g.scheme == "mix":
            return (
                "the metric gene applies to BASE genomes only in this build "
                "(pdhg/condat_vu on the LP cell); remove 'metric' or 'mix'"
            )
    return None


def _base_cert(scheme: str, tau: float, sigma, problem_kind: str,
               lp_opnorm: Optional[float] = None) -> ConstructionCert:
    """Level-1 cert for a bare (scheme, tau, sigma) on the canonical
    instance constants of problem_kind. Raises TypeCheckError.

    lp_opnorm: override of the lp_netflow gate operator norm for instance
    families whose ||A|| exceeds the degree-2 constant OPNORM_LP (see the
    SCOPE note at OPNORM_LP; sound value: max instance norm_bound, e.g.
    atlas.bench.lp_cell.gate_opnorm)."""
    if problem_kind == "tv_denoise":
        if scheme == "dys":
            raise TypeCheckError(
                "dys is structurally inapplicable to the TV cell: h is "
                "composed with L (prox of h∘L unavailable); dys needs a "
                "three-block composite without L (problem_kind="
                "'three_composite')"
            )
        if scheme == "fbs":
            return typecheck_fbs(L_smooth=L_SMOOTH_DUAL, step=tau)
        if scheme == "drs":
            return typecheck_drs(step=tau)
        if scheme == "pdhg":
            return typecheck_pdhg(t=tau, s=sigma, opnorm=OPNORM)
        return typecheck_condat_vu(
            t=tau, s=sigma, opnorm=OPNORM, L_smooth=L_SMOOTH_PRIMAL
        )
    if problem_kind == "lp_netflow":
        opn = OPNORM_LP if lp_opnorm is None else float(lp_opnorm)
        if scheme == "pdhg":
            return typecheck_pdhg(t=tau, s=sigma, opnorm=opn)
        if scheme == "condat_vu":
            return typecheck_condat_vu(
                t=tau, s=sigma, opnorm=opn, L_smooth=L_SMOOTH_LP
            )
        raise TypeCheckError(
            f"{scheme} is structurally inapplicable to the standard-form LP "
            "cell: f = <c, x> is linear (not strongly convex, so the fbs/drs "
            "dual paths do not exist) and h is composed with L (no dys); "
            "use pdhg or condat_vu"
        )
    # three_composite: f smooth + g + h without a linear map
    if scheme == "dys":
        return typecheck_dys(L_smooth=L_SMOOTH_PRIMAL, step=tau)
    raise TypeCheckError(
        f"{scheme} is structurally inapplicable to a three-block composite "
        "without a linear map; use dys"
    )


def validate(g: Genome, problem_kind: str = "tv_denoise",
             lp_opnorm: Optional[float] = None) -> ValidationResult:
    """THE STAGE-0 GATE: typecheck the genome via atlas.typing_ rules for
    the canonical instance of `problem_kind`. Returns ok + ConstructionCert,
    or ok=False with the violated condition string (never clamps silently).

    lp_opnorm (lp_netflow only): operator-norm override so the gate is
    sound for instance families beyond degree-2 regular netflow; pass
    atlas.bench.lp_cell.gate_opnorm(instances) for the set being run."""
    if problem_kind not in PROBLEM_KINDS:
        raise ValueError(f"unknown problem_kind {problem_kind!r}")
    err = _structural_error(g)
    if err is not None:
        return ValidationResult(ok=False, error=err)
    try:
        if g.metric is not None and problem_kind not in METRIC_PROBLEM_KINDS:
            raise TypeCheckError(
                f"the metric gene is inapplicable to problem_kind "
                f"{problem_kind!r}: diagonal rescaling preserves the atom "
                "structure (hence every existing typing rule applies "
                "verbatim) only for the standard-form LP composite "
                f"(problem kinds {list(METRIC_PROBLEM_KINDS)}); the TV "
                "cell's quadratic-fidelity + scaled-L1 atoms are not "
                "closed under diagonal rescaling"
            )
        if g.scheme == "mix" and g.mix.kind == "avg":
            m = g.mix
            try:
                cert_a = _base_cert(m.a.scheme, m.a.tau, m.a.sigma,
                                    problem_kind, lp_opnorm)
            except TypeCheckError as e:
                raise TypeCheckError(f"mix.scheme_a invalid: {e}")
            try:
                cert_b = _base_cert(m.b.scheme, m.b.tau, m.b.sigma,
                                    problem_kind, lp_opnorm)
            except TypeCheckError as e:
                raise TypeCheckError(f"mix.scheme_b invalid: {e}")
            cert = typecheck_avg(cert_a, cert_b, m.lam)
        elif g.scheme == "mix" and g.mix.kind == "switch_k":
            m = g.mix
            try:
                tail_cert = _base_cert(
                    m.tail.scheme, m.tail.tau, m.tail.sigma,
                    problem_kind, lp_opnorm
                )
            except TypeCheckError as e:
                raise TypeCheckError(
                    f"switch_k tail must pass the FULL Stage-0 gate: {e}"
                )
            # aggressive branch: structural checks only (already done);
            # certified applicability of the SCHEME (not its params) to the
            # problem kind is still required to be runnable at all.
            if problem_kind == "tv_denoise" and m.aggressive.scheme == "dys":
                raise TypeCheckError(
                    "switch_k aggressive scheme dys cannot run on the TV cell"
                )
            if problem_kind == "lp_netflow" and m.aggressive.scheme not in (
                "pdhg",
                "condat_vu",
            ):
                raise TypeCheckError(
                    "switch_k aggressive scheme must be pdhg or condat_vu "
                    "on the LP cell"
                )
            agg = m.aggressive
            agg_desc = f"{agg.scheme}(tau={agg.tau:.6g}" + (
                f", sigma={agg.sigma:.6g})" if agg.sigma is not None else ")"
            )
            cert = typecheck_switch_k(
                tail_cert, int(m.K), agg_desc, max_k=MAX_SWITCH_K
            )
        elif g.scheme == "mix":  # switch
            m = g.mix
            try:
                fb_cert = _base_cert(
                    m.fallback.scheme, m.fallback.tau, m.fallback.sigma,
                    problem_kind, lp_opnorm
                )
            except TypeCheckError as e:
                raise TypeCheckError(
                    f"switch fallback must pass the Stage-0 gate: {e}"
                )
            # aggressive branch: structural checks only (already done);
            # certified applicability of the SCHEME (not its params) to the
            # problem kind is still required to be runnable at all.
            if problem_kind == "tv_denoise" and m.aggressive.scheme == "dys":
                raise TypeCheckError(
                    "switch aggressive scheme dys cannot run on the TV cell"
                )
            if problem_kind == "lp_netflow" and m.aggressive.scheme not in (
                "pdhg",
                "condat_vu",
            ):
                raise TypeCheckError(
                    "switch aggressive scheme must be pdhg or condat_vu on "
                    "the LP cell"
                )
            cert = typecheck_switch(fb_cert, m.window)
        else:
            eff_opnorm = lp_opnorm
            if g.metric is not None and g.metric.kind in (
                "pock_chambolle", "ruiz_pc"
            ):
                # Sound gate bound WITHOUT knowing the instance: the
                # Pock-Chambolle diagonals guarantee ||D1 A D2|| <= 1 (their
                # Lemma 2). "ruiz_pc" applies PC LAST, so the same universal
                # bound covers the composed operator. This is what makes the
                # STATIC gate instance-independent as well as the compiler's.
                # For kind "ruiz" alone no a-priori bound exists; the gate
                # stays on the UNSCALED constant (conservative screen) and
                # the step-region expansion is realized by the MANDATORY
                # compile-time typecheck against the recomputed bound.
                #
                # The constant MUST equal what the compiler feeds the same
                # rule, or validate() and solve_genome disagree about the
                # same genome. It was 1.02 while the compiler moved to
                # PC_SAFETY = 1 + 1e-9, which made the static gate STRICTER
                # (tau*sigma < 1/1.02^2 = 0.961) and rejected the frozen
                # B0 itself at tau*sigma = 0.980. Kept in sync by
                # tests/test_pc_certified.py::test_gate_and_compiler_agree.
                eff_opnorm = PC_GATE_OPNORM
            cert = _base_cert(g.scheme, g.params.tau, g.params.sigma,
                              problem_kind, eff_opnorm)
        if g.params.rho is not None:
            # relax-then-anchor: the anchor only needs nonexpansiveness,
            # so pass the flag through here too or validate() and the
            # compiler would disagree about the same genome.
            cert = typecheck_relaxation(
                cert, g.params.rho, with_anchor=bool(g.wrapper.halpern))
        if g.wrapper.halpern:
            cert = typecheck_halpern(cert)
        if g.wrapper.restart == "fixed_k":
            cert = typecheck_restart(cert, int(g.wrapper.restart_k))
        if g.wrapper.kind is not None:
            from atlas.wrappers.registry import gate as _outer_gate
            cert = _outer_gate(g.wrapper.kind, cert, g.scheme)
    except TypeCheckError as e:
        return ValidationResult(ok=False, error=str(e))
    return ValidationResult(ok=True, cert=cert)


# Compact schema + validity card for LLM prompts (kept colocated with the
# schema so prompt and code cannot drift).
SCHEMA_DOC = """Genome JSON schema:
{"scheme": "fbs"|"drs"|"pdhg"|"condat_vu"|"dys"|"mix",
 "params": {"tau": number>0 (base schemes only),
            "sigma": number>0 (required for pdhg/condat_vu, ignored otherwise),
            "alpha": null or number in [0,1] (reserved, not yet used),
            "rho": null or number in (0,2) (over-relaxation)},
 "wrapper": {"halpern": true|false,
             "restart": "none"|"fixed_k",
             "restart_k": null or integer>0 (needs restart="fixed_k" AND
                          halpern=true; resets the Halpern anchor every k)},
 "mix": null
   | {"kind":"avg", "lambda": number in (0,1),
      "scheme_a": {"scheme":..., "params":{"tau":..., "sigma":...}},
      "scheme_b": {"scheme":..., "params":{"tau":..., "sigma":...}}}
   | {"kind":"switch_k", "K": integer in [1,5000],
      "aggressive": {"scheme":..., "params":{"tau":..., "sigma":...}},
      "tail":      {"scheme":..., "params":{"tau":..., "sigma":...}}},
 "metric": null (default: no rescaling; LP cells only)
   | {"kind":"ruiz", "iters": integer in [0,30]}
   | {"kind":"pock_chambolle", "alpha": number in [0,2]},
 "backend": null (default: exact f64 baseline)
   | {"precision": "f64"|"f32_cert64"|"schedule",
      "K": integer in [1,10000] (certificate cadence, f64 gap every K iters),
      "theta": null, or for "schedule" a number in (0,1) and > tolerance
               (switch f32 -> f64 when the certified gap <= theta),
      "batch": false (reserved)}}
Validity (Stage-0 gate for the TV cell; ||L||^2 = 8, L_f = 1):
- fbs:       0 < tau < 0.25
- drs:       tau > 0
- pdhg:      tau*sigma*8 < 1
- condat_vu: tau/2 + tau*sigma*8 < 1
- dys:       REJECTED on the TV cell (needs a problem without L)
- rho and halpern are mutually exclusive; rho*averagedness must be < 1
  (averagedness: fbs 2/(4-8*tau), drs/pdhg 0.5, dys 2/(4-tau)).
- restart "fixed_k" requires halpern=true.
- mix avg: runs lambda*T_a + (1-lambda)*T_b; BOTH parents must be valid;
  parents must be the SAME scheme, or pdhg+condat_vu with IDENTICAL
  (tau, sigma); drs/dys parents additionally need IDENTICAL tau (their
  fixed-point encoding depends on the resolvent step; fbs stepsizes are
  free). Certified averagedness = lambda*a_a + (1-lambda)*a_b.
- mix switch_k: finite-prefix switch. The aggressive scheme runs the
  first K iterations (K in [1,5000]), then the tail runs forever from
  where the prefix stopped. The tail must be FULLY valid; the aggressive
  params may break the side conditions (only finite positive required).
  Branches same scheme or pdhg+condat_vu; no outer rho/halpern/restart;
  f64 backend only. A non-finite prefix auto-restarts the tail from the
  last finite state.
- metric (LP cells only; base pdhg/condat_vu genomes only): solves an
  equivalently RESCALED problem ("ruiz" = row/col equilibration of the
  constraint matrix, "pock_chambolle" = their diagonal rule); the
  certificate stays on the ORIGINAL problem's gap. Helps on badly
  scaled instances.
- backend: EXECUTION policy only (never changes the iteration map or its
  certificate): "f32_cert64" runs f32 iterates with the rigorous gap
  still evaluated at f64 every K iterations (fast on consumer GPUs, use
  for the 1e-4 tier); "schedule" starts f32 and switches to f64 at
  certified gap <= theta; bf16/f16 are refused (certificates cannot
  survive cancellation below f32); switch genomes take only "f64"."""
