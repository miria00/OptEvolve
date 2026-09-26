"""Level-1 construction certificates: side-condition checks per scheme.

Each `typecheck_*` validates the scheme's stepsize side condition against
the problem's operator constants and returns a ConstructionCert (with the
averagedness constant of the fixed-point map and the LSCOMO citation), or
raises TypeCheckError. Running typecheck is MANDATORY before any solve;
atlas.schemes enforces this by calling it inside solve().
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Optional


class TypeCheckError(Exception):
    """A scheme side condition is violated (Level-1 construction failure)."""


@dataclass(frozen=True)
class ConstructionCert:
    """Level-1 certificate: the scheme is a validly constructed averaged map.

    averagedness: alpha of the (possibly relaxed) fixed-point map; for the
    primal-dual schemes it is averagedness in the scheme's metric M.
    """

    scheme: str
    params: dict
    condition_str: str
    satisfied: bool
    averagedness: float
    citation: str

    def to_dict(self) -> dict:
        return asdict(self)


def _require(satisfied: bool, cert: ConstructionCert) -> ConstructionCert:
    if not satisfied:
        raise TypeCheckError(
            f"[{cert.scheme}] side condition violated: {cert.condition_str} "
            f"with params {cert.params} ({cert.citation})"
        )
    return cert


# ---------------------------------------------------------------------------
# Scheme rules
# ---------------------------------------------------------------------------


def typecheck_fbs(L_smooth: float, step: float) -> ConstructionCert:
    """FBS / proximal gradient [LSCOMO SS2.7].

    x+ = prox_{a g}(x - a grad f(x)); valid for a in (0, 2/L). The forward
    step I - a grad f is (aL/2)-averaged (grad f is 1/L-cocoercive,
    Baillon-Haddad [SS2.2.1]); composed with the (1/2)-averaged prox
    [SS2.5], T_FBS is alpha-averaged with alpha = 2/(4 - aL) < 1.
    """
    # L_smooth = 0 is the affine-f edge (grad f constant, 0-Lipschitz):
    # the forward step is a translation, any a > 0 is valid, alpha = 1/2.
    two_over_L = 2.0 / L_smooth if L_smooth > 0.0 else float("inf")
    ok = math.isfinite(step) and L_smooth >= 0.0 and 0.0 < step < two_over_L
    alpha = 2.0 / (4.0 - step * L_smooth) if ok else float("nan")
    cert = ConstructionCert(
        scheme="FBS",
        params={"step": step, "L_smooth": L_smooth},
        condition_str=f"0 < a < 2/L  (a={step:.6g}, 2/L={two_over_L:.6g})",
        satisfied=ok,
        averagedness=alpha,
        citation="LSCOMO SS2.7 (prox-grad), SS2.2.1 (Baillon-Haddad), SS2.5 (prox averagedness)",
    )
    return _require(ok, cert)


def typecheck_drs(step: float) -> ConstructionCert:
    """Douglas-Rachford splitting [LSCOMO SS2.7.1].

    z+ = z + J_aB(2 J_aA(z) - z) - J_aA(z). Valid for any REAL a > 0; the
    DRS map is (1/2)-averaged (it is the average of I and a composition of
    reflected resolvents, each nonexpansive). a must be finite: inf/nan is
    not a valid resolvent parameter (the other schemes reject non-finite
    steps arithmetically; DRS's open-ended condition must check explicitly).
    """
    ok = math.isfinite(step) and step > 0.0
    cert = ConstructionCert(
        scheme="DRS",
        params={"step": step},
        condition_str=f"0 < a < inf  (a={step:.6g})",
        satisfied=ok,
        averagedness=0.5,
        citation="LSCOMO SS2.7.1 (Douglas-Rachford)",
    )
    return _require(ok, cert)


def typecheck_pdhg(t: float, s: float, opnorm: float) -> ConstructionCert:
    """PDHG / Chambolle-Pock [LSCOMO SS3.3]: variable-metric PPM.

    x+ = prox_{t f}(x - t L^T u); u+ = prox_{s h*}(u + s L(2x+ - x)).
    Side condition t*s*||L||^2 < 1 makes the metric M positive definite;
    the iteration is a resolvent in M, hence (1/2)-averaged in M.
    """
    ok = (t > 0.0) and (s > 0.0) and (t * s * opnorm**2 < 1.0)
    cert = ConstructionCert(
        scheme="PDHG",
        params={"t": t, "s": s, "opnorm": opnorm},
        condition_str=(
            f"t>0, s>0, t*s*||L||^2 < 1  (t*s*||L||^2={t * s * opnorm**2:.6g})"
        ),
        satisfied=ok,
        averagedness=0.5,
        citation="LSCOMO SS3.3 (PDHG as variable-metric PPM)",
    )
    return _require(ok, cert)


def typecheck_condat_vu(
    t: float, s: float, opnorm: float, L_smooth: float
) -> ConstructionCert:
    """Condat-Vu [LSCOMO SS3.3, eq. 3.13]: f smooth + g prox + h(Lx).

    x+ = prox_{t g}(x - t grad f(x) - t L^T u);
    u+ = prox_{s h*}(u + s L(2x+ - x)).
    Side condition t*L_f/2 + t*s*||L||^2 < 1. The map is (1/delta)-averaged
    in the PDHG metric M with delta = 2 - (L_f/2)/(1/t - s*||L||^2) > 1.
    """
    lhs = t * L_smooth / 2.0 + t * s * opnorm**2
    ok = (t > 0.0) and (s > 0.0) and (lhs < 1.0)
    if ok:
        delta = 2.0 - (L_smooth / 2.0) / (1.0 / t - s * opnorm**2)
        alpha = 1.0 / delta
    else:
        alpha = float("nan")
    cert = ConstructionCert(
        scheme="CondatVu",
        params={"t": t, "s": s, "opnorm": opnorm, "L_smooth": L_smooth},
        condition_str=f"t*L_f/2 + t*s*||L||^2 < 1  (lhs={lhs:.6g})",
        satisfied=ok,
        averagedness=alpha,
        citation="LSCOMO SS3.3 eq. 3.13 (Condat-Vu)",
    )
    return _require(ok, cert)


def typecheck_dys(L_smooth: float, step: float) -> ConstructionCert:
    """Davis-Yin three-operator splitting [LSCOMO SS2.7 p.48; SS13.4.2 Thm 28].

    For f smooth (grad f is beta-cocoercive, beta = 1/L by Baillon-Haddad),
    g and h prox-friendly, the DYS map

        x_g = prox_{a g}(z);  x_h = prox_{a h}(2 x_g - z - a grad f(x_g));
        z+ = z + x_h - x_g

    is theta-averaged with theta = 2*beta/(4*beta - a) = 2/(4 - a*L), valid
    for a in (0, 2*beta) = (0, 2/L) [LSCOMO Thm 28; Davis & Yin 2017].
    Same form as FBS (DYS reduces to FBS when g = 0, to DRS when f = 0).
    """
    ok = (
        math.isfinite(step)
        and math.isfinite(L_smooth)
        and L_smooth > 0.0
        and 0.0 < step < 2.0 / L_smooth
    )
    alpha = 2.0 / (4.0 - step * L_smooth) if ok else float("nan")
    cert = ConstructionCert(
        scheme="DYS",
        params={"step": step, "L_smooth": L_smooth},
        condition_str=(
            f"0 < a < 2/L  (a={step:.6g}, 2/L={2.0 / L_smooth:.6g})"
            if L_smooth > 0
            else f"0 < a < 2/L  (a={step:.6g}, L={L_smooth:.6g})"
        ),
        satisfied=ok,
        averagedness=alpha,
        citation=(
            "LSCOMO SS2.7 (Davis-Yin splitting), SS13.4.2 Thm 28 "
            "(theta = 2*beta/(4*beta - a)); Davis & Yin 2017"
        ),
    )
    return _require(ok, cert)


# ---------------------------------------------------------------------------
# Wrapper rules (threaded through scheme params)
# ---------------------------------------------------------------------------


def typecheck_relaxation(base_cert: ConstructionCert, rho: float,
                        with_anchor: bool = False) -> ConstructionCert:
    """Over-relaxation x+ = (1-rho) x + rho T(x) on an alpha-averaged T.

    Combination rule [LSCOMO SS2.5]: T_rho is (rho*alpha)-averaged.

    TWO ADMISSIBLE REGIMES, and which one applies depends on what FOLLOWS
    the relaxation (2026-09-05):

    with_anchor=False (relaxation alone, iterated by plain KM). Convergence
    of the KM iteration needs T_rho AVERAGED, so rho in (0, 2) and
    rho*alpha < 1 STRICTLY. At rho*alpha = 1 the map is merely nonexpansive
    and KM can cycle: the classic example is the reflection 2T - I, which
    does not converge on its own.

    with_anchor=True (an anchored/Halpern step follows). Halpern converges
    for any NONEXPANSIVE map, so the requirement relaxes to rho in (0, 2]
    and rho*alpha <= 1. For a (1/2)-averaged base such as PDHG this admits
    the full reflection rho = 2, i.e. Halpern composed with 2T - I. Sun,
    Yuan, Zhang & Zhao (SIOPT 35(2) 2025, arXiv:2403.18618) prove that for
    That = (M + A)^-1 M with M >= 0, the relaxed map F_rho = (1-rho)I +
    rho*That is M-nonexpansive for every rho in (0, 2] with NO extra side
    condition, and that rho = 2 HALVES the Halpern rate constant. This is
    the r2HPDHG composition (Lu & Yang, arXiv:2407.16144), reported at
    1.27-1.33x over plain restarted Halpern.

    Previously rho and the anchor were mutually exclusive, which made that
    composition unrepresentable: the single highest-value structure in the
    GPU-LP lineage was outside the search space by construction.
    """
    alpha = base_cert.averagedness
    new_alpha = rho * alpha
    if with_anchor:
        ok = (0.0 < rho <= 2.0) and (new_alpha <= 1.0 + 1e-12)
        cond = (f"; 0<rho<=2 and rho*alpha<=1 (rho*alpha={new_alpha:.6g}), "
                "nonexpansive suffices because an anchored step follows")
        cite = ("; LSCOMO SS2.5 (averagedness combination); Sun, Yuan, Zhang "
                "& Zhao SIOPT 35(2) 2025 (relax-then-anchor, rho in (0,2] "
                "M-nonexpansive, rho=2 halves the anchor constant)")
    else:
        ok = (0.0 < rho < 2.0) and (new_alpha < 1.0)
        cond = f"; 0<rho<2 and rho*alpha<1 (rho*alpha={new_alpha:.6g})"
        cite = "; LSCOMO SS2.5 (averagedness combination)"
    cert = ConstructionCert(
        scheme=base_cert.scheme + "+relax",
        params={**base_cert.params, "rho": rho, "relax_with_anchor": with_anchor},
        condition_str=base_cert.condition_str + cond,
        satisfied=ok,
        averagedness=min(new_alpha, 1.0) if ok else float("nan"),
        citation=base_cert.citation + cite,
    )
    return _require(ok, cert)


def typecheck_avg(
    cert_a: ConstructionCert, cert_b: ConstructionCert, lam: float
) -> ConstructionCert:
    """Averaged blend  T = lam*T_a + (1-lam)*T_b,  lam in (0, 1).

    RULE (documented for the paper): if T_a is alpha_a-averaged and T_b is
    alpha_b-averaged IN THE SAME METRIC, their convex combination is
    (lam*alpha_a + (1-lam)*alpha_b)-averaged [LSCOMO SS2.5 (averaged
    operators; convex combination with a nonexpansive map is averaged);
    Bauschke & Combettes, Convex Analysis and Monotone Operator Theory,
    Prop. 4.42]. Fixed points: for averaged maps with
    Fix(T_a) n Fix(T_b) != 0, Fix(lam*T_a + (1-lam)*T_b) =
    Fix(T_a) n Fix(T_b) [Bauschke & Combettes Prop. 4.47], so the KM limit
    solves BOTH schemes' inclusions. The same-metric and common-fixed-point
    prerequisites are enforced structurally by the Stage-0 gate
    (atlas.genome: same scheme with a step-independent fixed-point
    encoding (fbs; stepsizes free), same scheme with IDENTICAL tau where
    the encoding depends on the resolvent step (drs/dys), or the
    pdhg/condat_vu pair with identical (tau, sigma) so the primal-dual
    metric M coincides).
    """
    ok = (
        cert_a.satisfied
        and cert_b.satisfied
        and math.isfinite(lam)
        and 0.0 < lam < 1.0
    )
    combined = (
        lam * cert_a.averagedness + (1.0 - lam) * cert_b.averagedness
        if ok
        else float("nan")
    )
    ok = ok and combined < 1.0
    cert = ConstructionCert(
        scheme=f"avg({cert_a.scheme},{cert_b.scheme})",
        params={
            "lambda": lam,
            "a": dict(cert_a.params),
            "b": dict(cert_b.params),
        },
        condition_str=(
            f"0<lambda<1; parents certified; combined alpha = "
            f"lambda*a_a + (1-lambda)*a_b = {combined:.6g} < 1"
        ),
        satisfied=ok,
        averagedness=combined,
        citation=(
            cert_a.citation
            + " | "
            + cert_b.citation
            + "; LSCOMO SS2.5 + Bauschke-Combettes Prop 4.42/4.47 "
            "(convex combination of averaged maps)"
        ),
    )
    return _require(ok, cert)


def typecheck_switch(
    fallback_cert: ConstructionCert, window: int
) -> ConstructionCert:
    """Safeguarded switching: aggressive step accepted ONLY when the
    certified relative duality gap decreased below the minimum certified
    gap over the last `window` iterations,
    otherwise the certified fallback step is taken from the last accepted
    state. The aggressive branch is EXEMPT from side conditions by design:
    every accepted aggressive step strictly decreased the rigorous merit
    (Level-3 runtime gap certificate), and the fallback KM iteration
    (Level-1 certified averaged map) supplies the convergence guarantee.
    This is an L1(fallback)+L3(monitor) certificate, the certified home of
    the PDLP/HPR-LP-style controller family.
    """
    ok = fallback_cert.satisfied and isinstance(window, int) and window >= 1
    cert = ConstructionCert(
        scheme=f"switch[{fallback_cert.scheme}]",
        params={**fallback_cert.params, "window": window},
        condition_str=(
            fallback_cert.condition_str
            + f"; safeguard: accept aggressive iff cert gap < min certified "
            f"gap over the last {window} iters, else certified fallback step"
        ),
        satisfied=ok,
        averagedness=fallback_cert.averagedness,
        citation=(
            fallback_cert.citation
            + "; safeguarded switching (monitor inherits fallback "
            "convergence; cf. PDLP primal-weight controller lineage)"
        ),
    )
    return _require(ok, cert)


def typecheck_switch_k(
    tail_cert: ConstructionCert, K: int, aggressive_desc: str,
    max_k: int = 5000,
) -> ConstructionCert:
    """Finite-prefix switching (switch_k): run the aggressive map for the
    first K iterations, then run the tail map forever, the tail simply
    starting from wherever the prefix left off.

    TYPING RULE (documented for the paper): the tail must pass the FULL
    Stage-0 gate, so its scheme is a certified averaged map whose KM
    iteration is globally convergent from ANY starting point under its
    certified side conditions; therefore any finite prefix cannot affect
    asymptotic convergence: whatever (finite) state the prefix hands over
    is just another starting point for the certified tail. The aggressive
    block must be STRUCTURALLY well-formed (valid scheme name, finite
    positive parameters) but is EXEMPT from convergence side conditions;
    its certificate line reads "finite prefix, exempt". The compiled
    runtime additionally guards the handoff: if the prefix state is
    non-finite (checked periodically and at k = K), the tail restarts from
    the last recorded finite state (the initial point if none), so the
    exemption can never hand the certified tail a non-finite start.

    Constraints enforced here and by the Stage-0 gate: 1 <= K <= max_k;
    aggressive and tail share the same state space (same scheme, or both
    in the pdhg/condat_vu saddle family); f64 backend only.
    """
    ok = (
        tail_cert.satisfied
        and isinstance(K, int)
        and not isinstance(K, bool)
        and 1 <= K <= max_k
    )
    cert = ConstructionCert(
        scheme=f"switch_k[{tail_cert.scheme}]",
        params={**tail_cert.params, "K": K, "aggressive": aggressive_desc},
        condition_str=(
            f"aggressive prefix {aggressive_desc} for k < K={K}: finite "
            "prefix, exempt (structurally well-formed only; non-finite "
            "handoff restarts the tail from the last finite state); tail "
            f"certified: {tail_cert.condition_str}"
        ),
        satisfied=ok,
        averagedness=tail_cert.averagedness,
        citation=(
            tail_cert.citation
            + "; finite-prefix switching (tail globally convergent from any "
            "start under its certified conditions, so a finite prefix "
            "cannot affect asymptotic convergence)"
        ),
    )
    return _require(ok, cert)


def typecheck_metric_rescale(
    base_cert: ConstructionCert, kind: str, param_desc: str,
    opnorm_rescaled: float, bound_provenance: str = "power iteration x1.02",
) -> ConstructionCert:
    """Metric gene as PROBLEM RESCALING (zero new convergence theory).

    The compiled path replaces the problem by the diagonally rescaled
    equivalent (L' = D1 L D2 with correspondingly transformed data; see
    atlas.schemes.rescale for the derivation) and re-runs the EXISTING
    side-condition typecheck of the base scheme against the recomputed
    power-iteration bound of ||L'|| (x1.02 safety). All existing rules
    apply verbatim to the rescaled problem, so `base_cert` here IS the
    scheme's certificate on the rescaled problem; this rule only records
    the metric gene and the recomputed bound on the certificate line.
    Termination remains the ORIGINAL problem's f64 duality gap (the
    runtime metric maps iterates back before evaluating the certificate),
    so the certificate stays a statement about the original problem.
    """
    ok = (
        base_cert.satisfied
        and math.isfinite(opnorm_rescaled)
        and opnorm_rescaled >= 0.0
    )
    cert = ConstructionCert(
        scheme=base_cert.scheme + f"+metric[{kind}]",
        params={
            **base_cert.params,
            "metric_kind": kind,
            "metric_param": param_desc,
            "opnorm_rescaled": opnorm_rescaled,
        },
        condition_str=(
            base_cert.condition_str
            + f"; metric gene {kind}({param_desc}): problem rescaled, "
            f"||L'|| bound {opnorm_rescaled:.6g} ({bound_provenance}) "
            "fed to the same side condition; certificate evaluated on the "
            "ORIGINAL problem's gap"
        ),
        satisfied=ok,
        averagedness=base_cert.averagedness,
        citation=(
            base_cert.citation
            + "; diagonal rescaling (Ruiz 2001 equilibration"
            + (
                "; Pock & Chambolle 2011 diagonal preconditioning rule"
                if kind in ("pock_chambolle", "ruiz_pc")
                else ""
            )
            + "), existing conditions applied verbatim to the rescaled problem"
        ),
    )
    return _require(ok, cert)


def typecheck_restart(base_cert: ConstructionCert, restart_k: int) -> ConstructionCert:
    """Fixed-k restart of the Halpern anchor: every k iterations the anchor
    x0 is reset to the current iterate and the anchor weight sequence
    restarts. Each segment is a fresh Halpern iteration on the SAME
    nonexpansive map from a new anchor, so the construction requirement is
    unchanged (T nonexpansive, i.e. averagedness <= 1) and the map's
    averagedness constant is untouched: restart is a schedule on the
    anchor coefficients, not a change of the operator [LSCOMO SS12.2].
    Requires a Halpern cert (restart has no state to reset otherwise).
    """
    ok = (
        base_cert.satisfied
        and "halpern" in base_cert.scheme
        and isinstance(restart_k, int)
        and restart_k >= 1
    )
    cert = ConstructionCert(
        scheme=base_cert.scheme + f"+restart{restart_k}",
        params={**base_cert.params, "restart_k": restart_k},
        condition_str=(
            base_cert.condition_str
            + f"; fixed-k anchor restart (k={restart_k}) preserves the map"
        ),
        satisfied=ok,
        averagedness=base_cert.averagedness,
        citation=base_cert.citation + "; anchor restart (schedule only)",
    )
    return _require(ok, cert)


def typecheck_halpern(base_cert: ConstructionCert) -> ConstructionCert:
    """Halpern/OHM anchor [LSCOMO SS12.2]: x_{k+1} = x0/(k+2) + (k+1)/(k+2) T(x_k).

    Requires T nonexpansive; any alpha-averaged map with alpha <= 1 is
    nonexpansive (in its metric), so we require averagedness <= 1.
    """
    ok = base_cert.satisfied and base_cert.averagedness <= 1.0
    cert = ConstructionCert(
        scheme=base_cert.scheme + "+halpern",
        params=dict(base_cert.params),
        condition_str=base_cert.condition_str + "; T nonexpansive (alpha<=1)",
        satisfied=ok,
        averagedness=base_cert.averagedness,
        citation=base_cert.citation + "; LSCOMO SS12.2 (Halpern/OHM)",
    )
    return _require(ok, cert)


def typecheck_haugazeau(base_cert: ConstructionCert, scheme: str) -> ConstructionCert:
    """Haugazeau best-approximation outer step: x_{k+1} = Q(x_0, x_k, F x_k).

    Q(a, b, c) is the projection of a onto H(a, b) ∩ H(b, c), with
    H(y, z) = {w : <w - z, y - z> <= 0}. The step needs a firmly
    (quasi-)nonexpansive map. An alpha-averaged T with alpha <= 1 gives one
    with the same fixed points, F = I + (T - I)/(2 alpha), and F is firmly
    nonexpansive in the SAME metric in which T's averagedness is certified,
    so Q must be evaluated in that metric: M = [[I/t, -L*], [-L, I/s]] for
    pdhg and condat_vu (BuiltScheme.gram). Only those two schemes supply and
    admission-test their metric in this build. Guarantee: strong convergence
    to the projection of x_0 onto Fix T in the M-norm. The returned
    certificate carries no averagedness (the outer map depends on x_0), so
    no wrapper may be stacked on it.
    """
    a = base_cert.averagedness
    # genome names ("condat_vu") and certificate display names ("CondatVu") both accepted
    admitted = scheme.lower().replace("_", "").replace("-", "") in ("pdhg", "condatvu")
    ok = (base_cert.satisfied and admitted and a is not None
          and math.isfinite(a) and 0.0 < a <= 1.0)
    cert = ConstructionCert(
        scheme=base_cert.scheme + "+haugazeau",
        params={**base_cert.params, "haugazeau_alpha": a},
        condition_str=(
            base_cert.condition_str
            + f"; Haugazeau outer step on F = I + (T - I)/(2 alpha), alpha = {a:.6g} <= 1"
            + " (F firmly nonexpansive in the averagedness metric, Fix F = Fix T),"
            + " Q evaluated in that metric"
            + ("" if admitted else f"; REJECTED: scheme {scheme!r} supplies no admission-tested metric")
        ),
        satisfied=ok,
        averagedness=float("nan"),
        citation=(base_cert.citation
                  + "; Haugazeau 1968 (best-approximation outer step); Bauschke & Combettes,"
                  + " Convex Analysis and Monotone Operator Theory in Hilbert Spaces,"
                  + " best-approximation chapter (closed form of Q)"),
    )
    return _require(ok, cert)


def typecheck_pdhg_entropic(t: float, s: float, opnorm_1to2: float, mass: float) -> ConstructionCert:
    """PDHG with an entropic (Bregman) primal step on a mass simplex.

    x+ = argmin <c + L*u, x> + (1/t) D(x, x_k) over {x >= 0, sum x = mass},
    with kernel D the Bregman distance of mass * sum x log x, which gives the
    multiplicative step x+ proportional to x_k exp(-(t/mass)(c + L*u)); the
    dual step is Euclidean, u+ = prox_{s h*}(u + s L(2x+ - x)).

    By Pinsker's inequality the kernel is 1-strongly convex in the l1 norm on
    that simplex, so the Bregman primal-dual step condition (Chambolle & Pock
    2016) is t*s*||L||_{1->2}^2 < 1, with ||L||_{1->2} the largest column
    2-norm. On the unbounded orthant negative entropy is not strongly convex
    in any norm and no such condition exists, so the builder refuses any
    composite that does not declare the simplex. The iteration is not an
    averaged map in a Hilbert metric: averagedness is NaN, so no relaxation,
    anchor or outer wrapper can be stacked on it.
    """
    ok = (t > 0.0 and s > 0.0 and mass > 0.0 and math.isfinite(mass)
          and t * s * opnorm_1to2 ** 2 < 1.0)
    cert = ConstructionCert(
        scheme="PDHG-entropic",
        params={"t": t, "s": s, "opnorm_1to2": opnorm_1to2, "mass": mass},
        condition_str=(
            f"primal block on the simplex {{x >= 0, sum x = {mass:g}}} (implied by the constraints);"
            f" kernel mass*sum x log x 1-strongly convex in l1 there (Pinsker);"
            f" t>0, s>0, t*s*||L||_{{1->2}}^2 < 1 (= {t * s * opnorm_1to2 ** 2:.6g})"
        ),
        satisfied=ok,
        averagedness=float("nan"),
        citation=("Chambolle & Pock 2016, On the ergodic convergence rates of a first-order"
                  " primal-dual algorithm (Bregman proximal steps); Bregman 1967; Pinsker's inequality"),
    )
    return _require(ok, cert)


def typecheck_metric_schedule(base_cert: ConstructionCert, scheme: str, eta_budget: float,
                              omega_range: float, fill: float,
                              max_updates=None) -> ConstructionCert:
    """Certified adaptive step ratio for pdhg / condat_vu (variable metric).

    The saddle iteration is a proximal step in M_n = [[I/t, -L*], [-L, I/s]].
    Changing the ratio omega = sqrt(s/t) changes the metric. With
    theta = sqrt(t s) ||L|| < 1, M_n >= (1 - theta) D_n (D_n its block
    diagonal), so moving to (t', s') satisfies M_{n+1} <= (1 + eta_n) M_n with
        eta_n = max(0, t/t' - 1, s/s' - 1) / (1 - theta)
    (derived here from that bound; checked numerically in tests). The variable
    metric forward-backward theorem of Combettes and Vu (2013, 2014) keeps the
    base scheme convergent when sum eta_n < inf and the metrics are uniformly
    bounded. The controller enforces sum eta_n <= eta_budget (and optionally a
    finite number of updates), keeps omega within a factor omega_range of its
    start (uniform bounds), and places every (t, s) on the admissible boundary
    t L_f / 2 + t s ||L||^2 = fill < 1, so each metric also passes the base gate.
    """
    admitted = scheme.lower().replace("_", "") in ("pdhg", "condatvu")
    ok = (base_cert.satisfied and admitted and math.isfinite(eta_budget) and eta_budget >= 0.0
          and math.isfinite(omega_range) and omega_range >= 1.0 and 0.0 < fill < 1.0
          and (max_updates is None or (isinstance(max_updates, int) and max_updates >= 0)))
    cert = ConstructionCert(
        scheme=base_cert.scheme + "+adaptive_ratio",
        params={**base_cert.params, "eta_budget": eta_budget, "omega_range": omega_range,
                "fill": fill, "max_updates": max_updates},
        condition_str=(base_cert.condition_str
                       + f"; adaptive ratio: every (t, s) on t L_f/2 + t s ||L||^2 = {fill:g}; "
                       f"sum eta_n <= {eta_budget:g} with eta_n = max(0, t/t'-1, s/s'-1)/(1-theta); "
                       f"omega within x{omega_range:g} of its start"
                       + ("" if admitted else f"; REJECTED: scheme {scheme!r} has no saddle metric")),
        satisfied=ok,
        averagedness=float("nan"),
        citation=(base_cert.citation + "; Combettes & Vu 2013 (variable metric quasi-Fejer monotonicity), "
                  "2014 (variable metric forward-backward); PDLP primal-weight update (Applegate et al. 2021)"),
    )
    return _require(ok, cert)


def typecheck_diag_metric_schedule(base_cert, scheme: str, eta_budget: float, diag_range: float,
                                   fill: float, max_updates=None):
    """Certified adaptive DIAGONAL metric for pdhg / condat_vu (per-coordinate variable metric).

    The saddle iteration is a proximal step in M_n = [[T^-1, -L*], [-L, Sigma^-1]] with diagonal steps T
    (primal) and Sigma (dual). M_n is positive definite when theta = ||Sigma^{1/2} L T^{1/2}|| < 1, and then
    M_n >= (1 - theta) D_n for its block diagonal D_n. Moving to (T', Sigma') gives

        M_{n+1} <= (1 + eta_n) M_n,   eta_n = rho_n / (1 - theta),
        rho_n  = max(0, max_i T_i/T'_i - 1, max_j Sigma_j/Sigma'_j - 1),

    derived as M' = D' - Off <= (1 + rho) D - Off = (1 + rho) M + rho Off and +-Off <= theta D <= theta M /
    (1 - theta). This is the scalar rule of typecheck_metric_schedule with the ratio replaced by a maximum over
    coordinates, and it reduces to it exactly when the diagonals move uniformly (checked in
    tests/test_diag_metric.py, which also verifies the matrix inequality numerically).

    Convergence rests on the same theorem as the scalar case: the variable metric quasi-Fejer / forward-backward
    results of Combettes and Vu (2013, 2014) need sum_n eta_n < inf and uniformly bounded metrics. The controller
    enforces sum eta_n <= eta_budget, keeps every entry of T and Sigma within a factor diag_range of its start
    (the uniform bounds), optionally caps the number of updates, and re-checks the base gate on the rescaled
    operator after each proposed update, rejecting any update that would violate it.
    """
    admitted = scheme.lower().replace("_", "") in ("pdhg", "condatvu")
    ok = (base_cert.satisfied and admitted and math.isfinite(eta_budget) and eta_budget >= 0.0
          and math.isfinite(diag_range) and diag_range >= 1.0 and 0.0 < fill < 1.0
          and (max_updates is None or (isinstance(max_updates, int) and max_updates >= 0)))
    cert = ConstructionCert(
        scheme=base_cert.scheme + "+adaptive_diag_metric",
        params={**base_cert.params, "eta_budget": eta_budget, "diag_range": diag_range,
                "fill": fill, "max_updates": max_updates},
        condition_str=(base_cert.condition_str
                       + f"; adaptive diagonal metric: theta = ||Sigma^1/2 L T^1/2|| <= {fill:g} after every "
                       f"update; sum eta_n <= {eta_budget:g} with eta_n = max(0, max_i T_i/T'_i - 1, "
                       f"max_j S_j/S'_j - 1)/(1 - theta); every entry within x{diag_range:g} of its start"
                       + ("" if admitted else f"; REJECTED: scheme {scheme!r} has no saddle metric")),
        satisfied=ok,
        averagedness=base_cert.averagedness,
        citation=("Combettes & Vu 2013/2014 (variable metric quasi-Fejer, forward-backward); "
                  "Condat 2013 / Vu 2013 (base scheme); Pock & Chambolle 2011 (static diagonal preconditioning)"))
    if not ok:
        raise TypeCheckError(f"adaptive diagonal metric REJECTED: {cert.condition_str}")
    return cert


def diag_change_cost(T, S, Tp, Sp, theta: float) -> float:
    """eta_n for a diagonal metric change; theta is measured on the OLD metric, where the bound is derived."""
    import numpy as _np
    if not theta < 1.0:
        raise TypeCheckError(f"saddle metric not positive definite: theta = {theta:.6g}")
    rho = max(0.0, float(_np.max(_np.asarray(T) / _np.asarray(Tp))) - 1.0,
              float(_np.max(_np.asarray(S) / _np.asarray(Sp))) - 1.0)
    return rho / (1.0 - theta)


def typecheck_condat_vu_metric(s: float, lamP: float, nAT: float, g_is_zero: bool) -> ConstructionCert:
    """Condat-Vu with a matrix primal step T (quasi-Newton / variable metric).

    x+ = x - T(grad f(x) + A^T u),  u+ = prox_{s h*}(u + s A(2x+ - x)),  g = 0.
    It is forward-backward in M = [[T^{-1}, -A^T], [-A, I/s]]; with
    theta = s ||A T A^T|| < 1, T^{-1} - s A^T A >= (1 - theta) T^{-1}, so the
    forward operator is cocoercive enough when
        lambda_max(T P)/2 + s ||A T A^T|| < 1,
    which reduces to Condat's t L_f/2 + t s ||A||^2 < 1 for T = t I. This
    matrix form is a sufficient condition derived here and checked by the
    numerical admission test; the two norms come from power iteration with
    safety factors. g = 0 is required: a prox in a non-diagonal metric is not
    available, and the OSQP form of a QP needs none.
    """
    theta = s * nAT
    ok = g_is_zero and s > 0.0 and lamP >= 0.0 and nAT >= 0.0 and lamP / 2.0 + theta < 1.0
    delta = 2.0 - (lamP / 2.0) / (1.0 - theta) if theta < 1.0 else float("nan")
    cert = ConstructionCert(
        scheme="CondatVu-metric",
        params={"s": s, "lamP": lamP, "nAT": nAT, "theta": theta},
        condition_str=(f"g = 0 ({g_is_zero}); lambda_max(T P)/2 + s ||A T A^T|| < 1 "
                       f"(= {lamP / 2.0 + theta:.6g}), power-iteration estimates x1.05 / x1.02"),
        satisfied=ok,
        averagedness=(1.0 / delta) if (ok and delta > 1.0) else float("nan"),
        citation=("Condat 2013 / Vu 2013 (Condat-Vu); Combettes & Vu 2014 (variable metric forward-backward); "
                  "Bonnans, Gilbert, Lemarechal & Sagastizabal 1995 (quasi-Newton proximal metric)"),
    )
    return _require(ok, cert)


def typecheck_fista(L_smooth: float, step: float, restart: bool) -> ConstructionCert:
    """FISTA (Beck & Teboulle 2009) on f + g, f with L-Lipschitz gradient, g prox-friendly:
        x+ = prox_{a g}(y - a grad f(y)),  t+ = (1 + sqrt(1 + 4 t^2))/2,  y+ = x+ + ((t-1)/t+)(x+ - x),
    with the O'Donoghue-Candes (2015) gradient restart (t <- 1, y <- x+ when <y - x+, x+ - x> > 0).
    Side condition 0 < a <= 1/L. Guarantee: F(x_k) - F* = O(1/k^2) without restarts; the restart
    keeps that bound between restarts. Momentum makes the map non-averaged: no wrapper may be stacked.
    """
    ok = step > 0.0 and L_smooth >= 0.0 and step * L_smooth <= 1.0
    cert = ConstructionCert(
        scheme="FISTA" + ("+restart" if restart else ""),
        params={"step": step, "L_smooth": L_smooth, "restart": restart},
        condition_str=f"0 < a <= 1/L (a L = {step * L_smooth:.6g})",
        satisfied=ok,
        averagedness=float("nan"),
        citation="Beck & Teboulle 2009 (FISTA); O'Donoghue & Candes 2015 (adaptive restart)",
    )
    return _require(ok, cert)


def typecheck_adaptive_restart(base_cert: ConstructionCert, beta: float) -> ConstructionCert:
    """Adaptive restart of the Halpern anchor: restart when the fixed-point residual
    ||x - T x|| falls to beta times its value at the last restart (the PDLP restart
    idea, Applegate et al. 2021, on the fixed-point residual). Each segment is a Halpern
    iteration on the same nonexpansive map, whose residual obeys
    ||x_k - T x_k|| <= 2 ||x_0 - x*|| / (k + 1), so every segment triggers its restart in
    finitely many steps and the residual shrinks at least geometrically across restarts.
    Requires a Halpern cert and beta in (0, 1)."""
    ok = base_cert.satisfied and "halpern" in base_cert.scheme and 0.0 < beta < 1.0
    cert = ConstructionCert(
        scheme=base_cert.scheme + f"+adaptive_restart{beta:g}",
        params={**base_cert.params, "restart_beta": beta},
        condition_str=base_cert.condition_str + f"; anchor restarted when the residual falls by x{beta:g}",
        satisfied=ok, averagedness=base_cert.averagedness,
        citation=base_cert.citation + "; adaptive restart (Applegate et al. 2021; O'Donoghue & Candes 2015)")
    return _require(ok, cert)


def typecheck_anderson(base_cert: ConstructionCert, memory: int, D: float = 1e6, eps: float = 1e-6) -> ConstructionCert:
    """Type-II Anderson acceleration (Anderson 1965) of an averaged map with the
    Zhang-O'Donoghue-Boyd safeguard (SIAM J. Optim. 2020): the extrapolated point is
    accepted only if its fixed-point residual is at most D ||g_0|| (n_aa + 1)^-(1+eps),
    otherwise the plain averaged step is taken. Accepted residuals are summable and the
    rejected steps are ordinary KM steps, which keeps global convergence. Requires an
    averaged base map (alpha < 1) and memory >= 1; not composed with an anchor here."""
    a = base_cert.averagedness
    ok = (base_cert.satisfied and a is not None and math.isfinite(a) and a < 1.0
          and "halpern" not in base_cert.scheme and isinstance(memory, int) and memory >= 1)
    cert = ConstructionCert(
        scheme=base_cert.scheme + f"+anderson{memory}",
        params={**base_cert.params, "aa_memory": memory, "aa_D": D, "aa_eps": eps},
        condition_str=base_cert.condition_str + f"; safeguarded Anderson(m={memory}), accept iff ||g|| <= D||g0||(n+1)^-(1+eps)",
        satisfied=ok, averagedness=float("nan"),
        citation=base_cert.citation + "; Anderson 1965; Zhang, O'Donoghue & Boyd 2020 (safeguarded type-I/II AA)")
    return _require(ok, cert)
