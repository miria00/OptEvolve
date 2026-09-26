"""Generate OptAtlas_tex_2026/appendix-instantiation.tex from the code.

The theorem-to-compiler instantiation appendix: one entry per implemented
base scheme with {problem decomposition, state space, compiled update
(display equation), metric H_G, decoder pi_G, certificate condition}, plus
the wrapper genes, and the statement that the equivalence is machine
checked by tests/test_compiler_equivalence.py.

Anti-drift design (the table is generated FROM the code wherever the code
holds the ground truth, and every hand-typeset equation is anchored):
  1. The scheme list comes from atlas.schemes.base_schemes._BUILDERS; a
     scheme without a table entry aborts generation.
  2. Certificate conditions and citations are extracted from the
     ConstructionCert objects returned by atlas.typing_ (the symbolic part
     of condition_str, converted to LaTeX mechanically).
  3. Every display equation carries SOURCE ANCHORS: literal code fragments
     that must appear in the corresponding builder (or the km loop) source.
     If the update is edited, the anchor vanishes and generation fails
     instead of emitting a stale equation.
  4. The measured worst iterate deviations are re-measured at generation
     time by running layer A of tests/test_compiler_equivalence.py (50
     iterations per case, f64).

Style laws (enforced by assertion): no em-dash (neither --- nor the
character), no \\emph, no \\textit.

Run from the v2 root:
    JAX_PLATFORMS=cpu python scripts/gen_instantiation_table.py
"""
from __future__ import annotations

import inspect
import os
import re
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")

V2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, V2)
sys.path.insert(0, os.path.join(V2, "tests"))

OUT = os.path.join(V2, "OptAtlas_tex_2026", "appendix-instantiation.tex")


# ---------------------------------------------------------------------------
# 1. Certificate conditions + citations, from the typing rules themselves
# ---------------------------------------------------------------------------


def _symbolic_condition(cert) -> str:
    """The symbolic part of a ConstructionCert.condition_str (strip the
    numeric instantiation in the trailing parenthetical)."""
    s = cert.condition_str
    return s.split("  (")[0].strip()


def _cond_to_latex(cond: str) -> str:
    """Mechanical ASCII -> LaTeX for the condition strings of
    atlas.typing_.rules (kept intentionally dumb so an unrecognized token
    survives visibly rather than being silently dropped)."""
    parts = [p.strip() for p in cond.split(",")]
    out = []
    for p in parts:
        p = p.replace("||L||^2", r"\ell^2")
        p = p.replace("2/L", r"2/L_f")
        p = p.replace("inf", r"\infty")
        p = p.replace("*", r"\cdot ")
        p = p.replace("<=", r"\leq ")
        # code parameter names -> paper notation (word-boundary aware so
        # L_f, \ell, \cdot, \infty survive)
        p = re.sub(r"(?<![A-Za-z_\\])a(?![A-Za-z_])", r"\\alpha", p)
        p = re.sub(r"(?<![A-Za-z_\\])t(?![A-Za-z_])", r"\\tau", p)
        p = re.sub(r"(?<![A-Za-z_\\])s(?![A-Za-z_])", r"\\sigma", p)
        out.append("$" + p + "$")
    return ", ".join(out)


def collect_certs():
    from atlas.typing_ import (
        typecheck_condat_vu,
        typecheck_drs,
        typecheck_dys,
        typecheck_fbs,
        typecheck_pdhg,
    )

    return {
        "fbs": typecheck_fbs(L_smooth=1.0, step=0.5),
        "drs": typecheck_drs(step=0.5),
        "pdhg": typecheck_pdhg(t=0.1, s=0.1, opnorm=1.0),
        "condat_vu": typecheck_condat_vu(t=0.1, s=0.1, opnorm=1.0, L_smooth=1.0),
        "dys": typecheck_dys(L_smooth=1.0, step=0.5),
    }


# ---------------------------------------------------------------------------
# 2. Source anchors: the emitted updates, pinned to the builder source
# ---------------------------------------------------------------------------

ANCHORS = {
    "fbs": [
        'L.forward(L.adjoint(u)) / mu - dyn["Dc"]',
        '_prox_conj(dyn["h"], u - dyn["a"] * grad, dyn["a"])',
        'dyn["p"].prox(x - dyn["a"] * dyn["f"].grad(x), dyn["a"])',
    ],
    "drs": [
        'u = _prox_conj(dyn["h"], z, a)',
        "rhs = v + a * dyn[\"Dc\"]",
        "return z + w - u",
        'xh = dyn["h"].prox(2.0 * xg - z, a)',
        "return z + xh - xg",
    ],
    "pdhg": [
        'dyn["phi"].prox(x - t * L.adjoint(u), t)',
        '_prox_conj(dyn["h"], u + s * L.forward(2.0 * x_new - x), s)',
    ],
    "condat_vu": [
        'dyn["g"].prox(x - t * dyn["f"].grad(x) - t * L.adjoint(u), t)',
        '_prox_conj(dyn["h"], u + s * L.forward(2.0 * x_new - x), s)',
    ],
    "dys": [
        'xg = dyn["g"].prox(z, a)',
        'xh = dyn["h"].prox(2.0 * xg - z - a * dyn["f"].grad(xg), a)',
        "return z + xh - xg",
    ],
}

KM_ANCHORS = [
    "(1.0 - rho) * s + rho * t",  # relaxation applied to the state pytree
    "lam_j = 1.0 / (j.astype(jnp.float64) + 2.0)",  # Halpern coefficient
    'do_restart = j_next >= dyn_["restart_k"]',  # fixed-k anchor restart
    "lambda a, t: lam_j * a + (1.0 - lam_j) * t",  # anchor combination on T
]


def check_anchors():
    from atlas.schemes import base_schemes
    from atlas.wrappers import km

    builders = {
        "fbs": base_schemes._build_fbs,
        "drs": base_schemes._build_drs,
        "pdhg": base_schemes._build_pdhg,
        "condat_vu": base_schemes._build_condat_vu,
        "dys": base_schemes._build_dys,
    }
    schemes = set(base_schemes._BUILDERS)
    if schemes != set(builders):
        raise SystemExit(
            f"scheme registry drifted: _BUILDERS={sorted(schemes)} vs table "
            f"entries {sorted(builders)}; add the missing entry before "
            "regenerating the appendix"
        )
    for name, fn in builders.items():
        src = inspect.getsource(fn)
        for anchor in ANCHORS[name]:
            if anchor not in src:
                raise SystemExit(
                    f"source anchor missing in _build_{name}: {anchor!r}\n"
                    "the compiled update changed; update the display "
                    "equation AND the anchor together"
                )
    km_src = inspect.getsource(km)
    for anchor in KM_ANCHORS:
        if anchor not in km_src:
            raise SystemExit(
                f"source anchor missing in atlas/wrappers/km.py: {anchor!r}"
            )


# ---------------------------------------------------------------------------
# 3. Measured deviations: rerun layer A of the equivalence battery
# ---------------------------------------------------------------------------


def measure_deviations():
    import jax
    import jax.numpy as jnp

    from test_compiler_equivalence import (  # tests/ on sys.path
        CASES,
        N_ITERS,
        rel_err,
        run_reference,
        to_flat,
    )
    from atlas.schemes.base_schemes import build_scheme

    worst_by_scheme: dict = {}
    n_cases_by_scheme: dict = {}
    for cid in sorted(CASES):
        problem, scheme, params, ref = CASES[cid]()
        built = build_scheme(problem, scheme, params, check=True)
        dyn = jax.tree_util.tree_map(jnp.asarray, built.dyn)
        step = jax.jit(built.T)
        state = jax.tree_util.tree_map(jnp.asarray, built.state0)
        traj = run_reference(ref, N_ITERS)
        worst = 0.0
        for k in range(N_ITERS):
            state = step(state, jnp.asarray(k), dyn)
            worst = max(worst, rel_err(to_flat(state), to_flat(traj[k])))
        worst_by_scheme[scheme] = max(worst_by_scheme.get(scheme, 0.0), worst)
        n_cases_by_scheme[scheme] = n_cases_by_scheme.get(scheme, 0) + 1
    assert all(v == 3 for v in n_cases_by_scheme.values()), n_cases_by_scheme
    return worst_by_scheme


def _sci(x: float) -> str:
    if x == 0.0:
        return "0"
    m, e = f"{x:.1e}".split("e")
    return rf"{m}\!\times\!10^{{{int(e)}}}"


# ---------------------------------------------------------------------------
# 4. The document
# ---------------------------------------------------------------------------


def build_tex(certs, worst) -> str:
    cond = {k: _cond_to_latex(_symbolic_condition(c)) for k, c in certs.items()}
    dev = {k: _sci(v) for k, v in worst.items()}
    dev_max = _sci(max(worst.values()))

    L = []
    L.append(r"""% AUTO-GENERATED by scripts/gen_instantiation_table.py; do not edit by
% hand. Regenerate from the v2 root with
%   JAX_PLATFORMS=cpu python scripts/gen_instantiation_table.py
% Certificate conditions and citations are extracted from atlas.typing_;
% every display equation is anchored to the builder source in
% atlas/schemes/base_schemes.py (generation fails if the code drifts); the
% deviation numbers are re-measured at generation time by layer A of
% tests/test_compiler_equivalence.py.
% Style rules: never use the em-dash; no italics commands.
\section{Instantiation of the certificates in the compiler}
\label{app:instantiation}

This appendix connects Lemma~\ref{lem:base-certificates} to the emitted
code. For each base scheme of Table~\ref{tab:certificates} it records how
the compiler decomposes problem~\eqref{eq:composite}, the compiled state
space $\mathcal Z_G$ and metric $H_G$, the exact update the compiler
emits, the decoder $\pi_G$, the fixed-point correspondence, and the
certificate condition the gate checks (extracted verbatim, up to LaTeX
conversion, from the typing rules in the implementation). The compiler
dispatches on the shape of the composite: FBS and DRS instantiate the
three-term objective either on a primal two-block form or, when $h$ is
composed with $L$, on the dual problem
\begin{equation}
 \min_{u}\; f^*(-L^{\top}u)+h^*(u),
 \label{eq:inst-dual}
\end{equation}
which requires $g=0$ and $f$ to be $\mu$-strongly convex, so that
$F=f^*\circ(-L^{\top})$ is $(\ell^2/\mu)$-smooth and $\nabla f^*$ is
single-valued; the implemented strongly convex atoms have affine
$\nabla f^*$, which the dual updates below use. PDHG and Condat--V\~u
instantiate the saddle form directly, and Davis--Yin requires the
three-block form without $L$. A structurally inapplicable (problem,
scheme) pair is rejected by the gate as a certified construction failure.
Table~\ref{tab:instantiation} summarizes; the updates and correspondences
follow, one block per scheme.

\begin{table}[h]
\centering
\small
\setlength{\tabcolsep}{3pt}
\begin{tabular}{llllll}
\toprule
scheme & decomposition & state $\mathcal Z_G$ & $H_G$ & update &
decoder $\pi_G$ \\
\midrule
FBS & $f$ smooth $+$ one prox block & primal $x$ &
 $I$ & \eqref{eq:inst-fbs-primal} & $x$ \\
FBS & dual \eqref{eq:inst-dual} & dual $u$ &
 $I$ & \eqref{eq:inst-fbs-dual} & $\nabla f^*(-L^{\top}u)$ \\
DRS & $g$ prox $+$ $h$ prox ($f=0$, no $L$) & auxiliary $z$ &
 $I$ & \eqref{eq:inst-drs-primal} & $\operatorname{prox}_{\alpha g}(z)$ \\
DRS & dual \eqref{eq:inst-dual} & auxiliary $z$ &
 $I$ & \eqref{eq:inst-drs-dual} &
 $\nabla f^*(-L^{\top}\operatorname{prox}_{\alpha h^*}(z))$ \\
PDHG & one prox block $\varphi$ $+$ $h(Lx)$ & saddle $(x,u)$ &
 $M_{\tau\sigma}$ & \eqref{eq:inst-pdhg} & $x$ \\
Condat--V\~u & $f$ smooth $+$ $g$ prox $+$ $h(Lx)$ & saddle $(x,u)$ &
 $M_{\tau\sigma}$ & \eqref{eq:inst-cv} & $x$ \\
Davis--Yin & $f$ smooth $+$ $g$ prox $+$ $h$ prox (no $L$) &
 auxiliary $z$ & $I$ & \eqref{eq:inst-dys} &
 $\operatorname{prox}_{\alpha g}(z)$ \\
\bottomrule
\end{tabular}
\caption{Instantiation of the certificates of Table~\ref{tab:certificates}
in the compiler, per base scheme and accepted decomposition of
problem~\eqref{eq:composite}. $M_{\tau\sigma}$ is the primal--dual metric
of Table~\ref{tab:certificates}; on the dual instantiations the FBS/DRS
certificate is evaluated with the dual smoothness constant
$L_{F}=\ell^2/\mu$ in place of $L_f$.}
\label{tab:instantiation}
\end{table}
""")

    L.append(rf"""
\paragraph{{FBS.}}
Primal form ($L$ absent, exactly one nonzero prox block
$p\in\{{g,h\}}$):
\begin{{equation}}
 x_{{k+1}}=\operatorname{{prox}}_{{\alpha p}}\!\bigl(x_k-\alpha\nabla
 f(x_k)\bigr).
 \label{{eq:inst-fbs-primal}}
\end{{equation}}
Dual form \eqref{{eq:inst-dual}} (with $\nabla f^*$ affine, so
$\nabla F(u)=\tfrac1\mu LL^{{\top}}u-L\nabla f^*(0)$):
\begin{{equation}}
 u_{{k+1}}=\operatorname{{prox}}_{{\alpha h^*}}\!\Bigl(u_k-\alpha\bigl(
 \tfrac{{1}}{{\mu}}LL^{{\top}}u_k-L\nabla f^*(0)\bigr)\Bigr).
 \label{{eq:inst-fbs-dual}}
\end{{equation}}
Certificate condition (from the typing rules): {cond['fbs']}, evaluated
with $L_f$ on the primal form and $L_F=\ell^2/\mu$ on the dual form.
Fixed-point correspondence: on the primal form
$\operatorname{{Fix}}T=\mathcal X^\star$; on the dual form fixed points are
the dual solutions, and $\pi_G(u)=\nabla f^*(-L^{{\top}}u)$ recovers the
unique primal solution (strong convexity of $f$). On the TV form the
decoder specializes to $x=y-L^{{\top}}\operatorname{{clip}}(u;[-\lambda,
\lambda])$.
""")

    L.append(rf"""
\paragraph{{DRS.}}
Primal two-prox form ($f=0$, $L$ absent):
\begin{{equation}}
 x_g^k=\operatorname{{prox}}_{{\alpha g}}(z_k),\qquad
 x_h^k=\operatorname{{prox}}_{{\alpha h}}(2x_g^k-z_k),\qquad
 z_{{k+1}}=z_k+x_h^k-x_g^k.
 \label{{eq:inst-drs-primal}}
\end{{equation}}
Dual form \eqref{{eq:inst-dual}} with $A=\partial h^*$ and $B=\nabla F$:
$J_{{\alpha A}}=\operatorname{{prox}}_{{\alpha h^*}}$, and because
$\nabla f^*$ is affine the resolvent $J_{{\alpha B}}$ is the exact linear
solve
\begin{{equation}}
 u^k=\operatorname{{prox}}_{{\alpha h^*}}(z_k),\quad
 w^k=\Bigl(I+\tfrac{{\alpha}}{{\mu}}LL^{{\top}}\Bigr)^{{-1}}
 \bigl(2u^k-z_k+\alpha L\nabla f^*(0)\bigr),\quad
 z_{{k+1}}=z_k+w^k-u^k,
 \label{{eq:inst-drs-dual}}
\end{{equation}}
computed by conjugate gradients to relative residual $10^{{-12}}$.
Certificate condition: {cond['drs']}.
Fixed-point correspondence: every fixed point $z$ decodes to a solution
through the resolvent ($\operatorname{{prox}}_{{\alpha g}}(z)$ on the
primal form, the dual solution $\operatorname{{prox}}_{{\alpha h^*}}(z)$
then $\nabla f^*(-L^{{\top}}\cdot)$ on the dual form), and every solution
arises from at least one fixed point.
""")

    L.append(rf"""
\paragraph{{PDHG.}}
The primal block $\varphi$ is $f$ or $g$ (whichever is nonzero), or, when
$f$ is affine and $g$ is prox-friendly, the exact combined block
$\varphi=f+g$ through the linear-tilt identity
$\operatorname{{prox}}_{{\tau(q+p)}}=\operatorname{{prox}}_{{\tau p}}
\circ\operatorname{{prox}}_{{\tau q}}$ for affine $q$; this is how the
standard-form LP block $\langle c,x\rangle+I_{{x\geq0}}$ compiles to
$\max(x-\tau c,0)$. The emitted update is
\begin{{equation}}
 x_{{k+1}}=\operatorname{{prox}}_{{\tau\varphi}}\!\bigl(x_k-\tau
 L^{{\top}}u_k\bigr),\qquad
 u_{{k+1}}=\operatorname{{prox}}_{{\sigma h^*}}\!\bigl(u_k+\sigma
 L(2x_{{k+1}}-x_k)\bigr).
 \label{{eq:inst-pdhg}}
\end{{equation}}
Certificate condition: {cond['pdhg']}. Fixed points are the saddle points
of the $(\varphi,h,L)$ Lagrangian; $\pi_G$ is the primal coordinate.
""")

    L.append(rf"""
\paragraph{{Condat--V\~u.}}
\begin{{equation}}
 x_{{k+1}}=\operatorname{{prox}}_{{\tau g}}\!\bigl(x_k-\tau\nabla
 f(x_k)-\tau L^{{\top}}u_k\bigr),\qquad
 u_{{k+1}}=\operatorname{{prox}}_{{\sigma h^*}}\!\bigl(u_k+\sigma
 L(2x_{{k+1}}-x_k)\bigr).
 \label{{eq:inst-cv}}
\end{{equation}}
Certificate condition: {cond['condat_vu']}. Fixed points are the saddle
points of problem~\eqref{{eq:composite}}; $\pi_G$ is the primal
coordinate. With $L_f=0$ the update and its condition reduce exactly to
PDHG, which the LP instantiation exercises.
""")

    L.append(rf"""
\paragraph{{Davis--Yin.}}
\begin{{equation}}
 x_g^k=\operatorname{{prox}}_{{\alpha g}}(z_k),\qquad
 x_h^k=\operatorname{{prox}}_{{\alpha h}}\!\bigl(2x_g^k-z_k-\alpha\nabla
 f(x_g^k)\bigr),\qquad
 z_{{k+1}}=z_k+x_h^k-x_g^k.
 \label{{eq:inst-dys}}
\end{{equation}}
Certificate condition: {cond['dys']}. Fixed points correspond to
solutions through $\pi_G(z)=\operatorname{{prox}}_{{\alpha g}}(z)$.
""")

    L.append(r"""
\paragraph{Wrapper genes.}
The wrappers compile to state-space transformations of the certified map
$T_G$, exactly as certified: relaxation emits
$z_{k+1}=(1-\rho)z_k+\rho\,T_Gz_k$ (Lemma~\ref{lem:relaxation}); the
Halpern gene emits
\begin{equation}
 z_{k+1}=\tfrac{1}{j_k+2}\,z_{\rm anc}+\tfrac{j_k+1}{j_k+2}\,T_Gz_k,
 \label{eq:inst-halpern}
\end{equation}
where $j_k$ counts iterations since the last anchor reset and the restart
gene resets $z_{\rm anc}$ to the current iterate and $j_k$ to zero every
$k_{\rm r}$ iterations. The compiled anchor combination is applied to
$T_G$ itself, which is nonexpansive by (C1), rather than to the
normalized map $S_G$ of Proposition~\ref{prop:halpern}; the proposition
applies with $T_G$ in the role of the nonexpansive map when no anchor
restarts are used. It does not automatically give the same projection or rate
for periodically restarted trajectories.
""")

    L.append(rf"""
\paragraph{{Machine check.}}
The correspondence between this table and the emitted code is checked
numerically, not formally proved: tests/test\_compiler\_equivalence.py holds an independent
reference implementation of every update above (plain NumPy, double
precision, written directly from the formulas, sharing no code with the
compiler) and asserts agreement with the compiled iteration
iterate-by-iterate for 50 iterations on 3 random small problems per
scheme, drawn from a TV $16\times16$ instance, a tiny netflow LP, and
random dense composites, at relative tolerance $10^{{-10}}$ in the
maximum norm, for the full solver state and the decoded primal iterate.
The battery covers the plain map, the production genome-to-phenotype path,
and each wrapper gene (relaxation, Halpern, Halpern with fixed-$k$
restart). The largest deviation observed at generation time was
${dev_max}$ (DRS dual form, where the reference solves the resolvent
system exactly while the compiled path runs conjugate gradients to
$10^{{-12}}$); per scheme: FBS ${dev['fbs']}$, DRS ${dev['drs']}$, PDHG
${dev['pdhg']}$, Condat--V\~u ${dev['condat_vu']}$, Davis--Yin
${dev['dys']}$. This appendix is generated by
scripts/gen\_instantiation\_table.py, which extracts the certificate
conditions from the typing rules, verifies every display equation against
literal fragments of the builder source, and refuses to regenerate if a
scheme is added to the compiler without a corresponding entry and
reference implementation.
""")
    return "".join(L)


def main():
    check_anchors()
    certs = collect_certs()
    worst = measure_deviations()
    tex = build_tex(certs, worst)
    # Style laws (user-mandated): no em-dash, no italics commands.
    assert "---" not in tex, "em-dash (---) found in generated tex"
    assert "—" not in tex, "em-dash character found in generated tex"
    assert "\\emph" not in tex and "\\textit" not in tex, "italics found"
    with open(OUT, "w") as fh:
        fh.write(tex)
    print(f"wrote {OUT} ({len(tex)} bytes)")
    for k, v in sorted(worst.items()):
        print(f"  measured worst deviation {k:10s} {v:.3e}")


if __name__ == "__main__":
    main()
