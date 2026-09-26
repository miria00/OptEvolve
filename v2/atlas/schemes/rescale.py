"""Metric gene compilation: certified diagonal PROBLEM RESCALING.

The metric gene ({"kind": "ruiz", "iters": n} or {"kind":
"pock_chambolle", "alpha": a}) is compiled as an equivalent rescaling of
the standard-form LP composite, NOT as new typing rules; every existing
side condition applies verbatim to the rescaled problem, so zero new
convergence theory is introduced.

VARIABLE TRANSFORMS (documented per the metric-gene contract). Original
LP and its composite mapping (atlas.bench.lp_cell):

    min_x  c'x   s.t.  A x = b,  x >= 0
    f(x) = <c, x>, g = I_{x >= 0}, h(z) = I_{z = b}, L = A.

Let D1 (m x m) and D2 (n x n) be POSITIVE diagonal matrices. Substitute
x = D2 x' (a bijection of the nonnegative orthant onto itself, since
D2 > 0 entrywise):

    c'x = (D2 c)'x',   A x = b  <=>  A D2 x' = b  <=>  D1 A D2 x' = D1 b

(the last step left-multiplies the equality by the invertible D1). The
rescaled problem is therefore the SAME standard-form LP family:

    min_{x'} c~'x'  s.t.  A~ x' = b~,  x' >= 0,
    c~ = D2 c,  A~ = D1 A D2,  b~ = D1 b,

with the identical composite atoms (LinearCost, NonnegOrthant,
AffineEqualitySet), so every typing rule, prox identity and the LP gap
certificate of the existing machinery apply unchanged. Solution maps:

    primal:  x = D2 x'.
    dual:    the rescaled dual  max b~'y' s.t. A~'y' <= c~  satisfies
             A~'y' <= c~  <=>  D2 A'(D1 y') <= D2 c  <=>  A'(D1 y') <= c,
             so y = D1 y' is original-dual feasible with the same
             objective b~'y' = (D1 b)'y' = b'(D1 y') = b'y; in the PDHG
             iterate convention u = -y this reads  u = D1 u'.

Both maps are verified numerically in tests/test_metric_gene.py.

CERTIFICATE CONTRACT: termination is decided by the ORIGINAL problem's
f64 duality gap: the runtime metric maps (x', u') back through (D2, D1)
and evaluates atlas.certify.residuals.lp_rel_gap with the ORIGINAL
(c, A, b), so certificates remain statements about the original problem.
The rescaled operator's norm bound ||D1 A D2|| is fed to the EXISTING
gates (typecheck_pdhg / typecheck_condat_vu). For "ruiz" it is recomputed
by power iteration with the x1.02 safety factor. For "pock_chambolle" it
is the UNIVERSAL Lemma-2 constant 1 (x PC_SAFETY), so no power iteration
runs. The theorem supplies the universal bound 1, under which the gate
condition tau*sigma*||L'||^2 < 1 would be exactly tau*sigma < 1: a
condition on the GENOME ALONE, with the instance geometry dropped out.
The IMPLEMENTATION conservatively inflates that bound by PC_SAFETY, so
what actually runs is tau*sigma < 1/PC_SAFETY^2. Do not state the exact
form while a margin is in force; quote 1/PC_SAFETY^2.

DIAGONALS:
- "ruiz": Ruiz equilibration (Ruiz 2001, "A scaling algorithm for
  equilibrating both rows and columns norms in matrices"): `iters`
  rounds of  M <- diag(r)^{-1/2} M diag(c)^{-1/2}  with r/c the row/col
  infinity norms, accumulated into D1, D2. iters = 0 is the identity.
- "pock_chambolle": the Pock & Chambolle (2011) diagonal
  preconditioning rule, used here as a rescaling: their diagonal steps
  T_j = 1/sum_i |A_ij|^{2-alpha}, S_i = 1/sum_j |A_ij|^alpha give
  ||S^{1/2} A T^{1/2}|| <= 1 (their Lemma 2), so D1 = S^{1/2},
  D2 = T^{1/2}. The bound is taken from the theorem, not estimated.

Explicit matrix entries come from the lp_standard composite's
data["A_dense"] (netflow incidence and MPS matrices carry them; for a
matvec-only operator this is exactly the dense probe).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np

from atlas.ir.spec import CompositeProblem
from atlas.typing_ import TypeCheckError


# Pock & Chambolle (2011) Lemma 2: ||diag(d1) A diag(d2)|| <= 1 for every
# alpha in [0, 2], by construction of the diagonals. The diagonals are
# instance-DEPENDENT; the bound they achieve is UNIVERSAL.
PC_CERTIFIED_NORM = 1.0
# Multiplied into the certified bound to absorb floating-point drift in the
# computed diagonals. Soundness does not depend on it: the theorem gives
# ||D1 A D2|| <= 1, so tau*sigma < 1 already suffices.
#
# SIZING (2026-09-04). This was 1.01, which is far too blunt. The gate checks
# tau*sigma*nb^2 < 1, so nb = 1.01 implements tau*sigma < 1/1.01^2 = 0.980296
# and REJECTS the band [0.980296, 1). Phase-2 champions pressed exactly that
# boundary (t*s*||L||^2 = 0.950 and 0.9925), so a genome sitting where a real
# champion sat was being refused by the margin alone. Meanwhile the drift the
# margin exists to absorb is about n*eps ~ 2e-12 for n ~ 1e4, and the audited
# true norm over 137 instances x 3 alphas was exactly 1.000000000. 1e-9 is
# still ~1000x the plausible drift and costs only tau*sigma < 1 - 2e-9.
PC_SAFETY = 1.0 + 1e-9
# Slack allowed before verify=True declares the theorem violated.
PC_VERIFY_TOL = 1e-9


def _is_sparse(A) -> bool:
    """scipy.sparse duck-check without importing scipy at module import
    time (scipy is an atlas dependency anyway, but the schemes package
    only needs it when a sparse composite actually arrives)."""
    return hasattr(A, "tocsr") and hasattr(A, "nnz")


def power_iteration_norm_bound(
    A: np.ndarray, iters: int = 500, tol: float = 1e-13, seed: int = 0,
    safety: float = 1.02,
) -> float:
    """Power-iteration estimate of ||A||_2 inflated by `safety` (x1.02).

    Deterministic in `seed`; mirrors atlas.bench.lp_cell.power_iteration_norm
    (kept local so atlas.schemes does not import the bench package).
    Correctness (bound >= true norm on random matrices) is tested in
    tests/test_metric_gene.py.
    """
    if not _is_sparse(A):
        A = np.asarray(A, dtype=np.float64)
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(A.shape[1])
    nv = np.linalg.norm(v)
    if nv == 0.0:
        raise RuntimeError("power iteration: degenerate start vector")
    v = v / nv
    sigma = 0.0
    for _ in range(iters):
        w = A @ v
        s_new = float(np.linalg.norm(w))
        if s_new == 0.0:
            return 0.0
        z = A.T @ (w / s_new)
        nz = np.linalg.norm(z)
        if nz == 0.0:
            break
        v = z / nz
        if abs(s_new - sigma) <= tol * max(1.0, s_new):
            sigma = s_new
            break
        sigma = s_new
    return float(safety * sigma)


def ruiz_equilibrate(A: np.ndarray, iters: int) -> tuple[np.ndarray, np.ndarray]:
    """Ruiz row/col infinity-norm equilibration diagonals (d1, d2).

    Returns positive vectors with M = diag(d1) A diag(d2) approximately
    balanced (row and column infinity norms -> 1). Zero rows/columns are
    left unscaled (divisor 1). iters = 0 returns the identity scaling.

    Accepts a dense ndarray or a scipy.sparse matrix (the sparse path
    computes the identical diagonals on the explicit entries; needed for
    the MPS composites, whose matrices are never materialized densely).
    """
    if _is_sparse(A):
        return _ruiz_equilibrate_sparse(A, iters)
    A = np.asarray(A, dtype=np.float64)
    if A.ndim != 2:
        raise ValueError(f"ruiz_equilibrate expects a matrix, got shape {A.shape}")
    if iters < 0:
        raise ValueError("ruiz iters must be >= 0")
    m, n = A.shape
    d1 = np.ones(m)
    d2 = np.ones(n)
    M = A.copy()
    for _ in range(int(iters)):
        r = np.sqrt(np.max(np.abs(M), axis=1))
        c = np.sqrt(np.max(np.abs(M), axis=0))
        r[r == 0.0] = 1.0
        c[c == 0.0] = 1.0
        d1 /= r
        d2 /= c
        M = M / r[:, None] / c[None, :]
    return d1, d2


def _ruiz_equilibrate_sparse(A_sp, iters: int) -> tuple[np.ndarray, np.ndarray]:
    """Sparse Ruiz: same fixed point and update as the dense path, on the
    explicit nonzero entries only (zero rows/cols stay unscaled)."""
    import scipy.sparse as sp

    if iters < 0:
        raise ValueError("ruiz iters must be >= 0")
    M = A_sp.tocsr().astype(np.float64, copy=True)
    m, n = M.shape
    d1 = np.ones(m)
    d2 = np.ones(n)
    for _ in range(int(iters)):
        absM = abs(M)
        r = np.sqrt(absM.max(axis=1).toarray().ravel())
        c = np.sqrt(absM.max(axis=0).toarray().ravel())
        r[r == 0.0] = 1.0
        c[c == 0.0] = 1.0
        d1 /= r
        d2 /= c
        M = sp.diags(1.0 / r) @ M @ sp.diags(1.0 / c)
    return d1, d2


def pock_chambolle_diagonals(
    A: np.ndarray, alpha: float
) -> tuple[np.ndarray, np.ndarray]:
    """Pock-Chambolle (2011) diagonal rule as rescaling diagonals (d1, d2).

    d1_i = (sum_j |A_ij|^alpha)^{-1/2}, d2_j = (sum_i |A_ij|^{2-alpha})^{-1/2}
    (zero rows/columns unscaled), so ||diag(d1) A diag(d2)|| <= 1 by their
    Lemma 2; the actual bound used by the gates is still recomputed by
    power iteration x1.02. Accepts dense or scipy.sparse input.

    STRUCTURAL-ZERO CONVENTION (2026-09-04). The sums run over the
    STRUCTURAL NONZEROS only, i.e. 0^0 := 0, which is the convention
    Lemma 2's proof uses: it splits |K_ij| = |K_ij|^(alpha/2) *
    |K_ij|^(1-alpha/2) and applies Cauchy-Schwarz, and a structural zero
    contributes nothing to either factor. This matters only at the
    endpoints alpha = 0 (row sums) and alpha = 2 (column sums), where the
    naive dense rule 0^0 = 1 would count ambient dimension instead of
    nnz. Counting ambient dimension only ever INFLATES the denominators,
    so it stays valid (smaller steps) but is strictly looser, and it made
    the dense and sparse paths disagree. Previously the sparse path did
    not merely disagree, it RAISED: scipy's `.power(0.0)` refuses to
    densify, so alpha = 0 and alpha = 2 (via 2 - alpha) crashed on every
    sparse composite, which is every MPS/OOD instance. Both endpoints are
    inside the certified range and are now reachable.
    """
    if not (0.0 <= alpha <= 2.0):
        raise ValueError("pock_chambolle alpha must be in [0, 2]")
    if _is_sparse(A):
        absA = abs(A.tocsr().astype(np.float64))
        # Take the power on the explicit data only: this both avoids
        # scipy's zero-power densification refusal and realizes the
        # 0^0 := 0 convention above.
        row_m = absA.copy()
        row_m.data = np.power(row_m.data, alpha)
        col_m = absA.copy()
        col_m.data = np.power(col_m.data, 2.0 - alpha)
        row = np.asarray(row_m.sum(axis=1)).ravel()
        col = np.asarray(col_m.sum(axis=0)).ravel()
        d1 = np.where(row > 0.0, 1.0 / np.sqrt(np.where(row > 0.0, row, 1.0)), 1.0)
        d2 = np.where(col > 0.0, 1.0 / np.sqrt(np.where(col > 0.0, col, 1.0)), 1.0)
        return d1, d2
    A = np.asarray(A, dtype=np.float64)
    absA = np.abs(A)
    nz = absA > 0.0
    row = np.sum(np.where(nz, absA**alpha, 0.0), axis=1)
    col = np.sum(np.where(nz, absA ** (2.0 - alpha), 0.0), axis=0)
    d1 = np.where(row > 0.0, 1.0 / np.sqrt(np.where(row > 0.0, row, 1.0)), 1.0)
    d2 = np.where(col > 0.0, 1.0 / np.sqrt(np.where(col > 0.0, col, 1.0)), 1.0)
    return d1, d2


@dataclass(frozen=True, eq=False)
class ScaledLinOp:
    """L' = diag(d1) L diag(d2) without materializing the product.

    forward(x) = d1 * L(d2 * x), adjoint(u) = d2 * L'(d1 * u); norm_bound
    is the recomputed power-iteration bound of the rescaled matrix.
    eq=False keeps the identity-based hash for the kernel-cache
    static_key (the MatrixLinOp convention). DTYPE CONTRACT: like
    IncidenceLinOp, forward/adjoint preserve the input dtype (the
    diagonals are cast to it), so the op is safe under the f32-iterate
    backend policies; certificates still evaluate through the ORIGINAL
    f64 operator (atlas.schemes.compile._solve_with_metric)."""

    base: Any  # the original LinOp
    d1: Any  # jnp (m,)
    d2: Any  # jnp (n,)
    norm_bound: float

    def forward(self, x):
        return self.d1.astype(x.dtype) * self.base.forward(
            self.d2.astype(x.dtype) * x
        )

    def adjoint(self, u):
        return self.d2.astype(u.dtype) * self.base.adjoint(
            self.d1.astype(u.dtype) * u
        )


def rescale_lp_problem(
    problem: CompositeProblem, kind: str, iters=None, alpha=None,
    pc_certified: bool = True, pc_safety: float = PC_SAFETY,
    verify: bool = False,
) -> tuple[CompositeProblem, np.ndarray, np.ndarray, float]:
    """Build the equivalently rescaled lp_standard composite.

    Returns (rescaled_problem, d1, d2, opnorm_rescaled). See the module
    docstring for the transforms; the rescaled problem is the same atom
    family with c~ = d2*c, b~ = d1*b, L' = diag(d1) A diag(d2).

    NORM BOUND. For kind "ruiz" the bound is the recomputed power-iteration
    estimate (x1.02): Ruiz equilibration carries no a-priori bound. For kind
    "pock_chambolle" with pc_certified=True (the default) the bound is the
    CERTIFIED constant of Lemma 2 scaled by `pc_safety`, computed without
    touching the matrix, so it is identical on every instance. Pass
    pc_certified=False to fall back to power iteration, or verify=True to
    compute the true norm anyway and refuse to certify if the theorem does
    not hold (used by the tests and by any first run on a new instance
    family).
    """
    if problem.name != "lp_standard":
        raise TypeCheckError(
            "the metric gene rescales the standard-form LP composite only; "
            f"got problem {problem.name!r} (diagonal rescaling does not "
            "preserve this problem's atom structure)"
        )
    A_entries = problem.data.get("A_dense")
    sparse_entries = False
    if A_entries is None:
        A_entries = problem.data.get("A_csr")  # MPS composites (sparse)
        sparse_entries = A_entries is not None
    if A_entries is None:
        raise TypeCheckError(
            "metric gene needs explicit matrix entries (data['A_dense'] or "
            "data['A_csr']) for equilibration; this lp_standard composite "
            "carries none"
        )
    if not sparse_entries:
        A_entries = np.asarray(A_entries, dtype=np.float64)
    if kind == "ruiz":
        d1, d2 = ruiz_equilibrate(A_entries, int(iters))
    elif kind == "pock_chambolle":
        d1, d2 = pock_chambolle_diagonals(A_entries, float(alpha))
    elif kind == "ruiz_pc":
        # Ruiz THEN Pock-Chambolle: the frozen LP substrate. PC is applied to
        # the RUIZ-SCALED entries, so its universal bound covers the composed
        # operator and the composed diagonals are the elementwise products.
        # Routed through this function (rather than chaining two rescales in
        # the caller) so that _solve_with_metric's ORIGINAL-coordinate
        # certificate machinery applies unchanged: a solve handed a bare
        # preconditioned problem would certify the SCALED gap, which is a
        # different number (measured on netflow: scaled 9.99e-7 "converged"
        # while the original-coordinate gap was 1.098e-6, and the primal
        # residual differed by 2x).
        d1r, d2r = ruiz_equilibrate(A_entries, int(iters))
        if sparse_entries:
            import scipy.sparse as sp

            A_r = sp.diags(d1r) @ A_entries.tocsr() @ sp.diags(d2r)
        else:
            A_r = A_entries * d1r[:, None] * d2r[None, :]
        d1p, d2p = pock_chambolle_diagonals(A_r, float(alpha))
        d1, d2 = d1p * d1r, d2r * d2p
    else:
        raise TypeCheckError(f"unknown metric kind {kind!r}")
    if sparse_entries:
        import scipy.sparse as sp

        A_scaled = sp.diags(d1) @ A_entries.tocsr() @ sp.diags(d2)
    else:
        A_scaled = A_entries * d1[:, None] * d2[None, :]
    if kind in ("pock_chambolle", "ruiz_pc") and pc_certified:
        # CERTIFIED BOUND (Pock & Chambolle 2011, Lemma 2): for any alpha in
        # [0, 2] the diagonal rule gives ||D1 A D2|| <= 1 as a THEOREM ABOUT
        # THE FORMULA, so no power iteration is needed. WORDING: the DIAGONALS
        # d1, d2 depend on the instance; what is UNIVERSAL is the BOUND they
        # achieve. The consequence is that the gate condition
        # tau*sigma*||L'||^2 < 1 collapses to tau*sigma < 1, a condition on the
        # GENOME ALONE with the instance geometry dropped out. A genome
        # certified against this bound is certified on EVERY instance, which is
        # Phase 2 lacked (absolute steps certified against the netflow opnorm
        # 2.885 met OOD opnorms up to 3.4e6 and were rejected 1340/1340).
        # Strictness: the gates require tau*sigma*nb^2 < 1; with nb = 1 this
        # reduces to tau*sigma < 1, which is already strict, so soundness does
        # NOT rest on the safety factor. PC_SAFETY > 1 is defence against
        # floating-point drift in the diagonals only (measured max true norm
        # over 137 Mittelmann/MIPLIB instances x 3 alphas: exactly 1.000000000).
        nb = PC_CERTIFIED_NORM * float(pc_safety)
        if verify:
            true_nb = power_iteration_norm_bound(A_scaled, safety=1.0)
            if true_nb > PC_CERTIFIED_NORM + PC_VERIFY_TOL:
                raise TypeCheckError(
                    "Pock-Chambolle Lemma 2 violated by the computed "
                    f"diagonals: ||D1 A D2|| = {true_nb:.12g} > 1 "
                    f"(alpha={alpha}); the scaling is wrong, refusing to "
                    "certify against the theoretical bound"
                )
    else:
        nb = power_iteration_norm_bound(A_scaled)
    c = np.asarray(problem.data["c"], dtype=np.float64)
    b = np.asarray(problem.data["b"], dtype=np.float64)
    c_s = jnp.asarray(d2 * c)
    b_s = jnp.asarray(d1 * b)
    L_s = ScaledLinOp(
        base=problem.L,
        d1=jnp.asarray(d1),
        d2=jnp.asarray(d2),
        norm_bound=float(nb),
    )
    # Rebuild the atoms with the same classes as the original problem so
    # the scheme builders and _gap_kind dispatch identically (imported
    # lazily from the bench cell that defined them; the classes travel
    # with the problem instance, so this stays instance-driven).
    f_s = type(problem.f)(c=c_s)
    g_s = type(problem.g)()
    h_s = type(problem.h)(b=b_s)
    data = dict(problem.data)
    data.update({"c": c_s, "b": b_s})
    if sparse_entries:
        data["A_csr"] = A_scaled
    else:
        data["A_dense"] = A_scaled
    rescaled = CompositeProblem(
        f=f_s,
        g=g_s,
        h=h_s,
        L=L_s,
        data=data,
        shape=problem.shape,
        name="lp_standard",
    )
    return rescaled, d1, d2, float(nb)


# ---------------------------------------------------------------------------
# LP PRECONDITIONING SUBSTRATE (2026-09-04)
# ---------------------------------------------------------------------------

# PDLP's published pipeline: 10 Ruiz iterations, then Pock-Chambolle diagonal
# scaling. Their ablation (none 207 -> Ruiz 252 -> PC 256 -> both 283 solved)
# is why both are applied rather than either alone.
LP_RUIZ_ITERS = 10
LP_PC_ALPHA = 1.0


def precondition_lp(
    problem: CompositeProblem,
    ruiz_iters: int = LP_RUIZ_ITERS,
    pc_alpha: float = LP_PC_ALPHA,
    verify: bool = False,
) -> tuple[CompositeProblem, np.ndarray, np.ndarray, float]:
    """Ruiz(n) THEN Pock-Chambolle(alpha) as fixed LP infrastructure.

    Returns (preconditioned_problem, d1, d2, certified_norm_bound).

    WARNING: this returns a BARE rescaled problem. Solving it directly
    certifies the SCALED gap, which is a different number from the original
    problem's (measured on netflow: 9.99e-7 scaled vs 1.098e-6 original, and
    a 2x difference in primal residual). Use it only for analysis. To RUN a
    genome under this substrate, attach the metric gene
    {"kind": "ruiz_pc", "iters": n, "alpha": a} and call solve_genome on the
    ORIGINAL problem, which routes through _solve_with_metric and evaluates
    termination on the original problem's f64 duality gap.

    WHY THE CERTIFICATE SURVIVES THE COMPOSITION. Ruiz produces
    A_R = D1r A D2r; Pock-Chambolle is then applied TO A_R and, by Lemma 2,
    yields ||D1p A_R D2p|| <= 1 regardless of what A_R happens to be. Because
    PC is applied LAST, the universal bound holds for the composed operator
    too, and the composed diagonals are d1 = d1p * d1r, d2 = d2r * d2p. Any
    Ruiz count therefore enters the SAME certificate pathway, which is what
    makes n_ruiz safe to expose as a gene later with no new proof obligation.
    """
    return rescale_lp_problem(
        problem, "ruiz_pc", iters=int(ruiz_iters), alpha=float(pc_alpha),
        pc_certified=True, verify=verify,
    )


def lp_substrate_metric(ruiz_iters: int = LP_RUIZ_ITERS,
                        pc_alpha: float = LP_PC_ALPHA) -> dict:
    """The frozen LP substrate as a metric-gene dict, to be attached
    IDENTICALLY to every configuration on the cell (vanilla, tuned baseline,
    headroom families, evolved candidates) so that no comparison can be won
    by rediscovering numerical hygiene."""
    return {"kind": "ruiz_pc", "iters": int(ruiz_iters),
            "alpha": float(pc_alpha)}


# ---------------------------------------------------------------------------
# CONVEX QP (2026-09-10)
# ---------------------------------------------------------------------------
#
# The OSQP-form QP  min (1/2)x'Px + q'x + r  s.t.  l <= Ax <= u  is closed
# under positive diagonal rescaling. Substitute x = D2 x' and scale row i of
# the constraint by d1_i > 0 (which maps the interval [l_i, u_i] onto
# [d1_i l_i, d1_i u_i], infinities preserved):
#
#     P~ = D2 P D2,  q~ = D2 q,  A~ = D1 A D2,  l~ = D1 l,  u~ = D1 u.
#
# The rescaled problem is the same atom family (QuadraticForm, Zero, Box), so
# every typing rule applies verbatim, with L_f recomputed as ||P~|| and ||L'||
# as ||A~||. Solution maps: x = D2 x', y = D1 y' (the multiplier of a row
# scaled by d1_i is scaled by d1_i). The certificate is qp_rel_kkt evaluated
# on the ORIGINAL (P, q, A, l, u) after mapping back.
#
# "ruiz" on a QP equilibrates the KKT matrix [[P, A'], [A, 0]] (Ruiz 2001, in
# the form OSQP uses): column scaling of x from max(||P e_j||, ||A e_j||),
# row scaling of the constraints from ||A' e_i||, infinity norms, norms below
# QP_MIN_SCALING treated as 1, scalings clipped to [1e-4, 1e4].

QP_MIN_SCALING, QP_MAX_SCALING = 1e-4, 1e4


def kkt_ruiz_equilibrate(P, A, iters: int) -> tuple[np.ndarray, np.ndarray]:
    """Ruiz on the QP KKT matrix; returns (d1 rows of A, d2 columns)."""
    import scipy.sparse as sp

    Ps, As = sp.csc_matrix(P, dtype=np.float64), sp.csc_matrix(A, dtype=np.float64)
    d1, d2 = np.ones(As.shape[0]), np.ones(As.shape[1])
    for _ in range(int(iters)):
        col = np.maximum(abs(Ps).max(axis=0).toarray().ravel(),
                         abs(As).max(axis=0).toarray().ravel())
        row = abs(As).max(axis=1).toarray().ravel()
        col = np.where(col < QP_MIN_SCALING, 1.0, col)
        row = np.where(row < QP_MIN_SCALING, 1.0, row)
        dk = 1.0 / np.sqrt(np.clip(col, QP_MIN_SCALING, QP_MAX_SCALING))
        ek = 1.0 / np.sqrt(np.clip(row, QP_MIN_SCALING, QP_MAX_SCALING))
        Ps = (sp.diags(dk) @ Ps @ sp.diags(dk)).tocsc()
        As = (sp.diags(ek) @ As @ sp.diags(dk)).tocsc()
        d1 *= ek
        d2 *= dk
    return d1, d2


def rescale_qp_problem(
    problem: CompositeProblem, kind: str, iters=None, alpha=None,
    pc_safety: float = PC_SAFETY, power_iters: int | None = None,
    gpu_threshold: int | None = None,
) -> tuple[CompositeProblem, np.ndarray, np.ndarray, float]:
    """Build the equivalently rescaled qp_standard composite.

    Returns (rescaled_problem, d1, d2, opnorm_rescaled). L_f of the rescaled
    quadratic is recomputed by power iteration x1.02 inside the loader; the
    operator bound is power iteration x1.02 for "ruiz" and the Pock-Chambolle
    Lemma-2 constant x pc_safety when PC is applied last.
    """
    import dataclasses
    import scipy.sparse as sp

    from atlas.bench.qp_cell import to_composite

    if problem.name != "qp_standard":
        raise TypeCheckError(
            f"rescale_qp_problem needs the qp_standard composite, got {problem.name!r}")
    d = problem.data
    P, A = sp.csc_matrix(d["P"]), sp.csc_matrix(d["A"])
    if kind == "ruiz":
        d1, d2 = kkt_ruiz_equilibrate(P, A, int(iters))
    elif kind == "pock_chambolle":
        d1, d2 = pock_chambolle_diagonals(A, float(alpha))
    elif kind == "ruiz_pc":
        d1r, d2r = kkt_ruiz_equilibrate(P, A, int(iters))
        d1p, d2p = pock_chambolle_diagonals(sp.diags(d1r) @ A @ sp.diags(d2r), float(alpha))
        d1, d2 = d1p * d1r, d2r * d2p
    else:
        raise TypeCheckError(f"unknown metric kind {kind!r}")
    scaled = {
        "P": (sp.diags(d2) @ P @ sp.diags(d2)).tocsc(),
        "q": d2 * np.asarray(d["q"], dtype=np.float64),
        "r": d["r"],
        "A": (sp.diags(d1) @ A @ sp.diags(d2)).tocsc(),
        "l": d1 * np.asarray(d["l"], dtype=np.float64),
        "u": d1 * np.asarray(d["u"], dtype=np.float64),
    }
    rescaled = to_composite(scaled, power_iters=power_iters, gpu_threshold=gpu_threshold)
    if kind in ("pock_chambolle", "ruiz_pc"):
        rescaled = dataclasses.replace(
            rescaled, L=dataclasses.replace(rescaled.L, norm_bound=PC_CERTIFIED_NORM * float(pc_safety)))
    return rescaled, d1, d2, float(rescaled.L.norm_bound)
