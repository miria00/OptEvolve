"""Standard-form LP benchmark cell: problem side + composite mapping.

LP:                 min_x  c'x   s.t.   A x = b,  x >= 0.

Composite mapping onto  min_x f(x) + g(x) + h(L x)  (atlas.ir.spec):

    f(x) = <c, x>            affine: grad f = c (constant), L_f = 0,
                             prox_{t f}(v) = v - t c (exact, translation)
    g(x) = I_{x >= 0}        prox_{t g}(v) = max(v, 0)  (clip at 0), any t
    h(z) = I_{z = b}         h*(u) = <b, u>  (support of the point {b}),
                             prox_{s h*}(v) = v - s b   for any s > 0
    L    = A                 sparse-friendly matvec/adjoint LinOp with a
                             power-iteration norm bound

Derivation of prox_{s h*} (verified numerically in tests/test_lp_cell.py):
directly, argmin_u s<b,u> + (1/2)||u - v||^2 has stationarity
s b + u - v = 0, so u* = v - s b; via the Moreau identity,
prox_{s h*}(v) = v - s prox_{h/s}(v/s) = v - s b since prox of the
indicator of {b} is the constant map b.

With this mapping, Condat-Vu (L_f = 0 makes it plain PDHG / Chambolle-Pock,
the vanilla-PDLP iteration) reads

    x+ = max(x - t c - t A'u, 0)
    u+ = u + s (A(2x+ - x) - b)

side condition t*s*||A||^2 < 1 (typecheck_pdhg / typecheck_condat_vu).

Certificate: lp_rel_gap(x, u, c, A, b) = max of the three PDLP-style
relative criteria (primal feasibility, dual feasibility, duality gap) for
the dual pair  min c'x (Ax=b, x>=0)  <->  max b'y (A'y <= c)  with the
PDHG dual iterate u = -y. Weak duality gives c'x - b'y >= 0 on feasible
points (tested).

Oracle: HiGHS via scipy.optimize.linprog(method="highs"), cached by
content hash to V2/data/oracle_cache_lp/; stores the optimal value AND an
optimal dual y (standard sign: A'y <= c, b'y = c'x*).

INTEGRATION STATUS (docstring corrected 2026-08-26; the previous
"INTEGRATOR TODO" here was stale): the schemes side IS wired up.
atlas/schemes/base_schemes.py dispatches on the CompositeProblem atoms and
recognizes name == "lp_standard" (gap_kind "lp_gap", certified by
lp_rel_gap); the mini-E-R and mix-discovery LP experiments ran through
that path. reference_solve_composite() below remains as an independent
numerical check of the composite mapping only.

GATE NOTE: the Stage-0 constant atlas.genome.OPNORM_LP covers ONLY
degree-2 regular netflow. build_instances defaults family "regular" to
gen_regular_flow's degree=4 and also offers "grid"/"dense", whose operator
norms exceed that constant; when gating genomes for such instance sets,
pass gate_opnorm(instances) as validate(..., lp_opnorm=...).
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from typing import Any, Optional

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import linprog

from atlas.bench.netflow import NetflowLP, gen_grid_flow, gen_regular_flow
from atlas.certify.residuals import lp_rel_gap  # canonical home (jit-safe)
from atlas.ir.spec import CompositeProblem
from atlas.operators.base import PropSet
from atlas.typing_ import typecheck_condat_vu
from atlas.wrappers.km import Budget, km_run

ORACLE_CACHE_LP_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data",
    "oracle_cache_lp",
)


# ---------------------------------------------------------------------------
# Linear operators (sparse-friendly matvec/adjoint + norm bound)
# ---------------------------------------------------------------------------


def power_iteration_norm(
    matvec,
    rmatvec,
    n_in: int,
    iters: int = 500,
    tol: float = 1e-13,
    seed: int = 0,
    safety: float = 1.02,
) -> float:
    """Power-iteration estimate of ||A||_2, inflated by `safety`.

    Runs power iteration on A'A via numpy callables until the singular-value
    estimate stabilizes to relative `tol` (or `iters` cap), then multiplies
    by `safety` so the returned value upper-bounds ||A||_2 in practice
    (verified against exact SVD in tests for both instance families).
    Deterministic in `seed`.
    """
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(n_in)
    nv = np.linalg.norm(v)
    if nv == 0.0:
        raise RuntimeError("power iteration: degenerate start vector")
    v = v / nv
    sigma = 0.0
    for _ in range(iters):
        w = np.asarray(matvec(v))
        s_new = float(np.linalg.norm(w))
        if s_new == 0.0:
            return 0.0  # A v = 0 exactly: treat as the zero map
        z = np.asarray(rmatvec(w / s_new))
        nz = np.linalg.norm(z)
        if nz == 0.0:
            break
        v = z / nz
        if abs(s_new - sigma) <= tol * max(1.0, s_new):
            sigma = s_new
            break
        sigma = s_new
    return float(safety * sigma)


@dataclass(frozen=True, eq=False)
class MatrixLinOp:
    """Dense matrix as a LinOp: forward = A @ x, adjoint = A.T @ u.

    eq=False keeps the identity-based __hash__ so instances are usable in
    the kernel-cache static_key (a content hash over A would defeat the
    point: the operator lives in the T closure, so each instance compiles
    its own kernel anyway)."""

    A: Any  # jnp (m, n)
    norm_bound: float

    @staticmethod
    def from_dense(A_np: np.ndarray, safety: float = 1.02) -> "MatrixLinOp":
        A_np = np.asarray(A_np, dtype=np.float64)
        nb = power_iteration_norm(
            lambda x: A_np @ x, lambda u: A_np.T @ u, A_np.shape[1], safety=safety
        )
        return MatrixLinOp(A=jnp.asarray(A_np), norm_bound=nb)

    def forward(self, x):
        return self.A.astype(x.dtype) @ x

    def adjoint(self, u):
        return self.A.T.astype(u.dtype) @ u


@dataclass(frozen=True, eq=False)
class IncidenceLinOp:
    """Node-arc incidence as a LinOp WITHOUT materializing the matrix.

    forward(x)[i] = sum_{e: tail(e)=i} x_e - sum_{e: head(e)=i} x_e
    adjoint(y)[e] = y[tail(e)] - y[head(e)]
    (scatter-add / gather; cost O(n_arcs), sparse-friendly).
    """

    tail: Any  # jnp int64 (n_arcs,)
    head: Any  # jnp int64 (n_arcs,)
    n_nodes: int
    norm_bound: float

    @staticmethod
    def from_arcs(
        tail: np.ndarray, head: np.ndarray, n_nodes: int, safety: float = 1.02
    ) -> "IncidenceLinOp":
        tail = np.asarray(tail, dtype=np.int64)
        head = np.asarray(head, dtype=np.int64)

        def mv(x):
            y = np.zeros(n_nodes)
            np.add.at(y, tail, x)
            np.add.at(y, head, -x)
            return y

        def rmv(y):
            return y[tail] - y[head]

        nb = power_iteration_norm(mv, rmv, tail.shape[0], safety=safety)
        return IncidenceLinOp(
            tail=jnp.asarray(tail),
            head=jnp.asarray(head),
            n_nodes=int(n_nodes),
            norm_bound=nb,
        )

    def forward(self, x):
        y = jnp.zeros(self.n_nodes, dtype=x.dtype)
        return y.at[self.tail].add(x).at[self.head].add(-x)

    def adjoint(self, y):
        return y[self.tail] - y[self.head]


# ---------------------------------------------------------------------------
# Atoms (typed; carry PropSet like atlas.operators.prox atoms)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LinearCost:
    """f(x) = <c, x>. Affine: grad f = c is constant, i.e. 0-Lipschitz
    (L_f = 0); prox_{t f}(v) = v - t c exactly (a translation, hence a
    prox and (1/2)-averaged)."""

    c: Any
    props: PropSet = field(
        default=PropSet(lipschitz=0.0, averagedness=0.5, prox_friendly=True)
    )

    def grad(self, x):
        return jnp.broadcast_to(self.c, jnp.shape(x))

    def prox(self, v, step):
        return v - step * self.c


@dataclass(frozen=True)
class NonnegOrthant:
    """g(x) = I_{x >= 0}; prox = projection = clip at 0, any step."""

    props: PropSet = field(default=PropSet(averagedness=0.5, prox_friendly=True))

    def prox(self, v, step=None):
        return jnp.maximum(v, 0.0)


@dataclass(frozen=True)
class AffineEqualitySet:
    """h(z) = I_{z = b}. prox_{t h}(v) = b (constant map, any t).

    h*(u) = <b, u>, so prox_{s h*}(v) = v - s b (see module docstring for
    the two derivations; verified numerically in tests)."""

    b: Any
    props: PropSet = field(default=PropSet(averagedness=0.5, prox_friendly=True))

    def prox(self, v, step=None):
        return jnp.broadcast_to(self.b, jnp.shape(v))

    def prox_conj(self, v, s):
        return v - s * self.b


# Pytree registration: atom DATA (c, b) are dynamic leaves so the cached
# solver kernels of atlas.wrappers.km take LP instance data as runtime
# arguments (same convention as atlas.operators.prox atoms).
jax.tree_util.register_dataclass(LinearCost, data_fields=["c"], meta_fields=["props"])
jax.tree_util.register_dataclass(NonnegOrthant, data_fields=[], meta_fields=["props"])
jax.tree_util.register_dataclass(
    AffineEqualitySet, data_fields=["b"], meta_fields=["props"]
)


# ---------------------------------------------------------------------------
# Instances + composite adapter
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LPInstance:
    """Standard-form LP min c'x s.t. Ax=b, x>=0 with a solver-side LinOp."""

    c: np.ndarray  # (n,)
    b: np.ndarray  # (m,)
    A_dense: np.ndarray  # (m, n) dense copy for the oracle and checks
    L: Any  # LinOp (MatrixLinOp or IncidenceLinOp)
    name: str
    meta: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return int(self.c.shape[0])

    @property
    def m(self) -> int:
        return int(self.b.shape[0])


def to_composite(inst: LPInstance) -> CompositeProblem:
    """The one-call adapter: LPInstance -> E1 composite (see module docstring)."""
    c = jnp.asarray(inst.c, dtype=jnp.float64)
    b = jnp.asarray(inst.b, dtype=jnp.float64)
    return CompositeProblem(
        f=LinearCost(c=c),
        g=NonnegOrthant(),
        h=AffineEqualitySet(b=b),
        L=inst.L,
        data={"c": c, "b": b, "A_dense": inst.A_dense, "meta": dict(inst.meta)},
        shape=(inst.n,),
        name="lp_standard",
    )


def gen_lp_dense(n: int, m: int, seed: int, slack_scale: float = 0.1) -> LPInstance:
    """v1 gen_lp dense recipe (ported construction, numpy RNG; no v1 import).

    A ~ N(0, 1) of shape (m, n); x_feas ~ U(slack_scale, 1+slack_scale) > 0
    and b = A x_feas (primal feasible by construction);
    c = A'y_aux + s_aux with s_aux ~ U(slack_scale, 1+slack_scale) > 0 so
    the dual A'y + s = c, s >= 0 is feasible at (y_aux, s_aux): bounded.
    """
    if not (0 < m < n):
        raise ValueError("dense LP recipe expects 0 < m < n")
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((m, n))
    x_feas = rng.uniform(slack_scale, 1.0 + slack_scale, size=n)
    b = A @ x_feas
    y_aux = rng.standard_normal(m)
    s_aux = rng.uniform(slack_scale, 1.0 + slack_scale, size=n)
    c = A.T @ y_aux + s_aux
    meta = {
        "family": "lp_dense",
        "n": int(n),
        "m": int(m),
        "seed": int(seed),
        "x_feas": x_feas,
        "y_feas": y_aux,  # dual-feasible point: c - A'y_aux = s_aux > 0
    }
    return LPInstance(
        c=c, b=b, A_dense=A, L=MatrixLinOp.from_dense(A), name="lp_dense", meta=meta
    )


def from_netflow(nf: NetflowLP) -> LPInstance:
    """Wrap a NetflowLP as an LPInstance with a sparse-friendly LinOp."""
    L = IncidenceLinOp.from_arcs(nf.tail, nf.head, nf.n_nodes)
    meta = dict(nf.meta)
    meta["x_feas"] = nf.x_feas
    return LPInstance(
        c=nf.c,
        b=nf.b,
        A_dense=nf.incidence_dense(),
        L=L,
        name=str(nf.meta.get("family", "netflow")),
        meta=meta,
    )


# ---------------------------------------------------------------------------
# Certificate: LP relative duality gap (PDLP-style, jit-safe)
# ---------------------------------------------------------------------------


# lp_rel_gap now lives in atlas.certify.residuals (imported above) so the
# generalized schemes terminate on it without importing the bench package.


def _fwd(A, x):
    return A.forward(x) if hasattr(A, "forward") else jnp.asarray(A) @ x


def _adj(A, u):
    return A.adjoint(u) if hasattr(A, "adjoint") else jnp.asarray(A).T @ u


def lp_gap_terms(x, u, c, A, b) -> dict:
    """The certificate's components as floats (reporting / tests)."""
    x = jnp.asarray(x)
    u = jnp.asarray(u)
    c = jnp.asarray(c)
    b = jnp.asarray(b)
    xp = jnp.maximum(x, 0.0)
    P = float(jnp.vdot(c, xp))
    D = float(-jnp.vdot(b, u))
    pres = float(jnp.linalg.norm(_fwd(A, xp) - b))
    slack = c + _adj(A, u)
    dres = float(jnp.linalg.norm(jnp.maximum(-slack, 0.0)))
    return {
        "primal_value": P,
        "dual_value": D,
        "gap_raw": P - D,  # = c'x + b'u; >= 0 on feasible pairs
        "primal_infeas": pres,
        "dual_infeas": dres,
        "cert": float(lp_rel_gap(x, u, c, A, b)),
    }


# ---------------------------------------------------------------------------
# Oracle: HiGHS via scipy.optimize.linprog, content-hash cached
# ---------------------------------------------------------------------------


def _lp_arrays(problem) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(c, A_dense, b) from an LPInstance or an lp_standard CompositeProblem."""
    if isinstance(problem, LPInstance):
        return (
            np.asarray(problem.c, dtype=np.float64),
            np.asarray(problem.A_dense, dtype=np.float64),
            np.asarray(problem.b, dtype=np.float64),
        )
    if isinstance(problem, CompositeProblem) and problem.name == "lp_standard":
        return (
            np.asarray(problem.data["c"], dtype=np.float64),
            np.asarray(problem.data["A_dense"], dtype=np.float64),
            np.asarray(problem.data["b"], dtype=np.float64),
        )
    raise NotImplementedError(
        "lp_cell.oracle expects an LPInstance or an 'lp_standard' CompositeProblem"
    )


def _content_key(c: np.ndarray, A: np.ndarray, b: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(c.tobytes())
    h.update(A.tobytes())
    h.update(b.tobytes())
    h.update(repr(A.shape).encode())
    h.update(b"scipy-linprog-highs")
    return h.hexdigest()[:32]


def _dual_from_marginals(marg, c, A, b, fun) -> np.ndarray:
    """Return y with the STANDARD sign convention (A'y <= c, b'y = c'x*).

    scipy's eqlin.marginals sign convention has flipped across versions;
    resolve it numerically: pick the sign with the smaller dual
    infeasibility and verify strong duality b'y ~= fun.
    """
    cands = [np.asarray(marg, dtype=np.float64), -np.asarray(marg, dtype=np.float64)]
    infeas = [float(np.linalg.norm(np.maximum(A.T @ y - c, 0.0))) for y in cands]
    y = cands[int(np.argmin(infeas))]
    scale = 1.0 + abs(float(fun))
    if min(infeas) > 1e-6 * (1.0 + float(np.linalg.norm(c))):
        raise RuntimeError("HiGHS duals fail dual feasibility under both signs")
    if abs(float(b @ y) - float(fun)) > 1e-6 * scale:
        raise RuntimeError("HiGHS duals fail strong duality b'y == c'x*")
    return y


def oracle(problem, cache_dir: str = ORACLE_CACHE_LP_DIR) -> dict:
    """HiGHS reference solution of a standard-form LP, content-hash cached.

    Returns {"P_star", "x_star", "y_star", "primal_infeas", "cache_hit",
    "key"} with y_star in the standard dual sign (A'y <= c; the PDHG dual
    iterate corresponding to it is u = -y_star). Cached as .npz under
    `cache_dir` keyed by a content hash of (c, A, b).
    """
    c, A, b = _lp_arrays(problem)
    key = _content_key(c, A, b)
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"lp_{key}.npz")
    if os.path.exists(path):
        z = np.load(path)
        return {
            "P_star": float(z["P_star"]),
            "x_star": np.asarray(z["x_star"]),
            "y_star": np.asarray(z["y_star"]),
            "primal_infeas": float(z["primal_infeas"]),
            "cache_hit": True,
            "key": key,
        }

    res = linprog(c, A_eq=A, b_eq=b, bounds=(0, None), method="highs")
    if res.status != 0:
        raise RuntimeError(f"HiGHS failed (status {res.status}): {res.message}")
    y = _dual_from_marginals(res.eqlin.marginals, c, A, b, res.fun)
    pinf = float(np.linalg.norm(A @ res.x - b))
    np.savez_compressed(
        path, P_star=float(res.fun), x_star=res.x, y_star=y, primal_infeas=pinf
    )
    return {
        "P_star": float(res.fun),
        "x_star": np.asarray(res.x),
        "y_star": y,
        "primal_infeas": pinf,
        "cache_hit": False,
        "key": key,
    }


# ---------------------------------------------------------------------------
# Reference composite solve (numerical verification of the mapping ONLY)
# ---------------------------------------------------------------------------


def reference_solve_composite(
    problem: CompositeProblem,
    budget: Optional[Budget] = None,
    t: Optional[float] = None,
    s: Optional[float] = None,
    rho: Optional[float] = None,
) -> dict:
    """Solve an lp_standard composite by vanilla PDHG built ONLY from the
    composite's atoms (f.grad, g.prox, h.prox_conj, L.forward/adjoint).

    This function exists to verify the LP -> composite mapping numerically
    against the HiGHS oracle. It is NOT a scheme: the certified schemes in
    atlas/schemes/base_schemes.py still hard-code TV pieces, and hooking
    them to this mapping is the integrator's TODO (see module docstring).
    The step side condition IS typechecked (typecheck_condat_vu with
    L_f = f.props.lipschitz = 0, which reduces to the PDHG condition
    t*s*||A||^2 < 1), so the run carries a Level-1 cert.
    """
    if problem.name != "lp_standard":
        raise NotImplementedError("reference_solve_composite expects lp_standard")
    f, g, h, L = problem.f, problem.g, problem.h, problem.L
    c, b = problem.data["c"], problem.data["b"]
    if t is None:
        t = 0.99 / L.norm_bound
    if s is None:
        s = 0.99 / L.norm_bound
    L_f = f.props.lipschitz if f.props.lipschitz is not None else 0.0
    cert = typecheck_condat_vu(t=t, s=s, opnorm=L.norm_bound, L_smooth=L_f)
    if budget is None:
        budget = Budget(max_iters=1_000_000, tol=1e-8, sample_every=1_000)

    def T(state, k):
        x, u = state
        x_new = g.prox(x - t * f.grad(x) - t * L.adjoint(u), t)
        u_new = h.prox_conj(u + s * L.forward(2.0 * x_new - x), s)
        return (x_new, u_new)

    def metric_fn(state):
        x, u = state
        return lp_rel_gap(x, u, c, L, b)

    n = problem.shape[0]
    m = b.shape[0]
    state0 = (
        jnp.zeros(n, dtype=jnp.float64),
        jnp.zeros(m, dtype=jnp.float64),
    )
    state, iters, res_h, gap_h, wall = km_run(
        T, state0, budget, metric_fn, rho=rho, halpern=False
    )
    x, u = state
    return {
        "x": np.asarray(x),
        "u": np.asarray(u),
        "iters": int(iters),
        "fp_residual_history": res_h,
        "rel_gap_history": gap_h,
        "wall_time": wall,
        "cert": cert,
    }


# ---------------------------------------------------------------------------
# Held-out protocol (mirrors tv_cell.build_instances)
# ---------------------------------------------------------------------------

FAMILIES = ("dense", "grid", "regular")


def _make_instance(family: str, size: int, seed: int, degree: int = 4) -> LPInstance:
    if family == "dense":
        n = int(size)
        m = max(2, n // 2)
        return gen_lp_dense(n=n, m=m, seed=seed)
    if family == "grid":
        return from_netflow(gen_grid_flow(k=int(size), seed=seed))
    if family == "regular":
        return from_netflow(
            gen_regular_flow(n_nodes=int(size), seed=seed, degree=int(degree))
        )
    raise ValueError(f"unknown LP family {family!r}; choose from {FAMILIES}")


def build_instances(
    n_train: int, n_holdout: int, family: str, size: int, seed: int,
    degree: int = 4,
) -> tuple[list, list]:
    """(train, holdout) lists of lp_standard CompositeProblems.

    Mirrors tv_cell: deterministic in (family, size, seed); instance i uses
    derived seed seed*10007 + i, train = first n_train, holdout = the next
    n_holdout (disjoint seeds, same family/size).

    family: "dense" (size = n vars, m = n//2, v1 gen_lp recipe),
            "grid" (size = grid side k, netflow),
            "regular" (size = n_nodes, degree-`degree` netflow; default 4.
            NOTE: the Stage-0 constant atlas.genome.OPNORM_LP is sound only
            for degree=2; for anything else gate with
            validate(..., lp_opnorm=gate_opnorm(train + holdout))).
    """
    problems = [
        to_composite(_make_instance(family, size, seed * 10007 + i, degree=degree))
        for i in range(n_train + n_holdout)
    ]
    return problems[:n_train], problems[n_train:]


def gate_opnorm(problems) -> float:
    """Sound Stage-0 gate operator norm for a concrete LP instance set: the
    max power-iteration norm_bound over the instances (each bound already
    carries the 1.02 safety factor). Gate acceptance under this value
    implies instance-level typecheck acceptance for every instance in
    `problems`. Accepts LPInstances or lp_standard CompositeProblems."""
    bounds = []
    for p in problems:
        if isinstance(p, LPInstance):
            bounds.append(float(p.L.norm_bound))
        elif isinstance(p, CompositeProblem) and p.name == "lp_standard":
            bounds.append(float(p.L.norm_bound))
        else:
            raise NotImplementedError(
                "gate_opnorm expects LPInstances or lp_standard CompositeProblems"
            )
    if not bounds:
        raise ValueError("gate_opnorm needs at least one instance")
    return max(bounds)
