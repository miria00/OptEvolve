"""Tests for the LP cell: composite mapping, generators, oracle, certificate.

NOTE (integration): atlas/schemes/base_schemes.py still hard-codes TV
pieces, so the mapping-correctness test solves the composite with
lp_cell.reference_solve_composite (PDHG built ONLY from the composite's
atoms, side condition typechecked). Hooking the certified schemes to the
lp_standard composite is the integrator's TODO (see lp_cell docstring).
"""
import numpy as np
import pytest
import scipy.linalg

from atlas.bench.lp_cell import (
    AffineEqualitySet,
    IncidenceLinOp,
    LinearCost,
    MatrixLinOp,
    NonnegOrthant,
    build_instances,
    from_netflow,
    gen_lp_dense,
    lp_gap_terms,
    lp_rel_gap,
    oracle,
    reference_solve_composite,
    to_composite,
)
from atlas.bench.netflow import gen_grid_flow, gen_regular_flow

import jax.numpy as jnp


@pytest.fixture()
def rng():
    """Local RNG: shadows conftest's SESSION-scoped rng so this module does
    not consume draws from the stream other test modules rely on."""
    return np.random.default_rng(1234)


# ---------------------------------------------------------------------------
# 1. Composite mapping: atom derivations (numerical verification)
# ---------------------------------------------------------------------------


def test_prox_conj_affine_equality_is_translation(rng):
    """prox_{s h*}(v) = v - s b for h = indicator{z=b}: verified as the
    argmin of u -> s<b,u> + (1/2)||u - v||^2 against random perturbations,
    and against the Moreau identity with prox_h = const b."""
    m, s = 9, 0.37
    b = rng.standard_normal(m)
    v = rng.standard_normal(m)
    atom = AffineEqualitySet(b=jnp.asarray(b))
    u_star = np.asarray(atom.prox_conj(jnp.asarray(v), s))
    np.testing.assert_allclose(u_star, v - s * b, rtol=0, atol=1e-14)

    def prox_obj(u):
        return s * (b @ u) + 0.5 * np.sum((u - v) ** 2)

    for _ in range(100):
        pert = u_star + 1e-3 * rng.standard_normal(m)
        assert prox_obj(u_star) <= prox_obj(pert) + 1e-15
    # Moreau: prox_{s h*}(v) = v - s * prox_{h/s}(v/s); prox of I_{z=b} is b
    moreau = v - s * np.asarray(atom.prox(jnp.asarray(v / s), 1.0 / s))
    np.testing.assert_allclose(u_star, moreau, rtol=0, atol=1e-14)


def test_linear_cost_atom(rng):
    n, t = 6, 0.21
    c = rng.standard_normal(n)
    x = rng.standard_normal(n)
    atom = LinearCost(c=jnp.asarray(c))
    np.testing.assert_allclose(np.asarray(atom.grad(jnp.asarray(x))), c)
    # prox of the affine f is the exact translation v - t c
    np.testing.assert_allclose(
        np.asarray(atom.prox(jnp.asarray(x), t)), x - t * c, atol=1e-14
    )
    assert atom.props.lipschitz == 0.0  # constant gradient: L_f = 0


def test_nonneg_orthant_prox(rng):
    v = rng.standard_normal(11)
    out = np.asarray(NonnegOrthant().prox(jnp.asarray(v), 0.5))
    np.testing.assert_allclose(out, np.maximum(v, 0.0))


# ---------------------------------------------------------------------------
# 2. Linear operators: adjoint exactness + norm bound validity
# ---------------------------------------------------------------------------


def test_matrix_linop_adjoint_and_norm_bound(rng):
    A = rng.standard_normal((13, 29))
    L = MatrixLinOp.from_dense(A)
    x = rng.standard_normal(29)
    u = rng.standard_normal(13)
    lhs = float(jnp.vdot(L.forward(jnp.asarray(x)), jnp.asarray(u)))
    rhs = float(jnp.vdot(jnp.asarray(x), L.adjoint(jnp.asarray(u))))
    assert abs(lhs - rhs) <= 1e-10 * max(1.0, abs(lhs))
    exact = np.linalg.norm(A, 2)
    assert L.norm_bound >= exact  # must UPPER-bound for the side conditions
    assert L.norm_bound <= 1.05 * exact  # and stay tight


def test_incidence_linop_matches_dense_and_norm_bound(rng):
    nf = gen_regular_flow(n_nodes=12, seed=5)
    L = IncidenceLinOp.from_arcs(nf.tail, nf.head, nf.n_nodes)
    A = nf.incidence_dense()
    x = rng.standard_normal(nf.n_arcs)
    y = rng.standard_normal(nf.n_nodes)
    np.testing.assert_allclose(np.asarray(L.forward(jnp.asarray(x))), A @ x, atol=1e-12)
    np.testing.assert_allclose(np.asarray(L.adjoint(jnp.asarray(y))), A.T @ y, atol=1e-12)
    exact = np.linalg.norm(A, 2)
    assert exact <= L.norm_bound <= 1.05 * exact


# ---------------------------------------------------------------------------
# 3. Mapping correctness: composite solve matches HiGHS on a 20x40 LP
# ---------------------------------------------------------------------------


def test_composite_solve_matches_highs_20x40(tmp_path):
    inst = gen_lp_dense(n=40, m=20, seed=3)
    orc = oracle(inst, cache_dir=str(tmp_path))
    problem = to_composite(inst)
    assert problem.name == "lp_standard"
    out = reference_solve_composite(problem)
    # Level-1 cert present: Condat-Vu with L_f = 0 (= PDHG side condition)
    assert out["cert"].satisfied and out["cert"].scheme == "CondatVu"
    P = float(np.maximum(out["x"], 0.0) @ inst.c)
    assert abs(P - orc["P_star"]) <= 1e-6 * max(1.0, abs(orc["P_star"]))
    # terminated on the certificate, not the iteration cap
    assert out["rel_gap_history"][-1] <= 1e-8
    assert out["iters"] < 1_000_000


# ---------------------------------------------------------------------------
# 4. Generator determinism + feasibility by construction
# ---------------------------------------------------------------------------


def test_gen_lp_dense_deterministic():
    a = gen_lp_dense(30, 12, seed=11)
    b = gen_lp_dense(30, 12, seed=11)
    c = gen_lp_dense(30, 12, seed=12)
    np.testing.assert_array_equal(a.A_dense, b.A_dense)
    np.testing.assert_array_equal(a.c, b.c)
    np.testing.assert_array_equal(a.b, b.b)
    assert not np.array_equal(a.A_dense, c.A_dense)


@pytest.mark.parametrize("gen,kw", [
    (gen_grid_flow, {"k": 4}),
    (gen_regular_flow, {"n_nodes": 12}),
])
def test_netflow_deterministic(gen, kw):
    a = gen(seed=7, **kw)
    b = gen(seed=7, **kw)
    c = gen(seed=8, **kw)
    np.testing.assert_array_equal(a.tail, b.tail)
    np.testing.assert_array_equal(a.head, b.head)
    np.testing.assert_array_equal(a.c, b.c)
    np.testing.assert_array_equal(a.b, b.b)
    assert not (np.array_equal(a.c, c.c) and np.array_equal(a.b, c.b))


@pytest.mark.parametrize("nf", [gen_grid_flow(k=4, seed=1), gen_regular_flow(n_nodes=12, seed=1)])
def test_netflow_feasible_and_bounded_by_construction(nf):
    A = nf.incidence_dense()
    np.testing.assert_allclose(A @ nf.x_feas, nf.b, atol=1e-12)  # feasible
    assert np.all(nf.x_feas >= 0.0)
    assert np.all(nf.c >= 1.0) and np.allclose(nf.c, np.round(nf.c))  # bounded
    assert abs(nf.b.sum()) <= 1e-12  # balanced supplies (rows of A sum to 0)


def test_regular_graph_is_simple_and_regular():
    nf = gen_regular_flow(n_nodes=16, seed=2, degree=4)
    assert nf.n_arcs == 16 * 4
    assert np.all(nf.tail != nf.head)  # no self-loops
    deg_out = np.bincount(nf.tail, minlength=16)
    deg_in = np.bincount(nf.head, minlength=16)
    assert np.all(deg_out == 4) and np.all(deg_in == 4)
    keys = nf.tail * 16 + nf.head
    assert np.unique(keys).size == keys.size  # no duplicate arcs


# ---------------------------------------------------------------------------
# 5. Certificate: gap nonnegativity on random FEASIBLE pairs
# ---------------------------------------------------------------------------


def test_gap_nonnegative_on_random_feasible_points(rng):
    """Weak duality: c'x + b'u >= 0 for primal-feasible x and dual-feasible
    u = -y (A'y <= c), sampled around the constructed feasible points."""
    inst = gen_lp_dense(n=40, m=20, seed=21)
    A, c, b = inst.A_dense, inst.c, inst.b
    x_feas = inst.meta["x_feas"]  # >= slack_scale = 0.1 componentwise
    y_feas = inst.meta["y_feas"]  # slack c - A'y_feas >= 0.1 componentwise
    N = scipy.linalg.null_space(A)  # (n, n-m)
    for _ in range(50):
        # random primal-feasible point: x_feas + null-space step, kept >= 0
        w = rng.standard_normal(N.shape[1])
        step = N @ w
        step *= 0.05 / max(np.abs(step).max(), 1e-12)
        x = x_feas + step
        assert np.all(x >= 0.0)
        # random dual-feasible point: keep slack c - A'y >= 0
        d = rng.standard_normal(A.shape[0])
        atd = A.T @ d
        d *= 0.05 / max(np.abs(atd).max(), 1e-12)
        y = y_feas + d
        assert np.all(c - A.T @ y >= 0.0)
        terms = lp_gap_terms(x, -y, c, A, b)
        assert terms["gap_raw"] >= -1e-9
        assert terms["primal_infeas"] <= 1e-9
        assert terms["dual_infeas"] <= 1e-12


def test_gap_nonnegative_netflow_with_oracle_dual(tmp_path):
    nf = gen_grid_flow(k=4, seed=9)
    inst = from_netflow(nf)
    orc = oracle(inst, cache_dir=str(tmp_path))
    # (x_feas, -y_star) is a feasible pair up to HiGHS tolerances
    terms = lp_gap_terms(nf.x_feas, -orc["y_star"], inst.c, inst.A_dense, inst.b)
    assert terms["gap_raw"] >= -1e-7
    # certificate at the ORACLE pair is essentially zero
    cert = float(lp_rel_gap(orc["x_star"], -orc["y_star"], inst.c, inst.A_dense, inst.b))
    assert cert <= 1e-9


def test_lp_rel_gap_accepts_linop_and_dense(rng):
    inst = gen_lp_dense(n=20, m=8, seed=4)
    x = rng.standard_normal(20)
    u = rng.standard_normal(8)
    via_dense = float(lp_rel_gap(x, u, inst.c, inst.A_dense, inst.b))
    via_linop = float(lp_rel_gap(x, u, inst.c, inst.L, inst.b))
    assert abs(via_dense - via_linop) <= 1e-12 * max(1.0, via_dense)


# ---------------------------------------------------------------------------
# 6. Oracle: HiGHS + duals + content-hash cache
# ---------------------------------------------------------------------------


def test_oracle_dual_is_valid(tmp_path):
    inst = gen_lp_dense(n=30, m=12, seed=6)
    orc = oracle(inst, cache_dir=str(tmp_path))
    A, c, b = inst.A_dense, inst.c, inst.b
    y = orc["y_star"]
    assert np.max(A.T @ y - c) <= 1e-8  # dual feasible (standard sign)
    assert abs(b @ y - orc["P_star"]) <= 1e-8 * max(1.0, abs(orc["P_star"]))
    assert orc["primal_infeas"] <= 1e-8


def test_oracle_cache_hits(tmp_path):
    inst = gen_lp_dense(n=25, m=10, seed=13)
    first = oracle(inst, cache_dir=str(tmp_path))
    second = oracle(inst, cache_dir=str(tmp_path))
    assert first["cache_hit"] is False
    assert second["cache_hit"] is True
    assert second["P_star"] == first["P_star"]
    np.testing.assert_array_equal(second["x_star"], first["x_star"])
    np.testing.assert_array_equal(second["y_star"], first["y_star"])
    assert (tmp_path / f"lp_{first['key']}.npz").exists()
    # a DIFFERENT instance gets a different key (content hash, not filename)
    other = oracle(gen_lp_dense(n=25, m=10, seed=14), cache_dir=str(tmp_path))
    assert other["key"] != first["key"]
    # composite and instance hash identically
    via_composite = oracle(to_composite(inst), cache_dir=str(tmp_path))
    assert via_composite["cache_hit"] is True and via_composite["key"] == first["key"]


# ---------------------------------------------------------------------------
# 7. Held-out protocol (mirrors tv_cell)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("family,size", [("dense", 30), ("grid", 4), ("regular", 12)])
def test_build_instances_protocol(family, size):
    train, hold = build_instances(3, 2, family=family, size=size, seed=42)
    train2, hold2 = build_instances(3, 2, family=family, size=size, seed=42)
    assert len(train) == 3 and len(hold) == 2
    for p in train + hold:
        assert p.name == "lp_standard"
        assert p.shape == (int(np.asarray(p.data["c"]).shape[0]),)
    # deterministic across calls
    for a, b in zip(train + hold, train2 + hold2):
        np.testing.assert_array_equal(np.asarray(a.data["c"]), np.asarray(b.data["c"]))
        np.testing.assert_array_equal(np.asarray(a.data["b"]), np.asarray(b.data["b"]))
    # train and holdout instances are distinct (disjoint derived seeds)
    assert not np.array_equal(
        np.asarray(train[0].data["c"]), np.asarray(hold[0].data["c"])
    )
