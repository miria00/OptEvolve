"""Haugazeau outer wrapper: formula, metric, gate, genome, end-to-end certificate."""
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)
ROOT = Path(__file__).resolve().parents[1]
MM = ROOT / "data" / "maros_meszaros"


def _euclid_gram(p, q):
    return float(p @ p), float(p @ q), float(q @ q)


def test_q_formula_is_the_projection_onto_two_halfspaces():
    """Q(a,b,c) = P_{H(a,b) ∩ H(b,c)} a, checked against a brute-force QP."""
    from scipy.optimize import minimize
    from atlas.wrappers.km import haugazeau_coefficients
    rng = np.random.default_rng(3)
    for _ in range(40):
        a, b, c = rng.normal(size=(3, 4))
        pp, pq, qq = _euclid_gram(a - b, b - c)
        ca, cb, cc, _ = map(float, haugazeau_coefficients(pp, pq, qq))
        Q = ca * a + cb * b + cc * c
        cons = [{"type": "ineq", "fun": lambda z: -(z - b) @ (a - b)},
                {"type": "ineq", "fun": lambda z: -(z - c) @ (b - c)}]
        ref = minimize(lambda z: 0.5 * np.sum((z - a) ** 2), c, constraints=cons,
                       method="SLSQP", options={"ftol": 1e-14, "maxiter": 500}).x
        assert np.allclose(Q, ref, atol=1e-6), (Q, ref)


def _qp(name):
    from atlas.bench import qp_cell
    return qp_cell.load(str(MM / f"{name}.mat"))


def _built(prob, scheme, frac=0.99):
    from atlas.schemes.base_schemes import (CondatVuParams, PDHGParams, build_scheme)
    nb = float(prob.L.norm_bound)
    if scheme == "condat_vu":
        Lf = float(prob.f.props.lipschitz); t = 1.0 / Lf; s = frac * 0.5 / (t * nb ** 2)
        return build_scheme(prob, "condat_vu", CondatVuParams(t=t, s=s), check=True)
    t = 1.0 / nb; s = frac / (t * nb ** 2)
    return build_scheme(prob, "pdhg", PDHGParams(t=t, s=s), check=True)


@pytest.mark.skipif(not (MM / "HS21.mat").exists(), reason="Maros-Meszaros not downloaded")
def test_firmified_map_is_firmly_nonexpansive_in_the_scheme_metric():
    """F = I + (T - I)/(2 alpha) must satisfy ||Fa-Fb||_M^2 <= <a-b, Fa-Fb>_M."""
    built = _built(_qp("DUALC1"), "condat_vu")
    alpha = built.cert.averagedness
    assert 0.5 < alpha < 1.0
    rng = np.random.default_rng(0)
    x0, u0 = built.state0
    for _ in range(20):
        a = (jnp.asarray(rng.normal(size=x0.shape)), jnp.asarray(rng.normal(size=u0.shape)))
        b = (jnp.asarray(rng.normal(size=x0.shape)), jnp.asarray(rng.normal(size=u0.shape)))
        F = lambda z: jax.tree_util.tree_map(lambda s, t: s + (t - s) / (2 * alpha), z, built.T(z, 0, built.dyn))
        Fa, Fb = F(a), F(b)
        p = jax.tree_util.tree_map(jnp.subtract, Fa, Fb)
        q = jax.tree_util.tree_map(jnp.subtract, a, b)
        pp, pq, _ = map(float, built.gram(p, q, built.dyn))
        assert pp <= pq * (1 + 1e-9) + 1e-12, (pp, pq)


def test_gate_admits_saddle_schemes_and_rejects_the_rest():
    from atlas.typing_ import TypeCheckError
    from atlas.typing_.rules import typecheck_haugazeau, typecheck_pdhg
    base = typecheck_pdhg(t=0.5, s=0.5, opnorm=1.0)
    c = typecheck_haugazeau(base, "pdhg")
    assert c.satisfied and "Haugazeau" in c.citation and "+haugazeau" in c.scheme
    with pytest.raises(TypeCheckError):
        typecheck_haugazeau(base, "fbs")


def test_genome_kind_roundtrip_validate_and_legacy_hashes():
    from atlas.genome import Genome, GenomeParams, Wrapper, schema as S
    pinned = json.load(open(ROOT / "tests" / "data_legacy_genome_hashes.json"))
    for k, v in pinned.items():
        g = S.genome_from_dict(v["dict"])
        assert S.content_hash(g) == v["hash"], k          # legacy hashes unchanged
        assert "kind" not in S.genome_to_dict(g)["wrapper"]
    g = Genome("pdhg", GenomeParams(tau=0.1, sigma=0.1), wrapper=Wrapper(kind="haugazeau"))
    d = S.genome_to_dict(g)
    assert d["wrapper"]["kind"] == "haugazeau"
    g2 = S.genome_from_dict(d)
    assert g2 == g and S.content_hash(g2) == S.content_hash(g)
    assert S.content_hash(g) != S.content_hash(Genome("pdhg", GenomeParams(tau=0.1, sigma=0.1)))
    for bad in (Wrapper(kind="haugazeau", halpern=True), Wrapper(kind="nope")):
        assert S._structural_error(Genome("pdhg", GenomeParams(tau=0.3, sigma=0.5), wrapper=bad))
    assert S._structural_error(Genome("pdhg", GenomeParams(tau=0.3, sigma=0.5, rho=1.5),
                                      wrapper=Wrapper(kind="haugazeau")))
    assert S.validate(g, problem_kind="lp_netflow").ok
    assert not S.validate(Genome("fbs", GenomeParams(tau=0.1), wrapper=Wrapper(kind="haugazeau")),
                          problem_kind="tv_denoise").ok


def test_registry_matches_kb_and_schema():
    import atlas.kb as kb
    from atlas.genome import schema as S
    from atlas.wrappers import registry
    assert tuple(S.WRAPPER_KINDS) == registry.kinds()
    for kind, spec in registry.REGISTRY.items():
        entry = kb.by_id(spec["kb_id"])
        assert entry["genome"]["wrapper_kind"] == kind
        assert entry["implementation"] and entry["gate_rule"]


def test_proposer_draws_registry_wrappers_only_when_enabled():
    from atlas.evolve.random_backend import RandomBackend
    a = [g.wrapper for g in RandomBackend(seed=5, problem_kind="lp_netflow").ask(30)]
    b = [g.wrapper for g in RandomBackend(seed=5, problem_kind="lp_netflow", p_outer=0.0).ask(30)]
    assert a == b and all(w.kind is None for w in a)        # default draw sequence unchanged
    c = [g.wrapper for g in RandomBackend(seed=5, problem_kind="lp_netflow", p_outer=0.5).ask(40)]
    assert any(w.kind == "haugazeau" for w in c)


@pytest.mark.skipif(not (MM / "HS21.mat").exists(), reason="Maros-Meszaros not downloaded")
@pytest.mark.parametrize("backend", [None, "f64"])
def test_haugazeau_certifies_a_real_qp_on_both_runners(backend):
    from atlas.backend.hardware import prepare_base
    from atlas.certify.residuals import qp_rel_kkt
    from atlas.genome import Backend, Genome, GenomeParams, Wrapper
    from atlas.schemes.base_schemes import Budget, run_built
    prob = _qp("HS21")
    Lf, nb = float(prob.f.props.lipschitz), float(prob.L.norm_bound)
    t = 1.0 / Lf; s = 0.49 / (t * nb ** 2)
    be = None if backend is None else Backend(precision=backend, K=64)
    g = Genome("condat_vu", GenomeParams(tau=t, sigma=s), wrapper=Wrapper(kind="haugazeau"), backend=be)
    built, cert, params, _ = prepare_base(prob, g)
    assert "+haugazeau" in cert.scheme
    r = run_built(built, cert, params, Budget(max_iters=40000, tol=1e-6, sample_every=200), backend=be)
    kkt = float(qp_rel_kkt(np.asarray(r.x), np.asarray(r.u), prob.f.P, prob.f.q, prob.L, prob.h.lo, prob.h.hi))
    assert r.iters < 40000 and kkt <= 1e-6, (r.iters, kkt)
