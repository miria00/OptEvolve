"""Entropic (Bregman) PDHG: gate, invariants, end-to-end certificate against POT."""
import dataclasses
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)
MM = Path(__file__).resolve().parents[1] / "data" / "maros_meszaros"


def _ot(m=6, n=5, seed=0, declare=True):
    from atlas.bench import ot
    prob = ot.transport_composite(m, n, seed)
    if declare:
        mass = float(np.asarray(prob.data["b"])[:m].sum())
        meta = {**prob.data["meta"], "m": m, "n": n}
        prob = dataclasses.replace(prob, data={**prob.data, "simplex_mass": mass, "meta": meta})
    return prob


def test_gate_rejects_the_orthant_and_the_qp():
    from atlas.schemes.base_schemes import PDHGParams, build_scheme
    from atlas.typing_ import TypeCheckError
    with pytest.raises(TypeCheckError, match="not strongly convex"):
        build_scheme(_ot(declare=False), "pdhg_entropic", PDHGParams(t=0.1, s=0.1))
    if (MM / "HS21.mat").exists():
        from atlas.bench import qp_cell
        with pytest.raises(TypeCheckError):
            build_scheme(qp_cell.load(str(MM / "HS21.mat")), "pdhg_entropic", PDHGParams(t=0.1, s=0.1))
    with pytest.raises(TypeCheckError):  # t*s*2 = 1.0 is not < 1
        build_scheme(_ot(), "pdhg_entropic", PDHGParams(t=1.0, s=0.5))


def test_step_preserves_mass_and_positivity():
    from atlas.schemes.base_schemes import PDHGParams, build_scheme
    prob = _ot()
    built = build_scheme(prob, "pdhg_entropic", PDHGParams(t=0.3, s=0.9 / (2 * 0.3)))
    mass = prob.data["simplex_mass"]
    state = built.state0
    assert abs(float(state[0].sum()) - mass) < 1e-12 * mass
    for k in range(50):
        state = built.T(state, k, built.dyn)
        assert float(state[0].min()) > 0.0
        assert abs(float(state[0].sum()) - mass) < 1e-12 * max(1.0, mass)


def test_entropic_pdhg_certifies_small_ot_and_matches_network_simplex():
    ot = pytest.importorskip("ot")
    from atlas.certify.residuals import lp_rel_gap
    from atlas.schemes.base_schemes import Budget, PDHGParams, build_scheme, run_built
    m, n = 6, 5
    prob = _ot(m, n, seed=3)
    b = np.asarray(prob.data["b"]); a_, bt = b[:m], -b[m:]
    C = np.asarray(prob.data["c"]).reshape(m, n)
    ref = float(ot.emd2(a_, bt, C, numItermax=10**7))
    t = 0.5 / np.sqrt(2.0); s = 0.98 / (2.0 * t)
    params = PDHGParams(t=t, s=s)
    built = build_scheme(prob, "pdhg_entropic", params)
    r = run_built(built, built.cert, params, Budget(max_iters=200000, tol=1e-6, sample_every=1000))
    gap = float(lp_rel_gap(jnp.asarray(r.x), jnp.asarray(r.u), prob.data["c"], prob.L, prob.data["b"]))
    assert r.iters < 200000 and gap <= 1e-6, (r.iters, gap)
    assert abs(float(np.asarray(prob.data["c"]) @ r.x) - ref) <= 1e-4 * max(1.0, abs(ref))
