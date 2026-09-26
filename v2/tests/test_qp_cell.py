"""QP atom, loader, certificate and an end-to-end certified solve on real
Maros-Meszaros instances."""
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

import atlas  # noqa: F401
from atlas.bench import qp_cell
from atlas.certify.residuals import qp_rel_kkt

DATA = Path(__file__).resolve().parents[1] / "data" / "maros_meszaros"
pytestmark = pytest.mark.skipif(not (DATA / "HS21.mat").exists(), reason="Maros-Meszaros data not present")


def _osqp_solution(d, eps):
    osqp = pytest.importorskip("osqp")
    s = osqp.OSQP()
    s.setup(P=sp.triu(d["P"]).tocsc(), q=d["q"], A=d["A"], l=d["l"], u=d["u"],
            verbose=False, eps_abs=eps, eps_rel=eps, max_iter=200000, polishing=True)
    r = s.solve()
    return np.asarray(r.x), np.asarray(r.y)


def _steps(prob):
    """Condat-Vu steps meeting t*L_f/2 + t*s*||A||^2 = 0.99 < 1."""
    Lf, nA = float(prob.f.props.lipschitz), float(prob.L.norm_bound)
    t = 1.0 / Lf
    return t, 0.49 / (t * nA ** 2)


def test_loader_builds_a_symmetric_qp_composite():
    d = qp_cell.read_mm(str(DATA / "HS21.mat"))
    assert abs(d["P"] - d["P"].T).max() == 0
    prob = qp_cell.to_composite(d)
    assert prob.name == "qp_standard" and prob.shape == (d["P"].shape[0],)
    x = np.linspace(-1, 1, prob.shape[0])
    np.testing.assert_allclose(np.asarray(prob.f.grad(x)), d["P"] @ x + d["q"], atol=1e-12)


def test_certificate_is_small_at_a_solution_and_large_away_from_it():
    for name in ("HS21.mat", "DUALC1.mat"):
        d = qp_cell.read_mm(str(DATA / name)); prob = qp_cell.to_composite(d)
        x, y = _osqp_solution(d, 1e-10)
        at = float(qp_rel_kkt(x, y, prob.f.P, prob.f.q, prob.L, prob.h.lo, prob.h.hi))
        off = float(qp_rel_kkt(x + 1.0, y, prob.f.P, prob.f.q, prob.L, prob.h.lo, prob.h.hi))
        assert at < 1e-6, (name, at)
        assert off > 1e-3, (name, off)


def test_pdhg_rejects_qp_and_condat_vu_admits_it():
    from atlas.backend.hardware import prepare_base
    from atlas.genome import Genome, GenomeParams
    from atlas.typing_ import TypeCheckError
    prob = qp_cell.load(str(DATA / "HS21.mat"))
    t, s = _steps(prob)
    with pytest.raises(TypeCheckError):
        prepare_base(prob, Genome("pdhg", GenomeParams(tau=t, sigma=s)))
    prepare_base(prob, Genome("condat_vu", GenomeParams(tau=t, sigma=s)))


def test_condat_vu_certifies_a_real_qp_and_matches_osqp():
    from atlas.backend.hardware import prepare_base
    from atlas.genome import Backend, Genome, GenomeParams
    from atlas.schemes.base_schemes import Budget, run_built
    d = qp_cell.read_mm(str(DATA / "HS21.mat")); prob = qp_cell.to_composite(d)
    t, s = _steps(prob)
    g = Genome("condat_vu", GenomeParams(tau=t, sigma=s), backend=Backend(precision="f64", K=64))
    built, cert, params, _ = prepare_base(prob, g)
    cap = 200000
    r = run_built(built, cert, params, Budget(max_iters=cap, tol=1e-6, sample_every=250), backend=g.backend)
    assert r.iters < cap, "did not certify within the cap"
    x_ref, _ = _osqp_solution(d, 1e-10)
    obj = lambda x: 0.5 * x @ (d["P"] @ x) + d["q"] @ x + d["r"]
    assert abs(obj(r.x) - obj(x_ref)) <= 1e-4 * max(1.0, abs(obj(x_ref)))
