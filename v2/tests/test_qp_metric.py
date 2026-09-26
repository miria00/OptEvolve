"""Metric gene on the convex QP cell: rescaling maps and original-problem certificate."""
from pathlib import Path

import numpy as np
import pytest

DATA = Path(__file__).resolve().parents[1] / "data" / "maros_meszaros"
pytestmark = pytest.mark.skipif(not (DATA / "HS21.mat").exists(), reason="Maros-Meszaros not downloaded")


def _osqp(d, eps=1e-10):
    osqp = pytest.importorskip("osqp")
    import scipy.sparse as sp
    s = osqp.OSQP()
    s.setup(P=sp.triu(d["P"]).tocsc(), q=d["q"], A=d["A"], l=d["l"], u=d["u"], verbose=False,
            eps_abs=eps, eps_rel=eps, max_iter=200000, polishing=True)
    r = s.solve()
    return r.x, r.y


@pytest.mark.parametrize("name", ["HS21", "DUALC1"])
@pytest.mark.parametrize("kind,kw", [("ruiz", {"iters": 10}), ("ruiz_pc", {"iters": 10, "alpha": 1.0})])
def test_rescaled_optimum_maps_back(name, kind, kw):
    from atlas.bench import qp_cell
    from atlas.certify.residuals import qp_rel_kkt
    from atlas.schemes.rescale import rescale_qp_problem
    d = qp_cell.read_mm(str(DATA / f"{name}.mat")); prob = qp_cell.to_composite(d)
    x, y = _osqp(d)
    rs, d1, d2, nb = rescale_qp_problem(prob, kind, **kw)
    assert np.all(d1 > 0) and np.all(d2 > 0)
    # the original optimum, mapped into scaled coordinates (x' = x/d2, y' = y/d1), is optimal there
    c = float(qp_rel_kkt(x / d2, y / d1, rs.f.P, rs.f.q, rs.L, rs.h.lo, rs.h.hi))
    assert c < 1e-6, c
    # infinite bounds stay infinite, finite ones scale by d1
    assert np.array_equal(np.isinf(np.asarray(rs.h.hi)), np.isinf(d["u"]))
    if kind == "ruiz_pc":
        assert nb == pytest.approx(1.0, abs=1e-8)


def test_ruiz_condat_vu_certifies_original_problem():
    from atlas.backend.hardware import prepare_base
    from atlas.bench import qp_cell
    from atlas.certify.residuals import qp_rel_kkt
    from atlas.genome import Backend, Genome, GenomeParams, Metric
    from atlas.schemes.base_schemes import Budget, run_built
    from atlas.schemes.rescale import rescale_qp_problem
    d = qp_cell.read_mm(str(DATA / "HS21.mat")); prob = qp_cell.to_composite(d)
    rs = rescale_qp_problem(prob, "ruiz", iters=10)[0]
    tau = 1.0 / float(rs.f.props.lipschitz); sigma = 0.49 / (tau * float(rs.L.norm_bound) ** 2)
    g = Genome("condat_vu", GenomeParams(tau=tau, sigma=sigma), metric=Metric(kind="ruiz", iters=10),
               backend=Backend(precision="f64", K=64))
    built, c, params, _ = prepare_base(prob, g)
    assert "metric[ruiz]" in c.scheme and "recomputed" in c.condition_str
    r = run_built(built, c, params, Budget(max_iters=20000, tol=1e-6, sample_every=200), backend=g.backend)
    # result variables are ORIGINAL-problem variables: recheck on the original data directly
    kkt = float(qp_rel_kkt(np.asarray(r.x), np.asarray(r.u), prob.f.P, prob.f.q, prob.L, prob.h.lo, prob.h.hi))
    assert kkt <= 1e-6, kkt
    x_ref, _ = _osqp(d)
    obj = lambda x: 0.5 * x @ (d["P"] @ x) + d["q"] @ x + d["r"]
    assert abs(obj(np.asarray(r.x)) - obj(x_ref)) <= 1e-4 * max(1.0, abs(obj(x_ref)))
