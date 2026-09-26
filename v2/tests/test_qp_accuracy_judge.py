"""The per-row accuracy judge rejects points that the global KKT certificate passes."""
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.sparse as sp

jax.config.update("jax_enable_x64", True)
MM = Path(__file__).resolve().parents[1] / "data" / "maros_meszaros"
pytestmark = pytest.mark.skipif(not (MM / "BOYD2.mat").exists(), reason="Maros-Meszaros not downloaded")


def _judge(d, x, ref=None):
    from atlas.certify.residuals import qp_accuracy_judge
    return qp_accuracy_judge(x, d["P"], d["q"], d["r"], d["A"], d["l"], d["u"], ref_obj=ref, feas_tol=1e-4)


def test_boyd2_anderson_point_passes_kkt_but_is_infeasible_row_by_row():
    from atlas.bench import qp_cell
    from atlas.certify.residuals import qp_rel_kkt
    from atlas.evolve.kb_composer import Recipe, run_recipe
    d = qp_cell.read_mm(str(MM / "BOYD2.mat")); prob = qp_cell.to_composite(d)
    out = run_recipe(prob, Recipe("condat_vu", 0.98, anderson=5), tol=1e-4, max_evals=64, K=64)
    kkt = float(qp_rel_kkt(jnp.asarray(out["x"]), jnp.asarray(out["u"]), prob.f.P, prob.f.q, prob.L, prob.h.lo, prob.h.hi))
    assert kkt <= 1e-4                                     # the global certificate is fooled
    assert _judge(d, out["x"])["verdict"] == "infeasible"   # the per-row judge is not


def test_primalc8_point_passing_the_global_certificate_is_rejected():
    # Deterministic, independent of any solver trajectory: Clarabel's solution of PRIMALC8 moved by 1 in
    # its largest coordinate (3.3e4). Against the largest quantities of the problem that is 4.6e-5, so the
    # global OSQP-style certificate passes at 1e-4; against that variable's own bound row it is a violation
    # of 1, which the per-row judge reports.
    pytest.importorskip("clarabel")
    from atlas.bench import qp_cell
    from atlas.certify.residuals import qp_rel_kkt
    d = qp_cell.read_mm(str(MM / "PRIMALC8.mat")); prob = qp_cell.to_composite(d)
    x, y, status, obj = qp_cell.clarabel_reference(d, eps=1e-9)
    assert status == "Solved"
    kkt = lambda x_: float(qp_rel_kkt(jnp.asarray(x_), jnp.asarray(y), prob.f.P, prob.f.q, prob.L, prob.h.lo, prob.h.hi))
    assert kkt(x) <= 1e-8 and _judge(d, x, obj)["verdict"] == "solution"
    xb = x.copy(); xb[int(np.argmax(np.abs(x)))] += 1.0
    assert kkt(xb) <= 1e-4                                  # the global certificate is fooled
    assert _judge(d, xb, obj)["verdict"] == "infeasible"    # the per-row judge is not


@pytest.mark.parametrize("name", ["PRIMALC1", "PRIMALC2"])
def test_primalc1_primalc2_are_dual_infeasible_as_distributed(name):
    # Data provenance guard: as distributed (q in {-1, 0}, bounds at +-1e20) and loaded, Clarabel reports
    # these two instances dual infeasible and no solver in the head-to-head solves them, so no accuracy
    # claim may rest on them.
    pytest.importorskip("clarabel")
    from atlas.bench import qp_cell
    d = qp_cell.read_mm(str(MM / (name + ".mat")))
    assert qp_cell.clarabel_reference(d)[2] == "DualInfeasible"


@pytest.mark.parametrize("name", ["HS21", "DUALC1"])
def test_accurate_solutions_are_judged_solutions(name):
    osqp = pytest.importorskip("osqp")
    from atlas.bench import qp_cell
    d = qp_cell.read_mm(str(MM / f"{name}.mat"))
    s = osqp.OSQP(); s.setup(P=sp.triu(d["P"]).tocsc(), q=d["q"], A=d["A"], l=d["l"], u=d["u"], verbose=False,
                             eps_abs=1e-10, eps_rel=1e-10, max_iter=200000, polishing=True)
    x = s.solve().x
    ref = float(0.5 * x @ (d["P"] @ x) + d["q"] @ x + d["r"])
    assert _judge(d, x, ref)["verdict"] == "solution"
