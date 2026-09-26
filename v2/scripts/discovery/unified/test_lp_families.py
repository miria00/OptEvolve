"""Tests for lp_families (feasibility and boundedness by construction, structure the planner must see, the LP
adapters) and for the Netlib loader on two small problems.

Boundedness is tested as COMPACTNESS: HiGHS must return Optimal for the family's own cost and for random costs of both
signs, which fails on any instance with a recession direction. Netlib tests skip when data/netlib is not fetched
(python netlib.py).

    JAX_PLATFORMS=cpu /home/miria/jaxenv2/bin/python -m pytest -q test_lp_families.py
"""
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pytest
import scipy.sparse as sp

import lp_families as lf
import netlib
import uni

SMALL = {"transport": 6, "netflow_cap": 5, "mcf_joint": 4, "gub_packing": 12, "production": 8}
HARD = {"transport": {"spread": 0.05}, "netflow_cap": {"tightness": 1.0, "cost_max": 2},
        "mcf_joint": {"tightness": 1.0}, "gub_packing": {"tightness": 1.0, "correlation": 1.0, "gub": "eq"},
        "production": {"tightness": 1.0, "seasonality": 1.0, "hold_ratio": 1e-3}}


def _with_cost(d, q):
    return dict(d, q=np.asarray(q, dtype=float))


@pytest.mark.parametrize("family", lf.FAMILIES)
@pytest.mark.parametrize("knobs", ["default", "hard"])
def test_feasible_and_compact(family, knobs):
    kw = {} if knobs == "default" else HARD[family]
    for seed in (0, 1):
        d = lf.make_lp(family, SMALL[family], seed=seed, **kw)
        n = d["P"].shape[0]
        A = sp.csc_matrix(d["A"])
        assert d["P"].nnz == 0 and A.shape[1] == n and d["l"].shape == d["u"].shape == (A.shape[0],)
        assert np.all(d["l"] <= d["u"])
        assert lf.row_violation(d, d["x_feas"]) <= 1e-12, "construction point must be feasible"
        st, obj = netlib.highs_objective(d)
        assert st == "Optimal"
        assert obj <= float(d["q"] @ d["x_feas"]) + 1e-9 * max(1.0, abs(obj))
        rng = np.random.default_rng(seed)
        for _ in range(3):
            st_r, _ = netlib.highs_objective(_with_cost(d, rng.standard_normal(n)))
            assert st_r == "Optimal", f"{family} seed {seed}: random cost gave {st_r}, feasible set is not compact"


@pytest.mark.parametrize("family", lf.FAMILIES)
def test_deterministic(family):
    a, b = lf.make_lp(family, SMALL[family], seed=3), lf.make_lp(family, SMALL[family], seed=3)
    assert (sp.csc_matrix(a["A"]) != sp.csc_matrix(b["A"])).nnz == 0
    for k in ("q", "l", "u", "x_feas"):
        np.testing.assert_array_equal(a[k], b[k])


def test_knob_ranges():
    with pytest.raises(ValueError):
        lf.make_lp("netflow_cap", 4, tightness=1.5)
    with pytest.raises(ValueError):
        lf.make_lp("gub_packing", 4, correlation=-0.1)
    with pytest.raises(ValueError):
        lf.make_lp("nope", 4)


def test_difficulty_knobs_bind():
    """The tightness knobs must move the optimum: tight capacities cost more (or earn less) than loose ones."""
    loose, tight = lf.make_lp("production", 24, tightness=0.0), lf.make_lp("production", 24, tightness=1.0)
    assert netlib.highs_objective(tight)[1] > netlib.highs_objective(loose)[1] * (1 + 1e-6)
    loose, tight = lf.make_lp("gub_packing", 40, tightness=0.0), lf.make_lp("gub_packing", 40, tightness=1.0)
    assert netlib.highs_objective(tight)[1] > netlib.highs_objective(loose)[1] + 1e-6   # min -v'x: less value
    loose, tight = lf.make_lp("netflow_cap", 6, tightness=0.0), lf.make_lp("netflow_cap", 6, tightness=1.0)
    assert netlib.highs_objective(tight)[1] > netlib.highs_objective(loose)[1] * (1 + 1e-6)


def test_structure_seen_by_planner():
    """The registry's claims, on permuted encodings."""
    d = uni.permute_instance(lf.make_lp("transport", 6, aspect=1.5), 7)            # sinks 9 > sources 6
    p, _ = uni.plan_product(d)
    assert p["K"] == 6 and p["cover"] == 1.0
    d = uni.permute_instance(lf.make_lp("mcf_joint", 4, n_commodities=9), 7)        # K = 9 > grid node degree 8
    p, _ = uni.plan_product(d)
    assert p["cover"] == 1.0 and set(np.bincount(p["bid"]).tolist()) == {9}
    d = uni.permute_instance(lf.make_lp("gub_packing", 12, knap_density=1.0), 7)    # one dense knapsack row wins
    p, _ = uni.plan_product(d)
    assert p["K"] == 1 and p["cover"] == 1.0
    d = uni.permute_instance(lf.make_lp("gub_packing", 12, n_knap=1, knap_density=0.0), 7)
    p, _ = uni.plan_product(d)                    # the knapsack keeps one entry, which reads as a bound: GUB rows win
    assert p["cover"] == 1.0 and p["K"] == 12
    d = uni.permute_instance(lf.make_lp("production", 8, n_products=5, n_resources=2), 7)
    p, _ = uni.plan_product(d)
    assert p["K"] == 8 and abs(p["cover"] - 0.5) < 1e-12


def test_lp_adapters_run_and_decode():
    d = uni.permute_instance(lf.make_lp("transport", 6), 1)
    obj_ref, _ = uni.reference(d)
    assert obj_ref is not None
    dg = lf.lp_diagnose(d)
    assert set(dg) == {"top_row", "row_mass", "singular_gap", "fires"}
    # the reason lp_diagnose exists; if this starts passing, uni takes LPs directly and the adapter is redundant
    with pytest.raises(ValueError):
        uni.diagnose(d)
    plan, _ = uni.plan_product(d)
    for cand in ("std_ruiz", "reloc_rows"):
        form = lf.lp_formulation(d, cand, plan=plan)
        Lf, nb = form.prob.f.props.lipschitz, form.prob.L.norm_bound
        assert abs(Lf - nb) <= 1e-12 * nb                 # omega = 1: t = 1/||L||
        z = np.random.default_rng(0).standard_normal(d["P"].shape[0])
        assert form.decode(z).shape == z.shape
        r = uni.measure_ladder(form, d, obj_ref, tols=(1e-2,), cap=20000)
        assert r["N_at"]["0.01"] is not None, f"{cand} did not reach 1e-2"


# ----------------------------------------------------------------------------------------------------------------
# Netlib
# ----------------------------------------------------------------------------------------------------------------
HAVE_NETLIB = os.path.exists(os.path.join(netlib.NETLIB_DIR, "MANIFEST.json"))
needs_netlib = pytest.mark.skipif(not HAVE_NETLIB, reason="Netlib not fetched (python netlib.py)")


@needs_netlib
def test_readme_table():
    t = netlib.parse_readme(open(os.path.join(netlib.NETLIB_DIR, "readme"), encoding="latin-1").read())
    assert len(t) == 98
    assert t["afiro"]["minos"] == -4.6475314286e+02 and t["standgub"]["minos"] is None
    assert t["25fv47"]["minos_mips"] == 5.5018467791e+03 and t["25fv47"]["cplex_sparc"] is None


@needs_netlib
@pytest.mark.parametrize("name", ["afiro", "boeing1"])       # boeing1: RANGES, UP and LO bounds, empty rows
def test_netlib_small(name):
    d = netlib.load_netlib(name)
    A = sp.csc_matrix(d["A"])
    n = A.shape[1]
    pub, _ = netlib.published_optimum(name)
    assert d["P"].shape == (n, n) and d["P"].nnz == 0
    assert np.all(d["l"] <= d["u"]) and not np.any(np.isinf(d["l"]) & np.isinf(d["u"]))   # no free rows left
    assert np.all(np.diff(sp.csr_matrix(A).indptr) > 0)                                    # no empty rows left
    raw = netlib.read_mps_raw(name)
    nb = int(np.sum((raw.col_lower > -netlib.INF) | (raw.col_upper < netlib.INF)))
    assert d["meta"]["bound_rows"] == nb
    st, obj = netlib.highs_objective(d)
    # the published tables leave out the objective constant r (netlib docstring), and uni.reference returns q'x
    assert st == "Optimal" and abs(obj - d["r"] - pub) <= 1e-9 * max(1.0, abs(pub))
    obj_ref, _ = uni.reference(d)                         # Clarabel at 1e-9
    assert obj_ref is not None and abs(obj_ref - pub) <= 1e-6 * max(1.0, abs(pub))


@needs_netlib
def test_standgub_is_standata():
    a, b = netlib.load_netlib("standgub"), netlib.load_netlib("standata")
    assert (sp.csc_matrix(a["A"]) != sp.csc_matrix(b["A"])).nnz == 0
    np.testing.assert_array_equal(a["q"], b["q"])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
