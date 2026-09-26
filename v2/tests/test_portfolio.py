"""Portfolio fitting: one route when one candidate dominates, a feature split when the domain needs two."""
import math


def _feats(ps):
    return {f"i{k}": {"log10_p": math.log10(p), "lam_frac": 0.1, "alpha": 0.5} for k, p in enumerate(ps)}


def test_single_route_when_one_candidate_dominates():
    from atlas.evolve.portfolio import fit_portfolio, dispatch
    feats = _feats([54, 2000, 5000, 54])
    T = {i: {"A": 1.0, "B": 2.0} for i in feats}
    out = fit_portfolio(T, feats)
    assert out["members"] == ["A"] and out["rule"] == {"leaf": "A"}


def test_split_on_p_when_the_domain_needs_two_routes():
    from atlas.evolve.portfolio import fit_portfolio, dispatch
    feats = _feats([54, 60, 2000, 5000])
    T = {"i0": {"A": 0.1, "B": 1.0}, "i1": {"A": 0.1, "B": 1.0}, "i2": {"A": 2.0, "B": 0.2}, "i3": {"A": float("inf"), "B": 0.3}}
    out = fit_portfolio(T, feats)
    assert set(out["members"]) == {"A", "B"} and out["rule"]["feature"] == "log10_p"
    assert dispatch(out["rule"], {"log10_p": math.log10(54), "lam_frac": 0.1, "alpha": 0.5}) == "A"
    assert dispatch(out["rule"], {"log10_p": math.log10(4000), "lam_frac": 0.1, "alpha": 0.5}) == "B"
    assert out["tune_geo_dispatch"] <= out["tune_geo_single"]
