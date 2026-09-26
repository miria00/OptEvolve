"""Certificate stack: Level-1 construction certs, wrapper combination
rules, and the Level-4 benchmark card."""
import json
import math

import numpy as np
import pytest

from atlas import (
    Budget,
    ConstructionCert,
    PDHGParams,
    TypeCheckError,
    solve_pdhg,
    tv_denoise_problem,
    typecheck_condat_vu_solve,
    typecheck_fbs_solve,
    typecheck_pdhg_solve,
)
from atlas.certify.residuals import benchmark_card
from atlas.schemes.base_schemes import CondatVuParams, FBSParams
from atlas.typing_.rules import (
    typecheck_condat_vu,
    typecheck_drs,
    typecheck_fbs,
    typecheck_halpern,
    typecheck_pdhg,
    typecheck_relaxation,
)


def test_fbs_cert_fields():
    cert = typecheck_fbs(L_smooth=8.0, step=0.2)
    assert isinstance(cert, ConstructionCert)
    assert cert.scheme == "FBS"
    assert cert.satisfied
    # alpha = 2/(4 - a L) = 2/(4 - 1.6) = 0.8333...
    assert cert.averagedness == pytest.approx(2.0 / (4.0 - 1.6))
    assert 0.5 <= cert.averagedness < 1.0
    assert "2.7" in cert.citation and "Baillon-Haddad" in cert.citation
    d = cert.to_dict()
    json.dumps(d)
    assert d["params"]["step"] == 0.2


def test_drs_cert_any_positive_step():
    for a in (1e-3, 1.0, 50.0):
        cert = typecheck_drs(a)
        assert cert.satisfied and cert.averagedness == 0.5
        assert "2.7.1" in cert.citation
    with pytest.raises(TypeCheckError):
        typecheck_drs(-1.0)


def test_drs_cert_rejects_nonfinite_step():
    """'any a > 0' means any REAL a > 0: inf/nan are not valid resolvent
    parameters and must fail the MANDATORY Level-1 gate (regression: inf
    used to pass `step > 0` and produce a satisfied cert)."""
    for bad in (float("inf"), float("nan")):
        with pytest.raises(TypeCheckError):
            typecheck_drs(bad)


def test_pdhg_cert_boundary():
    opnorm = math.sqrt(8.0)
    cert = typecheck_pdhg(t=0.5, s=0.99 / 4.0, opnorm=opnorm)
    assert cert.satisfied and cert.averagedness == 0.5
    with pytest.raises(TypeCheckError):
        typecheck_pdhg(t=0.5, s=0.25, opnorm=opnorm)  # t*s*8 == 1 exactly
    with pytest.raises(TypeCheckError):
        typecheck_pdhg(t=-0.5, s=0.1, opnorm=opnorm)


def test_condat_vu_cert_reduces_to_pdhg():
    """With L_f = 0 the Condat-Vu condition/averagedness match PDHG."""
    opnorm = math.sqrt(8.0)
    cert = typecheck_condat_vu(t=0.5, s=0.99 / 4.0, opnorm=opnorm, L_smooth=0.0)
    assert cert.satisfied and cert.averagedness == pytest.approx(0.5)
    cert2 = typecheck_condat_vu(t=0.5, s=0.15, opnorm=opnorm, L_smooth=1.0)
    assert cert2.satisfied
    assert 0.5 < cert2.averagedness < 1.0  # smooth term costs averagedness
    with pytest.raises(TypeCheckError):
        typecheck_condat_vu(t=1.0, s=0.2, opnorm=opnorm, L_smooth=1.0)


def test_relaxation_combination_rule():
    base = typecheck_drs(1.0)  # alpha = 1/2
    relaxed = typecheck_relaxation(base, rho=1.5)
    assert relaxed.averagedness == pytest.approx(0.75)
    assert relaxed.scheme == "DRS+relax"
    with pytest.raises(TypeCheckError):
        typecheck_relaxation(base, rho=2.0)  # rho must be < 2
    tight = typecheck_fbs(L_smooth=8.0, step=1.9 / 8.0)  # alpha ~ 0.952
    with pytest.raises(TypeCheckError):
        typecheck_relaxation(tight, rho=1.1)  # rho*alpha > 1


def test_halpern_requires_nonexpansive():
    base = typecheck_drs(1.0)
    cert = typecheck_halpern(base)
    assert "12.2" in cert.citation and cert.scheme == "DRS+halpern"


def test_solve_level_typechecks(tv_instance_32):
    y, lam = tv_instance_32
    p = tv_denoise_problem(y, lam)
    c1 = typecheck_fbs_solve(p, FBSParams(step=0.2))
    assert c1.params["L_smooth"] == pytest.approx(8.0)
    c2 = typecheck_pdhg_solve(p, PDHGParams(t=0.5, s=0.99 / 4.0))
    assert c2.satisfied
    c3 = typecheck_condat_vu_solve(p, CondatVuParams(t=0.5, s=0.1))
    assert c3.params["L_smooth"] == 1.0  # from f's PropSet
    with pytest.raises(TypeCheckError):
        typecheck_fbs_solve(p, FBSParams(step=0.3))


def test_benchmark_card(tv_instance_32):
    y, lam = tv_instance_32
    p = tv_denoise_problem(y, lam)
    tol = 1e-4
    res = solve_pdhg(
        p, PDHGParams(t=0.5, s=0.99 / 4.0), Budget(max_iters=100_000, tol=tol, sample_every=100)
    )
    card = benchmark_card(res, p, tol=tol)
    s = json.dumps(card)  # serializable
    card2 = json.loads(s)
    assert card2["instance"]["shape"] == [32, 32]
    assert card2["iterations"] == res.iters
    assert card2["final_rel_gap"] <= tol
    assert card2["cert"]["scheme"] == "PDHG"
    assert card2["hardware"]["jax_backend"] == "cpu"
    assert card2["hardware"]["x64"] is True
    assert card2["wall_time_s"] > 0.0
