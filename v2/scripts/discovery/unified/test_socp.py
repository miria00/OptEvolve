"""Tests for socp.py: exact projections against numpy references and against optimality conditions that do not reuse
the closed form, the conic conversion, the Clarabel reference on every generator, convergence of std_ruiz under the
cone-aware acceptance, and equality of the relocated and standard solutions on randomly permuted encodings.

Run: JAX_PLATFORMS=cpu python test_socp.py        (prints the iteration table)
 or: JAX_PLATFORMS=cpu python -m pytest -q test_socp.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import scipy.sparse as sp
import jax
import jax.numpy as jnp

import socp
import uni
from atlas.bench.conic_baseline_proc import run_conic_baseline_proc
from atlas.schemes.base_schemes import _prox_conj
from atlas.schemes.rescale import kkt_ruiz_equilibrate

PROJ_TOL = 1e-12


def rel_err(a, b, v=None):
    """max |a - b| / (1 + max(|b|, |v|)) with v the projection's INPUT. A projection is nonexpansive, so a one-ulp
    difference in a norm near a branch boundary moves the output by up to ulp * |v|; normalizing by the output alone
    reported 6e-8 on the polar boundary at scale 1e8, where the exact output is 0 and the input is 7e8."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if not b.size:
        return 0.0
    den = 1.0 + max(np.max(np.abs(b)), 0.0 if v is None or not np.size(v) else np.max(np.abs(v)))
    return float(np.max(np.abs(a - b)) / den)


# ----------------------------------------------------------------------------------------------------------------
# numpy references, written per cone with explicit branches
# ----------------------------------------------------------------------------------------------------------------
def np_soc(t, z):
    nz = np.linalg.norm(z)
    if nz <= t:
        return float(t), z.copy()
    if nz <= -t:
        return 0.0, np.zeros_like(z)
    a = 0.5 * (t + nz)
    return a, a * z / nz


def np_ball(z, c, r):
    dz = z - c
    nd = np.linalg.norm(dz)
    return z.copy() if nd <= r else c + r * dz / nd


def soc_cases(rng):
    """(t, z) pairs: generic, strictly inside, polar, both boundaries, zero vectors, size-1 cones, extreme scales."""
    out = []
    for dim in (0, 1, 2, 3, 7, 50):
        for scale in (1e-8, 1.0, 1e8):
            z = rng.standard_normal(dim) * scale
            nz = np.linalg.norm(z)
            out += [(rng.standard_normal() * scale, z), (nz * 1.5 + scale, z), (-nz * 1.5 - scale, z), (nz, z),
                    (-nz, z), (0.3 * nz, z), (-0.3 * nz, z)]
        out += [(0.0, np.zeros(dim)), (1.0, np.zeros(dim)), (-1.0, np.zeros(dim))]
    return out


# ----------------------------------------------------------------------------------------------------------------
# 1. projections
# ----------------------------------------------------------------------------------------------------------------
def _check_soc_single():
    rng = np.random.default_rng(0)
    worst = 0.0
    atom = socp.SOCProj()
    prox = jax.jit(lambda a, v: a.prox(v))
    for t, z in soc_cases(rng):
        tr, zr = np_soc(t, z)
        tj, zj = socp.project_soc(jnp.asarray(t), jnp.asarray(z))
        vj = np.asarray(prox(atom, jnp.asarray(np.concatenate([[t], z]))))
        ref = np.concatenate([[tr], zr])
        vin = np.concatenate([[t], z])
        worst = max(worst, rel_err(np.concatenate([[float(tj)], np.asarray(zj)]), ref, vin), rel_err(vj, ref, vin))
        # conditions that do not reuse the formula: Q is self-dual, so v = P(v) - P(-v) with P(v) orthogonal to P(-v)
        v = np.concatenate([[t], z])
        p = np.asarray(prox(atom, jnp.asarray(v)))
        pm = np.asarray(prox(atom, jnp.asarray(-v)))
        sc = 1.0 + np.max(np.abs(v))
        assert np.linalg.norm(p[1:]) <= p[0] + 1e-12 * sc
        assert np.max(np.abs(v - (p - pm))) <= 1e-12 * sc
        assert abs(p @ pm) <= 1e-12 * sc * sc
    assert worst <= PROJ_TOL, worst
    return worst


def test_soc_single():
    _check_soc_single()


def _random_product(rng, K, scale=1.0, shuffle=True):
    dims = rng.integers(1, 12, K)                                   # size 1 is the cone t >= 0
    bid = np.repeat(np.arange(K), dims)
    head = np.concatenate([np.arange(k) == 0 for k in dims])
    w = rng.standard_normal(bid.size) * scale
    for k in range(K):                                              # force every regime to appear
        sel = np.flatnonzero(bid == k)
        nz = np.linalg.norm(w[sel[1:]])
        w[sel[0]] = [1.5 * nz + 0.1, -1.5 * nz - 0.1, nz, -nz, 0.2 * nz, w[sel[0]]][k % 6]
        if k % 11 == 7:
            w[sel] = 0.0
    if shuffle:                                                     # segments need not be contiguous or head-first
        order = rng.permutation(bid.size)
        bid, head, w = bid[order], head[order], w[order]
    return w, bid, head


def _check_soc_product():
    rng = np.random.default_rng(1)
    worst = 0.0
    fn = jax.jit(socp.project_soc_product, static_argnums=3)
    for trial in range(30):
        K = int(rng.integers(1, 80))
        w, bid, head = _random_product(rng, K, scale=[1e-6, 1.0, 1e6][trial % 3], shuffle=trial % 2 == 1)
        p = np.asarray(fn(jnp.asarray(w), jnp.asarray(bid), jnp.asarray(head), K))
        ref = np.empty_like(w)
        for k in range(K):
            sel = np.flatnonzero(bid == k)
            hs, ts = sel[head[sel]], sel[~head[sel]]
            tr, zr = np_soc(w[hs[0]], w[ts])
            ref[hs[0]], ref[ts] = tr, zr
        worst = max(worst, rel_err(p, ref, w))
    assert worst <= PROJ_TOL, worst
    return worst


def test_soc_product():
    _check_soc_product()


def _check_shifted_signed_product_and_conjugate():
    """SOCProductProj with sign and offset: numpy reference, variational inequality against random feasible points,
    and the conjugate prox that Condat-Vu uses. For C = {v : sign v + c in Q} and s > 0, Moreau gives
    v - s P_C(v/s); independently, h* = support function of C, whose prox is sign (P_{-Q}(sign v + s c))."""
    rng = np.random.default_rng(2)
    worst = 0.0
    for trial in range(20):
        K = int(rng.integers(1, 30))
        w, bid, head = _random_product(rng, K, shuffle=False)
        sign = rng.choice([-1.0, 1.0], w.size)
        off = rng.standard_normal(w.size) * 2.0
        atom = socp.SOCProductProj(bid=jnp.asarray(bid), head=jnp.asarray(head), sign=jnp.asarray(sign),
                                   offset=jnp.asarray(off), K=K)
        p = np.asarray(jax.jit(lambda a, v: a.prox(v))(atom, jnp.asarray(w)))
        ref = np.empty_like(w)
        y = sign * w + off
        for k in range(K):
            sel = np.flatnonzero(bid == k)
            tr, zr = np_soc(y[sel[0]], y[sel[1:]])
            ref[sel] = sign[sel] * (np.concatenate([[tr], zr]) - off[sel])
        worst = max(worst, rel_err(p, ref, y))
        for _ in range(20):                                          # <v - p, x - p> <= 0 for every x in C
            kq = rng.standard_normal(w.size) * 3.0
            for k in range(K):
                sel = np.flatnonzero(bid == k)
                kq[sel[0]] = np.linalg.norm(kq[sel[1:]]) + abs(kq[sel[0]])
            x = sign * (kq - off)
            assert (w - p) @ (x - p) <= 1e-10 * (1.0 + np.abs(w).max()) ** 2
        s = float(rng.uniform(0.1, 10.0))
        conj = np.asarray(_prox_conj(atom, jnp.asarray(w), s))
        u = sign * w + s * off
        pq = np.empty_like(u)
        for k in range(K):
            sel = np.flatnonzero(bid == k)
            tr, zr = np_soc(-u[sel[0]], -u[sel[1:]])
            pq[sel] = -np.concatenate([[tr], zr])
        worst = max(worst, rel_err(conj, sign * pq, u))
    assert worst <= PROJ_TOL, worst
    return worst


def test_shifted_signed_product_and_conjugate():
    _check_shifted_signed_product_and_conjugate()


def _check_balls():
    rng = np.random.default_rng(3)
    worst = 0.0
    for dim in (1, 2, 5, 40):
        for scale in (1e-6, 1.0, 1e6):
            c = rng.standard_normal(dim) * scale
            r = abs(rng.standard_normal()) * scale
            dirn = rng.standard_normal(dim)
            dirn /= np.linalg.norm(dirn)
            for z in (c + 0.5 * r * dirn, c + r * dirn, c + 3.0 * r * dirn, c.copy(), rng.standard_normal(dim) * scale):
                worst = max(worst, rel_err(socp.project_l2_ball(jnp.asarray(z), jnp.asarray(c), r), np_ball(z, c, r)))
                atom = socp.L2BallProj(center=jnp.asarray(c), radius=jnp.asarray(r))
                worst = max(worst, rel_err(jax.jit(lambda a, v: a.prox(v))(atom, jnp.asarray(z)), np_ball(z, c, r)))
            worst = max(worst, rel_err(socp.project_l2_ball(jnp.asarray(c), jnp.asarray(c), 0.0), c))
    for trial in range(20):                                          # product of balls of different sizes
        K = int(rng.integers(1, 40))
        dims = rng.integers(1, 10, K)
        bid = np.repeat(np.arange(K), dims)
        cen = rng.standard_normal(bid.size)
        rad = np.abs(rng.standard_normal(K)) * rng.choice([0.0, 1.0, 5.0], K)
        w = cen + rng.standard_normal(bid.size) * rng.choice([0.1, 1.0, 10.0])
        atom = socp.BallProductProj(bid=jnp.asarray(bid), center=jnp.asarray(cen), radius=jnp.asarray(rad), K=K)
        p = np.asarray(jax.jit(lambda a, v: a.prox(v))(atom, jnp.asarray(w)))
        ref = np.concatenate([np_ball(w[bid == k], cen[bid == k], rad[k]) for k in range(K)])
        worst = max(worst, rel_err(p, ref))
    assert worst <= PROJ_TOL, worst
    return worst


def test_balls():
    _check_balls()


def test_slice_product():
    rng = np.random.default_rng(4)
    w, bid, head = _random_product(rng, 7, shuffle=False)
    lo, hi = -rng.random(5), rng.random(5)
    v = rng.standard_normal(5 + w.size) * 2
    atom = socp._slices([(socp.Box(lo=jnp.asarray(lo), hi=jnp.asarray(hi)), 5),
                         (socp.SOCProductProj(bid=jnp.asarray(bid), head=jnp.asarray(head), sign=jnp.ones(w.size),
                                              offset=jnp.zeros(w.size), K=7), w.size)])
    p = np.asarray(jax.jit(lambda a, x: a.prox(x))(atom, jnp.asarray(v)))
    ref2 = np.asarray(socp.project_soc_product(jnp.asarray(v[5:]), jnp.asarray(bid), jnp.asarray(head), 7))
    assert rel_err(p, np.concatenate([np.clip(v[:5], lo, hi), ref2])) == 0.0


# ----------------------------------------------------------------------------------------------------------------
# 2. conic form, scaling, plan, reference
# ----------------------------------------------------------------------------------------------------------------
def test_to_clarabel_rows():
    """s = b - A x must be exactly (u - A_eq x | u - A_up x, A_lo x - l | f'x + h, F x + g per cone)."""
    rng = np.random.default_rng(5)
    d = socp.robust_lp(20, seed=1)
    d["l"] = d["l"].copy()
    d["u"] = d["u"].copy()
    d["l"][:3] = d["u"][:3] = 0.25                                  # equality rows
    d["l"][3:6] = -np.inf                                           # one-sided rows
    x = rng.standard_normal(20)
    c = socp.to_clarabel(d)
    s = c["b"] - c["A"] @ x
    A, l, u = d["A"].tocsr(), d["l"], d["u"]
    eq = np.isfinite(l) & np.isfinite(u) & (l == u)
    up, lo = np.isfinite(u) & ~eq, np.isfinite(l) & ~eq
    lin = np.concatenate([(u - A @ x)[eq], (u - A @ x)[up], (A @ x - l)[lo]])
    socs = np.concatenate([np.concatenate([socp.frow(c_["f"], 20) @ x + c_["h"], c_["F"] @ x + c_["g"]])
                           for c_ in d["cones"]])
    assert c["n_zero"] == eq.sum() and c["n_nonneg"] == up.sum() + lo.sum()
    assert list(c["soc_dims"]) == [1 + c_["F"].shape[0] for c_ in d["cones"]]
    assert rel_err(s, np.concatenate([lin, socs])) <= 1e-14


def test_cone_ruiz_reduces_to_kkt_ruiz():
    d = uni.knapsack_ridge(300, rho=0.1, seed=0)
    m, n = d["A"].shape
    d1, d2 = socp.cone_ruiz(d["P"], d["A"], np.arange(m), np.arange(n), iters=10)
    e1, e2 = kkt_ruiz_equilibrate(d["P"], d["A"], 10)
    assert np.array_equal(d1, e1) and np.array_equal(d2, e2)


def small_instances():
    return {"portfolio_risk": socp.portfolio_risk(30, seed=1),
            "portfolio_risk_lift": socp.portfolio_risk(30, seed=1, lift=True),
            "group_lasso": socp.group_lasso(40, m=80, seed=1),
            "min_enclosing_ball": socp.min_enclosing_ball(30, dim=4, seed=1),
            "robust_lp": socp.robust_lp(30, seed=1),
            "facility_location": socp.facility_location(30, seed=1)}


_REF = {}


def refs():
    if not _REF:
        for name, d in small_instances().items():
            obj, x = socp.reference(d)
            _REF[name] = (d, obj, x)
    return _REF


def test_generators_reference():
    for name, (d, obj, x) in refs().items():
        assert obj is not None, (name, x)
        ok, viol, gap = socp.accept_socp(x, d, obj, 1e-8)
        assert ok, (name, viol, gap)
        assert abs(obj) > 1e-3, (name, obj)                          # the relative objective test needs obj away from 0
        act = socp.socp_violation(x, d)
        assert max(act) <= 1e-8, (name, act)


def test_generators_seedable():
    for fam, fn in socp.FAMILIES.items():
        a, b = fn(20, seed=7), fn(20, seed=7)
        assert np.array_equal(a["q"], b["q"]) and (sp.csc_matrix(a["A"]) != sp.csc_matrix(b["A"])).nnz == 0
        assert all(np.array_equal(ca["g"], cb["g"]) and ca["h"] == cb["h"] for ca, cb in zip(a["cones"], b["cones"]))


def test_plan_permutation_robust():
    expect = {"portfolio_risk": 0, "portfolio_risk_lift": 1, "group_lasso": 8, "min_enclosing_ball": 1,
              "robust_lp": 0, "facility_location": 1}
    for name, d in small_instances().items():
        for dd in (d, socp.permute_socp(d, 11)[0]):
            plan, _ = socp.plan_cone(dd)
            assert (0 if plan is None else plan["K"]) == expect[name], (name, plan and plan["K"])


def test_scs_runner():
    d, obj, _ = refs()["facility_location"]
    rb = run_conic_baseline_proc(d, "scs", 1e-7)
    if "error" in rb:
        print("  scs unavailable:", rb["error"][:80])
        return
    assert abs(socp.socp_objective(rb["x"], d) - obj) / abs(obj) <= 1e-4, rb["status"]


# ----------------------------------------------------------------------------------------------------------------
# 3. convergence under the cone-aware acceptance, and relocation equivalence
# ----------------------------------------------------------------------------------------------------------------
TOLS = (1e-2, 1e-3, 1e-4, 1e-6)
CAP = 100000
# Equivalence is judged on LIMITS, not on first-accepted iterates: at objective accuracy tol a quadratic-growth problem
# only pins x to about sqrt(tol) (unlifted group_lasso: two exact formulations both accepted at 1e-8 were 9e-5 apart),
# and Clarabel's own x is off by about 1e-5 at its 1e-9 gap tolerance. Run past acceptance, std and reloc agree to
# 1e-14 or better on every instance here, so LIMIT_TOL leaves four orders of margin and would still catch a
# formulation that solves a nearby problem.
LIMIT_ITERS, LIMIT_TOL, REF_X_TOL = 30000, 1e-10, 1e-4
_MEAS = {}


def limit_x(d, cand, obj):
    """Iterate after LIMIT_ITERS iterations (a tolerance of 0 is never met, so the ladder runs to the cap)."""
    return socp.measure_ladder_socp(socp.build_socp_formulation(d, cand), d, obj, tols=(0.0,), cap=LIMIT_ITERS)["x"]


def meas(name, cand, perm_seed=None):
    key = (name, cand, perm_seed)
    if key not in _MEAS:
        d, obj, xr = refs()[name]
        pc = None
        if perm_seed is not None:
            d, pc = socp.permute_socp(d, perm_seed)
            xr = xr[pc]                                             # x_perm = x_orig[pc]
        form = socp.build_socp_formulation(d, cand)
        r = socp.measure_ladder_socp(form, d, obj, tols=TOLS, cap=CAP)
        r["form"], r["x_ref"] = form, xr
        _MEAS[key] = r
    return _MEAS[key]


def test_std_ruiz_accepts():
    for name in refs():
        r = meas(name, "std_ruiz")
        assert r["N_at"]["0.001"] is not None, (name, r["row_violation"], r["obj_gap"])


def _equivalent(d, obj, xr, cands=("std_ruiz", "reloc_cone", "reloc_cone_raw")):
    """Limits of every formulation accepted at 1e-6, equal to LIMIT_TOL, and within REF_X_TOL of Clarabel's x."""
    scale = 1.0 + np.linalg.norm(xr)
    lim = {c: limit_x(d, c, obj) for c in cands}
    rows = []
    for c in cands:
        assert socp.accept_socp(lim[c], d, obj, 1e-6)[0], (d.get("name"), c)
        dl = np.linalg.norm(lim[c] - lim[cands[0]]) / scale
        dr = np.linalg.norm(lim[c] - xr) / scale
        assert dl <= LIMIT_TOL, (d.get("name"), c, dl)
        assert dr <= REF_X_TOL, (d.get("name"), c, dr)
        rows.append((c, dl, dr))
    return rows


def _check_reloc_same_solution():
    """On a PERMUTED encoding, reloc_cone(_raw) and std_ruiz are each accepted at 1e-6 by the ladder and converge to
    the same point (the optimum is unique on these instances), which is also Clarabel's."""
    out = []
    for name in ("portfolio_risk_lift", "group_lasso", "min_enclosing_ball", "facility_location"):
        for cand in ("std_ruiz", "reloc_cone", "reloc_cone_raw"):
            r = meas(name, cand, perm_seed=5)
            assert r["N_at"]["1e-06"] is not None, (name, cand)
            assert cand == "std_ruiz" or r["form"].absorbed > 0
        d, obj, xr = refs()[name]
        dp, pc = socp.permute_socp(d, 5)
        out += [(name,) + row for row in _equivalent(dp, obj, xr[pc])]
    return out


def test_reloc_same_solution():
    _check_reloc_same_solution()


def signed_similarity_instance(seed=0):
    """min 1/2 ||x - v||^2 with cones the generators never produce: a relocatable cone with NEGATIVE head sign,
    mixed tail signs, modulus 2.5 and nonzero offsets; a relocatable ball of modulus 0.4 with mixed signs; a dense
    cone that must stay in L; and linear rows. x = 0 is strictly feasible and P = I, so the optimum is unique."""
    rng = np.random.default_rng(seed)
    n = 12
    e = lambda j: np.eye(1, n, j).ravel()
    F1 = sp.csc_matrix((np.array([-2.5, 2.5, -2.5]), (np.arange(3), np.array([7, 0, 5]))), shape=(3, n))
    F2 = sp.csc_matrix((np.array([0.4, -0.4]), (np.arange(2), np.array([1, 9]))), shape=(2, n))
    F3 = sp.csc_matrix(rng.standard_normal((2, n)) * (rng.random((2, n)) < 0.5))
    cones = [{"F": F1, "g": np.array([0.3, -1.2, 0.7]), "f": -2.5 * e(3), "h": 2.0},
             {"F": F2, "g": np.array([0.2, 0.1]), "f": np.zeros(n), "h": 0.5},
             {"F": F3, "g": np.zeros(2), "f": 0.3 * rng.standard_normal(n), "h": 1.0}]
    A = sp.csc_matrix(np.vstack([e(10) + e(11), e(8)]))
    v = rng.standard_normal(n) * 2.0
    return {"P": sp.identity(n, format="csc"), "q": -v, "r": 0.5 * float(v @ v), "A": A, "l": np.array([-1.0, -0.5]),
            "u": np.array([1.0, np.inf]), "cones": cones, "name": "signed_similarity"}


def ball_only_instance(n=15, seed=0):
    """min 1/2 ||x - v||^2 s.t. ||-0.4 x + g|| <= h: a single ball over ALL variables, so nothing is left for L."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(n) * 3.0
    return {"P": sp.identity(n, format="csc"), "q": -v, "r": 0.5 * float(v @ v), "A": sp.csc_matrix((0, n)),
            "l": np.zeros(0), "u": np.zeros(0), "name": "ball_only",
            "cones": [{"F": -0.4 * sp.identity(n, format="csc"), "g": rng.standard_normal(n) * 0.1, "f": np.zeros(n),
                       "h": 0.5}]}


def _check_similarity_and_full_absorption():
    """Sign, modulus and offset handling in g, and the implied redundant row when every constraint is absorbed."""
    cases = [(signed_similarity_instance(), 2), (socp.group_lasso(30, m=60, seed=2, lift=False), 6),
             (ball_only_instance(), 1)]
    out = []
    for d, K in cases:
        obj, xr = socp.reference(d)
        assert obj is not None, (d["name"], xr)
        for dd, pc in ((d, np.arange(d["P"].shape[0])), socp.permute_socp(d, 3)):
            plan, _ = socp.plan_cone(dd)
            assert plan is not None and plan["K"] == K, (d["name"], plan and plan["K"])
            for cand in ("std_ruiz", "reloc_cone", "reloc_cone_raw"):
                form = socp.build_socp_formulation(dd, cand)
                r = socp.measure_ladder_socp(form, dd, obj, tols=(1e-3, 1e-6), cap=CAP)
                assert r["N_at"]["1e-06"] is not None, (d["name"], cand, r["row_violation"], r["obj_gap"])
                out.append((d["name"][:18], cand, r["N_at"]["0.001"], r["N_at"]["1e-06"], form.L_csc.shape[0]))
            _equivalent(dd, obj, xr[pc])
    return out


def test_similarity_and_full_absorption():
    _check_similarity_and_full_absorption()


def test_uni_measure_compatible():
    """The Formulation objects run through uni.measure_ladder unchanged (its verdict ignores the cones, which is why
    measure_ladder_socp exists)."""
    for name in ("group_lasso", "robust_lp"):
        d, obj, _ = refs()[name]
        r = uni.measure_ladder(socp.build_socp_formulation(d, "std_ruiz"), d, obj, tols=(1e-2,), cap=100)
        assert "N_at" in r


def main():
    for fn in (_check_soc_single, _check_soc_product, _check_shifted_signed_product_and_conjugate, _check_balls):
        print(f"{fn.__name__[7:]:45s} worst relative error {fn():.2e}")
    for fn in (test_slice_product, test_to_clarabel_rows, test_cone_ruiz_reduces_to_kkt_ruiz, test_generators_reference,
               test_generators_seedable, test_plan_permutation_robust, test_scs_runner, test_std_ruiz_accepts,
               test_uni_measure_compatible):
        fn()
        print(f"{fn.__name__:45s} ok")
    for row in _check_reloc_same_solution():
        print("  %-20s %-15s limit vs std_ruiz %.1e   limit vs clarabel %.1e" % row)
    print(f"{'test_reloc_same_solution':45s} ok")
    for row in _check_similarity_and_full_absorption():
        print("  %-18s %-16s N@1e-3=%-6s N@1e-6=%-6s rows_L=%d" % row)
    print(f"{'test_similarity_and_full_absorption':45s} ok")
    print()
    print(f"{'instance':22s} {'formulation':24s} {'n':>5s} {'rows L':>6s} {'|L|':>6s} {'w':>6s}  "
          + " ".join(f"N@{t:g}".rjust(8) for t in TOLS) + "   dist_to_clarabel")
    names = list(refs())
    for name in names:
        d = refs()[name][0]
        cands = [("std_raw", None), ("std_ruiz", None)]
        if socp.plan_cone(d)[0] is not None:
            cands += [("std_ruiz", 5), ("reloc_cone", 5), ("reloc_cone_raw", 5)]
        for cand, ps in cands:
            r = meas(name, cand, ps)
            f = r["form"]
            w = f.prob.data.get("primal_weight")
            dist = np.linalg.norm(r["x"] - r["x_ref"]) / (1.0 + np.linalg.norm(r["x_ref"]))
            label = f.name + ("/perm" if ps is not None else "")
            print(f"{name:22s} {label:24s} {d['P'].shape[0]:5d} {f.L_csc.shape[0]:6d} {r['normL']:6.2f} "
                  f"{'-' if w is None else format(w, '.3f'):>6s}  "
                  + " ".join(str(r["N_at"][f"{t:g}"]).rjust(8) for t in TOLS) + f"   {dist:.1e}")


if __name__ == "__main__":
    main()
