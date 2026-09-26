"""Budgeted LLM routing on RouterBench as a structured LP in the solver's (P, q, r, A, l, u) form (2026-09-15).

TEXT DATA THAT IS SECRETLY AN OPTIMIZATION PROBLEM. RouterBench is a table of prompts x candidate LLMs carrying, per
cell, a quality score in [0, 1] and the dollar cost of that model answering that prompt. "Serve every prompt with some
model, spend at most B in total, maximize total quality" is the LP

    max  sum_{p,m} quality[p,m] x[p,m]
    s.t. sum_m x[p,m] = 1           for every prompt p      (one simplex per prompt: n_prompts EQUALITY rows)
         sum_{p,m} cost[p,m] x[p,m] <= B                    (ONE dense coupling row: the budget)
         x >= 0

written here as the minimization of -quality'x, the convention of the rest of the pipeline. Variable bounds are
identity rows of A (the realdata.load_qplib / lp_families convention), P is the zero matrix, so uni.diagnose and
uni.build_formulation take the LP path through lp_families.

WHY THIS IS THE STRUCTURE THE PROJECT RELOCATES. Two DISJOINT absorbable families cover every variable, and they
overlap, so exactly one can move into the proximal step (as for the two marginals of transport):

  (S) the per-prompt simplices  {x_p >= 0, 1'x_p = 1}, K = n_prompts blocks of M columns each. Absorbing them leaves
      the single dense budget row in L, whose norm is ||cost||_2 and grows like sqrt(n).
  (B) the budget set  {x >= 0, cost'x <= B}, K = 1 block of ALL n columns. Absorbing it leaves the simplex rows in L,
      a block-diagonal 0/1 matrix with EXACTLY ||L|| = sqrt(M) (M = number of models) at every size.

uni.plan_product takes a disjoint family greedily by DECREASING ROW LENGTH, so on this instance it always takes the
budget row first (nnz n) and can then add no simplex row (every column is used): it returns family (B), K = 1,
cover 1. This is the gub_packing behaviour of lp_families' registry ("the greedy takes ONE knapsack row first ... and
leaves every GUB row in L"), here with the opposite sign of usefulness: the row it takes is the dense coupling row
that alone sets ||L||, which is exactly the row the portfolio result says to relocate. plan_simplex_family() below
builds family (S) in the same schema so both can be measured; see its docstring for the shared-file change that
would let uni.plan_product return it.

TWO VARIANTS, AND WHY BOTH ARE HERE.
  make_routing              the budgeted problem above. ONE coupling row, so the LP is a fractional multiple-choice
                            knapsack: brute_force_lp solves it in closed form in under a second at 36497 prompts, and
                            HiGHS reproduces that value to 1e-16. It is a faithful reduction and a clean structure
                            demonstration, but it is NOT hard: no honest "a solver alone cannot do it" claim survives
                            it.
  make_routing_capacitated  the same problem with PER-MODEL CAPACITY rows, the operational constraint the plain
                            version is missing (provider rate limits). Three overlapping disjoint families instead of
                            two, no closed form, and a formulation spread of 1233x in iteration count (measured at
                            1024 prompts: std_ruiz 184950 iterations to 1e-3, the budget-row relocation that
                            uni.plan_product picks 23350, the simplex-family relocation 150).

DATA. withmartian/routerbench, revision 784021482c3f320c6619ed4b3bb3b41a21424fcb (2024-03-27), file
routerbench_0shot.pkl (36497 prompts x 11 models). data/routerbench/MANIFEST.json records the revision, the URL and
the file's sha256; routerbench_0shot_qc.npz is a 0.9 MB numeric cache of the two matrices (quality, cost) plus
eval_name and sample_id, so nothing downstream needs pandas or the 100 MB pickle.
"""
from __future__ import annotations

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_V2 = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
if _V2 not in sys.path:
    sys.path.insert(0, _V2)

import numpy as np
import scipy.sparse as sp

DATA_DIR = os.environ.get("ROUTERBENCH_DIR", os.path.join(_V2, "data", "routerbench"))
CACHE = "routerbench_0shot_qc.npz"
PICKLE = "routerbench_0shot.pkl"
REVISION = "784021482c3f320c6619ed4b3bb3b41a21424fcb"
HF_REPO = "withmartian/routerbench"
MODELS = ("WizardLM/WizardLM-13B-V1.2", "claude-instant-v1", "claude-v1", "claude-v2", "gpt-3.5-turbo-1106",
          "gpt-4-1106-preview", "meta/code-llama-instruct-34b-chat", "meta/llama-2-70b-chat",
          "mistralai/mistral-7b-chat", "mistralai/mixtral-8x7b-chat", "zero-one-ai/Yi-34B-Chat")
# n_prompts ladder; the last entry is the full table (36497 prompts x 11 models = 401467 variables)
SIZES = (256, 1024, 4096, 16384, 36497)
FULL = 36497


# ----------------------------------------------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------------------------------------------
def fetch(dest=None):
    """Download the pinned revision of routerbench_0shot.pkl and write the numeric cache and MANIFEST.json.

    Kept out of the import path: the cache is committed alongside the data, so a normal run never touches the
    network. Needs pandas (the published artifact is a pickled DataFrame, not parquet)."""
    import hashlib
    import urllib.request

    import pandas as pd

    dest = dest or DATA_DIR
    os.makedirs(dest, exist_ok=True)
    url = f"https://huggingface.co/datasets/{HF_REPO}/resolve/{REVISION}/{PICKLE}"
    pkl = os.path.join(dest, PICKLE)
    if not os.path.exists(pkl):
        urllib.request.urlretrieve(url, pkl)
    df = pd.read_pickle(pkl)
    Q = np.column_stack([pd.to_numeric(df[m], errors="raise").to_numpy(dtype=np.float64) for m in MODELS])
    C = np.column_stack([df[m + "|total_cost"].to_numpy(dtype=np.float64) for m in MODELS])
    # str() per element, not .astype(str): pandas 3 returns an OBJECT array there, and np.load(allow_pickle=False)
    # then refuses the file, so the cache has to hold fixed-width unicode
    txt = lambda col: np.array([str(v) for v in df[col]], dtype=np.str_)
    np.savez_compressed(os.path.join(dest, CACHE), quality=Q, cost=C,
                        models=np.array(MODELS, dtype=np.str_),
                        eval_name=txt("eval_name"), sample_id=txt("sample_id"))
    meta = {"repo": HF_REPO, "repo_type": "dataset", "revision": REVISION, "revision_date": "2024-03-27T07:27:17Z",
            "doi": "10.57967/hf/1996", "file": PICKLE, "file_bytes": os.path.getsize(pkl),
            "file_sha256": hashlib.sha256(open(pkl, "rb").read()).hexdigest(), "n_prompts": int(Q.shape[0]),
            "n_models": int(Q.shape[1]), "models": list(MODELS), "url": url, "cache": CACHE}
    with open(os.path.join(dest, "MANIFEST.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    return meta


_TABLE = {}


def load_table(dest=None):
    """{'quality': (n_prompts, M), 'cost': (n_prompts, M), 'models', 'eval_name', 'sample_id'} from the npz cache."""
    dest = dest or DATA_DIR
    key = os.path.abspath(dest)
    if key not in _TABLE:
        z = np.load(os.path.join(dest, CACHE), allow_pickle=False)
        _TABLE[key] = {"quality": z["quality"].astype(float), "cost": z["cost"].astype(float),
                       "models": [str(s) for s in z["models"]], "eval_name": z["eval_name"],
                       "sample_id": z["sample_id"]}
    return _TABLE[key]


def subset(n_prompts=None, seed=0, dest=None):
    """A seeded prompt subset (sorted, so the instance is reproducible and the eval_name mix is the table's own)."""
    t = load_table(dest)
    N = t["quality"].shape[0]
    if n_prompts is None or int(n_prompts) >= N:
        idx = np.arange(N)
    else:
        idx = np.sort(np.random.default_rng([seed, 99]).choice(N, int(n_prompts), replace=False))
    return idx, t


# ----------------------------------------------------------------------------------------------------------------
# The LP
# ----------------------------------------------------------------------------------------------------------------
def budget_range(Q, C):
    """(cheapest routing cost, most expensive routing cost, cost of the quality-optimal routing).

    The third is where the budget STOPS binding: above it the coupling row is slack and the LP separates into
    independent per-prompt argmaxes, which no first-order method finds hard."""
    c_min = float(C.min(axis=1).sum())
    c_max = float(C.max(axis=1).sum())
    # ties in quality broken toward the cheaper model, which is what an unconstrained optimum would pick
    best = np.lexsort((C, -Q), axis=1)[:, 0]
    c_q = float(C[np.arange(C.shape[0]), best].sum())
    return c_min, c_max, c_q


def make_routing(n_prompts=None, budget_frac=0.25, seed=0, dest=None, upper_bounds=False, permute=False,
                 budget_ref="max"):
    """Budgeted routing LP in OSQP form.

    SIZE: n_prompts (n_var = n_prompts * 11). DIFFICULTY: budget_frac in [0, 1] places B between the cheapest feasible
    routing (0: only the cheapest model per prompt is affordable, the budget row is tight and the LP is nearly
    determined) and, for budget_ref="max", the most expensive one (1: the row is slack and every prompt independently
    takes its best model).

    RouterBench's cost spread is very skewed (mean cost per prompt 4.6e-5 for mistral-7b against 3.3e-3 for
    gpt-4-1106-preview) and the quality-optimal routing does NOT take the dearest model everywhere, so the budget row
    stops binding at frac_slack = (cost of the quality-optimal routing - c_min)/(c_max - c_min), measured at 0.057 to
    0.081 over these sizes. Only 0 < budget_frac < frac_slack makes the problem non-separable, so budget_ref="opt"
    rescales the knob to that band: B = c_min + budget_frac * (c_q - c_min), where the whole of [0, 1] binds and 1 is
    exactly the unconstrained quality optimum. meta records both fractions.

    upper_bounds adds the redundant x <= 1 identity rows (implied by the simplex rows); off by default so the
    bound-row block is exactly one row per variable, as in lp_families.transport.
    permute applies uni.permute_instance, as the unified experiments do, so no plan can read the generator's order.
    Extra keys: x_feas, name, meta.
    """
    idx, t = subset(n_prompts, seed, dest)
    Q, C = t["quality"][idx], t["cost"][idx]
    P_, M = Q.shape
    n = P_ * M
    c_min, c_max, c_q = budget_range(Q, C)
    if budget_ref not in ("max", "opt"):
        raise ValueError("budget_ref must be 'max' or 'opt'")
    top = c_max if budget_ref == "max" else c_q
    B = c_min + float(budget_frac) * (top - c_min)
    rows = np.repeat(np.arange(P_), M)
    cols = np.arange(n)
    S = sp.csr_matrix((np.ones(n), (rows, cols)), shape=(P_, n))            # one simplex row per prompt
    br = sp.csr_matrix((C.ravel(), (np.zeros(n, dtype=int), cols)), shape=(1, n))   # the dense coupling budget row
    G = sp.vstack([S, br], format="csr")
    gl = np.concatenate([np.ones(P_), [-np.inf]])
    gu = np.concatenate([np.ones(P_), [B]])
    lo = np.zeros(n)
    hi = np.ones(n) if upper_bounds else np.full(n, np.inf)
    x_feas = np.zeros(n)
    x_feas[np.arange(P_) * M + np.argmin(C, axis=1)] = 1.0                  # cheapest routing: feasible for any frac
    b = np.flatnonzero(np.isfinite(lo) | np.isfinite(hi))
    Bm = sp.csr_matrix((np.ones(b.size), (np.arange(b.size), b)), shape=(b.size, n))
    meta = {"family": "routerbench_routing", "dataset": HF_REPO, "revision": REVISION, "shot": "0shot",
            "n_prompts": int(P_), "n_models": int(M), "seed": int(seed), "budget_frac": float(budget_frac),
            "budget_ref": budget_ref, "budget": B, "cost_cheapest": c_min, "cost_dearest": c_max,
            "cost_quality_optimal": c_q, "frac_of_max": float((B - c_min) / (c_max - c_min)),
            "frac_of_opt": float((B - c_min) / (c_q - c_min)),
            "frac_slack": float((c_q - c_min) / (c_max - c_min)), "prompt_idx": idx,
            "general_rows": int(G.shape[0]), "bound_rows": int(b.size), "upper_bounds": bool(upper_bounds)}
    d = {"P": sp.csc_matrix((n, n)), "q": -Q.ravel().astype(float), "r": 0.0,
         "A": sp.vstack([G, Bm], format="csc"), "l": np.concatenate([gl, lo[b]]), "u": np.concatenate([gu, hi[b]]),
         "x_feas": x_feas, "name": f"routing_{P_}x{M}_f{budget_frac:g}", "meta": meta,
         "quality": Q, "cost": C}
    if permute:
        import uni
        p = uni.permute_instance({k: d[k] for k in ("P", "q", "r", "A", "l", "u")}, 1000 + seed)
        p.update({k: d[k] for k in ("x_feas", "name", "meta", "quality", "cost")})
        p["meta"] = dict(meta, permuted=True)
        p["x_feas"] = None                        # the feasible point is in unpermuted coordinates; not carried over
        d = p
    return d


def _greedy_under_caps(key, C, cap):
    """Deterministic routing that minimizes `key` per prompt subject to per-model capacities.

    Prompts are visited in order of how much they LOSE by being denied their first choice (largest regret first),
    and each takes the best model that still has capacity. Not optimal, but feasible by construction, which is all a
    right-hand side needs: the generators in lp_families draw a feasible point first for the same reason."""
    P_, M = key.shape
    order = np.argsort(key, axis=1)
    regret = key[np.arange(P_), order[:, 1]] - key[np.arange(P_), order[:, 0]] if M > 1 else np.zeros(P_)
    left = cap.astype(float).copy()
    j = np.empty(P_, dtype=int)
    for p in np.argsort(-regret):
        for m in order[p]:
            if left[m] >= 1.0 - 1e-12:
                j[p] = m
                left[m] -= 1.0
                break
        else:
            j[p] = order[p, 0]                     # cannot happen while sum(cap) >= P_, kept so the loop is total
            left[j[p]] -= 1.0
    return j, float(C[np.arange(P_), j].sum())


def make_routing_capacitated(n_prompts=None, budget_frac=0.5, cap_frac=0.25, seed=0, dest=None, permute=False):
    """The same routing LP plus PER-MODEL CAPACITY rows: sum_p x[p,m] <= cap_frac * n_prompts for every model.

    This is the natural operational constraint the plain budgeted problem is missing (provider rate limits, vendor
    diversification: "no single model serves more than a quarter of traffic"). It matters structurally and
    computationally, and the difference is the point of including it:

      structure   the capacity rows partition the variables BY MODEL, exactly as the simplex rows partition them BY
                  PROMPT. The instance therefore carries THREE overlapping disjoint families (prompt marginals,
                  model marginals, the budget row), which is the two-marginal geometry of lp_families.transport with
                  a dense coupling row added, not a knapsack.
      difficulty  with more than one coupling row the Lagrangian is no longer a scalar bisection, and the LP stops
                  being a fractional multiple-choice knapsack. The closed-form reference of brute_force_lp does NOT
                  apply here and is not offered.

    cap_frac must be at least 1/n_models for the capacity rows to admit any routing; at 1 they never bind.
    The budget is placed between two capacity-feasible routings built by _greedy_under_caps, so x_feas is feasible
    for every budget_frac in [0, 1]."""
    idx, t = subset(n_prompts, seed, dest)
    Q, C = t["quality"][idx], t["cost"][idx]
    P_, M = Q.shape
    n = P_ * M
    cap = np.full(M, float(cap_frac) * P_)
    if cap.sum() < P_ - 1e-9:
        raise ValueError(f"cap_frac {cap_frac} is below 1/{M}: no routing meets every capacity")
    j_cheap, c_lo = _greedy_under_caps(C, C, cap)
    _, c_hi = _greedy_under_caps(-Q, C, cap)
    B = c_lo + float(budget_frac) * max(c_hi - c_lo, 0.0)
    rows = np.repeat(np.arange(P_), M)
    cols = np.arange(n)
    S = sp.csr_matrix((np.ones(n), (rows, cols)), shape=(P_, n))
    K = sp.csr_matrix((np.ones(n), (np.tile(np.arange(M), P_), cols)), shape=(M, n))
    br = sp.csr_matrix((C.ravel(), (np.zeros(n, dtype=int), cols)), shape=(1, n))
    G = sp.vstack([S, K, br], format="csr")
    gl = np.concatenate([np.ones(P_), np.full(M, -np.inf), [-np.inf]])
    gu = np.concatenate([np.ones(P_), cap, [B]])
    Bm = sp.identity(n, format="csr")
    x_feas = np.zeros(n)
    x_feas[np.arange(P_) * M + j_cheap] = 1.0
    meta = {"family": "routerbench_routing_cap", "dataset": HF_REPO, "revision": REVISION, "shot": "0shot",
            "n_prompts": int(P_), "n_models": int(M), "seed": int(seed), "budget_frac": float(budget_frac),
            "cap_frac": float(cap_frac), "budget": B, "cost_cheapest_capped": c_lo, "cost_quality_capped": c_hi,
            "capacity": float(cap[0]), "prompt_idx": idx, "general_rows": int(G.shape[0]), "bound_rows": n,
            "budget_ref": "capped", "frac_of_max": float("nan"), "frac_of_opt": float(budget_frac),
            "frac_slack": float("nan")}
    d = {"P": sp.csc_matrix((n, n)), "q": -Q.ravel().astype(float), "r": 0.0,
         "A": sp.vstack([G, Bm], format="csc"), "l": np.concatenate([gl, np.zeros(n)]),
         "u": np.concatenate([gu, np.full(n, np.inf)]), "x_feas": x_feas,
         "name": f"routing_cap_{P_}x{M}_f{budget_frac:g}_c{cap_frac:g}", "meta": meta, "quality": Q, "cost": C}
    if permute:
        import uni
        p = uni.permute_instance({k: d[k] for k in ("P", "q", "r", "A", "l", "u")}, 1000 + seed)
        p.update({k: d[k] for k in ("name", "meta", "quality", "cost")})
        p["meta"] = dict(meta, permuted=True)
        p["x_feas"] = None
        d = p
    return d


def instance_stats(d):
    A = sp.csc_matrix(d["A"])
    return {"n_var": int(A.shape[1]), "rows": int(A.shape[0]), "nnz": int(A.nnz),
            "general_rows": d["meta"]["general_rows"], "bound_rows": d["meta"]["bound_rows"],
            "budget": d["meta"]["budget"], "frac_of_max": d["meta"]["frac_of_max"],
            "frac_of_opt": d["meta"]["frac_of_opt"], "frac_slack": d["meta"]["frac_slack"]}


# ----------------------------------------------------------------------------------------------------------------
# Brute force: the Lagrangian (knapsack dual) on the single coupling row, and the natural routing baselines
# ----------------------------------------------------------------------------------------------------------------
def lagrangian_value(Q, C, B, lam):
    """g(lam) = sum_p max_m (q[p,m] - lam c[p,m]) + lam B, the dual of the routing LP at lam >= 0.

    Dropping the one coupling row leaves n_prompts independent simplices, so the inner maximum is a per-prompt
    argmax: the dual is evaluated in closed form with no solver at all. Weak duality gives g(lam) >= LP optimum for
    every lam >= 0, and min_{lam>=0} g(lam) = LP optimum (the LP is feasible and bounded)."""
    return float(np.max(Q - lam * C, axis=1).sum() + lam * B)


def lagrangian_sweep(Q, C, B, lam_hi=None, iters=200):
    """Bisect the dual on the budget: returns (dual optimum, lam*, primal from the argmax at lam*, its cost).

    phi(lam) = cost of the argmax routing at lam is nonincreasing in lam, so bisection on phi(lam) - B brackets the
    lam where the budget row is exactly met. At that lam the argmax primal is an OPTIMAL INTEGRAL routing whenever it
    hits the budget exactly; in general it differs from the LP optimum by at most one fractionally split prompt."""
    def route(lam):
        # ties in q - lam c broken toward the CHEAPER model, so phi(lam) is the lower envelope and stays monotone
        j = np.lexsort((C, -(Q - lam * C)), axis=1)[:, 0]
        r = np.arange(Q.shape[0])
        return j, float(C[r, j].sum()), float(Q[r, j].sum())

    if lam_hi is None:
        # at lam above max over p of (q spread / c spread) every prompt takes its cheapest model
        lam_hi = 1.0
        for _ in range(80):
            if route(lam_hi)[1] <= B:
                break
            lam_hi *= 2.0
    lo_, hi_ = 0.0, float(lam_hi)
    for _ in range(iters):
        mid = 0.5 * (lo_ + hi_)
        if route(mid)[1] > B:
            lo_ = mid
        else:
            hi_ = mid
    lam = hi_
    j, cost, qual = route(lam)
    # the dual is minimized over a grid around the bracket: the minimum can sit at either endpoint of the flat piece
    cands = [lo_, hi_, 0.5 * (lo_ + hi_), 0.0]
    dual = min(lagrangian_value(Q, C, B, l) for l in cands if l >= 0.0)
    return dual, lam, j, cost, qual


def brute_force_lp(Q, C, B, tie_rtol=1e-9):
    """Exact LP optimum with NO LP solver, certified by a matching primal and dual.

    g(lam) = sum_p max_m (q - lam c) + lam B is convex piecewise linear with g'(lam) = B - cost(routing at lam), so
    the lam* found by bisecting cost(lam) - B is its minimizer and g(lam*) is the LP optimum. The matching primal is
    built from the TIE SET at lam*: RouterBench quality takes only 13 distinct values, so at lam* thousands of
    prompts are indifferent between two models, and an optimal vertex splits many of them, not one (an earlier
    single-split repair left a 2.3e-5 relative gap at 36497 prompts, which would have made a 1e-4 acceptance check
    meaningless). Within a tie set the quality gained per dollar is exactly lam*, so filling the budget with tied
    upgrades in any order attains g(lam*).

    Returns (objective, cost, n_fractional_prompts, certificate) where the certificate carries lam*, the dual value
    and the relative primal-dual gap: 0 means the returned number is the exact optimum."""
    r = np.arange(Q.shape[0])
    _, lam, _, _, _ = lagrangian_sweep(Q, C, B)
    dual = lagrangian_value(Q, C, B, lam)
    s = Q - lam * C
    best = s.max(axis=1, keepdims=True)
    tied = s >= best - tie_rtol * np.maximum(1.0, np.abs(best))
    # cheapest and dearest tied option per prompt: everything between them is optimal for the Lagrangian
    c_tied = np.where(tied, C, np.inf)
    j_lo = np.argmin(c_tied, axis=1)
    j_hi = np.argmax(np.where(tied, C, -np.inf), axis=1)
    cost = float(C[r, j_lo].sum())
    qual = float(Q[r, j_lo].sum())
    slack = B - cost
    n_frac = 0
    if slack > 0:
        dc = C[r, j_hi] - C[r, j_lo]
        dq = Q[r, j_hi] - Q[r, j_lo]
        order = np.flatnonzero(dc > 0)
        take = np.cumsum(dc[order])
        k = int(np.searchsorted(take, slack))                 # prompts fully moved, then one partial
        if k:
            qual += float(dq[order[:k]].sum())
            cost += float(take[k - 1])
        if k < order.size:
            theta = (B - cost) / dc[order[k]]
            if theta > 0:
                qual += float(theta * dq[order[k]])
                cost += float(theta * dc[order[k]])
                n_frac = 1
    gap = abs(dual - qual) / (1.0 + abs(dual))
    return qual, cost, n_frac, {"lambda": lam, "dual": dual, "rel_primal_dual_gap": gap,
                                "n_tied_prompts": int((tied.sum(axis=1) > 1).sum())}


def baselines(Q, C, B):
    """Natural routing policies, each reported as (quality, cost, feasible under B).

    all:<model>   send every prompt to one model (the RouterBench per-LLM rows)
    cheapest      per prompt, the cheapest model (always feasible: its cost is the budget's own lower end)
    best_quality  per prompt, the highest-quality model, ties to the cheaper: the zero-cost oracle of the paper
    greedy        start from cheapest and take upgrades in decreasing quality-per-dollar while the budget allows
                  (the textbook cost-benefit rule for a multiple-choice knapsack)
    """
    P_, M = Q.shape
    r = np.arange(P_)
    out = {}
    for m in range(M):
        out[f"all:{m}"] = (float(Q[:, m].sum()), float(C[:, m].sum()))
    jc = np.argmin(C, axis=1)
    out["cheapest"] = (float(Q[r, jc].sum()), float(C[r, jc].sum()))
    jb = np.lexsort((C, -Q), axis=1)[:, 0]
    out["best_quality"] = (float(Q[r, jb].sum()), float(C[r, jb].sum()))
    # greedy: every (prompt, model) upgrade over the cheapest choice, sorted by quality gained per dollar
    dq = (Q - Q[r, jc][:, None]).ravel()
    dc = (C - C[r, jc][:, None]).ravel()
    ok = (dc > 1e-18) & (dq > 0)
    order = np.argsort(-(dq[ok] / dc[ok]))
    cand = np.flatnonzero(ok)[order]
    cur_q, cur_c = out["cheapest"]
    chosen = jc.copy()
    taken = np.zeros(P_, dtype=bool)
    for k in cand:
        p, m = divmod(int(k), M)
        if taken[p]:
            continue                       # one upgrade per prompt keeps the routing a valid vertex
        add_c = C[p, m] - C[p, chosen[p]]
        if cur_c + add_c <= B:
            cur_c += add_c
            cur_q += Q[p, m] - Q[p, chosen[p]]
            chosen[p] = m
            taken[p] = True
    out["greedy"] = (float(cur_q), float(cur_c))
    return {k: (q, c, bool(c <= B + 1e-12)) for k, (q, c) in out.items()}


# ----------------------------------------------------------------------------------------------------------------
# Structure: the second absorbable family, in uni.plan_product's schema
# ----------------------------------------------------------------------------------------------------------------
def plan_product_filtered(d, min_nnz=2, keep_row=None, order=None):
    """uni.plan_product with a caller-supplied row FILTER and row ORDER.

    uni.plan_product is byte-for-byte this routine with keep_row = None and order = (-nnz, index). It cannot be
    called with a different preference because the greedy order is hard-coded, and uni.py is not edited here. The
    shared-file change this would need is one line in uni.plan_product: accept an optional
    key=lambda r: (-nnz[r], r) and an optional predicate, defaulting to today's behaviour.

    keep_row(r) -> bool decides whether row r may enter the family; order(r) is the greedy sort key.
    """
    A = sp.csr_matrix(d["A"])
    l = np.asarray(d["l"], dtype=float)
    u = np.asarray(d["u"], dtype=float)
    m, n = A.shape
    nnz = np.diff(A.indptr)
    lo, hi = np.full(n, -np.inf), np.full(n, np.inf)
    has = np.zeros(n, dtype=bool)
    bound_rows_of = {}
    for r in np.flatnonzero(nnz == 1):
        c, v = int(A.indices[A.indptr[r]]), float(A.data[A.indptr[r]])
        if v == 0.0:
            continue
        rl, ru = (l[r] / v, u[r] / v) if v > 0 else (u[r] / v, l[r] / v)
        lo[c], hi[c] = max(lo[c], rl), min(hi[c], ru)
        has[c] = True
        bound_rows_of.setdefault(c, []).append(int(r))
    fin = has & (np.isfinite(lo) | np.isfinite(hi))
    cands = []
    for r in np.flatnonzero(nnz >= min_nnz):
        if keep_row is not None and not keep_row(int(r)):
            continue
        cols = A.indices[A.indptr[r]:A.indptr[r + 1]]
        vals = A.data[A.indptr[r]:A.indptr[r + 1]]
        if fin[cols].all() and np.all(vals != 0.0):
            cands.append(int(r))
    if not cands:
        return None, "no absorbable rows under the filter"
    used = np.zeros(n, dtype=bool)
    rows = []
    key = order if order is not None else (lambda r: (-nnz[r], r))
    for r in sorted(cands, key=key):
        cols = A.indices[A.indptr[r]:A.indptr[r + 1]]
        if not used[cols].any():
            used[cols] = True
            rows.append(r)
    T, bid, a = [], [], []
    for k, r in enumerate(rows):
        cols = A.indices[A.indptr[r]:A.indptr[r + 1]]
        o = np.argsort(cols)
        T.append(cols[o]); a.append(A.data[A.indptr[r]:A.indptr[r + 1]][o]); bid.append(np.full(cols.size, k))
    T, a, bid = np.concatenate(T), np.concatenate(a), np.concatenate(bid)
    removed = set(rows)
    for c in T.tolist():
        removed.update(bound_rows_of.get(c, []))
    keep = np.array(sorted(set(range(m)) - removed), dtype=int)
    fully = keep.size == 0
    if fully:
        keep = np.array([bound_rows_of[int(T[0])][0]], dtype=int)
    rest = np.array(sorted(set(range(n)) - set(T.tolist())), dtype=int)
    return {"kind": "product", "T": T, "bid": bid, "a": a, "lo": lo[T], "hi": hi[T],
            "b_lo": l[np.array(rows)], "b_hi": u[np.array(rows)], "K": len(rows), "rows": np.array(rows),
            "keep": keep, "perm": np.concatenate([T, rest]), "n_x": int(T.size), "cover": float(T.size / n),
            "fully_absorbed": bool(fully)}, "RELOCATABLE_PRODUCT"


def _row_is_equality(d):
    l, u = np.asarray(d["l"], dtype=float), np.asarray(d["u"], dtype=float)
    return np.isclose(l, u) & np.isfinite(l)


def plan_simplex_family(d):
    """Absorb the per-prompt SIMPLEX family (S); the budget row stays in the linear operator.

    Filter: equality rows only, which on this instance are exactly the simplex rows (the budget and capacity rows
    are one-sided). No order change is needed once they are filtered out: the simplex rows are already disjoint."""
    eq = _row_is_equality(d)
    return plan_product_filtered(d, keep_row=lambda r: bool(eq[r]))


def plan_budget_row(d):
    """Absorb the BUDGET row (B); every simplex row stays in the linear operator. This is what uni.plan_product
    returns unaided, because the budget row is the longest row in A."""
    eq = _row_is_equality(d)
    return plan_product_filtered(d, keep_row=lambda r: not bool(eq[r]))


def plan_capacity_family(d, n_models=11):
    """Absorb the per-model CAPACITY family of the capacitated variant (the model marginals).

    Filter: inequality rows other than the budget row. The budget row is told apart by its coefficients, which are
    the costs and are not all 1, so the filter does not depend on the row order of the generator."""
    A = sp.csr_matrix(d["A"])
    eq = _row_is_equality(d)

    def ok(r):
        if eq[r]:
            return False
        v = A.data[A.indptr[r]:A.indptr[r + 1]]
        return v.size > 1 and bool(np.allclose(v, 1.0))

    return plan_product_filtered(d, keep_row=ok)


PLANS = {"budget": plan_budget_row, "simplex": plan_simplex_family}
PLANS_CAP = {"budget": plan_budget_row, "simplex": plan_simplex_family, "capacity": plan_capacity_family}


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--sizes", nargs="*", type=int, default=list(SIZES))
    ap.add_argument("--budget-frac", type=float, default=0.5)
    ap.add_argument("--budget-ref", default="opt")
    args = ap.parse_args()
    if args.fetch:
        print(json.dumps(fetch(), indent=1))
    for s in args.sizes:
        d = make_routing(s, args.budget_frac, budget_ref=args.budget_ref)
        print(json.dumps({"size": s, **instance_stats(d)}, default=float), flush=True)
