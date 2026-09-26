"""Portfolio and dispatch for the generalisation stage (pure functions, no GPU).

T[instance][candidate] is the certified wall time of a candidate configuration on a tune instance
(inf when it does not certify); feats[instance] holds declared features (log10 n, log10 p, lam_frac,
alpha). fit_portfolio picks at most max_routes candidates greedily by the geometric mean of the
per-instance best member, then fits a dispatch tree of depth at most max_depth on the features that
minimises the geometric-mean time of the dispatched member. Everything is fitted on tune instances only;
the result is frozen before any held-out instance is run."""
from __future__ import annotations

import math

PENALTY = 1e6          # seconds charged for a configuration that did not certify


def _log(t: float) -> float:
    return math.log(t if (t is not None and math.isfinite(t) and t > 0) else PENALTY)


def geo_mean(xs) -> float:
    xs = list(xs)
    return math.exp(sum(_log(x) for x in xs) / len(xs)) if xs else float("nan")


def _cost(T, insts, cand) -> float:
    return sum(_log(T[i].get(cand)) for i in insts)


def _fit_tree(T, feats, insts, members, depth):
    leaf = min(members, key=lambda c: (_cost(T, insts, c), members.index(c)))
    best = ({"leaf": leaf}, _cost(T, insts, leaf))
    if depth == 0 or len(insts) < 2:
        return best
    for f in sorted(feats[insts[0]]):
        vals = sorted({feats[i][f] for i in insts})
        for lo, hi in zip(vals, vals[1:]):
            t = 0.5 * (lo + hi)
            L = [i for i in insts if feats[i][f] <= t]; R = [i for i in insts if feats[i][f] > t]
            nl, cl = _fit_tree(T, feats, L, members, depth - 1)
            nr, cr = _fit_tree(T, feats, R, members, depth - 1)
            if cl + cr < best[1] - 1e-9:          # a split must strictly beat the single leaf
                best = ({"feature": f, "threshold": t, "le": nl, "gt": nr}, cl + cr)
    return best


def dispatch(rule: dict, feat: dict):
    while "leaf" not in rule:
        rule = rule["le"] if feat[rule["feature"]] <= rule["threshold"] else rule["gt"]
    return rule["leaf"]


def fit_portfolio(T: dict, feats: dict, max_routes: int = 3, max_depth: int = 2, min_gain: float = 0.02) -> dict:
    insts = sorted(T)
    cands = sorted({c for i in insts for c in T[i]})
    single = min(cands, key=lambda c: (_cost(T, insts, c), c))
    members = [single]
    oracle = lambda ms: sum(min(_log(T[i].get(m)) for m in ms) for i in insts)
    while len(members) < max_routes:
        rest = [c for c in cands if c not in members]
        if not rest:
            break
        c = min(rest, key=lambda c: (oracle(members + [c]), c))
        if math.exp((oracle(members) - oracle(members + [c])) / len(insts)) < 1.0 + min_gain:
            break
        members.append(c)
    rule, _ = _fit_tree(T, feats, insts, members, max_depth)
    used = sorted({dispatch(rule, feats[i]) for i in insts}, key=members.index)
    return {"single_best": single, "members": used, "rule": rule,
            "tune_geo_single": geo_mean(T[i].get(single) for i in insts),
            "tune_geo_dispatch": geo_mean(T[i].get(dispatch(rule, feats[i])) for i in insts),
            "tune_geo_oracle": geo_mean(min((T[i].get(m) for m in cands), key=_log) for i in insts)}
