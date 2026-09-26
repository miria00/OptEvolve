"""Evolvable planner rules: which formulations the router explores on an instance it has no evidence for.

Exploration is the dominant cost on a NEW instance (one ladder run per formulation). A rule is Python code defining
    select(features: dict, neighbors: list[dict]) -> list[str]      (formulation keys to explore)
where `neighbors` summarizes vault evidence on other instances of the same family on this device. A proposed rule is
ADMITTED only if, on held-out instances with complete evidence, it keeps the decision (geometric-mean regret <= 1.05,
worst <= 1.5, no instance left without a formulation that reaches the target) while saving exploration time. This is
how a fix to the tool's decision logic becomes an agent action checked against held-out predictions, not a hand edit.
"""
from __future__ import annotations

import math
from collections import defaultdict

FORMULATIONS = ("std_raw", "std_ruiz", "reloc", "reloc_rows", "reloc_prod", "reloc_rows_prod", "reloc_cone",
                "reloc_cone_raw", "engine:clarabel", "engine:highs", "engine:piqp", "engine:scs", "engine:pdlp")


def family_of(case: str) -> str:
    p = case.split(":")
    if p[0] == "MM":
        return "maros_meszaros"
    return (p[1] if p[0] == "H" else p[0])


def instance_features(case, d, plan, pplan) -> dict:
    return {"family": family_of(case), "n": int(d["P"].shape[0]), "m": int(d["A"].shape[0]), "nnzA": int(d["A"].nnz),
            "single_block_kind": plan["kind"] if plan else None, "single_block_n": plan["n_x"] if plan else 0,
            "product_blocks": pplan["K"] if pplan else 0, "product_cover": pplan["cover"] if pplan else 0.0}


def neighbors(vault, case, device, tol) -> list[dict]:
    """Best measured time per formulation on OTHER instances of the same family on this device."""
    fam, key = family_of(case), f"{tol:g}"
    best = defaultdict(dict)
    for e in vault.query("evidence", lambda p: p.get("device") == device and p.get("case") != case
                         and family_of(p.get("case", "x:0")) == fam):
        p = e.payload
        N = (p.get("N_at") or {}).get(key)
        if N:
            best[p["case"]][p["formulation"]] = min(best[p["case"]].get(p["formulation"], 1e30), N)
    return [{"case": c, "N_by_formulation": v} for c, v in best.items()]


def full_evidence_table(vault, case, device, tol):
    """For a held-out case: per formulation, N at tol (or None) and the best calibrated T, plus exploration wall."""
    key = f"{tol:g}"
    rows = vault.evidence(case=case, device=device)
    N, T, wall = {}, defaultdict(lambda: math.inf), defaultdict(float)
    for r in rows:
        p = r.payload
        f = p.get("formulation")
        if f is None:
            continue
        if "N_at" in p:
            n = p["N_at"].get(key)
            if n and (N.get(f) is None or n < N[f]):
                N[f] = n
            N.setdefault(f, None)
            wall[f] += p.get("wall_s", 0.0)
    for r in rows:
        p = r.payload
        f = p.get("formulation")
        if p.get("calibrated") and N.get(f):
            T[f] = min(T[f], p["setup_s"] + N[f] * p["per_iter_s"])
    return N, dict(T), dict(wall)


def evaluate_rule(code: str, vault, heldout_cases, device, tol, featurize) -> dict:
    ns = {}
    exec(code, ns)
    select = ns["select"]
    regrets, saved, total, lost = [], 0.0, 0.0, []
    for case in heldout_cases:
        N, T, wall = full_evidence_table(vault, case, device, tol)
        if not T:
            continue
        best = min(T.values())
        chosen = set(select(featurize(case), neighbors(vault, case, device, tol)))
        avail = {f: t for f, t in T.items() if f in chosen}
        total += sum(wall.values())
        saved += sum(w for f, w in wall.items() if f not in chosen)
        if not avail:
            lost.append(case)
            continue
        regrets.append(min(avail.values()) / best)
    geo = math.exp(sum(map(math.log, regrets)) / len(regrets)) if regrets else math.inf
    ok = bool(regrets) and not lost and geo <= 1.05 and max(regrets) <= 1.5 and saved > 0
    return {"admitted": ok, "geo_regret": geo, "worst_regret": max(regrets) if regrets else None,
            "cases_without_solution": lost, "exploration_saved_s": saved, "exploration_total_s": total,
            "n_heldout": len(regrets)}


def candidate_summaries(vault, device):
    """What a cost-model term may use about each component: its certificate and its calibrated cost on this device,
    measured on OTHER instances (never on the instance being predicted)."""
    out = {}
    for c in vault.current("component"):
        cal = (c.payload.get("calibration") or {}).get(device) or {}
        out[c.id] = {"set": c.payload.get("set"), "certified": (c.payload.get("certificate") or {}).get("certified"),
                     "per_call_us": cal.get("per_call_us"), "per_iteration_us_in_solver": cal.get("per_iteration_us_in_solver")}
    return out


def evaluate_cost_term(code, vault, cases, device, tol, featurize) -> dict:
    """A cost-model term predicts the per-iteration cost of a (formulation, component) pair on an instance it has not
    been calibrated on:  predict(features, candidate, summaries, neighbors) -> seconds per iteration (or None).

    ADMISSION, on held-out instances with complete calibration evidence: for every calibrated pair, compare the
    prediction with the measurement; require the median absolute log ratio <= 0.35 (about 1.4x), and require that
    choosing by predicted total time keeps the decision (geometric-mean regret <= 1.05, worst <= 1.5). A term that
    passes lets the router skip calibration runs, which is where a first visit spends its budget.
    """
    ns = {}
    exec(code, ns)
    predict = ns["predict"]
    summaries = candidate_summaries(vault, device)
    errs, regrets, n_pairs = [], [], 0
    key = f"{tol:g}"
    for case in cases:
        rows = vault.evidence(case=case, device=device)
        N, meas = {}, {}
        for r in rows:
            p = r.payload
            f, c = p.get("formulation"), p.get("component")
            if f is None:
                continue
            n = (p.get("N_at") or {}).get(key)
            if n and (N.get(f) is None or n < N[f]):
                N[f] = n
            if p.get("calibrated") and p.get("per_iter_s") is not None:
                meas[(f, c)] = (p["setup_s"], p["per_iter_s"])
        if not meas or not N:
            continue
        feats = featurize(case)
        pred_T, true_T = {}, {}
        for (f, c), (setup, per_iter) in meas.items():
            if not N.get(f):
                continue
            n_pairs += 1
            true_T[(f, c)] = setup + N[f] * per_iter
            try:
                ph = predict(feats, {"formulation": f, "component": c}, summaries, neighbors(vault, case, device, tol))
            except Exception:
                ph = None
            if ph is None or not (ph > 0):
                continue
            errs.append(abs(math.log(ph / per_iter)))
            pred_T[(f, c)] = setup + N[f] * ph
        if pred_T and true_T:
            pick = min(pred_T, key=pred_T.get)
            regrets.append(true_T[pick] / min(true_T.values()))
    med = sorted(errs)[len(errs) // 2] if errs else math.inf
    geo = math.exp(sum(map(math.log, regrets)) / len(regrets)) if regrets else math.inf
    ok = bool(errs) and bool(regrets) and med <= 0.35 and geo <= 1.05 and max(regrets) <= 1.5
    return {"admitted": ok, "median_abs_log_ratio": med, "coverage_pairs": len(errs), "pairs_seen": n_pairs,
            "geo_regret": geo, "worst_regret": max(regrets) if regrets else None, "n_cases": len(regrets)}


def evaluate_product_rule(code, vault, cases, device, tol, make_case, measure_fn, cap=20000, max_cases=3) -> dict:
    """A product ROW-SELECTION rule: score(info) -> priority per candidate row (higher is absorbed first), where info
    has arrays row, nnz, l2, amax, is_eq.

    ADMISSION is by MEASUREMENT, because the whole point of the MIPLIB finding is that neither coverage nor the
    operator norm predicts the iteration count: on up to `max_cases` workload instances with product structure, run the
    relocated formulation under the proposed rule and under the shipped rules, and require no instance worse than 1.05x
    the best shipped rule's iteration count and at least one instance at 0.8x or better.
    """
    ns = {}
    exec(code, ns)
    score = ns["score"]
    rows, wins = [], 0
    for case in cases[:max_cases]:
        d = make_case(case)
        base_N, new_N = None, None
        for rule in ("width", "norm"):
            N = measure_fn(d, rule, cap, tol)
            if N and (base_N is None or N < base_N):
                base_N = N
        new_N = measure_fn(d, score, cap, tol)
        rows.append({"case": case, "shipped_best_N": base_N, "proposed_N": new_N})
        if base_N and new_N:
            if new_N > 1.05 * base_N:
                return {"admitted": False, "reason": f"worse on {case}: {new_N} vs {base_N}", "cases": rows}
            if new_N <= 0.8 * base_N:
                wins += 1
        elif base_N and not new_N:
            return {"admitted": False, "reason": f"did not reach the target on {case} while a shipped rule did",
                    "cases": rows}
    return {"admitted": wins > 0, "wins": wins, "cases": rows,
            "reason": "" if wins else "no instance improved by 20% or more"}
