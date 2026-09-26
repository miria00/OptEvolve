"""Diagnosis for the discovery loop: turn measurements into symptoms and symptoms into
knowledge-base mutations, the step that picked Halpern's anchor for PDHG because PDHG was a
nonexpansive map, and restarts because the anchor was slow late.

Inputs are plain records (ours, the baseline, hardware measurements); the output lists each
symptom with the evidence behind it and the KB ingredients whose effect addresses it, with the
factor (on N, iterations, or t, time per iteration) each is expected to change. This rule
table is the context the LLM generator in SkyDiscover would receive and extend; here it is
deterministic so its choices are auditable.
"""
from __future__ import annotations

import math

RULES = [
    # (symptom, test, KB ingredients, which factor, expectation)
    ("sparse_solution", lambda o, b, h: o["nnz"] / o["p"] < 0.3,
     ["gap_safe_screening"], "t", "tail iterations cost about nnz/p of a full step once screened"),
    ("memory_bound_step", lambda o, b, h: h.get("step_GBps", 0) >= 0.7 * h.get("peak_GBps", float("inf")),
     ["precision_f32_storage", "fused_gradient_kernel"], "t", "halving bytes per step halves t when bandwidth bound"),
    ("f32_transposed_product_slow", lambda o, b, h: h.get("f32_XTv_GBps", 1e9) < 0.5 * h.get("f64_XTv_GBps", 0),
     ["fused_gradient_kernel"], "t", "plain float32 cannot pay until the transposed product reaches bandwidth"),
    ("iteration_gap", lambda o, b, h: o["evals"] > 2.0 * b["evals"],
     ["halpern_anchor", "anderson", "variable_metric_schedule", "quasi_newton_metric"], "N",
     "baseline needs fewer gradient evaluations"),
]


def diagnose(ours: dict, baseline: dict, hw: dict) -> dict:
    t_ours = ours["seconds"] / max(ours["evals"], 1)
    t_base = baseline["seconds"] / max(baseline["evals"], 1)
    out = {"time_ratio": ours["seconds"] / baseline["seconds"], "N_ratio": ours["evals"] / baseline["evals"],
           "t_ratio": t_ours / t_base, "symptoms": [], "mutations": []}
    for name, test, ingredients, factor, why in RULES:
        if test(ours, baseline, hw):
            out["symptoms"].append({"symptom": name, "targets": factor, "why": why})
            for i in ingredients:
                if i not in [m["ingredient"] for m in out["mutations"]]:
                    out["mutations"].append({"ingredient": i, "because": name, "targets": factor})
    return out


# ---- the automatic chain: diagnose, mutate one ingredient, re-run, re-diagnose ----

# KB ingredient -> config change. Two engines are reachable from one configuration space:
# the screened FISTA runner (atlas.evolve.screened_runner), which owns the t-factor genes, and the
# KB composer (atlas.evolve.kb_composer), which owns the operator genes and runs any admitted
# composition of the knowledge base's ingredients. `route` decides which engine a configuration
# names. None still marks an ingredient neither engine can express; NOT_RUNNABLE says why.
MUTATIONS = {
    "gap_safe_screening": {"screen": True},
    "precision_f32_storage": {"dtype": "float32"},
    "halpern_anchor": {"anchor": True, "base": "fbs"},
    "anderson": {"anderson": 5, "base": "fbs"},
    "fused_gradient_kernel": None,
    "variable_metric_schedule": None,
    "quasi_newton_metric": None,
}

NOT_RUNNABLE = {
    "fused_gradient_kernel": "needs a hand-written fused gradient kernel; the FFI path is separate from both engines",
    "variable_metric_schedule": "the adaptive metric controller (atlas.backend.adaptive) runs primal-dual schemes, "
                                "not the forward-backward bases this cell uses",
    "quasi_newton_metric": "no quasi-Newton metric in the knowledge base's gate rules",
}

# Operator genes name a composed recipe; screened-only genes name the screened runner. A configuration
# carrying an operator gene is run by the composer, which cannot honour the screened-only genes: `route`
# reports them as ignored so the chain never claims a screening win it did not measure.
OPERATOR_GENES = ("anchor", "restart", "rho", "anderson", "step_frac", "ratio")
SCREENED_GENES = ("screen", "working_set", "buckets", "ws_init", "shrink", "momentum",
                  "step_rule", "L", "power_iters", "dtype")
_OFF = {"anchor": False, "restart": None, "rho": None, "anderson": 0, "step_frac": 1.0, "ratio": None}


def route(cfg: dict) -> dict:
    """Which engine runs this configuration, and which of its genes that engine ignores."""
    on = [g for g in OPERATOR_GENES if g in cfg and cfg[g] != _OFF[g]]
    if not on:
        return {"engine": "screened", "ignored": [], "operator_genes": []}
    ignored = [g for g in SCREENED_GENES if g in cfg and cfg[g] not in (None, False)]
    return {"engine": "composed", "ignored": ignored, "operator_genes": on}


def config_to_recipe(cfg: dict):
    """The composer recipe a configuration names. The gate (typecheck_recipe) still decides admissibility."""
    from atlas.evolve.kb_composer import Recipe
    return Recipe(base=cfg.get("base", "fbs"), step_frac=float(cfg.get("step_frac", 1.0)),
                  ratio=cfg.get("ratio"), rho=cfg.get("rho"), anchor=bool(cfg.get("anchor", False)),
                  restart=cfg.get("restart"), anderson=int(cfg.get("anderson", 0) or 0))


def _stalled(c: dict, factor: float = 4.0) -> bool:
    """The certificate falls, then crawls: decay per evaluation over the last third of the run is at
    least `factor` times worse than over the first third. Needs a recorded gap history."""
    h = [(e, g) for e, g in (c.get("gap_history") or []) if g > 0.0 and math.isfinite(g)]
    if len(h) < 6:
        return False
    k = max(len(h) // 3, 2)

    def rate(seg):
        de = seg[-1][0] - seg[0][0]
        return (math.log10(seg[0][1]) - math.log10(seg[-1][1])) / de if de > 0 else 0.0
    early, late = rate(h[:k]), rate(h[-k:])
    return early > 0.0 and late < early / factor

# Second-round symptoms compare a run with its parent (the run it was mutated from).
CHAIN_RULES = [
    ("replan_resets_momentum",
     lambda c, par: c.get("replans", 0) > 0 and c["config"].get("momentum", "reset") == "reset",
     lambda cfg: {"momentum": "keep"}, "N",
     "each re-plan restarts FISTA at t = 1; carry x, y and t onto the surviving columns"),
    ("shape_churn",
     lambda c, par: c.get("replans", 0) >= 1 and not c["config"].get("buckets", False),
     lambda cfg: {"buckets": True}, "t",
     "every exact-size re-plan compiles a new shape inside the solve; pad to halving buckets compiled from (n, p) alone"),
    ("cadence_quantisation",
     lambda c, par: c["config"].get("K", 64) >= 32 and c["evals"] % c["config"].get("K", 64) == 0,
     lambda cfg: {"K": cfg.get("K", 64) // 2}, "N",
     "the certificate is checked every K steps, so N is rounded up to a multiple of K"),
    ("gap_limited_screening",
     lambda c, par: bool(c.get("replan_at")) and c["replan_at"][0] >= 0.4 * c["evals"]
     and not c["config"].get("working_set", False),
     lambda cfg: {"working_set": True, "screen": True, "buckets": True}, "t",
     "safe screening waits for a small gap, so early steps pay for all p columns; start on a working set "
     "ranked by dual scores and let the full-problem certificate add violators"),
    ("oversized_working_set",
     lambda c, par: c["config"].get("working_set", False) and bool(c.get("active_sizes"))
     and c["active_sizes"][0] >= c.get("nnz", 0) > 0 and c["config"].get("ws_init", 0.125) > 1.0 / 64,
     lambda cfg: {"ws_init": cfg.get("ws_init", 0.125) / 2}, "t",
     "the first working set already holds as many columns as the final support; start smaller and let "
     "the certificate add what is missing"),
    ("step_estimate_cost",
     lambda c, par: c["config"].get("working_set", False) and c["config"].get("step_rule") == "power"
     and c["config"].get("power_iters", 30) > 10,
     lambda cfg: {"power_iters": 10}, "t",
     "the power-iteration step estimate is repeated at every re-plan; on a small working set 10 iterations "
     "suffice and the certificate guards the result (found by a blind run)"),
    ("step_size_cost",
     lambda c, par: c["config"].get("working_set", False) and c["config"].get("step_rule") == "power",
     lambda cfg: {"step_rule": "gram"}, "t",
     "on a working set of a few hundred columns the exact Lipschitz constant (top eigenvalue of X_W^T X_W) "
     "costs less than power iteration and is never an underestimate"),
    # The operator chain: anchor, then restart the anchor, then reflect under it. Each step is ONE gene,
    # each is gate-checked before it runs, and each is FORCED (expanded even when slower than its parent):
    # every intermediate step of the PDHG -> restart -> anchor -> reflection lineage is a loss on its own,
    # so a beam that only ever expands its fastest node cannot walk it.
    ("residual_stall",
     lambda c, par: _stalled(c) and not c["config"].get("anchor", False) and not c["config"].get("anderson", 0),
     lambda cfg: {"anchor": True, "base": "fbs"}, "N",
     "the certificate falls quickly and then crawls: the averaged map's tail is the O(1/sqrt k) residual decay "
     "the Halpern anchor replaces with O(1/k); FISTA momentum is not an averaged map, so the anchor needs the "
     "forward-backward base"),
    ("anchor_never_restarts",
     lambda c, par: c["config"].get("anchor", False) and not c["config"].get("restart"),
     lambda cfg: {"restart": ("adaptive", 0.5)}, "N",
     "the anchor pulls every step back toward one x0 that is never refreshed, so it is slow late; restart it "
     "when the fixed-point residual has halved (Applegate et al. 2021, on the residual)"),
    ("anchor_restarts_too_rarely",
     lambda c, par: (c["config"].get("anchor", False) and isinstance(c["config"].get("restart"), tuple)
                     and c["config"]["restart"][0] == "adaptive" and c.get("restarts", 0) <= 1),
     lambda cfg: {"restart": ("adaptive", min(0.9, float(cfg["restart"][1]) * 2.0))}, "N",
     "the anchor restarted at most once in the whole solve; a larger beta triggers it sooner"),
    ("contraction_budget_unspent",
     lambda c, par: (c["config"].get("anchor", False) and c["config"].get("restart")
                     and c["config"].get("rho") is None and c.get("averagedness", 2.0) <= 1.0),
     lambda cfg: {"rho": 1.0}, "N",
     "the anchored map is nonexpansive and the relaxation is still 1, so the contraction budget is unspent: "
     "the gate admits the maximum reflection (rho_abs = rho/alpha), which halves the Halpern rate constant "
     "(Sun et al. SIOPT 2025; the r2HPDHG composition)"),
    ("late_first_replan",
     lambda c, par: bool(c.get("replan_at")) and c["replan_at"][0] >= 0.4 * c["evals"] and c["config"].get("K", 64) > 8,
     lambda cfg: {"K": cfg.get("K", 64) // 2}, "t",
     "screening waits for a certificate check; a shorter cadence screens earlier"),
]


# Steps of the operator chain are followed even when they are slower than their parent, because every
# intermediate step of that lineage is a loss on its own. Nothing else is forced.
FORCED_SYMPTOMS = {"residual_stall", "anchor_never_restarts", "anchor_restarts_too_rarely",
                   "contraction_budget_unspent"}
FORCED_INGREDIENTS = {"halpern_anchor"}


def propose(cur: dict, parent, baseline: dict, hw: dict, mutations: dict | None = None) -> list:
    out = []
    for name, test, delta, factor, why in CHAIN_RULES:
        if test(cur, parent):
            out.append({"symptom": name, "ingredient": None, "delta": delta(cur["config"]), "targets": factor,
                        "why": why, "force": name in FORCED_SYMPTOMS})
    for m in diagnose(cur, baseline, hw)["mutations"]:
        out.append({"symptom": m["because"], "ingredient": m["ingredient"],
                    "delta": (mutations or MUTATIONS).get(m["ingredient"]),
                    "targets": m["targets"], "why": next(r[4] for r in RULES if r[0] == m["because"]),
                    "force": m["ingredient"] in FORCED_INGREDIENTS})
    return out


# ---- the control arm: timing-only mutation over the same gene space ----

GENE_SPACE = {
    "K": [8, 16, 32, 64],
    "screen": [False, True],
    "momentum": ["reset", "keep"],
    "buckets": [False, True],
    "working_set": [False, True],
    "ws_init": [0.25, 0.125, 0.0625, 0.03125],
    "step_rule": ["power", "gram"],
    "dtype": ["float64", "float32"],
}
GENE_DEFAULTS = {"K": 64, "screen": False, "momentum": "reset", "buckets": False, "working_set": False,
                 "ws_init": 0.125, "step_rule": "power", "dtype": "float64", "power_iters": 30, "shrink": 0.8,
                 "base": "fista"}
# A larger space of real solver knobs, most of which matter little on a given instance: it asks whether
# the diagnosis matters more as the space a blind search must cover grows (the guided arm is unchanged).
GENE_SPACE_LARGE = {**GENE_SPACE, "K": [4, 8, 16, 32, 64, 128],
                    "ws_init": [0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625],
                    "power_iters": [10, 30, 100], "shrink": [0.5, 0.8, 0.95], "base": ["fista", "fbs"]}


def blind_proposals(cfg: dict, rng, n: int, space: dict | None = None) -> list:
    """n random single-gene changes of cfg, chosen without any diagnosis: the control arm of the
    guided-versus-blind ablation. Same gene space and same beam as the guided arm; only the choice of
    mutation differs, and it sees nothing but certified wall time."""
    full = {**GENE_DEFAULTS, **cfg}
    moves = [(gname, v) for gname, vals in (space or GENE_SPACE).items() for v in vals if v != full[gname]]
    pick = rng.choice(len(moves), size=min(n, len(moves)), replace=False)
    return [{"symptom": "random", "ingredient": None, "delta": {moves[i][0]: moves[i][1]}, "targets": None,
             "why": "timing only"} for i in pick]


def evolve(evaluate, start: dict, baseline: dict, hw: dict, *, max_rounds: int = 6, min_gain: float = 0.02,
           beam: int = 2, patience: int = 2, log=print, proposer: str = "guided", seed: int = 0, per_node: int = 4,
           budget: int | None = None, space: dict | None = None, guided_cap: int | None = None,
           abandon_factor: float = 10.0, mutations: dict | None = None):
    """Beam search over single mutations. Each round expands the `beam` fastest certified configurations
    not yet expanded: diagnose each against its parent, evaluate every runnable mutation. A mutation is
    kept in the pool even if it is slower, because a diagnosed step can enable the next one (screening
    only pays once its re-planning costs are fixed); a strictly greedy chain stops there. Stops after
    `patience` rounds without a min_gain improvement of the best. The reported chain is the ancestry of
    the best configuration. `evaluate(config)` returns seconds, evals, converged (and replans,
    replan_at, nnz, p when relevant). proposer="blind" replaces the diagnosis by `per_node` random
    single-gene changes (blind_proposals, seeded); `budget` caps the number of evaluated configurations,
    start included, so both arms can be compared at the same cost. `trace` records the best certified
    time after every evaluation."""
    import numpy as _np
    rng = _np.random.default_rng(seed)
    key = lambda cfg: tuple(sorted(cfg.items()))
    # A forced chain step is followed ONCE. The same delta merged with different parents makes different
    # node keys, so without this a catastrophic move is re-run from every parent: on the logistic cell the
    # anchor cost 292 s against 0.143 s for the best configuration and was evaluated twice, and Anderson
    # 51.7 s twice, which is most of an hour spent re-learning the same answer. A delta that lands more
    # than `abandon_factor` times worse than the best certified time is not proposed again.
    abandoned, delta_seen = {}, set()
    dkey = lambda d: tuple(sorted((k, str(v)) for k, v in d.items()))
    root = key(start)
    nodes = {root: {"rec": {**evaluate(start), "config": dict(start)}, "parent": None, "via": None, "round": 0}}
    expanded, skipped = set(), []
    best, stale = root, 0
    best_so_far = lambda: min((nodes[k]["rec"]["seconds"] for k in nodes if nodes[k]["rec"]["converged"]), default=float("inf"))
    trace = [(1, best_so_far())]
    spent = lambda: budget is not None and len(nodes) >= budget
    log(f"round 0 {start}: {nodes[root]['rec']['seconds']:.3f}s {nodes[root]['rec']['evals']} evals; "
        f"baseline {baseline['seconds']:.3f}s")
    for rnd in range(1, max_rounds + 1):
        ranked = sorted((k for k in nodes if nodes[k]["rec"]["converged"] and k not in expanded),
                        key=lambda k: nodes[k]["rec"]["seconds"])
        frontier = ranked[:beam]
        # a forced node is expanded even if it is slow or did not certify: that is what lets the chain
        # walk through an intermediate step that loses on its own
        # A forced node is expanded even when slower than its parent, but NOT when it is hopeless. The
        # `abandoned` set stops a delta being re-PROPOSED; without this a hopeless node is still EXPANDED
        # and its own chain rules propose fresh deltas from it (the anchor at 292 s converged, so
        # anchor_never_restarts then proposed restarting it, another ~290 s, and each evaluation is a
        # warm-up plus reps). A merely slow or uncertified node is still expanded: that is the case
        # forcing exists for.
        hopeless = lambda k: nodes[k]["rec"]["seconds"] > abandon_factor * best_so_far()
        forced_ready = [k for k in nodes if k not in expanded and k not in frontier
                        and nodes[k].get("force") and nodes[k]["rec"].get("error") is None]
        for k in forced_ready:
            if hopeless(k):
                expanded.add(k)                       # retire it: never expanded, never reconsidered
                log(f"round {rnd}: not expanding {nodes[k]['rec']['config']} "
                    f"({nodes[k]['rec']['seconds']:.3f}s is over {abandon_factor:g}x the best "
                    f"{best_so_far():.3f}s); that lineage is abandoned")
        frontier = frontier + [k for k in forced_ready if k not in expanded]
        if not frontier:
            log(f"round {rnd}: nothing left to expand; stop"); break
        for k in frontier:
            expanded.add(k)
            node = nodes[k]; cur = node["rec"]
            par = nodes[node["parent"]]["rec"] if node["parent"] is not None else None
            if proposer == "guided":
                props = propose(cur, par, baseline, hw, mutations=mutations)
                if guided_cap:
                    # attack the dominant factor of T = N x t first: compare our evaluation count and time per
                    # evaluation with the baseline's, then keep the first guided_cap runnable proposals
                    n_ratio = cur["evals"] / max(baseline["evals"], 1)
                    t_ratio = (cur["seconds"] / max(cur["evals"], 1)) / (baseline["seconds"] / max(baseline["evals"], 1))
                    dom = "N" if n_ratio >= t_ratio else "t"
                    runnable = [pr for pr in props if pr["delta"] is not None]
                    runnable.sort(key=lambda pr: pr["targets"] != dom)          # stable: rule order within a factor
                    seen_d, kept = set(), []
                    for pr in runnable:
                        kd = tuple(sorted(pr["delta"].items()))
                        if kd not in seen_d and tuple(sorted({**cur["config"], **pr["delta"]}.items())) not in nodes:
                            seen_d.add(kd); kept.append(pr)
                    props = kept[:guided_cap] + [pr for pr in props if pr["delta"] is None]
            else:
                props = blind_proposals(cur["config"], rng, per_node, space)
            for pr in props:
                if spent():
                    break
                if pr["delta"] is None:
                    skipped.append({"round": rnd, **pr}); continue
                dk = dkey(pr["delta"])
                if dk in abandoned:
                    skipped.append({"round": rnd, **pr, "abandoned_after_s": abandoned[dk]}); continue
                cfg = {**cur["config"], **pr["delta"]}
                kk = key(cfg)
                if kk in nodes:
                    continue
                r = {**evaluate(cfg), "config": cfg}
                nodes[kk] = {"rec": r, "parent": k, "via": pr, "round": rnd, "force": bool(pr.get("force"))}
                delta_seen.add(dk)
                if r["seconds"] > abandon_factor * best_so_far():
                    abandoned[dk] = r["seconds"]
                    log(f"round {rnd}: abandoning {pr['delta']} ({r['seconds']:.3f}s is over {abandon_factor:g}x "
                        f"the best {best_so_far():.3f}s); it will not be tried from other parents")
                trace.append((len(nodes), best_so_far()))
                log(f"round {rnd} from {cur['seconds']:.3f}s: {pr['symptom']} -> {pr['delta']}: {r['seconds']:.3f}s "
                    f"{r['evals']} evals{'' if r['converged'] else ' NOT CERTIFIED'}")
        if spent():
            log(f"round {rnd}: evaluation budget {budget} spent; stop")
        top = min((k for k in nodes if nodes[k]["rec"]["converged"]), key=lambda k: nodes[k]["rec"]["seconds"])
        if nodes[top]["rec"]["seconds"] < (1 - min_gain) * nodes[best]["rec"]["seconds"]:
            best, stale = top, 0
            log(f"round {rnd}: best {nodes[best]['rec']['config']} {nodes[best]['rec']['seconds']:.3f}s "
                f"vs baseline {baseline['seconds']:.3f}s")
        else:
            pending = any(k not in expanded and nodes[k].get("force") and nodes[k]["rec"].get("error") is None
                          for k in nodes)
            stale = stale if pending else stale + 1
            if stale >= patience:
                log(f"round {rnd}: no improvement for {patience} rounds; stop"); break
        if spent():
            break
    path, k = [], best
    while k is not None:
        path.append(k); k = nodes[k]["parent"]
    chain = []
    for k in reversed(path):
        nd = nodes[k]; r = nd["rec"]; via = nd["via"] or {}
        chain.append({"round": nd["round"], "config": dict(r["config"]), "seconds": r["seconds"], "evals": r["evals"],
                      "symptom": via.get("symptom"), "ingredient": via.get("ingredient"), "delta": via.get("delta"),
                      "why": via.get("why")})
    tried = [{"config": nodes[k]["rec"]["config"], "seconds": nodes[k]["rec"]["seconds"], "evals": nodes[k]["rec"]["evals"],
              "converged": nodes[k]["rec"]["converged"], "round": nodes[k]["round"]} for k in nodes]
    return {"chain": chain, "best": nodes[best]["rec"], "beats_baseline": nodes[best]["rec"]["seconds"] < baseline["seconds"],
            "not_runnable": skipped, "tried": tried, "trace": trace, "evaluated": len(nodes)}
