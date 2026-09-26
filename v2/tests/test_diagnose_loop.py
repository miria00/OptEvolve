"""The diagnose -> mutate -> re-run loop reproduces the hand-built chain on a scripted evaluator,
and runs end to end on the real screened runner."""
import jax
import numpy as np

jax.config.update("jax_enable_x64", True)


def _scripted(cfg):
    # screening cuts t but its resets cost N; keeping momentum restores N; buckets remove churn
    s, ev, rp = 1.0, 192, 0
    if cfg.get("screen"):
        s, ev, rp = 0.80, 256, 3
        if cfg.get("momentum") == "keep":
            s, ev = 0.62, 192
        if cfg.get("buckets"):
            s, rp = s - 0.08, 2
    if cfg.get("dtype") == "float32":
        s *= 1.05
    return {"seconds": s, "evals": ev, "converged": True, "replans": rp, "replan_at": [64] if rp else [],
            "nnz": 197, "p": 2000}


def test_chain_follows_the_diagnosis():
    from atlas.evolve.diagnose import evolve
    hw = {"step_GBps": 909, "peak_GBps": 1008, "f32_XTv_GBps": 298, "f64_XTv_GBps": 871}
    out = evolve(_scripted, {"K": 64, "screen": False}, {"seconds": 0.5, "evals": 35}, hw, log=lambda *_: None)
    kept = [c["symptom"] for c in out["chain"][1:]]
    assert kept[0] == "sparse_solution" and "replan_resets_momentum" in kept and "shape_churn" in kept
    assert out["best"]["config"]["momentum"] == "keep" and out["best"]["config"]["buckets"] is True
    # anderson and the Halpern anchor now route to the KB composer, so they are runnable; what stays
    # unrunnable is what neither engine expresses
    stubs = {s["ingredient"] for s in out["not_runnable"]}
    assert stubs == {"fused_gradient_kernel", "variable_metric_schedule", "quasi_newton_metric"}
    assert "anderson" not in stubs and "halpern_anchor" not in stubs


def test_evolve_runs_on_the_screened_runner():
    from atlas.bench import logreg_cell as lc
    from atlas.evolve.diagnose import evolve
    from atlas.evolve.screened_runner import run_screened
    rng = np.random.default_rng(3); n, p = 1500, 200
    X = rng.normal(size=(n, p)); w = np.zeros(p); w[:10] = rng.normal(size=10)
    y = np.sign(X @ w + 0.3 * rng.normal(size=n)); y[y == 0] = 1
    lam = 0.1 * lc.lam_max(X, y, 0.5)

    def ev(cfg):
        # the loop now spans two engines; this evaluator owns only the screened one, so it routes and
        # declines composed configurations rather than handing operator genes to a runner without them
        from atlas.evolve.diagnose import route
        if route(cfg)["engine"] == "composed":
            return {"seconds": float("inf"), "evals": 0, "converged": False, "p": p, "nnz": p}
        r = run_screened(X, y, lam, 0.5, tol=1e-8, **cfg)
        return {**r, "seconds": r["wall_s"], "p": p}
    out = evolve(ev, {"K": 64, "screen": False}, {"seconds": 1e-9, "evals": 10}, {}, max_rounds=3, min_gain=-1e9,
                 log=lambda *_: None)
    assert all(c["config"] is not None for c in out["chain"]) and out["best"]["converged"]


def test_oversized_working_set_halves_the_start():
    from atlas.evolve.diagnose import propose
    cur = {"seconds": 0.23, "evals": 64, "converged": True, "replans": 1, "replan_at": [32], "nnz": 197, "p": 2000,
           "active_sizes": [250, 500], "config": {"K": 32, "working_set": True, "screen": True, "buckets": True}}
    deltas = [pr["delta"] for pr in propose(cur, None, {"seconds": 0.57, "evals": 42}, {})]
    assert {"ws_init": 0.0625} in deltas


def test_blind_arm_uses_the_same_budget_and_records_a_monotone_trace():
    from atlas.evolve.diagnose import evolve
    hw = {"step_GBps": 909, "peak_GBps": 1008, "f32_XTv_GBps": 298, "f64_XTv_GBps": 871}
    kw = dict(max_rounds=50, patience=50, budget=12, log=lambda *_: None)
    g = evolve(_scripted, {"K": 64, "screen": False}, {"seconds": 0.1, "evals": 35}, hw, **kw)
    b = evolve(_scripted, {"K": 64, "screen": False}, {"seconds": 0.1, "evals": 35}, hw, proposer="blind", seed=3, **kw)
    assert b["evaluated"] == 12 and g["evaluated"] <= 12
    for tr in (g["trace"], b["trace"]):
        vals = [v for _, v in tr]
        assert all(x >= y for x, y in zip(vals, vals[1:]))
    assert {c["symptom"] for c in b["chain"][1:]} <= {"random"}


def test_blind_arm_on_the_large_space_only_proposes_real_genes():
    import numpy as np
    from atlas.evolve.diagnose import GENE_SPACE_LARGE, blind_proposals, evolve
    rng = np.random.default_rng(0)
    for pr in blind_proposals({"K": 64}, rng, 50, GENE_SPACE_LARGE):
        (g, v), = pr["delta"].items()
        assert g in GENE_SPACE_LARGE and v in GENE_SPACE_LARGE[g]
    out = evolve(_scripted, {"K": 64, "screen": False}, {"seconds": 0.1, "evals": 35}, {}, proposer="blind", seed=1,
                 budget=10, max_rounds=50, patience=50, space=GENE_SPACE_LARGE, log=lambda *_: None)
    assert out["evaluated"] == 10


def test_rules_found_by_a_blind_run_are_proposed():
    from atlas.evolve.diagnose import propose
    cur = {"seconds": 0.064, "evals": 32, "converged": True, "replans": 1, "replan_at": [16], "nnz": 196, "p": 2000,
           "active_sizes": [125, 250], "config": {"K": 16, "working_set": True, "screen": True, "buckets": True,
                                                  "step_rule": "power", "ws_init": 0.0625}}
    props = propose(cur, None, {"seconds": 0.166, "evals": 37}, {})
    assert {"power_iters": 10} in [pr["delta"] for pr in props]
    # the finer-cadence rule was removed (K=8 on the best configuration took 0.089 s against 0.067 s); an older
    # rule (late_first_replan) may still halve K for its own reason
    assert "finer_cadence_on_working_set" not in {pr["symptom"] for pr in props}


def test_cadence_rule_at_K32_and_capped_guided_respects_the_cap():
    from atlas.evolve.diagnose import propose, evolve
    cur = {"seconds": 0.064, "evals": 64, "converged": True, "replans": 1, "replan_at": [32], "nnz": 196, "p": 2000,
           "active_sizes": [125, 250], "config": {"K": 32, "working_set": True, "screen": True, "buckets": True,
                                                  "step_rule": "power", "ws_init": 0.0625}}
    assert {"K": 16} in [pr["delta"] for pr in propose(cur, None, {"seconds": 0.166, "evals": 37}, {})]
    counts = []
    def ev(cfg):
        counts.append(cfg); return _scripted(cfg)
    evolve(ev, {"K": 64, "screen": False}, {"seconds": 0.1, "evals": 200}, {}, guided_cap=2, beam=1, max_rounds=3,
           patience=10, log=lambda *_: None)
    assert len(counts) <= 1 + 3 * 2


def test_route_separates_the_two_engines_and_names_ignored_genes():
    from atlas.evolve.diagnose import route
    screened = {"K": 64, "screen": True, "working_set": True, "step_rule": "power"}
    assert route(screened)["engine"] == "screened"
    composed = {**screened, "anchor": True, "base": "fbs"}
    r = route(composed)
    assert r["engine"] == "composed" and r["operator_genes"] == ["anchor"]
    # the composer cannot honour the screened-runner genes, and says so instead of silently dropping them
    assert set(r["ignored"]) >= {"screen", "working_set", "step_rule"}


def test_the_three_chain_moves_build_recipes_the_gate_admits():
    """anchor -> restart on the anchor -> reflection under the anchor: each step is a single gene, and
    the gate (not the loop) decides admissibility. Reflection is admissible ONLY under the anchor."""
    import pytest
    from atlas.evolve.diagnose import config_to_recipe
    from atlas.evolve.kb_composer import typecheck_recipe
    from atlas.typing_ import TypeCheckError
    from atlas.bench import logreg_cell as lc
    rng = np.random.default_rng(5); n, p = 600, 60
    X = rng.normal(size=(n, p)) / np.sqrt(p)
    y = np.sign(X @ rng.normal(size=p) + 0.3 * rng.normal(size=n)); y[y == 0] = 1
    prob = lc.composite(X, y, lam=0.05 * lc.lam_max(X, y, 0.5), alpha=0.5)
    cfg = {"K": 64, "base": "fbs"}
    for delta in ({"anchor": True}, {"restart": ("adaptive", 0.5)}, {"rho": 1.0}):
        cfg = {**cfg, **delta}
        typecheck_recipe(config_to_recipe(cfg), prob)          # each step of the chain is admitted
    assert config_to_recipe(cfg).ingredients() == ["halpern", "adaptive_restart", "krasnoselskii_mann"] or set(
        config_to_recipe(cfg).ingredients()) == {"halpern", "adaptive_restart", "krasnoselskii_mann"}
    # the same reflection WITHOUT the anchor is refused: rho*alpha = 1 is merely nonexpansive, and KM can cycle
    with pytest.raises(TypeCheckError):
        typecheck_recipe(config_to_recipe({"base": "fbs", "rho": 1.0}), prob)


def test_anchor_chain_rules_fire_on_recorded_evidence():
    """The chain symptoms read quantities the composer records, not invented fields."""
    from atlas.evolve.diagnose import propose
    base = {"seconds": 1.0, "evals": 4096, "converged": True, "nnz": 50, "p": 2000}
    anchored = {**base, "config": {"K": 64, "base": "fbs", "anchor": True}, "restarts": 0, "averagedness": 0.667}
    deltas = [pr["delta"] for pr in propose(anchored, None, {"seconds": 0.5, "evals": 100}, {})]
    assert {"restart": ("adaptive", 0.5)} in deltas
    restarted = {**base, "config": {"K": 64, "base": "fbs", "anchor": True, "restart": ("adaptive", 0.5)},
                 "restarts": 9, "averagedness": 0.667}
    assert {"rho": 1.0} in [pr["delta"] for pr in propose(restarted, None, {"seconds": 0.5, "evals": 100}, {})]


def _operator_scripted(cfg):
    """The shape of the real lineage: the anchor alone is SLOWER than the start, restarting it is still
    slower, and only reflection under the anchor wins. A beam that expands only its fastest node cannot
    reach the third step."""
    if not cfg.get("anchor"):
        return {"seconds": 1.00, "evals": 4096, "converged": True, "nnz": 50, "p": 2000}
    s, ev, restarts = 1.40, 5600, 0
    if cfg.get("restart"):
        s, ev, restarts = 1.10, 4400, 9
        if cfg.get("rho") == 1.0:
            s, ev = 0.55, 2200
    return {"seconds": s, "evals": ev, "converged": True, "nnz": 50, "p": 2000,
            "restarts": restarts, "averagedness": 0.667}


def test_forcing_walks_the_three_step_operator_chain_through_two_losses():
    from atlas.evolve.diagnose import evolve
    out = evolve(_operator_scripted, {"K": 64, "screen": False}, {"seconds": 0.30, "evals": 100}, {},
                 max_rounds=6, log=lambda *_: None)
    best = out["best"]["config"]
    assert best.get("anchor") and best.get("restart") and best.get("rho") == 1.0
    assert [c["symptom"] for c in out["chain"][1:]] == ["iteration_gap", "anchor_never_restarts",
                                                        "contraction_budget_unspent"]
    # the two intermediate steps really were losses, and were expanded anyway
    mid = [c["seconds"] for c in out["chain"][1:3]]
    assert all(t > out["chain"][0]["seconds"] for t in mid) and out["best"]["seconds"] < out["chain"][0]["seconds"]
