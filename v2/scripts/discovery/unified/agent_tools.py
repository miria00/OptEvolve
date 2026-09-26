"""Solver-side implementations of the agent's tools, run in the JAX environment (one process per call).
Usage: agent_tools.py <overview|solve|evolve|rule> '<json payload>'  -> prints one JSON line."""
import json, os, sys, time, traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vault import Vault

HERE = os.path.dirname(os.path.abspath(__file__))


def current_cost_term(vault):
    terms = vault.query("transformation", lambda p: p.get("name") == "cost_term" and p.get("admitted"))
    if not terms:
        return None
    ns = {}
    exec(open(terms[-1].payload["file"]).read(), ns)
    return ns["predict"]


def current_rule(vault):
    rules = vault.query("transformation", lambda p: p.get("name") == "planner_rule" and p.get("admitted"))
    if not rules:
        return None
    ns = {}
    exec(open(rules[-1].payload["file"]).read(), ns)
    return ns["select"]


def main():
    fn, payload = sys.argv[1], json.loads(sys.argv[2])
    V = Vault(payload["vault"])
    if fn == "overview":
        out = {"vault": V.describe(), "summary": V.summary(), "progress": payload.get("progress")}
    elif fn == "solve":
        from router import solve
        r = solve(payload["case"], V, device=payload["device"], tol=payload["tol"], budget_s=payload["budget_s"],
                  selector=current_rule(V), cost_term=current_cost_term(V))
        out = {k: r[k] for k in ("case", "solved", "time_to_solution_s", "wall_total_s", "choice", "explored", "feedback")}
    elif fn == "evolve":
        from evolve import evolve_operator
        out = evolve_operator(V, payload["spec"], payload["device"], payload["iterations"],
                              run_root=os.path.join(os.path.dirname(payload["vault"]), "runs"),
                              llm_base=payload["llm_base"],
                              lock=os.path.join(os.path.dirname(payload["vault"]), "gpu.lock"))
    elif fn == "product_rule":
        from rules import evaluate_product_rule
        from router import make_case, _rule_fn
        import problem_adapters as PA
        from uni import plan_product

        def measure_fn(d, rule, cap, tol):
            pl, _ = plan_product(d, rule=rule)
            if pl is None:
                return None
            form = PA.build(d, "reloc_rows", pl, {"impl": "bisect"})
            m = PA.ladder(form, d, obj_ref[0], (tol,), cap, 50, time.perf_counter() + 300)
            return m["N_at"].get(f"{tol:g}")

        obj_ref = [None]

        def mk(case):
            d = PA.normalize(make_case(case))
            obj_ref[0] = PA.reference(d)[0]
            return d

        ev = evaluate_product_rule(payload["code"], V, [c for c in payload["cases"]], payload["device"],
                                   payload["tol"], mk, measure_fn)
        if ev["admitted"]:
            name = payload.get("rule_name") or f"rule{int(time.time())}"
            path = os.path.join(os.path.dirname(payload["vault"]), "components", f"product_rule_{name}.py")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            open(path, "w").write(payload["code"])
            V.add("transformation", {"name": "product_rule", "rule_name": name, "admitted": True, "file": path,
                                     "summary": payload.get("rationale", ""), "evaluation": ev},
                  provenance={"origin": "agent propose_product_rule"})
        out = ev
    elif fn == "cost_term":
        from rules import evaluate_cost_term, instance_features
        from router import make_case
        from uni import diagnose, plan_block, plan_product
        import problem_adapters as PA

        def featurize(case):
            d = PA.normalize(make_case(case))
            pl = PA.plans(d)
            return instance_features(case, d, pl["single"], pl["product"])

        ev = evaluate_cost_term(payload["code"], V, payload["cases"], payload["device"], payload["tol"], featurize)
        if ev["admitted"]:
            path = os.path.join(os.path.dirname(payload["vault"]), "components", f"cost_term_{int(time.time())}.py")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            open(path, "w").write(payload["code"])
            V.add("transformation", {"name": "cost_term", "admitted": True, "file": path,
                                     "summary": payload.get("rationale", ""), "evaluation": ev},
                  provenance={"origin": "agent propose_cost_term"})
        out = ev
    elif fn == "rule":
        from rules import evaluate_rule, instance_features
        from router import make_case
        from uni import diagnose, plan_block, plan_product

        def featurize(case):
            d = make_case(case)
            plan, _ = plan_block(d, diagnose(d))
            pplan, _ = plan_product(d)
            return instance_features(case, d, plan, pplan)

        ev = evaluate_rule(payload["code"], V, payload["cases"], payload["device"], payload["tol"], featurize)
        if ev["admitted"]:
            path = os.path.join(os.path.dirname(payload["vault"]), "components", f"rule_{int(time.time())}.py")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            open(path, "w").write(payload["code"])
            V.add("transformation", {"name": "planner_rule", "admitted": True, "file": path,
                                     "summary": payload.get("rationale", ""), "evaluation": ev},
                  provenance={"origin": "agent propose_planner_rule"})
        out = ev
    else:
        out = {"error": f"unknown fn {fn}"}
    print(json.dumps(out, default=str))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print(json.dumps({"error": traceback.format_exc()[-1500:]}))
