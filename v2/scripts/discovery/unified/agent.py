"""The merged system's agent: one LLM that pursues a workload goal by choosing, at every step, between USING the vault's
tool on the task and EVOLVING part of the tool. Both actions are function calls; all share one budget.

  vault_overview()                 components, transformations, lessons, open specs, and workload progress
  solve(case)                      route, run, verify, record evidence; returns time, choice, binding factor, spec
  evolve_operator(case, iters)     synthesize an exact operator against the spec solve() emitted for that case;
                                   certified results are admitted to the vault (charged to the synthesis budget)
  propose_planner_rule(code, why)  a select(features, neighbors) rule deciding which candidates to explore on new
                                   instances; admitted only if it keeps held-out decisions while saving exploration
  propose_product_rule(code, ...)  a score(info) rule deciding which family of rows the product planner absorbs;
                                   admitted only if measured iterations improve without regressing any tested instance
  propose_cost_term(code, why)     a predictor of a candidate's per-iteration cost, so calibration runs can be skipped
  finish(summary)

The goal given to the agent is the paper's definition of success: budgeted solve rate at the accuracy target, then
total time to correct solutions. Every step is logged for the paper.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from openai import OpenAI  # noqa: E402  (runs in the skyenv interpreter, which has openai; the solver runs in subprocesses)

HERE = os.path.dirname(os.path.abspath(__file__))
PY = "/home/miria/jaxenv2/bin/python"

TOOLS = [
    {"type": "function", "function": {"name": "vault_overview", "description": "Show the vault (components, "
     "transformations, lessons, open specs) and workload progress.", "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "solve", "description": "Solve one workload instance with the vault's "
     "router: returns solved, time to solution, the chosen formulation and operator, what binds (the formulation's "
     "iteration count or the operator's cost), a spec (kind decision_flip: a faster exact operator would change the "
     "decision; kind improvement: a faster exact operator would lower this and every same-structure instance's time), "
     "and exploration cost on a first visit.",
     "parameters": {"type": "object", "properties": {"case": {"type": "string"}}, "required": ["case"]}}},
    {"type": "function", "function": {"name": "evolve_operator", "description": "Synthesize a faster exact operator "
     "against the spec that solve() last emitted for this case. Costs synthesis iterations from the shared budget; "
     "certified results are added to the vault. Call solve(case) again afterwards to use them.",
     "parameters": {"type": "object", "properties": {"case": {"type": "string"},
                    "iterations": {"type": "integer", "description": "synthesis iterations to spend (8-40)"}},
                    "required": ["case", "iterations"]}}},
    {"type": "function", "function": {"name": "propose_planner_rule", "description": "Propose Python code defining "
     "select(features, neighbors) -> list of CANDIDATE keys to explore on a NEW instance. Keys are formulations "
     "(std_raw, std_ruiz, reloc, reloc_rows, reloc_prod, reloc_rows_prod, reloc_cone, reloc_cone_raw) AND engines the "
     "vault may route to (engine:clarabel, engine:highs, engine:piqp, engine:scs, engine:pdlp). A key you leave out is "
     "NOT explored, so dropping the engine that solves a family loses those instances. features: family, n, m, nnzA, "
     "single_block_kind, single_block_n, "
     "product_blocks, product_cover. neighbors: list of {case, N_by_formulation} for other instances of the same family. "
     "Admitted only if held-out decisions are kept (geomean regret <= 1.05) while exploration time is saved.",
     "parameters": {"type": "object", "properties": {"code": {"type": "string"}, "rationale": {"type": "string"}},
                    "required": ["code", "rationale"]}}},
    {"type": "function", "function": {"name": "propose_product_rule", "description": "Propose Python code defining "
     "score(info) -> a priority per candidate row for the product planner (higher is absorbed first). info has numpy "
     "arrays: row, nnz (support size), l2 (norm of the row's coefficients), amax, is_eq. The planner absorbs a "
     "DISJOINT family greedily in priority order, so a single wide row can block a whole family of narrow ones. "
     "Shipped rules are width (nnz), norm (l2) and narrow (-nnz). Admitted only if measured iterations on workload "
     "instances are no worse than 1.05x the best shipped rule everywhere and at least 0.8x somewhere.",
     "parameters": {"type": "object", "properties": {"code": {"type": "string"}, "rule_name": {"type": "string"},
                    "rationale": {"type": "string"}}, "required": ["code", "rule_name", "rationale"]}}},
    {"type": "function", "function": {"name": "propose_cost_term", "description": "Propose Python code defining "
     "predict(features, candidate, summaries, neighbors) -> seconds per iteration for a (formulation, component) pair "
     "on an instance it has NOT been calibrated on (return None when unsure). summaries: per component, its set, "
     "certificate and calibrated cost on this device from other instances. Admitted only if held-out predictions are "
     "within about 1.4x (median) and decisions are kept; an admitted term lets the router skip calibration runs, which "
     "is where a first visit spends its budget.",
     "parameters": {"type": "object", "properties": {"code": {"type": "string"}, "rationale": {"type": "string"}},
                    "required": ["code", "rationale"]}}},
    {"type": "function", "function": {"name": "finish", "description": "End the session with a short summary.",
     "parameters": {"type": "object", "properties": {"summary": {"type": "string"}}, "required": ["summary"]}}},
]

SYSTEM = """You operate an optimization system on a {device} GPU. Your workload is a list of optimization instances that
must be solved to relative accuracy {tol:g} (per-row feasibility and objective error, checked on the original problem).
SUCCESS, in order: (1) the number of instances solved correctly within {budget_s:.0f} s each on a FIRST visit, (2) the
total time to correct solutions. You share one budget: {max_llm_tokens} of your own generated tokens and
{max_evolve} synthesis iterations for evolve_operator.
You have two kinds of action. USE the tool: solve(case) routes the instance through the vault (formulation, operator,
device), verifies the answer, and tells you what binds and what spec would change the decision. EVOLVE the tool:
evolve_operator(case, iterations) synthesizes and certifies a faster exact operator against that spec, and it is then
reused on every instance with the same structure. propose_planner_rule changes which candidates are explored on a new
instance. propose_product_rule changes WHICH family of rows the planner absorbs, which matters because the widest row
can block a family that converges far faster. propose_cost_term predicts a candidate's cost so the router can skip
calibration runs. Every proposal is checked before admission. Evolve when solve() feedback shows a spec worth meeting,
exploration cost dominating, or a formulation that should have been considered; otherwise keep solving. Read the lessons in vault_overview before
proposing rules. Be concise. Call finish when the budget is nearly spent or nothing more is worth doing."""


def run_solver_tool(fn: str, payload: dict, timeout=3600) -> dict:
    """Solver-side tools run in the JAX environment through agent_tools.py (one process per call)."""
    import subprocess
    env = dict(os.environ)
    env.pop("LD_LIBRARY_PATH", None)
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    p = subprocess.run([PY, os.path.join(HERE, "agent_tools.py"), fn, json.dumps(payload)], capture_output=True,
                       text=True, timeout=timeout, env=env)
    lines = [l for l in p.stdout.splitlines() if l.startswith("{")]
    return json.loads(lines[-1]) if lines else {"error": (p.stderr or p.stdout)[-800:]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload", required=True, help="comma-separated cases")
    ap.add_argument("--vault", required=True, help="working vault path (copied from --seed-vault if missing)")
    ap.add_argument("--seed-vault", default="/home/miria/OptEvolve/v2/vault/vault.jsonl")
    ap.add_argument("--device", default="rtx4090")
    ap.add_argument("--tol", type=float, default=1e-3)
    ap.add_argument("--budget-s", type=float, default=30.0)
    ap.add_argument("--max-llm-tokens", type=int, default=60000)
    ap.add_argument("--max-evolve", type=int, default=60)
    ap.add_argument("--max-turns", type=int, default=80)
    ap.add_argument("--llm", default="http://127.0.0.1:8001/v1")
    ap.add_argument("--synth-llm", default="http://127.0.0.1:8002/v1")
    ap.add_argument("--log", required=True)
    a = ap.parse_args()
    if not os.path.exists(a.vault):
        os.makedirs(os.path.dirname(os.path.abspath(a.vault)), exist_ok=True)
        shutil.copy(a.seed_vault, a.vault)
        comps = os.path.join(os.path.dirname(a.seed_vault), "components")
        if os.path.isdir(comps) and not os.path.exists(os.path.join(os.path.dirname(a.vault), "components")):
            shutil.copytree(comps, os.path.join(os.path.dirname(a.vault), "components"))
    cases = a.workload.split(",")
    client = OpenAI(base_url=a.llm, api_key="EMPTY")
    log = open(a.log, "a")
    state = {"tokens": 0, "evolve_used": 0, "progress": {c: None for c in cases}, "last_spec": {}}
    msgs = [{"role": "system", "content": SYSTEM.format(device=a.device, tol=a.tol, budget_s=a.budget_s,
                                                           max_llm_tokens=a.max_llm_tokens, max_evolve=a.max_evolve)},
            {"role": "user", "content": "Workload: " + ", ".join(cases)}]
    common = {"vault": a.vault, "device": a.device, "tol": a.tol, "budget_s": a.budget_s}

    def emit(kind, **kw):
        log.write(json.dumps({"t": time.time(), "kind": kind, **kw}, default=str) + "\n"); log.flush()

    for turn in range(a.max_turns):
        if state["tokens"] >= a.max_llm_tokens:
            emit("budget_exhausted", what="llm_tokens")
            break
        r = client.chat.completions.create(model="Qwen/Qwen3.8-27B", messages=msgs, tools=TOOLS, max_tokens=8000,
                                           extra_body={"reasoning_effort": "low"})
        state["tokens"] += r.usage.completion_tokens
        m = r.choices[0].message
        msgs.append({"role": "assistant", "content": m.content or "",
                     "tool_calls": [tc.model_dump() for tc in (m.tool_calls or [])]})
        emit("assistant", content=m.content, tool_calls=[(tc.function.name, tc.function.arguments) for tc in (m.tool_calls or [])],
             tokens=r.usage.completion_tokens)
        if not m.tool_calls:
            msgs.append({"role": "user", "content": "Use a tool, or call finish."})
            continue
        done = False
        for tc in m.tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            t0 = time.time()
            if name == "vault_overview":
                res = run_solver_tool("overview", {**common, "progress": state["progress"]})
            elif name == "solve":
                res = run_solver_tool("solve", {**common, "case": args.get("case")})
                if "solved" in res:
                    state["progress"][args.get("case")] = {"solved": res["solved"], "time_s": res.get("time_to_solution_s")}
                    if (res.get("feedback") or {}).get("spec"):
                        state["last_spec"][args.get("case")] = res["feedback"]["spec"]
            elif name == "evolve_operator":
                case, iters = args.get("case"), int(args.get("iterations", 16))
                iters = max(0, min(iters, a.max_evolve - state["evolve_used"]))
                spec = state["last_spec"].get(case)
                if spec is None:
                    res = {"error": "no spec for this case: call solve(case) first (its feedback carries the spec)"}
                elif iters < 4:
                    res = {"error": "synthesis budget exhausted"}
                else:
                    state["evolve_used"] += iters
                    res = run_solver_tool("evolve", {**common, "spec": spec, "iterations": iters,
                                                     "llm_base": a.synth_llm}, timeout=6 * 3600)
            elif name == "propose_product_rule":
                res = run_solver_tool("product_rule", {**common, "code": args.get("code", ""),
                                                        "rule_name": args.get("rule_name", ""),
                                                        "rationale": args.get("rationale", ""),
                                                        "cases": [c for c, v in state["progress"].items() if v]},
                                      timeout=3 * 3600)
            elif name == "propose_cost_term":
                res = run_solver_tool("cost_term", {**common, "code": args.get("code", ""),
                                                    "rationale": args.get("rationale", ""),
                                                    "cases": [c for c, v in state["progress"].items() if v]})
            elif name == "propose_planner_rule":
                res = run_solver_tool("rule", {**common, "code": args.get("code", ""), "rationale": args.get("rationale", ""),
                                               "cases": [c for c, v in state["progress"].items() if v]})
            elif name == "finish":
                res, done = {"ok": True}, True
            else:
                res = {"error": f"unknown tool {name}"}
            res["budget_left"] = {"llm_tokens": a.max_llm_tokens - state["tokens"],
                                  "synthesis_iterations": a.max_evolve - state["evolve_used"]}
            emit("tool", name=name, args=args, result=res, wall_s=time.time() - t0)
            msgs.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(res, default=str)[:6000]})
        if done:
            break
    emit("end", progress=state["progress"], tokens=state["tokens"], evolve_used=state["evolve_used"])
    print(json.dumps({"progress": state["progress"], "tokens": state["tokens"], "evolve_used": state["evolve_used"]}))


if __name__ == "__main__":
    main()
