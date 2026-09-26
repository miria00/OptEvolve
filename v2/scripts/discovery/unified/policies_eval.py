"""WHAT "BEATS" MEANS, measured: budgeted solve rate and time to a correct solution on HELD-OUT instances.

For every policy and every held-out case (a FIRST visit: the case has no evidence in the policy's vault):
  solved      the answer passes the original-problem check at tol within a wall-clock budget B that INCLUDES the
              policy's own exploration and calibration
  time        wall time to the verified solution (first visit), and the routed time on a repeat visit
Policies:
  tool_alone   router + the seed vault (shipped components and transformations, no evolution)
  merged       router + the vault after an agent session (admitted components and planner rules)
  agent_alone  a fixed pipeline with an operator evolved WITHOUT the tool (end-to-end fitness on a suite): relocated
               formulation with that operator when the structure exists, Ruiz-scaled standard otherwise
Baselines (the bar to beat) are measured separately by the virtual-best harness on the same cells.
Output: one JSONL record per (policy, case); summarize with --summarize.
"""
import argparse, json, math, os, shutil, sys, time, traceback
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ap = argparse.ArgumentParser()
ap.add_argument("--cases", default="")
ap.add_argument("--policy", choices=("tool_alone", "merged", "agent_alone"))
ap.add_argument("--vault", help="vault for tool_alone / merged (copied; held-out evidence is never written back)")
ap.add_argument("--agent-op", help="operator file for agent_alone")
ap.add_argument("--device", default="rtx4090", help="the MACHINE key; with --device-select its CPU is <key>_cpu")
ap.add_argument("--device-select", action="store_true", help="offer {gpu, cpu} of this machine as candidates")
ap.add_argument("--no-engines", action="store_true", help="our own arm only: isolates the device decision")
ap.add_argument("--ref-cache", default="", help="cached reference objectives (the judge's oracle, not a policy cost)")
ap.add_argument("--case-cache", default="", help="serialized instances for a machine that cannot parse the source")
ap.add_argument("--tol", type=float, default=1e-3)
ap.add_argument("--budget-s", type=float, default=60.0)
ap.add_argument("--out")
ap.add_argument("--summarize", nargs="*")
a = ap.parse_args()


def summarize(files):
    rows = [json.loads(l) for f in files for l in open(f) if l.strip()]
    by = defaultdict(list)
    for r in rows:
        by[r["policy"]].append(r)
    cases = sorted({r["case"] for r in rows})
    print("%-12s %-8s %-10s %-14s %-14s" % ("policy", "solved", "of", "geo_time_s", "geo_repeat_s"))
    for p, rs in by.items():
        ok = [r for r in rs if r["solved"]]
        g = lambda v: math.exp(sum(map(math.log, v)) / len(v)) if v else float("nan")
        print("%-12s %-8d %-10d %-14.3f %-14.3f" % (p, len(ok), len(rs), g([r["first_visit_s"] for r in ok]),
                                                  g([r["repeat_s"] for r in ok if r.get("repeat_s")])))
    print("\nper case (first-visit seconds, FAIL = not solved within budget):")
    for c in cases:
        cells = []
        for p in by:
            r = next((r for r in by[p] if r["case"] == c), None)
            cells.append(f"{p}={'FAIL' if not r or not r['solved'] else '%.2f' % r['first_visit_s']}")
        print("  %-34s %s" % (c, "  ".join(cells)))


if a.summarize:
    summarize(a.summarize)
    sys.exit(0)

from router import make_case, solve  # noqa: E402
REFS = json.load(open(a.ref_cache)) if a.ref_cache else {}
if a.case_cache:
    sys.path.insert(0, os.path.dirname(os.path.abspath(a.case_cache.rstrip("/"))))
    from case_cache import load_case as _load_case
    _mk = lambda c: _load_case(c, a.case_cache, make_case)
else:
    _mk = make_case
from uni import diagnose, plan_block, build_formulation, reference, measure_ladder  # noqa: E402
from vault import Vault  # noqa: E402

out = open(a.out, "a")
work = os.path.join(os.path.dirname(os.path.abspath(a.out)), f"vault_{a.policy}_{int(time.time())}")
if a.policy in ("tool_alone", "merged"):
    os.makedirs(work, exist_ok=True)
    shutil.copy(a.vault, os.path.join(work, "vault.jsonl"))
    comps = os.path.join(os.path.dirname(a.vault), "components")
    if os.path.isdir(comps):
        shutil.copytree(comps, os.path.join(work, "components"), dirs_exist_ok=True)
    V = Vault(os.path.join(work, "vault.jsonl"))
    rule = None
    rules = V.query("transformation", lambda p: p.get("name") == "planner_rule" and p.get("admitted"))
    if rules:
        ns = {}
        exec(open(rules[-1].payload["file"]).read(), ns)
        rule = ns["select"]

for case in a.cases.split(","):
    rec = {"policy": a.policy, "case": case, "tol": a.tol, "budget_s": a.budget_s, "device": a.device}
    try:
        import problem_adapters as PA
        d = PA.normalize(_mk(case))
        obj_ref = REFS.get(case)
        if obj_ref is None:
            obj_ref, _ = PA.reference(d)
        if a.policy in ("tool_alone", "merged"):
            if V.evidence(case=case, device=a.device):
                rec["warning"] = "case already had evidence in this vault: NOT a first visit"
            t0 = time.perf_counter()
            kw = {"devices": ("gpu", "cpu") if a.device_select else None, "engines": not a.no_engines}
            r = solve(case, V, a.device, tol=a.tol, budget_s=a.budget_s, d=d, obj_ref=obj_ref, selector=rule, **kw)
            first = time.perf_counter() - t0
            rep = solve(case, V, a.device, tol=a.tol, budget_s=a.budget_s, d=d, obj_ref=obj_ref, selector=rule, **kw)
            fs = r.get("first_solution_s")
            rec.update({"solved": bool(r["solved"]) and fs is not None and fs <= a.budget_s,
                        "first_visit_s": fs if fs is not None else first, "wall_total_s": first,
                        "routed_time_s": r["time_to_solution_s"], "repeat_s": rep["wall_total_s"] if rep["solved"] else None,
                        "choice": r["choice"], "explored": r["explored"],
                        "device_chosen": r.get("device_chosen"), "devices": r.get("devices"),
                        "table": r.get("table"), "repeat_choice": rep.get("choice"),
                        "repeat_device": rep.get("device_chosen")})
        else:
            t0 = time.perf_counter()
            plan = PA.plans(d)["single"] if PA.kind(d) != "socp" else None
            if plan is not None and plan["kind"] in ("box_hyperplane", "box_slab"):
                form = PA.build(d, "reloc_rows", plan, {"impl": "file:" + a.agent_op})
            else:
                form = PA.build(d, "std_ruiz", None, {})            # without the tool there is no other structure to use
            m = PA.ladder(form, d, obj_ref, (a.tol,), 200000, 50, t0 + a.budget_s)
            first = time.perf_counter() - t0
            N = m["N_at"].get(f"{a.tol:g}")
            rec.update({"solved": N is not None and first <= a.budget_s, "first_visit_s": first,
                        "repeat_s": (m["setup_s"] + N * m["t_iter"]) if N else None,
                        "choice": {"formulation": form.name}})
    except Exception:
        rec.update({"solved": False, "error": traceback.format_exc()[-400:]})
    out.write(json.dumps(rec, default=str) + "\n"); out.flush()
    print(case, a.policy, "solved" if rec.get("solved") else "FAIL", "%.2f" % rec.get("first_visit_s", -1), flush=True)
