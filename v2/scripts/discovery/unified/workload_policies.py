"""Compare TOOL ALONE, AGENT ALONE and MERGED on the timed workload (see workload_eval.py).

A policy picks one candidate per instance from the candidates it is allowed to use; its cost on that instance is the
candidate's measured total time, or a FAILURE if that candidate did not reach original-problem acceptance. Reported:
total workload time over instances every policy solved, geometric-mean slowdown vs the best candidate anyone had,
failures, and the split between ladder instances and held-out ones.
"""
import argparse, json, math
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("--file", required=True)
ap.add_argument("--agent-op", required=True, help="candidate name of the black-box arm's operator, e.g. reloc:bb0")
ap.add_argument("--certified-ops", default="", help="comma list of candidate names certified from the spec arm")
a = ap.parse_args()
rows = [json.loads(l) for l in open(a.file)]
by = defaultdict(dict)
held = {}
for r in rows:
    by[r["case"]][r["cand"]] = r
    held[r["case"]] = r["heldout"]
cert = [c for c in a.certified_ops.split(",") if c]
T = lambda r: r.get("total_s") if r and r.get("N") else None
policies = {
    "tool_alone(shipped: std, reloc:bisect)": lambda c: [k for k in ("std_raw", "std_ruiz", "reloc:bisect") if k in c],
    "tool_alone(+hand breakpoint,newton12)": lambda c: [k for k in ("std_raw", "std_ruiz", "reloc:bisect",
                                                                    "reloc:breakpoint", "reloc:newton12") if k in c],
    "agent_alone(black-box operator, relocated)": lambda c: [a.agent_op] if a.agent_op in c else [],
    "merged(tool + certified synthesized ops)": lambda c: [k for k in ("std_raw", "std_ruiz", "reloc:bisect")
                                                           if k in c] + [k for k in cert if k in c],
}
best = {case: min([T(r) for r in c.values() if T(r)] or [None]) if any(T(r) for r in c.values()) else None
        for case, c in by.items()}
print("%-44s %-8s %-9s %-10s %-10s %s" % ("policy", "solved", "failed", "geo_ladder", "geo_heldout", "choices"))
for name, allowed in policies.items():
    solved, failed, ratios = 0, [], {"ladder": [], "heldout": []}
    picks = {}
    for case, c in by.items():
        if best[case] is None:
            continue
        opts = [(T(c[k]), k) for k in allowed(c) if T(c[k])]
        if not opts:
            failed.append(case); continue
        t, k = min(opts)
        solved += 1
        picks[case] = k
        ratios["heldout" if held[case] else "ladder"].append(t / best[case])
    g = lambda v: ("%.3f" % math.exp(sum(map(math.log, v)) / len(v))) if v else "-"
    print("%-44s %-8d %-9d %-10s %-10s" % (name, solved, len(failed), g(ratios["ladder"]), g(ratios["heldout"])))
    for case in by:
        print("      %-34s %s" % (case, picks.get(case, "FAILED" if case in failed else "-")))
