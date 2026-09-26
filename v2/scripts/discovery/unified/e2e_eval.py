"""BLACK-BOX ARM evaluator: what an agent sees without the principled tool.

The candidate operator is dropped into the solver on the target instance and the whole solve is timed; the answer is
accepted only if it passes the original-problem check (row violation and objective gap <= 1e-6), because any honest
agent evaluator checks the answer. There is NO per-call spec, NO operator certificate over the contract (mixed signs,
infinite bounds, edge cases), and NO cost-model decomposition: the only signal is end-to-end time on this instance.
Prints one JSON line.
"""
import json, os, sys, time, traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from uni import make_instance, permute_instance, diagnose, plan_block, build_formulation, reference, measure, time_clean

CASE = os.environ.get("E2E_CASE", "knapsack_ridge:1500:0.003")
CACHE = os.environ.get("E2E_CACHE", "")


def main(path):
    out = {"converged": 0.0, "N": None, "total_s": None, "error": ""}
    try:
        fam, n, rho = CASE.split(":")[0], int(CASE.split(":")[1]), float(CASE.split(":")[2])
        d = permute_instance(make_instance(fam, n, seed=0, rho=rho), 7)
        plan, _ = plan_block(d, diagnose(d))
        if CACHE and os.path.exists(CACHE):
            obj_ref = json.load(open(CACHE))["obj_ref"]
        else:
            obj_ref, _ = reference(d)
            if CACHE:
                json.dump({"obj_ref": obj_ref}, open(CACHE, "w"))
        form = lambda: build_formulation(d, "reloc", plan=plan, impl="file:" + path)
        m = measure(form(), d, obj_ref, tol=1e-6, cap=20000, K=10)
        out.update({"N": m["N"], "rv": m["row_violation"], "gap": m["obj_gap"]})
        if m["N"]:
            ts = [time_clean(form(), m["N"], K=10)["total_s"] for _ in range(3)]
            out.update({"converged": 1.0, "total_s": float(np.median(ts))})
    except Exception:
        out["error"] = traceback.format_exc()[-600:]
    print(json.dumps(out))


if __name__ == "__main__":
    main(sys.argv[1])
