"""AGENT-ALONE evaluator over a SUITE (the fairness fix for the black-box arm): the candidate operator is put in the
relocated formulation of every suite case that has the structure, each solve is checked on the original problem at
E2E_TOL, and the fitness is the geometric mean over cases of T_best_standard / T_candidate, with a failed case counted
as a 10x slowdown. No certificate, no cost model: only end-to-end time and correctness on the suite. One JSON line."""
import json, math, os, sys, traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from router import make_case
from uni import diagnose, plan_block, build_formulation, reference, measure_ladder, time_clean

CASES = os.environ["E2E_CASES"].split(",")
TOL = float(os.environ.get("E2E_TOL", "1e-3"))
CACHE = os.environ["E2E_CACHE"]


def main(path):
    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    per = []
    for case in CASES:
        rec = {"case": case}
        try:
            d = make_case(case)
            if case not in cache:
                obj_ref, _ = reference(d)
                best = None
                for cand in ("std_raw", "std_ruiz"):
                    m = measure_ladder(build_formulation(d, cand), d, obj_ref, cap=200000, K=50)
                    N = m["N_at"].get(f"{TOL:g}")
                    if N:
                        t = time_clean(build_formulation(d, cand), N, K=50)["total_s"]
                        best = t if best is None else min(best, t)
                cache[case] = {"obj_ref": obj_ref, "T_std": best}
                json.dump(cache, open(CACHE, "w"))
            plan, _ = plan_block(d, diagnose(d))
            if plan is None:
                continue
            form = lambda: build_formulation(d, "reloc_rows", plan=plan, impl="file:" + path)
            m = measure_ladder(form(), d, cache[case]["obj_ref"], cap=20000, K=10)
            N = m["N_at"].get(f"{TOL:g}")
            if N:
                T = float(np.median([time_clean(form(), N, K=10)["total_s"] for _ in range(2)]))
                ratio = (cache[case]["T_std"] / T) if cache[case]["T_std"] else 10.0
                rec.update({"accepted": True, "N": N, "T": T, "ratio": ratio})
            else:
                rec.update({"accepted": False, "ratio": 0.1})
        except Exception:
            rec.update({"accepted": False, "ratio": 0.1, "error": traceback.format_exc()[-300:]})
        per.append(rec)
    ratios = [r["ratio"] for r in per if "ratio" in r]
    geo = math.exp(sum(math.log(max(x, 1e-6)) for x in ratios) / len(ratios)) if ratios else 0.0
    print(json.dumps({"geo_ratio": geo, "cases": per, "failed": sum(1 for r in per if not r.get("accepted"))}))


if __name__ == "__main__":
    main(sys.argv[1])
