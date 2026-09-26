"""In-solver calibration of a candidate operator file on one case: setup and per-iteration cost of the relocated
formulation over a short clean run (lesson:isolated_timing_optimistic). Prints one JSON line."""
import json, os, sys, traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from router import make_case
from uni import diagnose, plan_block, plan_product, build_formulation, time_clean

case, formulation, path = sys.argv[1], sys.argv[2], sys.argv[3]
iters = int(os.environ.get("CAL_ITERS", "300"))
out = {"ok": False}
try:
    d = make_case(case)
    plan, why = plan_product(d) if os.environ.get("SPEC_PLAN") == "product" else plan_block(d, diagnose(d))
    if plan is None:
        raise RuntimeError(f"no plan: {why}")
    c = min((time_clean(build_formulation(d, formulation, plan=plan, impl="file:" + path), iters, K=50)
             for _ in range(2)), key=lambda c: c["per_iter_s"])
    out = {"ok": True, "setup_s": c["setup_s"], "per_iter_s": c["per_iter_s"]}
except Exception:
    out["error"] = traceback.format_exc()[-500:]
print(json.dumps(out))
