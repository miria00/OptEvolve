"""THE BAR TO BEAT: every class-appropriate open-source solver on every held-out (instance, accuracy target) cell.

Each solver runs in its isolated interpreter at eps in {tol, tol/10, tol/100} (virtual_best.py takes the min over
settings our checker accepts, which is generous to the baselines; --matched-eps restricts to eps == tol) and every
returned point is judged by OUR original-problem check at tol. Output rows use virtual_best.py's record format.
Run on the same machine as the policies it is compared with, with no GPU work in flight.
"""
import argparse, json, os, sys, time, traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("OPTEVOLVE_BASELINE_ENV_ROOT", "/home/miria/baseline_envs" if os.path.isdir("/home/miria/baseline_envs")
                      else "/workspace:/root/baseline_envs")
import numpy as np
import problem_adapters as PA
from router import make_case

# ECOS speaks second-order cones natively and was already installed, so leaving it out of the SOCP field was the
# very error this measurement is about: a field that looks complete because every instance is inside every solver's
# class. Specialists (specialist.py, specialist_struct.py) are offered to the field on the same terms as any other
# engine: they are only RUN on an instance whose structure has been recognized, and they are judged by the same
# original-problem check, so the virtual best may use them against us.
SOLVERS = {"qp": ("osqp", "scs", "clarabel", "highs", "piqp", "proxqp", "ecos"),
           "lp": ("highs", "pdlp", "cupdlpx", "clarabel", "scs", "osqp", "piqp"),
           "socp": ("clarabel", "scs", "ecos")}
SPECIALIST_PREFIXES = ("lasso", "flow", "glasso", "meb")

ap = argparse.ArgumentParser()
ap.add_argument("--cases", required=True)
ap.add_argument("--tols", default="1e-3,1e-4")
ap.add_argument("--timeout", type=float, default=30.0)
ap.add_argument("--workers", type=int, default=6, help="cases in parallel; baselines are CPU-only")
ap.add_argument("--eps-mults", default="1,0.01", help="eps settings as multiples of tol (min over settings is taken)")
ap.add_argument("--ref-cache", default="", help="JSON file of case -> reference objective, so a costly reference "
                "solve (384 s on one SOCP) is paid once across runs")
ap.add_argument("--no-specialists", action="store_true", help="field of generic solvers only")
ap.add_argument("--out", required=True)
a = ap.parse_args()
done = set()
if os.path.exists(a.out):
    for line in open(a.out):
        r = json.loads(line)
        done.add((r["instance"], r["tol"], r["solver"], r.get("eps")))
MULTS = [float(x) for x in a.eps_mults.split(",")]


def reference_cached(case, d):
    if not a.ref_cache:
        return PA.reference(d)[0]
    cache = json.load(open(a.ref_cache)) if os.path.exists(a.ref_cache) else {}
    if case not in cache:
        cache[case] = PA.reference(d)[0]
        json.dump(cache, open(a.ref_cache, "w"))
    return cache[case]


def run_case(case):
    recs = []
    try:
        d = PA.normalize(make_case(case))
        k = PA.kind(d)
        obj_ref = reference_cached(case, d)
    except Exception:
        return [{"instance": case, "error": traceback.format_exc()[-300:]}]
    extra = tuple(PA.structural_engines(d)) if not a.no_specialists else ()
    for tol in (float(t) for t in a.tols.split(",")):
        for solver in tuple(SOLVERS[k]) + extra:
            for eps in (tol * m for m in MULTS):
                if (case, tol, solver, eps) in done:
                    continue
                rec = {"instance": case, "tol": tol, "solver": solver, "eps": eps, "kind": k}
                try:
                    if solver.split(":", 1)[0] in SPECIALIST_PREFIXES:
                        # A specialist is run once, at the requested tolerance: it has no eps ladder to sweep.
                        if eps != tol:
                            continue
                        rb = PA.run_engine(d, solver, obj_ref, tol, a.timeout)
                        rec["t_s"] = rb.get("t_solver_s")
                        rec["wall_s"] = rb.get("wall_s")
                        rec.update({"accepted": bool(rb.get("accepted")), "rv": rb.get("rv"), "gap": rb.get("gap"),
                                    "status": rb.get("error") or "specialist",
                                    "unsupported": bool(rb.get("unsupported"))})
                        recs.append(rec)
                        continue
                    if k == "socp":
                        from atlas.bench.conic_baseline_proc import run_conic_baseline_proc
                        rb = run_conic_baseline_proc(d, solver, eps, timeout_s=a.timeout)
                        rec["t_s"] = rb.get("t_solver_s", rb.get("t_s"))
                    else:
                        from atlas.bench.baseline_proc import run_baseline_proc
                        rb = run_baseline_proc(d, solver, eps, timeout_s=a.timeout)
                        rec["t_s"] = rb.get("t_s")
                    if rb.get("unsupported"):
                        rec.update({"accepted": False, "unsupported": True})
                    elif "error" in rb or rb.get("x") is None:
                        rec.update({"accepted": False, "error": str(rb.get("error"))[:200]})
                    else:
                        ok, rv, gap = PA.accept(np.asarray(rb["x"]), d, obj_ref, tol)
                        rec.update({"accepted": bool(ok), "rv": rv, "gap": gap, "status": rb.get("status")})
                except Exception:
                    rec.update({"accepted": False, "error": traceback.format_exc()[-200:]})
                recs.append(rec)
    return recs


if __name__ == "__main__":
    from multiprocessing import Pool
    out = open(a.out, "a")
    cases = a.cases.split(",")
    with Pool(min(a.workers, len(cases))) as pool:
        for recs in pool.imap_unordered(run_case, cases):
            for rec in recs:
                out.write(json.dumps(rec, default=float) + "\n")
            out.flush()
            print(recs[0]["instance"] if recs else "?", "done", len(recs), "records", flush=True)
    print("BASELINES_DONE", flush=True)
