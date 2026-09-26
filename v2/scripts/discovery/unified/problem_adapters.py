"""Problem-class adapters for the router: one interface over QP, LP (P = 0) and SOCP instances, and over CPU/GPU
ENGINES (external solvers the vault may route an instance to, e.g. an interior-point method for tiny instances whose
GPU iterations are kernel-launch bound).

  kind(d)                       "socp" | "lp" | "qp"
  reference(d)                  (objective, info) from a high-accuracy reference solve in an isolated interpreter
  plans(d)                      structure plans: single dominant block, product family, cone family
  build(d, cand, plan, kw)      a Formulation for our first-order method
  ladder(form, d, obj_ref, ...) iterations to acceptance at the target, with deadline
  accept(x, d, obj_ref, tol)    (ok, violation, objective gap) on the ORIGINAL problem
  engines(kind)                 engine names the vault can route to for this class
  structural_engines(d)         extra engines unlocked by RECOGNIZING the instance (see specialist.py)
  run_engine(d, name, tol, s)   {"accepted", "t_solver_s", "wall_s"} judged by the same acceptance check
"""
from __future__ import annotations

import os
import time

import numpy as np
import scipy.sparse as sp

import uni

# ECOS is a native second-order cone solver that was installed the whole time and was never offered a cone program,
# by the baseline field OR by the router. Measured 2026-09-16 on the held-out set: it is the ONLY solver in the field
# that reaches our bar on CBLIB:qssp60 (1.38 s at tol 1e-3, where Clarabel and SCS both miss), so leaving it out of
# this list would have completed the field for the oracle we are compared against and not for us.
ENGINES = {"qp": ("clarabel", "piqp", "highs"), "lp": ("highs", "clarabel", "pdlp"),
           "socp": ("clarabel", "scs", "ecos")}


def kind(d) -> str:
    if d.get("cones"):
        return "socp"
    return "lp" if sp.csc_matrix(d["P"]).nnz == 0 else "qp"


def normalize(d):
    """CBLIB's loader names the cone list 'soc'; socp.py names it 'cones'."""
    if "soc" in d and "cones" not in d:
        d = dict(d)
        d["cones"] = d.pop("soc")
    return d


def reference(d):
    if kind(d) == "socp":
        import socp
        obj, x = socp.reference(d)
        return obj, (None if obj is None else "clarabel 1e-9")
    return uni.reference(d)


def plans(d):
    k = kind(d)
    if k == "socp":
        import socp
        cplan, _ = socp.plan_cone(d)
        return {"single": None, "product": None, "cone": cplan}
    plan, _ = uni.plan_block(d, uni.diagnose(d))
    pplan, _ = uni.plan_product(d)
    return {"single": plan, "product": pplan, "cone": None}


def build(d, cand, plan, kw):
    if kind(d) == "socp":
        import socp
        return socp.build_socp_formulation(d, cand, plan=plan)
    return uni.build_formulation(d, cand, plan=plan, **kw)


def ladder(form, d, obj_ref, tols, cap, K, deadline):
    if kind(d) == "socp":
        import socp
        return socp.measure_ladder_socp(form, d, obj_ref, tols=tols, cap=cap, K=K, deadline=deadline)
    return uni.measure_ladder(form, d, obj_ref, tols=tols, cap=cap, K=K, deadline=deadline)


def accept(x, d, obj_ref, tol):
    if kind(d) == "socp":
        import socp
        return socp.accept_socp(x, d, obj_ref, tol)
    return uni.accept(x, d, obj_ref, tol)


def engines(k):
    return ENGINES.get(k, ())


def structural_engines(d):
    """Engines that become available only once the instance's FORMULATION is recognized.

    A generic QP solver is chosen by problem class; a lasso coordinate-descent solver can only be offered after the
    block structure of (P, q, A, l, u) has been checked to be a lasso encoding. Measured on our own family (4090,
    2026-09-15): at n = 100,000 the recognized route finishes in 0.048 s where the unrecognized routes take 17.6 s
    (ours, GPU) and roughly 10^3 s (Clarabel). The whole gap is that the specialist solves the n-dimensional problem
    on its 2,807-column support instead of a 250,000-variable QP.
    """
    out = ()
    try:
        import specialist
        if specialist.detect_lasso(d) is not None:
            out += tuple(f"lasso:{s}" for s in specialist.SPECIALISTS)
    except ImportError:
        pass
    try:
        import specialist_struct
        out += tuple(specialist_struct.structural_engines(d))
    except ImportError:
        pass
    return out


# A solver's eps is ITS OWN residual measure; our acceptance bar is a per-row relative feasibility violation plus a
# relative objective gap against an independent reference. They are not the same quantity, so running a solver at the
# eps whose numeral matches our tol understates it. Measured on the held-out set (2026-09-16): with only the two rungs
# (tol, tol/100) there were 176 records where a solver returned a point and lost to our check alone, several of them
# converging monotonically on the bar and stopping one rung short. Four rungs, and the fastest one our check accepts
# wins. The loop is still bounded by the caller's remaining budget, so a deep rung cannot run past it.
EPS_MULTS = (1.0, 1e-2, 1e-4, 1e-6)


def run_engine(d, name, obj_ref, tol, timeout_s):
    """Run an external solver up the eps ladder until OUR check accepts, or the caller's budget runs out.
    Wall time (process start, conversion, solve) is what a first visit pays; t_solver_s is the routed cost."""
    t0 = time.perf_counter()
    os.environ.setdefault("OPTEVOLVE_BASELINE_HARD_MARGIN", "5")     # engines must respect the router's budget
    if name.startswith("lasso:"):
        import specialist
        return specialist.run_specialist(d, name.split(":", 1)[1], obj_ref, tol, timeout_s)
    if name.split(":", 1)[0] in ("flow", "glasso", "meb"):
        import specialist_struct
        return specialist_struct.run_struct_specialist(d, name, obj_ref, tol, timeout_s)
    last = {"accepted": False}
    for eps in (tol * m for m in EPS_MULTS):
        left = timeout_s - (time.perf_counter() - t0)
        if left <= 0.5:
            break
        if kind(d) == "socp":
            from atlas.bench.conic_baseline_proc import run_conic_baseline_proc
            rb = run_conic_baseline_proc(d, name, eps, timeout_s=left)
            t_solver = rb.get("t_solver_s", rb.get("t_s"))
        else:
            from atlas.bench.baseline_proc import run_baseline_proc
            rb = run_baseline_proc(d, name, eps, timeout_s=left)
            t_solver = rb.get("t_s")
        if "error" in rb or rb.get("x") is None:
            last = {"accepted": False, "error": str(rb.get("error"))[:200]}
            continue
        ok, rv, gap = accept(np.asarray(rb["x"]), d, obj_ref, tol)
        last = {"accepted": bool(ok), "t_solver_s": t_solver, "eps": eps, "rv": rv, "gap": gap}
        if ok:
            break
    last["wall_s"] = time.perf_counter() - t0
    return last
