"""Alternative SELECTION RULES for the product relocation, and the measurement that compares them.

uni.plan_product admits a row as absorbable when every column in its support carries a bound row, then packs a
DISJOINT family greedily in decreasing nonzero count. Disjointness is forced by the product structure (two blocks
sharing a column are not a product), so the only thing a planner can change is WHICH rows it takes and in what order.
The shipped order is one choice among several, and it optimizes column coverage, not iteration count.

The rules here vary that order only; admission, bound folding and the returned plan dict are identical to
uni.plan_product's, so build_formulation consumes them unchanged.

  greedy    (-nnz, r)                     the shipped rule: widest rows first
  rownorm   (-||a_r||_2, r)               widest rows are not the rows that dominate the operator. A row with many
                                          small entries can be wide and spectrally irrelevant, while a two-entry
                                          big-M row can dominate ||A||: on lotsize the widest absorbable rows have
                                          119 nonzeros and norm 10.9 while the largest-norm ones have 2 nonzeros and
                                          norm 20,019, so the greedy rule absorbs the wide rows and LEAVES the big-M
                                          rows in L.

MECHANISM, as measured rather than as assumed. The unscaled ||A_keep|| is what separates the two rules (47x on
nexp-150-20-8-5, 1835x on lotsize, 4192x on cbs-cta), and it is a good SCREEN for which instances to look at. It is
not the explanation: the reloc_rows candidate row-equilibrates whatever stays in L, so on nexp-150-20-8-5 the two
plans end up with nearly the same operator norm (6.69 greedy, 6.98 rownorm) and nearly the same per-iteration cost,
and the iteration counts still differ by more than 6x (greedy not converged at 18,550, rownorm 2,900). That is this
project's portfolio lesson on real data: row rescaling normalizes a dominant row's NORM but does not remove its
geometry from the operator, and only relocation does. The selection rule decides which rows get that treatment.
  eqfirst   (not equality, -nnz)          an equality row absorbs as a hyperplane block, an inequality row as a slab;
                                          the hyperplane pins a degree of freedom that the slab only bounds
  mindeg    (conflicts, -nnz)             fewest-conflicts-first, the standard independent-set heuristic: taking a
                                          wide row can block several others, so decreasing width is not the rule that
                                          maximizes covered columns
  objaware  (-nnz, r) skipping rows       the study's suggested rule: keep the columns carrying the largest objective
            touching top-|q| columns      coefficients out of the absorbed set, so the linear term still drives them
                                          through the gradient step rather than through a projection

score_plan adds the no-solve screen: ||A_keep||_2, the norm of the UNSCALED operator left in L after relocation.

Run as a script, this measures iterations to 1e-3 for std_ruiz and for every rule, one instance per subprocess:
  python miplib_planner.py --names a,b,c --out rules.jsonl [--budget 120] [--tol 1e-3]
  python miplib_planner.py --one NAME --budget 120                   # the unit the driver spawns
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.08")
os.environ.setdefault("OPTEVOLVE_BASELINE_ENV_ROOT", "/workspace:/root/baseline_envs")

RULES = ("greedy", "rownorm", "eqfirst", "mindeg", "objaware")


def _bound_map(A, l, u):
    """Per column: the interval its single-nonzero rows impose, which rows those are, and whether it is usable.

    Same folding as uni.plan_product (divide by the coefficient, flipping the interval for a negative one), written
    vectorized because MIPLIB relaxations have one bound row per bounded column and that is most of m.
    """
    m, n = A.shape
    nnz = np.diff(A.indptr)
    lo, hi = np.full(n, -np.inf), np.full(n, np.inf)
    has = np.zeros(n, dtype=bool)
    bound_rows_of = {}
    single = np.flatnonzero(nnz == 1)
    for r in single:
        c, v = int(A.indices[A.indptr[r]]), float(A.data[A.indptr[r]])
        if v == 0.0:
            continue
        rl, ru = (l[r] / v, u[r] / v) if v > 0 else (u[r] / v, l[r] / v)
        lo[c], hi[c] = max(lo[c], rl), min(hi[c], ru)
        has[c] = True
        bound_rows_of.setdefault(c, []).append(int(r))
    fin = has & (np.isfinite(lo) | np.isfinite(hi))
    return lo, hi, fin, bound_rows_of, nnz


def candidate_rows(A, fin, nnz, min_nnz=2):
    """uni.plan_product's admission test: a multi-nonzero row whose whole support is finitely bounded."""
    out = []
    for r in np.flatnonzero(nnz >= min_nnz):
        cols = A.indices[A.indptr[r]:A.indptr[r + 1]]
        vals = A.data[A.indptr[r]:A.indptr[r + 1]]
        if fin[cols].all() and np.all(vals != 0.0):
            out.append(int(r))
    return out


def _order(rule, A, l, u, q, cands, nnz):
    """The candidate rows in the order the rule wants them offered to the disjointness filter."""
    if rule == "greedy":
        return sorted(cands, key=lambda r: (-nnz[r], r))
    if rule == "rownorm":
        nrm = {r: float(np.linalg.norm(A.data[A.indptr[r]:A.indptr[r + 1]])) for r in cands}
        return sorted(cands, key=lambda r: (-nrm[r], r))
    if rule == "eqfirst":
        eq = np.isclose(l, u)
        return sorted(cands, key=lambda r: (not bool(eq[r]), -nnz[r], r))
    if rule == "mindeg":
        # conflict degree: how many OTHER candidate rows share a column with this one. Built through a column ->
        # candidate-rows index so it stays linear in the candidate nonzeros rather than quadratic in the rows.
        by_col = {}
        for r in cands:
            for c in A.indices[A.indptr[r]:A.indptr[r + 1]]:
                by_col.setdefault(int(c), []).append(r)
        deg = {}
        for r in cands:
            nb = set()
            for c in A.indices[A.indptr[r]:A.indptr[r + 1]]:
                nb.update(by_col[int(c)])
            deg[r] = len(nb) - 1
        return sorted(cands, key=lambda r: (deg[r], -nnz[r], r))
    if rule == "objaware":
        aq = np.abs(np.asarray(q, dtype=float))
        thr = np.quantile(aq[aq > 0], 0.95) if np.any(aq > 0) else np.inf
        hot = aq >= thr
        keep = [r for r in cands if not hot[A.indices[A.indptr[r]:A.indptr[r + 1]]].any()]
        return sorted(keep or cands, key=lambda r: (-nnz[r], r))
    raise ValueError(f"unknown rule {rule!r}")


def plan_product_rule(d, rule="greedy", min_nnz=2):
    """uni.plan_product with a pluggable selection order. Returns (plan, reason) in the same format.

    rule="greedy" reproduces uni.plan_product exactly, which is what makes the comparison a controlled one.
    """
    A = sp.csr_matrix(d["A"])
    l = np.asarray(d["l"], dtype=float)
    u = np.asarray(d["u"], dtype=float)
    q = np.asarray(d["q"], dtype=float)
    m, n = A.shape
    lo, hi, fin, bound_rows_of, nnz = _bound_map(A, l, u)
    cands = candidate_rows(A, fin, nnz, min_nnz)
    if not cands:
        return None, "no absorbable rows"
    used = np.zeros(n, dtype=bool)
    rows = []
    for r in _order(rule, A, l, u, q, cands, nnz):
        cols = A.indices[A.indptr[r]:A.indptr[r + 1]]
        if not used[cols].any():
            used[cols] = True
            rows.append(r)
    if not rows:
        return None, "no absorbable rows"
    rows = sorted(rows)                           # block order must not depend on the offer order
    T, bid, a = [], [], []
    for k, r in enumerate(rows):
        cols = A.indices[A.indptr[r]:A.indptr[r + 1]]
        order = np.argsort(cols)
        T.append(cols[order])
        a.append(A.data[A.indptr[r]:A.indptr[r + 1]][order])
        bid.append(np.full(cols.size, k))
    T, a, bid = np.concatenate(T), np.concatenate(a), np.concatenate(bid)
    removed = set(rows)
    for c in T.tolist():
        removed.update(bound_rows_of.get(c, []))
    keep = np.array(sorted(set(range(m)) - removed), dtype=int)
    fully_absorbed = keep.size == 0
    if fully_absorbed:
        keep = np.array([bound_rows_of[int(T[0])][0]], dtype=int)
    rest = np.array(sorted(set(range(n)) - set(T.tolist())), dtype=int)
    return {"kind": "product", "T": T, "bid": bid, "a": a, "lo": lo[T], "hi": hi[T],
            "b_lo": l[np.array(rows)], "b_hi": u[np.array(rows)], "K": len(rows), "rows": np.array(rows),
            "keep": keep, "perm": np.concatenate([T, rest]), "n_x": int(T.size), "cover": float(T.size / n),
            "fully_absorbed": bool(fully_absorbed), "rule": rule}, "RELOCATABLE_PRODUCT"


def opnorm(M, k=1):
    M = sp.csc_matrix(M).astype(np.float64)
    if M.shape[0] == 0 or M.shape[1] == 0 or M.nnz == 0:
        return 0.0
    if min(M.shape) <= 2:
        return float(np.linalg.norm(M.toarray(), 2))
    return float(svds(M, k=k, return_singular_vectors=False)[0])


def score_plan(d, plan):
    """No-solve summary of what a plan leaves behind: K, covered columns, and ||A_keep||_2, which sets the step."""
    A = sp.csc_matrix(d["A"])
    if plan is None:
        return {"K": 0, "n_x": 0, "cover": 0.0, "keep_rows": int(A.shape[0]), "normL_keep": opnorm(A)}
    keep = plan["keep"]
    return {"K": int(plan["K"]), "n_x": int(plan["n_x"]), "cover": float(plan["cover"]),
            "keep_rows": int(keep.size), "normL_keep": opnorm(A[keep]),
            "eq_blocks": int(np.sum(np.isclose(plan["b_lo"], plan["b_hi"])))}


# ----------------------------------------------------------------------------------------------------------------
# Measurement: iterations to the target for std_ruiz and for every rule's relocated formulation
# ----------------------------------------------------------------------------------------------------------------
def measure_one(name, tol=1e-3, budget_s=120.0, ref_timeout=900.0, rules=RULES, impl="bisect"):
    import miplib
    import uni
    from miplib_gap import reference_obj

    rec = {"name": name, "tol": tol, "budget_s": budget_s}
    d = miplib.load_miplib(name)
    A = sp.csc_matrix(d["A"])
    rec.update(n=int(A.shape[1]), m=int(A.shape[0]), nnz=int(A.nnz))
    obj_ref, ref = reference_obj(d, timeout_s=ref_timeout)
    rec["reference"] = ref
    if obj_ref is None:
        rec["error"] = "no usable reference"
        return rec
    rec["obj_ref"] = obj_ref
    tols = (1e-2, tol)
    key = f"{tol:g}"
    rec["arms"] = {}

    def run(label, cand, plan):
        t0 = time.perf_counter()
        arm = {"cand": cand}
        try:
            if plan is not None:
                arm.update(score_plan(d, plan))
            form = uni.build_formulation(d, cand, plan=plan, impl=impl)
            # wall cap only: the point of the comparison is N, so the cap must come from the budget, not a guess
            m = uni.measure_ladder(form, d, obj_ref, tols=tols, cap=2_000_000, K=50,
                                   deadline=time.perf_counter() + budget_s)
            arm.update(N_at=m["N_at"], N=m["N_at"].get(key), iters_run=m["iters_run"], setup_s=m["setup_s"],
                       t_iter=m["t_iter"], normL=m["normL"], row_violation=m["row_violation"], obj_gap=m["obj_gap"])
        except Exception as e:
            arm["error"] = f"{type(e).__name__}: {str(e)[:250]}"
        arm["wall_s"] = round(time.perf_counter() - t0, 2)
        rec["arms"][label] = arm

    run("std_ruiz", "std_ruiz", None)
    for rule in rules:
        plan, reason = plan_product_rule(d, rule)
        if plan is None:
            rec["arms"][rule] = {"plan_reason": reason}
            continue
        run(rule, "reloc_rows", plan)
        rec["arms"][rule]["plan_reason"] = reason
    # SCORE ON TIME, NOT ITERATIONS. This used to be argmin over N, which is not the quantity anyone cares about and
    # on this benchmark it inverts the answer: nexp-150-20-8-5 picked an arm at 2,900 iterations and 25.03 s over
    # std_ruiz at 7,700 iterations and 7.94 s, and 50v-10 picked 6,850 iterations at 49.63 s over 25,200 at 11.26 s.
    # Every declared winner was 1.21x to 4.41x SLOWER than the arm it beat. A relocated formulation buys iterations
    # and pays for them per iteration, so N alone can never decide it; the whole point of the cost model is that the
    # two terms have to be multiplied before they are compared.
    # Both keys required: an arm that gave up quickly has a small wall_s and no N, and must not win on time.
    ts = {k: v["wall_s"] for k, v in rec["arms"].items() if v.get("wall_s") and v.get("N")}
    ns = {k: v.get("N") for k, v in rec["arms"].items() if v.get("N")}
    rec["best_arm"] = min(ts, key=ts.get) if ts else None
    rec["best_arm_by_N"] = min(ns, key=ns.get) if ns else None    # kept so the disagreement stays visible
    rec["N_and_time_disagree"] = bool(rec["best_arm"] and rec["best_arm_by_N"]
                                      and rec["best_arm"] != rec["best_arm_by_N"])
    return rec


def sweep_one(name, rules=RULES, timeout=600.0):
    """Every rule's plan for one instance with no solve: coverage and the norm left in L, which is the population
    evidence behind the measured iteration counts (Condat-Vu takes s = 0.49 / ||L||)."""
    import signal

    import miplib

    def handler(signum, frame):
        raise TimeoutError(f"exceeded {timeout}s")

    rec = {"name": name}
    t0 = time.perf_counter()
    try:
        signal.signal(signal.SIGALRM, handler)
        signal.alarm(int(timeout))
        d = miplib.load_miplib(name)
        A = sp.csc_matrix(d["A"])
        rec.update(n=int(A.shape[1]), m=int(A.shape[0]), nnz=int(A.nnz))
        for rule in rules:
            plan, reason = plan_product_rule(d, rule)
            rec[rule] = {"reason": reason} if plan is None else score_plan(d, plan)
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {str(e)[:200]}"
    finally:
        signal.alarm(0)
    rec["s"] = round(time.perf_counter() - t0, 2)
    return rec


def _sweep_worker(args):
    name, rules, timeout = args
    try:
        return sweep_one(name, rules, timeout)
    except Exception as e:
        return {"name": name, "fatal_error": f"{type(e).__name__}: {str(e)[:300]}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-out", default="", help="no-solve rule sweep over the suite; writes JSONL here")
    ap.add_argument("--sweep-max-nnz", type=float, default=5e5)
    ap.add_argument("--sweep-workers", type=int, default=16)
    ap.add_argument("--sweep-timeout", type=float, default=600.0)
    ap.add_argument("--one", default="")
    ap.add_argument("--names", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--tol", type=float, default=1e-3)
    ap.add_argument("--budget", type=float, default=120.0)
    ap.add_argument("--ref-timeout", type=float, default=900.0)
    ap.add_argument("--proc-timeout", type=float, default=2400.0)
    ap.add_argument("--rules", default=",".join(RULES))
    args = ap.parse_args()
    rules = tuple(x for x in args.rules.split(",") if x)

    if args.sweep_out:
        import miplib
        from multiprocessing import get_context
        ents = [e for e in miplib.list_miplib() if (e["nnz"] or 0) <= args.sweep_max_nnz]
        todo = [(e["name"], rules, args.sweep_timeout) for e in sorted(ents, key=lambda e: -(e["nnz"] or 0))]
        t0, done = time.perf_counter(), 0
        with open(args.sweep_out, "w") as fh, get_context("spawn").Pool(args.sweep_workers,
                                                                        maxtasksperchild=4) as pool:
            for rec in pool.imap_unordered(_sweep_worker, todo):
                fh.write(json.dumps(rec, default=str) + "\n")
                fh.flush()
                done += 1
                if done % 20 == 0:
                    print(f"{done}/{len(todo)} {time.perf_counter() - t0:.0f}s", flush=True)
        print(f"sweep done {done} in {time.perf_counter() - t0:.0f}s -> {args.sweep_out}")
        return

    if args.one:
        try:
            rec = measure_one(args.one, args.tol, args.budget, args.ref_timeout, rules)
        except Exception as e:
            import traceback
            rec = {"name": args.one, "fatal_error": f"{type(e).__name__}: {str(e)[:300]}",
                   "traceback": traceback.format_exc()[-1200:]}
        print("@@REC@@" + json.dumps(rec, default=str), flush=True)
        return

    names = [x for x in args.names.split(",") if x]
    script = os.path.abspath(__file__)
    with open(args.out, "w") as fh:
        for i, name in enumerate(names, 1):
            cmd = [sys.executable, script, "--one", name, "--tol", repr(args.tol), "--budget", repr(args.budget),
                   "--ref-timeout", repr(args.ref_timeout), "--rules", ",".join(rules)]
            t0 = time.perf_counter()
            try:
                pr = subprocess.run(cmd, capture_output=True, text=True, timeout=args.proc_timeout,
                                    env=dict(os.environ))
                line = next((l for l in reversed(pr.stdout.splitlines()) if l.startswith("@@REC@@")), None)
                rec = json.loads(line[len("@@REC@@"):]) if line else {
                    "name": name, "driver_error": "no record emitted", "returncode": pr.returncode,
                    "stderr": (pr.stderr or "")[-1500:]}
            except subprocess.TimeoutExpired:
                rec = {"name": name, "driver_error": f"process timeout after {args.proc_timeout}s"}
            rec["driver_wall_s"] = round(time.perf_counter() - t0, 2)
            fh.write(json.dumps(rec, default=str) + "\n")
            fh.flush()
            ns = {k: (v or {}).get("N") for k, v in (rec.get("arms") or {}).items()}
            print(f"[{i}/{len(names)}] {name} best={rec.get('best_arm')} N={ns} {rec['driver_wall_s']}s", flush=True)


if __name__ == "__main__":
    main()
