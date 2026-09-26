"""Gap analysis for the Word Mover's Distance track on 20 Newsgroups (2026-09-15).

Stages, each writing one JSON line per record to --out:

  sizes      instance size distribution over a sample of pairs, the largest natural pairs, and the concatenated
             super-documents built to reach n = 1e5 .. 1e6
  verify     the reduction: our LP optimum (HiGHS, Clarabel) against POT's network simplex in its own venv, plus
             feasibility of the returned transport plan and of the construction's x_feas
  knn        the downstream number: k-NN classification error on a stratified subset of 20 Newsgroups with EXACT
             WMD distances, against the published WMD result and against an entropic (Sinkhorn) approximation
             implemented here
  structure  uni.plan_product (K, cover, which marginal family), uni.diagnose + uni.plan_block, and the
             rows-versus-columns question: both sides are absorbable, which one does the greedy pick and why
  solvers    HiGHS, PDLP, cuPDLP and Clarabel at 1e-3 and 1e-4 on the largest instances, judged by uni.accept
             against the network-simplex reference; failures are recorded, not dropped
  batch      the REALISTIC workload: thousands of small pairs end to end, per-LP cost for each solver
  gap        router.solve at the requested tol and budget: solved, choice, binding factor, spec, uncovered
  ladder     iterations to 1e-2 and 1e-3 for std_ruiz, rows-absorbed and columns-absorbed, the planner experiment

H100 WALL-CLOCK IS CONTAMINATED (a vLLM server holds 70 GB and the SMs). Iteration counts and structure are the
primary numbers; every second reported from that machine is indicative only.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(_HERE))))

import numpy as np
import scipy.sparse as sp

import wmd

POT_PY = os.environ.get("OPTEVOLVE_POT_PYTHON",
                        "/root/baseline_envs/venv_pot/bin/python" if os.path.isdir("/root/baseline_envs/venv_pot")
                        else "/home/miria/baseline_envs/venv_pot/bin/python")


def _jsonable(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist() if o.size <= 32 else {"shape": list(o.shape), "head": o.ravel()[:8].tolist()}
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)


def emit(out, rec):
    line = json.dumps(rec, default=_jsonable)
    if out:
        with open(out, "a") as fh:
            fh.write(line + "\n")
    print(line, flush=True)


# ----------------------------------------------------------------------------------------------------------------
# sizes
# ----------------------------------------------------------------------------------------------------------------
def stage_sizes(out, n_pairs, targets):
    c = wmd.load()
    s = wmd.doc_sizes()
    emit(out, {"stage": "sizes", "what": "corpus", **c["meta"],
               "docs_with_no_invocab_word": int((s == 0).sum()), "docs_at_cap": int((s == wmd.CAP_WORDS).sum()),
               "unique_words_mean": float(s.mean()), "unique_words_median": float(np.median(s)),
               "unique_words_p90": float(np.percentile(s, 90)), "unique_words_max": int(s.max())})
    pairs = wmd.sample_pairs(n_pairs, seed=0)
    emit(out, {"stage": "sizes", "what": "pair_distribution", **wmd.pair_table(pairs)})
    big = wmd.largest_pairs(8)
    emit(out, {"stage": "sizes", "what": "largest_natural_pairs",
               "pairs": [{"doc_a": a, "doc_b": b, "n_var": v} for a, b, v in big]})
    for tw in targets:
        g = wmd.concat_groups(tw, n_groups=2, seed=0)
        d = wmd.make_wmd_concat(g[0], g[1], permute=False, with_feas=True)
        m = d["meta"]
        emit(out, {"stage": "sizes", "what": "concat", "target_words": tw, "m": m["m"], "n": m["n"],
                   "n_var": m["n_var"], "rows": m["rows"], "nnz": m["nnz"],
                   "n_docs_a": m["n_docs_a"], "n_docs_b": m["n_docs_b"], "name": d["name"],
                   **wmd.feasibility(d)})


# ----------------------------------------------------------------------------------------------------------------
# verify
# ----------------------------------------------------------------------------------------------------------------
def pot_pair(i, j, p=1.0, timeout_s=3600):
    pr = subprocess.run([POT_PY, os.path.join(_HERE, "wmd_ref_pot.py"), "pair", str(i), str(j), str(p)],
                        capture_output=True, text=True, timeout=timeout_s,
                        env=dict(os.environ, OMP_NUM_THREADS="1"))
    if pr.returncode != 0:
        return {"error": (pr.stderr or pr.stdout).strip()[-400:]}
    return json.loads(pr.stdout.strip().splitlines()[-1])


def run_solver(d, solver, eps, timeout_s=1800):
    """One baseline in its isolated interpreter, returning objective, plan feasibility and timing."""
    from atlas.bench.baseline_proc import run_baseline_proc
    import lp_families
    t0 = time.perf_counter()
    rb = run_baseline_proc(wmd.solver_dict(d), solver, eps, timeout_s=timeout_s,
                           scratch=os.environ.get("OPTEVOLVE_SCRATCH"))
    wall = time.perf_counter() - t0
    if "error" in rb or rb.get("x") is None:
        return {"solver": solver, "eps": eps, "error": str(rb.get("error"))[:300], "wall_s": wall}
    x = np.asarray(rb["x"])
    return {"solver": solver, "eps": eps, "obj": float(np.asarray(d["q"]) @ x), "t_solver_s": rb.get("t_s"),
            "row_violation": lp_families.row_violation(d, x), "x_min": float(x.min()), "wall_s": wall}


def stage_verify(out, n_small, p, solvers):
    rng = np.random.default_rng(7)
    s = wmd.doc_sizes()
    ok = np.flatnonzero((s >= 5) & (s <= 60))                 # small enough that every solver finishes instantly
    for t in range(n_small):
        i, j = (int(x) for x in rng.choice(ok, size=2, replace=False))
        ref = pot_pair(i, j, p)
        if "error" in ref:
            emit(out, {"stage": "verify", "doc_a": i, "doc_b": j, "pot_error": ref["error"]})
            continue
        d = wmd.make_wmd((i, j), p=p, permute=True, seed=t)
        rec = {"stage": "verify", "doc_a": i, "doc_b": j, "m": ref["m"], "n": ref["n"], "n_var": ref["n_var"],
               "pot_obj": ref["obj"], "pot_plan_row_err": ref["plan_row_err"],
               "pot_plan_col_err": ref["plan_col_err"], "pot_plan_min": ref["plan_min"],
               "pot_t_solve_s": ref["t_solve_s"], **wmd.feasibility(d)}
        for solver in solvers:
            r = run_solver(d, solver, 1e-9)
            if "error" in r:
                rec[solver] = {"error": r["error"]}
            else:
                rec[solver] = {"obj": r["obj"], "row_violation": r["row_violation"], "x_min": r["x_min"],
                               "rel_err_vs_pot": abs(r["obj"] - ref["obj"]) / (1.0 + abs(ref["obj"]))}
        emit(out, rec)


# ----------------------------------------------------------------------------------------------------------------
# knn: the downstream number
# ----------------------------------------------------------------------------------------------------------------
def _stratified(split, per_class, seed, exclude=None):
    c = wmd.load()
    s = wmd.doc_sizes()
    rng = np.random.default_rng(seed)
    pick = []
    for lab in range(20):
        pool = np.flatnonzero((c["labels"] == lab) & (c["split"] == split) & (s > 0))
        if exclude is not None:
            pool = np.setdiff1d(pool, exclude)
        rng.shuffle(pool)
        pick.append(pool[:per_class])
    return np.sort(np.concatenate(pick))


def wmd_matrix(rows, cols, p, workers, tag, scratch):
    """Exact WMD distance matrix, one network-simplex LP per entry, fanned over `workers` POT processes."""
    os.makedirs(scratch, exist_ok=True)
    rpath = os.path.join(scratch, f"{tag}_cols.npy")
    np.save(rpath, cols)
    chunks = np.array_split(rows, workers)
    procs, outs = [], []
    t0 = time.perf_counter()
    for w, ch in enumerate(chunks):
        if ch.size == 0:
            continue
        cp = os.path.join(scratch, f"{tag}_rows{w}.npy")
        op = os.path.join(scratch, f"{tag}_out{w}.npy")
        np.save(cp, ch)
        procs.append(subprocess.Popen(
            [POT_PY, os.path.join(_HERE, "wmd_ref_pot.py"), "matrix", cp, rpath, op, str(p)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            env=dict(os.environ, OMP_NUM_THREADS="1")))
        outs.append((op, ch))
    D = np.zeros((rows.size, cols.size))
    off = 0
    for pr, (op, ch) in zip(procs, outs):
        so, se = pr.communicate()
        if pr.returncode != 0:
            raise RuntimeError(f"POT worker failed: {(se or so)[-500:]}")
        D[off:off + ch.size] = np.load(op)
        off += ch.size
    return D, time.perf_counter() - t0


def sinkhorn_matrix(rows, cols, p, eps_rel, n_iter, scratch=None):
    """Entropic approximation of the same distances; see wmd_sinkhorn for the method and why it is batched."""
    import wmd_sinkhorn
    D, dt, mass = wmd_sinkhorn.sinkhorn_matrix(rows, cols, p, eps_rel, n_iter)
    if mass < 0.99:
        print(json.dumps({"warning": "entropic plan lost mass", "min_plan_mass": mass}), flush=True)
    return D, dt


def _knn_error(D, y_rows, y_cols, k):
    """k-NN error of rows classified by their k nearest columns (ties broken by the nearer neighbour)."""
    idx = np.argsort(D, axis=1)[:, :k]
    lab = y_cols[idx]
    pred = np.empty(D.shape[0], dtype=np.int64)
    for r in range(D.shape[0]):
        vals, counts = np.unique(lab[r], return_counts=True)
        best = vals[counts == counts.max()]
        pred[r] = lab[r][np.isin(lab[r], best)][0] if best.size > 1 else best[0]
    return float((pred != y_rows).mean()), pred


def _select_k(Dtt, y_tr, ks):
    """Leave-one-out k selection on the training subsample, the role the paper's 80/20 validation split plays."""
    Dm = Dtt.copy()
    np.fill_diagonal(Dm, np.inf)
    errs = {}
    for k in ks:
        errs[k], _ = _knn_error(Dm, y_tr, y_tr, k)
    best = min(errs, key=lambda k: (errs[k], k))
    return best, errs


def stage_knn(out, per_class_train, per_class_test, p, workers, scratch, eps_rels, sink_iter, ks, full_train):
    """k is chosen by leave-one-out on a TRAIN SUBSAMPLE (the role the paper's 80/20 validation split plays), then
    the test documents are classified against the FULL training set, which is the published protocol."""
    c = wmd.load()
    s = wmd.doc_sizes()
    tr = _stratified(0, per_class_train, seed=11)                  # k-selection pool
    te = _stratified(1, per_class_test, seed=12)
    ref_tr = np.flatnonzero((c["split"] == 0) & (s > 0)) if full_train else tr
    y_tr, y_te, y_ref = c["labels"][tr], c["labels"][te], c["labels"][ref_tr]
    emit(out, {"stage": "knn", "what": "subsample", "n_ksel_train": int(tr.size), "n_test": int(te.size),
               "n_reference_train": int(ref_tr.size), "full_train": bool(full_train),
               "per_class_train": per_class_train, "per_class_test": per_class_test,
               "train_unique_words_mean": float(s[ref_tr].mean()), "test_unique_words_mean": float(s[te].mean()),
               "lps_k_selection": int(tr.size ** 2), "lps_test_train": int(te.size * ref_tr.size),
               "lps_total": int(tr.size ** 2 + te.size * ref_tr.size), "p": p,
               "note": "every entry of both matrices is one optimal-transport LP"})

    Dtt, t_tt = wmd_matrix(tr, tr, p, workers, "trtr", scratch)
    Dtt = 0.5 * (Dtt + Dtt.T)                                  # network simplex is symmetric up to float rounding
    k_best, k_errs = _select_k(Dtt, y_tr, ks)
    emit(out, {"stage": "knn", "what": "k_selection", "k_best": int(k_best),
               "loo_errors": {str(k): v for k, v in k_errs.items()}, "wall_s": t_tt,
               "lps": int(tr.size ** 2), "us_per_lp": 1e6 * t_tt / max(tr.size ** 2, 1), "workers": workers})

    Dte, t_te = wmd_matrix(te, ref_tr, p, workers, "tetr", scratch)
    err_exact, _ = _knn_error(Dte, y_te, y_ref, k_best)
    per_k = {str(k): _knn_error(Dte, y_te, y_ref, k)[0] for k in ks}
    emit(out, {"stage": "knn", "what": "exact_wmd", "k": int(k_best), "test_error": err_exact,
               "test_error_per_k": per_k, "best_k_error": float(min(per_k.values())),
               "published_kusner2015_20news": 0.27,
               "published_note": "Kusner et al. ICML 2015 Figure 3, full 11,293-document training set",
               "wall_s": t_te, "lps": int(te.size * ref_tr.size),
               "us_per_lp": 1e6 * t_te / max(te.size * ref_tr.size, 1), "workers": workers})
    np.save(os.path.join(scratch, "D_test_train.npy"), Dte)
    np.save(os.path.join(scratch, "knn_test_idx.npy"), te)
    np.save(os.path.join(scratch, "knn_train_idx.npy"), ref_tr)

    for er in eps_rels:
        Ste, t_s = sinkhorn_matrix(te, ref_tr, p, er, sink_iter)
        err_s, _ = _knn_error(Ste, y_te, y_ref, k_best)
        per_k_s = {str(k): _knn_error(Ste, y_te, y_ref, k)[0] for k in ks}
        rel = np.abs(Ste - Dte) / (1.0 + np.abs(Dte))
        emit(out, {"stage": "knn", "what": "sinkhorn", "eps_rel": er, "sink_iter": sink_iter, "k": int(k_best),
                   "test_error": err_s, "exact_test_error": err_exact,
                   "test_error_per_k": per_k_s, "best_k_error": float(min(per_k_s.values())),
                   "k_note": "k is the one chosen by exact WMD, so the two rows differ only in the distance",
                   "median_rel_dist_err_vs_exact": float(np.median(rel)),
                   "mean_rel_dist_err_vs_exact": float(rel.mean()),
                   "max_rel_dist_err_vs_exact": float(rel.max()),
                   "nearest_neighbour_agreement": float(_rank_agree(Dte, Ste)),
                   "wall_s": t_s, "lps": int(te.size * ref_tr.size),
                   "us_per_lp": 1e6 * t_s / max(te.size * ref_tr.size, 1)})


def stage_sinkhorn(out, scratch, p, eps_rels, sink_iter, ks, k_exact):
    """The entropic comparison, against the EXACT matrix a previous knn run saved.

    Split out from `knn` so the Sinkhorn sweep can be re-run (or a new eps added) without re-solving the 11 million
    network-simplex LPs the exact matrix cost.
    """
    c = wmd.load()
    Dte = np.load(os.path.join(scratch, "D_test_train.npy"))
    te = np.load(os.path.join(scratch, "knn_test_idx.npy"))
    ref_tr = np.load(os.path.join(scratch, "knn_train_idx.npy"))
    y_te, y_ref = c["labels"][te], c["labels"][ref_tr]
    err_exact, _ = _knn_error(Dte, y_te, y_ref, k_exact)
    for er in eps_rels:
        Ste, t_s = sinkhorn_matrix(te, ref_tr, p, er, sink_iter)
        err_s, _ = _knn_error(Ste, y_te, y_ref, k_exact)
        per_k_s = {str(k): _knn_error(Ste, y_te, y_ref, k)[0] for k in ks}
        rel = np.abs(Ste - Dte) / (1.0 + np.abs(Dte))
        np.save(os.path.join(scratch, f"S_test_train_{er}.npy"), Ste)
        emit(out, {"stage": "sinkhorn", "eps_rel": er, "sink_iter": sink_iter, "k": int(k_exact),
                   "test_error": err_s, "exact_test_error": err_exact,
                   "test_error_per_k": per_k_s, "best_k_error": float(min(per_k_s.values())),
                   "k_note": "k is the one chosen by exact WMD, so the two rows differ only in the distance",
                   "median_rel_dist_err_vs_exact": float(np.median(rel)),
                   "mean_rel_dist_err_vs_exact": float(rel.mean()),
                   "max_rel_dist_err_vs_exact": float(rel.max()),
                   "nearest_neighbour_agreement": float(_rank_agree(Dte, Ste)),
                   "wall_s": t_s, "lps": int(te.size * ref_tr.size),
                   "us_per_pair": 1e6 * t_s / max(te.size * ref_tr.size, 1)})


def _rank_agree(D, S):
    """Fraction of row-wise nearest-neighbour choices on which the approximation agrees with the exact distance."""
    return float((np.argmin(D, axis=1) == np.argmin(S, axis=1)).mean())


# ----------------------------------------------------------------------------------------------------------------
# structure
# ----------------------------------------------------------------------------------------------------------------
def stage_structure(out, instances):
    import uni
    for label, d in instances:
        A = sp.csr_matrix(d["A"])
        fam = np.asarray(d["meta"]["row_family"])
        nnz = np.diff(A.indptr)
        gen = np.flatnonzero(fam != 2)
        m, n = d["meta"]["m"], d["meta"]["n"]
        rec = {"stage": "structure", "label": label, "name": d["name"], "m": m, "n": n,
               "n_var": d["meta"]["n_var"], "rows": int(A.shape[0]), "nnz": int(A.nnz),
               "general_rows": int(gen.size), "general_row_nnz": sorted(set(nnz[gen].tolist()))[:6],
               "n_distinct_general_nnz": int(len(set(nnz[gen].tolist()))),
               "source_row_nnz": int(n), "sink_row_nnz": int(m)}
        t0 = time.perf_counter()
        pplan, pmsg = uni.plan_product(d)
        rec["plan_product_s"] = time.perf_counter() - t0
        if pplan is None:
            rec["plan_product"] = {"status": pmsg}
        else:
            rec["plan_product"] = {"status": pmsg, "kind": pplan["kind"], "K": int(pplan["K"]),
                                   "cover": float(pplan["cover"]), "n_x": int(pplan["n_x"]),
                                   "rows_kept": int(pplan["keep"].size),
                                   "block_sizes": sorted(set(np.bincount(pplan["bid"]).tolist()))[:6],
                                   "fully_absorbed": bool(pplan["fully_absorbed"]),
                                   "side": wmd.plan_side_of(d, pplan)}
            for side in ("source", "sink"):
                t0 = time.perf_counter()
                ps = wmd.plan_side(d, side)
                rec[f"plan_{side}"] = {"K": int(ps["K"]), "cover": float(ps["cover"]), "n_x": int(ps["n_x"]),
                                       "rows_kept": int(ps["keep"].size), "build_s": time.perf_counter() - t0,
                                       "block_size": int(np.bincount(ps["bid"]).max()),
                                       "agrees_with_greedy": wmd.plans_agree(ps, pplan),
                                       "fully_absorbed": bool(ps["fully_absorbed"])}
            rec["both_sides_absorbable"] = bool(rec["plan_source"]["cover"] == 1.0
                                                and rec["plan_sink"]["cover"] == 1.0)
            rec["greedy_picks_shorter_document_side"] = bool(
                (m < n and rec["plan_product"]["side"] == "source") or
                (n < m and rec["plan_product"]["side"] == "sink") or m == n)
        t0 = time.perf_counter()
        diag = uni.diagnose(d)
        rec["diagnose"] = dict(diag, wall_s=time.perf_counter() - t0)
        bplan, bmsg = uni.plan_block(d, diag)
        rec["plan_block"] = {"status": bmsg} if bplan is None else {
            "status": bmsg, "kind": bplan["kind"], "n_x": int(bplan["n_x"]),
            "rows_kept": int(bplan["keep"].size), "dense_row": int(bplan["dense_row"]),
            "family_of_absorbed_row": ["source", "sink", "bound"][int(fam[bplan["dense_row"]])]}
        emit(out, rec)


# ----------------------------------------------------------------------------------------------------------------
# solvers
# ----------------------------------------------------------------------------------------------------------------
def stage_solvers(out, instances, solvers, tols, obj_refs, timeout_s):
    import uni
    for label, d in instances:
        ref = obj_refs.get(label)
        for solver in solvers:
            for tol in tols:
                r = run_solver(d, solver, tol, timeout_s=timeout_s)
                rec = {"stage": "solvers", "label": label, "name": d["name"], "n_var": d["meta"]["n_var"],
                       "m": d["meta"]["m"], "n": d["meta"]["n"], "solver": solver, "tol": tol,
                       "wall_s": r["wall_s"]}
                if "error" in r:
                    rec["error"] = r["error"]
                    rec["accepted"] = False
                else:
                    rec.update({"obj": r["obj"], "t_solver_s": r["t_solver_s"],
                                "row_violation": r["row_violation"], "x_min": r["x_min"]})
                    if ref is not None:
                        rec["obj_gap"] = abs(r["obj"] - ref) / (1.0 + abs(ref))
                        rec["accepted"] = bool(rec["row_violation"] <= tol and rec["obj_gap"] <= tol)
                        rec["obj_ref"] = ref
                emit(out, rec)


def stage_batch(out, n_pairs, solvers, tol, p, seed):
    """The realistic workload: many small pairs, one LP each, end to end per solver."""
    pairs = wmd.sample_pairs(n_pairs, seed=seed)
    s = wmd.doc_sizes()
    keep = pairs[(s[pairs[:, 0]] > 0) & (s[pairs[:, 1]] > 0)][:n_pairs]
    build_t = 0.0
    ds = []
    for (i, j) in keep:
        t0 = time.perf_counter()
        ds.append(wmd.make_wmd((i, j), p=p, permute=False, with_feas=False))
        build_t += time.perf_counter() - t0
    nv = np.array([d["meta"]["n_var"] for d in ds])
    emit(out, {"stage": "batch", "what": "build", "n_pairs": len(ds), "build_s": build_t,
               "ms_per_build": 1e3 * build_t / max(len(ds), 1), "n_var_mean": float(nv.mean()),
               "n_var_median": float(np.median(nv))})
    # POT, the specialist, on exactly these pairs
    scratch = os.environ.get("OPTEVOLVE_SCRATCH", "/tmp")
    os.makedirs(scratch, exist_ok=True)
    pp = os.path.join(scratch, "batch_pairs.npy")
    np.save(pp, np.asarray(keep))
    t0 = time.perf_counter()
    pr = subprocess.run([POT_PY, os.path.join(_HERE, "wmd_ref_pot.py"), "pairs", pp, str(p)],
                        capture_output=True, text=True, env=dict(os.environ, OMP_NUM_THREADS="1"))
    rec = json.loads(pr.stdout.strip().splitlines()[-1]) if pr.returncode == 0 else {"error": pr.stderr[-300:]}
    emit(out, {"stage": "batch", "what": "pot_network_simplex", "n_pairs": len(keep),
               "wall_s": time.perf_counter() - t0, **rec})
    for solver in solvers:
        t0, fails, objs = time.perf_counter(), 0, []
        for d in ds:
            r = run_solver(d, solver, tol, timeout_s=300)
            if "error" in r:
                fails += 1
            else:
                objs.append(r["obj"])
        w = time.perf_counter() - t0
        emit(out, {"stage": "batch", "what": "solver", "solver": solver, "tol": tol, "n_pairs": len(ds),
                   "wall_s": w, "ms_per_lp": 1e3 * w / max(len(ds), 1), "failures": fails,
                   "note": "includes the isolated-interpreter launch, which is the honest cost of calling a "
                           "general LP solver once per document pair"})


# ----------------------------------------------------------------------------------------------------------------
# gap and ladder
# ----------------------------------------------------------------------------------------------------------------
def stage_gap(out, instances, obj_refs, vault_path, device, tol, budget_s):
    import router
    from vault import Vault
    v = Vault(vault_path)
    for label, d in instances:
        case = f"WMD:{label}"
        t0 = time.perf_counter()
        try:
            r = router.solve(case, v, device, tol=tol, budget_s=budget_s, d=wmd.solver_dict(d),
                             obj_ref=obj_refs.get(label))
            r["wall_total_measured_s"] = time.perf_counter() - t0
        except Exception as e:
            r = {"error": f"{type(e).__name__}: {e}"[:500], "wall_total_measured_s": time.perf_counter() - t0}
        emit(out, {"stage": "gap", "label": label, "case": case, "n_var": d["meta"]["n_var"],
                   "m": d["meta"]["m"], "n": d["meta"]["n"], "tol": tol, "budget_s": budget_s, **r})


def stage_ladder(out, instances, obj_refs, cap, deadline_s, tols):
    import uni
    for label, d in instances:
        ref = obj_refs.get(label)
        if ref is None:
            emit(out, {"stage": "ladder", "label": label, "error": "no reference objective"})
            continue
        cands = [("std_ruiz", "std_ruiz", None),
                 ("rows_absorbed", "reloc_rows", wmd.plan_side(d, "source")),
                 ("cols_absorbed", "reloc_rows", wmd.plan_side(d, "sink"))]
        for name, cand, plan in cands:
            t0 = time.perf_counter()
            rec = {"stage": "ladder", "label": label, "cand": name, "n_var": d["meta"]["n_var"],
                   "m": d["meta"]["m"], "n": d["meta"]["n"]}
            try:
                form = uni.build_formulation(wmd.solver_dict(d), cand, plan=plan, impl="bisect")
                m = uni.measure_ladder(form, wmd.solver_dict(d), ref, tols=tuple(tols), cap=cap,
                                       deadline=None if deadline_s is None else time.perf_counter() + deadline_s)
                L = sp.csc_matrix(form.L_csc)
                rec.update({"N_at": m["N_at"], "iters_run": m["iters_run"], "normL": m["normL"], "Lf": m["Lf"],
                            "setup_s": m["setup_s"], "t_iter_s": m["t_iter"],
                            "row_violation": m["row_violation"], "obj_gap": m["obj_gap"],
                            "absorbed": int(form.absorbed), "form": form.name,
                            "K": None if plan is None else int(plan["K"]),
                            "L_shape": list(L.shape), "L_nnz": int(L.nnz)})
            except Exception as e:
                rec["error"] = f"{type(e).__name__}: {e}"[:400]
            rec["wall_s"] = time.perf_counter() - t0
            emit(out, rec)


# ----------------------------------------------------------------------------------------------------------------
def build_instances(spec, p, permute, seed):
    """`spec` entries are 'pair:I:J' or 'concat:TARGET_WORDS'."""
    out = []
    for s in spec:
        parts = s.split(":")
        if parts[0] == "pair":
            i, j = int(parts[1]), int(parts[2])
            out.append((s, wmd.make_wmd((i, j), p=p, permute=permute, seed=seed, with_feas=False)))
        elif parts[0] == "concat":
            tw = int(parts[1])
            g = wmd.concat_groups(tw, n_groups=2, seed=0)
            out.append((s, wmd.make_wmd_concat(g[0], g[1], p=p, permute=permute, seed=seed, with_feas=False)))
        else:
            raise ValueError(f"bad instance spec {s!r}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["sizes", "verify", "knn", "structure", "solvers", "batch", "gap", "ladder", "sinkhorn"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--p", type=float, default=1.0)
    ap.add_argument("--instances", nargs="*", default=["concat:300"])
    ap.add_argument("--no-permute", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-pairs", type=int, default=5000)
    ap.add_argument("--targets", type=int, nargs="*", default=[300, 700, 1100])
    ap.add_argument("--solvers", nargs="*", default=["highs", "clarabel", "pdlp"])
    ap.add_argument("--tols", type=float, nargs="*", default=[1e-3, 1e-4])
    ap.add_argument("--tol", type=float, default=1e-3)
    ap.add_argument("--timeout", type=float, default=1800.0)
    ap.add_argument("--ref-json", default=None, help="jsonl carrying {'label':..., 'obj_ref':...} records")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--scratch", default=os.environ.get("OPTEVOLVE_SCRATCH", "/tmp/wmd"))
    ap.add_argument("--per-class-train", type=int, default=50)
    ap.add_argument("--per-class-test", type=int, default=25)
    ap.add_argument("--eps-rels", type=float, nargs="*", default=[0.5, 0.1, 0.05, 0.01])
    ap.add_argument("--sink-iter", type=int, default=200)
    ap.add_argument("--sub-train", action="store_true",
                    help="classify against the train SUBSAMPLE instead of the full training set")
    ap.add_argument("--ks", type=int, nargs="*", default=list(range(1, 20)))
    ap.add_argument("--vault", default="/workspace/optevolve/vault/vault.jsonl")
    ap.add_argument("--device", default="h100")
    ap.add_argument("--budget", type=float, default=120.0)
    ap.add_argument("--cap", type=int, default=200000)
    ap.add_argument("--ladder-deadline", type=float, default=900.0)
    ap.add_argument("--n-small", type=int, default=12)
    ap.add_argument("--k-exact", type=int, default=14, help="sinkhorn stage: the k the exact run chose")
    args = ap.parse_args()

    os.makedirs(args.scratch, exist_ok=True)
    refs = {}
    if args.ref_json and os.path.exists(args.ref_json):
        with open(args.ref_json) as fh:
            for line in fh:
                r = json.loads(line)
                if r.get("obj_ref") is not None:
                    refs[r["label"]] = float(r["obj_ref"])

    if args.stage == "sizes":
        stage_sizes(args.out, args.n_pairs, args.targets)
        return
    if args.stage == "verify":
        stage_verify(args.out, args.n_small, args.p, args.solvers)
        return
    if args.stage == "sinkhorn":
        stage_sinkhorn(args.out, args.scratch, args.p, args.eps_rels, args.sink_iter, args.ks, args.k_exact)
        return
    if args.stage == "knn":
        stage_knn(args.out, args.per_class_train, args.per_class_test, args.p, args.workers, args.scratch,
                  args.eps_rels, args.sink_iter, args.ks, not args.sub_train)
        return

    inst = build_instances(args.instances, args.p, not args.no_permute, args.seed)
    if args.stage == "structure":
        stage_structure(args.out, inst)
    elif args.stage == "solvers":
        stage_solvers(args.out, inst, args.solvers, args.tols, refs, args.timeout)
    elif args.stage == "batch":
        stage_batch(args.out, args.n_pairs, args.solvers, args.tol, args.p, args.seed)
    elif args.stage == "gap":
        stage_gap(args.out, inst, refs, args.vault, args.device, args.tol, args.budget)
    elif args.stage == "ladder":
        stage_ladder(args.out, inst, refs, args.cap, args.ladder_deadline, args.tols)


if __name__ == "__main__":
    main()
