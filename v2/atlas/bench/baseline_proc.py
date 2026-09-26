"""Run a baseline solver in its own interpreter, so it never shares an environment with ours.

The sweep used to `import osqp` inside the process that also holds our JAX solver. That breaks the
standing rule that baselines live in isolated environments: on a machine where the solver env
deliberately has no baselines, every baseline arm reported
`ModuleNotFoundError: No module named 'osqp'` and the sweep printed our time against three FAILs,
which reads like a win and is not one (2026-09-13, portfolio at n_var 2044).

Running out of process also gives each baseline its own CPU-pinned interpreter, which is how OSQP is
kept on CPU while ours runs on the GPU, and it means a baseline segfault cannot take our run with it.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np
import scipy.sparse as sp

SOLVERS = ("osqp", "scs", "clarabel")
# The virtual-best field (2026-09-15). Kept out of SOLVERS so every sweep that iterates SOLVERS still runs exactly the
# original three. Problem classes: highs, piqp, proxqp take any convex QP; ecos takes LPs directly and QPs through an
# SOC epigraph (needs a factor of P); pdlp takes LPs and diagonal-P QPs; cupdlpx (GPU) takes LPs only.
EXTRA_SOLVERS = ("highs", "piqp", "proxqp", "ecos", "pdlp", "cupdlpx")
ALL_SOLVERS = SOLVERS + EXTRA_SOLVERS
# Import name per solver, so the pre-clock import loads the real module (the solver name is not always the module).
_MODULES = {"highs": "highspy", "proxqp": "proxsuite", "pdlp": "ortools.pdlp.python.pdlp"}
# ECOS needs F with F'F = P. A non-diagonal P is factored by a dense eigendecomposition (what CVXPY's quad_form does),
# which is O(n^3) and fills the cone block, so larger non-diagonal instances are refused rather than timed out.
ECOS_DENSE_MAX_N = 3000
UNSUPPORTED_EXIT = 3


class Unsupported(ValueError):
    """The instance is outside the solver's problem class. Reported as an error record, never as a loss."""


def _env_python(solver: str) -> str:
    """Interpreter for `solver`: explicit override, then a venv_<solver> beside the repo, then ours."""
    explicit = os.environ.get(f"OPTEVOLVE_BASELINE_ENV_{solver.upper()}")
    if explicit:
        return explicit
    root = os.environ.get("OPTEVOLVE_BASELINE_ENV_ROOT", "")
    # The root may list several directories (os.pathsep-separated): the H100 keeps the original three venvs on
    # /workspace and the newer ones under /root/baseline_envs because /workspace has no room. One root behaves as before.
    for r in (p for p in root.split(os.pathsep) if p):
        cand = os.path.join(r, f"venv_{solver}", "bin", "python")
        if os.path.exists(cand):
            return cand
    return sys.executable


def _threads():
    """BASELINE_THREADS as an int, or None when unset, in which case every solver keeps its own default."""
    v = os.environ.get("BASELINE_THREADS", "").strip()
    return int(v) if v else None


def dump_instance(d: dict, path: str) -> None:
    P, A = sp.csc_matrix(d["P"]), sp.csc_matrix(d["A"])
    np.savez(path, Pd=P.data, Pi=P.indices, Pp=P.indptr, Ad=A.data, Ai=A.indices, Ap=A.indptr,
             q=np.asarray(d["q"]), l=np.asarray(d["l"]), u=np.asarray(d["u"]),
             shape=np.array([P.shape[0], A.shape[0]]))


def load_instance(path: str) -> dict:
    z = np.load(path)
    nvar, ncon = int(z["shape"][0]), int(z["shape"][1])
    return {"P": sp.csc_matrix((z["Pd"], z["Pi"], z["Pp"]), shape=(nvar, nvar)),
            "A": sp.csc_matrix((z["Ad"], z["Ai"], z["Ap"]), shape=(ncon, nvar)),
            "q": z["q"], "l": z["l"], "u": z["u"], "r": 0.0}


def solve(d: dict, solver: str, eps: float, timeout_s: float):
    """Solve in THIS process. The clock starts before setup, matching the timing contract."""
    P, A, q, l, u = d["P"], d["A"], d["q"], d["l"], d["u"]
    # IMPORT BEFORE THE CLOCK. The import used to sit inside the timed region, charging every baseline its module load
    # (measured on the H100: osqp 0.19-0.26 s, scs 0.02-0.04 s, clarabel 0.01-0.02 s) while our side never pays for
    # importing jax. On small instances that was most of OSQP's reported time (2026-09-15, Maros-Meszaros DUAL*).
    import importlib
    importlib.import_module(_MODULES.get(solver, solver))
    if solver in EXTRA_SOLVERS:
        return _solve_extra(d, solver, eps, timeout_s)
    t0 = time.perf_counter()
    if solver == "osqp":
        import osqp
        m = osqp.OSQP()
        m.setup(P=sp.triu(P, format="csc"), q=q, A=A, l=l, u=u, verbose=False,
                eps_abs=eps, eps_rel=eps, max_iter=200000, time_limit=timeout_s)
        r = m.solve()
        x, y, st, it = r.x, r.y, str(r.info.status), int(r.info.iter)
    else:
        eqm = np.isfinite(l) & np.isfinite(u) & (l == u)
        up, lo = np.isfinite(u) & ~eqm, np.isfinite(l) & ~eqm
        Ac = sp.vstack([A[eqm], A[up], -A[lo]]).tocsc()
        bc = np.concatenate([u[eqm], u[up], -l[lo]])
        nz, nl = int(eqm.sum()), int(up.sum() + lo.sum())
        if solver == "scs":
            import scs
            sol = scs.SCS({"P": sp.triu(P, format="csc"), "A": Ac, "b": bc, "c": q},
                          {"z": nz, "l": nl}, eps_abs=eps, eps_rel=eps,
                          time_limit_secs=timeout_s, verbose=False).solve()
            x, y, st = np.asarray(sol["x"]), np.asarray(sol["y"]), str(sol["info"]["status"])
            it = int(sol["info"]["iter"])
        elif solver == "clarabel":
            import clarabel
            cones = [clarabel.ZeroConeT(nz), clarabel.NonnegativeConeT(nl)]
            cfg = clarabel.DefaultSettings()
            cfg.verbose = False
            cfg.tol_gap_abs = cfg.tol_gap_rel = cfg.tol_feas = eps
            cfg.time_limit = timeout_s
            if _threads() is not None:      # only under BASELINE_THREADS, so unset runs keep the old settings exactly
                cfg.max_threads = _threads()
            sol = clarabel.DefaultSolver(sp.triu(P, format="csc"), q, Ac, bc, cones, cfg).solve()
            x, y, st, it = np.array(sol.x), np.array(sol.z), str(sol.status), int(sol.iterations)
        else:
            raise ValueError(f"unknown baseline solver {solver!r}")
        y = _dual_from_conic(y, eqm, up, lo, A.shape[0])
    t = time.perf_counter() - t0
    out = {"t_s": t, "x": x, "y": y, "status": st, "iters": it}
    if _threads() is not None:
        out["threads"] = _threads()
    return out


def _is_diagonal(P) -> bool:
    P = sp.csc_matrix(P)
    return (P - sp.diags(P.diagonal())).count_nonzero() == 0


def _check_class(d: dict, solver: str) -> None:
    """Refuse instances outside the solver's class BEFORE the clock, so a refusal is never timed as a failure."""
    P = sp.csc_matrix(d["P"])
    n = P.shape[0]
    if solver == "cupdlpx" and P.count_nonzero():
        raise Unsupported("cupdlpx solves LPs only (P is nonzero)")
    if solver == "pdlp" and not _is_diagonal(P):
        raise Unsupported("OR-Tools PDLP accepts only a diagonal objective matrix")
    if solver == "ecos" and P.count_nonzero() and not _is_diagonal(P) and n > ECOS_DENSE_MAX_N:
        raise Unsupported(f"ecos needs a dense factor of a non-diagonal P; n={n} > {ECOS_DENSE_MAX_N}")


def _psd_factor(P):
    """Sparse F (k x n) with F'F = P, for the SOC epigraph of 1/2 x'Px.

    Diagonal P: F = sqrt(diag). Positive definite P: the Cholesky factor, which is triangular and so about half as dense
    as an eigenvector factor (DUAL1: 3,653 vs 7,225 nonzeros). Singular P: eigendecomposition with eigenvalues below
    1e-12 * max dropped. Entries below 1e-14 of the largest are dropped in both dense cases (CVXQP1_S: 9,472 to 6,476).
    """
    P = sp.csc_matrix(P)
    n = P.shape[0]
    if _is_diagonal(P):
        dg = P.diagonal()
        nz = np.flatnonzero(dg > 0)
        return sp.csc_matrix((np.sqrt(dg[nz]), (np.arange(nz.size), nz)), shape=(nz.size, n))
    Pd = P.toarray()
    try:
        F = np.linalg.cholesky(Pd).T
    except np.linalg.LinAlgError:
        w, V = np.linalg.eigh(Pd)
        keep = w > 1e-12 * max(float(w.max()), 1e-300)
        F = np.sqrt(w[keep])[:, None] * V[:, keep].T
    F[np.abs(F) < 1e-14 * np.abs(F).max()] = 0.0
    return sp.csc_matrix(F)


def _solve_extra(d: dict, solver: str, eps: float, timeout_s: float):
    """The solvers added for the virtual best. Same contract as `solve`: imports and any one-off device warm-up happen
    before the clock, the clock starts before setup (including every format conversion), x and y are returned on the
    original rows with the OSQP sign convention (P x + q + A'y = 0, y > 0 on an active upper bound).

    Tolerance mapping (the accuracy target eps is the rung under test; OUR checker is the judge, never the status):
      highs    primal_feasibility_tolerance = dual_feasibility_tolerance = max(eps, 1e-10) (HiGHS's lower limit),
               ipm_optimality_tolerance = max(eps, 1e-12); solver "choose" (simplex for LPs, active set for QPs)
      piqp     eps_abs = eps_rel = eps, eps_duality_gap_abs = eps_duality_gap_rel = eps (gap check is on by default)
      proxqp   eps_abs = eps_rel = eps, check_duality_gap on with eps_duality_gap_abs = eps_duality_gap_rel = eps;
               off by default, turned on because our judge requires the objective, not only the residuals;
               eps_primal_inf = eps_dual_inf = 1e-14 (see below: the default flags feasible LPs infeasible)
      ecos     abstol = reltol = eps; feastol stays at its default 1e-8 because it also gates ECOS's infeasibility
               certificate: at feastol = eps >= 1e-4 ECOS declared the feasible CVXQP1_S, CVXQP2_S and QSHARE2B
               "Primal infeasible" (2026-09-15), and at 1e-8 it solves all three in 20 to 26 iterations
      pdlp     simple_optimality_criteria eps_optimal_absolute = eps_optimal_relative = eps (default l2 norm)
      cupdlpx  OptimalityTol = FeasibilityTol = eps (relative, default l2 norm)
    The osqp / scs / clarabel mapping in `solve` is eps_abs = eps_rel = eps (tol_gap_abs = tol_gap_rel = tol_feas).
    Threads (BASELINE_THREADS): highs `threads`, pdlp `num_threads`, clarabel `max_threads`; piqp, proxqp (sparse),
    ecos, osqp and scs are single-threaded builds; cupdlpx runs on the GPU. The parent also caps the BLAS / OpenMP /
    rayon pools through the environment so no solver oversubscribes behind its own setting.
    """
    _check_class(d, solver)
    P, A = sp.csc_matrix(d["P"]), sp.csc_matrix(d["A"])
    q, l, u = np.asarray(d["q"], float), np.asarray(d["l"], float), np.asarray(d["u"], float)
    n, m = P.shape[0], A.shape[0]
    k = _threads()
    if solver == "pdlp":
        from ortools.pdlp import solve_log_pb2, solvers_pb2
        from ortools.pdlp.python import pdlp
    if solver == "cupdlpx":
        import cupdlpx
        # A throwaway solve before the clock creates the CUDA context and loads cuBLAS / cuSPARSE, the device analogue
        # of the pre-clock import; our GPU side is likewise timed on a warm device, never on context creation.
        w = cupdlpx.Model(objective_vector=np.ones(1), constraint_matrix=sp.csr_matrix(np.ones((1, 1))),
                          constraint_lower_bound=np.ones(1), constraint_upper_bound=np.ones(1))
        w.setParams(OutputFlag=False)
        w.optimize()
    t0 = time.perf_counter()
    if solver == "highs":
        import highspy
        h = highspy.Highs()
        h.setOptionValue("output_flag", False)
        h.setOptionValue("primal_feasibility_tolerance", max(eps, 1e-10))
        h.setOptionValue("dual_feasibility_tolerance", max(eps, 1e-10))
        h.setOptionValue("ipm_optimality_tolerance", max(eps, 1e-12))
        h.setOptionValue("time_limit", float(timeout_s))
        if k is not None:
            h.setOptionValue("threads", k)
        lp = highspy.HighsLp()
        lp.num_col_, lp.num_row_ = n, m
        lp.col_cost_ = q
        lp.col_lower_, lp.col_upper_ = np.full(n, -highspy.kHighsInf), np.full(n, highspy.kHighsInf)
        lp.row_lower_, lp.row_upper_ = l, u
        lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
        lp.a_matrix_.num_col_, lp.a_matrix_.num_row_ = n, m
        lp.a_matrix_.start_, lp.a_matrix_.index_, lp.a_matrix_.value_ = A.indptr, A.indices, A.data
        model = highspy.HighsModel()
        model.lp_ = lp
        Pl = sp.tril(P, format="csc")          # HiGHS reads the LOWER triangle, column-wise
        if Pl.count_nonzero():
            hess = highspy.HighsHessian()
            hess.dim_, hess.format_ = n, highspy.HessianFormat.kTriangular
            hess.start_, hess.index_, hess.value_ = Pl.indptr, Pl.indices, Pl.data
            model.hessian_ = hess
        h.passModel(model)
        h.run()
        sol, info = h.getSolution(), h.getInfo()
        x, y = np.asarray(sol.col_value, float), -np.asarray(sol.row_dual, float)   # HiGHS: c - A'y_h = reduced cost
        st = h.modelStatusToString(h.getModelStatus())
        detail = {"simplex": int(info.simplex_iteration_count), "ipm": int(info.ipm_iteration_count),
                  "crossover": int(info.crossover_iteration_count), "qp": int(info.qp_iteration_count),
                  "pdlp": int(info.pdlp_iteration_count)}
        it = sum(max(v, 0) for v in detail.values())
    elif solver in ("piqp", "proxqp"):
        eq = np.isfinite(l) & np.isfinite(u) & (l == u)
        Ae, Ai = A[eq].tocsc(), A[~eq].tocsc()
        ne, ni = int(eq.sum()), int((~eq).sum())
        Pu = sp.triu(P, format="csc")          # both read the UPPER triangle (checked: full and triu solve identically)
        if solver == "piqp":
            import piqp
            s = piqp.SparseSolver()
            cfg = s.settings
            cfg.verbose = False
            cfg.eps_abs = cfg.eps_rel = eps
            cfg.eps_duality_gap_abs = cfg.eps_duality_gap_rel = eps
            s.setup(Pu, q, Ae if ne else None, u[eq] if ne else None, Ai if ni else None,
                    l[~eq] if ni else None, u[~eq] if ni else None)
            status = s.solve()
            r = s.result
            x = np.asarray(r.x, float)
            y = np.zeros(m)
            if ne:
                y[eq] = np.asarray(r.y)
            if ni:
                y[~eq] = np.asarray(r.z_u) - np.asarray(r.z_l)      # PIQP: P x + q + A_e'y + G'(z_u - z_l) = 0
            st, it = str(status), int(r.info.iter)
        else:
            from proxsuite import proxqp
            qp = proxqp.sparse.QP(n, ne, ni)
            cfg = qp.settings
            cfg.verbose = False
            cfg.eps_abs = cfg.eps_rel = eps
            cfg.check_duality_gap = True
            cfg.eps_duality_gap_abs = cfg.eps_duality_gap_rel = eps
            # With the default 1e-4 infeasibility thresholds ProxQP (sparse AND dense) declared a feasible boxed LP
            # PRIMAL_INFEASIBLE after 22 iterations; at 1e-14 it solves it (2026-09-15). Every benchmark cell has a
            # Clarabel reference, so the instance is feasible and this can only make the baseline stronger.
            cfg.eps_primal_inf = cfg.eps_dual_inf = 1e-14
            # ProxQP documents +-1e20 as its infinity; a literal inf can reach its equilibration arithmetic
            qp.init(Pu, q, Ae if ne else None, u[eq] if ne else None, Ai if ni else None,
                    np.maximum(l[~eq], -1e20) if ni else None, np.minimum(u[~eq], 1e20) if ni else None)
            qp.solve()
            r = qp.results
            x = np.asarray(r.x, float)
            y = np.zeros(m)
            if ne:
                y[eq] = np.asarray(r.y)
            if ni:
                y[~eq] = np.asarray(r.z)                            # ProxQP: P x + q + A'y + C'z = 0
            st, it = str(r.info.status), int(r.info.iter)
    elif solver == "ecos":
        import ecos
        eqm = np.isfinite(l) & np.isfinite(u) & (l == u)
        up, lo = np.isfinite(u) & ~eqm, np.isfinite(l) & ~eqm
        Gl = sp.vstack([A[up], -A[lo]]).tocsc()
        hl = np.concatenate([u[up], -l[lo]])
        nl = Gl.shape[0]
        if P.count_nonzero():
            # min q'x + t  s.t.  x'Px <= 2t, the rotated cone ||F x||^2 <= 2 t * 1 written as the standard cone
            # ||((t - 1)/sqrt2, F x)|| <= (t + 1)/sqrt2 over z = (x, t)
            F = _psd_factor(P)
            kf, r2 = F.shape[0], 1.0 / np.sqrt(2.0)
            zc = sp.csc_matrix((nl, 1))
            tcol = sp.csc_matrix(np.array([[-r2], [-r2]]))
            G = sp.vstack([sp.hstack([Gl, zc]), sp.hstack([sp.csc_matrix((2, n)), tcol]),
                           sp.hstack([-F, sp.csc_matrix((kf, 1))])]).tocsc()
            h = np.concatenate([hl, [r2, -r2], np.zeros(kf)])
            c = np.concatenate([q, [1.0]])
            dims = {"l": nl, "q": [kf + 2]}
            Ae = sp.hstack([A[eqm], sp.csc_matrix((int(eqm.sum()), 1))]).tocsc()
        else:
            G, h, c, dims, Ae = Gl, hl, q, {"l": nl, "q": []}, A[eqm].tocsc()
        ne = int(eqm.sum())
        sol = ecos.solve(c, G if G.shape[0] else None, h if G.shape[0] else None, dims,
                         Ae if ne else None, u[eqm] if ne else None,
                         abstol=eps, reltol=eps, verbose=False)
        x = np.asarray(sol["x"], float)[:n]
        yc = np.concatenate([np.asarray(sol["y"], float).ravel()[:ne], np.asarray(sol["z"], float)[:nl]])
        y = _dual_from_conic(yc, eqm, up, lo, m)
        st, it = f'{sol["info"]["exitFlag"]}:{sol["info"]["infostring"]}', int(sol["info"]["iter"])
    elif solver == "pdlp":
        qp = pdlp.QuadraticProgram()
        qp.resize_and_initialize(n, m)
        qp.objective_vector = q
        qp.constraint_matrix = A
        qp.constraint_lower_bounds, qp.constraint_upper_bounds = l, u
        qp.variable_lower_bounds, qp.variable_upper_bounds = np.full(n, -np.inf), np.full(n, np.inf)
        if P.count_nonzero():
            qp.set_objective_matrix_diagonal(P.diagonal())
        prm = solvers_pb2.PrimalDualHybridGradientParams()
        prm.termination_criteria.simple_optimality_criteria.eps_optimal_absolute = eps
        prm.termination_criteria.simple_optimality_criteria.eps_optimal_relative = eps
        prm.termination_criteria.time_sec_limit = float(timeout_s)
        if k is not None:
            prm.num_threads = k
        res = pdlp.primal_dual_hybrid_gradient(qp, prm)
        x, y = np.asarray(res.primal_solution, float), -np.asarray(res.dual_solution, float)
        st = solve_log_pb2.TerminationReason.Name(res.solve_log.termination_reason)
        it = int(res.solve_log.iteration_count)
    elif solver == "cupdlpx":
        mdl = cupdlpx.Model(objective_vector=q, constraint_matrix=A.tocsr(), constraint_lower_bound=l,
                            constraint_upper_bound=u)
        mdl.setParams(OutputFlag=False, TimeLimit=float(timeout_s), OptimalityTol=eps, FeasibilityTol=eps)
        mdl.optimize()
        x = np.full(n, np.nan) if mdl.X is None else np.asarray(mdl.X, float)
        y = np.full(m, np.nan) if mdl.Pi is None else -np.asarray(mdl.Pi, float)     # cuPDLPx: c - A'pi = rc
        st, it = str(mdl.StatusName), int(mdl.IterCount)
    else:
        raise ValueError(f"unknown baseline solver {solver!r}")
    t = time.perf_counter() - t0
    # a solver that stopped without a point returns an empty vector; NaN keeps the record judgeable (and rejected)
    x = x if x.shape == (n,) else np.full(n, np.nan)
    y = y if y.shape == (m,) else np.full(m, np.nan)
    out = {"t_s": t, "x": x, "y": y, "status": st, "iters": it}
    if solver == "highs":
        out["iters_detail"] = detail
    if k is not None:
        out["threads"] = k
    return out


def _dual_from_conic(yc, eqm, up, lo, m):
    """Scatter the conic multiplier back onto the original l <= A x <= u rows."""
    # A row with BOTH bounds finite and l != u appears in `up` AND `lo`, so its multiplier is the SUM
    # of the two contributions. Plain assignment let the lower branch overwrite the upper one, which
    # corrupted every two-sided row. Only `control` has any (540 of 980 rows; lasso, huber, svm and
    # portfolio have zero), and it made Clarabel and SCS look like they FAILED our certificate
    # (cert 2.2e-01 and 7.0e-01) on points that were feasible to 1e-15. atlas/bench/qp_cell.py
    # y_from_conic already does this correctly; this function did not.
    y = np.zeros(m)
    n_eq, n_up = int(eqm.sum()), int(up.sum())
    y[eqm] = yc[:n_eq]
    y[up] += yc[n_eq:n_eq + n_up]
    y[lo] -= yc[n_eq + n_up:]
    return y


def run_baseline_proc(d: dict, solver: str, eps: float, timeout_s: float = 600.0,
                      scratch: str | None = None):
    """Solve `d` with `solver` in its isolated interpreter. Returns the same dict `solve` does.

    An import failure here is reported as an error record rather than raising, so one missing
    baseline environment cannot be mistaken for that baseline losing.
    """
    py = _env_python(solver)
    tmpdir = scratch or tempfile.gettempdir()
    with tempfile.NamedTemporaryFile(suffix=".npz", dir=tmpdir, delete=False) as fh:
        inst = fh.name
    out = inst + ".out.npz"
    try:
        dump_instance(d, inst)
        cmd = [py, os.path.abspath(__file__), inst, solver, repr(float(eps)),
               repr(float(timeout_s)), out]
        env = dict(os.environ, JAX_PLATFORMS="cpu")     # baselines are CPU, OSQP's CUDA build is unstable
        k = os.environ.get("BASELINE_THREADS", "").strip()
        pools = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "RAYON_NUM_THREADS")
        if k:
            # a matched-thread comparison must also bound the pools a solver does not expose (BLAS, OpenMP, rayon)
            env.update({v: k for v in pools})
        elif solver == "ecos":
            # ECOS itself is single-threaded; the only BLAS call is our dense factor of P, and a 20-thread OpenBLAS pool
            # spent 0.13 s on CVXQP1_S's 100x100 eigh that takes 1 ms on one thread (2026-09-15), charging ECOS 7x
            env.update({v: "1" for v in pools})
        # the grace period beyond the solver's own time limit: generous by default, but a ROUTER under a wall-clock
        # budget sets it to a few seconds so one engine call cannot overshoot the budget it was given
        margin = float(os.environ.get("OPTEVOLVE_BASELINE_HARD_MARGIN", "120"))
        pr = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s + margin, env=env)
        if pr.returncode == UNSUPPORTED_EXIT and pr.stdout.strip():
            return {**json.loads(pr.stdout.strip().splitlines()[-1]), "solver_python": py}
        if pr.returncode != 0:
            return {"error": (pr.stderr or pr.stdout).strip()[:400], "solver_python": py}
        rec = json.loads(pr.stdout.strip().splitlines()[-1])
        z = np.load(out)
        rec["x"], rec["y"] = z["x"], z["y"]
        rec["solver_python"] = py
        return rec
    except subprocess.TimeoutExpired:
        return {"error": f"timeout after {timeout_s}s", "solver_python": py}
    finally:
        for p in (inst, out):
            try:
                os.unlink(p)
            except OSError:
                pass


if __name__ == "__main__":
    inst, solver, eps, tmo, out = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
    try:
        r = solve(load_instance(inst), solver, eps, tmo)
    except Unsupported as e:
        print(json.dumps({"error": f"unsupported: {e}", "unsupported": True}))
        sys.exit(UNSUPPORTED_EXIT)
    np.savez(out, x=np.asarray(r.pop("x")), y=np.asarray(r.pop("y")))
    print(json.dumps(r))
