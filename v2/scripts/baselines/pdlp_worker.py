"""Run PDLP (OR-Tools, the PDLP authors' own code) on one standard-form LP.

Runs in an isolated interpreter because the ortools wheel bundles its own HiGHS
and its libortools.so collides with the separately installed highspy at load
time. Reads an npz of (c, A, b) and prints one JSON line.
"""
import json, sys, time
import numpy as np, scipy.sparse as sp
from ortools.pdlp.python import pdlp
from ortools.pdlp import solvers_pb2

d = np.load(sys.argv[1], allow_pickle=False)
tol, tl = float(sys.argv[2]), float(sys.argv[3])
c = d["c"]; b = d["b"]
A = sp.csr_matrix((d["Ad"], d["Ai"], d["Ap"]), shape=tuple(d["Ashape"]))
n = c.shape[0]

qp = pdlp.QuadraticProgram()
qp.resize_and_initialize(A.shape[0], n)
qp.objective_vector = np.asarray(c, np.float64)
qp.objective_offset = 0.0
qp.constraint_matrix = A
qp.constraint_lower_bounds = np.asarray(b, np.float64)
qp.constraint_upper_bounds = np.asarray(b, np.float64)
qp.variable_lower_bounds = np.zeros(n)
qp.variable_upper_bounds = np.full(n, np.inf)

p = solvers_pb2.PrimalDualHybridGradientParams()
p.termination_criteria.eps_optimal_absolute = tol
p.termination_criteria.eps_optimal_relative = tol
p.termination_criteria.time_sec_limit = tl
t0 = time.perf_counter()
res = pdlp.primal_dual_hybrid_gradient(qp, p)
dt = time.perf_counter() - t0
log = res.solve_log
print(json.dumps({
    "t_s": dt,
    "iters": int(getattr(log, "iteration_count", -1)),
    "termination": int(getattr(log, "termination_reason", -1)),
    "objective": float(np.dot(np.asarray(c), np.asarray(res.primal_solution))),
}))
