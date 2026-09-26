"""Worst-case fixed-point residual of a composed fixed-coefficient recipe, by performance
estimation (PEPit). Runs in the isolated environment /home/miria/pepenv.

Class: T = (1 - alpha) I + alpha N, N nonexpansive (T alpha-averaged). Recipe: F =
(1 - rho) I + rho T = (1 - rho alpha) I + rho alpha N, iterated N times plainly
(Krasnoselskii-Mann) or with the Halpern anchor lambda_k = 1/(k+2). Metric: ||x_N - T x_N||^2
under ||x_0 - x*||^2 <= 1. Reads a JSON list of {alpha, rho, anchor, N} on stdin, prints a JSON
list of worst cases (None when the SDP fails, e.g. an expansive recipe)."""
import json, sys
from PEPit import PEP
from PEPit.operators import LipschitzOperator


def worst(alpha, rho, anchor, N):
    problem = PEP()
    Nop = problem.declare_function(LipschitzOperator, L=1.)
    xs, _, _ = Nop.fixed_point()
    x0 = problem.set_initial_point()
    problem.set_initial_condition((x0 - xs) ** 2 <= 1)
    a = rho * alpha
    x = x0
    for k in range(N):
        Fx = (1 - a) * x + a * Nop.gradient(x)
        x = (1 / (k + 2)) * x0 + (1 - 1 / (k + 2)) * Fx if anchor else Fx
    problem.set_performance_metric((x - ((1 - alpha) * x + alpha * Nop.gradient(x))) ** 2)
    return float(problem.solve(verbose=0))


out = []
for spec in json.load(sys.stdin):
    try:
        out.append(worst(**spec))
    except Exception:
        out.append(None)
print(json.dumps(out))
