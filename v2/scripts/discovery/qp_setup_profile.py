"""Attribute the QP setup cost, which is 97% of our lasso time, before trying to remove any of it.

At lasso n_var 50,000 our 5.268 s decomposed as rescale 2.187 + prepare 2.419 + compile 0.619 +
SOLVE 0.044. Reading the code says the rescale runs twice (qp_scale_sweep's run_ours calls
rescale_qp_problem for tau/sigma, then prepare_base -> _prepare_with_metric calls it again), and that
each rescale pays two 500-iteration CPU power iterations at tol 1e-13. This measures each piece
separately so the fix is aimed at what actually costs, rather than at what looks expensive.

Everything here except the build is numpy/scipy on CPU, so it can run without a free GPU.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import scipy.sparse as sp

ap = argparse.ArgumentParser()
ap.add_argument("--family", default="lasso")
ap.add_argument("--size", type=int, default=20000)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--ruiz-iters", type=int, default=10)
ap.add_argument("--power-iters", default="500,100,50,20",
                help="power-iteration budgets to time and compare; 500 is the default the solver uses")
ap.add_argument("--out", required=True)
a = ap.parse_args()

import jax

jax.config.update("jax_enable_x64", True)

from atlas.bench import qp_cell
from atlas.bench.qp_gen import FAMILIES
from atlas.schemes.rescale import (kkt_ruiz_equilibrate, power_iteration_norm_bound,
                                   rescale_qp_problem)


def clock(label, fn):
    t0 = time.perf_counter()
    out = fn()
    dt = time.perf_counter() - t0
    print(f"  {label:<46s} {dt:8.3f} s", flush=True)
    return dt, out


rec = {"family": a.family, "size": a.size, "seed": a.seed, "phases": {}, "power_iteration": {}}
d = FAMILIES[a.family](a.size, seed=a.seed)
P, A = sp.csc_matrix(d["P"]), sp.csc_matrix(d["A"])
rec.update(n_var=P.shape[0], n_con=A.shape[0], nnz_P=int(P.nnz), nnz_A=int(A.nnz))
print(f"  {a.family}({a.size}): n_var={P.shape[0]} n_con={A.shape[0]} "
      f"nnz(P)={P.nnz} nnz(A)={A.nnz}", flush=True)

rec["phases"]["to_composite"], prob = clock("to_composite", lambda: qp_cell.to_composite(d))
rec["phases"]["kkt_ruiz_equilibrate"], (d1, d2) = clock(
    f"kkt_ruiz_equilibrate(iters={a.ruiz_iters})", lambda: kkt_ruiz_equilibrate(P, A, a.ruiz_iters))

A_s = (sp.diags(d1) @ A @ sp.diags(d2)).tocsc()
P_s = (sp.diags(d2) @ P @ sp.diags(d2)).tocsc()

# The two power iterations a single rescale pays: operator bound on A, and L_f on P.
for budget in [int(v) for v in a.power_iters.split(",")]:
    tA, nbA = clock(f"power_iteration_norm_bound(A, iters={budget})",
                    lambda b=budget: power_iteration_norm_bound(A_s, iters=b))
    tP, nbP = clock(f"power_iteration_norm_bound(P, iters={budget})",
                    lambda b=budget: power_iteration_norm_bound(P_s, iters=b))
    rec["power_iteration"][str(budget)] = {"t_A_s": tA, "nb_A": float(nbA),
                                           "t_P_s": tP, "nb_P": float(nbP), "total_s": tA + tP}

# One full rescale, which is what _prepare_with_metric ALSO does internally.
rec["phases"]["rescale_qp_problem_once"], _ = clock(
    "rescale_qp_problem (one call)",
    lambda: rescale_qp_problem(prob, "ruiz", iters=a.ruiz_iters))

base = rec["power_iteration"].get("500", {})
if base.get("nb_A"):
    print("\n  accuracy cost of the 500-iteration budget (bound is inflated x1.02 anyway):", flush=True)
    for k, v in sorted(rec["power_iteration"].items(), key=lambda kv: -int(kv[0])):
        dA = abs(v["nb_A"] - base["nb_A"]) / base["nb_A"]
        dP = abs(v["nb_P"] - base["nb_P"]) / max(base["nb_P"], 1e-300)
        print(f"    iters={k:<4s} total {v['total_s']:7.3f} s   rel change vs 500: "
              f"||A|| {dA:.2e}  ||P|| {dP:.2e}", flush=True)
        v["rel_change_A_vs_500"], v["rel_change_P_vs_500"] = dA, dP

json.dump(rec, open(a.out, "w"), indent=1, default=str)
print("\n  NOTE: the solver pays TWO of these rescales per solve (run_ours + _prepare_with_metric).")
print("QP_SETUP_PROFILE_DONE")
