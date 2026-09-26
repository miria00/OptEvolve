"""Exact network-simplex reference for a DOTmark transport instance, run in POT's OWN venv.

POT is not installed in jaxenv2, /workspace/venv or any baseline env, and must not be: this script is executed by
OPTEVOLVE_POT_PYTHON (default /root/baseline_envs/venv_pot/bin/python) as a subprocess and prints one JSON line.

The optimum it reports is the optimum of the SAME LP run_dotmark builds, up to the floating-point difference between
the marginals here (image / image.sum()) and the marginals of the OSQP dict (the row and column sums of the product
coupling, which equal these to a relative 1e-16). Network simplex is exact in the sense that matters: it terminates
at a vertex with a certificate, so the objective is not a tolerance-limited number the way an interior-point or
first-order reference is.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import dotmark_ot


def main():
    cls, pair, res, p = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4])
    i1, i2 = dotmark_ot.PAIRS[pair]
    im1 = dotmark_ot.load_image(cls, dotmark_ot.IMAGE_IDS[i1], res)
    im2 = dotmark_ot.load_image(cls, dotmark_ot.IMAGE_IDS[i2], res)
    a = np.ascontiguousarray(im1.ravel() / im1.sum())
    b = np.ascontiguousarray(im2.ravel() / im2.sum())
    b = b * (a.sum() / b.sum())                      # POT rejects marginals whose sums differ
    M = res * res
    C = np.ascontiguousarray(dotmark_ot.cost_vector(res, p).reshape(M, M))
    import ot
    t0 = time.perf_counter()
    val, log = ot.emd2(a, b, C, numItermax=10_000_000, log=True, numThreads=int(os.environ.get("OMP_NUM_THREADS", 1)))
    rec = {"solver": "pot_network_simplex", "pot_version": ot.__version__, "obj": float(val),
           "t_solve_s": time.perf_counter() - t0, "warning": log.get("warning"),
           "m": M, "n": M, "mass_a": float(a.sum()), "mass_b": float(b.sum())}
    print(json.dumps(rec))


if __name__ == "__main__":
    main()
