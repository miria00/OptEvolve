"""Regime 5 baselines: qpth and cvxpylayers, strictly as shipped.

RULE: nothing is done to make these faster.  QPFunction() is constructed with
NO arguments, so eps=1e-12, maxIter=20, PDIPM_BATCHED and check_Q_spd=True are
whatever the library ships.  CvxpyLayer is called with no solver_args.  The
forward solve is what is timed, which is what dominates a layer's runtime.

Both are graded by our acceptance check against a reference objective computed
once outside every timed region, never by their own convergence report.
"""
import argparse, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qp_core import make_problem, score

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=64); ap.add_argument("--m", type=int, default=48)
ap.add_argument("--B", type=int, default=1024)
ap.add_argument("--arm", default="qpth", choices=["qpth", "cvxpylayers"])
ap.add_argument("--device", default="cuda")
ap.add_argument("--budget", type=float, default=600.0)
ap.add_argument("--out", default="")
a = ap.parse_args()

Q, G, h, qs = make_problem(n=a.n, m=a.m, B=a.B, seed=0)
fstar = np.load(f"/home/miria/data/qp/ref_n{a.n}_m{a.m}_B{a.B}.npy")
res = {"arm": a.arm, "n": a.n, "m": a.m, "B": a.B, "device": a.device,
       "rule": "shipped defaults, no arguments"}

import torch
dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
res["torch_device"] = str(dev)

if a.arm == "qpth":
    from qpth.qp import QPFunction
    Qt = torch.tensor(Q, dtype=torch.float64, device=dev)
    pt = torch.tensor(qs, dtype=torch.float64, device=dev)
    Gt = torch.tensor(G, dtype=torch.float64, device=dev)
    ht = torch.tensor(h, dtype=torch.float64, device=dev)
    e = torch.empty(0, dtype=torch.float64, device=dev)
    torch.cuda.synchronize() if dev.type == "cuda" else None
    t0 = time.perf_counter()
    try:
        Z = QPFunction()(Qt, pt, Gt, ht, e, e)        # NO ARGUMENTS
        torch.cuda.synchronize() if dev.type == "cuda" else None
        el = time.perf_counter() - t0
        Zh = Z.detach().cpu().numpy().astype(np.float64)
        gw, gmed, feas, _ = score(Q, G, h, qs, Zh, fstar)
        res.update({"wall": el, "rel_worst": gw, "rel_median": gmed, "feas": feas,
                    "accept": max(gw, feas)})
    except Exception as ex:
        res.update({"error": "%s: %s" % (type(ex).__name__, str(ex)[:200]),
                    "wall": time.perf_counter() - t0})

else:
    import cvxpy as cp
    from cvxpylayers.torch import CvxpyLayer
    t0 = time.perf_counter()
    try:
        z = cp.Variable(a.n); qp = cp.Parameter(a.n)
        prob = cp.Problem(cp.Minimize(0.5 * cp.quad_form(z, cp.psd_wrap(Q)) + qp @ z),
                          [G @ z <= h])
        layer = CvxpyLayer(prob, parameters=[qp], variables=[z])
        pt = torch.tensor(qs, dtype=torch.float64, device=dev)
        Z, = layer(pt)                                  # NO solver_args
        torch.cuda.synchronize() if dev.type == "cuda" else None
        el = time.perf_counter() - t0
        Zh = Z.detach().cpu().numpy().astype(np.float64)
        gw, gmed, feas, _ = score(Q, G, h, qs, Zh, fstar)
        res.update({"wall": el, "rel_worst": gw, "rel_median": gmed, "feas": feas,
                    "accept": max(gw, feas)})
    except Exception as ex:
        res.update({"error": "%s: %s" % (type(ex).__name__, str(ex)[:200]),
                    "wall": time.perf_counter() - t0})

print(json.dumps(res, indent=1))
if a.out: open(a.out, "w").write(json.dumps(res, indent=1))
