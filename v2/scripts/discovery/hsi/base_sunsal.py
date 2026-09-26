"""Baseline specialist: SUnSAL-style ADMM for fully constrained unmixing.

Bioucas-Dias and Figueiredo's SUnSAL solves exactly this problem by ADMM with
one Cholesky of (D^T D + rho I) shared across pixels, which is the strongest
CPU method for it and the one the unmixing literature uses.  It is implemented
here as published: numpy, whole matrix at once, multithreaded BLAS, and its
OWN stopping rule based on primal and dual residuals.

It is then graded by our acceptance check, not by its stopping rule, and it is
offered a ladder of its own tolerance settings.  Setup (Gram, factorization) is
inside the clock, as it is for our arm.
"""
import argparse, json, os, sys, time
import numpy as np
from scipy.linalg import cho_factor, cho_solve

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fcls_core import fw_gap_and_feas, project_simplex_np


def proj_fast(V):
    """Sort-based exact simplex projection, measured faster on CPU than a
    bisection on the dual scalar (13.3 ms against 58.3 ms at 2000 columns)."""
    n = V.shape[0]
    U = -np.sort(-V, axis=0)
    css = np.cumsum(U, axis=0) - 1.0
    ind = np.arange(1, n + 1).reshape(-1, 1)
    r = n - 1 - np.argmax((U - css / ind > 0)[::-1], axis=0)
    th = css[r, np.arange(V.shape[1])] / (r + 1.0)
    return np.maximum(V - th, 0.0)


def sunsal(D, Y, rho=1.0, tol=1e-8, max_iter=1000000, budget_s=1800.0, alpha=1.6,
           use_inverse=False, check_every=1):
    """SUnSAL's ADMM, given its best CPU form.

    BASELINE POLICY: naive, out of the box.  This runs SUnSAL as published,
    factorizing (D^T D + rho I) once and calling cho_solve each iteration, with
    its residual norms computed every iteration as its own stopping rule
    requires.  We measured a faster form for it (an explicit inverse, which
    turns the step into one GEMM, plus residuals only on check iterations, 1.74x
    overall) and that form is available behind --use-inverse as a disclosed
    control, but it is NOT the headline: a reviewer can check the published
    implementation and cannot check our private improvement of it.
    """
    n = D.shape[1]
    t0 = time.perf_counter()
    DtD = D.T @ D
    B = D.T @ Y
    A = DtD + rho * np.eye(n)
    if use_inverse:
        Minv = np.linalg.inv(A)
        step = lambda R: Minv @ R
    else:
        c = cho_factor(A)
        step = lambda R: cho_solve(c, R)
    Z = np.full((n, Y.shape[1]), 1.0 / n)
    U = np.zeros_like(Z)
    it = 0
    scale = np.sqrt(Z.size)
    while it < max_iter and time.perf_counter() - t0 < budget_s:
        X = step(B + rho * (Z - U))
        XA = alpha * X + (1.0 - alpha) * Z
        Zn = proj_fast(XA + U)
        U = U + XA - Zn
        it += 1
        if it % check_every == 0:
            pr = np.linalg.norm(XA - Zn)
            dr = rho * np.linalg.norm(Zn - Z)
            Z = Zn
            if pr <= tol * scale and dr <= tol * scale:
                break
        else:
            Z = Zn
    return Z, it, time.perf_counter() - t0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mat", default="/home/miria/data/hsi/Cuprite.mat")
    ap.add_argument("--pixels", type=int, default=0)
    ap.add_argument("--rho", type=float, default=1.0)
    ap.add_argument("--alpha", type=float, default=1.6)
    ap.add_argument("--tol", type=float, default=1e-8)
    ap.add_argument("--budget", type=float, default=1800.0)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--use-inverse", action="store_true",
                    help="disclosed control only; the headline runs the published form")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    if a.threads:
        for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ[v] = str(a.threads)
    import scipy.io as sio
    d = sio.loadmat(a.mat)
    D = np.asarray(d["D"], dtype=np.float64); Y = np.asarray(d["Y"], dtype=np.float64)
    if a.pixels: Y = Y[:, : a.pixels]
    Z, it, el = sunsal(D, Y, rho=a.rho, tol=a.tol, budget_s=a.budget,
                       alpha=a.alpha, use_inverse=a.use_inverse)
    gw, gmed, feas, fsum = fw_gap_and_feas(D, Y, Z)
    r = {"arm": "sunsal_admm_cpu", "form": "inverse_control" if a.use_inverse else "published_out_of_box", "rho": a.rho, "alpha": a.alpha,
         "own_tol": a.tol, "iters": it, "pixels": Y.shape[1], "wall": el,
         "worst_gap": gw, "median_gap": gmed, "feas": feas, "obj": fsum,
         "it_per_s": it / el}
    print(json.dumps(r, indent=1))
    if a.out: open(a.out, "w").write(json.dumps(r, indent=1))
