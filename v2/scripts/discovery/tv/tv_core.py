"""Isotropic total variation denoising, with a rigorous primal-dual gap.

    min_x  1/2||x - y||^2 + lam * sum_i ||(Dx)_i||_2      (optionally x in [0,1])

D is the 2D forward-difference gradient with Neumann boundary, so ||D||^2 <= 8.
The dual variable p lives in the product of 2-norm balls of radius lam and

    G(p) = min_x { 1/2||x-y||^2 + <x, D^T p> }

is available in closed form, so P(x) - G(p) is a certificate computable from
ANY x and any feasible p, including a competitor's output.  Isotropic TV has
no exact combinatorial solver: parametric max-flow solves the anisotropic case
only, which is why the earlier TV cell in this project had an empty field.
"""
from __future__ import annotations
import numpy as np


def grad(x):
    gx = np.zeros_like(x); gy = np.zeros_like(x)
    gx[:-1, :] = x[1:, :] - x[:-1, :]
    gy[:, :-1] = x[:, 1:] - x[:, :-1]
    return gx, gy


def div(px, py):
    """Negative adjoint of grad: <grad x, p> = <x, -div p>."""
    d = np.zeros_like(px)
    d[0, :] = px[0, :]
    d[1:-1, :] = px[1:-1, :] - px[:-2, :]
    d[-1, :] = -px[-2, :]
    e = np.zeros_like(py)
    e[:, 0] = py[:, 0]
    e[:, 1:-1] = py[:, 1:-1] - py[:, :-2]
    e[:, -1] = -py[:, -2]
    return d + e


def tv_iso(x):
    gx, gy = grad(x)
    return float(np.sqrt(gx * gx + gy * gy).sum())


def primal(x, y, lam, box=False):
    if box and (x.min() < -1e-9 or x.max() > 1 + 1e-9):
        return np.inf
    return 0.5 * float(((x - y) ** 2).sum()) + lam * tv_iso(x)


def dual_from_p(px, py, y, lam, box=False):
    """G(p) after clipping p into its feasible set, so the bound is valid."""
    nrm = np.sqrt(px * px + py * py)
    sc = np.minimum(1.0, lam / np.maximum(nrm, 1e-300))
    px, py = px * sc, py * sc
    dtp = -div(px, py)                    # D^T p
    if box:
        xs = np.clip(y - dtp, 0.0, 1.0)
    else:
        xs = y - dtp
    return 0.5 * float(((xs - y) ** 2).sum()) + float((xs * dtp).sum())


def certify(x, y, lam, px, py, box=False):
    P = primal(x, y, lam, box)
    G = dual_from_p(px, py, y, lam, box)
    return (P - G) / (1.0 + abs(P)), P, G


def dual_from_x(x, y, lam, box=False):
    """A feasible dual point built from a primal iterate, so a competitor that
    returns only x can still be certified: take p proportional to the
    subgradient direction of the TV term, scaled into the ball."""
    gx, gy = grad(x)
    nrm = np.sqrt(gx * gx + gy * gy)
    sc = lam / np.maximum(nrm, 1e-12)
    px, py = gx * sc, gy * sc
    best = None
    for s in (1.0, 0.99, 0.95, 0.9, 0.75, 0.5, 0.25):
        g = dual_from_p(px * s, py * s, y, lam, box)
        best = g if best is None else max(best, g)
    P = primal(x, y, lam, box)
    return (P - best) / (1.0 + abs(P)), P, best
