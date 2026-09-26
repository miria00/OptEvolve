"""TV-regularized Poisson deconvolution.

    min_x  sum_i (Ax)_i - y_i log((Ax)_i)  +  lam * TV_iso(x)    s.t. x >= 0

This is the convex problem that fluorescence microscopy and PET actually pose:
photon counting makes the noise Poisson, not Gaussian.  The packaged baseline
everyone reaches for, skimage's richardson_lucy, solves a DIFFERENT problem,
the unregularized Poisson maximum likelihood, so it is reported here as such
rather than as a competitor on this objective.  The convex TV-regularized
problem has no packaged solver, so the field is generic conic solvers.

The data term has no Lipschitz gradient (it blows up as Ax -> 0), so the
smooth-plus-proximable split used for the Gaussian case does not apply.  The
formulation gene relocates BOTH the data term and the TV term behind the
linear map, K = [A; D], leaving g = the non-negativity indicator whose
projection is a clip.  Both dual proxes are closed form: the Poisson conjugate
has an exact root, and the TV block is a ball projection.
"""
from __future__ import annotations
import numpy as np
import sys, os
sys.path.insert(0, "/home/miria/OptEvolve/v2/scripts/discovery/tv")
from tv_core import grad, div, tv_iso
from deblur_core import make_kernel, blur, blurT


def poisson_nll(Ax, y, eps=1e-12):
    Ax = np.maximum(Ax, eps)
    return float(np.sum(Ax - y * np.log(Ax)))


def primal(x, y, Kf, lam):
    if x.min() < -1e-12:
        return np.inf
    return poisson_nll(blur(x, Kf), y) + lam * tv_iso(x)


def prox_poisson_conj(u, y, sig):
    """prox of sigma * (Poisson NLL)^* at u, elementwise closed form."""
    return 0.5 * (u + 1.0 - np.sqrt((u - 1.0) ** 2 + 4.0 * sig * y))


def make_data(clean, Kf, peak=100.0, seed=0):
    """Poisson counts from a blurred image, scaled to a photon budget."""
    rng = np.random.default_rng(seed)
    lam_img = blur(np.maximum(clean, 0.0), Kf)
    lam_img = lam_img / lam_img.max() * peak
    return rng.poisson(lam_img).astype(np.float64), lam_img
