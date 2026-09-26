"""Isotropic TV deblurring with a rigorous primal-dual gap.

    min_x  1/2||A x - y||^2 + mu/2 ||x||^2 + lam * sum_i ||(Dx)_i||_2

A is a circular convolution, so A and A^T are FFTs.  The small ridge mu is
what makes the certificate computable: without it A^T A is singular wherever
the blur kernel's spectrum vanishes, the inner minimization is unbounded below
and there is no finite dual value to compare against.  With it,

    G(p) = min_x { 1/2||Ax-y||^2 + mu/2||x||^2 + <x, D^T p> }

is solved exactly in Fourier space, so P(x) - G(p) certifies ANY x, including
a competitor's.

There is no packaged solver for this problem in the standard imaging stack:
scikit-image offers Wiener filtering and Richardson-Lucy, which solve
different problems, so the comparison field here is generic conic solvers plus
a hand-written first-order competitor.
"""
from __future__ import annotations
import numpy as np
from tv_core import grad, div, tv_iso


def make_kernel(shape, sigma=2.0, ksz=15):
    ax = np.arange(ksz) - ksz // 2
    g = np.exp(-(ax ** 2) / (2 * sigma ** 2)); g /= g.sum()
    k = np.outer(g, g)
    K = np.zeros(shape)
    h = ksz // 2
    K[:ksz, :ksz] = k
    K = np.roll(np.roll(K, -h, axis=0), -h, axis=1)
    return np.fft.rfft2(K)


def blur(x, Kf):
    return np.fft.irfft2(np.fft.rfft2(x) * Kf, s=x.shape)


def blurT(x, Kf):
    return np.fft.irfft2(np.fft.rfft2(x) * np.conj(Kf), s=x.shape)


def primal(x, y, Kf, lam, mu):
    r = blur(x, Kf) - y
    return 0.5 * float((r * r).sum()) + 0.5 * mu * float((x * x).sum()) + lam * tv_iso(x)


def dual_from_p(px, py, y, Kf, lam, mu):
    nrm = np.sqrt(px * px + py * py)
    sc = np.minimum(1.0, lam / np.maximum(nrm, 1e-300))
    px, py = px * sc, py * sc
    dtp = -div(px, py)
    rhs = blurT(y, Kf) - dtp
    denom = np.abs(Kf) ** 2 + mu
    xs = np.fft.irfft2(np.fft.rfft2(rhs) / denom, s=y.shape)
    r = blur(xs, Kf) - y
    return (0.5 * float((r * r).sum()) + 0.5 * mu * float((xs * xs).sum())
            + float((xs * dtp).sum()))


def certify(x, y, Kf, lam, mu, px, py):
    P = primal(x, y, Kf, lam, mu)
    G = dual_from_p(px, py, y, Kf, lam, mu)
    return (P - G) / (1.0 + abs(P)), P, G


def dual_from_x(x, y, Kf, lam, mu):
    """Certificate for a competitor that returns only an image."""
    gx, gy = grad(x)
    nrm = np.sqrt(gx * gx + gy * gy)
    sc = lam / np.maximum(nrm, 1e-12)
    best = None
    for s in (1.0, 0.99, 0.95, 0.9, 0.75, 0.5, 0.25, 0.0):
        g = dual_from_p(gx * sc * s, gy * sc * s, y, Kf, lam, mu)
        best = g if best is None else max(best, g)
    P = primal(x, y, Kf, lam, mu)
    return (P - best) / (1.0 + abs(P)), P, best
