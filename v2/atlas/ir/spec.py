"""Problem IR: the E1 composite form  min_x f(x) + g(x) + h(L x).

Each block is optional. The canonical P0 instance is anisotropic TV
denoising of an image y:

    min_x (1/2)||x - y||^2 + lam * ||D x||_1

with f = (1/2)||. - y||^2 (1-smooth, 1-strongly convex), g = 0,
h = lam*||.||_1, L = D the 2D forward-difference gradient with Neumann
boundary (2 output channels), ||D||^2 <= 8.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import jax.numpy as jnp

from atlas.operators.linear import Grad2D, LinOp
from atlas.operators.prox import Box, QuadraticDataFidelity, ScaledL1, Zero


@dataclass(frozen=True)
class CompositeProblem:
    """E1 composite problem  min_x f(x) + g(x) + h(L x); blocks optional.

    f, g, h are atoms (see atlas.operators); L is a LinOp. `data` carries
    the problem arrays/scalars (e.g. y, lam for TV denoising); `shape` is
    the primal variable shape; `name` tags the instance family.
    """

    f: Optional[Any] = None
    g: Optional[Any] = None
    h: Optional[Any] = None
    L: Optional[LinOp] = None
    data: dict = field(default_factory=dict)
    shape: tuple = ()
    name: str = "composite"

    @property
    def is_tv_denoise(self) -> bool:
        return self.name == "tv_denoise"

    @property
    def has_linear_map(self) -> bool:
        return self.L is not None and self.h is not None


def tv_denoise_problem(y, lam: float) -> CompositeProblem:
    """Build the canonical TV-denoising instance for image y (H, W)."""
    y = jnp.asarray(y, dtype=jnp.float64)
    if y.ndim != 2:
        raise ValueError(f"tv_denoise_problem expects a 2D image, got shape {y.shape}")
    if lam <= 0:
        raise ValueError("lam must be > 0")
    return CompositeProblem(
        f=QuadraticDataFidelity(y=y),
        g=Zero(),
        h=ScaledL1(lam=float(lam)),
        L=Grad2D(shape=tuple(y.shape)),
        data={"y": y, "lam": float(lam)},
        shape=tuple(y.shape),
        name="tv_denoise",
    )


def lasso_box_problem(y, lam: float, lo: float = 0.0, hi: float = 1.0) -> CompositeProblem:
    """Three-block composite WITHOUT a linear map (canonical DYS instance):

        min_x (1/2)||x - y||^2 + lam*||x||_1 + indicator_{[lo, hi]}(x)

    f = quadratic data fidelity (1-smooth), g = lam*||.||_1 (prox: soft
    threshold), h = box indicator (prox: clip), L = None. This is the E1
    form with the identity in place of L, i.e. exactly the setting of
    Davis-Yin three-operator splitting [LSCOMO SS2.7].
    """
    y = jnp.asarray(y, dtype=jnp.float64)
    if lam <= 0:
        raise ValueError("lam must be > 0")
    if not lo < hi:
        raise ValueError("need lo < hi")
    return CompositeProblem(
        f=QuadraticDataFidelity(y=y),
        g=ScaledL1(lam=float(lam)),
        h=Box(lo=float(lo), hi=float(hi)),
        L=None,
        data={"y": y, "lam": float(lam), "lo": float(lo), "hi": float(hi)},
        shape=tuple(y.shape),
        name="lasso_box",
    )
