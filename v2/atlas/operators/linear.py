"""Linear operator protocol and the 2D forward-difference gradient.

LinOp contract: forward(x), adjoint(u), norm_bound with
||L|| <= norm_bound, verified by the adjoint unit test
<L x, u> == <x, L^T u> on random inputs (tests/test_operators.py).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import jax.numpy as jnp


@runtime_checkable
class LinOp(Protocol):
    norm_bound: float

    def forward(self, x): ...

    def adjoint(self, u): ...


@dataclass(frozen=True)
class Grad2D:
    """2D forward-difference gradient with Neumann boundary conditions.

    forward: x of shape (H, W) -> u of shape (2, H, W):
        u[0, i, j] = x[i+1, j] - x[i, j]  (0 in the last row)
        u[1, i, j] = x[i, j+1] - x[i, j]  (0 in the last column)
    adjoint: negative divergence (exact matrix transpose of forward).
    Operator norm: ||D||^2 <= 8 (standard bound for this discretization,
    cf. Chambolle-Pock), so norm_bound = sqrt(8).
    """

    shape: tuple  # (H, W)
    norm_bound: float = math.sqrt(8.0)

    def forward(self, x):
        d0 = jnp.concatenate([x[1:, :] - x[:-1, :], jnp.zeros_like(x[:1, :])], axis=0)
        d1 = jnp.concatenate([x[:, 1:] - x[:, :-1], jnp.zeros_like(x[:, :1])], axis=1)
        return jnp.stack([d0, d1], axis=0)

    def adjoint(self, u):
        u0, u1 = u[0], u[1]
        # Rows H-1 / cols W-1 of the forward output are structurally zero;
        # the adjoint must ignore those input entries.
        t0 = u0.at[-1, :].set(0.0)
        t1 = u1.at[:, -1].set(0.0)
        # (D0^T t0)[j] = t0[j-1] - t0[j] with t0[-1] := 0 (negative divergence).
        v0 = jnp.concatenate([jnp.zeros_like(t0[:1, :]), t0[:-1, :]], axis=0) - t0
        v1 = jnp.concatenate([jnp.zeros_like(t1[:, :1]), t1[:, :-1]], axis=1) - t1
        return v0 + v1
