"""Primal step operators T (inverse metrics) for Condat-Vu with g = 0.

T is H0 = diag(d) refined by limited-memory BFGS pairs (s_i, y_i) with
(y_i, s_i) > 0, applied by the two-loop recursion (Nocedal 1980), times a
scale c. It satisfies the secant equations T y_i = c s_i and is symmetric
positive definite. Used as the matrix step of the quasi-Newton proximal idea
of Bonnans, Gilbert, Lemarechal and Sagastizabal (Math. Prog. 68, 1995):
for a QP the secant pairs are exact, y = (P + mu I) s.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class PrimalMetric:
    d: Any   # (n,) positive diagonal of H0
    S: Any   # (m, n) secant steps
    Y: Any   # (m, n) secant gradient differences
    c: Any   # scalar scale

    def apply(self, v):
        m = self.S.shape[0]
        rho = 1.0 / jnp.sum(self.Y * self.S, axis=1) if m else None
        q, alphas = v, []
        for i in reversed(range(m)):
            a = rho[i] * jnp.vdot(self.S[i], q)
            q = q - a * self.Y[i]
            alphas.append(a)
        r = self.d * q
        for j, i in enumerate(range(m)):
            b = rho[i] * jnp.vdot(self.Y[i], r)
            r = r + self.S[i] * (alphas[m - 1 - j] - b)
        return self.c * r

    def dense(self) -> np.ndarray:
        n = int(self.d.shape[0])
        return np.stack([np.asarray(self.apply(jnp.eye(n, dtype=jnp.float64)[k])) for k in range(n)], axis=1)


jax.tree_util.register_dataclass(PrimalMetric, data_fields=["d", "S", "Y", "c"], meta_fields=[])


def lbfgs_metric(S, Y, d=None) -> PrimalMetric:
    """Keep pairs with (y, s) > 1e-12 |y||s|; H0 = gamma I (gamma = (s,y)/(y,y) of the last pair) unless d given."""
    S, Y = np.atleast_2d(np.asarray(S, float)), np.atleast_2d(np.asarray(Y, float))
    keep = [i for i in range(S.shape[0]) if S[i] @ Y[i] > 1e-12 * np.linalg.norm(S[i]) * np.linalg.norm(Y[i])]
    S, Y = S[keep], Y[keep]
    if d is None:
        gamma = float(S[-1] @ Y[-1] / (Y[-1] @ Y[-1])) if len(keep) else 1.0
        d = np.full(S.shape[1] if len(keep) else 0, gamma)
    return PrimalMetric(d=jnp.asarray(d), S=jnp.asarray(S), Y=jnp.asarray(Y), c=jnp.asarray(1.0))


def krylov_pairs(P_apply, n: int, m: int, mu: float, seed: int = 0):
    """m normalised Krylov directions of P and their exact secant images (P + mu I) s."""
    rng = np.random.default_rng(seed)
    s = rng.normal(size=n); S, Y = [], []
    for _ in range(m):
        s = s / np.linalg.norm(s)
        y = np.asarray(P_apply(jnp.asarray(s))) + mu * s
        S.append(s.copy()); Y.append(y)
        s = y - (y @ s) * s          # next Krylov direction, orthogonalised against the last
        for v in S:
            s = s - (s @ v) * v
        if np.linalg.norm(s) < 1e-12:
            break
    return np.array(S), np.array(Y)


def power_max(apply, n: int, iters: int = 300, seed: int = 0) -> float:
    """Largest eigenvalue modulus of a linear map with real nonnegative spectrum."""
    v = jnp.asarray(np.random.default_rng(seed).normal(size=n))
    v = v / jnp.linalg.norm(v)

    def body(_, carry):
        v, _ = carry
        w = apply(v)
        lam = jnp.linalg.norm(w)
        return w / lam, lam

    _, lam = jax.lax.fori_loop(0, iters, body, (v, jnp.asarray(0.0)))
    return float(lam)


def gate_quantities(metric: PrimalMetric, P, A, n: int, m_rows: int) -> tuple[float, float]:
    """(lambda_max(T P), ||A T A^T||) by power iteration, with safety factors 1.05 and 1.02."""
    lamP = power_max(jax.jit(lambda v: metric.apply(P.forward(v))), n) * 1.05
    nAT = power_max(jax.jit(lambda u: A.forward(metric.apply(A.adjoint(u)))), m_rows) * 1.02
    return lamP, nAT


def scale_for_gate(metric: PrimalMetric, P, A, n: int, m_rows: int, fill: float = 0.99):
    """Scale T so lambda_max(T P)/2 = fill/2 and put s on the gate boundary."""
    lam0, nat0 = gate_quantities(metric, P, A, n, m_rows)
    if not lam0 > 0:
        raise ValueError("lambda_max(T P) = 0: the quadratic is zero, use an LP scheme")
    c = fill / lam0
    s = (fill / 2.0) / (c * nat0)
    return replace(metric, c=jnp.asarray(c)), s, c * lam0, c * nat0
