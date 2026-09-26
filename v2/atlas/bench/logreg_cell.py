"""Elastic-net logistic regression as a composite problem.

    min_w  F(w) + G(w),  F(w) = (1/n) sum_i log(1 + exp(-y_i x_i^T w)),
                         G(w) = lam * (alpha ||w||_1 + (1 - alpha)/2 ||w||_2^2),  y_i in {-1, +1}.

This is the objective of skglm's SparseLogisticRegression(alpha=lam,
l1_ratio=alpha) and of scikit-learn / cuML LogisticRegression with
penalty="elasticnet", l1_ratio=alpha, C = 1/(n lam), all without intercept.
grad F is Lipschitz with L_f = ||X||_2^2 / (4 n); the prox of G is a scaled
soft threshold, closed form in any diagonal metric. The certificate is the
relative duality gap of atlas.certify.residuals.logreg_enet_rel_gap.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from atlas.ir.spec import CompositeProblem
from atlas.operators.base import PropSet


@dataclass(frozen=True)
class LogisticLoss:
    X: Any  # (n, p) design (dense array), dynamic leaf
    y: Any  # (n,) labels in {-1, +1}, dynamic leaf
    props: PropSet = field(default=PropSet(prox_friendly=False))

    def value(self, w):
        return jnp.mean(jnp.logaddexp(0.0, -self.y * (self.X @ w)))

    def grad(self, w):
        z = self.y * (self.X @ w)
        return -(self.X.T @ (self.y * jax.nn.sigmoid(-z))) / self.X.shape[0]


@dataclass(frozen=True)
class ElasticNet:
    lam: Any    # dynamic, so a regularisation path does not recompile
    alpha: Any  # l1 ratio in [0, 1]
    props: PropSet = field(default=PropSet(averagedness=0.5, prox_friendly=True))

    def value(self, w):
        return self.lam * (self.alpha * jnp.sum(jnp.abs(w)) + 0.5 * (1.0 - self.alpha) * jnp.sum(w * w))

    def prox(self, v, step):
        thr = step * self.lam * self.alpha
        return jnp.sign(v) * jnp.maximum(jnp.abs(v) - thr, 0.0) / (1.0 + step * self.lam * (1.0 - self.alpha))


jax.tree_util.register_dataclass(LogisticLoss, data_fields=["X", "y"], meta_fields=["props"])
jax.tree_util.register_dataclass(ElasticNet, data_fields=["lam", "alpha"], meta_fields=["props"])


def lipschitz_bound(X, iters: int = 100, safety: float = 1.02) -> float:
    """||X||_2^2 / (4n) by power iteration on X^T X, times a safety factor."""
    n, p = X.shape
    v = np.random.default_rng(0).normal(size=p); v /= np.linalg.norm(v)
    for _ in range(iters):
        w = X.T @ (X @ v); lam = float(np.linalg.norm(w)); v = w / lam
    return safety * lam / (4.0 * n)


def lam_max(X, y, alpha: float) -> float:
    """Smallest lam with w = 0 optimal (alpha > 0): ||X^T y||_inf / (2 n alpha)."""
    return float(np.max(np.abs(X.T @ y)) / (2.0 * X.shape[0] * alpha))


def composite(X, y, lam: float, alpha: float, dtype=jnp.float64) -> CompositeProblem:
    Xj, yj = jnp.asarray(X, dtype=dtype), jnp.asarray(y, dtype=dtype)
    Lf = lipschitz_bound(np.asarray(X, dtype=np.float64))
    f = LogisticLoss(X=Xj, y=yj, props=PropSet(lipschitz=Lf, cocoercivity=1.0 / Lf, prox_friendly=False,
                                                 strong_convexity=0.0))
    g = ElasticNet(lam=jnp.asarray(lam, dtype=dtype), alpha=jnp.asarray(alpha, dtype=dtype))
    return CompositeProblem(f=f, g=g, h=None, L=None, data={"n": int(X.shape[0]), "p": int(X.shape[1])},
                            shape=(int(X.shape[1]),), name="logreg_enet")
