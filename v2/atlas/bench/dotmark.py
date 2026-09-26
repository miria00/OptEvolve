"""DOTmark optimal transport instances as matrix-free lp_standard composites.

Data: DOTmark 1.0 (Schrieber, Schuhmacher, Gottschlich 2017, IEEE Access;
arXiv 1610.03368), downloaded from
https://www.stochastik.math.uni-goettingen.de/DOTmark_1.0.zip into
data/dotmark (MANIFEST.json records the zip and per-file sha256). Ten classes
of grayscale images, resolutions 32 to 512, ten images per class and size.

An instance is a pair of images of one class and resolution r: the source and
target measures are the two images normalised to mass 1, and the cost is the
squared Euclidean distance between pixel centres on the integer grid, as in the
DOTmark paper. With m = n = r^2 this is the transport LP on K_{m,n} of
atlas.bench.ot, built WITHOUT a dense incidence matrix (IncidenceLinOp with the
closed-form norm sqrt(m + n) x 1.02).

The marginals imply the total-mass constraint sum(x) = 1, which the composite
records as data["simplex_mass"]: a Bregman (entropic) primal step may place the
primal block on that simplex without changing the feasible set.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from atlas.bench.lp_cell import AffineEqualitySet, IncidenceLinOp, LinearCost, NonnegOrthant
from atlas.bench.ot import bipartite_arcs, transport_opnorm
from atlas.ir.spec import CompositeProblem

ROOT = Path(__file__).resolve().parents[2] / "data" / "dotmark"
CLASSES = ("CauchyDensity", "ClassicImages", "GRFmoderate", "GRFrough", "GRFsmooth",
           "LogGRF", "LogitGRF", "MicroscopyImages", "Shapes", "WhiteNoise")
IMAGE_IDS = tuple(range(1001, 1011))


@dataclass(frozen=True, eq=False)
class TransportLinOp:
    """The K_{m,n} incidence operator of atlas.bench.ot (arc e = i*n + j, tail i,
    head m + j) as DENSE REDUCTIONS instead of scatter-add: with P = x.reshape(m, n),

        A x   = [P 1_n ; -P^T 1_m]          (row sums, minus column sums)
        A^T y = vec(y_S 1_n^T - 1_m y_T^T)   (outer difference)

    Identical in exact arithmetic to IncidenceLinOp on the same arcs (tested);
    what changes is the memory access pattern: coalesced reductions and a
    broadcast instead of atomics. eq=False keeps identity hashing for the
    compile-cache static key, as for the other LinOps."""

    m: int
    n: int
    norm_bound: float

    @property
    def n_nodes(self) -> int:
        return self.m + self.n

    @property
    def shape(self) -> tuple:
        return (self.m + self.n, self.m * self.n)

    def forward(self, x):
        P = x.reshape(self.m, self.n)
        return jnp.concatenate([P.sum(axis=1), -P.sum(axis=0)])

    def adjoint(self, y):
        return (y[: self.m, None] - y[None, self.m:]).reshape(-1)


def image(cls: str, res: int, image_id: int) -> np.ndarray:
    return np.loadtxt(ROOT / "Data" / cls / f"data{res}_{image_id}.csv", delimiter=",")


def grid_cost(res: int) -> np.ndarray:
    """C[p, q] = ||p - q||^2 for pixel centres p, q of an res x res grid (row-major)."""
    ii, jj = np.meshgrid(np.arange(res), np.arange(res), indexing="ij")
    P = np.stack([ii.ravel(), jj.ravel()], 1).astype(np.float64)
    return ((P[:, None, :] - P[None, :, :]) ** 2).sum(-1)


def pair(cls: str, res: int, id_a: int, id_b: int):
    """(a, b, C): the two normalised images and the grid cost."""
    a, b = image(cls, res, id_a).ravel(), image(cls, res, id_b).ravel()
    return a / a.sum(), b / b.sum(), grid_cost(res)


def composite(cls: str, res: int, id_a: int, id_b: int, operator: str = "incidence") -> CompositeProblem:
    """operator "incidence" (scatter-add IncidenceLinOp) or "reduction"
    (TransportLinOp): same operator, different execution structure."""
    a, b, C = pair(cls, res, id_a, id_b)
    m = n = res * res
    if operator == "reduction":
        L = TransportLinOp(m=m, n=n, norm_bound=float(transport_opnorm(m, n)))
    elif operator == "incidence":
        tail, head, n_nodes = bipartite_arcs(m, n)
        L = IncidenceLinOp(tail=jnp.asarray(tail), head=jnp.asarray(head), n_nodes=int(n_nodes),
                           norm_bound=float(transport_opnorm(m, n)))
    else:
        raise ValueError(f"operator must be 'incidence' or 'reduction', got {operator!r}")
    c = jnp.asarray(C.ravel(), dtype=jnp.float64)
    bvec = jnp.asarray(np.concatenate([a, -b]), dtype=jnp.float64)
    meta = {"family": "ot_dotmark", "class": cls, "resolution": int(res), "ids": (int(id_a), int(id_b)),
            "m": m, "n": n, "n_arcs": m * n, "opnorm_exact": math.sqrt(m + n),
            # every column of the incidence matrix is e_tail - e_head: ||L||_{1->2} = sqrt(2)
            "opnorm_1to2": math.sqrt(2.0), "operator": operator,
            "cost": "squared Euclidean distance on the integer pixel grid (DOTmark)"}
    return CompositeProblem(f=LinearCost(c=c), g=NonnegOrthant(), h=AffineEqualitySet(b=bvec), L=L,
                            data={"c": c, "b": bvec, "meta": meta, "simplex_mass": 1.0},
                            shape=(m * n,), name="lp_standard")
