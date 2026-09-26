"""Atom protocol and property sets.

Property tags follow LSCOMO:
- Lipschitz constant L of a gradient map; its cocoercivity constant is
  beta = 1/L by the Baillon-Haddad theorem [LSCOMO SS2.2.1].
- Proximal operators of proper closed convex functions are resolvents of
  maximal monotone operators and hence (1/2)-averaged [LSCOMO SS2.5].
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable


@dataclass(frozen=True)
class PropSet:
    """Operator property tags used by the type system.

    lipschitz:     Lipschitz constant of the map (for gradients: L).
    cocoercivity:  cocoercivity constant beta; for grad f with L-Lipschitz
                   gradient, beta = 1/L [LSCOMO SS2.2.1, Baillon-Haddad].
    averagedness:  alpha such that the map is alpha-averaged; proxes and
                   projections are (1/2)-averaged [LSCOMO SS2.5].
    prox_friendly: True if the atom has a cheap closed-form prox.
    strong_convexity: mu of the underlying function (0.0 if merely convex).
    """

    lipschitz: Optional[float] = None
    cocoercivity: Optional[float] = None
    averagedness: Optional[float] = None
    prox_friendly: bool = False
    strong_convexity: float = 0.0


@runtime_checkable
class Atom(Protocol):
    """A typed operator atom: a prox map, gradient map, or projection.

    Concrete atoms expose `props: PropSet` plus whichever of
    `prox(v, step)`, `grad(x)`, `project(v)` apply to them.
    """

    props: PropSet
