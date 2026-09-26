"""Outer-wrapper registry: the knowledge base entries the search may draw.

Each entry ties a genome `wrapper.kind` to (a) its knowledge-base id in
atlas/kb/schemes.json, (b) its gate rule in atlas.typing_.rules, (c) the
iteration mode that implements it in atlas.wrappers.km / the policy runner,
and (d) the schemes on which it is admitted in this build. Legacy wrapper
genes (wrapper.halpern / wrapper.restart) are not listed here; they keep their
original fields so every existing genome hash is unchanged.

A new KB scheme enters the search by adding one entry here, one gate rule,
one implementation, and an admission test; nothing else in the proposer
changes.
"""
from __future__ import annotations

from atlas.typing_ import TypeCheckError


def _gate_haugazeau(cert, scheme):
    from atlas.typing_.rules import typecheck_haugazeau
    return typecheck_haugazeau(cert, scheme)


REGISTRY = {
    "haugazeau": {
        "kb_id": "haugazeau",
        "gate": _gate_haugazeau,
        "mode": "haugazeau",
        # the averagedness metric is supplied (BuiltScheme.gram) and
        # admission-tested for these schemes only
        "schemes": ("pdhg", "condat_vu"),
    },
}


def kinds() -> tuple:
    return tuple(REGISTRY)


def gate(kind: str, cert, scheme: str):
    if kind not in REGISTRY:
        raise TypeCheckError(f"unknown wrapper kind {kind!r}; registered: {list(REGISTRY)}")
    return REGISTRY[kind]["gate"](cert, scheme)


def admitted_for(scheme: str) -> tuple:
    return tuple(k for k, v in REGISTRY.items() if scheme in v["schemes"])
