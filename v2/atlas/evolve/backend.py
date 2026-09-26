"""SearchBackend protocol: the seam between the evolution loop and any
proposal engine (random, LLM-guided, SkyDiscover, OpenEvolve, ...).

Contract:
- ask(k) returns k candidate Genomes. Backends SHOULD propose only genomes
  that pass the Stage-0 gate (atlas.genome.validate); the loop re-checks and
  rejects (never clamps) anything that slips through.
- tell(genome, fitness, metrics) feeds back an evaluated candidate. Fitness
  is a scalar, HIGHER IS BETTER (e.g. negative time-to-tolerance or negative
  relative gap at budget); metrics is a free-form dict of auxiliary
  observations (iters, wall_time, rel_gap, ...) that context-building
  backends may surface in prompts.

Optional counters the loop reads if present (per-generation deltas):
- stage0_rejections: candidates the backend itself discarded at the gate.
- n_fallbacks: LLM proposals abandoned for a random fallback.
Optional: reseed(seed) for deterministic re-runs.

SkyDiscover adapter (plan SS3.1): SkyDiscover's four programmable
components map 1:1 onto this seam — its Solution Generator + Context
Builder implement ask() (the mutation menu and archive snapshot become the
context), and its Evaluator + Solution Selector consume tell() (Stage-0 is
the free first Evaluator stage; canonical content_hash dedup lives in the
Selector). A `sky_adapter.py` implementing SearchBackend slots in here in
P2 with no change to loop.evolve; the vendored-OpenEvolve adapter is the
same shape (engine-agnosticism ablation via a config switch).
"""
from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from atlas.genome import Genome


@runtime_checkable
class SearchBackend(Protocol):
    def ask(self, k: int) -> Sequence[Genome]:
        """Propose k candidate genomes (Stage-0-valid by construction)."""
        ...

    def tell(self, genome: Genome, fitness: float, metrics: dict) -> None:
        """Report an evaluated candidate; fitness higher is better."""
        ...
