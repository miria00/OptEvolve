"""Evolution wing: SearchBackend seam, random/LLM backends, ask-tell loop.

A SkyDiscover adapter (plan SS3.1) implements SearchBackend in P2; the
vendored-OpenEvolve adapter is the same shape. See backend.py.
"""
from atlas.evolve.backend import SearchBackend
from atlas.evolve.llm_backend import LLMBackend
from atlas.evolve.loop import evolve
from atlas.evolve.parallel import make_pool
from atlas.evolve.random_backend import RandomBackend

__all__ = ["SearchBackend", "RandomBackend", "LLMBackend", "evolve", "make_pool"]
