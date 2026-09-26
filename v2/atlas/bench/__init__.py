"""Benchmark / data wing for Optimization Atlas v2 (P0 slice).

Real-data benchmark cells built on the P0 core: Kermany2017 OCT patch
loader (`kermany`), the TV-denoising cell on real clinical speckle
(`tv_cell`), time-to-tol / SGM-10 metrics (`metrics`), and the cell
runner emitting benchmark cards (`runner`).

All entrypoints pin JAX to CPU + float64 (the GPU belongs to the vLLM
server; contention silently hangs evaluations).
"""
from atlas.bench.kermany import load_patches
from atlas.bench.metrics import sgm, speedup_vs_baseline, time_to_tol
from atlas.bench.runner import run_cell, save_card
from atlas.bench.tv_cell import build_instances, oracle

__all__ = [
    "load_patches",
    "build_instances",
    "oracle",
    "time_to_tol",
    "sgm",
    "speedup_vs_baseline",
    "run_cell",
    "save_card",
]
