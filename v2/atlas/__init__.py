"""Optimization Atlas v2 - typed operator-splitting solvers in JAX (P0 core).

NOTE: all JAX code in this project runs on CPU with float64. The package
enforces both HERE, before any submodule imports jax: JAX_PLATFORMS=cpu
(the GPU is occupied by a vLLM server; GPU contention silently hangs
evaluations) and jax_enable_x64 (otherwise every jnp.float64 request would
silently downcast to float32 and the 1e-9 oracle tolerance is unreachable).
Entrypoints should still set both defensively in case they touch jax before
importing atlas.
"""
import os as _os

_os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax as _jax

_jax.config.update("jax_enable_x64", True)

# Persistent XLA compilation cache (speed lever): kernels survive process
# restarts under ~/.cache/atlas_jax (override via ATLAS_JAX_CACHE_DIR;
# set it to "" to disable). Best-effort: failures never break imports.
try:
    _cache_dir = _os.environ.get(
        "ATLAS_JAX_CACHE_DIR", _os.path.expanduser("~/.cache/atlas_jax")
    )
    if _cache_dir:
        _os.makedirs(_cache_dir, exist_ok=True)
        _jax.config.update("jax_compilation_cache_dir", _cache_dir)
        _jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.1)
except Exception:  # pragma: no cover - cache is an optimization only
    pass

__version__ = "0.3.0"

from atlas.ir.spec import CompositeProblem, lasso_box_problem, tv_denoise_problem
from atlas.operators.base import Atom, PropSet
from atlas.operators.linear import Grad2D, LinOp
from atlas.schemes.base_schemes import (
    CondatVuParams,
    DRSParams,
    DYSParams,
    FBSParams,
    PDHGParams,
    SolveResult,
    solve_condat_vu,
    solve_drs,
    solve_dys,
    solve_fbs,
    solve_pdhg,
    typecheck_condat_vu_solve,
    typecheck_drs_solve,
    typecheck_dys_solve,
    typecheck_fbs_solve,
    typecheck_pdhg_solve,
)
from atlas.schemes.compile import solve_genome
from atlas.typing_ import ConstructionCert, TypeCheckError
from atlas.wrappers.km import Budget

__all__ = [
    "__version__",
    "CompositeProblem",
    "tv_denoise_problem",
    "lasso_box_problem",
    "Atom",
    "PropSet",
    "Grad2D",
    "LinOp",
    "Budget",
    "SolveResult",
    "FBSParams",
    "DRSParams",
    "PDHGParams",
    "CondatVuParams",
    "DYSParams",
    "solve_fbs",
    "solve_drs",
    "solve_pdhg",
    "solve_condat_vu",
    "solve_dys",
    "solve_genome",
    "typecheck_fbs_solve",
    "typecheck_drs_solve",
    "typecheck_pdhg_solve",
    "typecheck_condat_vu_solve",
    "typecheck_dys_solve",
    "ConstructionCert",
    "TypeCheckError",
]
