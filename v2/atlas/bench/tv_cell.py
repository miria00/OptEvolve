"""TV-denoising benchmark cell on REAL Kermany OCT patches.

Instances: min_x (1/2)||x - y||^2 + lam ||D x||_1 where y IS the raw
clinical B-scan patch (real speckle; no synthetic noise is added).

Oracle: high-accuracy reference solve via the core PDHG at float64 with
a tight budget (target rel duality gap <= 1e-9; the rigorous TV gap of
atlas.certify.residuals certifies it). P* and x* are cached to
V2/data/oracle_cache/ keyed by a content hash of (y, lam).
"""
from __future__ import annotations

import hashlib
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

from atlas.bench.kermany import load_patches
from atlas.ir.spec import CompositeProblem, tv_denoise_problem
from atlas.schemes.base_schemes import PDHGParams, solve_pdhg
from atlas.wrappers.km import Budget

ORACLE_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data",
    "oracle_cache",
)

# PDHG steps for the oracle solve: t*s*||D||^2 = 0.99 < 1 (typechecked
# again inside solve_pdhg). Empirically ~10k iters / ~4 s to rel_gap
# 1e-9 on a 128x128 OCT patch (CPU, f64).
_ORACLE_T = 0.5
_ORACLE_S = 0.99 / 4.0


def build_instances(n: int, size: int, lam: float = 0.1, seed: int = 0):
    """n TV-denoising CompositeProblems on real OCT patches.

    The patch itself is y — we denoise actual clinical speckle. Patch
    selection/cropping is deterministic in `seed` (kermany.load_patches).
    """
    patches = load_patches(n, size=size, split="test", seed=seed)
    return [tv_denoise_problem(p, lam=lam) for p in patches]


def _content_key(problem: CompositeProblem, tol: float) -> str:
    """Content hash of the instance (y bytes, shape, lam) + oracle tol."""
    y = np.asarray(problem.data["y"], dtype=np.float64)
    h = hashlib.sha256()
    h.update(y.tobytes())
    h.update(repr(tuple(y.shape)).encode())
    h.update(repr(float(problem.data["lam"])).encode())
    h.update(f"tol={tol:.3e}".encode())
    return h.hexdigest()[:32]


def oracle(
    problem: CompositeProblem,
    tol: float = 1e-9,
    max_iters: int = 5_000_000,
    cache_dir: str = ORACLE_CACHE_DIR,
) -> dict:
    """High-accuracy reference solution of a TV instance.

    Runs the core PDHG (f64, CPU) until the RIGOROUS relative duality
    gap <= tol (default 1e-9). Returns {"P_star", "x_star", "rel_gap",
    "iters", "wall_time_s", "cache_hit", "key"}. Results are cached as
    .npz under `cache_dir`, keyed by a content hash of (y, lam, tol).
    """
    if not problem.is_tv_denoise:
        raise NotImplementedError("oracle() supports the canonical TV-denoise cell")
    key = _content_key(problem, tol)
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"tv_{key}.npz")
    if os.path.exists(path):
        z = np.load(path)
        return {
            "P_star": float(z["P_star"]),
            "x_star": np.asarray(z["x_star"]),
            "rel_gap": float(z["rel_gap"]),
            "iters": int(z["iters"]),
            "wall_time_s": float(z["wall_time_s"]),
            "cache_hit": True,
            "key": key,
        }

    result = solve_pdhg(
        problem,
        PDHGParams(t=_ORACLE_T, s=_ORACLE_S),
        Budget(max_iters=max_iters, tol=tol, sample_every=1000),
    )
    rel_gap = float(result.rel_gap_history[-1])
    if not rel_gap <= tol:
        raise RuntimeError(
            f"oracle failed to reach rel_gap <= {tol:g} "
            f"(achieved {rel_gap:g} in {result.iters} iters)"
        )
    y = np.asarray(problem.data["y"])
    lam = float(problem.data["lam"])
    x_star = np.asarray(result.x)
    P_star = _tv_primal_np(x_star, y, lam)
    np.savez_compressed(
        path,
        P_star=P_star,
        x_star=x_star,
        rel_gap=rel_gap,
        iters=result.iters,
        wall_time_s=result.wall_time,
    )
    return {
        "P_star": P_star,
        "x_star": x_star,
        "rel_gap": rel_gap,
        "iters": int(result.iters),
        "wall_time_s": float(result.wall_time),
        "cache_hit": False,
        "key": key,
    }


def _tv_primal_np(x: np.ndarray, y: np.ndarray, lam: float) -> float:
    """P(x) = (1/2)||x-y||^2 + lam||Dx||_1 in pure numpy (forward diff)."""
    d0 = x[1:, :] - x[:-1, :]
    d1 = x[:, 1:] - x[:, :-1]
    return float(0.5 * np.sum((x - y) ** 2) + lam * (np.abs(d0).sum() + np.abs(d1).sum()))
