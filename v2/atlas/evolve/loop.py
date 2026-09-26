"""The ask/evaluate/tell evolution loop with elitism keep-1 and exact dedup.

evolve(backend, fitness_fn, generations, pop_size, seed) drives any
SearchBackend. Every proposed genome passes the Stage-0 gate again at the
loop level (defense in depth: invalid candidates are rejected and counted,
never evaluated, never clamped). Duplicate genomes (canonical content
hash) reuse their cached fitness instead of re-running the evaluator.

Elitism keep-1: the best (genome, fitness) pair survives across
generations unconditionally - per-generation best_fitness is monotone
non-decreasing - and is re-told to the backend each generation so
context-building backends keep it in their prompt archive.

fitness_fn(genome) -> float (higher is better) or (float, metrics_dict).

PROCESS-PARALLEL EVALUATION (speed lever): pass `pool` (any object with a
.map(fn, iterable) method, e.g. atlas.evolve.parallel.make_pool(n)) and
the unique un-cached candidates of each generation are evaluated with one
pool.map call; fitness_fn and genomes must then be picklable. Results are
bitwise the same as the serial path (same candidates, same dedup, same
tell order).
"""
from __future__ import annotations

import math
import warnings
from typing import Callable, Optional

from atlas.genome import Genome, canonical_json, content_hash, validate
from atlas.evolve.backend import SearchBackend


def _counter(backend, name: str) -> int:
    v = getattr(backend, name, 0)
    return int(v) if isinstance(v, (int, float)) else 0


def evolve(
    backend: SearchBackend,
    fitness_fn: Callable,
    generations: int,
    pop_size: int,
    seed: Optional[int] = None,
    flat_warn_rel: float = 0.01,
    pool=None,
    problem_kind: str = "tv_denoise",
) -> dict:
    """Run the loop; returns a history dict:

    {"generations": [{"gen", "best_fitness", "best_genome", "fitnesses",
                      "n_rejected_stage0", "n_llm_fallbacks",
                      "n_evaluated", "n_dedup_hits"}, ...],
     "best_genome": dict, "best_fitness": float,
     "n_unique_evaluated": int,
     "fitness_spread": float, "flat_fitness_warning": bool}

    Flatness diagnostic (the v1 failure mode was a silently constant
    fitness): per-candidate fitnesses are recorded per generation, and
    flat_fitness_warning is set when >= 2 unique genomes received FINITE
    fitness but their spread is <= flat_warn_rel * max(|fitness|) - i.e.
    selection is operating on noise. A warnings.warn is also emitted so
    interactive runs cannot miss it.
    """
    if seed is not None and hasattr(backend, "reseed"):
        backend.reseed(seed)

    best_genome: Optional[Genome] = None
    best_fitness = -math.inf
    best_metrics: dict = {}
    cache: dict[str, tuple[float, dict]] = {}  # content_hash -> (fitness, metrics)
    history: dict = {"generations": []}

    def _parse(out):
        fit, metrics = out if isinstance(out, tuple) else (float(out), {})
        return float(fit), metrics

    for gen in range(generations):
        rej_before = _counter(backend, "stage0_rejections")
        fb_before = _counter(backend, "n_fallbacks")

        candidates = list(backend.ask(pop_size))
        n_rejected_loop = 0
        n_evaluated = 0
        n_dedup = 0
        gen_fitnesses: list[float] = []

        # Pass 1: gate + dedup; collect the unique un-cached genomes.
        gated: list[tuple[Genome, str] | None] = []
        to_eval: list[Genome] = []
        to_eval_hashes: set[str] = set()
        for g in candidates:
            v = validate(g, problem_kind=problem_kind)  # Stage-0, loop level
            if not v.ok:
                n_rejected_loop += 1
                gated.append(None)
                continue
            h = content_hash(g)
            gated.append((g, h))
            if h not in cache and h not in to_eval_hashes:
                to_eval.append(g)
                to_eval_hashes.add(h)

        # Pass 2: evaluate (serial map or process pool).
        mapper = pool.map if pool is not None else map
        for g, out in zip(to_eval, mapper(fitness_fn, to_eval)):
            cache[content_hash(g)] = _parse(out)
            n_evaluated += 1

        # Pass 3: tell in the original candidate order (identical to the
        # serial semantics; duplicates count as dedup hits).
        seen_this_gen: set[str] = set()
        for item in gated:
            if item is None:
                continue
            g, h = item
            fit, metrics = cache[h]
            if h in seen_this_gen or h not in to_eval_hashes:
                n_dedup += 1
            seen_this_gen.add(h)
            gen_fitnesses.append(fit)
            backend.tell(g, fit, metrics)
            if fit > best_fitness:
                best_fitness, best_genome, best_metrics = fit, g, metrics

        # elitism keep-1: the champion survives and stays in the archive
        if best_genome is not None:
            backend.tell(best_genome, best_fitness, best_metrics)

        history["generations"].append(
            {
                "gen": gen,
                "best_fitness": best_fitness,
                "best_genome": (
                    None if best_genome is None else canonical_json(best_genome)
                ),
                "fitnesses": gen_fitnesses,
                "n_rejected_stage0": n_rejected_loop
                + (_counter(backend, "stage0_rejections") - rej_before),
                "n_llm_fallbacks": _counter(backend, "n_fallbacks") - fb_before,
                "n_evaluated": n_evaluated,
                "n_dedup_hits": n_dedup,
            }
        )

    history["best_genome"] = None if best_genome is None else canonical_json(best_genome)
    history["best_fitness"] = best_fitness
    history["n_unique_evaluated"] = len(cache)

    # Flatness diagnostic over the UNIQUE evaluated genomes (v1 failure
    # mode: a constant fitness silently produced normal-looking curves).
    finite = [f for f, _ in cache.values() if math.isfinite(f)]
    spread = (max(finite) - min(finite)) if finite else 0.0
    scale = max((abs(f) for f in finite), default=0.0)
    # Flag only when >= 2 unique genomes got FINITE fitness yet the spread is
    # negligible (a mostly-infeasible run with one finite survivor is not a
    # flat landscape; failure counts already expose that situation).
    flat = len(finite) >= 2 and spread <= flat_warn_rel * max(scale, 1e-300)
    history["fitness_spread"] = spread
    history["flat_fitness_warning"] = bool(flat)
    if flat:
        warnings.warn(
            f"evolve(): fitness landscape is flat/near-flat - spread "
            f"{spread:.3g} over {len(cache)} unique genomes (<= "
            f"{flat_warn_rel:g} * max|fitness| {scale:.3g}); selection is "
            f"operating on noise and the run's ranking is not meaningful.",
            RuntimeWarning,
            stacklevel=2,
        )
    return history

def evolve_clk(
    backend_factory: Callable[[int], SearchBackend],
    fitness_fn: Callable,
    C: int,
    L: int,
    K: int,
    seed: Optional[int] = None,
    pool=None,
    problem_kind: str = "tv_denoise",
    share_elite: bool = True,
) -> dict:
    """Search on three axes instead of one, after SimpleTES.

    The single-trajectory loop above is (C=1, K=pop_size) with global
    elitism: one chain, one archive, selection over the whole generation.
    SimpleTES parameterizes the same budget as C parallel trajectories, L
    sequential refinements each, and a LOCAL best-of-K at every step, and
    reports that dropping any one axis degrades sharply.  The budget is the
    same either way, C * L * K evaluations, so the split is an ablation and
    not a bigger run.

    C  independent trajectories, each with its own backend and therefore its
       own archive, racing rather than cooperating.
    L  refinement steps per trajectory.
    K  candidates per step, of which the best is the one that shapes that
       trajectory's next context.

    share_elite is a minimal context composition (the Phi axis): the global
    champion is told to every trajectory, so a discovery made in one chain is
    visible to the others without merging their archives.

    The evaluation cache is GLOBAL, so trajectories that rediscover the same
    genome pay for it once; that is a saving, not a shortcut, because the
    fitness is a deterministic function of the genome.
    """
    backends = [backend_factory(c) for c in range(C)]
    if seed is not None:
        for c, b in enumerate(backends):
            if hasattr(b, "reseed"):
                b.reseed(seed + 1009 * c)

    cache: dict[str, tuple[float, dict]] = {}
    best_genome: Optional[Genome] = None
    best_fitness = -math.inf
    best_metrics: dict = {}
    local_best = [(-math.inf, None, {}) for _ in range(C)]
    history: dict = {"steps": [], "C": C, "L": L, "K": K,
                     "budget": C * L * K, "share_elite": share_elite}

    def _parse(out):
        fit, metrics = out if isinstance(out, tuple) else (float(out), {})
        return float(fit), metrics

    for l in range(L):
        step_rec = {"step": l, "per_trajectory": []}
        for c in range(C):
            b = backends[c]
            cands = list(b.ask(K))
            gated: list[tuple[Genome, str]] = []
            to_eval: list[Genome] = []
            seen: set[str] = set()
            n_rej = 0
            for g in cands:
                if not validate(g, problem_kind=problem_kind).ok:
                    n_rej += 1
                    continue
                h = content_hash(g)
                gated.append((g, h))
                if h not in cache and h not in seen:
                    to_eval.append(g)
                    seen.add(h)
            mapper = pool.map if pool is not None else map
            for g, out in zip(to_eval, mapper(fitness_fn, to_eval)):
                cache[content_hash(g)] = _parse(out)
            # local best-of-K: this is the K axis
            step_best = (-math.inf, None, {})
            for g, h in gated:
                fit, metrics = cache[h]
                b.tell(g, fit, metrics)
                if fit > step_best[0]:
                    step_best = (fit, g, metrics)
                if fit > best_fitness:
                    best_fitness, best_genome, best_metrics = fit, g, metrics
            if step_best[1] is not None and step_best[0] > local_best[c][0]:
                local_best[c] = step_best
            # elitism within the trajectory, then the shared champion
            if local_best[c][1] is not None:
                b.tell(local_best[c][1], local_best[c][0], local_best[c][2])
            if share_elite and best_genome is not None:
                b.tell(best_genome, best_fitness, best_metrics)
            step_rec["per_trajectory"].append(
                {"c": c, "n_rejected_stage0": n_rej,
                 "step_best": step_best[0] if step_best[1] is not None else None,
                 "traj_best": local_best[c][0] if local_best[c][1] else None}
            )
        step_rec["best_fitness"] = best_fitness
        history["steps"].append(step_rec)

    history["best_genome"] = None if best_genome is None else canonical_json(best_genome)
    history["best_fitness"] = best_fitness
    history["n_unique_evaluated"] = len(cache)
    history["trajectory_bests"] = [lb[0] if lb[1] is not None else None for lb in local_best]
    return history
