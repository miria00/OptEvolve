"""Versioned hardware pilot: profiles, cadence/factorial, and proposer controls.

Runs only the declared diagnostic data. The frozen LP final holdout is never
loaded. No existing experiment file is overwritten. Examples in
docs/HARDWARE_WORKFLOW.md. Set JAX_PLATFORMS before invoking this script.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
import jax
from atlas.backend.hardware import operator_profile, prepare_base
from atlas.evolve.llm_backend import LLMBackend
from atlas.genome import (Genome, GenomeParams, Wrapper, Backend, Metric,
                          content_hash, genome_to_dict, genome_from_dict, validate)
from atlas.ir.spec import tv_denoise_problem
from atlas.schemes.base_schemes import run_built
from atlas.wrappers.km import Budget

KS = (1, 10, 50, 200)


def write_json(path, data):
    def clean(x):
        if isinstance(x, dict):
            return {str(k): clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [clean(v) for v in x]
        if isinstance(x, np.ndarray):
            return clean(x.tolist())
        if isinstance(x, np.generic):
            return clean(x.item())
        if isinstance(x, float) and not np.isfinite(x):
            return str(x)
        return x
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(clean(data), indent=2, allow_nan=False))
    tmp.replace(path)


def source_hashes():
    files = (list((ROOT / "atlas").rglob("*.py")) +
             list((ROOT / "atlas").rglob("*.cu")) + [Path(__file__)])
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(files)}


def load_data(args):
    n = args.n_train + args.n_holdout
    if args.cell in ("oct", "toy_tv"):
        if args.cell == "oct":
            from atlas.bench.kermany import load_patches
            patches, meta = load_patches(n, size=args.size, seed=args.data_seed,
                                         split="test", return_meta=True)
        else:
            rng = np.random.default_rng(args.data_seed)
            patches = rng.random((n, args.size, args.size))
            meta = [{"index": i, "synthetic": True} for i in range(n)]
        problems = [tv_denoise_problem(y, .1) for y in patches]
        for p, m in zip(problems, meta):
            m["input_sha256"] = hashlib.sha256(np.asarray(p.f.y).tobytes()).hexdigest()
        return problems, meta
    from atlas.bench.mps_cell import load_standard_form, to_composite
    split_path = ROOT / "results/sprint/phase3/cell_split.json"
    split = json.loads(split_path.read_text())
    rows = split["calibration"]["instances"]
    if args.lp_names:
        names = args.lp_names.split(",")
        by_name = {r["name"]: r for r in rows}
        rows = [by_name[name] for name in names]  # calibration membership enforced
    else:
        rows = sorted(rows, key=lambda r: (r["nnz"], r["name"]))
    if len(rows) < n:
        raise ValueError("not enough calibration instances for diagnostic split")
    rows = rows[:n]
    problems, meta = [], []
    for row in rows:
        path = ROOT / "data/lp_suites" / row["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError(f"frozen input hash mismatch: {path}")
        problems.append(to_composite(load_standard_form(str(path))))
        meta.append({**row, "source_partition": "frozen_calibration_only"})
    return problems, meta


def algorithms(cell):
    if cell in ("oct", "toy_tv"):
        return {
            "textbook": Genome("condat_vu", GenomeParams(tau=.5, sigma=.186)),
            "banked_policy": Genome("condat_vu", GenomeParams(tau=.05, sigma=1., rho=1.9)),
        }
    metric = Metric(kind="ruiz_pc", iters=10, alpha=1.)
    return {
        "frozen_B0": Genome("pdhg", GenomeParams(tau=1.98, sigma=.495), metric=metric),
        "expert_relaxation": Genome("pdhg", GenomeParams(tau=1.98, sigma=.495, rho=1.5), metric=metric),
        "expert_anchor_restart": Genome("pdhg", GenomeParams(tau=1.98, sigma=.495, rho=1.5),
             Wrapper(halpern=True, restart="fixed_k", restart_k=500), metric=metric),
    }


def policies():
    return {"legacy": None, **{f"f64_K{k}": Backend(K=k) for k in KS},
            "f32_K50": Backend(precision="f32_cert64", K=50),
            "f32_K200": Backend(precision="f32_cert64", K=200),
            "schedule_K50": Backend(precision="schedule", K=50, theta=.01),
            "schedule_K200": Backend(precision="schedule", K=200, theta=.01)}


def policy_catalog(generated=False):
    menu = policies()
    if generated:
        from atlas.backend.cuda_lp import VARIANTS
        menu.update({f"{variant}_K{k}": Backend(K=k, kernel=variant)
                     for variant in VARIANTS for k in (50, 200)})
    return menu


def independent_gap(p, result):
    """NumPy/SciPy check of returned ORIGINAL coordinates, outside fitness."""
    x, u = np.asarray(result.x, dtype=np.float64), np.asarray(result.u, dtype=np.float64)
    if p.name == "lp_standard":
        A, c, b = p.data["A_csr"], np.asarray(p.data["c"]), np.asarray(p.data["b"])
        xp = np.maximum(x, 0)
        primal, dual = float(c @ xp), float(-b @ u)
        return max(np.linalg.norm(A @ xp - b) / (1 + np.linalg.norm(b)),
                   np.linalg.norm(np.maximum(-(c + A.T @ u), 0)) / (1 + np.linalg.norm(c)),
                   abs(primal - dual) / (1 + abs(primal) + abs(dual)))
    y = np.asarray(p.f.y)
    lam = float(p.h.lam)
    primal = .5 * np.sum((x - y) ** 2) + lam * (
        np.sum(np.abs(np.diff(x, axis=0))) + np.sum(np.abs(np.diff(x, axis=1))))
    clipped = np.clip(u, -lam, lam)
    dt = np.zeros_like(x)
    dt[:-1, :] -= clipped[0, :-1, :]
    dt[1:, :] += clipped[0, :-1, :]
    dt[:, :-1] -= clipped[1, :, :-1]
    dt[:, 1:] += clipped[1, :, :-1]
    dual = np.sum(y * dt) - .5 * np.sum(dt ** 2)
    return (primal - dual) / (abs(primal) + 1)


class Evaluator:
    def __init__(self, args, problems):
        self.args, self.problems = args, problems
        self.prepared = {}

    def evaluate(self, genome, indices, tol, repeats=None):
        repeats = self.args.repeats if repeats is None else repeats
        rows = []
        for i in indices:
            p = self.problems[i]
            key = (i, content_hash(replace(genome, backend=None)),
                   getattr(genome.backend, "kernel", "jax"))
            t0 = time.perf_counter()
            if key not in self.prepared:
                self.prepared[key] = prepare_base(p, genome)
            built, cert, params, metadata = self.prepared[key]
            prepare_s = time.perf_counter() - t0
            budget = Budget(max_iters=self.args.max_iters, tol=tol, sample_every=250)
            # One complete warm solve materializes all used precision phases.
            t0 = time.perf_counter()
            cold = run_built(built, cert, params, budget, backend=genome.backend)
            cold_s = time.perf_counter() - t0
            samples, end_to_end, gaps, its = [], [], [], []
            for _ in range(repeats):
                t0 = time.perf_counter()
                result = run_built(built, cert, params, budget, backend=genome.backend)
                end_to_end.append(time.perf_counter() - t0)
                samples.append(result.wall_time)
                gaps.append(float(independent_gap(p, result)))
                its.append(result.iters)
                reported = float(result.rel_gap_history[-1])
                if not np.isclose(gaps[-1], reported, atol=1e-10, rtol=1e-7):
                    raise AssertionError(f"independent certificate mismatch: {gaps[-1]} vs {reported}")
            ok = all(np.isfinite(g) and 0 <= g <= tol for g in gaps)
            backend = (result.extra or {}).get("backend", {})
            rows.append({"instance_index": i, "certified": ok,
                "solver_s": float(np.median(samples)), "solver_samples_s": samples,
                "warm_call_s": float(np.median(end_to_end)),
                "prepare_s": prepare_s, "cold_call_s": cold_s,
                "iters": its, "independent_gaps": gaps,
                "checks": backend.get("certificate_checks", result.iters),
                "backend": backend, "metric": metadata,
                "construction_satisfied": bool(cert and cert.satisfied)})
        solved = sum(r["certified"] for r in rows)
        # A fixed censoring penalty is part of this diagnostic protocol.
        times = [r["solver_s"] if r["certified"] else self.args.penalty_s for r in rows]
        aggregate = float(np.expm1(np.mean(np.log1p(times))))
        return {"n_solved": solved, "n": len(rows), "penalized_sgm1_s": aggregate,
                "rows": rows}


def rank(result):
    return (-result["n_solved"], result["penalized_sgm1_s"])


class TypedMutation:
    """A small population control over the same base-genome representation."""
    def __init__(self, cell, seed, independent=False, generated=False):
        self.cell, self.independent = cell, independent
        self.policy_menu = policy_catalog(generated)
        self.rng = np.random.default_rng(seed)
        self.archive = []
        self.seed_genome = next(iter(algorithms(cell).values()))
        self.stage0_rejections = 0

    def reseed(self, seed):
        self.rng = np.random.default_rng(seed)

    def tell(self, genome, fitness, metrics=None):
        self.archive.append((fitness, genome))
        self.archive.sort(key=lambda v: v[0], reverse=True)
        self.archive = self.archive[:4]

    def ask(self, k=1):
        return [self._ask_one() for _ in range(k)]

    def _ask_one(self):
        for _ in range(200):
            parent = (self.seed_genome if self.independent or not self.archive else
                      self.archive[int(self.rng.integers(len(self.archive)))][1])
            d = genome_to_dict(parent)
            moves = ("steps", "wrapper", "backend") if self.independent else (
                str(self.rng.choice(("steps", "wrapper", "backend"))),)
            if "steps" in moves:
                if self.cell == "lp":
                    eta = float(self.rng.choice((.7, .9, .99)))
                    omega = float(self.rng.choice((.25, .5, 1., 2.)))
                    d["params"]["tau"], d["params"]["sigma"] = eta / omega, eta * omega
                else:
                    d["scheme"] = str(self.rng.choice(("pdhg", "condat_vu")))
                    tau = float(self.rng.choice((.025, .05, .1, .3, .5)))
                    d["params"]["tau"] = tau
                    d["params"]["sigma"] = float(self.rng.choice((.4, .7, .95))) / (8 * tau)
            if "wrapper" in moves:
                h = bool(self.rng.integers(2))
                d["params"]["rho"] = float(self.rng.choice((1., 1.5, 1.9)))
                d["wrapper"] = {"halpern": h, "restart": "fixed_k" if h else "none",
                                "restart_k": int(self.rng.choice((50, 200, 500))) if h else None}
            if "backend" in moves:
                b = self.policy_menu[str(self.rng.choice(list(self.policy_menu)[1:]))]
                d["backend"] = {"precision": b.precision, "K": b.K,
                                "theta": b.theta, "kernel": b.kernel}
            g = genome_from_dict(d)
            if validate(g, problem_kind="lp_netflow" if self.cell == "lp" else "tv_denoise").ok:
                return g
            self.stage0_rejections += 1
        raise RuntimeError("typed mutation could not produce an admissible genome")


def pilot(args, evaluator, profile, tol, train, holdout, output):
    runs = []
    for arm_index, arm in enumerate(args.arms.split(",")):
        for replicate in range(args.n_seeds):
            seed = args.search_seed + replicate
            proposer_seed = seed + 100003 * (arm_index + 1)
            fallback_seed = seed + 900001 + 100003 * arm_index
            fallback = TypedMutation(args.cell, fallback_seed, generated=args.generated_kernels)
            if arm.startswith("llm_"):
                backend = LLMBackend(seed=proposer_seed, fallback=fallback,
                    hardware=profile, hardware_context=arm[4:],
                    base_url=args.llm_url, max_tokens=600,
                    problem_kind="lp_netflow" if args.cell == "lp" else "tv_denoise")
            else:
                backend = TypedMutation(args.cell, proposer_seed, independent=arm == "random",
                                        generated=args.generated_kernels)
            baseline = next(iter(algorithms(args.cell).values()))
            base_result = evaluator.evaluate(baseline, train, tol, repeats=1)
            def fitness(r):
                return r["n_solved"] * (args.penalty_s + 1) - r["penalized_sgm1_s"]
            backend.tell(baseline, fitness(base_result), {"n_solved": base_result["n_solved"]})
            best, best_result = baseline, base_result
            seen, candidates = {content_hash(baseline)}, []
            proposal_count, duplicates, rejected = 0, 0, 0
            start = time.perf_counter()
            while len(candidates) < args.pilot_evals and proposal_count < args.pilot_evals * 30:
                proposal_count += 1
                prior_fallbacks = getattr(backend, "n_fallbacks", 0)
                g = backend.ask(1)[0]
                # The executable pilot deliberately restricts all arms to the
                # same two saddle schemes and frozen LP substrate.
                valid = (g.scheme in ("pdhg", "condat_vu") and
                         not (g.backend and g.backend.batch) and
                         (g.backend is None or g.backend in policy_catalog(args.generated_kernels).values()) and
                         g.metric == baseline.metric)
                if not valid:
                    rejected += 1
                    continue
                h = content_hash(g)
                if h in seen:
                    duplicates += 1
                    continue
                seen.add(h)
                r = evaluator.evaluate(g, train, tol, repeats=1)
                backend.tell(g, fitness(r), {"n_solved": r["n_solved"],
                                            "solver_s": r["penalized_sgm1_s"]})
                if rank(r) < rank(best_result):
                    best, best_result = g, r
                candidates.append({"genome": genome_to_dict(g), "hash": h,
                    "source": ("fallback" if getattr(backend, "n_fallbacks", 0) > prior_fallbacks
                               else "llm" if arm.startswith("llm_") else arm),
                    "train": r, "best_hash": content_hash(best),
                    "transcript": getattr(backend, "last_transcript", [])})
                print(f"PILOT {arm} seed={seed} fresh={len(candidates)} best={rank(best_result)}", flush=True)
            run = {"arm": arm, "seed": seed, "proposer_seed": proposer_seed,
                "fallback_seed": fallback_seed, "fresh_evaluations": len(candidates),
                "proposal_count": proposal_count, "duplicates": duplicates,
                "scope_rejections": rejected, "search_s": time.perf_counter() - start,
                "llm_calls": getattr(backend, "n_llm_calls", 0),
                "fallbacks": getattr(backend, "n_fallbacks", 0),
                "token_usage": getattr(backend, "token_usage", {}),
                "budget_complete": len(candidates) == args.pilot_evals,
                "champion": genome_to_dict(best), "train": best_result,
                "holdout": evaluator.evaluate(best, holdout, tol),
                "candidates": candidates,
                "transcripts": getattr(backend, "last_transcript", [])}
            runs.append(run)
            write_json(output, {"status": "running", "runs": runs})
    write_json(output, {"status": "complete", "runs": runs})
    return runs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cell", choices=("oct", "toy_tv", "lp"), default="oct")
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--n-train", type=int, default=3)
    ap.add_argument("--n-holdout", type=int, default=3)
    ap.add_argument("--lp-names", default="")
    ap.add_argument("--max-iters", type=int, default=20000)
    ap.add_argument("--tolerances", default="0.0001,0.000001")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--penalty-s", type=float, default=120.)
    ap.add_argument("--data-seed", type=int, default=20260906)
    ap.add_argument("--search-seed", type=int, default=42)
    ap.add_argument("--pilot-evals", type=int, default=0)
    ap.add_argument("--n-seeds", type=int, default=2)
    ap.add_argument("--arms", default="random,typed,llm_none,llm_identity,llm_measured")
    ap.add_argument("--llm-url", default="http://localhost:8000/v1")
    ap.add_argument("--profile-only", action="store_true")
    ap.add_argument("--pilot-only", action="store_true")
    ap.add_argument("--trace", action="store_true")
    ap.add_argument("--algorithm-filter", default="",
                    help="comma-separated catalog names, chosen before the run")
    ap.add_argument("--policy-filter", default="",
                    help="comma-separated policy names, chosen before the run")
    ap.add_argument("--generated-kernels", action="store_true",
                    help="include bounded CUDA LP update templates (f64 only)")
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(f"new output directory required: {out}")
    out.mkdir(parents=True)
    write_json(out / "protocol.json", {
        "version": "hardware-pilot-v1", "args": vars(args),
        "source_sha256": source_hashes(), "host": platform.node(),
        "device": jax.devices()[0].device_kind, "jax": jax.__version__,
        "status": "exploratory diagnostic, not the final paper holdout",
        "timing": "warm synchronized solver execution; compilation and preparation separate",
        "censoring": "iteration cap, fixed penalty for non-solves; not a wall-time stop",
        "selection": "train only; fixed candidate menu; all chosen comparisons also reported",
    })
    problems, meta = load_data(args)
    train = list(range(args.n_train))
    holdout = list(range(args.n_train, len(problems)))
    write_json(out / "data.json", {"instances": meta, "train": train, "holdout": holdout})
    algs = algorithms(args.cell)
    if args.algorithm_filter:
        algs = {k: algs[k] for k in args.algorithm_filter.split(",")}
    policy_menu = policy_catalog(args.generated_kernels)
    if args.generated_kernels:
        if args.cell != "lp":
            raise ValueError("generated kernels currently support the LP cell only")
    if args.policy_filter:
        policy_menu = {k: policy_menu[k] for k in args.policy_filter.split(",")}
    profile = operator_profile(problems[0], next(iter(algs.values())), hlo_dir=out / "hlo")
    write_json(out / "profile.json", profile)
    print("PROFILE", jax.devices()[0], flush=True)
    if args.profile_only:
        return
    evaluator = Evaluator(args, problems)
    results = {"status": "running", "tolerances": {}}
    for tol in map(float, args.tolerances.split(",")):
        if not args.pilot_only:
            rows = []
            pairs = [(a, b) for a in algs for b in policy_menu]
            np.random.default_rng(args.search_seed).shuffle(pairs)
            for aname, bname in pairs:
                g = replace(algs[aname], backend=policy_menu[bname])
                row = {"algorithm": aname, "policy": bname,
                    "genome": genome_to_dict(g),
                    "train": evaluator.evaluate(g, train, tol),
                    "holdout": evaluator.evaluate(g, holdout, tol)}
                rows.append(row)
                print(f"BENCH tol={tol:g} {aname}/{bname} train={rank(row['train'])} holdout={rank(row['holdout'])}", flush=True)
                results["tolerances"][str(tol)] = {"rows": rows}
                write_json(out / "benchmark.json", results)
            chosen = min(rows, key=lambda r: rank(r["train"]))
            baseline_name = next(iter(algs))
            backend_only = min((r for r in rows if r["algorithm"] == baseline_name),
                               key=lambda r: rank(r["train"]))
            reference_policy = "f64_K1" if "f64_K1" in policy_menu else next(iter(policy_menu))
            algorithm_only = min((r for r in rows if r["policy"] == reference_policy),
                                 key=lambda r: rank(r["train"]))
            sequential = min((r for r in rows if r["algorithm"] == algorithm_only["algorithm"]),
                             key=lambda r: rank(r["train"]))
            reverse_sequential = min((r for r in rows if r["policy"] == backend_only["policy"]),
                                     key=lambda r: rank(r["train"]))
            results["tolerances"][str(tol)].update(
                joint_selected=chosen, backend_only_selected=backend_only,
                algorithm_only_selected=algorithm_only, sequential_selected=sequential,
                reverse_sequential_selected=reverse_sequential,
                selection_status=("uninformative_all_non_solves" if chosen["train"]["n_solved"] == 0
                                  else "selected_on_training_only"))
            write_json(out / "benchmark.json", results)
        if args.pilot_evals:
            pilot(args, evaluator, profile, tol, train, holdout, out / f"pilot_{tol:g}.json")
    if args.trace:
        trace_dir = out / "trace"
        with jax.profiler.trace(str(trace_dir)):
            evaluator.evaluate(replace(next(iter(algs.values())), backend=Backend(K=50)),
                               train[:1], float(args.tolerances.split(",")[0]), repeats=2)
    results["status"] = "complete"
    write_json(out / "benchmark.json", results)
    print("COMPLETE", out, flush=True)


if __name__ == "__main__":
    main()
