"""HAWKEYE CROSS-HARDWARE experiment, R6-lite. One evolution run on the
GPU of THIS machine with backend/precision genes active.

TV cell (real Kermany OCT patches, size 128, lam 0.1), 6 train / 3
held-out drawn with data seed 7. Gene menu: single schemes (fbs, drs,
pdhg, condat_vu) + rho + halpern(+restart) + BACKEND genes
{precision: f64 | f32_cert64 | schedule(K, theta), batch}. Mix genomes
(avg/switch) are OUTSIDE the R6-lite menu and are replaced by fallback
draws. Proposals: LLMBackend conditioned on the hardware record (vLLM
Qwen-1.5B), RandomBackend fallback.

Fitness = -SGM10 of wall-clock time-to-CERTIFIED rel duality gap <= 1e-4
over the train instances. The certificate is ALWAYS evaluated at f64:
asserted per solve in this harness (policy path: certificate_dtype ==
"float64"; default path: the whole run is f64 under jax_enable_x64).
Timing convention: solve once to warm compile caches; if it certified,
solve again and score the second wall time (the policy path includes
compile/trace in wall_time, the km path does not; the warm re-run makes
the two comparable). A solve that never certifies scores inf (strict
SGM) and its gap floor is RECORDED (Hawkeye-lane data, not failure).

This is the ONE lane where the solver runs on the GPU: JAX_PLATFORMS is
forced to cuda before any atlas import, preallocation off (LOCAL shares
the 4090 with vLLM, ~9 GB free; 128x128 TV fits easily).

Usage:
  python scripts/hawkeye_r6lite.py --machine LOCAL --seed 42 \
      --llm-url http://localhost:8000/v1
"""
from __future__ import annotations

import os

os.environ["JAX_PLATFORMS"] = "cuda"  # GPU lane: overrides the atlas CPU pin
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import argparse
import dataclasses
import json
import math
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import jax  # noqa: E402

from atlas.bench.metrics import sgm  # noqa: E402
from atlas.bench.kermany import load_patches  # noqa: E402
from atlas.bench.runner import sanitize_json  # noqa: E402
from atlas.evolve.llm_backend import LLMBackend  # noqa: E402
from atlas.evolve.loop import evolve  # noqa: E402
from atlas.evolve.random_backend import RandomBackend  # noqa: E402
from atlas.genome import (  # noqa: E402
    Backend,
    Genome,
    canonical_json,
    genome_from_dict,
)
from atlas.ir.spec import tv_denoise_problem  # noqa: E402
from atlas.schemes.compile import genome_cap_scheme, solve_genome  # noqa: E402
from atlas.typing_ import TypeCheckError  # noqa: E402
from atlas.wrappers.km import Budget  # noqa: E402

V2_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPRINT_DIR = os.path.join(V2_ROOT, "results", "sprint")

MENU = ("fbs", "drs", "pdhg", "condat_vu")
ITER_CAPS = {"fbs": 10_000, "drs": 1_500, "pdhg": 10_000, "condat_vu": 10_000}
SAMPLE_EVERY = 50  # default-path certificate cadence; policy path uses gene K

# The all-f64 default of the TV cell (textbook condat_vu, side condition
# t*L_f/2 + t*s*||D||^2 = 0.9925 < 1), run through the P0 km path.
DEFAULT_GENOME = {
    "scheme": "condat_vu",
    "params": {"tau": 0.5, "sigma": 0.99 * 0.1875, "alpha": None, "rho": None},
    "wrapper": {"halpern": False, "restart": "none", "restart_k": None},
    "mix": None,
    "backend": None,
}


class MenuFilter:
    """SearchBackend wrapper enforcing the R6-lite gene menu: mix genomes
    (avg/switch) are replaced by fallback draws (which have p_mix = 0)."""

    def __init__(self, inner: LLMBackend):
        self.inner = inner
        self.n_menu_replacements = 0

    def ask(self, k: int):
        out = []
        for g in self.inner.ask(k):
            tries = 0
            while g.scheme == "mix" and tries < 50:
                self.n_menu_replacements += 1
                g = self.inner.fallback.ask(1)[0]
                tries += 1
            out.append(g)
        return out

    def tell(self, genome, fitness, metrics):
        self.inner.tell(genome, fitness, metrics)

    def reseed(self, seed: int):
        self.inner.reseed(seed)

    @property
    def stage0_rejections(self):
        return self.inner.stage0_rejections + self.inner.fallback.stage0_rejections

    @property
    def n_fallbacks(self):
        return self.inner.n_fallbacks

    @property
    def n_llm_calls(self):
        return self.inner.n_llm_calls


def assert_cert_f64(result):
    """The certificate is ALWAYS evaluated at f64 (harness assertion)."""
    assert jax.config.jax_enable_x64, "jax_enable_x64 is off"
    be = (result.extra or {}).get("backend") if result.extra else None
    if be is not None:
        assert be["certificate_dtype"] == "float64", (
            f"non-f64 certificate: {be['certificate_dtype']}"
        )
    else:
        # default P0 path: the whole run (iterates + gap) is f64
        assert np.asarray(result.rel_gap_history).dtype == np.float64, (
            "default-path gap history is not float64"
        )


def timed_solve(problem, g: Genome, budget: Budget, stall_log, tag):
    """(time_to_certified_or_inf, record). Warm solve, then timed solve if
    it certified. Gap floors of non-certifying runs are recorded."""
    tol = budget.tol
    r1 = solve_genome(problem, g, budget)
    assert_cert_f64(r1)
    gaps = np.asarray(r1.rel_gap_history, dtype=float)
    final_gap = float(gaps[-1])
    floor = float(np.nanmin(gaps)) if gaps.size else float("inf")
    rec = {
        "iters": int(r1.iters),
        "final_rel_gap": final_gap,
        "gap_floor": floor,
        "wall_cold_s": float(r1.wall_time),
    }
    be = (r1.extra or {}).get("backend") if r1.extra else None
    if be is not None:
        rec["backend"] = {k: be[k] for k in
                          ("precision", "K", "theta", "switch_iter",
                           "iterate_dtypes", "certificate_dtype")}
    if final_gap > tol:
        prec = g.backend.precision if g.backend is not None else "f64(default)"
        stall_log.append({
            "tag": tag,
            "genome": canonical_json(g),
            "precision": prec,
            "iters": int(r1.iters),
            "gap_floor": floor,
            "final_rel_gap": final_gap,
        })
        rec["time_to_certified_s"] = None
        return float("inf"), rec
    r2 = solve_genome(problem, g, budget)
    assert_cert_f64(r2)
    t = float(r2.wall_time)
    rec["time_to_certified_s"] = t
    rec["iters_warm"] = int(r2.iters)
    return t, rec


def eval_genome(g: Genome, instances, tol, stall_log, tag):
    cap = ITER_CAPS.get(genome_cap_scheme(g), 10_000)
    budget = Budget(max_iters=cap, tol=tol, sample_every=SAMPLE_EVERY)
    times, rows, n_fail = [], [], 0
    for ii, prob in enumerate(instances):
        try:
            t, rec = timed_solve(prob, g, budget, stall_log, f"{tag}/inst{ii}")
        except TypeCheckError as e:
            times.append(float("inf"))
            rows.append({"instance": ii, "error": str(e)})
            n_fail += 1
            continue
        rec["instance"] = ii
        rows.append(rec)
        times.append(t)
        if not math.isfinite(t):
            n_fail += 1
    return {
        "genome": canonical_json(g),
        "sgm10_time_s": sgm(times),
        "times_s": times,
        "failure_rate": n_fail / max(len(instances), 1),
        "rows": rows,
    }


def make_fitness(train, tol, stall_log, counter):
    cache: dict = {}

    def fitness_fn(g: Genome):
        key = canonical_json(g)
        if key in cache:
            return cache[key]
        ev = eval_genome(g, train, tol, stall_log, "train")
        s = ev["sgm10_time_s"]
        fit = -s if math.isfinite(s) else float("-inf")
        be = "none" if g.backend is None else (
            f"{g.backend.precision}/K={g.backend.K}"
            + (f"/theta={g.backend.theta:.3g}" if g.backend.theta else "")
        )
        metrics = {
            "sgm10_time_s": round(s, 4) if math.isfinite(s) else float("inf"),
            "failure_rate": ev["failure_rate"],
            "backend": be,
        }
        counter["n"] += 1
        print(f"  [eval {counter['n']:3d}] sgm10={metrics['sgm10_time_s']} "
              f"fail={ev['failure_rate']:.2f} scheme={g.scheme} backend={be}",
              flush=True)
        cache[key] = (fit, metrics)
        return cache[key]

    return fitness_fn


def strip_to_f64(g: Genome) -> Genome:
    """Champion with ONLY the precision gene forced to f64 (same K cadence,
    same everything else): isolates the precision policy's contribution."""
    if g.backend is None:
        return g
    return dataclasses.replace(
        g, backend=Backend(precision="f64", K=g.backend.K, theta=None,
                           batch=g.backend.batch)
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--machine", required=True, choices=["LOCAL", "REMOTE"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--llm-url", default="http://localhost:8000/v1")
    ap.add_argument("--generations", type=int, default=10)
    ap.add_argument("--pop-size", type=int, default=8)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--lam", type=float, default=0.1)
    ap.add_argument("--data-seed", type=int, default=7)
    ap.add_argument("--n-train", type=int, default=6)
    ap.add_argument("--n-holdout", type=int, default=3)
    ap.add_argument("--tol", type=float, default=1e-4)
    args = ap.parse_args()

    t0 = time.time()
    dev = jax.devices()[0]
    assert dev.platform != "cpu", "R6 requires the GPU; got CPU"
    print(f"[hawkeye] device={dev.device_kind} jax={jax.__version__} "
          f"machine={args.machine} seed={args.seed}", flush=True)

    # hardware record: prefer the measured probe already on disk
    hw_path = os.path.join(SPRINT_DIR, f"hardware_{args.machine}.json")
    if os.path.exists(hw_path):
        with open(hw_path) as fh:
            hardware = json.load(fh)
    else:
        from atlas.backend.device import detect_device
        hardware = detect_device(prefer_gpu=True)
    print(f"[hawkeye] hardware={hardware['kind']} "
          f"f32={hardware['tflops_f32']} f64={hardware['tflops_f64']} TFLOPS",
          flush=True)

    patches, meta = load_patches(args.n_train + args.n_holdout, size=args.size,
                                 split="test", seed=args.data_seed,
                                 return_meta=True)
    problems = [tv_denoise_problem(p, lam=args.lam) for p in patches]
    train, holdout = problems[: args.n_train], problems[args.n_train:]
    print(f"[hawkeye] labels={[m['label_name'] for m in meta]}", flush=True)

    stall_log: list = []
    counter = {"n": 0}
    fitness_fn = make_fitness(train, args.tol, stall_log, counter)

    fallback = RandomBackend(seed=args.seed + 1, schemes=MENU,
                             p_mix=0.0, p_switch=0.0, p_backend=0.5)
    lb = LLMBackend(base_url=args.llm_url, seed=args.seed, fallback=fallback,
                    hardware=hardware)
    backend = MenuFilter(lb)

    hist = evolve(backend, fitness_fn, args.generations, args.pop_size,
                  seed=args.seed)
    champ = genome_from_dict(json.loads(hist["best_genome"]))
    print(f"[hawkeye] champion: {canonical_json(champ)}", flush=True)

    # held-out evaluations: champion, champion-forced-f64, all-f64 default
    default_g = genome_from_dict(DEFAULT_GENOME)
    champ_f64 = strip_to_f64(champ)
    ev_champ = eval_genome(champ, holdout, args.tol, stall_log, "holdout/champ")
    ev_champ_f64 = eval_genome(champ_f64, holdout, args.tol, stall_log,
                               "holdout/champ_f64")
    ev_default = eval_genome(default_g, holdout, args.tol, stall_log,
                             "holdout/default")
    # champion on train under the same warm-timing convention (= -fitness)
    ev_champ_train = eval_genome(champ, train, args.tol, stall_log,
                                 "train/champ_final")

    def _speedup(base, new):
        if not math.isfinite(new["sgm10_time_s"]):
            return 0.0
        if not math.isfinite(base["sgm10_time_s"]):
            return float("inf")
        return base["sgm10_time_s"] / new["sgm10_time_s"]

    out = {
        "experiment": "hawkeye_r6lite",
        "machine": args.machine,
        "hardware": hardware,
        "jax_version": jax.__version__,
        "device": dev.device_kind,
        "config": {
            "generations": args.generations, "pop_size": args.pop_size,
            "size": args.size, "lam": args.lam, "tol": args.tol,
            "data_seed": args.data_seed, "n_train": args.n_train,
            "n_holdout": args.n_holdout, "evolve_seed": args.seed,
            "menu": list(MENU), "iter_caps": ITER_CAPS,
            "sample_every": SAMPLE_EVERY, "p_backend": 0.5,
            "llm_url": args.llm_url,
            "llm_model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "timing": "warm re-run wall time to certification; strict SGM10",
        },
        "history": hist,
        "champion": {
            "genome": canonical_json(champ),
            "train_sgm10_s": ev_champ_train["sgm10_time_s"],
            "holdout": ev_champ,
        },
        "champion_f64": {
            "genome": canonical_json(champ_f64),
            "holdout": ev_champ_f64,
            "note": "champion with precision forced to f64, same K cadence",
        },
        "default_all_f64": {
            "genome": canonical_json(default_g),
            "holdout": ev_default,
            "note": "textbook condat_vu t=0.5 s=0.185625, P0 f64 km path",
        },
        "speedups_holdout": {
            "champion_vs_default_f64": _speedup(ev_default, ev_champ),
            "champion_vs_champion_f64": _speedup(ev_champ_f64, ev_champ),
        },
        "stall_log": stall_log,
        "counters": {
            "n_unique_evaluated": counter["n"],
            "stage0_rejections": backend.stage0_rejections,
            "n_llm_calls": backend.n_llm_calls,
            "n_llm_fallbacks": backend.n_fallbacks,
            "n_menu_replacements": backend.n_menu_replacements,
        },
        "wall_time_s": time.time() - t0,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    os.makedirs(SPRINT_DIR, exist_ok=True)
    dest = os.path.join(SPRINT_DIR,
                        f"hawkeye_r6lite_{args.machine}_seed{args.seed}.json")
    with open(dest, "w") as fh:
        json.dump(sanitize_json(out), fh, indent=2, allow_nan=False)
    print(f"[hawkeye] wrote {dest} ({out['wall_time_s']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
