"""Structural headroom assay on the FROZEN calibration cell.

THE QUESTION: starting from a competent static first-order method, does
certified structural machinery move us materially beyond B0's 7/20 at 1e-6?

This is a DIAGNOSTIC, not a redefinition of success. The pre-registered
verdict stands: B0 failed its 60-80% calibration band at every cap. What the
assay decides is whether structural headroom exists on this cell, and
therefore whether to launch evolution here or move structural discovery to an
L_f > 0 cell.

NOTHING IS MUTATED: same cell, same memberships, same Ruiz+PC substrate, same
frozen B0 parameters as the seed point. B0 is NOT retroactively strengthened.

FAMILIES (all under the identical substrate, all sharing one tuning budget):
  G0     static PDHG                       (the frozen seed baseline)
  Grho   + over-relaxation
  GH     + Halpern anchor
  GHR    + Halpern + fixed-k restart
  GrH    + relaxation THEN anchor          (includes the reflection rho=2)
  GrHR   + relaxation, anchor, restart
  Gphase two-phase stepsize switch_K
Condat-Vu is EXCLUDED: on LP L_f = 0 makes it identical to PDHG (measured),
so a scheme switch here is degenerate and would test nothing.

MEASURES, per family and per tolerance:
  - certified solve count and SGM time (lexicographic, censored at the cap)
  - WHICH instances solve, so the UNION across families can be computed: if
    different genes rescue different instances, a search that composes them
    has more headroom than any single family's count reveals
  - basin width W10 = fraction of sampled settings that beat B0 (more solves,
    or equal solves with >=10% lower SGM), i.e. broad basin vs lucky needle
  - the four near-miss instances tracked individually, with gap trajectories
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cuda")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
TARGET_GPU_SUBSTRING = os.environ.get("ATLAS_TARGET_GPU", "5090")

import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from atlas.bench import mps_cell  # noqa: E402
from atlas.genome import genome_from_dict, validate  # noqa: E402
from atlas.schemes.compile import solve_genome  # noqa: E402
from atlas.schemes.rescale import lp_substrate_metric  # noqa: E402
from atlas.typing_ import TypeCheckError  # noqa: E402
from atlas.wrappers.km import Budget  # noqa: E402

SPLIT = os.path.join(ROOT, "results", "sprint", "phase3", "cell_split.json")
OUT = os.path.join(ROOT, "results", "sprint", "phase3", "headroom_assay.json")

# Frozen B0 (calibration artifact; NOT re-tuned here).
B0_ETA, B0_OMEGA, B0_ALPHA = 0.99, 0.5, 1.0
# The four instances B0 censored within reach of the certificate.
NEAR_MISS = ["neos8", "rocII-5-11", "trento1", "unitcal_7"]
SGM_SHIFT = 10.0


def sgm(vals, shift=SGM_SHIFT):
    v = np.asarray([x for x in vals if np.isfinite(x)], dtype=float)
    return float(np.exp(np.mean(np.log(v + shift))) - shift) if v.size else float("inf")


def _mk(eta, omega, rho=None, halpern=False, restart_k=None, phase=None):
    tau, sigma = eta / omega, eta * omega
    m = lp_substrate_metric(pc_alpha=B0_ALPHA)
    if phase is not None:
        k, eta2, om2 = phase
        return {"scheme": "mix", "metric": m,
                "mix": {"kind": "switch_k", "K": int(k),
                        "aggressive": {"scheme": "pdhg", "tau": tau, "sigma": sigma},
                        "tail": {"scheme": "pdhg", "tau": eta2 / om2,
                                 "sigma": eta2 * om2}}}
    d = {"scheme": "pdhg", "params": {"tau": tau, "sigma": sigma}, "metric": m}
    if rho is not None:
        d["params"]["rho"] = rho
    w = {}
    if halpern:
        w["halpern"] = True
    if restart_k is not None:
        w["halpern"] = True
        w["restart"] = "fixed_k"
        w["restart_k"] = int(restart_k)
    if w:
        d["wrapper"] = w
    return d


def families(n_per_family: int, seed: int) -> dict:
    """Pre-registered sampling distributions, one shared budget per family."""
    rng = np.random.default_rng(seed)
    E = [0.7, 0.9, 0.99]                    # step scale
    W = [0.25, 0.5, 1.0, 2.0]               # primal weight
    RHO = [1.2, 1.5, 1.8, 1.95, 2.0]        # 2.0 = reflection (anchor only)
    RK = [100, 500, 2000, 10000]            # fixed-k restart period
    K = [200, 1000, 5000]                   # phase switch point

    def take(cands):
        idx = rng.choice(len(cands), size=min(n_per_family, len(cands)),
                         replace=False)
        return [cands[i] for i in sorted(idx)]

    fam = {}
    fam["G0"] = [_mk(B0_ETA, B0_OMEGA)]  # the frozen seed point, exactly
    fam["Grho"] = take([_mk(e, w, rho=r) for e in E for w in W
                        for r in RHO if r < 2.0])
    fam["GH"] = take([_mk(e, w, halpern=True) for e in E for w in W])
    fam["GHR"] = take([_mk(e, w, restart_k=k) for e in E for w in W for k in RK])
    fam["GrH"] = take([_mk(e, w, rho=r, halpern=True) for e in E for w in W
                       for r in RHO])
    fam["GrHR"] = take([_mk(e, w, rho=r, restart_k=k) for e in E for w in W
                        for r in RHO for k in RK])
    fam["Gphase"] = take([_mk(e, w, phase=(k, e2, w2)) for e in E for w in W
                          for k in K for e2 in E for w2 in W
                          if (e2, w2) != (e, w)])
    return fam


def evaluate(probs, d, tol, max_iters, cap_s, want_traj=()):
    g = genome_from_dict(d)
    v = validate(g, problem_kind="lp_netflow")
    if not v.ok:
        return {"rejected": v.error[:160]}
    rows, traj = {}, {}
    for name, prob in probs.items():
        t0 = time.time()
        try:
            r = solve_genome(prob, g, Budget(max_iters=max_iters, tol=tol,
                                             sample_every=250))
            el = time.time() - t0
            h = r.rel_gap_history[np.isfinite(r.rel_gap_history)]
            gap = float(h[-1]) if h.size else float("nan")
            rows[name] = {"ok": bool(gap <= tol) and el <= cap_s,
                          "iters": int(r.iters), "wall": el, "gap": gap}
            if name in want_traj:
                traj[name] = [float(x) for x in h[::max(1, len(h) // 60)]]
        except TypeCheckError as ex:
            rows[name] = {"ok": False, "rejected": str(ex)[:120]}
    solved = sorted(n for n, r in rows.items() if r.get("ok"))
    times = [rows[n]["wall"] if rows[n].get("ok") else cap_s for n in rows]
    return {"n_solved": len(solved), "solved": solved, "sgm_s": sgm(times),
            "rows": rows, "traj": traj}


def beats(cand, base, margin=0.10):
    """Lexicographic improvement over B0: more certified solves, or equal
    solves with at least `margin` lower SGM time."""
    if cand.get("rejected") or base.get("rejected"):
        return False
    if cand["n_solved"] > base["n_solved"]:
        return True
    return (cand["n_solved"] == base["n_solved"]
            and cand["sgm_s"] <= (1.0 - margin) * base["sgm_s"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-family", type=int, default=10)
    ap.add_argument("--screen-max-iters", type=int, default=20_000)
    ap.add_argument("--full-max-iters", type=int, default=200_000)
    ap.add_argument("--cap-s", type=float, default=120.0)
    ap.add_argument("--seed", type=int, default=20260905)
    args = ap.parse_args()

    import subprocess

    import jax
    import jaxlib
    gpu = "unknown"
    try:
        gpu = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True).strip().splitlines()[0]
    except Exception:  # noqa: BLE001
        pass
    env = {"devices": [str(d) for d in jax.devices()], "gpu_name": gpu,
           "jax": jax.__version__, "jaxlib": jaxlib.__version__,
           "iterate_precision": "float64", "certificate_precision": "float64",
           "host": os.uname().nodename}
    print("ENVIRONMENT", flush=True)
    for k, v in env.items():
        print(f"  {k:24s} {v}", flush=True)
    if TARGET_GPU_SUBSTRING and TARGET_GPU_SUBSTRING not in gpu:
        sys.exit(f"FATAL: declared target {TARGET_GPU_SUBSTRING!r} absent "
                 f"(found {gpu!r})")

    split = json.load(open(SPLIT))
    rows = split["calibration"]["instances"]
    probs = {}
    for e in rows:
        try:
            std = mps_cell.load_standard_form(
                os.path.join(ROOT, "data", "lp_suites", e["file"]))
            probs[e["name"]] = mps_cell.to_composite(std)
        except Exception as ex:  # noqa: BLE001
            print(f"  ! load {e['name']}: {type(ex).__name__}", flush=True)
    print(f"calibration n={len(probs)} "
          f"sha={split['calibration']['sha256'][:16]}", flush=True)
    warm = _mk(B0_ETA, B0_OMEGA)
    for _p in probs.values():
        try:
            solve_genome(probs and _p, genome_from_dict(warm),
                         Budget(max_iters=2, tol=0.0, sample_every=1))
        except Exception:  # noqa: BLE001
            pass

    fam = families(args.n_per_family, args.seed)
    result = {"environment": env, "config": vars(args),
              "B0": {"eta": B0_ETA, "omega": B0_OMEGA, "alpha": B0_ALPHA},
              "near_miss": NEAR_MISS, "tolerances": {}}

    for tol in (1e-4, 1e-6):
        print(f"\n{'='*66}\nTOLERANCE {tol:g}\n{'='*66}", flush=True)
        base = evaluate(probs, fam["G0"][0], tol, args.full_max_iters,
                        args.cap_s, want_traj=NEAR_MISS)
        print(f"B0 (seed baseline): {base['n_solved']}/{len(probs)} "
              f"sgm={base['sgm_s']:.2f}s", flush=True)
        per_family = {"G0": {"base": base}}
        for fname, cands in fam.items():
            if fname == "G0":
                continue
            print(f"\n-- {fname}: screening {len(cands)} settings", flush=True)
            scr = []
            for d in cands:
                r = evaluate(probs, d, tol, args.screen_max_iters, args.cap_s)
                scr.append({"genome": d, **{k: v for k, v in r.items()
                                            if k != "rows" and k != "traj"}})
                tag = ("REJECT" if r.get("rejected")
                       else f"{r['n_solved']:>2}/{len(probs)} sgm={r['sgm_s']:7.2f}s")
                print(f"   {tag}", flush=True)
            ok = [s for s in scr if not s.get("rejected")]
            ok.sort(key=lambda s: (-s["n_solved"], s["sgm_s"]))
            best = ok[0] if ok else None
            conf = (evaluate(probs, best["genome"], tol, args.full_max_iters,
                             args.cap_s, want_traj=NEAR_MISS) if best else None)
            w10 = (sum(1 for s in ok if beats(s, base)) / len(ok)) if ok else 0.0
            per_family[fname] = {"screened": scr, "best_screen": best,
                                 "confirmed_full": conf, "W10": w10}
            if conf:
                print(f"   -> FULL BUDGET: {conf['n_solved']}/{len(probs)} "
                      f"sgm={conf['sgm_s']:.2f}s   W10={w10:.0%}", flush=True)
        # union across families at full budget
        union = set(base["solved"])
        for fname, blk in per_family.items():
            c = blk.get("confirmed_full")
            if c:
                union |= set(c["solved"])
        print(f"\nUNION across families: {len(union)}/{len(probs)}  "
              f"(B0 alone {base['n_solved']})", flush=True)
        per_family["_union"] = {"n": len(union), "instances": sorted(union)}
        result["tolerances"][f"{tol:g}"] = per_family

    json.dump(result, open(OUT, "w"), indent=1, default=float)
    print(f"\nwrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
