"""Admissible region in ABSOLUTE (t, s) coordinates, per family / size / degree.

structure_probe.py parameterises  s = theta / (t*||L||^2), so the gate verdict
there is a function of theta ALONE, by construction -- it cannot show how the
admissible SET moves when the family or the size changes.  This companion
sweeps a fixed catalog of RAW (t, s) pairs, identical for every cell, and asks
prepare_base for a verdict.  Because the gate is  t*s*||L||^2 < 1  and ||L||
is a property of the instance, the SAME raw pair flips verdict across cells.

For every admitted pair it also solves, so the absolute-coordinate map carries
both a verdict and a time-to-certificate.

Writes results/structure_20260909/rawgrid_<device>.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np

import structure_probe as SP  # noqa: E402  (same scripts/ dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default=(
        "grid:12:0,grid:16:0,grid:24:0,grid:32:0,"
        "regular:1024:3,regular:1024:4,regular:1024:6,regular:1024:8,"
        "regular:512:4,regular:2048:4"),
        help="comma list of family:size:degree")
    ap.add_argument("--taus", default="0.01,0.025,0.05,0.1,0.25,0.5,1.0")
    ap.add_argument("--sigmas", default="0.01,0.025,0.05,0.1,0.25,0.5,1.0")
    ap.add_argument("--n-instances", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scheme", default="pdhg", choices=("pdhg", "condat_vu"))
    ap.add_argument("--rho", type=float, default=1.0)
    ap.add_argument("--K", type=int, default=64)
    ap.add_argument("--max-iters", type=int, default=150000)
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--no-solve", action="store_true",
                    help="gate verdicts only (symbolic, no execution)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out-dir", default=SP.DEFAULT_OUT_DIR)
    args = ap.parse_args()

    taus = [float(v) for v in args.taus.split(",")]
    sigmas = [float(v) for v in args.sigmas.split(",")]
    out = {
        "version": 1,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "args": vars(args),
        "device": {"kind": SP.jax.devices()[0].device_kind,
                   "tag": SP.device_tag(args.device)},
        "parameterisation": (
            "RAW absolute (t, s) pairs, identical across every cell; the gate "
            "lhs t*s*||L||^2 therefore varies across cells only through ||L||"),
        "instances": [],
        "runs": [],
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"rawgrid_{SP.device_tag(args.device)}.json"

    def flush():
        out_path.write_text(json.dumps(out, indent=1, default=SP._json_default))

    for spec in args.cells.split(","):
        fam, size_s, deg_s = spec.split(":")
        size, degree = int(size_s), int(deg_s)
        cell_id = f"{fam}_s{size}" + (f"_d{degree}" if fam == "regular" else "")
        problems = SP.build_cell(fam, size, max(degree, 1), args.n_instances,
                                 args.seed, 0.1, "test", "center")
        print(f"\n=== raw-grid cell {cell_id} ===", flush=True)
        for i, prob in enumerate(problems):
            feats = SP.structure_features(prob, max_dense_elems=8e6,
                                          max_gram_dim=4096, max_nnz=4e7)
            feats.update(cell_id=cell_id, family=fam, size=size,
                         degree=degree, instance=i)
            out["instances"].append(feats)
            nb = feats["norm_bound"]
            print(f"  [{cell_id} #{i}] ||L||_gate={nb:.6f} "
                  f"sigma_max={feats['sigma_max']} "
                  f"kappa+={feats['kappa_plus']} "
                  f"(1/||L||^2 = {1.0/nb**2:.6f})", flush=True)
            for t in taus:
                for s in sigmas:
                    if args.no_solve:
                        lhs, prod = SP.gate_lhs(args.scheme, t, s, nb,
                                                SP.smooth_lipschitz(prob))
                        rec = {"scheme": args.scheme, "t": t, "s": s,
                               "rho": args.rho, "ratio_t_over_s": t / s,
                               "opnorm_gate": nb, "theta_gate": prod,
                               "gate_lhs": lhs, "gate_admit": lhs < 1.0,
                               "N": None, "exec_s": None, "converged": None}
                    else:
                        rec = SP.run_setting(prob, args.scheme, t, s, args.rho,
                                             K=args.K, max_iters=args.max_iters,
                                             tol=args.tol, repeats=args.repeats,
                                             loop="static")
                    rec.update(cell_id=cell_id, family=fam, size=size,
                               degree=degree, instance=i,
                               kappa_plus=feats["kappa_plus"],
                               sigma_max=feats["sigma_max"],
                               theta_true=(None if feats["sigma_max"] is None
                                           else t * s * feats["sigma_max"] ** 2))
                    out["runs"].append(rec)
                    tag = "ADMIT " if rec["gate_admit"] else "REJECT"
                    extra = ("" if rec.get("N") is None else
                             f" N={rec['N']:>7d} exec={rec['exec_s']:.4f}s "
                             f"conv={rec['converged']}")
                    print(f"    t={t:<6g} s={s:<6g} lhs={rec['gate_lhs']:<9.4f} "
                          f"{tag}{extra}", flush=True)
                    flush()
    flush()
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
