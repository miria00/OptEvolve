"""HiGHS wall-time CONTEXT lines for the Phase 2 OOD holdout tiers.

CROSS-ECOSYSTEM FLAG (stated wherever these numbers appear): HiGHS is a
presolve + simplex/IPM C++ solver at its own default tolerances (primal/
dual feasibility 1e-7); our configs are first-order primal-dual methods
in JAX-on-CPU without presolve, stopped at rel duality gap 1e-4 / 1e-6.
These numbers are context, never a same-rules comparison.

Uses the SAME instance selection as scripts/eval_holdout.py (stratified
48-instance small subsample, seed 20260903, + full medium tier).
Median-of-3 wall seconds per instance, fresh Highs object each solve.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

from atlas.bench.mps_cell import DATA_ROOT, MANIFEST_PATH  # noqa: E402
from atlas.bench.runner import sanitize_json  # noqa: E402
from eval_holdout import ood_specs  # noqa: E402
from phase2_common import machine_record  # noqa: E402


def solve_one(path: str) -> dict:
    import highspy

    from atlas.bench.mps_cell import _highs_readable_path

    rpath, tmp = _highs_readable_path(path)
    times = []
    rec = {}
    try:
        for _rep in range(3):
            h = highspy.Highs()
            h.setOptionValue("output_flag", False)
            t0 = time.time()
            if h.readModel(rpath) != highspy.HighsStatus.kOk:
                return {"error": "readModel failed"}
            t_read = time.time() - t0
            # LP RELAXATION, matching oracle_mps: all vars continuous
            # (readModel keeps MIP integrality otherwise).
            ncols = int(h.getLp().num_col_)
            h.changeColsIntegrality(
                ncols,
                np.arange(ncols, dtype=np.int32),
                np.full(ncols, highspy.HighsVarType.kContinuous),
            )
            t1 = time.time()
            h.run()
            times.append(time.time() - t1)
            rec = {
                "status": str(h.getModelStatus()),
                "objective": float(h.getObjectiveValue()),
                "read_s": t_read,
            }
    finally:
        if tmp is not None and os.path.exists(tmp):
            os.unlink(tmp)
    rec["wall_s_median3"] = float(np.median(times))
    rec["wall_s_all"] = times
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ood-small-n", type=int, default=48)
    ap.add_argument("--ood-seed", type=int, default=20260903)
    ap.add_argument("--ood-smallest", action="store_true")
    ap.add_argument("--no-medium", action="store_true")
    ap.add_argument("--manifest", default=MANIFEST_PATH)
    ap.add_argument("--data-root", default=DATA_ROOT)
    ap.add_argument("--machine", choices=("LOCAL", "REMOTE"), default="LOCAL")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    specs, sel = ood_specs(args)
    rows = []
    for s in specs:
        r = solve_one(s["path"])
        r.update(instance=s["name"], tier=s["tier"], nnz=s["nnz"])
        rows.append(r)
        print(
            f"  {s['name']} ({s['tier']}): "
            f"{r.get('wall_s_median3', 'ERR'):>10} s {r.get('status')}",
            flush=True,
        )
    out = {
        "experiment": "phase2_highs_context",
        "note": (
            "CROSS-ECOSYSTEM CONTEXT ONLY: HiGHS presolve+simplex/IPM at "
            "HiGHS default tolerances vs first-order no-presolve configs "
            "at rel-gap 1e-4/1e-6; never a same-rules comparison"
        ),
        **machine_record(args.machine),
        "selection": sel,
        "rows": rows,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(sanitize_json(out), fh, indent=2, allow_nan=False)
        fh.write("\n")
    print(f"[{args.machine}] highs context -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
