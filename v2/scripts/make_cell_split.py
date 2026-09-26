"""Freeze the hard-LP cell split: CALIBRATION / SEARCH-POOL / FINAL holdout.

PRE-REGISTRATION CONTRACT
  FINAL   never used for tuning, sweeps, search, seed selection or any
          measurement until the frozen champions are evaluated once.
  CALIB   used for caps, B0 tuning, the structural headroom assay, and the
          1e-6 tractability question.
  SEARCH  the pool evolution draws fitness from. Its FINAL membership is
          fixed in a second step, AFTER calibration tells us what is
          tractable at 1e-6; freezing it before that would either be broken
          or force us to break the freeze. The pool is frozen here; the
          drawn subset is stamped later by --finalize-search.

PROVENANCE (recorded in the artifact, and belongs in the paper). ALL THREE
SETS are drawn from instances never touched by any sweep, solve, tuning or
selection, so there is no contamination argument to make.

The 137 instances used in the earlier operator-norm / gate-admission sweep
(scripts/pc_ood_sweep.py, plus one 20k-iteration diagnostic solve of app1-1)
are EXCLUDED from this cell entirely. They were excluded rather than reused
for calibration because they are not size-comparable to the untouched pool:
seen median nnz 20,000 against untouched median 63,995, with exactly ONE of
the 137 falling inside the untouched nnz range. Reusing them would have made
the final holdout a size-shift transfer test rather than held-out
generalization, which is precisely the Phase-2 error this split exists to
prevent. Fetching the remainder of the MIPLIB 2017 benchmark set raised the
untouched pool to 140 and made the clean design affordable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUITES = os.path.join(ROOT, "data", "lp_suites")
PHASE3 = os.path.join(ROOT, "results", "sprint", "phase3")
OUT = os.path.join(PHASE3, "cell_split.json")
SEED = 20260904


def sha_of(names) -> str:
    h = hashlib.sha256()
    for n in sorted(names):
        h.update(n.encode())
        h.update(b"\0")
    return h.hexdigest()


def load_seen() -> set[str]:
    seen = set()
    for tier in ("small", "medium"):
        p = os.path.join(PHASE3, f"pc_ood_{tier}.json")
        if os.path.exists(p):
            d = json.load(open(p))
            for e in d.get("instances", d if isinstance(d, list) else []):
                seen.add(e["name"])
    return seen


def stratify(rows, k, seed):
    """Pick k rows stratified across nnz quartiles, deterministically."""
    rows = sorted(rows, key=lambda e: e.get("nnz") or 0)
    if k >= len(rows):
        return list(rows)
    rng = random.Random(seed)
    nb = 4
    buckets = [rows[i * len(rows) // nb:(i + 1) * len(rows) // nb]
               for i in range(nb)]
    out, per = [], k // nb
    for b in buckets:
        out.extend(rng.sample(b, min(per, len(b))))
    rest = [e for e in rows if e not in out]
    rng.shuffle(rest)
    out.extend(rest[:k - len(out)])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-calibration", type=int, default=20)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--final-frac", type=float, default=0.4,
                    help="fraction of the non-calibration pool held out")
    ap.add_argument("--nnz-max", type=int, default=313260,
                    help="cell upper nnz bound (tractability at 1e-6)")
    args = ap.parse_args()

    man = json.load(open(os.path.join(SUITES, "MANIFEST.json")))
    med_max = man["medium_tier_nnz"]
    usable = [e for e in man["instances"]
              if e.get("usable", True) and 0 < (e.get("nnz") or 0) <= med_max]
    seen = load_seen()
    fresh_all = [e for e in usable if e["name"] not in seen]
    if not fresh_all:
        sys.exit("no untouched instances: cannot form an uncontaminated FINAL")

    # THE CELL IS AN NNZ BAND. We already downloaded the smallest instances
    # first, so everything untouched clusters at larger nnz: without a band,
    # FINAL (41550..104420) and the seen pool (48..1955388) overlap in ONE
    # instance out of 137, and "held-out generalization" would silently
    # become a size-shift transfer test, which is exactly the Phase-2 error
    # this split exists to avoid. So the cell is defined as the untouched
    # band widened by --band-pad, and ALL THREE sets are drawn inside it.
    lo = min(e["nnz"] for e in fresh_all)
    hi = int(args.nnz_max)
    in_band = lambda e: lo <= (e.get("nnz") or 0) <= hi  # noqa: E731
    fresh_rows = [e for e in fresh_all if in_band(e)]
    print(f"CELL nnz band: {lo} .. {hi}   untouched in band {len(fresh_rows)}")
    print(f"  (excluded from the cell: {len(seen)} previously-swept instances)")
    if len(fresh_rows) < args.n_calibration + 20:
        sys.exit(f"only {len(fresh_rows)} untouched instances in band; "
                 "raise --nnz-max or fetch more")

    # ALL THREE SETS COME FROM UNTOUCHED INSTANCES (2026-09-04, second pass).
    # The first pass drew calibration/search from the 137 previously-swept
    # instances and FINAL from untouched ones, which made FINAL a size-shifted
    # holdout: seen median nnz 20,000 vs untouched median 63,995, with ONE
    # instance of 137 overlapping FINAL's range. Fetching the rest of the
    # MIPLIB benchmark set raised the untouched pool to 140, enough to draw
    # every set from it. The 137 seen instances are now excluded from the cell
    # entirely, so no contamination argument is needed at all.
    pool = sorted(fresh_rows, key=lambda e: e.get("nnz") or 0)
    calib = stratify(pool, args.n_calibration, args.seed)
    taken = {e["name"] for e in calib}
    rest = [e for e in pool if e["name"] not in taken]
    n_final = max(1, int(round(len(rest) * args.final_frac)))
    final_rows = stratify(rest, n_final, args.seed + 1)
    taken |= {e["name"] for e in final_rows}
    search_pool = [e for e in rest if e["name"] not in taken]
    fresh_rows = final_rows

    def block(rows):
        return {
            "n": len(rows),
            "nnz_min": min((e.get("nnz") or 0) for e in rows) if rows else None,
            "nnz_max": max((e.get("nnz") or 0) for e in rows) if rows else None,
            "sha256": sha_of(e["name"] for e in rows),
            "instances": [
                {"name": e["name"], "file": e["file"], "nnz": e.get("nnz"),
                 "rows": e.get("rows"), "cols": e.get("cols"),
                 "source": e["source"], "sha256": e.get("sha256")}
                for e in sorted(rows, key=lambda r: r["name"])
            ],
        }

    out = {
        "created": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc).isoformat(),
        "seed": args.seed,
        "cell_nnz_band": [lo, hi],
        "nnz_max": args.nnz_max,
        "final_frac": args.final_frac,
        "contract": {
            "FINAL": "never used for tuning, sweeps, search or seed selection",
            "CALIBRATION": "caps, B0 tuning, headroom assay, 1e-6 tractability",
            "SEARCH_POOL": ("evolution fitness pool; drawn subset stamped "
                            "after calibration via --finalize-search"),
        },
        "disclosure": (
            "ALL THREE SETS are drawn from instances never touched by any "
            "sweep, solve, tuning or selection. The 137 instances used in "
            "the earlier operator-norm/gate-admission sweep are EXCLUDED "
            "from this cell entirely, so no contamination argument is "
            "required. They were excluded rather than reused because they "
            "are not size-comparable to the untouched pool (seen median nnz "
            "20,000 vs untouched median 63,995, one instance of 137 "
            "overlapping), which would have turned held-out generalization "
            "into a size-shift transfer test: the Phase-2 error."
        ),
        "excluded_seen_n": len(seen),
        "calibration": block(calib),
        "search_pool": block(search_pool),
        "final": block(fresh_rows),
        "search_frozen": None,
    }
    os.makedirs(PHASE3, exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)

    print("FROZEN CELL SPLIT")
    for k in ("calibration", "search_pool", "final"):
        b = out[k]
        print(f"  {k:12s} n={b['n']:4d}  nnz {b['nnz_min']}..{b['nnz_max']}  "
              f"sha256={b['sha256'][:16]}")
    overlap = (set(e["name"] for e in fresh_rows) & seen)
    print(f"  FINAL n seen-before: {len(overlap)} (must be 0)")
    assert not overlap
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
