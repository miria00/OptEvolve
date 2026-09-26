"""Audit the frozen cell split for hidden distribution shift beyond nnz.

The nnz match was fixed deliberately; this checks that nothing ELSE is badly
skewed across CALIBRATION / SEARCH_POOL / FINAL before any solver result
exists. Reports per-set summaries and a two-sided KS test of each set against
the pooled remainder, for:

  rows, cols, nnz, density, m/n aspect ratio, coefficient dynamic range
  (max|A_ij| / min nonzero |A_ij|), mean |A_ij|, and the standard-form
  expansion factor (converted cols / original cols), which reflects how much
  bound and inequality structure the conversion introduces.

A small p-value is not automatically fatal (with n ~ 20-35 per set some
imbalance is expected), but anything extreme should be reshuffled NOW rather
than explained later.
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from atlas.bench import mps_cell  # noqa: E402

SPLIT = os.path.join(ROOT, "results", "sprint", "phase3", "cell_split.json")
SUITES = os.path.join(ROOT, "data", "lp_suites")
SETS = ("calibration", "search_pool", "final")


def features(entry):
    path = os.path.join(SUITES, entry["file"])
    f = {"rows": entry.get("rows") or 0, "cols": entry.get("cols") or 0,
         "nnz": entry.get("nnz") or 0}
    f["density"] = f["nnz"] / max(1, f["rows"] * f["cols"])
    f["aspect_m_over_n"] = f["rows"] / max(1, f["cols"])
    try:
        std = mps_cell.load_standard_form(path)
        A = std.A.tocsr()
        d = np.abs(A.data)
        d = d[d > 0]
        f["dyn_range"] = float(d.max() / d.min()) if d.size else 1.0
        f["mean_abs"] = float(d.mean()) if d.size else 0.0
        f["stdform_expansion"] = A.shape[1] / max(1, f["cols"])
    except Exception as e:  # noqa: BLE001
        print(f"  ! {entry['name']}: {type(e).__name__}", flush=True)
        f["dyn_range"] = f["mean_abs"] = f["stdform_expansion"] = float("nan")
    return f


def ks_2samp(a, b):
    a, b = np.sort(np.asarray(a)), np.sort(np.asarray(b))
    if a.size == 0 or b.size == 0:
        return float("nan"), float("nan")
    allv = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, allv, "right") / a.size
    cdf_b = np.searchsorted(b, allv, "right") / b.size
    d = float(np.max(np.abs(cdf_a - cdf_b)))
    en = math.sqrt(a.size * b.size / (a.size + b.size))
    lam = (en + 0.12 + 0.11 / en) * d
    p = 2.0 * sum((-1) ** (k - 1) * math.exp(-2.0 * k * k * lam * lam)
                  for k in range(1, 101))
    return d, min(1.0, max(0.0, p))


def main() -> None:
    split = json.load(open(SPLIT))
    feats = {}
    for s in SETS:
        rows = split[s]["instances"]
        print(f"loading {s} ({len(rows)})", flush=True)
        feats[s] = [features(e) for e in rows]

    keys = ["rows", "cols", "nnz", "density", "aspect_m_over_n",
            "dyn_range", "mean_abs", "stdform_expansion"]
    print(f"\n{'feature':20s} " + "".join(f"{s:>26s}" for s in SETS))
    print(f"{'':20s} " + "".join(f"{'median [IQR]':>26s}" for _ in SETS))
    report = {}
    for k in keys:
        line = f"{k:20s} "
        for s in SETS:
            v = np.array([f[k] for f in feats[s]], dtype=float)
            v = v[np.isfinite(v)]
            if v.size:
                q1, med, q3 = np.percentile(v, [25, 50, 75])
                line += f"{med:>10.4g} [{q1:.3g},{q3:.3g}]".rjust(26)
            else:
                line += "n/a".rjust(26)
        print(line)
        report[k] = {}
        for s in SETS:
            v = np.array([f[k] for f in feats[s]], dtype=float)
            v = v[np.isfinite(v)]
            other = np.concatenate([
                np.array([f[k] for f in feats[o]], dtype=float)
                for o in SETS if o != s]) if v.size else np.array([])
            other = other[np.isfinite(other)]
            d, p = ks_2samp(v, other)
            report[k][s] = {"n": int(v.size),
                            "median": float(np.median(v)) if v.size else None,
                            "ks_D_vs_rest": d, "ks_p_vs_rest": p}

    print("\nKS vs pooled remainder (two-sided; flagging p < 0.05):")
    flagged = []
    for k in keys:
        for s in SETS:
            p = report[k][s]["ks_p_vs_rest"]
            mark = "  <== FLAG" if (p == p and p < 0.05) else ""
            if mark:
                flagged.append((k, s, p))
            print(f"  {k:20s} {s:12s} D={report[k][s]['ks_D_vs_rest']:.3f} "
                  f"p={p:.3f}{mark}")
    out = os.path.join(ROOT, "results", "sprint", "phase3",
                       "cell_split_audit.json")
    json.dump({"report": report, "flagged": flagged}, open(out, "w"), indent=1)
    print(f"\n{len(flagged)} flagged. wrote {out}")


if __name__ == "__main__":
    main()
