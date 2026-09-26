"""Collect every measured result into one table.  Reads artifacts, never
transcribes numbers by hand."""
import glob, json, os, sys

R = "/home/miria/OptEvolve/v2/results"

def load(p):
    try:
        return json.load(open(p))
    except Exception:
        return None

print("=" * 78)
print("REGIME 1  hyperspectral unmixing, Cuprite 188 x 47750, 498-signature library")
print("=" * 78)
for tag, p in [("ours 4090 f64", f"{R}/hsi_unmix_20260924/ours_4090_f64_full.json"),
               ("SUnSAL cpu   ", f"{R}/hsi_unmix_20260924/sunsal_cpu_full.json")]:
    r = load(p)
    if not r: print(f"  {tag}: (not finished)"); continue
    if "hit" in r:
        h = r["hit"]
        g = lambda k: ("%.1fs" % h[k]["t"]) if k in h else "-"
        print("  %s  1e-3 %-9s 1e-4 %-9s  final worst %.2e  (%.0fs total)"
              % (tag, g("0.001"), g("0.0001"), r["final_worst_gap"], r["wall_total"]))
    else:
        print("  %s  %d iters %.0fs  worst %.2e  median %.2e"
              % (tag, r["iters"], r["wall"], r["worst_gap"], r["median_gap"]))

print()
print("=" * 78)
print("REGIME 2  isotropic TV denoising, DIV2K 0801, lam = 0.1")
print("=" * 78)
print("  %-14s %8s %9s %9s %9s %9s" % ("arm", "size", "1e-3", "1e-4", "1e-6", "1e-8"))
for tag, pat in [("ours 4090 f64", f"{R}/tv_20260924/ours_4090f64_*.json")]:
    rows = []
    for p in glob.glob(pat):
        r = load(p)
        if r: rows.append(r)
    for r in sorted(rows, key=lambda r: r["shape"][0]):
        h = r["hit"]; g = lambda k: ("%.2fs" % h[k]["t"]) if k in h else "-"
        print("  %-14s %8s %9s %9s %9s %9s"
              % (tag, "%d^2" % r["shape"][0], g("0.001"), g("0.0001"), g("1e-06"), g("1e-08")))
for p in sorted(glob.glob(f"{R}/tv_20260924/skimage_ref_*.json")):
    r = load(p)
    if not r: continue
    h = r["hit"]; g = lambda k: ("%.2fs" % h[k]["t"]) if k in h else "-"
    print("  %-14s %8s %9s %9s %9s %9s   best rel %.1e"
          % ("skimage cpu", "%d^2" % r["shape"][0], g("0.001"), g("0.0001"),
             g("1e-06"), g("1e-08"), r["best_rel"]))

print()
print("=" * 78)
print("REGIME 3  isotropic TV deblurring, no packaged solver exists")
print("=" * 78)
rows = [load(p) for p in glob.glob(f"{R}/deblur_20260924/ours_*.json")]
rows = [r for r in rows if r]
if not rows:
    print("  (sweep still running)")
for r in sorted(rows, key=lambda r: r["shape"][0]):
    h = r["hit"]; g = lambda k: ("%.2fs" % h[k]["t"]) if k in h else "-"
    print("  %-14s %8s %9s %9s %9s %9s" % ("ours 4090 f64", "%d^2" % r["shape"][0],
          g("0.001"), g("0.0001"), g("1e-06"), g("1e-08")))
