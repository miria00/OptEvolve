"""Score the roofline predictor against the measured fusion sweep.

A predictor is only worth reporting if it ranks tiles the way the hardware
does. This scores three nested models by rank correlation against measurement:
  M1 roofline only
  M2 roofline + occupancy (shared memory limits blocks per SM)
  M3 roofline + occupancy + wave quantisation (too few blocks to fill the GPU)
"""
import json, math, sys
from pathlib import Path
R = Path('/home/miria/OptEvolve/v2/results')

# Ada AD102 (RTX 4090): 128 SMs, 100 KB shared/SM, 1536 threads/SM.
SM_COUNT, SMEM_PER_SM, MAX_THREADS_SM = 128, 100 * 1024, 1536
THREADS = {8: 64, 16: 128, 24: 256, 32: 256, 40: 256}

def spearman(a, b):
    def rank(v):
        s = sorted(range(len(v)), key=lambda i: v[i])
        r = [0] * len(v)
        for pos, i in enumerate(s): r[i] = pos
        return r
    ra, rb = rank(a), rank(b)
    n = len(a)
    d2 = sum((ra[i] - rb[i]) ** 2 for i in range(n))
    return 1 - 6 * d2 / (n * (n * n - 1)) if n > 1 else float('nan')

rf = json.load(open(R / 'diagnostics_20260909/roofline_4090.json'))
base = {t['tile']: t for t in rf['tiles']}

def occupancy(B, T):
    S = B + 2 * T
    smem = 5 * S * S * 8
    blocks = max(1, SMEM_PER_SM // smem)
    thr = THREADS.get(B, 256)
    return min(1.0, blocks * thr / MAX_THREADS_SM), blocks

def predict(tile, n, model):
    B, T = (int(v) for v in tile.split('x'))
    t = base[tile]['pred_s_per_pixel_iter']
    if model >= 2:
        occ, _ = occupancy(B, T)
        t /= max(occ, 1e-6)                      # low occupancy cannot hide latency
    if model >= 3:
        nblocks = math.ceil(n / B) ** 2
        waves = nblocks / SM_COUNT
        t *= math.ceil(waves) / waves if waves > 0 else 1   # partial last wave wastes SMs
    return t

SRC = [(128,'kernel_fusion_20260909/e2e/full_4090_sz128.json'),
       (256,'kernel_fusion_20260909/e2e/full_4090_sz256.json'),
       (384,'kernel_fusion_20260909/e2e/full_4090_sz384.json'),
       (496,'kernel_fusion_20260909/e2e/full_4090_sz496.json'),
       (992,'kernel_fusion_20260909/large/large_4090_m2.json'),
       (1488,'kernel_fusion_20260909/large/large_4090_m3.json'),
       (1984,'kernel_fusion_20260909/large/large_4090_m4.json')]

print(f"{'n':>6s} {'tiles':>6s} " + " ".join(f"{'rho_M'+str(m):>8s}" for m in (1,2,3)) +
      f" {'measured best':>14s} " + " ".join(f"{'M'+str(m)+' pick':>10s}" for m in (1,2,3)))
agg = {1: [], 2: [], 3: []}
hits = {1: 0, 2: 0, 3: 0}; tot = 0
for n, f in SRC:
    p = R / f
    if not p.exists(): continue
    rows = json.load(open(p))['rows']
    keys = [k for k in rows[0] if k.startswith('tf')]
    meas, tiles = [], []
    for k in keys:
        tile = k[2:]
        if tile not in base: continue
        tiles.append(tile)
        meas.append(sum(r[k].get('t_s', float('nan')) for r in rows))
    if len(tiles) < 3: continue
    tot += 1
    best_m = tiles[meas.index(min(meas))]
    line = f"{n:6d} {len(tiles):6d} "
    picks = []
    for m in (1, 2, 3):
        pred = [predict(t, n, m) for t in tiles]
        rho = spearman(pred, meas)
        agg[m].append(rho)
        pick = tiles[pred.index(min(pred))]
        picks.append(pick)
        if pick == best_m: hits[m] += 1
        line += f"{rho:8.3f} "
    print(line + f"{best_m:>14s} " + " ".join(f"{p:>10s}" for p in picks))
print()
for m in (1, 2, 3):
    v = agg[m]
    name = {1: 'roofline only', 2: '+occupancy', 3: '+occupancy+waves'}[m]
    print(f"  M{m} {name:20s} mean Spearman {sum(v)/len(v):6.3f}   "
          f"picked the true best tile {hits[m]}/{tot}")

# --- M4: roofline plus a fixed per-launch cost -------------------------------
# M1's rho is strongly NEGATIVE at n=128 and strongly POSITIVE at n>=992, which
# says roofline is the right model only once the kernel is bandwidth-bound. The
# missing term at small n is launch/loop overhead, which fusion amortises by T.
# One fitted constant L (seconds per launch), spread over n^2 * T useful
# pixel-iterations.
print("\n--- M4: roofline + fitted per-launch cost ---")
import itertools
data = []
for n, f in SRC:
    p = R / f
    if not p.exists(): continue
    rows = json.load(open(p))['rows']
    for k in [k for k in rows[0] if k.startswith('tf')]:
        tile = k[2:]
        if tile not in base: continue
        B, T = (int(v) for v in tile.split('x'))
        tot_s = sum(r[k].get('t_s', float('nan')) for r in rows)
        iters = rows[0][k].get('iters') or 1
        n_img = len(rows)
        per_pixel_iter = tot_s / (n_img * iters * n * n)
        data.append((n, tile, B, T, per_pixel_iter))

best_L, best_score = None, -2
for L in [0] + [10 ** e for e in range(-8, -3)] + [3 * 10 ** e for e in range(-8, -4)]:
    rhos = []
    for n, _ in SRC:
        grp = [d for d in data if d[0] == n]
        if len(grp) < 3: continue
        pred = [base[t]['pred_s_per_pixel_iter'] + L / (T * n * n) for (_, t, B, T, _m) in grp]
        meas = [d[4] for d in grp]
        rhos.append(spearman(pred, meas))
    if rhos and sum(rhos) / len(rhos) > best_score:
        best_score, best_L = sum(rhos) / len(rhos), L
print(f"  best fitted launch cost L = {best_L:.2e} s, mean Spearman {best_score:.3f}")
hits4 = 0; tot4 = 0
print(f"\n{'n':>6s} {'rho_M4':>8s} {'measured best':>14s} {'M4 pick':>10s}")
for n, _ in SRC:
    grp = [d for d in data if d[0] == n]
    if len(grp) < 3: continue
    tot4 += 1
    pred = [base[t]['pred_s_per_pixel_iter'] + best_L / (T * n * n) for (_, t, B, T, _m) in grp]
    meas = [d[4] for d in grp]
    pick = grp[pred.index(min(pred))][1]; bm = grp[meas.index(min(meas))][1]
    if pick == bm: hits4 += 1
    print(f"{n:6d} {spearman(pred, meas):8.3f} {bm:>14s} {pick:>10s}")
print(f"\n  M4 picked the true best tile {hits4}/{tot4}")
