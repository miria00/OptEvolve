"""Low-memory LIBSVM -> X.npy (float32, dense, memory-mapped) + y.npy (int8, +-1).

Decompresses a .bz2 once to disk, then parses byte ranges with scikit-learn's
C parser (load_svmlight_file offset/length) and writes each chunk straight into
an open_memmap, so peak memory is one chunk rather than the whole matrix.
Runs in /home/miria/enetenv. Usage: SRC(.bz2) OUTDIR --n-features P [--chunk-mb 512]
"""
import argparse, bz2, hashlib, json, os, shutil, time
from pathlib import Path
import numpy as np
from sklearn.datasets import load_svmlight_file

ap = argparse.ArgumentParser(); ap.add_argument("src"); ap.add_argument("out")
ap.add_argument("--n-features", type=int, required=True); ap.add_argument("--chunk-mb", type=int, default=512)
ap.add_argument("--label-map", default=None,
                help="raw:signed pairs, e.g. 1:1,2:-1; required unless the raw labels are already -1/+1")
ap.add_argument("--zero-based", action="store_true",
                help="feature indices start at 0 (LIBSVM files are 1-based; auto-detection is unreliable per chunk)")
a = ap.parse_args()
LMAP = None if a.label_map is None else {float(k): int(v) for k, v in (kv.split(":") for kv in a.label_map.split(","))}
if a.chunk_mb < 1:
    ap.error("--chunk-mb must be at least 1")
t0 = time.time(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
src = Path(a.src)
h = hashlib.sha256()
with open(src, "rb") as f:
    for blk in iter(lambda: f.read(1 << 24), b""):
        h.update(blk)
txt = out / "raw.svm"
if src.suffix == ".bz2":
    with bz2.open(src, "rb") as fi, open(txt, "wb") as fo:
        shutil.copyfileobj(fi, fo, length=1 << 24)
else:
    txt = src
n = 0
with open(txt, "rb") as f:
    for _ in f:
        n += 1
X = np.lib.format.open_memmap(out / "X.npy", mode="w+", dtype=np.float32, shape=(n, a.n_features))
y = np.empty(n, dtype=np.int8)
size, step, off, row = os.path.getsize(txt), a.chunk_mb << 20, 0, 0
while off < size:
    Xc, yc = load_svmlight_file(str(txt), n_features=a.n_features, dtype=np.float32, offset=off, length=step,
                                zero_based=a.zero_based)
    k = Xc.shape[0]
    raw = set(np.unique(yc).tolist())
    if LMAP is None and not raw <= {-1.0, 1.0}:   # a sign test would silently merge classes (covtype: {1, 2})
        raise SystemExit(f"labels {sorted(raw)} are not -1/+1; pass --label-map")
    if LMAP is not None and not raw <= set(LMAP):
        raise SystemExit(f"labels {sorted(raw - set(LMAP))} missing from --label-map")
    X[row:row + k] = Xc.toarray()
    y[row:row + k] = np.vectorize(LMAP.get)(yc) if LMAP is not None else yc.astype(np.int8)
    row += k; off += step
assert row == n, (row, n)
X.flush(); np.save(out / "y.npy", y)
if txt != src:
    txt.unlink()
json.dump({"source": str(src), "sha256_source": h.hexdigest(), "n": n, "p": a.n_features,
           "positives": int((y > 0).sum()), "seconds": time.time() - t0}, open(out / "meta.json", "w"), indent=1)
print(open(out / "meta.json").read())
