"""Convert a LIBSVM binary-classification file to X.npy (float32, dense) + y.npy (int8, +-1).
Runs in /home/miria/enetenv (scikit-learn's C svmlight parser). Usage:
  libsvm_to_npy.py SRC.bz2 OUTDIR --n-features 2000"""
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
from sklearn.datasets import load_svmlight_file
ap = argparse.ArgumentParser(); ap.add_argument("src"); ap.add_argument("out"); ap.add_argument("--n-features", type=int, default=None)
a = ap.parse_args()
t0 = time.time()
X, y = load_svmlight_file(a.src, n_features=a.n_features, dtype=np.float32)
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
np.save(out / "X.npy", X.toarray().astype(np.float32))
y = np.where(y > 0, 1, -1).astype(np.int8); np.save(out / "y.npy", y)
json.dump({"source": str(a.src), "sha256_source": hashlib.sha256(open(a.src, "rb").read()).hexdigest(),
           "n": int(X.shape[0]), "p": int(X.shape[1]), "nnz": int(X.nnz), "positives": int((y > 0).sum()),
           "seconds": time.time() - t0}, open(out / "meta.json", "w"), indent=1)
print(open(out / "meta.json").read())
