"""Freeze search, validation and sealed test splits before any model call."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.io import loadmat

def synthetic(seed, n, p, m, condition=200):
    rng = np.random.default_rng(seed)
    u, _ = np.linalg.qr(rng.normal(size=(n, n)))
    v, _ = np.linalg.qr(rng.normal(size=(p, n)))
    D = u @ (np.geomspace(1, 1 / condition, n)[:, None] * v.T)
    W = np.zeros((p, m))
    for j in range(m):
        ids = rng.choice(p, 5, replace=False)
        W[ids, j] = rng.dirichlet(np.ones(5))
    Y = D @ W + rng.normal(size=(n, m)) * 1e-3
    return D, Y

def make(out, mat):
    out = Path(out)
    if (out / 'manifest.json').exists():
        raise FileExistsError('refusing to overwrite a frozen split')
    d = loadmat(mat)
    D, Y = np.asarray(d['D'], float), np.asarray(d['Y'], float)
    perm = np.random.default_rng(20260926).permutation(Y.shape[1])
    manifests = []
    def write(split, name, d, y, ids=None):
        path = out / split / (name + '.npz')
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, D=d, Y=y)
        row = {'split': split, 'name': name, 'shape': [*d.shape, y.shape[1]],
               'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        if ids is not None:
            row['pixel_indices'] = ids.tolist()
        manifests.append(row)
    write('train', 'cuprite256', D, Y[:, perm[:256]], perm[:256])
    write('train', 'synthetic96', *synthetic(1901, 96, 160, 32))
    write('validation', 'cuprite512', D, Y[:, perm[256:768]], perm[256:768])
    write('validation', 'synthetic128', *synthetic(1902, 128, 192, 64, 500))
    write('test', 'cuprite2048', D, Y[:, perm[768:2816]], perm[768:2816])
    write('test', 'cuprite8192', D, Y[:, perm[2816:11008]], perm[2816:11008])
    write('test', 'synthetic160', *synthetic(2901, 160, 256, 96, 1000))
    rng = np.random.default_rng(9001)
    for name, d, y in [
        ('identity', np.eye(6), rng.normal(size=(6, 5))),
        ('rank_deficient', np.ones((7, 9)), rng.normal(size=(7, 4))),
        ('zero', np.zeros((5, 8)), np.zeros((5, 3))),
    ]:
        write('admission', name, d, y)
    manifest = {'seed': 20260926, 'task': 'simplex_least_squares',
                'target': 1e-4, 'feasibility_tol': 1e-9, 'cases': manifests,
                'test_scope': 'disjoint pixels within one library plus synthetic shape/conditioning shift'}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps([{k:v for k,v in r.items() if k != 'pixel_indices'} for r in manifests], indent=2))

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    p.add_argument('--mat', default='/home/miria/data/hsi/Cuprite.mat')
    a = p.parse_args()
    make(a.out, a.mat)
