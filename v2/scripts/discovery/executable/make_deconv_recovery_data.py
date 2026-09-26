"""Freeze an application-shift simplex LS assay, disjoint from HSI search.

Both cell deconvolution datasets have the same mathematical simplex LS class;
this is a distribution/application shift, not a new optimization class.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--admission', required=True)
    ap.add_argument('--lm22', default='/home/miria/data/deconv/deconv_lm22.npz')
    ap.add_argument('--til10', default='/home/miria/data/deconv/deconv_til10.npz')
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    for split in ('train', 'admission', 'test'):
        (out / split).mkdir()
    for p in Path(a.admission).glob('*.npz'):
        shutil.copy2(p, out / 'admission' / p.name)
    rows = []
    inputs = [('lm22', Path(a.lm22), [
        ('train', 'lm22_2048', 0, 2048),
        ('train', 'lm22_4096', 2048, 6144),
        ('test', 'lm22_2048_holdout', 6144, 8192)]),
        ('til10', Path(a.til10), [
        ('test', 'til10_2048', 0, 2048),
        ('test', 'til10_4096', 2048, 6144)])]
    for label, source, slices in inputs:
        with np.load(source, allow_pickle=False) as d:
            D = np.asarray(d['A'], np.float64)
            Y = np.asarray(d['B'], np.float64)
        for split, name, lo, hi in slices:
            path = out / split / (name + '.npz')
            np.savez(path, D=D, Y=Y[:, lo:hi])
            rows.append({'split': split, 'name': name, 'signature': label,
                         'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                         'column_interval': [lo, hi], 'D': list(D.shape),
                         'Y': list(Y[:, lo:hi].shape),
                         'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    (out / 'manifest.json').write_text(json.dumps({
        'scope': 'cell deconvolution application shift; same simplex least-squares class',
        'seed': 0, 'cases': rows,
        'target': 1e-4, 'feasibility_tol': 1e-9,
        'test_never_deployed_to_search_worker': True}, indent=2))
    print(out / 'manifest.json')


if __name__ == '__main__':
    main()
