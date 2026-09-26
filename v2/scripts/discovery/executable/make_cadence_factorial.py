"""Freeze a post-search 2x2 cadence/chunk ablation of recovered ADMM."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import recovery_space as R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--record', required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    sources = out / 'sources'; sources.mkdir()
    data = out / 'data'; (data / 'test').mkdir(parents=True)
    original = json.loads(Path(a.record).read_text())['spec']
    assert original['algorithm'] == 'admm' and original['certificate'] == 'host'
    assert original['width']['chunk'] == 500
    combos = []
    for cert in ('host', 'device_screened'):
        for chunk in (500, 128):
            spec = json.loads(json.dumps(original))
            spec['certificate'] = cert
            spec['width']['chunk'] = chunk
            code, _ = R.compile_spec(spec)
            label = ('screen' if cert == 'device_screened' else 'host') + str(chunk)
            path = sources / (label + '.py')
            path.write_text(code)
            combos.append({'label': label, 'spec': spec,
                           'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    parent = Path(a.data)
    test = parent / 'test' / 'cuprite10000.npz'
    shutil.copy2(test, data / 'test' / test.name)
    (data / 'manifest.json').write_text(json.dumps({
        'scope': 'post-search mechanism ablation on one frozen held-out case; not model selection',
        'parent_manifest_sha256': hashlib.sha256((parent / 'manifest.json').read_bytes()).hexdigest(),
        'case_sha256': hashlib.sha256(test.read_bytes()).hexdigest(),
        'source_record_sha256': hashlib.sha256(Path(a.record).read_bytes()).hexdigest(),
        'combinations': combos}, indent=2))
    print(out)


if __name__ == '__main__':
    main()
