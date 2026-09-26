"""Add large training batches without consuming any frozen validation/test pixel.

This is a new exploratory distribution, not a retroactive change to prior runs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
from scipy.io import loadmat


def main():
    p=argparse.ArgumentParser();p.add_argument('--original',required=True)
    p.add_argument('--out',required=True);p.add_argument('--mat',default='/home/miria/data/hsi/Cuprite.mat')
    a=p.parse_args();original=Path(a.original);out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((original/'manifest.json').read_text())
    used=set(i for case in manifest['cases'] for i in case.get('pixel_indices',[]))
    mat=loadmat(a.mat);D=np.asarray(mat['D'],float);Y=np.asarray(mat['Y'],float)
    available=np.array(sorted(set(range(Y.shape[1]))-used))
    np.random.default_rng(2026092602).shuffle(available)
    if len(available)<20480:raise ValueError('not enough unallocated pixels')
    cases=[]
    for case in manifest['cases']:
        if case['split'] not in ('train','admission'):continue
        target=out/case['split']/(case['name']+'.npz');target.parent.mkdir(exist_ok=True)
        shutil.copy2(original/case['split']/target.name,target);cases.append(case)
    offset=0
    for size in [4096,16384]:
        ids=available[offset:offset+size];offset+=size
        assert not (used&set(ids.tolist()))
        path=out/'train'/f'cuprite{size}.npz';np.savez(path,D=D,Y=Y[:,ids])
        cases.append({'split':'train','name':f'cuprite{size}','shape':[*D.shape,size],
                      'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'pixel_indices':ids.tolist()})
    manifest.update(cases=cases,parent_manifest_sha256=hashlib.sha256((original/'manifest.json').read_bytes()).hexdigest(),
                    seed=2026092602,test_root=str(original),
                    scope='new scale curriculum; original validation/test pixels excluded; no test feedback to proposer')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps({'new_cases':[4096,16384],'excluded_original_pixels':len(used),'overlap':0}))


if __name__=='__main__':main()
