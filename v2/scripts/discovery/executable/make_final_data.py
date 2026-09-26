"""Seal new final cases for continuations after the initial test was inspected."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from make_data import synthetic


def main():
    p=argparse.ArgumentParser();p.add_argument('--exclude',nargs='+',required=True)
    p.add_argument('--out',required=True);p.add_argument('--mat',default='/home/miria/data/hsi/Cuprite.mat')
    a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    (out/'test').mkdir();used=set();parents=[]
    for directory in a.exclude:
        path=Path(directory)/'manifest.json';manifest=json.loads(path.read_text())
        used.update(i for row in manifest['cases'] for i in row.get('pixel_indices',[]))
        parents.append(hashlib.sha256(path.read_bytes()).hexdigest())
    raw=loadmat(a.mat);D=np.asarray(raw['D'],float);Y=np.asarray(raw['Y'],float)
    ids=np.array(sorted(set(range(Y.shape[1]))-used));np.random.default_rng(2026092603).shuffle(ids)
    if len(ids)<16144:raise ValueError('insufficient untouched pixels for final cases')
    rows=[];offset=0
    def write(name,d,y,indices=None):
        file=out/'test'/(name+'.npz');np.savez(file,D=d,Y=y)
        row={'name':name,'split':'test','shape':[*d.shape,y.shape[1]],
             'sha256':hashlib.sha256(file.read_bytes()).hexdigest()}
        if indices is not None:row['pixel_indices']=indices.tolist()
        rows.append(row)
    for n in [6144,10000]:
        selected=ids[offset:offset+n];offset+=n
        assert not (used&set(selected.tolist()))
        write('cuprite'+str(n),D,Y[:,selected],selected)
    write('synthetic192',*synthetic(2026092603,192,320,128,1500))
    (out/'manifest.json').write_text(json.dumps({'cases':rows,'seed':2026092603,
        'excluded_manifests':parents,'scope':'new final holdout for post-pilot continuations; same real library',
        'target':1e-4,'feasibility_tol':1e-9},indent=2))
    print(json.dumps({'new_final_real_pixels':16144,'excluded_pixels':len(used),'overlap':0}))


if __name__=='__main__':main()
