"""Freeze selected sources, then compare under randomized paired validation.

Never called by the search policy. All candidate sources are copied and hashed
before any validation/test result is visible. No selection on final test timings.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import random
import tarfile
import campaign as C

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--sources',nargs='+',required=True,help='label=/absolute/or/relative/program.py')
    p.add_argument('--data',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--split',choices=('validation','test'),default='validation')
    p.add_argument('--repeats',type=int,default=5)
    p.add_argument('--budget',type=float,default=60)
    p.add_argument('--slots',default='0,1,2,3,4,5')
    p.add_argument('--remote-root',default='/root/optevolve_discovery_confirmation')
    a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    source_dir=out/'sources';source_dir.mkdir()
    sources={}
    for item in a.sources:
        label,path=item.split('=',1)
        if not label.replace('_','').isalnum():
            raise ValueError('source label must be alphanumeric/underscore')
        data=Path(path).read_bytes(); frozen=source_dir/(label+'.py');frozen.write_bytes(data)
        sources[label]={'file':str(frozen),'sha256':hashlib.sha256(data).hexdigest(),'origin':str(path)}
    cases=[a.split+'/'+p.stem for p in sorted((Path(a.data)/a.split).glob('*.npz'))]
    if not cases:
        raise ValueError('no cases')
    C.write_json(out/'protocol.json',{'args':vars(a),'sources':sources,'cases':cases,
                    'data_manifest_sha256':hashlib.sha256((Path(a.data)/'manifest.json').read_bytes()).hexdigest(),
                    'design':'candidate order independently randomized within each repetition and case'})
    C.REMOTE=a.remote_root
    bundle=out/'bundle.tar.gz'
    with tarfile.open(bundle,'w:gz') as tar:
        for name in ('worker.py','space.py','judge.py'):
            tar.add(C.HERE/name,arcname=name)
        for path in (Path(a.data)/a.split).glob('*.npz'):
            tar.add(path,arcname='data/'+a.split+'/'+path.name)
    slots=[]
    for host,port,count in C.NODES:
        for gpu in range(count):slots.append((host,port,gpu))
    selected=[slots[int(s)] for s in a.slots.split(',')]
    for host,port in sorted({s[:2] for s in selected}):
        C.ssh(host,port,f'mkdir -p {C.REMOTE}/jobs')
        C.copy_to(host,port,bundle,C.REMOTE+'/bundle.tar.gz')
        C.ssh(host,port,f'tar -xzf {C.REMOTE}/bundle.tar.gz -C {C.REMOTE}')
    jobs=[(case,r) for case in cases for r in range(a.repeats)]
    # A given worker owns its GPU for its entire queue; candidates in each
    # comparison run serially on the same GPU, in a frozen random order.
    queues=[[] for _ in selected]
    for i,job in enumerate(jobs):queues[i%len(selected)].append(job)
    def work(slot,jobs):
        rows=[]
        for case,rep in jobs:
            order=list(sources)
            order_seed=int(hashlib.sha256(f'{case}:{rep}:4701'.encode()).hexdigest()[:8],16)
            random.Random(order_seed).shuffle(order)
            for index,label in enumerate(order):
                output=out/'raw'/case.replace('/','_')/f'r{rep:02d}'/label
                row=C.evaluate_on_slot(slot,Path(sources[label]['file']),[case],output,a.budget)[0]
                row.update(label=label,repetition=rep,order=index)
                rows.append(row)
                print(json.dumps({k:row.get(k) for k in ('case','label','repetition','accepted','elapsed_s','error')}),flush=True)
        return rows
    with ThreadPoolExecutor(max_workers=len(selected)) as pool:
        parts=list(pool.map(lambda x:work(*x),zip(selected,queues)))
    rows=[r for part in parts for r in part]
    C.write_json(out/'results.json',rows)

if __name__=='__main__':main()
