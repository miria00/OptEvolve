"""After first sealed recovery, confirm its frozen source at full paper batch size.

This is scale reproduction on the complete image (including training pixels),
not another unseen-data test. The single-target timing is distinct from the
manuscript's multi-target ladder. The reference is hidden from the proposer.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from scipy.io import loadmat
import campaign as C
from recovery_equivalence import audit

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True)
    ap.add_argument('--confirmation-dir',default='recovery_v1_equivalence_confirmation')
    ap.add_argument('--max-wait',type=float,default=10800);a=ap.parse_args()
    root=Path(a.root);data=root/'data_paper_scale'
    path=data/'test/cuprite_full.npz'
    if not (data/'manifest.json').exists():
        (data/'test').mkdir(parents=True,exist_ok=False)
        raw=loadmat('/home/miria/data/hsi/Cuprite.mat')
        np.savez(path,D=np.asarray(raw['D'],float),Y=np.asarray(raw['Y'],float))
        C.write_json(data/'manifest.json',{'scope':'full paper image; includes training and test pixels; scale reproduction, not independent generalization',
            'target':1e-4,'shape':[list(raw['D'].shape),list(raw['Y'].shape)],
            'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    manifest=json.loads((data/'manifest.json').read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest()==manifest['sha256']
    C.write_json(root/'paper_scale_equivalence_protocol.json',{
        'selection':'the first LLM arm to pass the predeclared sealed recovery confirmation',
        'reference':'hidden manually ported paper single-target schedule',
        'repetitions':3,'budget_s':300,'slot':2,'target':1e-4,'speedup_required':False,
        'source_changes_allowed':False,'scope':'full-image single-target reproduction; not paper multi-target timing'})
    marker=root/a.confirmation_dir/'first_confirmed_recovery.json'
    start=time.monotonic()
    while not marker.exists():
        if time.monotonic()-start>a.max_wait:raise TimeoutError('no sealed recovery; no full-scale candidate selected')
        time.sleep(20)
    selected=json.loads(marker.read_text());arm=selected['arm']
    source=root/a.confirmation_dir/(arm+'.py')
    out=root/'confirmation_recovery_paper_scale'
    command=[sys.executable,'-u',str(C.HERE/'confirm.py'),'--sources','recovered='+str(source),
             'paper_schedule='+str(C.HERE/'audit_paper_schedule.py'),
             '--data',str(data),'--out',str(out),'--split','test','--repeats','3',
             '--budget','300','--slots','2','--remote-root','/root/optevolve_recovery_full_scale']
    subprocess.run(command,check=True)
    record=json.loads(Path(selected['training_origin']).read_text())
    rows=json.loads((out/'results.json').read_text())
    got=[r for r in rows if r['label']=='recovered']
    result=audit(record['spec'],source.read_text(),got)
    result.update(arm=arm,all_three_full_scale_passed=len(got)==3 and all(r.get('accepted') for r in got))
    C.write_json(out/'recovery_audit.json',result)
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
