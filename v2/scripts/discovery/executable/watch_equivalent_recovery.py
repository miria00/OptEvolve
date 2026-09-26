"""Freeze the first training recovery per LLM arm, then audit sealed holdouts.

The selection rule never uses held-out times, and test evidence is never returned
to the proposer. Selection is across seeds within each arm, explicitly exploratory.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import campaign as C
from recovery_equivalence import audit


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True)
    p.add_argument('--campaign',default='recovery_v1')
    p.add_argument('--slot',default='2');p.add_argument('--wait-for',default='league_typed_hybrid_v2')
    p.add_argument('--max-seconds',type=float,default=10800)
    a=p.parse_args();root=Path(a.root);search=root/a.campaign
    out=root/(a.campaign+'_equivalence_confirmation');out.mkdir(exist_ok=True)
    protocol={
        'selection':'first chronologically completed training recovery per LLM arm, pooled over two seeds',
        'arms':['withheld_cards','full_cards','resample'],
        'target':'HSI single-target method composition up to the ADMM/DRS state conjugacy, including actual retirement and screened certification',
        'amendment':'Secondary semantic audit added before any candidate passed it; original literal ADMM audit and results retained',
        'amended_at_utc':datetime.now(timezone.utc).isoformat(),
        'threshold':'all three held-out cases accepted in all three fresh repetitions, plus execution audit',
        'speedup_required':False,'budget_s':120,'repeats':3,'slot':a.slot,
        'test_manifest_sha256':hashlib.sha256((root/'data_final/manifest.json').read_bytes()).hexdigest(),
        'limitations':'Supplied standard atoms; no claim of missing-operator synthesis, new theorem, or all paper solvers recovered.'}
    if not (out/'protocol.json').exists():C.write_json(out/'protocol.json',protocol)
    C.write_json(out/'transport_recheck_policy.json',{
        'rule':'one recorded repeat of an unambiguous SSH banner failure; original search results retained',
        'scope':'diagnostic method recovery; does not alter original campaign hit counts',
        'timestamp':datetime.now(timezone.utc).isoformat()})
    status=out/'status.json'
    done=json.loads(status.read_text()).get('arms',{}) if status.exists() else {}
    start=time.monotonic()
    while time.monotonic()-start<a.max_seconds:
        prerequisite=root/a.wait_for/'status.json'
        ready=prerequisite.exists() and json.loads(prerequisite.read_text()).get('complete')
        found=[];overrides={}
        for recheck in (root/'infrastructure_rechecks').glob('*/result.json'):
            rr=json.loads(recheck.read_text())
            overrides[rr['origin']]={**rr['record'],'rows':rr['rows'],
                                     'infrastructure_recheck':str(recheck)}
        for file in search.glob('*_s*/p*/record.json'):
            record=overrides.get(str(file),json.loads(file.read_text()))
            code=file.parent/'candidate.py'
            if 'rows' in record and code.exists() and audit(record['spec'],code.read_text(),record['rows'])['method_recovered']:
                found.append(file)
        found.sort(key=lambda f:f.stat().st_mtime_ns)
        for file in found:
            arm=file.parent.parent.name.rsplit('_s',1)[0]
            if arm=='random' or arm in done or not ready:continue
            record=overrides.get(str(file),json.loads(file.read_text()))
            source=(file.parent/'candidate.py').read_text()
            code=out/(arm+'.py');code.write_text(source)
            C.write_json(out/(arm+'_selection.json'),{'origin':str(file),'record':record,
                'source_sha256':hashlib.sha256(source.encode()).hexdigest()})
            trial=out/arm
            cmd=[sys.executable,'-u',str(C.HERE/'confirm.py'),'--sources',arm+'='+str(code),
                 '--data',str(root/'data_final'),'--out',str(trial),'--split','test',
                 '--repeats','3','--budget','120','--slots',a.slot,
                 '--remote-root','/root/optevolve_recovery_equiv_confirm_'+arm]
            result=subprocess.run(cmd)
            if result.returncode:
                done[arm]={'confirmed':False,'infrastructure_error':result.returncode}
            else:
                rows=json.loads((trial/'results.json').read_text())
                evidence=audit(record['spec'],source,rows)
                # Require every planned result, not just a successful subset.
                confirmed=len(rows)==9 and evidence['method_recovered'] and all(r.get('accepted') for r in rows)
                done[arm]={'confirmed':confirmed,'training_origin':str(file),'audit':evidence,
                           'rows':rows,'scope':'automatic assembly of the HSI paper-method components; single-target protocol'}
                if confirmed and not (out/'first_confirmed_recovery.json').exists():
                    C.write_json(out/'first_confirmed_recovery.json',{'arm':arm,**done[arm]})
                print(json.dumps({'CONFIRMED_METHOD_RECOVERY':confirmed,'arm':arm,'training_origin':str(file)}),flush=True)
            C.write_json(out/'status.json',{'complete':False,'arms':done})
        statuses=list(search.glob('*_s*/status.json'))
        complete=len(statuses)==8 and all(json.loads(f.read_text()).get('complete') for f in statuses)
        eligible={f.parent.parent.name.rsplit('_s',1)[0] for f in found}-{'random'}
        if complete and eligible.issubset(done):break
        time.sleep(20)
    C.write_json(out/'status.json',{'complete':complete,'arms':done,'elapsed_s':time.monotonic()-start})

if __name__=='__main__':main()
