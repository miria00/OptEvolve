"""Wait for scale leagues, freeze training-selected code, then run the sealed test.

No model calls or source edits occur here. This file makes the continuation and
final-selection rule inspectable before final test measurements are available.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import campaign as C
from judge import aggregate


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True)
    p.add_argument('--runs',default='league_scale_v1,league_scale_examples_v2')
    p.add_argument('--max-wait-seconds',type=float,default=10800)
    a=p.parse_args();root=Path(a.root);runs=a.runs.split(',');start=time.monotonic()
    C.write_json(root/'final_selection_protocol.json',{
        'runs':runs,'rule':'best training solved count, then penalized geometric mean time across these exploratory continuations',
        'final_data_manifest_sha256':hashlib.sha256((root/'data_final/manifest.json').read_bytes()).hexdigest(),
        'repetitions':5,'budget_s':120,'slots':[0,1],
        'claim_scope':'pooled exploratory selection; not a causal ablation or additional independent discovery seed'})
    while True:
        done=[]
        for name in runs:
            file=root/name/'status.json'
            done.append(file.exists() and json.loads(file.read_text()).get('complete'))
        subprocess.run([sys.executable,str(C.HERE/'report.py'),'--root',str(root)],check=True)
        if all(done):break
        if time.monotonic()-start>a.max_wait_seconds:
            raise TimeoutError('prerequisite league unfinished; final test has not been started')
        time.sleep(30)
    selections=[]
    for name in runs:
        league=root/name
        records=[json.loads(f.read_text()) for f in league.glob('p*/record.json')]
        records=[r for r in records if 'score' in r]
        rows=[json.loads(f.read_text()) for f in (league/'initial/proposer_seed').glob('*.json')]
        initial={'path':str(league/'sources/proposer_seed.py'),'rows':rows,'score':aggregate(rows,60)}
        best=max([initial,*records],key=C.rank);selections.append({'run':name,'record':best})
    selected=max(selections,key=lambda r:C.rank(r['record']))
    C.write_json(root/'scale_final_selection.json',{'per_run':selections,'pooled_selection':selected})
    active=root/'patch_v2_s1/withheld_cards_s1/winner.py'
    sources=['active_set='+str(active),'admm='+str(C.HERE/'audit_oracle.py'),
             'paper_schedule='+str(C.HERE/'audit_paper_schedule.py')]
    # This separate comparator was selected on the original small training
    # distribution, before seeing the new final cases. It is not a candidate
    # in the pooled four-case selection above.
    small=root/'league_typed_v1/champion.py'
    if small.exists():sources.append('typed_small_league='+str(small))
    chosen=Path(selected['record']['path'])
    if hashlib.sha256(chosen.read_bytes()).digest()!=hashlib.sha256(active.read_bytes()).digest():
        sources.append('league_selected='+str(chosen))
    command=[sys.executable,'-u',str(C.HERE/'confirm.py'),'--sources',*sources,
        '--data',str(root/'data_final'),'--out',str(root/'confirmation_scale_final'),
        '--split','test','--repeats','5','--budget','120','--slots','0,1',
        '--remote-root','/root/optevolve_discovery_scale_final']
    subprocess.run(command,check=True)
    subprocess.run([sys.executable,str(C.HERE/'report.py'),'--root',str(root)],check=True)


if __name__=='__main__':main()
