"""Render method-recovery evidence without redefining success after a run."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import time

def render(root):
    root=Path(root);search=root/'recovery_v1';confirmation=root/'recovery_v1_confirmation'
    lines=['# Automatic method recovery status','',
        'Updated '+datetime.now(timezone.utc).isoformat()+'.','',
        'Target: automatic composition of the paper HSI method from supplied standard operations. No reference speedup is required. This does not establish synthesis of missing operators or recovery of all paper solvers.','',
        '| Chain | Fresh evaluated | Programs passing all training cases | First literal-ADMM recovery | Finished |',
        '|---|---:|---:|---|---|']
    for chain in sorted(search.glob('*_s[0-9]*')):
        if not chain.is_dir():continue
        rows=[json.loads(p.read_text()) for p in chain.glob('p*/record.json')]
        fresh=[r for r in rows if 'evaluation' in r]
        solved=[r for r in fresh if r.get('score',{}).get('solved',-1)==3]
        first=chain/'first_recovery.json'
        hit=str(json.loads(first.read_text())['evaluation']) if first.exists() else 'none'
        status=chain/'status.json';done=status.exists() and json.loads(status.read_text()).get('complete')
        lines.append(f'| {chain.name} | {len(fresh)} / 8 | {len(solved)} | {hit} | {done} |')
    equivalents=root/'recovery_v1_equivalence_confirmation'
    for selection in sorted(equivalents.glob('*_selection.json')):
        selected=json.loads(selection.read_text())
        r=selected['record']
        lines+=['',f"Semantic training recovery selected: **{selection.stem.replace('_selection','')}**, proposal {r['proposal']}. Source: `{selected['origin']}`."]
        if 'infrastructure_recheck' in r:
            lines+=['','This uses one separately recorded rerun after an SSH banner failure; the original failed search score remains in the table above.']
    lines+=['','## Sealed confirmation','',
            'First training recovery per LLM arm is selected before test measurements. Three cases, three fresh repetitions each, fixed 120-second cap. All nine must pass; the method execution audit must also pass.','',
            '| Arm | Case | Passed / completed | Median accepted seconds |','|---|---|---:|---:|']
    for kind,folder in [('literal ADMM',confirmation),('equivalent ADMM/DRS',equivalents)]:
        for arm in ('withheld_cards','full_cards','resample'):
            trial=folder/arm;result=trial/'results.json'
            rr=json.loads(result.read_text()) if result.exists() else [json.loads(p.read_text()) for p in trial.glob('raw/*/r*/*/*.json')]
            groups=defaultdict(list)
            for r in rr:groups[r['case']].append(r)
            for case,rows in sorted(groups.items()):
                accepted=[r for r in rows if r.get('accepted')]
                median=f"{statistics.median(r['elapsed_s'] for r in accepted):.3f}" if accepted else '—'
                lines.append(f'| {arm} ({kind}) | {case} | {len(accepted)}/{len(rows)} | {median} |')
    firsts=[f/'first_confirmed_recovery.json' for f in (confirmation,equivalents) if (f/'first_confirmed_recovery.json').exists()]
    if firsts:
        first=min(firsts,key=lambda p:p.stat().st_mtime_ns)
        r=json.loads(first.read_text());lines+=['',f"**Confirmed first recovery: {r['arm']}.**",'',
            'Scope: single-target method composition on two real pixel holdouts and a synthetic shift. The real holdouts share the training spectral library. Full-image and multi-target paper timings remain separate checks.']
    else:lines+=['','**No complete held-out method recovery has been confirmed yet.**']
    lines+=['','Artifacts retain prompts, model responses, source hashes, all failures, compiled candidates and independent float64 certificate results. Expert hints are provided only to full_cards; all arms share the executable primitives.']
    (root/'RECOVERY_STATUS.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True)
    p.add_argument('--watch-seconds',type=float,default=0)
    a=p.parse_args();render(a.root);start=time.monotonic()
    while time.monotonic()-start<a.watch_seconds:
        time.sleep(20);render(a.root)
