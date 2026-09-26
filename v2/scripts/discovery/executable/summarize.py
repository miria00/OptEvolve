"""Summarize executed candidates, including every failure. No inferred wins."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re

def summarize(root):
    rows = []
    for chain in sorted(Path(root).glob('*_s*')):
        if not chain.is_dir() or not re.search(r'_s\d+$',chain.name):
            continue
        records = [json.loads(p.read_text()) for p in sorted(chain.glob('p*/record.json'))]
        responses = [json.loads(p.read_text()) for p in sorted(chain.glob('p*/response.json'))]
        scored = [r for r in records if 'score' in r]
        admitted = [r for r in scored if 'gate_error' not in r]
        solved = [r for r in admitted if r['score']['solved'] == r['score']['total']]
        best = min(solved,key=lambda r:r['score']['penalized_geomean_s']) if solved else None
        status = json.loads((chain/'status.json').read_text()) if (chain/'status.json').exists() else {}
        errors = Counter()
        for r in records:
            if 'proposal_error' in r:
                errors['proposal_or_api'] += 1
            elif r.get('duplicate'):
                errors['duplicate'] += 1
            elif r.get('gate_error'):
                errors['admission_execution_error' if any('error' in q for q in r['gate_error'])
                       else 'admission_numerical_failure'] += 1
            elif r.get('score',{}).get('solved',0) < r.get('score',{}).get('total',0):
                errors['search_nonconvergence'] += 1
        rows.append({'chain':chain.name,'complete':status.get('complete',False),
                     'proposals':len(records),'fresh_candidates':len(scored),
                     'admitted':len(admitted),'all_cases_solved':len(solved),
                     'errors':dict(errors),
                     'best':None if best is None else {'id':best['id'],
                       'proposal':best['proposal'],'score':best['score'],
                       'rationale':best['rationale'],
                       'path':str(chain/f"p{best['proposal']:03d}"/'candidate.py')},
                     'model_tokens':sum((r.get('usage') or {}).get('total_tokens',0) for r in responses),
                     'model_wall_s_recorded_lower_bound':sum((r.get('llm') or {}).get('seconds',0) for r in records)})
    return {'scope':'search only; not held-out confirmation or significance', 'chains':rows}

if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--run',required=True)
    p.add_argument('--out')
    a=p.parse_args()
    result=summarize(a.run)
    if a.out:
        Path(a.out).write_text(json.dumps(result,indent=2))
    for row in result['chains']:
        print(json.dumps({k:v for k,v in row.items() if k != 'best'}))
        if row['best']:
            print(' BEST',row['best']['path'],row['best']['score'])
