"""Build conservative execution-grounded training pairs from same-prompt siblings.

Default pairs compare complete certified success to numerical nonconvergence.
Syntax, API and admission errors are excluded: otherwise this repeats the old
format-repair experiment. Speed-only pairs require separate repeated measurements.
This exports data; it does not train model weights.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

def numerical_record(record):
    return ('score' in record and 'gate_error' not in record
            and not record.get('proposal_error') and not record.get('duplicate')
            and bool(record.get('rows')) and all('error' not in r for r in record['rows']))

def executable_answer(response):
    """Do not turn unchecked model explanations into reasoning supervision."""
    answer=json.loads(response['choices'][0]['message']['content'])
    if 'replacements' in answer:
        action={'replacements':answer['replacements']}
    elif 'code' in answer:
        action={'code':answer['code']}
    elif 'spec' in answer:
        action={'spec':answer['spec']}
    else:
        raise ValueError('response has no executable action')
    return json.dumps({'rationale':'',**action})

def build(root):
    groups = defaultdict(list)
    root=Path(root)
    paths=[*root.glob('*_s*/p*/record.json'),*root.glob('p*/record.json')]
    protocol=json.loads((root/'protocol.json').read_text()) if (root/'protocol.json').exists() else {}
    for path in sorted(paths):
        record = json.loads(path.read_text())
        if numerical_record(record):
            groups[(path.parent.parent.name,record['prompt_sha256'])].append((path,record))
    for entries in groups.values():
        good = [(p,r) for p,r in entries if r['score']['solved'] == r['score']['total']]
        bad = [(p,r) for p,r in entries if r['score']['solved'] < r['score']['total']]
        for (cp,chosen),(rp,rejected) in zip(good,bad):
            creq = json.loads((cp.parent/'request.json').read_text())
            rreq = json.loads((rp.parent/'request.json').read_text())
            if creq['messages'] != rreq['messages']:
                raise ValueError('preference outputs do not have identical conditioning')
            cres = json.loads((cp.parent/'response.json').read_text())
            rres = json.loads((rp.parent/'response.json').read_text())
            yield {'prompt':creq['messages'],
                   'chosen':[{'role':'assistant','content':executable_answer(cres)}],
                   'rejected':[{'role':'assistant','content':executable_answer(rres)}],
                   'provenance':{'chosen_id':chosen['id'], 'rejected_id':rejected['id'],
                                 'arm':chosen['arm'], 'seed':chosen.get('seed',protocol.get('args',{}).get('seed')),
                                 'reason':'certified every search case versus numerical nonconvergence',
                                 'split':'train', 'rationales_removed':True,
                                 'weight_training_performed':False}}

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    pairs = list(build(a.run))
    out = Path(a.out)
    out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists():
        raise FileExistsError('use a fresh training snapshot')
    out.write_text(''.join(json.dumps(row)+'\n' for row in pairs))
    print(json.dumps({'pairs':len(pairs),'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),
                      'trained':False}))
