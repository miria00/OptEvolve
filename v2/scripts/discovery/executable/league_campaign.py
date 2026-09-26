"""Reference-taught program search against baselines and successive champions.

Baseline source is never shown to the proposer. Its executable and all champions
are immutable opponents. Only fixed training cases influence search or promotion.
Use a separate confirm.py run for final validation/test performance.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import random
import time
import urllib.request
import campaign as C
from judge import aggregate
from league import compare,StopRule
from space import API,PATCH_API,TASK,REFERENCES,source_id,validate_source


def typed_proposal(prompt,endpoint,seed,out):
    from typed_space import response_schema,canonical_spec,compile_spec
    schema=response_schema();schema['properties']['rationale']['maxLength']=1000
    body={'model':'Qwen/Qwen3.8-27B','messages':[{'role':'user','content':prompt}],
          'temperature':.8,'seed':seed,'max_tokens':5000,
          'chat_template_kwargs':{'enable_thinking':False},
          'response_format':{'type':'json_schema','json_schema':{'name':'composition','schema':schema,'strict':True}}}
    C.write_json(out/'request.json',body);start=time.perf_counter()
    request=urllib.request.Request(endpoint+'/v1/chat/completions',data=json.dumps(body).encode(),
                                   headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(request,timeout=300) as response:raw=json.load(response)
    C.write_json(out/'response.json',raw)
    if raw['choices'][0]['finish_reason']=='length':raise ValueError('truncated typed proposal')
    answer=json.loads(raw['choices'][0]['message']['content']);answer['spec']=canonical_spec(answer['spec'])
    answer['code'],certificate=compile_spec(answer['spec']);C.write_json(out/'construction.json',certificate)
    return answer,{'seconds':time.perf_counter()-start,'usage':raw.get('usage',{})}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--baselines',nargs='+',required=True,help='name=path; source hidden from proposer')
    p.add_argument('--seed-source',default=str(C.HERE/'seed.py'))
    p.add_argument('--data',required=True);p.add_argument('--out',required=True)
    p.add_argument('--cases',default='train/cuprite256,train/synthetic96',
                   help='comma-separated training cases; validation/test are forbidden')
    p.add_argument('--remote-root',required=True);p.add_argument('--slot',type=int,required=True)
    p.add_argument('--endpoint',default='http://127.0.0.1:8011')
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--evaluations',type=int,default=24);p.add_argument('--budget',type=float,default=30)
    p.add_argument('--max-seconds',type=float,default=7200);p.add_argument('--patience',type=int,default=12)
    p.add_argument('--repeats',type=int,default=3);p.add_argument('--min-speedup',type=float,default=1.05)
    p.add_argument('--max-case-regression',type=float,default=1.20)
    p.add_argument('--target-speedup',type=float)
    p.add_argument('--arm',choices=['league','resample'],default='league')
    p.add_argument('--representation',choices=['patch','typed'],default='patch')
    p.add_argument('--executable-references',action='store_true')
    a=p.parse_args()
    if a.repeats<3:raise ValueError('use at least three paired repetitions')
    root=Path(a.out);root.mkdir(parents=True,exist_ok=False)
    frozen=root/'sources';frozen.mkdir()
    C.REMOTE=a.remote_root
    slots=[(h,port,g) for h,port,n in C.NODES for g in range(n)];slot=slots[a.slot]
    cases=a.cases.split(',')
    if not cases or any(not c.startswith('train/') or '..' in c or c.count('/')!=1 for c in cases):
        raise ValueError('league search may only use explicit training cases')
    rng=random.Random(79000+a.seed);parent_rng=random.Random(89000+a.seed)
    order_rng=random.Random(99000+a.seed)
    stop=StopRule(a.evaluations,a.max_seconds,a.patience,a.target_speedup)
    start=time.perf_counter();pool=[];archive=[];seen=set();promotions=[]
    def freeze(label,path):
        source=Path(path).read_text();validate_source(source)
        target=frozen/(label+'.py');target.write_text(source)
        return {'name':label,'path':str(target),'id':source_id(source),
                'sha256':hashlib.sha256(target.read_bytes()).hexdigest()}
    for item in a.baselines:
        label,path=item.split('=',1)
        if not label.replace('_','').isalnum():raise ValueError('invalid baseline label')
        if any(r['name']==label for r in pool):raise ValueError('duplicate baseline label')
        pool.append(freeze(label,path))
    if a.representation=='typed':
        from typed_space import DEFAULT,compile_spec,SPACE_DESCRIPTION
        code,_=compile_spec(DEFAULT);temporary=root/'typed_seed.py';temporary.write_text(code)
        initial=freeze('proposer_seed',temporary);initial['spec']=DEFAULT
    else:initial=freeze('proposer_seed',a.seed_source)
    C.write_json(root/'protocol.json',{'args':vars(a),'opponents':pool,'seed':initial,'cases':cases,
        'data_sha256':hashlib.sha256((Path(a.data)/'manifest.json').read_bytes()).hexdigest(),
        'source_hashes':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in C.HERE.glob('*.py')},
        'scope':'training-set adversarial league; fixed certificate; no weight training or test adaptation'})
    snap=root/'source_snapshot';snap.mkdir()
    for f in C.HERE.glob('*.py'):(snap/f.name).write_bytes(f.read_bytes())
    C.deploy(a.data,a.out)
    for baseline in [*pool,initial]:
        rows=C.evaluate_on_slot(slot,Path(baseline['path']),cases,root/'initial'/baseline['name'],a.budget)
        baseline.update(rows=rows,score=aggregate(rows,a.budget))
    champion=max(pool,key=C.rank);anchor=dict(champion)
    parent=initial;archive.append(initial)
    seen.add(initial['id']);evaluations=attempts=since_promotion=0;anchor_speedup=1.
    pending_prompt=None;pending_parent=None
    while True:
        reason=stop.reason(evaluations=evaluations,elapsed_s=time.perf_counter()-start,
                           since_promotion=since_promotion,anchor_speedup=anchor_speedup)
        if attempts>=4*a.evaluations:reason='proposal cap exhausted (including duplicates and format errors)'
        if reason:break
        attempts+=1;od=root/f'p{attempts:03d}';od.mkdir()
        record={'proposal':attempts,'incumbent_id':champion['id'],'arm':a.arm}
        try:
            # Two siblings share precisely the same input, supporting later
            # execution-grounded preference training without invented labels.
            if pending_prompt is None:
                if a.arm=='resample':parent=initial
                elif (evaluations//2)%2==1 and archive[-1].get('gate_error'):parent=archive[-1]
                elif promotions:parent=champion
                elif parent_rng.random()<.25:parent=parent_rng.choice(sorted(archive,key=C.rank,reverse=True)[:4])
                else:parent=max(archive,key=C.rank)
                if a.representation=='typed':
                    prompt=TASK+'\n'+SPACE_DESCRIPTION+'\n'+REFERENCES
                    prompt+='\nParent design: '+json.dumps(parent['spec'])
                    if a.arm=='league':
                        prompt+='\nAlready attempted designs (return a different design): '+json.dumps([r['spec'] for r in archive])
                else:
                    api=API.replace('Return JSON {"rationale": "short mechanism and expected benefit", "code": "full module"}.','')
                    prompt=TASK+'\n'+api+'\n'+PATCH_API+'\n'+REFERENCES
                    prompt+='\nParent source:\n'+Path(parent['path']).read_text()
                if a.executable_references:
                    prompt+='\nExecutable standard teaching examples:\n'+(C.HERE/'standard_reference_examples.py').read_text()
                    prompt+='\nThese examples establish mathematical updates and array shapes. The examples do not choose a fast execution strategy for this workload. Write your own functions; this module is not available to import in the worker.'
                if a.arm=='league':
                    feedback=[{k:r.get(k) for k in ('id','spec','score','rows','gate_error','promotion')} for r in archive[-6:]]
                    for item in feedback:
                        item['rows']=[{k:v for k,v in row.items() if k not in ('trace','state_shapes')} for row in item['rows']]
                    prompt+='\nExecuted feedback: '+json.dumps(feedback)
                    prompt+='\nCurrent opponent measurements (no source): '+json.dumps({'name':champion['name'],'rows':champion['rows'],'score':champion['score']})
                    prompt+='\nA challenger must solve every case, improve repeated geometric-mean time by at least '+str(a.min_speedup)+'x, and avoid per-case regressions above '+str(a.max_case_regression)+'x. The certificate never changes. Propose a concrete structural or hardware change from measured evidence.'
                pending_prompt=prompt;pending_parent=dict(parent)
            prompt=pending_prompt;parent=pending_parent
            record.update(parent=parent['id'],prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest())
            if a.representation=='typed':
                answer,meta=typed_proposal(prompt,a.endpoint,rng.randrange(2**31),od)
                record['spec']=answer['spec']
            else:answer,meta=C.llm(prompt,a.endpoint,rng.randrange(2**31),od,Path(parent['path']).read_text())
            code=answer['code'];sid=source_id(code);record.update(id=sid,llm=meta,rationale=answer.get('rationale',''))
            if sid in seen:
                record['duplicate']=True;C.write_json(od/'record.json',record);continue
            seen.add(sid);source=od/'candidate.py';source.write_text(code)
            evaluations+=1;since_promotion+=1
            record.update(evaluation=evaluations,path=str(source),name=f'candidate_{attempts:03d}')
            gate=C.evaluate_on_slot(slot,source,['admission/identity','admission/rank_deficient','admission/zero'],od,8)
            record['admission']=gate
            if all(r.get('accepted') for r in gate):
                rows=C.evaluate_on_slot(slot,source,cases,od,a.budget)
            else:
                record['gate_error']=[r for r in gate if not r.get('accepted')]
                rows=[{'case':c,'accepted':False,'elapsed_s':a.budget} for c in cases]
            record.update(rows=rows,score=aggregate(rows,a.budget));archive.append(record)
            # Single measurements screen candidates only; they never promote.
            if (record['score']['solved']==len(cases) and
                record['score']['penalized_geomean_s']<champion['score']['penalized_geomean_s']/a.min_speedup):
                # Current champion, fixed initial anchor, and one rotating old
                # opponent prevent reward drift and forgotten regressions.
                opponents={champion['id']:champion,anchor['id']:anchor}
                if pool:
                    older=pool[(evaluations-1)%len(pool)];opponents[older['id']]=older
                competitors={sid:record,**opponents};measured={key:[] for key in competitors}
                for case in cases:
                    for repetition in range(a.repeats):
                        order=list(competitors);order_rng.shuffle(order)
                        for position,key in enumerate(order):
                            competitor=competitors[key]
                            output=od/'contest'/case.replace('/','_')/f'r{repetition}'/key[:12]
                            row=C.evaluate_on_slot(slot,Path(competitor['path']),[case],output,a.budget)[0]
                            row.update(repetition=repetition,order=position);measured[key].append(row)
                decisions={key:compare(measured[sid],measured[key],cap_s=a.budget,
                    min_speedup=a.min_speedup if key==champion['id'] else 1/a.max_case_regression,
                    max_case_regression=a.max_case_regression) for key in opponents}
                promotion=all(d['promote'] for d in decisions.values())
                record['promotion']={'promote':promotion,'contests':decisions}
                C.write_json(od/'contest.json',{'measurements':measured,'decisions':decisions})
                if promotion:
                    champion=record;pool.append(record);since_promotion=0
                    anchor_speedup=decisions[anchor['id']]['speedup_geomean']
                    promotions.append({'id':sid,'evaluation':evaluations,'anchor_speedup':anchor_speedup})
                    (root/'champion.py').write_text(code)
                    C.write_json(root/'champion.json',record)
            C.write_json(od/'record.json',record)
            print(json.dumps({'evaluation':evaluations,'score':record['score'],'promotion':record.get('promotion')}),flush=True)
        except Exception as exc:
            record['proposal_error']=str(exc)[-2500:];C.write_json(od/'record.json',record)
            print(json.dumps({'proposal':attempts,'error':str(exc)[-300:]}),flush=True)
        finally:
            if attempts%2==0:pending_prompt=pending_parent=None
        C.write_json(root/'status.json',{'complete':False,'evaluations':evaluations,'proposals':attempts,
            'promotions':promotions,'champion':champion,'elapsed_s':time.perf_counter()-start})
    C.write_json(root/'status.json',{'complete':True,'termination':reason,'evaluations':evaluations,
        'proposals':attempts,'promotions':promotions,'champion':champion,'elapsed_s':time.perf_counter()-start})
    print(json.dumps({'termination':reason,'promotions':len(promotions)}),flush=True)


if __name__=='__main__':main()
