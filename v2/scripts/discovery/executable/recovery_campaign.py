"""Matched LLM/random composition search over the same canonical typed space."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import queue
import random
import time
import urllib.request
import campaign as C
from space import TASK,REFERENCES,HSI_CARDS
from judge import aggregate
from recovery_space import DEFAULT,SPACE_DESCRIPTION,compile_spec,canonical_spec,random_spec,response_schema,audit

def proposal(parent,history,arm,rng,out,endpoint,structured=False,context=None):
    if arm=='random':
        return random_spec(rng),'uniform sampling from the same admissible component space',None
    prompt=TASK+'\n'+REFERENCES+'\n'+SPACE_DESCRIPTION+'\n'+(C.HERE/'standard_reference_examples.py').read_text()
    prompt+='\nActual training distribution and evaluation budget: '+json.dumps(context)
    if arm=='full_cards':prompt+='\n'+HSI_CARDS
    if arm!='resample':
        prompt+='\nCurrent design: '+json.dumps(parent['spec'])
        prompt+='\nExecuted designs and measurements: '+json.dumps(history[-8:])
    prompt+='\nReturn one complete candidate design. Find a reliably certifiable efficient solver over the full distribution, including setup and certificates. No requirement to beat a reference runtime. Avoid repeating executed designs.'
    request={'model':'Qwen/Qwen3.8-27B','messages':[{'role':'user','content':prompt}],
             'temperature':.8,'seed':rng.randrange(2**31),'max_tokens':3000,
             'chat_template_kwargs':{'enable_thinking':False},'response_format':{'type':'json_object'}}
    if structured:
        request['response_format']={'type':'json_schema','json_schema':{
            'name':'solver_composition','schema':response_schema(),'strict':True}}
    C.write_json(out/'request.json',request)
    start=time.perf_counter()
    req=urllib.request.Request(endpoint+'/v1/chat/completions',data=json.dumps(request).encode(),
                               headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=300) as response:raw=json.load(response)
    C.write_json(out/'response.json',raw)
    if raw['choices'][0]['finish_reason']=='length':raise ValueError('truncated model response')
    result=json.loads(raw['choices'][0]['message']['content'])
    return result['spec'],result.get('rationale',''),{'seconds':time.perf_counter()-start,'usage':raw.get('usage',{})}

def parent_rank(record,progress=False):
    s=record['score']
    if progress and s['solved']<s['total']:
        import math
        failed=[r for r in record['rows'] if not r.get('accepted')]
        worst=max((max(r.get('worst_gap',1e6)/1e-4,r.get('feasibility',1e6)/1e-9,
                       -r.get('min_gap',0.)/1e-9)
                   for r in failed),default=1e6)
        return (s['solved'],-math.log(max(worst,1.0)),-s['penalized_geomean_s'])
    return (s['solved'],0.,-s['penalized_geomean_s'])

def chain(arm,seed,slot,args):
    out=Path(args.out)/f'{arm}_s{seed}';out.mkdir()
    rng=random.Random(33000+seed);archive=[];seen=set();n=0;attempt=0
    cases=args.cases.split(',')
    parent={'spec':DEFAULT,'evaluated':False,'rows':[],
            'score':{'solved':0,'total':len(cases),'penalized_geomean_s':args.budget}}
    # The plain seed specifies the starting program; it is not a measured
    # candidate or credited with any solution. Fresh budget counts proposals.
    code,cert=compile_spec(DEFAULT);(out/'seed.py').write_text(code)
    C.write_json(out/'seed.json',parent)
    while n<args.evaluations and attempt<4*args.evaluations:
        attempt+=1;od=out/f'p{attempt:03d}';od.mkdir()
        record={'arm':arm,'seed':seed,'proposal':attempt,'parent':parent['spec']}
        try:
            fields=['case','accepted','elapsed_s','worst_gap','feasibility','iterations','setup_s','step_s','certificate_s','error']
            if args.profile_feedback and arm!='scalar':
                fields.extend(['restriction_s','handoff_s','output_s','state_shape_changes'])
            history=[{'spec':r['spec'],'score':r['score'],
                      'rows':[{k:v for k,v in row.items() if k in fields}
                              for row in r['rows']]} for r in archive]
            spec,rationale,meta=proposal(parent,history,arm,rng,od,args.endpoint,args.structured,args.search_context)
            spec=canonical_spec(spec)
            sid=hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()
            record.update(spec=spec,id=sid,rationale=rationale,llm=meta)
            if sid in seen:
                record['duplicate']=True;C.write_json(od/'record.json',record);continue
            seen.add(sid)
            code,cert=compile_spec(spec);(od/'candidate.py').write_text(code)
            C.write_json(od/'construction.json',cert)
            n+=1;record['evaluation']=n
            gates=C.evaluate_on_slot(slot,od/'candidate.py',
                ['admission/identity','admission/rank_deficient','admission/zero'],od,8)
            record['admission']=gates
            if all(r.get('accepted') for r in gates):
                rows=C.evaluate_on_slot(slot,od/'candidate.py',cases,od,args.budget)
            else:
                record['gate_error']=[r for r in gates if not r.get('accepted')]
                rows=[{'case':c,'accepted':False,'elapsed_s':args.budget} for c in cases]
            record.update(rows=rows,score=aggregate(rows,args.budget))
            record['recovery_audit']=audit(spec,code,rows)
            C.write_json(od/'record.json',record);archive.append(record)
            if record['recovery_audit']['method_recovered'] and not (out/'first_recovery.json').exists():
                C.write_json(out/'first_recovery.json',record)
                (out/'recovered.py').write_text(code)
                print(json.dumps({'METHOD_RECOVERED':out.name,'evaluation':n,'source':str(od/'candidate.py')}),flush=True)
            parent=max(archive,key=lambda r:parent_rank(r,args.progress_selection))
            print(json.dumps({'chain':out.name,'eval':n,'spec':spec,'score':record['score'],'recovered':record['recovery_audit']['method_recovered']}),flush=True)
        except Exception as exc:
            record['proposal_error']=str(exc)[-2500:];C.write_json(od/'record.json',record)
            print(json.dumps({'chain':out.name,'error':str(exc)[-300:]}),flush=True)
        C.write_json(out/'status.json',{'complete':False,'evaluated':n,'proposals':attempt,'best':parent})
    code,cert=compile_spec(parent['spec']);(out/'winner.py').write_text(code)
    C.write_json(out/'winner.json',parent)
    C.write_json(out/'status.json',{'complete':True,'evaluated':n,'proposals':attempt,'best':parent})

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',required=True);ap.add_argument('--data',required=True)
    ap.add_argument('--cases',default='train/cuprite256,train/synthetic96')
    ap.add_argument('--endpoint',default='http://127.0.0.1:8011')
    ap.add_argument('--remote-root',required=True)
    ap.add_argument('--slots',default='0,1,2,3,4,5')
    ap.add_argument('--seeds',default='0,1');ap.add_argument('--arms',default='withheld_cards,full_cards,random,resample')
    ap.add_argument('--evaluations',type=int,default=16);ap.add_argument('--budget',type=float,default=30)
    ap.add_argument('--structured',action='store_true')
    ap.add_argument('--profile-feedback',action='store_true')
    ap.add_argument('--progress-selection',action='store_true')
    a=ap.parse_args()
    if any(not c.startswith('train/') or '..' in c or c.count('/')!=1 for c in a.cases.split(',')):
        raise ValueError('search may use only training cases')
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    C.REMOTE=a.remote_root
    import numpy as np
    a.search_context={'budget_seconds_per_case':a.budget,'cases':[]}
    for case in a.cases.split(','):
        with np.load(Path(a.data)/(case+'.npz'),allow_pickle=False) as data:
            a.search_context['cases'].append({'case':case,'D':list(data['D'].shape),'Y':list(data['Y'].shape)})
    C.write_json(out/'protocol.json',{'args':vars(a),'space_sha256':hashlib.sha256(Path(__file__).with_name('recovery_space.py').read_bytes()).hexdigest(),
       'source_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in C.HERE.glob('*.py')},
       'data_manifest_sha256':hashlib.sha256((Path(a.data)/'manifest.json').read_bytes()).hexdigest(),
       'claim_scope':'Prospective HSI standard-atom composition recovery; ADMM branch has numerical admission, not an Atlas proof; no lasso D3-D6 implementation',
       'success_rule':'private execution audit plus independent certification; no reference-speedup threshold; all arms retain the full fresh-evaluation budget',
       'fairness':'same canonical action support, same fresh evaluation budget; record actual elapsed/token cost separately'})
    import shutil
    snapshot=out/'source_snapshot';snapshot.mkdir()
    for file in C.HERE.glob('*.py'):shutil.copy2(file,snapshot/file.name)
    C.deploy(a.data,a.out)
    allslots=[(h,p,g) for h,p,count in C.NODES for g in range(count)]
    chosen=[allslots[int(i)] for i in a.slots.split(',')]
    slots=queue.Queue()
    for slot in chosen:slots.put(slot)
    def job(pair):
        slot=slots.get()
        try:chain(*pair,slot,a)
        finally:slots.put(slot)
    jobs=[(arm,int(seed)) for seed in a.seeds.split(',') for arm in a.arms.split(',')]
    with ThreadPoolExecutor(max_workers=len(chosen)) as pool:list(pool.map(job,jobs))

if __name__=='__main__':main()
