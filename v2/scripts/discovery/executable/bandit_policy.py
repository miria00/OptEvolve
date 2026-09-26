"""Online contextual bandit controlling the frozen LLM's proposal channel.

Actions are: query the model, mutate one legal gene of the current parent, or
sample a legal fresh genome. LinUCB updates only from executed numerical results.
This learns a small search policy; it does not fine-tune the 27B model.
"""
import json
import math
import random
import numpy as np
import recovery_space as R

ACTIONS=('model','mutate','random')
DIM=9


def context(parent, shapes, latest=None, gpu=True):
    shape=shapes[-1]
    D,Y=shape['D'],shape['Y']
    s=parent['spec']
    ratio=0.
    if latest:
        measurements=[r for r in latest.get('rows',[]) if 'step_s' in r]
        if measurements:
            ratio=np.median([r.get('certificate_s',0.)/(r['step_s']+1e-6) for r in measurements])
    return np.array([1., math.log1p(D[0])/7, math.log1p(D[1])/7,
        math.log1p(Y[1])/11, float(gpu),
        parent.get('score',{}).get('solved',0)/max(1,parent.get('score',{}).get('total',len(shapes))),
        min(2.,math.log1p(ratio)),float(s['algorithm']!='fbs'),
        float(s['certificate']=='device_screened')],float)


class LinUCB:
    def __init__(self, seed=0, alpha=.45, ridge=1.):
        self.A=np.stack([np.eye(DIM)*ridge for _ in ACTIONS])
        self.b=np.zeros((len(ACTIONS),DIM))
        self.alpha=float(alpha)
        self.rng=random.Random(seed)
        self.counts=[0]*len(ACTIONS)

    def choose(self,x):
        scores=[]
        for a in range(len(ACTIONS)):
            inverse=np.linalg.inv(self.A[a])
            mean=float(self.b[a]@inverse@x)
            uncertainty=math.sqrt(max(float(x@inverse@x),0.))
            scores.append(mean+self.alpha*uncertainty)
        top=max(scores)
        ties=[i for i,value in enumerate(scores) if abs(value-top)<1e-10]
        action=self.rng.choice(ties)
        return ACTIONS[action],scores

    def update(self,x,action,reward):
        if not 0<=reward<=1:raise ValueError('reward must be bounded')
        i=ACTIONS.index(action)
        self.A[i]+=np.outer(x,x)
        self.b[i]+=reward*x
        self.counts[i]+=1

    def dump(self):
        return {'method':'LinUCB','actions':ACTIONS,'feature_dim':DIM,
                'alpha':self.alpha,'A':self.A.tolist(),'b':self.b.tolist(),
                'counts':self.counts}

    @classmethod
    def restore(cls,value,seed):
        o=cls(seed,value['alpha']);o.A=np.array(value['A']);o.b=np.array(value['b']);o.counts=value['counts'];return o


def mutate(parent,rng):
    """Change one of the same exposed components without a target recipe."""
    s=json.loads(json.dumps(parent))
    genes=('algorithm','state','restrict','certificate','contract','admission',
           'chunk','unroll','step_fraction','resolvent_step','relaxation')
    for _ in range(64):
        p=json.loads(json.dumps(s));gene=rng.choice(genes)
        if gene=='algorithm':
            p['algorithm']=rng.choice([x for x in ('fbs','drs','admm') if x!=p['algorithm']])
            p['formulation']='forward' if p['algorithm']=='fbs' else 'resolvent'
            p['handoff']=0
            p['state']='none' if p['algorithm']=='fbs' else rng.choice(R.T.FACTORS)
        elif gene=='state':
            if p['algorithm']=='fbs':continue
            p['state']=rng.choice(R.T.FACTORS)
        elif gene=='restrict':p['restrict']='retire' if p['restrict']=='none' else 'none'
        elif gene=='certificate':p['certificate']='device_screened' if p['certificate']=='host' else 'host'
        elif gene=='contract':p['contract']='float32' if p['contract']=='float64' else 'float64'
        elif gene=='admission':p['admission']='cpu' if p['admission']=='gpu' else 'gpu'
        elif gene=='chunk':p['width']['chunk']=rng.choice(R.CHUNKS)
        elif gene=='unroll':p['width']['unroll']=rng.choice([1,2,4])
        elif gene=='step_fraction':p['step_fraction']=rng.choice([.25,.5,.9,1.5])
        elif gene=='resolvent_step':
            if p['algorithm']=='fbs':continue
            p['resolvent_step']=rng.choice(R.T.STEPS)
        elif gene=='relaxation':
            if p['algorithm']=='fbs':continue
            p['relaxation']=rng.choice(R.RELAXATIONS)
        try:
            q=R.canonical_spec(p)
            if q!=R.canonical_spec(s):return q,gene
        except ValueError:pass
    raise RuntimeError('could not find a different legal single-gene mutation')


def reward(rows,cap):
    """Fixed staged reward: independent solve count first, then wall time."""
    if not rows:return 0.
    accepted=[r for r in rows if r.get('accepted')]
    if len(accepted)==len(rows):
        elapsed=[max(1e-3,r['elapsed_s']) for r in rows]
        relative=np.mean([math.log(cap/t) for t in elapsed])
        return min(1.,.8+.2*max(0.,relative)/math.log(max(cap,math.e)))
    fraction=sum(r.get('fraction_certified',0.) for r in rows)/len(rows)
    return min(.3,.2*len(accepted)/len(rows)+.1*fraction)
