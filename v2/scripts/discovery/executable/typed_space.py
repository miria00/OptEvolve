"""A small executable, construction-gated eight-layer HSI composition space.

Only textbook FBS/relaxed DRS operators are compiled. No hand-built HSI solver
is a primitive. A mathematical construction certificate concerns exact atoms;
floating-point code still passes numerical admission and the external judge.
This family adapter does not implement the lasso D3-D6 transformations.
"""
from __future__ import annotations
from dataclasses import asdict
import json
import math
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

DEFAULT = {'formulation':'forward', 'algorithm':'fbs', 'state':'none',
           'restrict':'none', 'handoff':0, 'contract':'float64', 'admission':'gpu',
           'width':{'chunk':128,'unroll':1}, 'step_fraction':.9,
           'resolvent_step':1.0, 'relaxation':1.0}
CHUNKS = [16,64,128,256,1024]
STEPS = [.001,.01,.1,1.,10.,100.]
FACTORS = ['inverse','cholesky','recompute','host_inverse']

def check(spec):
    if not isinstance(spec,dict) or set(spec) != set(DEFAULT):
        raise ValueError('spec keys must be exactly ' + ', '.join(DEFAULT))
    choices = {'formulation':['forward','resolvent'], 'algorithm':['fbs','drs'],
               'state':['none',*FACTORS], 'restrict':['none','retire'],
               'handoff':[0,1,4,16], 'contract':['float64','float32'],
               'admission':['gpu','cpu']}
    for k,values in choices.items():
        if spec[k] not in values:
            raise ValueError(f'{k} must be one of {values}')
    if (spec['algorithm']=='fbs') != (spec['formulation']=='forward'):
        raise ValueError('fbs uses a forward quadratic; drs uses its resolvent')
    if spec['algorithm']=='drs' and spec['handoff']:
        raise ValueError('handoff is a finite FBS prefix into DRS; direct DRS has handoff=0')
    needs_factor = spec['algorithm']=='drs' or spec['handoff']>0
    if needs_factor != (spec['state']!='none'):
        raise ValueError('state must carry a factor policy iff a resolvent is used')
    w=spec['width']
    if not isinstance(w,dict) or set(w)!= {'chunk','unroll'} or w['chunk'] not in CHUNKS or w['unroll'] not in [1,2,4]:
        raise ValueError('width chunk in [16,64,128,256,1024], unroll in [1,2,4]')
    for k in ('step_fraction','resolvent_step','relaxation'):
        if isinstance(spec[k],bool) or not isinstance(spec[k],(int,float)) or not math.isfinite(spec[k]):
            raise ValueError('finite numerical parameters required')
    if (spec['step_fraction'] not in [.25,.5,.9,1.5] or spec['resolvent_step'] not in STEPS
            or spec['relaxation'] not in [.5,1.,1.5,1.9]):
        raise ValueError('parameters must belong to the shared LLM/random enumeration')
    from atlas.typing_.rules import typecheck_fbs,typecheck_drs,typecheck_relaxation
    # FBS step is a dimensionless fraction of a rigorously bounded L. The
    # per-instance compiler calculates L; the unit-L check verifies the ratio.
    certs=[asdict(typecheck_fbs(1.,spec['step_fraction']))]
    if needs_factor:
        certs.append(asdict(typecheck_relaxation(typecheck_drs(spec['resolvent_step']),spec['relaxation'])))
    return {'base_scheme_certificates':certs,
            'exact_atoms':'quadratic gradient/resolvent and Euclidean simplex projection',
            'scope':'base maps; finite-prefix handoff and pixel retirement require separate adapter checks',
            'numerical_tests_required':True}

def canonical_spec(spec):
    check(spec)
    out=json.loads(json.dumps(spec))
    if out['algorithm']=='drs':
        out['step_fraction']=.9
    elif out['handoff']==0:
        out['resolvent_step']=1.0
        out['relaxation']=1.0
    return out

def response_schema():
    """Constrain decoding to the same support used by random_spec.

    Cross-field validity is still checked by check(); the schema prevents
    format/type/enum errors without revealing a winning design.
    """
    enum=lambda xs:{'enum':xs}
    props={
        'formulation':enum(['forward','resolvent']), 'algorithm':enum(['fbs','drs']),
        'state':enum(['none',*FACTORS]),'restrict':enum(['none','retire']),
        'handoff':enum([0,1,4,16]),'contract':enum(['float32','float64']),
        'admission':enum(['cpu','gpu']),
        'width':{'type':'object','properties':{'chunk':enum(CHUNKS),'unroll':enum([1,2,4])},
                 'required':['chunk','unroll'],'additionalProperties':False},
        'step_fraction':enum([.25,.5,.9,1.5]),'resolvent_step':enum(STEPS),
        'relaxation':enum([.5,1.,1.5,1.9])}
    variants=[]
    for fixed in [
        {'algorithm':['fbs'],'formulation':['forward'],'state':['none'],'handoff':[0],
         'resolvent_step':[1.0],'relaxation':[1.0]},
        {'algorithm':['fbs'],'formulation':['forward'],'state':FACTORS,'handoff':[1,4,16]},
        {'algorithm':['drs'],'formulation':['resolvent'],'state':FACTORS,'handoff':[0],'step_fraction':[.9]},
    ]:
        pp=json.loads(json.dumps(props))
        pp.update({k:enum(v) for k,v in fixed.items()})
        variants.append({'type':'object','properties':pp,'required':list(pp),'additionalProperties':False})
    return {'type':'object','properties':{'rationale':{'type':'string'},
            'spec':{'oneOf':variants}},
            'required':['rationale','spec'],'additionalProperties':False}

def random_spec(rng):
    algorithm=rng.choice(['fbs','drs'])
    handoff=rng.choice([0,1,4,16]) if algorithm=='fbs' else 0
    return {'formulation':'forward' if algorithm=='fbs' else 'resolvent',
            'algorithm':algorithm,'state':rng.choice(FACTORS) if algorithm=='drs' or handoff else 'none',
            'restrict':rng.choice(['none','retire']), 'handoff':handoff,
            'contract':rng.choice(['float32','float64']), 'admission':rng.choice(['cpu','gpu']),
            'width':{'chunk':rng.choice(CHUNKS),'unroll':rng.choice([1,2,4])},
            'step_fraction':rng.choice([.25,.5,.9,1.5]),'resolvent_step':rng.choice(STEPS),
            'relaxation':rng.choice([.5,1.,1.5,1.9])}

SPACE_DESCRIPTION = '''Return JSON {"rationale":"short explanation", "spec":{...}}.
This is the complete executable eight-layer composition grammar:
1 formulation: "forward" uses the quadratic gradient; "resolvent" uses its exact prox.
2 algorithm: "fbs" requires forward; "drs" requires resolvent.
3 state: "none" for pure FBS; otherwise "inverse", "cholesky", "recompute", "host_inverse".
  These select carried linear algebra for quadratic proximal evaluations.
  host_inverse computes the shifted inverse using host float64 NumPy, then
  transfers it to the selected execution device/arithmetic. Its cost is measured.
4 restrict: "none" or "retire" (remove individually accepted RHS from active work,
  reconstruct the full batch and recheck all RHS externally).
5 handoff: 0,1,4,16. A positive value runs this many FBS chunks then switches to DRS
  from the current feasible point. Direct DRS must use 0. A handoff needs factor state.
6 contract: "float64" or "float32" internal arithmetic, external float64 judge unchanged.
7 admission: "cpu" or "gpu" actual execution device.
8 width: {"chunk":16|64|128|256|1024, "unroll":1|2|4}, compiled iteration grouping.
Also provide: step_fraction in [.25,.5,.9,1.5] for the FBS step fraction/L;
resolvent_step in [.001,.01,.1,1,10,100] for prox_(step*f);
relaxation in [.5,1,1.5,1.9] for DRS.
All keys must be supplied, even when a parameter is inactive. No arbitrary code.
The compiler implements the named standard operators. The model composes and prices
them for the problem/hardware; choosing a configuration is not invention of DRS.
'''

def compile_spec(spec):
    spec=canonical_spec(spec)
    admission=check(spec)
    source=TEMPLATE.replace('__SPEC_LITERAL__',repr(spec))
    from space import validate_source
    validate_source(source)
    return source,admission

TEMPLATE = '''import numpy as np
import jax
import jax.numpy as jnp
from functools import partial

def spec():
    return __SPEC_LITERAL__

def contract():
    return {'dtype':spec()['contract']}

def admit(meta):
    return spec()['admission']

def width(meta):
    return spec()['width']

def project(v):
    u=jnp.sort(v,axis=0)[::-1]
    cs=jnp.cumsum(u,axis=0)-1.0
    k=jnp.arange(1,v.shape[0]+1)[:,None]
    r=jnp.sum(u-cs/k>0,axis=0)-1
    th=jnp.take_along_axis(cs,r[None,:],axis=0)[0]/(r+1)
    return jnp.maximum(v-th,0.0)

def formulate(D,Y,options):
    dt=jnp.float64 if options['dtype']=='float64' else jnp.float32
    d,y=jnp.asarray(D,dtype=dt),jnp.asarray(Y,dtype=dt)
    h=d.T@d
    # A symmetric Gram's spectral norm is bounded by its absolute row norm.
    # Compute the bound in host float64 independently of the arithmetic gene.
    hf=D.T@D
    L=max(float(np.max(np.sum(np.abs(hf),axis=1)))*1.000001,1e-12)
    p,m=D.shape[1],Y.shape[1]
    return {'H':h,'B':d.T@y,'D':d,'Y':y,'L':L,'p':p,'m':m,
            'alive':np.arange(m),'full':np.full((p,m),1.0/p),
            'phase':spec()['algorithm'],'factor':None}

def make_factor(problem):
    h=problem['H']; eta=spec()['resolvent_step']
    if spec()['state']=='host_inverse':
        q=np.eye(problem['p'])+eta*np.asarray(h,dtype=np.float64)
        return jnp.asarray(np.linalg.inv(q),dtype=h.dtype)
    q=jnp.eye(problem['p'],dtype=h.dtype)+eta*h
    if spec()['state']=='cholesky':
        return jnp.linalg.cholesky(q)
    return jnp.linalg.inv(q)

def initialize(problem,options):
    x=jnp.full((problem['p'],problem['m']),1.0/problem['p'],dtype=problem['H'].dtype)
    if problem['phase']=='drs' and spec()['state']!='recompute':
        problem['factor']=make_factor(problem)
    return {'x':x,'z':x,'chunks':0}

@partial(jax.jit,static_argnames=('steps','unroll'))
def forward_kernel(x,H,B,step,steps,unroll):
    return jax.lax.fori_loop(0,steps,lambda i,v:project(v-step*(H@v-B)),x,unroll=unroll)

@partial(jax.jit,static_argnames=('steps','unroll','cholesky'))
def resolvent_kernel(z,B,factor,eta,relaxation,steps,unroll,cholesky):
    def body(i,z):
        x=project(z)
        rhs=2*x-z+eta*B
        if cholesky:
            q=jax.scipy.linalg.cho_solve((factor,True),rhs)
        else:
            q=factor@rhs
        return z+relaxation*(q-x)
    z=jax.lax.fori_loop(0,steps,body,z,unroll=unroll)
    return z,project(z)

def advance(problem,state,steps,options):
    s=spec()
    if problem['phase']=='fbs':
        x=forward_kernel(state['x'],problem['H'],problem['B'],s['step_fraction']/problem['L'],steps,options['unroll'])
        z=x
    else:
        factor=make_factor(problem) if s['state']=='recompute' else problem['factor']
        z,x=resolvent_kernel(state['z'],problem['B'],factor,s['resolvent_step'],s['relaxation'],steps,options['unroll'],s['state']=='cholesky')
    return {'x':x,'z':z,'chunks':state['chunks']+1}

def restrict(problem,state,options):
    if spec()['restrict']=='none' or len(problem['alive'])<=1:
        return problem,state
    x=state['x']; R=problem['D']@x-problem['Y']
    G=problem['D'].T@R
    gap=(jnp.sum(G*x,axis=0)-jnp.min(G,axis=0))/(1+0.5*jnp.sum(R*R,axis=0))
    keep=np.flatnonzero(np.asarray(gap)>options['tol']*0.25)
    if 0<len(keep)<.75*len(problem['alive']):
        # The driver's preceding solution() call restored float64 feasibility.
        # Preserve those checked values when freezing retired columns; copying
        # raw float32 state here would undo that restoration permanently.
        problem['alive']=problem['alive'][keep]
        problem['B']=problem['B'][:,keep]
        problem['Y']=problem['Y'][:,keep]
        state={**state,'x':state['x'][:,keep],'z':state['z'][:,keep]}
    return problem,state

def handoff(problem,state,feedback,options):
    threshold=spec()['handoff']
    if threshold and problem['phase']=='fbs' and state['chunks']>=threshold:
        problem['phase']='drs'
        if spec()['state']!='recompute':
            problem['factor']=make_factor(problem)
        state={**state,'z':state['x']}
    return problem,state

def solution(problem,state):
    x=np.asarray(state['x'],dtype=np.float64)
    # Numerical feasibility restoration is included in the measured clock.
    u=np.sort(x,axis=0)[::-1]
    cs=np.cumsum(u,axis=0)-1.0
    k=np.arange(1,x.shape[0]+1)[:,None]
    r=np.sum(u-cs/k>0,axis=0)-1
    th=cs[r,np.arange(x.shape[1])]/(r+1)
    problem['full'][:,problem['alive']]=np.maximum(x-th,0)
    return problem['full']
'''
