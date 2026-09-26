"""Prospective recovery assay: standard atoms, private method audit.

This is automatic composition of supplied operations, not invention of ADMM.
The new ADMM branch has numerical admission, not an Atlas proof certificate.
Old search grammars and recorded candidates remain unchanged.
"""
import ast
import copy
import json
import random
from pathlib import Path
import typed_space as T
from space import validate_source, apply_function_patch

DEFAULT={**T.DEFAULT,'certificate':'host'}
CHUNKS=[16,64,128,256,500,1024]
RELAXATIONS=[.5,1.,1.5,1.6,1.9]

def check(spec):
    if not isinstance(spec,dict) or set(spec)!=set(DEFAULT):
        raise ValueError('all recovery spec fields are required')
    if spec['certificate'] not in ('host','device_screened'):
        raise ValueError('certificate must be host or device_screened')
    if spec['algorithm'] not in ('fbs','drs','admm'):
        raise ValueError('unknown algorithm')
    if spec['width']['chunk'] not in CHUNKS or spec['relaxation'] not in RELAXATIONS:
        raise ValueError('invalid chunk or relaxation')
    base=copy.deepcopy(spec);base.pop('certificate')
    if base['algorithm']=='admm':base['algorithm']='drs'
    if base['width']['chunk']==500:base['width']['chunk']=256
    if base['relaxation']==1.6:base['relaxation']=1.5
    T.check(base) # Shared cross-field and primitive parameter checks only.
    return {'scope':'standard operator composition plus independent numerical admission',
            'atlas_admm_proof':False,
            'formal_claim':'No new end-to-end convergence theorem is asserted by this adapter.',
            'standard_admm_conditions':{'rho_positive':spec['resolvent_step']>0,
                                        'relaxation_in_0_2':0<spec['relaxation']<2}}

def canonical_spec(spec):
    check(spec);s=copy.deepcopy(spec)
    if s['algorithm']!='fbs':s['step_fraction']=.9
    elif s['handoff']==0:s.update(resolvent_step=1.,relaxation=1.)
    return s

def response_schema():
    schema=T.response_schema()
    variants=schema['properties']['spec']['oneOf']
    variants.append(copy.deepcopy(variants[-1]))
    variants[-1]['properties']['algorithm']={'enum':['admm']}
    for v in variants:
        v['properties']['certificate']={'enum':['host','device_screened']}
        v['required'].append('certificate')
        v['properties']['width']['properties']['chunk']={'enum':CHUNKS}
        if v['properties']['relaxation']['enum']!=[1.0]:
            v['properties']['relaxation']={'enum':RELAXATIONS}
    return schema

def random_spec(rng):
    s=T.random_spec(rng)
    algorithm=rng.choice(['fbs','drs','admm'])
    s['algorithm']=algorithm;s['formulation']='forward' if algorithm=='fbs' else 'resolvent'
    if algorithm!='fbs':s['handoff']=0
    s['state']=rng.choice(T.FACTORS) if algorithm!='fbs' or s['handoff'] else 'none'
    s['width']['chunk']=rng.choice(CHUNKS)
    s['relaxation']=rng.choice(RELAXATIONS)
    s['certificate']=rng.choice(['host','device_screened'])
    return canonical_spec(s)

SPACE_DESCRIPTION=T.SPACE_DESCRIPTION+'''
Recovery-space extension (same support for every experimental arm):
algorithm also permits "admm", using a standard equality split of the quadratic
and simplex indicator, with scaled dual state. It requires formulation=resolvent,
handoff=0, and a factor policy. resolvent_step=1/rho. Relaxation also allows 1.6.
width.chunk also allows 500. Add required field certificate: "host" or
"device_screened". The latter runs cheap device residual checks between compiled
blocks, requesting the independent host certificate near tolerance or after at
most 20 blocks. It never accepts a solution itself. restrict=retire is available
with all engines and both schedules; outputs always reconstruct the entire batch.
The private paper-reference implementation and private recovery audit are absent
from the prompt. Choose operations using measured execution feedback.
'''

# Keep the independently tested primitive projection and FBS/DRS operators.
# Explicit function replacement avoids duplicate definitions in emitted code.
REPLACEMENTS='''
def formulate(D,Y,options):
    dt=jnp.float64 if options['dtype']=='float64' else jnp.float32
    d,y=jnp.asarray(D,dtype=dt),jnp.asarray(Y,dtype=dt)
    hf=D.T@D
    L=max(float(np.max(np.sum(np.abs(hf),axis=1)))*1.000001,1e-12)
    p,m=D.shape[1],Y.shape[1]
    return {'H':d.T@d,'B':d.T@y,'D':d,'Y':y,'yy':jnp.sum(y*y,axis=0),
            'L':L,'p':p,'m':m,'alive':np.arange(m),'full':np.full((p,m),1.0/p),
            'phase':spec()['algorithm'],'factor':None}

def initialize(problem,options):
    x=jnp.full((problem['p'],problem['m']),1.0/problem['p'],dtype=problem['H'].dtype)
    if problem['phase'] in ('drs','admm') and spec()['state']!='recompute':
        problem['factor']=make_factor(problem)
    return {'x':x,'z':x,'u':jnp.zeros_like(x),'chunks':0,'actual_iterations':0}

@partial(jax.jit,static_argnames=('steps','unroll','cholesky'))
def admm_kernel(z,u,B,factor,eta,relaxation,steps,unroll,cholesky):
    def body(i,state):
        z,u=state
        rhs=z-u+eta*B
        if cholesky:
            x=jax.scipy.linalg.cho_solve((factor,True),rhs)
        else:
            x=factor@rhs
        xa=relaxation*x+(1-relaxation)*z
        zn=project(xa+u)
        return zn,u+xa-zn
    return jax.lax.fori_loop(0,steps,body,(z,u),unroll=unroll)

@jax.jit
def screen(x,H,B,yy):
    hx=H@x;g=hx-B
    f=.5*(jnp.sum(x*hx,axis=0)-2*jnp.sum(B*x,axis=0)+yy)
    return (jnp.sum(g*x,axis=0)-jnp.min(g,axis=0))/(1+jnp.abs(f))

def advance_block(problem,state,steps,options):
    s=spec();u=state['u']
    if problem['phase']=='fbs':
        x=forward_kernel(state['x'],problem['H'],problem['B'],s['step_fraction']/problem['L'],steps,options['unroll'])
        z=x
    else:
        factor=make_factor(problem) if s['state']=='recompute' else problem['factor']
        if problem['phase']=='admm':
            x,u=admm_kernel(state['x'],state['u'],problem['B'],factor,s['resolvent_step'],s['relaxation'],steps,options['unroll'],s['state']=='cholesky')
            z=x
        else:
            z,x=resolvent_kernel(state['z'],problem['B'],factor,s['resolvent_step'],s['relaxation'],steps,options['unroll'],s['state']=='cholesky')
    return {'x':x,'z':z,'u':u,'chunks':state['chunks']+1,'actual_iterations':state['actual_iterations']+steps}

def retire(problem,state,gap,options):
    if spec()['restrict']=='none' or len(problem['alive'])<=1:return state
    keep=np.flatnonzero(gap>options['tol']*.01)
    if 0<len(keep)<=.7*len(problem['alive']):
        # Save all current values before removing any column, even when called
        # inside advance before the driver's first solution() call.
        solution(problem,state)
        problem['alive']=problem['alive'][keep]
        problem['B']=jnp.take(problem['B'],keep,axis=1)
        problem['Y']=jnp.take(problem['Y'],keep,axis=1)
        problem['yy']=jnp.take(problem['yy'],keep)
        state={**state,'x':jnp.take(state['x'],keep,axis=1),
               'z':jnp.take(state['z'],keep,axis=1),'u':jnp.take(state['u'],keep,axis=1)}
    return state

def advance(problem,state,steps,options):
    blocks=20 if spec()['certificate']=='device_screened' else 1
    for block in range(blocks):
        state=advance_block(problem,state,steps,options)
        # A finite handoff counts actual compiled blocks, independently of the
        # external certificate cadence.
        problem,state=handoff(problem,state,{},options)
        if spec()['certificate']=='device_screened':
            gap=np.asarray(screen(state['x'],problem['H'],problem['B'],problem['yy']))
            if float(np.max(gap))<=2*options['tol'] or block==blocks-1:return state
            state=retire(problem,state,gap,options)
    return state

def restrict(problem,state,options):
    if spec()['restrict']=='retire':
        gap=np.asarray(screen(state['x'],problem['H'],problem['B'],problem['yy']))
        state=retire(problem,state,gap,options)
    return problem,state

def solution(problem,state):
    x=np.asarray(state['x'],dtype=np.float64)
    if spec()['contract']=='float32':
        # Restore simplex feasibility after float32 arithmetic, inside timing.
        u=np.sort(x,axis=0)[::-1];cs=np.cumsum(u,axis=0)-1.0
        k=np.arange(1,x.shape[0]+1)[:,None]
        r=np.sum(u-cs/k>0,axis=0)-1
        th=cs[r,np.arange(x.shape[1])]/(r+1)
        x=np.maximum(x-th,0)
    problem['full'][:,problem['alive']]=x
    return problem['full']
'''

def compile_spec(spec):
    spec=canonical_spec(spec)
    admission=check(spec)
    source=T.TEMPLATE.replace('__SPEC_LITERAL__',repr(spec))
    tree=ast.parse(REPLACEMENTS)
    replacements={n.name:ast.unparse(n) for n in tree.body}
    source=apply_function_patch(source,replacements)
    validate_source(source)
    return source,admission

def audit(spec,source,rows):
    """Private audit: compiled semantics PLUS observed execution, never rationale.

    A single-target structural recovery does not establish equivalence to the
    paper's multi-target timing protocol or newly synthesized missing operators.
    """
    import hashlib
    expected,_=compile_spec(spec)
    source_ok=source==expected
    sha=hashlib.sha256(source.encode()).hexdigest()
    evidence=[]
    for row in rows:
        calls=row.get('observed_calls',{})
        evidence.append({
            'case':row.get('case'),'source_matches':row.get('source_sha256')==sha,
            'certificate_passed':bool(row.get('accepted')),
            'admm_executed':spec['algorithm']=='admm' and calls.get('admm_kernel',0)>0,
            'shared_factor_executed':spec['state'] in ('inverse','cholesky','host_inverse') and
                calls.get('make_factor')==1 and calls.get('admm_kernel',0)>1,
            'simplex_operator':source_ok,
            'gpu_state_observed':'gpu' in row.get('state_devices',[]),
            'retirement_executed':row.get('minimum_active_rhs',row.get('meta',{}).get('n_rhs',0))<row.get('meta',{}).get('n_rhs',0),
            'screened_cadence_executed':spec['certificate']=='device_screened' and
                calls.get('screen',0)>row.get('certificate_calls',0),
        })
    # Retirement and sparse certification need at least one nontrivial case to
    # exercise them; requiring them on identity/zero cases is meaningless.
    recovered=bool(source_ok and evidence and all(r['source_matches'] and r['certificate_passed'] and
        r['admm_executed'] and r['simplex_operator'] and r['gpu_state_observed'] for r in evidence) and
        all(any(r[k] for r in evidence) for k in
            ('shared_factor_executed','retirement_executed','screened_cadence_executed')))
    return {'method_recovered':recovered,'evidence':evidence,
            'target':'paper HSI single-target ADMM/shared-factor/GPU/retirement/screened-certificate composition',
            'speedup_required':False,'scope':'supplied-atom composition; not invention of ADMM or lasso D3-D6'}
