"""HIDDEN comparison only. Never included in a prompt or a search seed.

Port of the existing hand-built HSI ADMM approach (ours_v5.py), with a fixed
penalty, overrelaxation, shared inverse and the same independent driver as the
generated candidates. This is an audit reference, not an LLM discovery.
"""
import numpy as np
import jax
import jax.numpy as jnp
from functools import partial

def contract():
    return {'dtype':'float64'}

def admit(meta):
    return 'gpu'

def width(meta):
    return {'chunk':128, 'unroll':1}

def formulate(D,Y,options):
    d,y = jnp.asarray(D),jnp.asarray(Y)
    h = d.T @ d
    rho = 1.0
    return {'M':jnp.linalg.inv(h+rho*jnp.eye(h.shape[0])), 'B':d.T@y,
            'rho':rho, 'p':D.shape[1], 'm':Y.shape[1]}

def initialize(problem,options):
    z=jnp.full((problem['p'],problem['m']),1.0/problem['p'])
    return {'x':z,'u':jnp.zeros_like(z)}

def project(v):
    u=jnp.sort(v,axis=0)[::-1]
    cs=jnp.cumsum(u,axis=0)-1
    k=jnp.arange(1,v.shape[0]+1)[:,None]
    r=jnp.sum(u-cs/k>0,axis=0)-1
    th=jnp.take_along_axis(cs,r[None,:],axis=0)[0]/(r+1)
    return jnp.maximum(v-th,0)

@partial(jax.jit,static_argnames=('steps','unroll'))
def kernel(z,u,M,B,rho,steps,unroll):
    def body(i,state):
        z,u=state
        x=M@(B+rho*(z-u))
        xa=1.6*x-.6*z
        zn=project(xa+u)
        return zn,u+xa-zn
    return jax.lax.fori_loop(0,steps,body,(z,u),unroll=unroll)

def advance(problem,state,steps,options):
    z,u=kernel(state['x'],state['u'],problem['M'],problem['B'],problem['rho'],steps,options['unroll'])
    return {'x':z,'u':u}

def restrict(problem,state,options):
    return problem,state

def handoff(problem,state,feedback,options):
    return problem,state

def solution(problem,state):
    return state['x']
