"""Hidden single-target adaptation of hsi/ours_v5.py, including its schedule.

This is a hand-built comparison, NEVER a proposer seed or teaching card. Unlike
audit_oracle.py it includes device prechecks, pixel retirement, and sparse host
certification. The paper's multi-target ladder is reduced to the driver's one
fixed target. Original solver source and the original timing regime are distinct.
"""
import numpy as np
import jax
import jax.numpy as jnp
from functools import partial


def contract():return {'dtype':'float64'}
def admit(meta):return 'gpu'
def width(meta):return {'chunk':500,'unroll':1}


def formulate(D,Y,options):
    d,y=jnp.asarray(D),jnp.asarray(Y)
    H=d.T@d;B=d.T@y
    return {'H':H,'Bfull':B,'yyfull':jnp.sum(y*y,axis=0),
            'M':jnp.linalg.inv(H+jnp.eye(H.shape[0])),
            'p':D.shape[1],'m':Y.shape[1]}


def initialize(problem,options):
    x=jnp.full((problem['p'],problem['m']),1./problem['p'])
    return {'x':x,'u':jnp.zeros_like(x),'alive':np.arange(problem['m']),
            'B':problem['Bfull'],'yy':problem['yyfull'],
            'full':np.full((problem['p'],problem['m']),1./problem['p']),
            'actual_iterations':0}


def project(v):
    u=jnp.sort(v,axis=0)[::-1];cs=jnp.cumsum(u,axis=0)-1.
    n=v.shape[0];k=jnp.arange(1,n+1,dtype=v.dtype)[:,None]
    r=n-1-jnp.argmax((u-cs/k>0)[::-1],axis=0)
    theta=jnp.take_along_axis(cs,r[None,:],axis=0)[0]/(r+1)
    return jnp.maximum(v-theta,0.)


@partial(jax.jit,static_argnames=('steps',))
def kernel(z,u,M,B,steps):
    def body(i,state):
        z,u=state
        x=M@(B+z-u);xa=1.6*x-.6*z
        zn=project(xa+u)
        return zn,u+xa-zn
    return jax.lax.fori_loop(0,steps,body,(z,u))


@jax.jit
def device_gap(x,H,B,yy):
    hx=H@x;g=hx-B
    objective=.5*(jnp.sum(x*hx,axis=0)-2*jnp.sum(B*x,axis=0)+yy)
    return (jnp.sum(g*x,axis=0)-jnp.min(g,axis=0))/(1+jnp.abs(objective))


def advance(problem,state,steps,options):
    # As in ours_v5, request host confirmation near the target or once per
    # twenty blocks. Every returned full batch is judged independently by the
    # driver, including all retired columns. The program cannot accept itself.
    for block in range(20):
        x,u=kernel(state['x'],state['u'],problem['M'],state['B'],steps)
        state={**state,'x':x,'u':u,'actual_iterations':state['actual_iterations']+steps}
        gap=np.asarray(device_gap(x,problem['H'],state['B'],state['yy']))
        if float(gap.max())<=2*options['tol'] or block==19:
            return state
        keep=np.flatnonzero(gap>options['tol']*.01)
        if 0<len(keep)<=.7*len(state['alive']):
            state['full'][:,state['alive']]=np.asarray(x,dtype=np.float64)
            alive=state['alive'][keep]
            state={**state,'alive':alive,'x':jnp.take(x,keep,axis=1),
                   'u':jnp.take(u,keep,axis=1),
                   'B':jnp.take(problem['Bfull'],alive,axis=1),
                   'yy':jnp.take(problem['yyfull'],alive)}
    return state


def restrict(problem,state,options):return problem,state
def handoff(problem,state,feedback,options):return problem,state


def solution(problem,state):
    state['full'][:,state['alive']]=np.asarray(state['x'],dtype=np.float64)
    return state['full']
