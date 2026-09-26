"""Executable textbook teaching examples, deliberately ordinary NumPy code.

These teach shapes and standard splitting equations. They contain no Cuprite
data, hardware policy, retirement, cached factorization, compilation, cadence
policy, or optimized solver implementation. Adaptation and measurement belong
to the proposer. Sources: Boyd et al. (2011), ADMM; Parikh and Boyd (2014),
Proximal Algorithms. For min 0.5*x.T*H*x-b.T*x + I_simplex(x), columns are RHS.
"""
import numpy as np


def simplex_projection(v):
    ordered=np.sort(v,axis=0)[::-1]
    cumulative=np.cumsum(ordered,axis=0)-1.
    count=np.arange(1,v.shape[0]+1)[:,None]
    rho=np.sum(ordered-cumulative/count>0,axis=0)-1
    threshold=cumulative[rho,np.arange(v.shape[1])]/(rho+1)
    return np.maximum(v-threshold,0.)


def quadratic_prox(v,H,b,step):
    # Correct but potentially costly: this example solves afresh on every call.
    return np.linalg.solve(np.eye(H.shape[0])+step*H,v+step*b)


def textbook_admm(H,b,z,rho,iterations):
    u=np.zeros_like(z)
    for iteration in range(iterations):
        x=quadratic_prox(z-u,H,b,1/rho)
        z_next=simplex_projection(x+u)
        u=u+x-z_next
        z=z_next
    return z


def textbook_drs(H,b,z,step,relaxation,iterations):
    for iteration in range(iterations):
        x=simplex_projection(z)
        q=quadratic_prox(2*x-z,H,b,step)
        z=z+relaxation*(q-x)
    return simplex_projection(z)
