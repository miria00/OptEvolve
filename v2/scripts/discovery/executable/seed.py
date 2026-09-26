"""Ordinary projected gradient starting point, not a hand-built winning solver."""
import numpy as np
import jax
import jax.numpy as jnp
from functools import partial

def contract():
    return {"dtype": "float64"}

def admit(meta):
    return "gpu"

def width(meta):
    return {"chunk": 128, "unroll": 1}

def formulate(D, Y, options):
    dtype = jnp.float64 if options['dtype'] == 'float64' else jnp.float32
    d = jnp.asarray(D, dtype=dtype)
    y = jnp.asarray(Y, dtype=dtype)
    h = d.T @ d
    return {'H': h, 'B': d.T @ y, 'L': jnp.maximum(jnp.linalg.norm(h, ord=jnp.inf), 1e-12),
            'p': D.shape[1], 'm': Y.shape[1]}

def initialize(problem, options):
    return {'x': jnp.full((problem['p'], problem['m']), 1.0 / problem['p'],
                          dtype=problem['H'].dtype)}

def project(v):
    u = jnp.sort(v, axis=0)[::-1]
    cs = jnp.cumsum(u, axis=0) - 1.0
    k = jnp.arange(1, v.shape[0] + 1)[:, None]
    r = jnp.sum(u - cs / k > 0, axis=0) - 1
    theta = jnp.take_along_axis(cs, r[None, :], axis=0)[0] / (r + 1)
    return jnp.maximum(v - theta, 0.0)

@partial(jax.jit, static_argnames=('steps', 'unroll'))
def kernel(x, h, b, L, steps, unroll):
    return jax.lax.fori_loop(0, steps,
        lambda i, z: project(z - (h @ z - b) / L), x, unroll=unroll)

def advance(problem, state, steps, options):
    return {'x': kernel(state['x'], problem['H'], problem['B'], problem['L'],
                        steps, options['unroll'])}

def restrict(problem, state, options):
    return problem, state

def handoff(problem, state, feedback, options):
    return problem, state

def solution(problem, state):
    return state['x']
