"""One candidate, one case, fresh process. No remote model calls on timing GPUs."""
from __future__ import annotations
import argparse
import json
import hashlib
import os
from pathlib import Path
import time
import traceback
import numpy as np
from space import validate_source, source_id
from judge import certify

def evaluate(source, case, budget_s, tol):
    validate_source(source)
    import jax
    jax.config.update('jax_enable_x64', True)
    jax.config.update('jax_enable_compilation_cache', False)
    # Runtime initialization is common, outside the algorithm clock. The fresh
    # process wall time is also recorded by the launcher, and must not be hidden.
    available = jax.devices()
    data = np.load(case, allow_pickle=False)
    D, Y = data['D'], data['Y']
    D.flags.writeable = False
    Y.flags.writeable = False
    meta = {'n_rows': D.shape[0], 'n_features': D.shape[1], 'n_rhs': Y.shape[1],
            'gpu_name': str(available[0].device_kind)}
    t0 = time.perf_counter()
    ns = {'__name__': 'candidate'}
    exec(compile(source, '<candidate>', 'exec'), ns)
    # Prospective execution witnesses for trusted compiled composition sources.
    # These count Python dispatches, not hardware kernels or solver iterations.
    observed_calls = {}
    def observe(name, function):
        def wrapped(*args, **kwargs):
            observed_calls[name] = observed_calls.get(name, 0) + 1
            return function(*args, **kwargs)
        return wrapped
    for name in ('make_factor', 'admm_kernel', 'resolvent_kernel',
                 'forward_kernel', 'screen', 'retire'):
        if callable(ns.get(name)):
            ns[name] = observe(name, ns[name])
    contract = ns['contract']()
    if contract.get('dtype') not in ('float32', 'float64'):
        raise ValueError('contract dtype must be float32 or float64')
    device = ns['admit'](dict(meta))
    if device not in ('gpu', 'cpu'):
        raise ValueError('admit must return gpu or cpu')
    width = ns['width'](dict(meta))
    if any(type(width.get(k)) is not int or not 1 <= width[k] <= hi
           for k, hi in [('chunk', 8192), ('unroll', 16)]):
        raise ValueError('invalid chunk/unroll')
    options = {'tol': tol, 'budget_s': budget_s, 'dtype': contract['dtype'],
               'device': device, **width}
    trace = []
    iterations, cert_s, step_s = 0, 0.0, 0.0
    restrict_s, handoff_s, output_s = 0.0, 0.0, 0.0
    shape_history = []
    state_devices = set()
    minimum_active_rhs = Y.shape[1]
    certificate_calls = 0
    with jax.default_device(jax.devices(device)[0]):
        # The candidate receives copies: mutation of its inputs cannot change
        # the independent problem used by the judge.
        problem = ns['formulate'](D.copy(), Y.copy(), dict(options))
        state = ns['initialize'](problem, dict(options))
        jax.block_until_ready(state)
        setup_s = time.perf_counter() - t0
        result = {'accepted': False, 'worst_gap': 1e300, 'feasibility': 1e300,
                  'fraction_certified': 0.0}
        while time.perf_counter() - t0 < budget_s:
            ts = time.perf_counter()
            state = ns['advance'](problem, state, width['chunk'], dict(options))
            jax.block_until_ready(state)
            for value in jax.tree.leaves(state):
                if isinstance(value, jax.Array):
                    state_devices.update(d.platform for d in value.devices())
            if isinstance(problem, dict) and 'alive' in problem:
                minimum_active_rhs = min(minimum_active_rhs, len(problem['alive']))
            step_s += time.perf_counter() - ts
            shapes = [list(v.shape) for v in jax.tree.leaves(state) if hasattr(v,'shape')]
            if not shape_history or shapes != shape_history[-1]:
                shape_history.append(shapes)
            iterations += width['chunk']
            tc = time.perf_counter()
            X = np.asarray(ns['solution'](problem, state), dtype=np.float64)
            output_s += time.perf_counter() - tc
            tc = time.perf_counter()
            result = certify(D, Y, X, tol)
            certificate_calls += 1
            cert_s += time.perf_counter() - tc
            elapsed = time.perf_counter() - t0
            trace.append({'iteration': iterations, 'elapsed_s': elapsed, **result})
            if result['accepted'] or 'error' in result:
                break
            tr = time.perf_counter()
            problem, state = ns['restrict'](problem, state, dict(options))
            jax.block_until_ready((problem,state))
            restrict_s += time.perf_counter() - tr
            feedback = {'iteration': iterations, 'elapsed_s': elapsed,
                        'worst_gap': result['worst_gap'], 'feasibility': result['feasibility']}
            th = time.perf_counter()
            problem, state = ns['handoff'](problem, state, feedback, dict(options))
            jax.block_until_ready((problem,state))
            handoff_s += time.perf_counter() - th
        elapsed = time.perf_counter() - t0
    return {**result, 'accepted': result['accepted'] and elapsed <= budget_s,
            'elapsed_s': elapsed, 'setup_s': setup_s, 'step_s': step_s,
            'certificate_s': cert_s, 'iterations': iterations, 'trace': trace[-12:],
            'restriction_s':restrict_s, 'handoff_s':handoff_s, 'output_s':output_s,
            'state_shape_changes':max(0,len(shape_history)-1), 'state_shapes':shape_history,
            'observed_calls':observed_calls, 'certificate_calls':certificate_calls,
            'state_devices':sorted(state_devices), 'minimum_active_rhs':minimum_active_rhs,
            'options': options, 'meta': meta, 'source_id': source_id(source),
            'regime': 'fresh algorithm incl compilation; initialized JAX runtime'}

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', required=True)
    ap.add_argument('--case', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--budget', type=float, default=30)
    ap.add_argument('--tol', type=float, default=1e-4)
    a = ap.parse_args()
    t = time.perf_counter()
    try:
        result = evaluate(Path(a.source).read_text(), a.case, a.budget, a.tol)
    except Exception:
        result = {'accepted': False, 'elapsed_s': a.budget,
                  'error': traceback.format_exc()[-3500:]}
    result['worker_wall_s'] = time.perf_counter() - t
    result['source_sha256'] = hashlib.sha256(Path(a.source).read_bytes()).hexdigest()
    result['case_sha256'] = hashlib.sha256(Path(a.case).read_bytes()).hexdigest()
    Path(a.out).write_text(json.dumps(result, allow_nan=False))
    print(json.dumps({k:v for k,v in result.items() if k != 'trace'}), flush=True)
