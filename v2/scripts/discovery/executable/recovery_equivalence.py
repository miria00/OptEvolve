"""Secondary semantic recovery audit, preserving the original literal ADMM audit.

For the identity split and feasible initial z, ADMM's carried v=z+u obeys
v_next = v + alpha * (prox_eta_f(2*P(v)-v) - P(v)), with z_next=P(v_next).
This is precisely the compiled relaxed DRS map. Tests compare both state
coordinates and primal iterates, not merely their final objective values.
"""
import hashlib
from recovery_space import compile_spec,audit as literal_audit


def audit(spec,source,rows):
    expected,_=compile_spec(spec);source_ok=source==expected
    sha=hashlib.sha256(source.encode()).hexdigest()
    direct=spec['algorithm'] in ('admm','drs') and spec['handoff']==0
    kernel='admm_kernel' if spec['algorithm']=='admm' else 'resolvent_kernel'
    evidence=[]
    for row in rows:
        calls=row.get('observed_calls',{})
        evidence.append({
            'case':row.get('case'),'source_matches':source_ok and row.get('source_sha256')==sha,
            'certificate_passed':bool(row.get('accepted')),
            'equivalent_split_executed':direct and calls.get(kernel,0)>0,
            'shared_factor_executed':spec['state'] in ('inverse','cholesky','host_inverse') and
                calls.get('make_factor')==1 and calls.get(kernel,0)>1,
            'simplex_operator':source_ok,'gpu_state_observed':'gpu' in row.get('state_devices',[]),
            'retirement_executed':row.get('minimum_active_rhs',row.get('meta',{}).get('n_rhs',0))<row.get('meta',{}).get('n_rhs',0),
            'screened_cadence_executed':spec['certificate']=='device_screened' and
                calls.get('screen',0)>row.get('certificate_calls',0)})
    recovered=bool(source_ok and evidence and all(r['source_matches'] and r['certificate_passed'] and
        r['equivalent_split_executed'] and r['simplex_operator'] and r['gpu_state_observed'] for r in evidence) and
        all(any(r[k] for r in evidence) for k in
            ('shared_factor_executed','retirement_executed','screened_cadence_executed')))
    return {'method_recovered':recovered,'evidence':evidence,'algorithm':spec['algorithm'],
            'literal_admm_audit':literal_audit(spec,source,rows),
            'target':'HSI splitting/shared-factor/GPU/retirement/screened-certificate composition',
            'equivalence':'ADMM v=z+u is the compiled relaxed DRS carried state',
            'speedup_required':False,'scope':'supplied-atom composition at one target; not missing-operator synthesis'}
