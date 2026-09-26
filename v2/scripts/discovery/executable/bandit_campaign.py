"""Prospective, on-policy contextual-bandit search over executable solver moves.

The frozen LLM, a legal one-gene mutation, and a fresh random genome are the
three actions. LinUCB sees only training execution feedback. Every distinct
candidate consumes one fresh evaluation regardless of which action produced it.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import shutil
import time

import numpy as np

import bandit_policy as B
import campaign as C
from judge import aggregate
import recovery_campaign as RC
import recovery_space as R

DECONV_TASK = '''For each column y of Y solve
    minimize 0.5 * ||D x - y||_2^2 subject to x >= 0 and sum(x) = 1.
D is shared across columns and is a gene-expression signature matrix; Y contains
synthetic bulk immune profiles. This is the same simplex least-squares objective
as spectral unmixing but a different application, scale, and design matrix.
The fixed acceptance bar on EVERY column is independent host float64
Frank-Wolfe gap <= 1e-4, feasibility error <= 1e-9. Setup, compilation,
transfers, iterations, output reconstruction, and checking count toward wall
time on an A100. Return the full coefficient matrix, not a subset.
Compose a legal algorithm, factorization, state, restriction, precision,
device, work width, and certificate schedule. Supply an executable design.
'''


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def training_context(data, cases, cap):
    answer = {'budget_seconds_per_case': cap, 'cases': []}
    for case in cases:
        with np.load(Path(data) / (case + '.npz'), allow_pickle=False) as d:
            answer['cases'].append({'case': case, 'D': list(d['D'].shape),
                                    'Y': list(d['Y'].shape)})
    return answer


def feedback(archive):
    fields = ('case', 'accepted', 'elapsed_s', 'worst_gap', 'feasibility',
              'iterations', 'setup_s', 'step_s', 'certificate_s', 'error')
    return [{'spec': r['spec'], 'score': r['score'],
             'rows': [{k: row[k] for k in fields if k in row}
                      for row in r['rows']]} for r in archive[-8:]]


def chain(args, slot):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    cases = args.cases.split(',')
    spec0 = R.canonical_spec(R.DEFAULT)
    parent = {'spec': spec0, 'rows': [], 'score': {'solved': 0,
              'total': len(cases), 'penalized_geomean_s': args.budget}}
    archive = []
    seen = set()
    model_rng = random.Random(33000 + args.seed)
    mutation_rng = random.Random(53000 + args.seed)
    random_rng = random.Random(73000 + args.seed)
    initial_bytes = Path(args.initial_policy).read_bytes() if args.initial_policy else None
    if initial_bytes is not None:
        (out / 'initial_policy.json').write_bytes(initial_bytes)
    policy = (B.LinUCB.restore(json.loads(initial_bytes), args.seed)
              if initial_bytes is not None else B.LinUCB(args.seed))
    context = training_context(args.data, cases, args.budget)
    if args.task == 'deconv':
        RC.TASK = DECONV_TASK
    shapes = context['cases']
    src0, _ = R.compile_spec(spec0)
    (out / 'seed.py').write_text(src0)
    C.write_json(out / 'seed.json', parent)
    C.write_json(out / 'protocol.json', {
        'args': vars(args), 'slot': slot, 'training_context': context,
        'space_sha256': sha(Path(__file__).with_name('recovery_space.py')),
        'policy_sha256': sha(Path(__file__).with_name('bandit_policy.py')),
        'runner_sha256': sha(__file__),
        'initial_policy_sha256': (hashlib.sha256(initial_bytes).hexdigest()
                                  if initial_bytes is not None else None),
        'data_manifest_sha256': sha(Path(args.data) / 'manifest.json'),
        'primary_target': 'private method recovery within eight fresh candidates',
        'policy_reward': 'training numerical solve fraction, then measured time',
        'fresh_budget': args.evaluations,
        'admission_counts_as_fresh_candidate': True,
        'proposal_errors_and_duplicates_not_free': 'logged; maximum four attempts per fresh slot',
        'model_weights': 'frozen Qwen 27B FP8 on separate H100',
        'source_freeze': 'training cases only on worker; heldout never deployed'})
    snap = out / 'source_snapshot'; snap.mkdir()
    for p in C.HERE.glob('*.py'):
        shutil.copy2(p, snap / p.name)
    C.deploy(args.data, out)
    fresh = attempts = model_calls = 0
    while fresh < args.evaluations and attempts < 4 * args.evaluations:
        attempts += 1
        od = out / f'p{attempts:03d}'
        od.mkdir()
        x = B.context(parent, shapes, archive[-1] if archive else None)
        action, scores = policy.choose(x) if args.policy == 'linucb' else ('model', [])
        rec = {'proposal': attempts, 'action': action, 'features': x.tolist(),
               'ucb': scores, 'parent': parent['spec'], 'policy_counts_before': policy.counts[:],
               'seed': args.seed}
        try:
            if action == 'model':
                model_calls += 1
                raw, rationale, meta = RC.proposal(parent, feedback(archive),
                    'withheld_cards', model_rng, od, args.endpoint, True, context)
                spec = R.canonical_spec(raw)
                rec.update(llm=meta, rationale=rationale)
            elif action == 'mutate':
                spec, gene = B.mutate(parent['spec'], mutation_rng)
                rec['mutated_gene'] = gene
            else:
                spec = R.random_spec(random_rng)
            sid = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
            rec.update(spec=spec, id=sid)
            if sid in seen:
                rec['duplicate'] = True
                C.write_json(od / 'record.json', rec)
                continue
            seen.add(sid)
            code, admission = R.compile_spec(spec)
            (od / 'candidate.py').write_text(code)
            C.write_json(od / 'construction.json', admission)
            fresh += 1
            rec['evaluation'] = fresh
            gates = C.evaluate_on_slot(slot, od / 'candidate.py',
                ['admission/identity', 'admission/rank_deficient', 'admission/zero'], od, 8)
            rec['admission'] = gates
            if all(row.get('accepted') for row in gates):
                rows = C.evaluate_on_slot(slot, od / 'candidate.py', cases, od, args.budget)
            else:
                rows = [{'case': case, 'accepted': False, 'elapsed_s': args.budget}
                        for case in cases]
                rec['gate_error'] = [row for row in gates if not row.get('accepted')]
            rec.update(rows=rows, score=aggregate(rows, args.budget))
            rec['recovery_audit'] = R.audit(spec, code, rows)
            rec['reward'] = B.reward(rows, args.budget)
            if args.policy == 'linucb':
                policy.update(x, action, rec['reward'])
            archive.append(rec)
            parent = max(archive, key=lambda r: RC.parent_rank(r, True))
            if rec['recovery_audit']['method_recovered'] and not (out / 'first_recovery.json').exists():
                C.write_json(out / 'first_recovery.json', rec)
                (out / 'recovered.py').write_text(code)
                print(json.dumps({'METHOD_RECOVERED': fresh, 'source': str(od / 'candidate.py')}), flush=True)
            C.write_json(od / 'record.json', rec)
            print(json.dumps({'evaluation': fresh, 'action': action, 'solved': rec['score']['solved'],
                              'reward': rec['reward'], 'recovered': rec['recovery_audit']['method_recovered']}), flush=True)
        except Exception as exc:
            rec['proposal_error'] = str(exc)[-2500:]
            C.write_json(od / 'record.json', rec)
            print(json.dumps({'attempt': attempts, 'error': rec['proposal_error'][-300:]}), flush=True)
        C.write_json(out / 'policy.json', policy.dump())
        C.write_json(out / 'status.json', {'complete': False, 'evaluated': fresh,
                     'proposals': attempts, 'model_calls': model_calls,
                     'action_counts': policy.counts, 'first_recovery': (out / 'first_recovery.json').exists(),
                     'best': parent})
    code, _ = R.compile_spec(parent['spec'])
    (out / 'winner.py').write_text(code)
    C.write_json(out / 'winner.json', parent)
    C.write_json(out / 'status.json', {'complete': True, 'evaluated': fresh,
                 'proposals': attempts, 'model_calls': model_calls,
                 'action_counts': policy.counts, 'first_recovery': (out / 'first_recovery.json').exists(),
                 'best': parent})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--remote-root', required=True)
    ap.add_argument('--slot', type=int, required=True)
    ap.add_argument('--cases', default='train/cuprite4096,train/cuprite16384,train/synthetic96')
    ap.add_argument('--budget', type=float, default=120)
    ap.add_argument('--evaluations', type=int, default=8)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--initial-policy')
    ap.add_argument('--policy', choices=('linucb', 'pure'), default='linucb')
    ap.add_argument('--task', choices=('hsi', 'deconv'), default='hsi')
    ap.add_argument('--endpoint', default='http://127.0.0.1:8011')
    a = ap.parse_args()
    if any(not c.startswith('train/') or c.count('/') != 1 or '..' in c for c in a.cases.split(',')):
        raise ValueError('only training cases may enter the policy')
    if not a.remote_root.startswith('/root/optevolve_discovery_'):
        raise ValueError('remote root must be isolated')
    C.REMOTE = a.remote_root
    slots = [(h, p, g) for h, p, n in C.NODES for g in range(n)]
    chain(a, slots[a.slot])


if __name__ == '__main__':
    main()
