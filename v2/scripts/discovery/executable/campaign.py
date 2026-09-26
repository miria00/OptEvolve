"""Executable LLM outer loop with isolated fleet workers and immutable lineage.

No model access to the repository, oracle implementations, validation or test data.
Each chain has its own RNG, archive, prompts, outputs and GPU. No shared output file.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
from pathlib import Path
import queue
import random
import shlex
import subprocess
import time
import urllib.request
from space import (API, TASK, REFERENCES, HSI_CARDS, PATCH_API, apply_function_patch,
                   validate_source, source_id, layer_ids)
from judge import aggregate

HERE = Path(__file__).resolve().parent
REMOTE = '/root/optevolve_discovery_20260926'
KEY = '/home/miria/.ssh/id_ed25519'
NODES = [('154.54.102.49', 17170, 4), ('154.54.102.42', 13024, 2)]
ARMS = ['full_cards', 'withheld_cards', 'no_references', 'resample']

def run(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if r.returncode:
        raise RuntimeError(f'command exit {r.returncode}: {r.stderr[-2000:]} {r.stdout[-1000:]}')
    return r.stdout

def ssh(host, port, command, timeout=180):
    return run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', '-p', str(port),
                '-i', KEY, 'root@' + host, command], timeout=timeout)

def copy_to(host, port, local, remote):
    # Copy is idempotent, unlike running a timed candidate. Retry only transport
    # here; never silently execute a candidate twice after an ambiguous timeout.
    command=['scp','-q','-o','BatchMode=yes','-o','ConnectTimeout=10',
             '-o','ServerAliveInterval=10','-o','ServerAliveCountMax=3',
             '-P',str(port),'-i',KEY,str(local),f'root@{host}:{remote}']
    for attempt in range(3):
        try:return run(command,timeout=90)
        except (RuntimeError,subprocess.TimeoutExpired):
            if attempt==2:raise
            time.sleep(1)

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    tmp.replace(path)

def deploy(data, output):
    import tarfile
    bundle = Path(output) / 'worker_bundle.tar.gz'
    with tarfile.open(bundle, 'w:gz') as tar:
        for name in ('worker.py', 'space.py', 'judge.py', 'seed.py'):
            tar.add(HERE / name, arcname=name)
        # Test and validation never go to a search worker at this stage.
        for split in ('train', 'admission'):
            for p in sorted((Path(data) / split).glob('*.npz')):
                tar.add(p, arcname=f'data/{split}/{p.name}')
    for host, port, _ in NODES:
        ssh(host, port, f'mkdir -p {REMOTE}/jobs')
        copy_to(host, port, bundle, REMOTE + '/bundle.tar.gz')
        ssh(host, port, f'tar -xzf {REMOTE}/bundle.tar.gz -C {REMOTE}')

def evaluate_on_slot(slot, code_path, cases, outdir, cap_s):
    host, port, gpu = slot
    source_sha = hashlib.sha256(code_path.read_bytes()).hexdigest()
    key = hashlib.sha256(json.dumps([str(code_path.resolve()),str(Path(outdir).resolve()),
                         host,port,gpu,source_sha,cap_s]).encode()).hexdigest()[:24]
    remote_src = f'{REMOTE}/jobs/{key}.py'
    copy_to(host, port, code_path, remote_src)
    rows = []
    # Confirmation queues can share a slot with longer full-image checks.
    # Queue waiting is outside the worker's algorithm clock and is recorded
    # separately in remote_call_wall_s. Never time out merely behind a valid
    # longer job, and never overlap GPU execution to avoid that wait.
    lock_wait_s = max(900, int(cap_s + 120))
    for case in cases:
        split, name = case.split('/')
        remote_out = f'{REMOTE}/jobs/{key}_{split}_{name}.json'
        command = shlex.join(['flock','--exclusive','--wait',str(lock_wait_s),
            f'/tmp/optevolve_gpu_{gpu}.lock',
            'timeout', '--signal=TERM', '--kill-after=10', str(int(cap_s + 90)),
            'env', '-u', 'LD_LIBRARY_PATH', f'CUDA_VISIBLE_DEVICES={gpu}',
            'JAX_PLATFORMS=cuda,cpu', 'XLA_PYTHON_CLIENT_PREALLOCATE=false',
            'OPENBLAS_NUM_THREADS=2', 'OMP_NUM_THREADS=2', 'MKL_NUM_THREADS=2',
            'ATLAS_JAX_CACHE_DIR=', '/root/jaxenv/bin/python', REMOTE + '/worker.py',
            '--source', remote_src, '--case', REMOTE + '/data/' + case + '.npz',
            '--budget', str(cap_s), '--out', remote_out])
        start = time.perf_counter()
        try:
            ssh(host, port, command, timeout=lock_wait_s + cap_s + 150)
            row = json.loads(ssh(host, port, 'cat ' + remote_out))
            if row.get('source_sha256') != source_sha:
                raise ValueError('worker source hash differs from the submitted candidate')
        except Exception as exc:
            row = {'accepted': False, 'elapsed_s': cap_s, 'error': str(exc)[-2000:]}
        row.update(case=case, host=host, gpu=gpu, remote_call_wall_s=time.perf_counter()-start)
        rows.append(row)
        write_json(Path(outdir) / (split + '_' + name + '.json'), row)
    return rows

def summarize_feedback(record):
    return {k:record[k] for k in ('id', 'rationale', 'score', 'gate_error', 'rows') if k in record}

def rank(record):
    s = record['score']
    # Partial progress only breaks all-failure ties during search. It is never
    # counted as an accepted solution or a speedup in results.
    progress = sum(r.get('fraction_certified', 0) for r in record.get('rows', []))
    gaps = [r.get('worst_gap', 1e6) for r in record.get('rows', [])]
    return (s['solved'], -s['penalized_geomean_s'], progress,
            -sum(math.log(max(g, 1e-15)) for g in gaps))

def llm(prompt, endpoint, seed, output, patch_parent=None):
    body = {'model': 'Qwen/Qwen3.8-27B', 'messages': [
        {'role': 'system', 'content': 'You are a numerical algorithm researcher. Return the requested JSON with complete executable code. Use execution evidence to improve mathematical and computational structure.'},
        {'role': 'user', 'content': prompt}], 'temperature': .8, 'top_p': .95,
        'seed': seed, 'max_tokens': 12000,
        'chat_template_kwargs': {'enable_thinking': False},
        'response_format': {'type': 'json_object'}}
    write_json(output / 'request.json', body)
    t = time.perf_counter()
    req = urllib.request.Request(endpoint.rstrip('/') + '/v1/chat/completions',
        data=json.dumps(body).encode(), headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req, timeout=600) as response:
        raw = json.load(response)
    write_json(output / 'response.json', raw)
    choice = raw['choices'][0]
    if choice['finish_reason'] == 'length' or not choice['message'].get('content'):
        raise ValueError('truncated or empty model answer; no silent salvage')
    parsed = json.loads(choice['message']['content'])
    if patch_parent is not None:
        parsed['code'] = apply_function_patch(patch_parent, parsed['replacements'])
    validate_source(parsed['code'])
    return parsed, {'seconds':time.perf_counter()-t, 'usage':raw.get('usage'), 'model':raw.get('model')}

def chain(arm, seed, slot, args):
    directory = Path(args.out) / f'{arm}_s{seed}'
    directory.mkdir(parents=True, exist_ok=False)
    rng = random.Random(81000 + seed)
    parent_rng = random.Random(91000 + seed)
    baseline = HERE / 'seed.py'
    train = ['train/cuprite256', 'train/synthetic96']
    admission = ['admission/identity', 'admission/rank_deficient', 'admission/zero']
    write_json(directory / 'config.json', {'arm':arm, 'seed':seed, 'slot':slot,
        'fresh_evaluations':args.evaluations, 'siblings':2, 'budget_s':args.budget,
        'train':train, 'admission':admission})
    base_rows = evaluate_on_slot(slot, baseline, train, directory / 'baseline', args.budget)
    parent = {'id': source_id(baseline.read_text()), 'code':baseline.read_text(),
              'rationale':'ordinary projected gradient seed', 'rows':base_rows,
              'score':aggregate(base_rows, args.budget)}
    seed_parent = parent
    archive = [parent]
    seen = {parent['id']}
    proposals, evaluated = 0, 0
    while evaluated < args.evaluations and proposals < 4 * args.evaluations:
        selected = seed_parent if arm == 'resample' else parent
        if (args.repair_failures and arm != 'resample' and (evaluated // 2) % 2 == 1
                and len(archive) > 1 and archive[-1].get('gate_error')):
            selected = archive[-1]
        prompt = TASK + '\n' + API
        if arm != 'no_references':
            prompt += '\n' + REFERENCES
        if arm == 'full_cards':
            prompt += '\n' + HSI_CARDS
        if args.function_patches:
            prompt = prompt.replace('Return JSON {"rationale": "short mechanism and expected benefit", "code": "full module"}.', '')
            prompt += '\n' + PATCH_API
        prompt += '\nInitial/current program:\n```python\n' + selected['code'] + '\n```\n'
        if arm != 'resample':
            feedback = [summarize_feedback(r) for r in archive[-4:]]
            # Strip long traces from prompts, but retain them in the artifacts.
            for rec in feedback:
                rec['rows'] = [{k:v for k,v in row.items() if k not in ('trace', 'source_id')}
                               for row in rec.get('rows', [])]
            prompt += '\nMeasured execution feedback:\n' + json.dumps(feedback)
        prompt += ('\nPropose function replacements that make a complete improved program. ' if args.function_patches
                   else '\nPropose a complete improved program. ')
        prompt += 'You may replace the algorithm, not just constants. '
        prompt += 'Do not copy instance-specific answers; evaluation uses unseen inputs.'
        siblings = []
        for j in range(min(2, args.evaluations-evaluated)):
            proposals += 1
            cand_dir = directory / f'p{proposals:03d}'
            cand_dir.mkdir()
            record = {'proposal':proposals, 'parent':selected['id'], 'arm':arm, 'seed':seed,
                      'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest()}
            try:
                proposal, meta = llm(prompt, args.endpoint, rng.randrange(2**31), cand_dir,
                                     selected['code'] if args.function_patches else None)
                code = proposal['code']
                cid = source_id(code)
                record.update(id=cid, rationale=proposal.get('rationale',''), llm=meta,
                              layer_ids=layer_ids(code))
                (cand_dir / 'candidate.py').write_text(code)
                if cid in seen:
                    record['duplicate'] = True
                    write_json(cand_dir / 'record.json', record)
                    continue
                seen.add(cid)
                # Each fresh candidate reaches the immutable environment once.
                # A gate rejection is an evaluated failure, never a free retry.
                evaluated += 1
                record['evaluation'] = evaluated
                gates = evaluate_on_slot(slot, cand_dir / 'candidate.py', admission,
                                         cand_dir, min(args.budget, 8))
                record['admission'] = gates
                if not all(r.get('accepted') for r in gates):
                    rows = [{'case':c, 'accepted':False, 'elapsed_s':args.budget} for c in train]
                    record['gate_error'] = [{k:v for k,v in r.items() if k != 'trace'}
                                            for r in gates if not r.get('accepted')]
                else:
                    rows = evaluate_on_slot(slot, cand_dir / 'candidate.py', train,
                                            cand_dir, args.budget)
                record.update(rows=rows, score=aggregate(rows, args.budget))
                write_json(cand_dir / 'record.json', record)
                record['code'] = code
                archive.append(record)
                siblings.append(record)
                print(json.dumps({'chain':directory.name, 'eval':evaluated,
                                  'score':record['score'], 'gate_pass': 'gate_error' not in record}), flush=True)
            except Exception as exc:
                record['proposal_error'] = str(exc)[-2500:]
                write_json(cand_dir / 'record.json', record)
                print(json.dumps({'chain':directory.name, 'proposal_error':str(exc)[-500:]}), flush=True)
        if siblings and arm != 'resample':
            # A small archive of alternatives allows escape from one failed lineage.
            best = max(archive, key=rank)
            if (evaluated // 2) % 4 == 0 and len(archive) > 3:
                parent = parent_rng.choice(sorted(archive, key=rank, reverse=True)[:4])
            else:
                parent = best
        write_json(directory / 'status.json', {'evaluated':evaluated, 'proposals':proposals,
                   'best':summarize_feedback(max(archive, key=rank)), 'complete':False})
    winner = max(archive, key=rank)
    (directory / 'winner.py').write_text(winner['code'])
    write_json(directory / 'winner.json', {k:v for k,v in winner.items() if k != 'code'})
    write_json(directory / 'status.json', {'evaluated':evaluated, 'proposals':proposals,
               'best':summarize_feedback(winner), 'complete':True})
    return directory.name

def main():
    global REMOTE
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--endpoint', default='http://127.0.0.1:8011')
    ap.add_argument('--evaluations', type=int, default=12)
    ap.add_argument('--budget', type=float, default=30)
    ap.add_argument('--seeds', default='0,1')
    ap.add_argument('--arms', default=','.join(ARMS))
    ap.add_argument('--deploy-only', action='store_true')
    ap.add_argument('--remote-root', default=REMOTE)
    ap.add_argument('--function-patches', action='store_true')
    ap.add_argument('--repair-failures', action='store_true')
    ap.add_argument('--slots', default='0,1,2,3,4,5', help='fleet slot IDs, node-0=0..3, node-1=4..5')
    args = ap.parse_args()
    REMOTE = args.remote_root
    if not REMOTE.startswith('/root/optevolve_discovery_') or any(c not in '/_-.0123456789abcdefghijklmnopqrstuvwxyz' for c in REMOTE):
        raise ValueError('invalid isolated remote root')
    Path(args.out).mkdir(parents=True, exist_ok=True)
    protocol = {'timestamp_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                'args':vars(args), 'nodes':NODES, 'source_hashes':{
                    p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.py')},
                'data_manifest_sha256':hashlib.sha256((Path(args.data)/'manifest.json').read_bytes()).hexdigest(),
                'scope':'HSI executable rediscovery pilot; NOT lasso D3-D6 ablation; no random-grammar arm',
                'claims':'finite numerical output acceptance; no general convergence proof for synthesized code'}
    protocol_path = Path(args.out) / 'protocol.json'
    if protocol_path.exists():
        raise FileExistsError('choose a fresh campaign directory; protocol is immutable')
    write_json(protocol_path, protocol)
    deploy(args.data, args.out)
    if args.deploy_only:
        return
    slots = queue.Queue()
    available_slots = []
    for host, port, count in NODES:
        for gpu in range(count):
            available_slots.append((host,port,gpu))
    selected_slots = [int(s) for s in args.slots.split(',')]
    if len(set(selected_slots)) != len(selected_slots):
        raise ValueError('duplicate GPU slot assignment')
    for s in selected_slots:
        slots.put(available_slots[s])
    def job(arm, seed):
        slot = slots.get()
        try:
            return chain(arm,seed,slot,args)
        finally:
            slots.put(slot)
    jobs = [(arm,int(seed)) for seed in args.seeds.split(',') for arm in args.arms.split(',')]
    with ThreadPoolExecutor(max_workers=len(selected_slots)) as pool:
        futures = {pool.submit(job,arm,seed):(arm,seed) for arm,seed in jobs}
        for f in as_completed(futures):
            try:
                print('CHAIN_DONE', f.result(), flush=True)
            except Exception as exc:
                print('CHAIN_ERROR', futures[f], repr(exc), flush=True)

if __name__ == '__main__':
    main()
