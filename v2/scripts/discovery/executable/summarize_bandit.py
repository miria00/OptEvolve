"""Render observed fixed-budget plain-loop and online-policy outcomes."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from recovery_equivalence import audit as semantic_audit


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def rows(chain, solver_only=False):
    records = [(p, read(p)) for p in chain.glob('p*/record.json')]
    fresh = sorted(((p, r) for p, r in records if r and 'evaluation' in r),
                   key=lambda pair: pair[1]['evaluation'])
    status = read(chain / 'status.json') or {}
    def hit(path, r):
        if solver_only:
            return (r.get('score', {}).get('total') is not None and
                    r.get('score', {}).get('solved') == r.get('score', {}).get('total'))
        source = path.with_name('candidate.py')
        if not source.exists() or 'rows' not in r or 'spec' not in r:
            return False
        try:
            return semantic_audit(r['spec'], source.read_text(), r['rows'])['method_recovered']
        except (ValueError, KeyError):
            return False
    first = next((r['evaluation'] for p, r in fresh if hit(p, r)), None)
    literal = next((r['evaluation'] for _, r in fresh
                    if r.get('recovery_audit', {}).get('method_recovered')), None)
    solved = sum(r.get('score', {}).get('total') is not None and
                 r.get('score', {}).get('solved') == r.get('score', {}).get('total')
                 for _, r in fresh)
    actions = Counter(r.get('action', 'model' if r.get('llm') else 'random')
                      for _, r in fresh)
    usage = Counter()
    for _, r in records:
        if r and r.get('llm'):
            usage.update({k: r['llm']['usage'].get(k, 0)
                          for k in ('prompt_tokens', 'completion_tokens')})
    return {'fresh': len(fresh), 'finished': bool(status.get('complete')),
            'first': first, 'literal': literal, 'solved': solved, 'actions': actions,
            'model_calls': sum(bool(r and r.get('llm')) for _, r in records),
            'tokens': usage}


def render(root):
    entries = []
    for directory in sorted((root / 'recovery_v1').glob('*_s*')):
        if directory.is_dir() and (directory / 'status.json').exists():
            entries.append(('pure '+directory.name, rows(directory)))
    for name in ('rl_bandit_s0', 'rl_bandit_s1', 'deconv_pure_s0',
                 'deconv_rl_s0'):
        directory = root / name
        if directory.exists():
            entries.append((name, rows(directory, name.startswith('deconv_'))))
    out = ['# Fixed-budget LLM and bandit pilot', '',
           'Updated '+datetime.now(timezone.utc).isoformat()+'.', '',
           'HSI recovery requires the pre-existing ADMM/DRS semantic execution '
           'audit plus independent '
           'numerical certification. Deconvolution reports first all-training '
           'solve instead; it has no HSI method-identity target. Incomplete '
           'chains are not counted as zero at eight.', '',
           '| Chain | Fresh / 8 | Finished | All-training solves | First semantic/solver hit | First literal ADMM | Executed actions | Model calls |',
           '|---|---:|---|---:|---|---|---|---:|']
    for name, r in entries:
        actions = ', '.join(f'{k}:{v}' for k, v in sorted(r['actions'].items()))
        first = str(r['first']) if r['first'] is not None else 'none yet'
        literal = str(r['literal']) if r['literal'] is not None else 'none yet'
        if name.startswith('deconv_'):
            literal = 'not applicable'
        out.append(f"| {name} | {r['fresh']} / 8 | {r['finished']} | {r['solved']} | {first} | {literal} | {actions} | {r['model_calls']} |")
    out += ['', 'The deconvolution assay is a different application and data '
            'distribution with the same simplex least-squares mathematics. '
            'It is not a new optimization class. H100 transfer has not been run.']
    (root / 'RL_PILOT_STATUS.md').write_text('\n'.join(out)+'\n')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    args = ap.parse_args()
    render(args.root)
