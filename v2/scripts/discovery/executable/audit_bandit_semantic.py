"""Apply the pre-existing ADMM/DRS equivalence audit to all bandit records.

This reads only training execution traces. It never changes the original
literal-ADMM records or exposes a sealed case to the search policy.
"""
import argparse
import json
from pathlib import Path

import campaign as C
from recovery_equivalence import audit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--chain', required=True)
    a = ap.parse_args()
    root = Path(a.chain)
    out = []
    for path in sorted(root.glob('p*/record.json')):
        record = json.loads(path.read_text())
        source = path.with_name('candidate.py')
        if 'evaluation' not in record or 'rows' not in record or not source.exists():
            continue
        result = audit(record['spec'], source.read_text(), record['rows'])
        C.write_json(path.with_name('semantic_audit.json'), result)
        out.append({'evaluation': record['evaluation'], 'proposal': record['proposal'],
                    'source': str(source), 'method_recovered': result['method_recovered'],
                    'literal_admm': result['literal_admm_audit']['method_recovered']})
    C.write_json(root / 'semantic_audit_status.json', {'audited': out,
        'first_semantic_recovery': next((r for r in out if r['method_recovered']), None),
        'audit_definition': 'recovery_equivalence.py, frozen before bandit campaign'})
    print(root / 'semantic_audit_status.json')


if __name__ == '__main__':
    main()
