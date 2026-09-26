"""Extract selected public OpenReview notes from a published archive.

Source: qhjqhj00/iclr-openreview-reviews, GitHub release v1.0.
Only notes explicitly readable by everyone are retained. No reviewer identity
resolution is performed. Run with the downloaded archive path as argument.
"""
import hashlib
import json
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parent
IDS = {'RQm2KQTM5r', 'hit3hGBheP', 'Jb1WkNSfUB', '0jHyEKHDyx',
       'T0EiEuhOOL', 'm2nmp8P5in', 'z5uVAKwmjf', 'IkmD3fKBPQ',
       'LU27DiW5ik', 'w696Vhv5B2'}
records = {k: {'paper': None, 'notes': []} for k in IDS}
needles = [x.encode() for x in IDS]

with tarfile.open(sys.argv[1], 'r|bz2') as tar:
    for member in tar:
        if not member.isfile() or not member.name.endswith('.jsonl'):
            continue
        if not any(f'iclr_{y}/' in member.name for y in (2024, 2025, 2026)):
            continue
        print('Reading', member.name, flush=True)
        for line in tar.extractfile(member):
            if not any(x in line for x in needles):
                continue
            n = json.loads(line)
            if 'everyone' not in [x.lower() for x in n.get('readers', [])]:
                continue
            forum = n.get('forum', n.get('id'))
            if forum not in IDS:
                continue
            if n['id'] == forum:
                c = n['content']
                records[forum]['paper'] = {'id': n['id'], 'content': {
                    k: v for k, v in c.items() if k in (
                        'title', 'abstract', 'venue', 'venueid', 'pdf', 'keywords')}}
            else:
                records[forum]['notes'].append({k: n[k] for k in (
                    'id', 'forum', 'replyto', 'invitations', 'invitation',
                    'signatures', 'readers', 'cdate', 'mdate', 'content') if k in n})
        for forum, record in records.items():
            if record['paper'] or record['notes']:
                (ROOT / (forum + '.json')).write_text(json.dumps(record, indent=2))

for forum, r in records.items():
    c = (r['paper'] or {}).get('content', {})
    print(forum, len(r['notes']), c.get('title'), c.get('venue'))
    if r['paper']:
        lines = [str(c.get('title')), str(c.get('venue'))]
        for n in sorted(r['notes'], key=lambda n: n.get('cdate', 0)):
            lines += ['', f"NOTE {n['id']} {n.get('invitations', n.get('invitation'))}"]
            for k, v in n['content'].items():
                lines += [k.upper(), str(v.get('value', v) if isinstance(v, dict) else v)]
        (ROOT / (forum + '.txt')).write_text('\n'.join(lines))

(ROOT / 'archive_provenance.json').write_text(json.dumps({
    'url': 'https://github.com/qhjqhj00/iclr-openreview-reviews/releases/download/v1.0/iclr.tar.bz2',
    'accessed': '2026-09-06',
    'filter': 'Selected forums; readers including everyone; public author identities excluded from paper metadata',
    'sha256': hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest()
}, indent=2))
