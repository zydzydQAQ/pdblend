#!/usr/bin/env python3
"""Bind final profile acceptance to the independent, fixed and adaptive evidence."""
import argparse, hashlib, json
from pathlib import Path

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
p=argparse.ArgumentParser();p.add_argument('profile',type=Path);a=p.parse_args();profile=a.profile.resolve();d=profile.parent
audit=json.loads((d/'audit.json').read_text());ind=json.loads((d/'independent-report.json').read_text());fixed=json.loads((d/'acceptance/fixed-summary.json').read_text());adaptive=json.loads((d/'adaptive-summary.json').read_text())
if not audit.get('basic_passed') or ind.get('gate',{}).get('status')!='PASS' or not adaptive.get('gate',{}).get('passed'):
    raise SystemExit('profile evidence is not complete')
if not all(all(r.get('passed') for r in rows) for rows in fixed.get('fixed',{}).values()):
    raise SystemExit('fixed M>=4 evidence is incomplete')
if sha(profile)!=audit.get('profile_sha256') or sha(profile)!=ind.get('candidate_sha256'):
    raise SystemExit('profile hash binding mismatch')
audit.update(status='accepted',formal_eligible=True,independent_report_sha256=sha(d/'independent-report.json'),fixed_summary_sha256=sha(d/'acceptance/fixed-summary.json'),adaptive_summary_sha256=sha(d/'adaptive-summary.json'))
(d/'audit.json').write_text(json.dumps(audit,indent=1))
(d/'AUDIT-PASSED').write_text(json.dumps(dict(profile_sha256=sha(profile),raw_sha256=audit.get('raw_sha256'),audit=str((d/'audit.json').resolve()),independent_report_sha256=sha(d/'independent-report.json'),fixed_summary_sha256=sha(d/'acceptance/fixed-summary.json'),adaptive_summary_sha256=sha(d/'adaptive-summary.json')),indent=1))
print(json.dumps(dict(status=audit['status'],formal_eligible=audit['formal_eligible'],profile_sha256=sha(profile)),indent=1))
