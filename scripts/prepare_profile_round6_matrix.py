#!/usr/bin/env python3
"""Create an isolated PDblend-only matrix using the accepted round6 profile."""
import argparse, hashlib, json
from pathlib import Path

p=argparse.ArgumentParser(); p.add_argument('source_spec',type=Path); p.add_argument('out_root',type=Path); p.add_argument('profile',type=Path); a=p.parse_args()
s=json.loads(a.source_spec.read_text()); points=[]
for point in s['points']:
    if point.get('policy') != 'pdblend':
        continue
    q=dict(point); q['profile']=str(a.profile.resolve()); points.append(q)
out=dict(s,root=str(a.out_root.resolve()),points=points)
a.out_root.mkdir(parents=True,exist_ok=True)
(a.out_root/'spec.json').write_text(json.dumps(out,indent=1))
(a.out_root/'matrix-provenance.json').write_text(json.dumps(dict(source_spec=str(a.source_spec.resolve()),source_spec_sha256=hashlib.sha256(a.source_spec.read_bytes()).hexdigest(),profile=str(a.profile.resolve()),profile_sha256=hashlib.sha256(a.profile.read_bytes()).hexdigest(),points=len(points),baseline_points_excluded=True),indent=1))
print(json.dumps(dict(root=str(a.out_root),points=len(points),profile=str(a.profile)),indent=1))
