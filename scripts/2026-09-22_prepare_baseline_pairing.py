#!/usr/bin/env python3
"""Create isolated single-seed baseline pairing specs for finalist points."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from pdblend.seed_config import SINGLE_SEED, SEEDS, SEED_POLICY, seed_metadata

BASELINES = ('mixed', 'distserve_static', 'dynamollm', 'ecoserve')

p = argparse.ArgumentParser()
p.add_argument('candidate_spec', type=Path)
p.add_argument('candidate_points', type=Path, help='JSON or CSV with finalist names')
p.add_argument('out_root', type=Path)
p.add_argument('--seeds', default=str(SINGLE_SEED))
p.add_argument('--profile', type=Path, default=Path('results/v2/profile-7b/profile.json'))
a = p.parse_args()
requested_seeds = tuple(int(x) for x in a.seeds.split(',') if x)
if requested_seeds != SEEDS:
    p.error(f'active campaign uses {SEED_POLICY}; pass --seeds {SINGLE_SEED}')
spec = json.loads(a.candidate_spec.read_text())
if a.candidate_points.suffix == '.json':
    data = json.loads(a.candidate_points.read_text())
    names = [r['name'] for r in data.get('rows', data) if r.get('status') in ('win', 'screen_pass', 'screen_m2')]
else:
    import csv
    with a.candidate_points.open(newline='') as fh:
        names = [r['name'] for r in csv.DictReader(fh) if r.get('status') in ('win', 'screen_pass', 'screen_m2')]
points = []
for candidate_name in names:
    base = next((p for p in spec['points'] if p['name'] == candidate_name), None)
    if base is None:
        base = next((p for p in spec['points'] if p['name'].replace('-pdblend_dominance', '') == candidate_name), None)
    if base is None:
        raise SystemExit(f'candidate point absent from spec: {candidate_name}')
    for seed in requested_seeds:
        for policy in BASELINES:
            q = dict(base, name=f"{base['name'].replace('-pdblend_dominance','')}-{policy}-seed-{seed}",
                     policy=policy, seed=seed, profile=str(a.profile.resolve()), **seed_metadata())
            points.append(q)
out = dict(spec, root=str(a.out_root.resolve()), points=points, **seed_metadata(),
           defaults=dict(spec.get('defaults', {}), profile=str(a.profile.resolve()),
                         seeds=[SINGLE_SEED], single_seed=True, seed_policy=SEED_POLICY))
a.out_root.mkdir(parents=True, exist_ok=True)
(a.out_root / 'spec.json').write_text(json.dumps(out, indent=1))
(a.out_root / 'provenance.json').write_text(json.dumps({
    'candidate_spec': str(a.candidate_spec.resolve()),
    'candidate_spec_sha256': hashlib.sha256(a.candidate_spec.read_bytes()).hexdigest(),
    'profile': str(a.profile.resolve()), 'profile_sha256': hashlib.sha256(a.profile.read_bytes()).hexdigest(),
    'baselines': BASELINES, 'seeds': [SINGLE_SEED], 'single_seed': True,
    'seed_policy': SEED_POLICY,
    'points': len(points), 'candidate_points': names, 'baseline_code_frozen': True,
}, indent=1))
print(json.dumps({'root': str(a.out_root), 'points': len(points), 'candidate_points': names}, indent=1))
