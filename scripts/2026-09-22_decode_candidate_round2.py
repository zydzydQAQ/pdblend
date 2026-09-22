#!/usr/bin/env python3
"""Freeze round-2 900 MHz candidate after using round-1 as development data."""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
from pdblend.profile.model import PerfModel
from pdblend.profile.decode_fit import compare, fit_candidate, errors, predict
from pdblend.profile.merge import sha256

p = argparse.ArgumentParser()
p.add_argument('base', type=Path, help='immutable profile-v2-parallel directory')
p.add_argument('development', type=Path, help='round-1 independent-report.json')
p.add_argument('out', type=Path)
p.add_argument('--kind', default='hinges4_64_ctx')
p.add_argument('--repeat-batch', type=int, default=0,
               help='repeat original/development rows for this batch in the fit')
a = p.parse_args()
if a.out.exists():
    raise FileExistsError(a.out)
a.out.mkdir(parents=True)
raw = json.loads((a.base/'raw.json').read_text())
model = PerfModel.load(a.base/'profile.json')
rows = [r for r in raw['decode'] if r['freq_mhz'] == 900]
dev = json.loads(a.development.read_text())
for r in dev['rows']:
    rows.append(dict(batch=r['batch'], context_tokens=r['context_tokens'],
                     effective_context_tokens=r['effective_context_tokens'],
                     step_seconds=r['observed'], power_w=None, freq_mhz=900))
if a.repeat_batch > 0:
    weighted = [r for r in rows if r['batch'] == a.repeat_batch]
    rows.extend(weighted)
spec = fit_candidate(rows, a.kind, raw['kv_capacity_tokens'])
spec['source'] = dict(raw_path=str((a.base/'raw.json').resolve()),
                      raw_sha256=sha256(a.base/'raw.json'),
                      development_report=str(a.development.resolve()),
                      development_sha256=sha256(a.development),
                      parent_profile_sha256=sha256(a.base/'profile.json'),
                      frozen_at_s=time.time())
model.decode_overrides[900] = spec
q = errors([r['step_seconds'] for r in rows],
           [predict(spec, r['batch'], r['effective_context_tokens']) for r in rows])
model.quality['decode_time@900'] = q
model.residuals['decode_time@900'] = q['max']
model.save(a.out/'profile.json')
(a.out/'raw.json').symlink_to((a.base/'raw.json').resolve())
for path in a.base.glob('pair-*'):
    if path.is_dir():
        (a.out/path.name).symlink_to(path.resolve(), target_is_directory=True)
comparison = compare([r for r in rows if r.get('power_w') is not None],
                     model, raw['kv_capacity_tokens'])
(a.out/'cpu-comparison.json').write_text(json.dumps(comparison, indent=1))
(a.out/'candidate-frozen.json').write_text(json.dumps(dict(
    profile_sha256=sha256(a.out/'profile.json'), raw_sha256=sha256(a.out/'raw.json'),
    development_sha256=sha256(a.development), frozen_at_s=time.time(),
    kind=a.kind, training=q, development_samples=len(dev['rows'])), indent=1))
print(json.dumps(dict(profile=str(a.out/'profile.json'), training=q,
                     development_samples=len(dev['rows'])), indent=1))
