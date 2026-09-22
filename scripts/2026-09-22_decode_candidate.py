#!/usr/bin/env python3
"""Freeze a decode-only 900 MHz candidate before collecting independent evidence."""
import argparse,json,time,os
from pathlib import Path
from pdblend.profile.model import PerfModel
from pdblend.profile.decode_fit import compare,fit_candidate,errors,predict
from pdblend.profile.merge import sha256

p=argparse.ArgumentParser();p.add_argument('base',type=Path);p.add_argument('out',type=Path)
p.add_argument('--kind',default='hinges4_64');a=p.parse_args()
if a.out.exists(): raise FileExistsError(a.out)
a.out.mkdir(parents=True)
raw=json.loads((a.base/'raw.json').read_text());model=PerfModel.load(a.base/'profile.json')
rows=[r for r in raw['decode'] if r['freq_mhz']==900]
comparison=compare(rows,model,raw['kv_capacity_tokens'])
spec=fit_candidate(rows,a.kind,raw['kv_capacity_tokens'])
spec['source']=dict(raw_path=str((a.base/'raw.json').resolve()),raw_sha256=sha256(a.base/'raw.json'),
                   parent_profile_sha256=sha256(a.base/'profile.json'), frozen_at_s=time.time())
model.decode_overrides[900]=spec
q=errors([r['step_seconds'] for r in rows],[predict(spec,r['batch'],r['effective_context_tokens']) for r in rows])
model.quality['decode_time@900']=q;model.residuals['decode_time@900']=q['max']
model.save(a.out/'profile.json')
# Evidence references remain valid without copying or altering immutable source shards.
(a.out/'raw.json').symlink_to((a.base/'raw.json').resolve())
for path in a.base.glob('pair-*'):
 if path.is_dir(): (a.out/path.name).symlink_to(path.resolve(),target_is_directory=True)
(a.out/'cpu-comparison.json').write_text(json.dumps(comparison,indent=1))
(a.out/'candidate-frozen.json').write_text(json.dumps(dict(profile_sha256=sha256(a.out/'profile.json'),
   raw_sha256=sha256(a.out/'raw.json'),frozen_at_s=time.time(),kind=a.kind,training=q),indent=1))
print(json.dumps(dict(profile=str(a.out/'profile.json'),training=q)))
