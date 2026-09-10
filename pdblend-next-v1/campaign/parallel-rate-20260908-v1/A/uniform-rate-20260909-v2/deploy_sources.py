"""Stage only absent exact immutable inputs for the new-A v2 contract."""
from pathlib import Path
import importlib.util,json
U=Path(__file__).resolve().parent;ROOT=U.parents[1];C=ROOT/'common/uniform-rate-20260909-v2'
spec=importlib.util.spec_from_file_location('safe_newA_stage',U.parent/'uniform-rate-20260909-v1/stage.py');s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)
d=json.loads((C/'release-002/declaration.json').read_text())
paths=set(C.glob('*.py'));paths.update(p for p in (C/'release-002').rglob('*') if p.is_file());paths.update([U/'pipeline.py',U/'build_pipeline.py'])
for row in d['cells']:
 if row['node']=='Anew20260909':
  paths.add(Path(row['trace']));paths.add(Path(row['source_300s_trace']['path']))
for r in d.get('reconstructed_missing_sources',[]):
 src,ref=r['exact_local_copy'],r['original_reference'];assert s.sha(src['path'])==src['sha256']==ref['sha256'];p=Path(ref['path'])
 if not p.exists():
  p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb') as f:f.write(Path(src['path']).read_bytes())
 assert s.sha(p)==ref['sha256'];paths.add(p)
paths.update([ROOT/'B/baseline-return-after-external-source-v1/execution.py',ROOT/'raw_metrics_v3.py',ROOT/'raw_metrics_v2.py',ROOT/'audit_cooperative_arrivals_v1.py'])
paths.update((ROOT.parent/'main-slo-improvement-v7').glob('*.py'))
print(json.dumps(s.stage(sorted(paths))))
