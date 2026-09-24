#!/usr/bin/env python3
"""Check the exact packaged PD imports, inputs and 36 plans without GPUs."""
import argparse
import json
from pathlib import Path
import subprocess
import time
from pdblend.bench.resident_session import write_new

CHECK = r'''
import hashlib,json,os
from pathlib import Path
from types import SimpleNamespace
from pdblend.bench.comparison_pdblend_observation import validate_observation_inputs
from pdblend.bench.comparison_pdblend_lifecycle import PDblendResidentBoundary
from pdblend.bench.comparison_runtime import pdblend_window_resources
from pdblend.bench.resident_session import digest
root=Path(os.environ['PDBLEND_PREFLIGHT_PACKAGE'])
c=json.load(open(root/'campaign.json'));execution=json.load(open(root/'execution-inputs.json'))
s=Path(execution['source']);manifest=json.load(open(s/'manifest.json'))
assert digest(manifest['files'])==execution['source_sha256']
for name,sha in manifest['files'].items():
 assert hashlib.sha256((s/name).read_bytes()).hexdigest()==sha
records=[]
for group in c['groups']:
 if {p['system'] for p in group['points']}!={'pdblend'}:continue
 assert len(group['points'])==12
 specs=[SimpleNamespace(tp=r['tp'],pp=r['pp'],generation=0) for r in group['engine_identity']['instances']]
 for point in group['points']:
  checked=validate_observation_inputs(point,point['inputs'])
  loaded,plan=pdblend_window_resources(point,specs)
  assert checked['formal_eligible'] is False and checked['profile_qualified'] is False
  records.append(dict(point=point['name'],tp=plan.tp,counts=plan.counts,frequency_mhz=plan.f_M))
assert len(records)==36
print(json.dumps(dict(status='passed',hardware_executed=False,points=len(records),source_files=len(manifest['files']),source_sha256=execution['source_sha256'],plans=records)))
'''


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package',type=Path,required=True)
    args=parser.parse_args();out=args.package.resolve()
    execution=json.loads((out/'execution-inputs.json').read_text());source=Path(execution['source'])
    argv=['docker','run','--rm','--network','none','--cpus','1','--memory','4g',
        '--entrypoint','/opt/venv/bin/python','-v',str(source)+':/opt/pdblend-src:ro',
        '-v','/home/pdblend4:/home/pdblend4:ro','-e','PYTHONPATH=/opt/pdblend-src',
        '-e','PYTHONDONTWRITEBYTECODE=1','-e','OMP_NUM_THREADS=1','-e','OPENBLAS_NUM_THREADS=1',
        '-e','PDBLEND_PREFLIGHT_PACKAGE='+str(out),execution['image_digest'],'-B','-c',CHECK]
    start=time.time();result=subprocess.run(argv,capture_output=True,text=True,timeout=60)
    write_new(out/'image-cpu-preflight.json',dict(returncode=result.returncode,
        elapsed_s=time.time()-start,image_digest=execution['image_digest'],
        source_sha256=execution['source_sha256'],hardware_executed=False,
        stdout=result.stdout,stderr=result.stderr))
    if result.returncode:raise RuntimeError(result.stderr[-2500:])
    data=json.loads(result.stdout.splitlines()[-1]);assert data['status']=='passed'
    print(json.dumps({k:v for k,v in data.items() if k!='plans'}))


if __name__=='__main__':main()
