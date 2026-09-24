#!/usr/bin/env python3
"""Validate an immutable tuning package inside its image without GPU access."""
import argparse
import json
from pathlib import Path
import subprocess

CODE = '''import json,sys
from pdblend.bench.capacity_floor_v2 import validate_manifest
from pdblend.bench.low_m_tuning import source_inventory
from pdblend.bench.resident_session import digest
from pdblend.bench.low_m_tuning_runtime import LowMTuningAdapter
from pdblend.bench.pdblend_runtime_options import control_options
from pdblend.control.policies import get_policy
group=json.load(open(sys.argv[1]))
manifest=validate_manifest(group['tuning_manifest']['path'])
assert digest(source_inventory())==manifest['context']['algorithm_source_sha256']
control_options(get_policy('pdblend'),manifest['recovery_policy'])
print(json.dumps(dict(passed=True,trials=len(manifest['trials']),
    context=manifest['context'],hardware_executed=False)))
'''


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package',required=True,type=Path)
    args=parser.parse_args(); package=args.package.resolve()
    output=package/'container-preflight.json'
    if output.exists(): raise ValueError('preflight record already exists')
    job=json.loads((package/'jobs.json').read_text())[0]
    payload=job['payload']; argv=payload['argv']
    group=Path(argv[argv.index('--group')+1]); data=json.loads(group.read_text())
    source=Path(data['source_manifest']['path']).parent
    cmd=['docker','run','--rm','--runtime=runc','--network=none','--entrypoint','/opt/venv/bin/python',
         '-v',str(source)+':/opt/pdblend-src:ro','-v','/home/pdblend4:/home/pdblend4:ro',
         '-v','/home/models:/models:ro','-e','NVIDIA_VISIBLE_DEVICES=void',
         '-e','PYTHONPATH=/opt/pdblend-src','-e','PYTHONDONTWRITEBYTECODE=1',
         payload['image_digest'],'-B','-c',CODE,str(group)]
    result=subprocess.run(cmd,text=True,capture_output=True,timeout=300)
    report=dict(job_id=job['job_id'],source_revision=source.name,returncode=result.returncode,
        stdout=result.stdout,stderr=result.stderr,hardware_executed=False,argv=cmd)
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(job_id=job['job_id'],returncode=result.returncode,
                         stdout=result.stdout,stderr=result.stderr)))
    return 0 if result.returncode==0 else 1


if __name__=='__main__':
    raise SystemExit(main())
