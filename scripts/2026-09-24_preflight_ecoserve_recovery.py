#!/usr/bin/env python3
"""Import the frozen Eco recovery bundle and validate inputs without GPUs."""
import argparse
import json
from pathlib import Path
import subprocess

from pdblend.bench.comparison_campaign import binding, load_bound
from pdblend.bench.resident_session import write_new


CHECK = r'''
import json, sys
from pathlib import Path
from pdblend.bench.comparison_campaign import binding, load_bound
from pdblend.bench.comparison_runtime import NativeResidentAdapter
from pdblend.bench.comparison_ecoserve_inputs import validate_ecoserve_inputs
from pdblend.bench.comparison_ecoserve_acceptance import audit_ecoserve_window
from pdblend.bench.comparison_meter_preflight import qualify_startup_snapshot
from pdblend.bench.isolated_comparison_meter import IsolatedComparisonMeter
from pdblend.bench.resident_session import digest, file_sha
root=Path(sys.argv[1]); campaign=json.loads((root/'campaign.json').read_text())
execution=json.loads((root/'execution-inputs.json').read_text())
source=Path(execution['source']); manifest=json.loads((source/'manifest.json').read_text())
assert digest(manifest['files'])==manifest['source_sha256']==execution['source_sha256']
for name,sha in manifest['files'].items():
    assert file_sha(source/name)==sha, name
checks=[]
for job in json.loads((root/'jobs.json').read_text()):
    argv=job['payload']['argv']; group=load_bound(binding(argv[argv.index('--group')+1]))
    assert NativeResidentAdapter.isolated_metering_requested(group)
    assert all(p['system']=='ecoserve' for p in group['points'])
    for point in group['points']:
        old=campaign.get('preserved_baseline_receipts',{}).get(point['name'])
        if old:
            receipt=load_bound(old)
            assert receipt['point_sha256']==digest(point) and receipt['baseline_frozen'] and receipt['evidence_valid']
            checks.append(dict(point=point['name'],passed=True,preserved_without_execution=True))
            continue
        result=validate_ecoserve_inputs(point, group['engine_identity'], source_manifest=binding(source/'manifest.json'))
        assert result['preflight_ready'], result
        checks.append(dict(point=point['name'],passed=True,preserved_without_execution=False))
assert len(checks)>0
print(json.dumps(dict(passed=True, hardware_executed=False, checks=checks, source_sha256=execution['source_sha256'])))
'''


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign-dir', type=Path, required=True)
    args=parser.parse_args(); out=args.campaign_dir.resolve()
    execution=load_bound(binding(out/'execution-inputs.json'))
    project=Path(__file__).resolve().parents[1]
    argv=['docker','run','--rm','--name',out.name+'-cpu',
        '--entrypoint','/opt/venv/bin/python',
        '-v',str(project)+':'+str(project)+':ro',
        '-v',execution['source']+':/opt/pdblend-src:ro',
        '-e','NVIDIA_VISIBLE_DEVICES=void','-e','PYTHONDONTWRITEBYTECODE=1',
        '-e','PYTHONPATH=/opt/pdblend-src',execution['image_digest'],'-B','-c',CHECK,str(out)]
    write_new(out/'preflight-command.json',dict(argv=argv,hardware_executed=False))
    process=subprocess.run(argv,capture_output=True,text=True,timeout=180)
    for name,value in [('stdout',process.stdout),('stderr',process.stderr)]:
        with (out/('preflight.'+name)).open('x') as stream:stream.write(value)
    parsed=json.loads(process.stdout) if process.returncode==0 else None
    review=dict(status='cpu_preflight_passed' if process.returncode==0 else 'preflight_failed',
        returncode=process.returncode, hardware_executed=False, queue_modified=False,
        campaign=binding(out/'campaign.json'),jobs=binding(out/'jobs.json'),
        source_manifest=binding(Path(execution['source'])/'manifest.json'),
        command=binding(out/'preflight-command.json'), stdout=binding(out/'preflight.stdout'),
        stderr=binding(out/'preflight.stderr'),result=parsed,implementation=binding(Path(__file__)))
    write_new(out/'review.json',review)
    print(json.dumps(dict(status=review['status'],returncode=process.returncode)))
    return 0 if process.returncode==0 else 2


if __name__=='__main__':raise SystemExit(main())
