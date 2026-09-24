#!/usr/bin/env python3
"""Validate 36 frozen initial plans and 9 fresh extensions in the actual CPU image."""
import argparse
import json
from pathlib import Path
import subprocess
import time

CHECK = r'''
import hashlib,importlib.util,json,os
from pathlib import Path
from types import SimpleNamespace
from pdblend.bench.comparison_campaign import load_bound,binding
from pdblend.bench.comparison_pdblend_observation import validate_observation_inputs
from pdblend.bench.comparison_runtime import pdblend_window_resources
from pdblend.bench.independent_dispatch import request_rows
from pdblend.bench.pdblend_runtime_options import comparison_options
from pdblend.bench.resident_session import digest
from pdblend.bench.single_observation_slo_boundary import make_extension_point,validate_generated_point,validate_policy
root=Path(os.environ['PDBLEND_PREFLIGHT_PACKAGE']);out=Path(os.environ['PDBLEND_PREFLIGHT_OUTPUT'])
c=json.loads((root/'campaign.json').read_text());execution=load_bound(c['execution_inputs'])
source=Path(execution['source']);manifest=load_bound(c['execution_source_manifest'])
assert digest(manifest['files'])==execution['source_sha256']
for name,sha in manifest['files'].items():
 assert hashlib.sha256((source/name).read_bytes()).hexdigest()==sha,name
helper_ref=c['startup_helper'];load_bound_bytes=Path(helper_ref['path']).read_bytes()
assert hashlib.sha256(load_bound_bytes).hexdigest()==helper_ref['sha256']
spec=importlib.util.spec_from_file_location('pdblend.bench.comparison_startup',helper_ref['path'])
startup=importlib.util.module_from_spec(spec);spec.loader.exec_module(startup)
import pdblend
assert Path(pdblend.__file__).resolve().is_relative_to('/opt/pdblend-src')
records=[]
def check(point,specs,extension=False):
 checked=validate_observation_inputs(point,point['inputs'])
 assert checked['formal_eligible'] is False and checked['profile_qualified'] is False
 assert point['revision']==execution['source_sha256']
 loaded,plan=pdblend_window_resources(point,specs)
 cfg=load_bound(point['inputs']['system_config']);choice=load_bound(point['inputs']['offline_choice'])
 options=comparison_options(cfg,point['inputs']['system_config']['path'],point=point)['values']
 tuning=load_bound(point['inputs']['planning_trace'])
 assert tuning['selection_split'] in ('calibration','tuning')
 assert options['capacity_floor_path'] is None
 actual=startup.preview_startup(point,loaded.model,plan,options,request_rows(tuning))
 for role in ('P','D','M'):
  assert getattr(plan,'f_'+role)<=options['safety_max_freq']
  assert actual['f_'+role]<=options['safety_max_freq']
 if extension:
  assert 'startup_contract' not in choice
  assert choice['startup_contract_pending']=='recompute_from_generated_point_independent_tuning'
  trace=load_bound(point['trace']);assert trace['duration_s']==150 and trace['seed']==701
  assert trace['selection_split']=='evaluation' and all(0<=r['arrival_s']<150 for r in trace['requests'])
 else:
  startup.validate_contract_bindings(choice['startup_contract'],cfg['profile'],point['inputs']['planning_trace'],options)
  assert actual==choice['startup_contract']['expected_first_plan']
 records.append(dict(point=point['name'],extension=extension,trace=point['trace'],planning_trace=point['inputs']['planning_trace'],actual_first_plan=actual))
for group in c['groups']:
 assert len(group['points'])==12 and {p['system'] for p in group['points']}=={'pdblend'}
 specs=[SimpleNamespace(tp=r['tp'],pp=r['pp'],generation=0) for r in group['engine_identity']['instances']]
 for point in group['points']:check(point,specs)
 refs=[r for r in c['active_extension_policy_refs'] if load_bound(r)['model_id']==group['model_id']]
 assert len(refs)==1;policy_ref=refs[0];policy=validate_policy(load_bound(policy_ref),group)
 for dataset in ('alpaca','sharegpt','longbench'):
  point,ref=make_extension_point(policy_ref,dataset,1.25,out/'extensions'/group['model_id']/dataset)
  validate_generated_point(policy,point)
  check(point,specs,True)
assert sum(not r['extension'] for r in records)==36 and sum(r['extension'] for r in records)==9
result=dict(status='passed',hardware_executed=False,source_sha256=execution['source_sha256'],source_manifest=c['execution_source_manifest'],campaign=binding(root/'campaign.json'),startup_helper=helper_ref,original_points=36,extension_points=9,records=records)
(out/'result.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='records'}))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    package, out = args.package.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    campaign = json.loads((package / 'campaign.json').read_text())
    execution = json.loads(Path(campaign['execution_inputs']['path']).read_text())
    source = execution['source']
    argv = ['docker', 'run', '--rm', '--network', 'none', '--cpus', '1', '--memory', '4g',
            '--entrypoint', '/opt/venv/bin/python', '-v', source + ':/opt/pdblend-src:ro',
            '-v', '/home/pdblend4:/home/pdblend4:ro', '-v', str(out) + ':' + str(out) + ':rw',
            '-e', 'PYTHONPATH=/opt/pdblend-src', '-e', 'PYTHONDONTWRITEBYTECODE=1',
            '-e', 'OMP_NUM_THREADS=1', '-e', 'OPENBLAS_NUM_THREADS=1',
            '-e', 'PDBLEND_PREFLIGHT_PACKAGE=' + str(package),
            '-e', 'PDBLEND_PREFLIGHT_OUTPUT=' + str(out), execution['image_digest'], '-B', '-c', CHECK]
    started = time.time()
    completed = subprocess.run(argv, capture_output=True, text=True, timeout=120)
    receipt = dict(returncode=completed.returncode, elapsed_s=time.time() - started,
                   image_digest=execution['image_digest'], hardware_executed=False,
                   source_sha256=execution['source_sha256'], argv=argv,
                   stdout=completed.stdout, stderr=completed.stderr)
    (out / 'execution.json').write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
    if completed.returncode:
        raise RuntimeError(completed.stderr[-4000:])
    result = json.loads((out / 'result.json').read_text())
    print(json.dumps({k: v for k, v in result.items() if k != 'records'}))


if __name__ == '__main__':
    main()
