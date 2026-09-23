#!/usr/bin/env python3
"""Validate immutable incremental GPU jobs without exposing any GPU."""
import argparse
import concurrent.futures
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
IMAGE='sha256:1c2d0bf96dfa752394a6aa4b5398a6105dcf060936a484a89729dcab6f9d9acc'
TESTS=['test_long_context_followup.py','test_long_context_followup_review.py','test_long_context_collect.py',
       'test_local_power_job.py','test_local_power.py','test_incremental_profile_wave.py']
PREFLIGHT='''import hashlib,json,os,sys
from pathlib import Path
from pdblend.model_registry import ModelRegistry
from pdblend.profile import long_context_collect as lc
member=json.loads(sys.argv[1]);kind=member['kind'];plan=None
if kind=='followup':
 from pdblend.profile.long_context_followup import load_package
 manifest,plan,raw=load_package(member['package'])
 expected_points=manifest['expected_training_points']
 assert expected_points==len(plan['training'])
 assert all(p['max_tokens']==512 and p['training_derived_min_output_tokens']<512 for p in plan['training'])
elif kind=='local_power':
 from pdblend.profile.local_power import load_package
 manifest,plan,model=load_package(member['package']);raw=json.loads(Path(manifest['inputs']['training_raw']['path']).read_text())
 expected_points=len(plan['points'])
 assert expected_points==12 and len(manifest['mixed_repair']['points'])==4
else:
 plan=json.loads(Path(member['plan']).read_text());raw=json.loads(Path(plan['training_source']).read_text())
 assert lc.digest(plan['training_source'])==plan['training_source_sha256']
 lc.validate_training_plan(plan,raw);expected_points=len(plan['training'])
 assert expected_points==36 and plan['training_only'] is True and not plan['fit_existing_holdout']
 manifest={'model_id':raw['model_id'],'tp':raw['tp'],'pp':raw['pp']}
spec=ModelRegistry('/models',verification_receipt=os.environ['PDBLEND_MODEL_VERIFICATION_RECEIPT']).get(member['model_id'])
spec.validate_config();spec.validate_topology(member['tp'],1)
assert (spec.model_id,spec.model_hash,spec.tokenizer_hash)==(raw['model_id'],raw['model_hash'],raw['tokenizer_hash'])
assert (manifest['model_id'],manifest['tp'],manifest['pp'])==(member['model_id'],member['tp'],1)
if kind!='local_power':
 for point in plan['training']:
  assert lc.point_capacity_error(point,raw['kv_capacity_tokens']) is None
print(json.dumps({'ready':True,'hardware_actions_started':False,'kind':kind,'model_id':spec.model_id,'tp':member['tp'],
 'expected_training_or_power_points':expected_points,'source_sha256':os.environ['PDBLEND_SOURCE_SHA256'],
 'module_file':lc.__file__,'module_sha256':lc.digest(lc.__file__),'model_hash':spec.model_hash,'tokenizer_hash':spec.tokenizer_hash}))
'''

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--jobs',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
 args=parser.parse_args();out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
 jobs=json.loads(args.jobs.read_text());source=Path(jobs[0]['payload']['source_snapshot'])
 assert len(jobs)==4 and sum(j['payload']['gpu_count'] for j in jobs)==8
 assert all(j['payload']['source_snapshot']==str(source) for j in jobs)
 helper_spec=importlib.util.spec_from_file_location('incremental_freezer_review',ROOT/'scripts/2026-09-22_enqueue_parallel_profiles.py')
 helper=importlib.util.module_from_spec(helper_spec);helper_spec.loader.exec_module(helper)
 manifest=json.loads((source/'manifest.json').read_text());helper.verify_snapshot(source,manifest['files'])
 def run(name,argv):
  start=time.time();result=subprocess.run(argv,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=180)
  (out/(name+'.log')).write_text(result.stdout)
  return dict(name=name,argv=argv,exit_code=result.returncode,elapsed_s=time.time()-start,log_sha256=sha(out/(name+'.log')))
 def preflight(job,index):
  p=job['payload'];original=p['argv'];image_index=original.index(IMAGE)
  argv=['docker','run','--rm','--runtime=runc','--network=none','--cpus','1','--workdir','/',
        '-e','NVIDIA_VISIBLE_DEVICES=void','-e','CUDA_VISIBLE_DEVICES=','--entrypoint','/opt/venv/bin/python']
  for i,arg in enumerate(original[:image_index]):
   if arg=='-v':
    mount=original[i+1]
    if '{attempt_dir}' in mount:continue
    host,target,mode=mount.rsplit(':',2)
    argv+=['-v',host+':'+target+':ro']
   elif arg=='-e':
    value=original[i+1].replace('{lease_gpu_uuids}',','.join('CPU-PREFLIGHT-'+str(k) for k in range(p['tp'])))
    argv+=['-e',value]
  argv += [IMAGE,'-B','-c',PREFLIGHT,json.dumps(p['measurement_scope'],sort_keys=True)]
  row=run('preflight-'+p['cohort_member'],argv)
  if row['exit_code']==0:
   text=(out/(row['name']+'.log')).read_text();row['receipt']=json.loads(text.splitlines()[-1])
  return row
 test_cmd=['docker','run','--rm','--runtime=runc','--network=none','--cpus','2','--workdir','/',
  '-e','NVIDIA_VISIBLE_DEVICES=void','-e','CUDA_VISIBLE_DEVICES=','-e','PYTHONDONTWRITEBYTECODE=1','-e','PYTHONPATH=/opt/pdblend-src',
  '-v',str(source)+':/opt/pdblend-src:ro','-v',str(source)+':/src:ro','-v',str(ROOT/'tests')+':/tests:ro',
  '-v',str(ROOT/'scripts')+':/scripts:ro','-v',str(out)+':/audit:rw','--entrypoint','/opt/venv/bin/python',
  IMAGE,'-B','-m','pytest','-q','-p','no:cacheprovider','--junitxml=/audit/focused-tests.xml',
  *['/tests/pdblend/'+name for name in TESTS]]
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
  futures=[pool.submit(preflight,job,index) for index,job in enumerate(jobs)]
  tests_future=pool.submit(run,'focused-tests',test_cmd)
  preflights=[f.result() for f in futures];tests=tests_future.result()
 helper.verify_snapshot(source,manifest['files'])
 no_cache=not list(source.rglob('__pycache__')) and not list(source.rglob('*.pyc'))
 counts={}
 if (out/'focused-tests.xml').exists():
  suites=ET.parse(out/'focused-tests.xml').getroot().findall('testsuite')
  counts={k:sum(int(s.get(k,0)) for s in suites) for k in ['tests','failures','errors','skipped']}
 report=dict(schema=1,jobs=str(args.jobs.resolve()),jobs_sha256=sha(args.jobs),source_sha256=source.name,
  snapshot_verified_before_and_after=True,no_source_bytecode=no_cache,preflights=preflights,tests=tests,test_counts=counts,
  hardware_executed=False,queue_modified=False,formal_eligible=False,energy_comparable=False,
  test_source_sha256={name:sha(ROOT/'tests/pdblend'/name) for name in TESTS},review_script_sha256=sha(__file__))
 report['ready']=tests['exit_code']==0 and bool(counts) and counts['skipped']==counts['errors']==counts['failures']==0 and no_cache and all(
  r['exit_code']==0 and r.get('receipt',{}).get('ready') is True for r in preflights)
 (out/'review.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps(dict(ready=report['ready'],test_counts=counts,preflights=[(r['name'],r['exit_code']) for r in preflights],review=str(out/'review.json'))))
 return 0 if report['ready'] else 1
if __name__=='__main__':raise SystemExit(main())
