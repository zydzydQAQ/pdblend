#!/usr/bin/env python3
"""CPU-only pinned image validation of an immutable resident job package."""
import concurrent.futures
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT/'results/2026-09-23/dist-eco-resident-v1'
OUT = PACKAGE/'cpu-review'
IMAGE = 'sha256:1c2d0bf96dfa752394a6aa4b5398a6105dcf060936a484a89729dcab6f9d9acc'
TESTS = ['test_resident_campaign.py','test_resident_campaign_http.py',
         'test_ecoserve_run_native.py','test_ecoserve_run_native_http.py']

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def run(name, argv):
    start=time.time()
    result=subprocess.run(argv,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=180)
    (OUT/(name+'.log')).write_text(result.stdout)
    return dict(name=name,exit_code=result.returncode,argv=argv,elapsed_s=time.time()-start,
                log_sha256=sha(OUT/(name+'.log')))
def preflight(job,index):
    payload=job['payload'];model=payload['model_id'];output=OUT/model;output.mkdir()
    source_argv=payload['argv']; image_index=source_argv.index(IMAGE)
    count=payload['gpu_count'];replace={'{lease_gpu_uuids}':','.join('CPU-PREFLIGHT-'+str(i) for i in range(count)),
        '{lease_local_indices}':','.join(map(str,range(count))),'{lease_port}':str(19000+index*100)}
    argv=['docker','run','--rm','--runtime=runc','--network=none','--cpus','1','--workdir','/',
          '-e','NVIDIA_VISIBLE_DEVICES=void','--entrypoint','/opt/venv/bin/python']
    for host,target,mode in payload['mounts']:
        argv+=['-v',f'{host}:{target}:{mode}']
    argv+=['-v',f'{output}:/output:rw']
    for i,arg in enumerate(source_argv[:image_index]):
        if arg=='-e':
            value=source_argv[i+1]
            for old,new in replace.items():value=value.replace(old,new)
            if value.startswith('CUDA_VISIBLE_DEVICES='):value='CUDA_VISIBLE_DEVICES='
            argv+=['-e',value]
    argv += [IMAGE] + [replace.get(value,value) for value in source_argv[image_index+1:]] + ['--preflight-only']
    checked=run('preflight-'+model,argv)
    receipt=output/'campaign/preflight.json'
    if receipt.exists():
        value=json.loads(receipt.read_text())
        checked.update(preflight_sha256=sha(receipt),ready=value.get('ready'),hardware_actions_started=value.get('hardware_actions_started'))
    checked['completion_created']=(output/'campaign/completion.json').exists()
    return checked

def main():
    OUT.mkdir(exist_ok=False)
    package=json.loads((PACKAGE/'manifest.json').read_text());source=Path(package['source_snapshot'])
    spec=importlib.util.spec_from_file_location('freezer',ROOT/'scripts/2026-09-22_enqueue_parallel_profiles.py')
    freezer=importlib.util.module_from_spec(spec);spec.loader.exec_module(freezer)
    files=json.loads((source/'manifest.json').read_text())['files']
    freezer.verify_snapshot(source,files)
    jobs=json.loads((PACKAGE/'jobs.json').read_text())
    tests=['/tests/independent_baselines/'+name for name in TESTS]
    test_cmd=['docker','run','--rm','--runtime=runc','--network=none','--cpus','2','--workdir','/',
      '-e','NVIDIA_VISIBLE_DEVICES=void','-e','CUDA_VISIBLE_DEVICES=','-e','PYTHONDONTWRITEBYTECODE=1',
      '-e','PYTHONPATH=/opt/pdblend-src','-v',str(source)+':/opt/pdblend-src:ro',
      '-v',str(source)+':/src:ro','-v',str(ROOT/'tests')+':/tests:ro','-v',str(OUT)+':/audit:rw',
      '--entrypoint','/opt/venv/bin/python',IMAGE,'-B','-m','pytest','-q','-p','no:cacheprovider',
      '--junitxml=/audit/http-tests.xml',*tests]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        checks=[pool.submit(preflight,job,index) for index,job in enumerate(jobs)]
        tests_future=pool.submit(run,'http-tests',test_cmd)
        preflights=[future.result() for future in checks]
        test=tests_future.result()
    help_cmd=test_cmd[:test_cmd.index('-m')]+['-m','pdblend_baselines.resident_campaign','--help']
    help_result=run('cli-help',help_cmd)
    freezer.verify_snapshot(source,files)
    no_cache=not list(source.rglob('__pycache__')) and not list(source.rglob('*.pyc'))
    result=dict(schema='dist-eco-resident-cpu-review-v1',hardware_executed=False,enqueued=False,
        package_manifest_sha256=sha(PACKAGE/'manifest.json'),jobs_sha256=sha(PACKAGE/'jobs.json'),
        source_sha256=package['source_sha256'],snapshot_verified_before_and_after=True,no_source_bytecode=no_cache,
        tests=test,preflights=preflights,cli_help=help_result,
        test_source_sha256={name:sha(ROOT/'tests/independent_baselines'/name) for name in TESTS},
        review_script_sha256=sha(__file__),formal_eligible=False,energy_comparable=False)
    result['ready']=all(row['exit_code']==0 and row.get('ready') is True and row.get('hardware_actions_started') is False and not row['completion_created'] for row in preflights) and test['exit_code']==0 and help_result['exit_code']==0 and no_cache
    (OUT/'review.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(ready=result['ready'],preflights=[(r['name'],r['exit_code']) for r in preflights],tests=test['exit_code'],review=str(OUT/'review.json'))))
    return 0 if result['ready'] else 1
if __name__=='__main__':raise SystemExit(main())
