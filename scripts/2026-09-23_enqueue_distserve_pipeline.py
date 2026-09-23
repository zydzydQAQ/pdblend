#!/usr/bin/env python3
"""Queue real independent DistServe pipelines without repeating profiles."""
import argparse
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from pdblend.experimentation.lease import GPULeaseQueue


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec',type=Path,required=True)
    p.add_argument('--prepare-only',action='store_true')
    args=p.parse_args();report=deepcopy(json.loads(args.spec.read_text()))
    root=ROOT/'results/2026-09-22/three-model'
    queue=GPULeaseQueue(root/'queue.json');state=queue.snapshot()
    freeze=module('freeze',ROOT/'scripts/2026-09-22_enqueue_parallel_profiles.py')
    native=module('native',ROOT/'scripts/2026-09-23_enqueue_native_acceptance.py')
    changed=['pdblend_runtime/serve.py','pdblend_runtime/native_v1.py',
        *['pdblend_baselines/distserve/'+name for name in
          ('runtime.py','request_runtime.py','run_native.py','gpu_probe.py')]]
    with tempfile.TemporaryDirectory(prefix='distserve-pipeline-source-') as tmp:
        staging=Path(tmp)/'src';shutil.copytree(report['source'],staging)
        for path in changed:shutil.copy2(ROOT/'src'/path,staging/path)
        source,sha=freeze.freeze_source(staging,root/'native-acceptance-sources')
    verification,_=freeze.freeze_receipt(root/'model-verification.json',root/'profile-receipts')
    jobs=native.build_jobs('probe',root=root,source=source,source_hash=sha,
        image=report['image_digest'],verification=verification,depends=[])
    existing={j['payload']['model_id']:j['job_id'] for j in report['jobs'] if j['payload'].get('kind')=='probe'}
    for job in jobs:
        old=job['job_id'];new=old.replace('native-probe-','native-distserve-pipeline-')
        job['job_id']=new;job['priority']=395
        payload=job['payload'];payload.update(kind='distserve_pipeline',container_name=new,
            depends_on=[existing[payload['model_id']]],timeout_s=1200,
            profiles_collected=False,evidence_class='independent_native_request_pipeline')
        argv=payload['argv'];argv[argv.index('--name')+1]=new
        argv[argv.index('pdblend_runtime.probe')]='pdblend_baselines.distserve.gpu_probe'
    changes=[]
    for job in report['jobs']:
        if '-quad-' not in job['job_id']:continue
        old=job['job_id'];current=state['jobs'][old]
        if current['status']!='queued' or current['attempts']:
            raise ValueError('protected profile wave already started')
        # Keep names bounded even after several audited dependency repairs.
        suffix=hashlib.sha256((old+sha).encode()).hexdigest()[:16]
        new=old.split('-quad-')[0]+'-quad-after-pipeline-'+suffix
        job['job_id']=new;payload=job['payload'];payload['container_name']=new
        payload['depends_on'] += [j['job_id'] for j in jobs]
        argv=payload['argv'];argv[argv.index('--name')+1]=new
        changes.append((old,job))
    report['jobs']+=jobs;report['parent_spec']=str(args.spec.resolve())
    report.setdefault('source_variants',{})['distserve_pipeline']=dict(path=str(source),sha256=sha,files=changed)
    output=root/f'native-with-distserve-pipeline-{sha[:16]}.json'
    output.write_text(json.dumps(report,indent=2)+'\n')
    if not args.prepare_only:
        for old,_ in changes:queue.block(old,reason='Protected profile wave follows independent DistServe pipeline GPU checks')
        for _,job in changes:queue.enqueue(**job)
        for job in jobs:queue.enqueue(**job)
    print(json.dumps(dict(spec=str(output),source_sha256=sha,
        jobs=[j['job_id'] for j in jobs],prepared_only=args.prepare_only),indent=2))


if __name__=='__main__':main()
