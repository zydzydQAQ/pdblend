#!/usr/bin/env python3
"""Replace only unstarted native probes, then schedule a protected profile wave."""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from pdblend.experimentation.lease import GPULeaseQueue


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args()
    root=ROOT/'results/2026-09-22/three-model'
    queue=GPULeaseQueue(root/'queue.json')
    state=queue.snapshot()
    old=[j for j in state['jobs'].values() if j['status']=='queued' and
         j['job_id'].startswith(('native-probe-','native-dynamo-','native-eco4-'))]
    if len(old)!=9 or any(j['attempts']!=0 or j.get('lease_id') for j in old):
        raise ValueError('expected exactly nine unstarted native jobs')
    depends=sorted(j['job_id'] for j in state['jobs'].values() if
        j['job_id'].startswith(('holdout-7b-tp4-22436','holdout-32b-tp4-5d5')) and
        j['job_id'].endswith('shared-bc3b8b870464') and j['status'] in ('running','succeeded'))
    if len(depends)!=2:
        raise ValueError('current protected TP4 calibration wave not found')
    freezing=module('freeze',ROOT/'scripts/2026-09-22_enqueue_parallel_profiles.py')
    native=module('native',ROOT/'scripts/2026-09-23_enqueue_native_acceptance.py')
    source,sha=freezing.freeze_source(ROOT/'src',root/'native-acceptance-sources')
    verification,_=freezing.freeze_receipt(root/'model-verification.json',root/'profile-receipts')
    image=subprocess.check_output(['docker','image','inspect','pdblend:l20-cu128-vllm-v1','--format','{{.Id}}'],text=True).strip()
    kw=dict(root=root,source=source,source_hash=sha,image=image,verification=verification,depends=depends)
    probes=native.build_jobs('probe',**kw)
    if any(j['job_id'] in {x['job_id'] for x in old} for j in probes):
        raise ValueError('source is unchanged; keep the existing queued native jobs')
    # Model-specific independent mechanisms follow their resident-pair probes.
    mechanisms=[]
    by_model={j['payload']['model_id']:j['job_id'] for j in probes}
    for kind in ('dynamo','eco4'):
        for job in native.build_jobs(kind,**kw):
            job['payload']['depends_on']=[by_model[job['payload']['model_id']]]
            mechanisms.append(job)
    jobs=probes+mechanisms
    quad=json.loads((ROOT/'results/2026-09-23/quad-wave3-review-9d2bc444f986-v2/jobs.json').read_text())
    old_quad=[j for j in state['jobs'].values() if j['status']=='queued' and
              any(j['job_id'].startswith(x['job_id']) for x in quad)]
    if old_quad:
        if len(old_quad)!=4 or any(j['attempts'] or j.get('lease_id') for j in old_quad):
            raise ValueError('profile phase replacement requires four unstarted members')
        for j in quad:
            j['job_id']+='-after-'+sha[:12]
            j['payload']['container_name']=j['job_id']
            a=j['payload']['argv'];a[a.index('--name')+1]=j['job_id']
        old+=old_quad
    # This stops a completed short member from being backfilled by a different
    # GPU workload while its peers still rely on the qualified cohort layout.
    phase_dependencies=[j['job_id'] for j in jobs]
    for job in quad:
        job['payload']['depends_on']=phase_dependencies
    jobs+=quad
    report=dict(source=str(source),source_sha256=sha,image_digest=image,
        replaces=[j['job_id'] for j in old],jobs=jobs,
        phases=['current_tp4_holdouts','native_pairs_and_independent_mechanisms','qualified_quad_profile_wave'],
        formal_eligible=False)
    out=root/f'native-and-quad-spec-{sha[:16]}.json'
    out.write_text(json.dumps(report,indent=2)+'\n')
    if not args.prepare_only:
        # Current TP4 dependencies are not complete, so these pending jobs
        # cannot be claimed during the replacement. Never touch a live lease.
        fresh=queue.snapshot()
        if all(fresh['jobs'][n]['status']=='succeeded' for n in depends):
            raise RuntimeError('TP4 wave completed; replan against current live leases before replacement')
        for j in old:
            current=fresh['jobs'][j['job_id']]
            if current['status']!='queued' or current['attempts'] or current.get('lease_id'):
                raise RuntimeError('native job started during preparation')
        for j in old:
            queue.block(j['job_id'],reason='Replaced before execution: native startup, physical GPU mapping, carry token, and owned cleanup fixes; previous immutable spec retained.')
        for j in jobs:
            queue.enqueue(**j)
    print(json.dumps(dict(prepared_only=args.prepare_only,spec=str(out),source_sha256=sha,
                         native_jobs=len(probes)+len(mechanisms),profile_jobs=len(quad)),indent=2))


if __name__=='__main__':main()
