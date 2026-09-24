#!/usr/bin/env python3
"""Prepare the qualified PD matrix and baseline facts; --enqueue opts into leases.

This external handoff never edits frozen runtime code, historical receipts, or
the baseline jobs file. No GPU command is executed by this process. Its optional
image preflight runs under runc with no GPU devices or network.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
DRIVER=ROOT/'scripts/2026-09-24_drive_low_m_tuning.py'
MATRIX=ROOT/'scripts/2026-09-24_prepare_repaired_standard_matrix.py'
HELPER=ROOT/'scripts/2026-09-24_prepare_energy_repair.py'
spec=importlib.util.spec_from_file_location('frozen_low_m_driver_helpers',DRIVER)
d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)
SCHEMA='pdblend-repair-followup/v1'
DONE={'enqueued_awaiting_result_review','awaiting_pd_optimization','needs_diagnosis'}


def source_ref_for_job(job):
    values=[arg.split('=',1)[1] for arg in job['payload']['argv'] if arg.startswith('PDBLEND_SOURCE_MANIFEST=')]
    if len(values)!=1:raise ValueError('job must name exactly one frozen source manifest')
    ref=d.binding(values[0]);source=d.verify_source(ref)
    if source.name!=job['payload']['source_sha256']:raise ValueError('job frozen source revision differs')
    return ref


def package_jobs(package,*,points,jobs,system=None):
    package=Path(package).resolve();campaign_ref=d.binding(package/'campaign.json')
    campaign=d.read_bound(campaign_ref);jobs_ref=d.binding(package/'jobs.json');specs=d.read_bound(jobs_ref)
    if len(campaign['points'])!=points or len(specs)!=jobs or len(campaign['groups'])!=jobs:
        raise ValueError('execution package point/job count differs')
    if len({point['name'] for point in campaign['points']})!=points:
        raise ValueError('execution package has duplicate points')
    if len({job['job_id'] for job in specs})!=jobs:raise ValueError('duplicate package job IDs')
    groups={group['session_id']:group for group in campaign['groups']};covered=[];inputs=[]
    for index,job in enumerate(specs):
        payload=job['payload'];argv=payload['argv']
        if (payload.get('gpu_count')!=8 or payload.get('exclusive') is not True
                or payload.get('reserve_host') is not True or payload.get('required_receipts')!=['session/completion.json']
                or payload.get('depends_on',[]) or job.get('max_attempts')!=1):
            raise ValueError('package lost exclusive eight-board single-attempt receipt contract')
        if index and payload.get('after_terminal')!=[specs[index-1]['job_id']]:
            raise ValueError('package jobs must retain their serial chain')
        if argv.count('--group')!=1:raise ValueError('package job needs one explicit group')
        group_ref=d.binding(argv[argv.index('--group')+1]);group=d.read_bound(group_ref)
        if group!=groups.get(payload['session_id']):raise ValueError('job group differs from frozen campaign')
        if system and {point['system'] for point in group['points']}!={system}:
            raise ValueError('matrix contains a different system')
        if not system and any(point['system']=='pdblend' for point in group['points']):
            raise ValueError('baseline package contains PDblend points')
        covered.extend(group['points']);inputs.append(dict(group=group_ref,source=source_ref_for_job(job)))
    if {point['name']:point for point in covered}!={point['name']:point for point in campaign['points']} or len(covered)!=points:
        raise ValueError('job groups do not exactly cover the frozen point cohort')
    prep_ref=d.binding(package/'preparation.json');prep=d.read_bound(prep_ref)
    if prep.get('campaign')!=campaign_ref or prep.get('jobs')!=jobs_ref:
        raise ValueError('preparation receipt differs from campaign/jobs bytes')
    return dict(campaign=campaign_ref,jobs=jobs_ref,preparation=prep_ref,inputs=inputs),campaign,specs


def verify_baseline_preflight(package,specs):
    ref=d.binding(Path(package)/'container-preflight.json');rows=d.read_bound(ref)
    if len(rows)!=len(specs) or {row['job_id'] for row in rows}!={job['job_id'] for job in specs}:
        raise ValueError('baseline image preflight omitted a job')
    total=0
    for row in rows:
        data=json.loads(row['stdout'].splitlines()[-1])
        if row['returncode']!=0 or row.get('hardware_executed') is not False or data.get('passed') is not True:
            raise ValueError('baseline image preflight did not pass')
        total+=data['points']
    if total!=23:raise ValueError('baseline image preflight does not cover all 23 points')
    return ref


def floor_structure(package,state):
    if state.get('status')!='finished':raise ValueError('low-M driver has no finished complete qualification')
    if state.get('package_identity')!=package['identity']:raise ValueError('low-M state belongs to a different package')
    result=state['result']
    if result.get('unqualified_rates') or not result.get('floor'):raise ValueError('low-M driver lacks a floor for every rate')
    artifact=d.read_bound(result['floor']);summary=d.read_bound(result['summary'])
    rates={trial['nominal_rate_rps'] for trial in package['manifest']['trials']}
    floors=artifact.get('floors',[])
    if (artifact.get('kind')!='pdblend_capacity_floor_v2' or artifact.get('formal_eligible') is not False
            or artifact.get('tuning_manifest')!=package['tuning_manifest']
            or artifact.get('identity')!=package['manifest']['identity']
            or len(floors)!=len(rates) or {row['nominal_rate_rps'] for row in floors}!=rates
            or summary.get('selected')!=floors or set(result['qualified_rates'])!=rates):
        raise ValueError('capacity floor identity or full nominal workload coverage differs')
    required={(seed,family) for seed in package['manifest']['seeds'] for family in package['manifest']['families']}
    lookup={row['trial_id']:row for row in summary['rows']}
    for floor in floors:
        rows=[lookup[identity] for identity in floor['trial_ids']]
        if len(rows)!=18 or len(required)!=18 or {(row['seed'],row['family']) for row in rows}!=required:
            raise ValueError('capacity floor lacks its complete 18 raw trials')
    if artifact['trial_receipts']!=[row['receipt'] for row in summary['rows']]:
        raise ValueError('capacity floor trial receipts differ from original summary')
    return artifact,summary


def validate_floor(package,state,*,python):
    """The same frozen validator rechecks raw evidence, including rejected domains."""
    artifact,summary=floor_structure(package,state)
    for ref in artifact['trial_receipts']:d.read_bound(ref)
    raw=d.frozen_call(package,'summarize',python=python,receipts=artifact['trial_receipts'])
    if raw['rejected'] or raw['selected']!=artifact['floors'] or raw['rows']!=summary['rows']:
        raise ValueError('original capacity floor no longer agrees with frozen raw validation')
    return dict(passed=True,floor=state['result']['floor'],summary=state['result']['summary'],
                source_manifest=package['source_manifest'],tuning_manifest=package['tuning_manifest'],
                qualified_rates=state['result']['qualified_rates'],hardware_executed=False)


def prepare_matrix(package,parent,out,floor,predecessor,*,python):
    argv=[str(python),'-B',str(MATRIX),'--parent',str(parent),'--out',str(out),
          '--source-manifest',package['source_manifest']['path'],'--capacity-floor',floor['path']]
    if predecessor:argv.extend(['--after-terminal',predecessor])
    env=dict(os.environ,PYTHONPATH=str(Path(package['source_manifest']['path']).parent),PYTHONDONTWRITEBYTECODE='1')
    process=subprocess.run(argv,text=True,capture_output=True,env=env,cwd=ROOT,timeout=3600)
    invocation=dict(argv=argv,returncode=process.returncode,stdout=process.stdout,stderr=process.stderr,hardware_executed=False)
    d.write_once(Path(out).parent/(Path(out).name+'-prepare-invocation.json'),invocation)
    if process.returncode:raise RuntimeError('standard matrix preparation failed: '+process.stderr[-4000:])


IMAGE_CHECK=r'''
import hashlib,json,sys
from pathlib import Path
from types import SimpleNamespace
from pdblend.bench.comparison_campaign import load_bound
from pdblend.bench.comparison_pdblend_observation import validate_observation_inputs
from pdblend.bench.comparison_runtime import pdblend_window_resources
from pdblend.bench.resident_session import digest
campaign=load_bound(json.loads(sys.argv[1]));ref=campaign['execution_source_manifest'];source=Path(ref['path']).parent
manifest=load_bound(ref)
assert digest(manifest['files'])==source.name
for name,sha in manifest['files'].items():assert hashlib.sha256((source/name).read_bytes()).hexdigest()==sha
import pdblend.bench.comparison_runtime as runtime
assert Path(runtime.__file__).resolve().is_relative_to(Path('/opt/pdblend-src'))
records=[]
for group in campaign['groups']:
 assert len(group['points'])==12
 specs=[SimpleNamespace(tp=row['tp'],pp=row['pp'],generation=0) for row in group['engine_identity']['instances']]
 for point in group['points']:
  checked=validate_observation_inputs(point,point['inputs']);loaded,plan=pdblend_window_resources(point,specs)
  assert checked['formal_eligible'] is False and point['source_manifest']==ref
  records.append(dict(point=point['name'],tp=plan.tp,counts=plan.counts,frequency_mhz=plan.f_M))
assert len(records)==36 and len(campaign['groups'])==3
print(json.dumps(dict(passed=True,points=36,jobs=3,source_sha256=source.name,hardware_executed=False,plans=records)))
'''


def image_command(package,matrix_ref,jobs):
    images={job['payload']['image_digest'] for job in jobs}
    if len(images)!=1:raise ValueError('standard matrix jobs differ in runtime image')
    source=Path(package['source_manifest']['path']).parent
    return ['docker','run','--rm','--runtime','runc','--network','none','--cpus','1','--memory','4g',
        '--entrypoint','/opt/venv/bin/python','-v',str(source)+':/opt/pdblend-src:ro',
        '-v',str(ROOT)+':'+str(ROOT)+':ro','-v','/home/models:/models:ro',
        '-e','NVIDIA_VISIBLE_DEVICES=void','-e','PYTHONPATH=/opt/pdblend-src',
        '-e','PYTHONDONTWRITEBYTECODE=1','-e','OMP_NUM_THREADS=1','-e','OPENBLAS_NUM_THREADS=1',
        '-e','PDBLEND_SOURCE_MANIFEST='+package['source_manifest']['path'],
        '-e','PDBLEND_SOURCE_SHA256='+source.name,next(iter(images)),'-B','-c',IMAGE_CHECK,json.dumps(matrix_ref)]


def preflight_matrix(package,refs,jobs,output):
    path=Path(output)/'matrix-image-preflight.json'
    argv=image_command(package,refs['campaign'],jobs)
    if path.exists():
        receipt=json.loads(path.read_text())
        if receipt['argv']!=argv or receipt['campaign']!=refs['campaign']:raise ValueError('cached matrix preflight binding differs')
    else:
        process=subprocess.run(argv,text=True,capture_output=True,timeout=3600)
        parsed=json.loads(process.stdout.splitlines()[-1]) if process.returncode==0 else None
        receipt=dict(argv=argv,campaign=refs['campaign'],returncode=process.returncode,stdout=process.stdout,
                     stderr=process.stderr,result=parsed,hardware_executed=False)
        d.write_once(path,receipt)
    result=receipt.get('result') or {}
    if (receipt['returncode']!=0 or result.get('passed') is not True or result.get('hardware_executed') is not False
            or result.get('points')!=36 or result.get('jobs')!=3
            or result.get('source_sha256')!=jobs[0]['payload']['source_sha256']):
        raise ValueError('matrix image CPU preflight did not pass its complete frozen cohort')
    return d.binding(path)


def serial_specs(matrix_jobs,baseline_jobs,predecessor=None):
    """Return queue specifications, retaining the baseline package byte for byte."""
    result=deepcopy(matrix_jobs)+deepcopy(baseline_jobs)
    for index,job in enumerate(result):
        job['payload']['after_terminal']=[result[index-1]['job_id']] if index else ([predecessor] if predecessor else [])
        job['payload'].setdefault('depends_on',[])
    return result


def released_predecessor(queue,driver_state):
    identity=driver_state['stages'][-1]['job_id'];job=queue.get('jobs',{}).get(identity)
    if (job is None or job.get('status') not in d.TERMINAL or job.get('lease_id') is not None
            or any(lease.get('job_id')==identity and lease.get('status')=='active' for lease in queue.get('leases',{}).values())):
        return False,None
    # The existing lease queue deliberately excludes blocked from after_terminal.
    # A verified released blocked attempt needs no impossible dependency edge.
    return True,identity if job['status'] in ('succeeded','failed','cancelled') else None


def load_inputs(low_m,baseline,parent):
    package=d.load_package(low_m);baseline_refs,baseline_campaign,baseline_jobs=package_jobs(baseline,points=23,jobs=4)
    preflight=verify_baseline_preflight(baseline,baseline_jobs)
    identity=dict(low_m=package['identity'],baseline=baseline_refs,baseline_preflight=preflight,
        matrix_parent=d.binding(parent),driver=d.binding(DRIVER),preparer=d.binding(MATRIX),helper=d.binding(HELPER),
        followup=d.binding(Path(__file__)))
    return dict(package=package,baseline_refs=baseline_refs,baseline_jobs=baseline_jobs,identity=identity,parent=Path(parent))


def verify_inputs(inputs):
    identity=inputs['identity']
    for key in ('matrix_parent','driver','preparer','helper','followup','baseline_preflight'):
        ref=identity[key]
        if d.binding(ref['path'])!=ref:raise ValueError('followup input bytes changed: '+key)
    refs=inputs['baseline_refs']
    for key in ('campaign','jobs','preparation'):d.read_bound(refs[key])
    for item in refs['inputs']:
        d.read_bound(item['group']);d.verify_source(item['source'])


def validate_matrix(package,refs,campaign,predecessor):
    protocol=campaign['repair_protocol']
    if (campaign['execution_source_manifest']!=package['source_manifest']
            or protocol['capacity_floor'] is None or len(protocol['capacity_points'])!=2
            or protocol.get('frozen_source_reused') is not True or protocol.get('canonical_startup_for_capacity_v2') is not True
            or any(item['source']!=package['source_manifest'] for item in refs['inputs'])):
        raise ValueError('standard matrix did not reuse the qualified algorithm and two measured low-M domains')


def tick(inputs,driver_path,queue_path,output,*,python,enqueue=False):
    output=Path(output);state_path=output/'followup-status.json';package=inputs['package']
    state=json.loads(state_path.read_text()) if state_path.exists() else dict(schema=SCHEMA,
        input_identity=inputs['identity'],status='waiting_low_m',actions=[],submitted=[],created_s=time.time())
    paths=dict(queue=str(Path(queue_path).resolve()),driver_state=str(Path(driver_path).resolve()))
    if not state_path.exists():state['execution_paths']=paths
    if state.get('schema')!=SCHEMA or state.get('input_identity')!=inputs['identity']:
        raise ValueError('followup state belongs to different immutable inputs or implementation')
    if state.get('execution_paths')!=paths:raise ValueError('followup state belongs to another queue or driver path')
    if state['status'] in DONE:return state
    try:
        if not state.get('driver_snapshot'):
            if not Path(driver_path).exists():d.save_state(state_path,state);return state
            driver_state=json.loads(Path(driver_path).read_text())
            if driver_state.get('status') not in ('finished','needs_diagnosis'):d.save_state(state_path,state);return state
            if driver_state.get('package_identity')!=package['identity']:raise ValueError('terminal low-M state package mismatch')
            if driver_state.get('queue_path')!=paths['queue']:raise ValueError('terminal low-M state uses a different lease queue')
            released,predecessor=released_predecessor(json.loads(Path(queue_path).read_text()),driver_state)
            if not released:
                state['status']='waiting_low_m_release';d.save_state(state_path,state);return state
            state['driver_snapshot']=d.write_once(output/'low-m-terminal-state.json',driver_state)
            state['predecessor']=predecessor
            d.save_state(state_path,state)
        driver_state=d.read_bound(state['driver_snapshot'])
        verify_inputs(inputs)
        if not state.get('qualification'):
            try:
                verdict=validate_floor(package,driver_state,python=python)
                state['mode']='qualified_matrix_then_baselines'
            except (ValueError,KeyError,TypeError,OSError,RuntimeError) as exc:
                verdict=dict(passed=False,reason=str(exc),hardware_executed=False)
                state['mode']='baseline_facts_only'
            state['qualification']=d.write_once(output/'floor-validation.json',verdict)
            d.save_state(state_path,state)
        d.read_bound(state['qualification'])
        matrix_jobs=[]
        if state['mode']=='qualified_matrix_then_baselines':
            if not state.get('matrix_path'):
                state['matrix_path']=str((output/'standard-matrix').resolve());d.save_state(state_path,state)
            target=Path(state['matrix_path'])
            if target.exists() and not all((target/name).exists() for name in ('campaign.json','jobs.json','preparation.json')):
                # Preserve an interrupted builder's partial files and choose a fresh output.
                number=1
                while (output/f'standard-matrix-recovery-{number:03d}').exists():number+=1
                target=output/f'standard-matrix-recovery-{number:03d}'
                state['matrix_path']=str(target.resolve());d.save_state(state_path,state)
            if not target.exists():
                prepare_matrix(package,inputs['parent'],target,driver_state['result']['floor'],state['predecessor'],python=python)
            refs,campaign,matrix_jobs=package_jobs(target,points=36,jobs=3,system='pdblend')
            validate_matrix(package,refs,campaign,state['predecessor'])
            if state.get('matrix') and state['matrix']!=refs:raise ValueError('prepared matrix inputs changed')
            if campaign['parent_campaign']!=inputs['identity']['matrix_parent']:raise ValueError('matrix original parent differs')
            if campaign['repair_protocol']['capacity_floor']!=driver_state['result']['floor']:
                raise ValueError('matrix selected a different capacity floor')
            state['matrix']=refs
            state['matrix_preflight']=preflight_matrix(package,refs,matrix_jobs,output)
        if not state.get('queue_specs'):
            specs=serial_specs(matrix_jobs,inputs['baseline_jobs'],state['predecessor'])
            state['queue_specs']=d.write_once(output/'derived-queue-specs.json',dict(
                jobs=specs,baseline_original=inputs['baseline_refs']['jobs'],matrix=state.get('matrix'),
                predecessor=state['predecessor'],low_m_state=state['driver_snapshot'],formal_eligible=False))
            state['status']='prepared';d.save_state(state_path,state)
        spec_doc=d.read_bound(state['queue_specs']);specs=spec_doc['jobs']
        if specs!=serial_specs(matrix_jobs,inputs['baseline_jobs'],state['predecessor']):
            raise ValueError('bound queue batch differs from reviewed original execution packages')
        if not enqueue:return state
        queue=json.loads(Path(queue_path).read_text())
        for job in specs:
            existing=queue.get('jobs',{}).get(job['job_id'])
            if existing is not None:d.assert_same_job(existing,job)
        for job in specs:
            queue=json.loads(Path(queue_path).read_text());existing=queue.get('jobs',{}).get(job['job_id'])
            if existing is not None:d.assert_same_job(existing,job)
            else:
                # The entire reviewed batch is durable before the first enqueue.
                # If interrupted after enqueue, exact queue reconciliation wins.
                d.frozen_call(package,'enqueue',python=python,queue=str(Path(queue_path).resolve()),job=job)
            if job['job_id'] not in state['submitted']:
                state['submitted'].append(job['job_id']);d.save_state(state_path,state)
        state['status']='awaiting_pd_optimization' if state['mode']=='baseline_facts_only' else 'enqueued_awaiting_result_review'
        state['goal_complete']=False
    except Exception as exc:
        state.update(status='needs_diagnosis',diagnosis=repr(exc),goal_complete=False)
    state['updated_s']=time.time();d.save_state(state_path,state)
    return state


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--low-m-package',type=Path,required=True)
    parser.add_argument('--driver-state',type=Path)
    parser.add_argument('--queue',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--baseline-package',type=Path,default=ROOT/'results/2026-09-24/pdblend-energy-repair-v1/baseline-supplements')
    parser.add_argument('--matrix-parent',type=Path,default=ROOT/'results/2026-09-24/profile-saturation-round-v3/campaign.json')
    parser.add_argument('--python',type=Path,default=Path(sys.executable))
    mode=parser.add_mutually_exclusive_group();mode.add_argument('--once',action='store_true');mode.add_argument('--watch',action='store_true')
    parser.add_argument('--enqueue',action='store_true',help='enable reviewed follow-up lease enqueueing')
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    inputs=load_inputs(args.low_m_package,args.baseline_package,args.matrix_parent)
    driver_path=args.driver_state or args.low_m_package/'driver/state.json';prior=None
    while True:
        with (args.out/'followup.lock').open('a+') as lock:
            fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
            state=tick(inputs,driver_path,args.queue,args.out,python=args.python,enqueue=args.enqueue)
        view={key:state.get(key) for key in ('status','mode','submitted','qualification','queue_specs','diagnosis')}
        signature=d.digest(view)
        if signature!=prior:print(json.dumps(view,sort_keys=True),flush=True);prior=signature
        if not args.watch or state['status'] in DONE or (state['status']=='prepared' and not args.enqueue):break
        time.sleep(5.)
    return 1 if state['status']=='needs_diagnosis' else 0


if __name__=='__main__':raise SystemExit(main())
