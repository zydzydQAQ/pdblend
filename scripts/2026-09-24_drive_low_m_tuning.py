#!/usr/bin/env python3
"""Drive an immutable low-M experiment; queue mutation requires --enqueue.

The manager is deliberately outside src: GPU jobs and every evidence verdict
continue to use the package's frozen Python implementation. It never claims an
evaluation/baseline result and never modifies runtime code, profiles or traces.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

SCOPE = 'independent_low_m_tuning/v2'
SCHEMA = 'pdblend-low-m-driver/v1'
TERMINAL = {'succeeded','failed','cancelled','blocked'}


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def binding(path):
    path=Path(path).resolve()
    return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def read_bound(ref):
    if binding(ref['path']) != ref:
        raise ValueError('immutable binding differs: '+str(ref.get('path')))
    return json.loads(Path(ref['path']).read_text())


def write_once(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError('refusing to replace immutable driver artifact: '+str(path))
        return binding(path)
    with path.open('x') as stream:
        json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n')
    return binding(path)


def save_state(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix='.'+path.name+'.',dir=path.parent)
    try:
        with os.fdopen(fd,'w') as stream:
            json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n')
            stream.flush();os.fsync(stream.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)


def candidate_trials(manifest,rate,frequency):
    return [trial for trial in manifest['trials'] if trial['nominal_rate_rps']==rate
            and trial['plan']['f_M']==frequency]


def next_decision(manifest,records):
    """Pure scheduling policy: nominal screening, then complete qualification.

    A failed nominal stops that rate's downward frequency sweep. Energy ranks
    only the same nominal selection-seed trace; stress is a feasibility gate.
    """
    rates=sorted({trial['nominal_rate_rps'] for trial in manifest['trials']})
    seed=manifest['selection_seed'];screen_next=[];by_rate={}
    for rate in rates:
        nominal={trial['plan']['f_M']:trial for trial in manifest['trials']
                 if trial['nominal_rate_rps']==rate and trial['seed']==seed and trial['family']=='nominal'}
        passed=[];failure=None
        for frequency in sorted(nominal,reverse=True):
            trial=nominal[frequency];record=records.get(trial['id'])
            if record is None:
                screen_next.append(trial['id']);break
            if record.get('accepted') is not True:
                failure=dict(trial_id=trial['id'],frequency_mhz=frequency,reason=record.get('reason'))
                break
            energy=record.get('energy_service_tail_j')
            if type(energy) not in (int,float) or not 0<energy<float('inf'):
                raise ValueError('accepted nominal lacks finite positive full energy')
            passed.append(dict(frequency_mhz=frequency,energy_service_tail_j=energy,trial_id=trial['id']))
        by_rate[str(rate)]=dict(rate_rps=rate,nominal_passed=passed,nominal_stop=failure)
    if screen_next:
        return dict(kind='screen',trial_ids=screen_next,rate_decisions=by_rate,
                    reason='next_lower_frequency_for_each_rate_without_a_failed_nominal')
    qualification=[]
    for rate in rates:
        info=by_rate[str(rate)];passed=info['nominal_passed']
        if not passed:
            info.update(candidates=[],qualified=[],reason='no_accepted_nominal_candidate');continue
        best=min(passed,key=lambda row:(row['energy_service_tail_j'],row['frequency_mhz']))
        higher=[row for row in passed if row['frequency_mhz']>best['frequency_mhz']]
        candidates=[best]+([min(higher,key=lambda row:row['frequency_mhz'])] if higher else [])
        info['candidates']=[];info['qualified']=[]
        chosen_pending=None
        for candidate in candidates:
            trials=candidate_trials(manifest,rate,candidate['frequency_mhz'])
            required={(s,f) for s in manifest['seeds'] for f in manifest['families']}
            if len(trials)!=len(required) or {(t['seed'],t['family']) for t in trials}!=required:
                raise ValueError('candidate does not contain the exact complete independent seed/family domain')
            rejected=[t['id'] for t in trials if t['id'] in records and records[t['id']].get('accepted') is not True]
            missing=[t['id'] for t in trials if t['id'] not in records]
            status='rejected' if rejected else 'pending' if missing else 'qualified'
            info['candidates'].append(dict(candidate,status=status,rejected_trial_ids=rejected,missing_trial_ids=missing))
            if status=='qualified':info['qualified'].append(candidate['frequency_mhz'])
            elif status=='pending' and chosen_pending is None:chosen_pending=missing
        if chosen_pending:qualification.extend(chosen_pending)
    if qualification:
        return dict(kind='qualify',trial_ids=qualification,rate_decisions=by_rate,
                    reason='complete_remaining_seed_family_trials_for_best_or_adjacent_conservative_candidate')
    return dict(kind='finalize',trial_ids=[],rate_decisions=by_rate,
                unqualified_rates=[info['rate_rps'] for info in by_rate.values() if not info['qualified']],
                reason='all_selected_candidates_qualified_or_rejected')


def replace_arg(argv,name,value):
    argv=list(argv)
    if argv.count(name)!=1:raise ValueError('template needs exactly one '+name)
    index=argv.index(name);argv[index+1]=value
    return argv


def derived_stage(template,base_group,*,directory,index,kind,trial_ids,predecessor):
    """Build one content-bound resident job without enqueueing or executing it."""
    if not trial_ids or len(trial_ids)!=len(set(trial_ids)):raise ValueError('empty or duplicate stage trials')
    directory=Path(directory).resolve();group=deepcopy(base_group)
    group.pop('session_id',None)
    group.update(driver_stage=index,driver_kind=kind,trial_ids=list(trial_ids))
    group['session_id']=f'low-m-stage-{index:03d}-'+digest(group)[:16]
    group_path=directory/f'stage-{index:03d}-group.json'
    group_ref=write_once(group_path,group)
    job=deepcopy(template);job_id='comparison-7b-'+digest(group)[:16]
    job.update(job_id=job_id,priority=800,max_attempts=1)
    payload=job['payload'];argv=payload['argv']
    if 'pdblend.bench.low_m_tuning_runtime' not in argv:
        raise ValueError('template does not execute the frozen independent tuning module')
    argv=replace_arg(argv,'--name',job_id);argv=replace_arg(argv,'--group',str(group_path))
    if argv.count('--trial-ids')!=1:raise ValueError('template needs one explicit trial list')
    left=argv.index('--trial-ids');right=left+1
    while right<len(argv) and not argv[right].startswith('--'):right+=1
    argv=argv[:left+1]+list(trial_ids)+argv[right:]
    payload.update(argv=argv,container_name=job_id,session_id=group['session_id'],
                   trial_ids=list(trial_ids),after_terminal=[predecessor])
    # A qualification stage may contain 34 independent service windows. Keep
    # its execution bound explicit instead of inheriting the two-trial limit.
    payload['timeout_s']=max(payload.get('timeout_s',0),1200+len(trial_ids)*450)
    payload.setdefault('depends_on',[])
    if (payload.get('gpu_count')!=8 or payload.get('exclusive') is not True
            or payload.get('reserve_host') is not True or payload.get('required_receipts')!=['session/completion.json']):
        raise ValueError('stage lost its exclusive eight-board/receipt contract')
    spec_ref=write_once(directory/f'stage-{index:03d}-job.json',job)
    return dict(index=index,kind=kind,job_id=job_id,trial_ids=list(trial_ids),job=job,
                group=group_ref,spec=spec_ref,processed=False,status='prepared')


BRIDGE = r'''
import hashlib,json,sys
from pathlib import Path
task=json.load(sys.stdin)
source=Path(task['source']).resolve()
import pdblend.bench.capacity_floor_v2 as audit
if not Path(audit.__file__).resolve().is_relative_to(source):
    raise ValueError('validator import escaped frozen source')
from pdblend.bench.comparison_campaign import binding
from pdblend.bench.comparison_acceptance import _bound
from pdblend.bench.low_m_tuning import source_inventory
from pdblend.bench.resident_session import digest
manifest=_bound(task['manifest'])
if digest(source_inventory(source))!=manifest['context']['algorithm_source_sha256']:
    raise ValueError('algorithm runtime/source identity mismatch')
op=task['operation']
if op=='preflight':
    checked=audit.validate_manifest(task['manifest']['path'])
    answer=dict(validated=True,trials=len(checked['trials']),context=checked['context'])
elif op=='audit':
    answer=[]
    for ref in task['receipts']:
        receipt=_bound(ref)
        row=dict(trial_id=receipt['trial_id'],receipt=ref)
        if receipt['manifest']!=task['manifest']:
            raise ValueError('trial manifest binding differs')
        for artifact in receipt['artifacts'].values():
            if hashlib.sha256(Path(artifact['path']).read_bytes()).hexdigest()!=artifact['sha256']:
                raise ValueError('trial raw artifact checksum differs: '+artifact['path'])
        try:
            result=audit.validate_trial(ref['path'],manifest=manifest,manifest_ref=task['manifest'])
            row.update(result,accepted=True)
        except ValueError as exc:
            row.update(accepted=False,reason=str(exc))
        answer.append(row)
elif op=='summarize':
    for ref in task['receipts']:_bound(ref)
    answer=audit.summarize(task['manifest']['path'],[ref['path'] for ref in task['receipts']])
elif op=='enqueue':
    from pdblend.experimentation.lease import GPULeaseQueue
    job=task['job']
    queue=GPULeaseQueue(task['queue'])
    actual=queue.enqueue(job['job_id'],job['payload'],priority=job['priority'],max_attempts=job['max_attempts'])
    answer=dict(job_id=actual.job_id,status=actual.status)
else:raise ValueError('unknown frozen bridge operation')
print(json.dumps(answer,sort_keys=True,allow_nan=False))
'''


def verify_source(source_ref):
    source=read_bound(source_ref);root=Path(source_ref['path']).parent.resolve()
    if source.get('source_sha256')!=digest(source['files']):raise ValueError('frozen source inventory digest differs')
    for name,sha in source['files'].items():
        path=(root/name).resolve()
        if not path.is_relative_to(root) or binding(path)['sha256']!=sha:
            raise ValueError('frozen algorithm source bytes changed: '+name)
    return root


def frozen_call(package,operation,*,python,**kwargs):
    source=verify_source(package['source_manifest'])
    payload=dict(operation=operation,source=str(source),manifest=package['tuning_manifest'],**kwargs)
    env=dict(os.environ,PYTHONPATH=str(source),PYTHONDONTWRITEBYTECODE='1')
    result=subprocess.run([str(python),'-B','-c',BRIDGE],input=json.dumps(payload),text=True,
                          capture_output=True,cwd=source,env=env,timeout=1800)
    if result.returncode:
        raise RuntimeError('frozen '+operation+' failed: '+result.stderr[-6000:])
    return json.loads(result.stdout)


def load_package(directory):
    directory=Path(directory).resolve();campaign_ref=binding(directory/'campaign.json')
    campaign=read_bound(campaign_ref);manifest=read_bound(campaign['tuning_manifest'])
    group=read_bound(campaign['group']);jobs_ref=binding(directory/'jobs.json');jobs=read_bound(jobs_ref)
    if (campaign.get('scope')!=SCOPE or group.get('scope')!=SCOPE or len(jobs)!=1
            or group['tuning_manifest']!=campaign['tuning_manifest'] or group['source_manifest']!=campaign['source_manifest']):
        raise ValueError('inconsistent frozen low-M package')
    source=Path(campaign['source_manifest']['path']).parent
    job=jobs[0]
    if (job['payload']['source_sha256']!=source.name
            or job['payload']['trial_ids']!=campaign['first_trials']):
        raise ValueError('initial job source/trial binding differs')
    identity=dict(campaign=campaign_ref,group=campaign['group'],jobs=jobs_ref,
                  tuning_manifest=campaign['tuning_manifest'],source_manifest=campaign['source_manifest'])
    return dict(campaign,identity=identity,template=job,base_group=group,manifest=manifest)


def normalized_payload(payload):
    payload=deepcopy(payload);payload.setdefault('depends_on',[])
    return payload


def assert_same_job(actual,spec):
    if (normalized_payload(actual['payload'])!=normalized_payload(spec['payload'])
            or actual.get('max_attempts')!=spec['max_attempts'] or actual.get('priority')!=spec['priority']):
        raise ValueError('existing queue job has a different immutable stage specification')


def successful_attempt(queue,job_id):
    if queue['jobs'][job_id].get('lease_id') is not None:
        raise ValueError('terminal job still owns a lease')
    leases=[lease for lease in queue.get('leases',{}).values() if lease.get('job_id')==job_id]
    if not leases or any(lease.get('status')=='active' for lease in leases):
        raise ValueError('terminal job lacks a released attempt')
    latest=max(leases,key=lambda row:row['attempt'])
    if latest.get('status')!='succeeded':raise ValueError('latest queue attempt did not succeed')
    return Path(latest['attempt_dir']).resolve()


def harvest(package,stage,queue,*,python,output):
    attempt=successful_attempt(queue,stage['job_id']);session=attempt/'session'
    complete_ref=binding(session/'completion.json');complete=read_bound(complete_ref)
    if complete.get('scope')!=SCOPE or complete.get('status')!='passed' or complete.get('complete') is not True:
        raise ValueError('session completion is unsuccessful')
    cleanup=read_bound(complete['cleanup_receipt'])
    if cleanup!=complete['cleanup'] or cleanup.get('passed') is not True or cleanup.get('process_cleanup_verified') is not True:
        raise ValueError('actual session cleanup is unverified')
    if set(complete['results'])!=set(stage['trial_ids']):raise ValueError('session trial list differs from stage')
    refs=[]
    for trial_id in stage['trial_ids']:
        ref=binding(session/'trials'/trial_id/'trial-receipt.json');receipt=read_bound(ref)
        if receipt.get('trial_id')!=trial_id or receipt.get('manifest')!=package['tuning_manifest']:
            raise ValueError('trial receipt identity differs from stage')
        refs.append(ref)
    rows=frozen_call(package,'audit',python=python,receipts=refs)
    if len(rows)!=len(refs) or {row['trial_id'] for row in rows}!=set(stage['trial_ids']):
        raise ValueError('frozen validator did not return exactly the stage trials')
    audit_ref=write_once(Path(output)/f"stage-{stage['index']:03d}-audit.json",dict(
        completion=complete_ref,cleanup=complete['cleanup_receipt'],rows=rows,
        source_manifest=package['source_manifest'],tuning_manifest=package['tuning_manifest']))
    return rows,dict(completion=complete_ref,audit=audit_ref,attempt_dir=str(attempt))


def freeze_result(package,state,decision,*,python,output):
    refs=[state['records'][identity]['receipt'] for identity in sorted(state['records'])]
    report=frozen_call(package,'summarize',python=python,receipts=refs)
    accepted={trial_id for trial_id,row in state['records'].items() if row.get('accepted') is True}
    if {row['trial_id'] for row in report['rows']}!=accepted or len(report['rows'])!=len(accepted):
        raise ValueError('final raw validation differs from the accepted stage evidence')
    expected_rates={info['rate_rps'] for info in decision['rate_decisions'].values() if info['qualified']}
    if {row['nominal_rate_rps'] for row in report['selected']}!=expected_rates:
        raise ValueError('final qualification differs from the completed seed/family domains')
    report_ref=write_once(Path(output)/'summary.json',report)
    floor_ref=None
    if report['selected']:
        lookup={row['trial_id']:row for row in report['rows']}
        required={(seed,family) for seed in package['manifest']['seeds'] for family in package['manifest']['families']}
        for selected in report['selected']:
            rows=[lookup[trial_id] for trial_id in selected['trial_ids']]
            if len(rows)!=18 or len(required)!=18 or {(row['seed'],row['family']) for row in rows}!=required:
                raise ValueError('frozen summary tried to issue an incomplete capacity domain')
        floor_ref=write_once(Path(output)/'capacity-floor-v2.json',dict(kind='pdblend_capacity_floor_v2',
            identity=package['manifest']['identity'],tuning_manifest=package['tuning_manifest'],
            floors=report['selected'],trial_receipts=[row['receipt'] for row in report['rows']],formal_eligible=False))
    return dict(summary=report_ref,floor=floor_ref,qualified_rates=[row['nominal_rate_rps'] for row in report['selected']],
                unqualified_rates=decision['unqualified_rates'],formal_eligible=False)


def initialize_state(package):
    job=package['template']
    return dict(schema=SCHEMA,package_identity=package['identity'],status='waiting',preflight=None,
        stages=[dict(index=0,kind='initial_screen',job_id=job['job_id'],job=job,
                     trial_ids=package['first_trials'],processed=False,status='external_initial_job')],
        records={},decisions=[],diagnosis=None,created_s=time.time())


def tick(package,queue_path,state_path,*,python,enqueue=False):
    state_path=Path(state_path);output=state_path.parent
    state=json.loads(state_path.read_text()) if state_path.exists() else initialize_state(package)
    if not state_path.exists():state['queue_path']=str(Path(queue_path).resolve())
    if state.get('schema')!=SCHEMA or state.get('package_identity')!=package['identity']:
        raise ValueError('driver state belongs to different immutable package')
    if state.get('queue_path')!=str(Path(queue_path).resolve()):
        raise ValueError('driver state belongs to a different lease queue')
    if state['status'] in ('finished','needs_diagnosis'):return state
    try:
        if state['preflight'] is None:
            result=frozen_call(package,'preflight',python=python)
            state['preflight']=write_once(output/'preflight.json',dict(result,package_identity=package['identity']))
            save_state(state_path,state)
        else:read_bound(state['preflight'])
        queue=json.loads(Path(queue_path).read_text()) # read-only: no implicit reclaim/probe
        stage=state['stages'][-1]
        if not stage['processed']:
            actual=queue.get('jobs',{}).get(stage['job_id'])
            if actual is None:
                if stage['index']==0:
                    state.update(status='waiting',waiting_for='initial_job_registration')
                elif not enqueue:
                    state.update(status='prepared',waiting_for='explicit_enqueue')
                else:
                    # The stage and its deterministic ID are already persisted.
                    # If interrupted after enqueue, the next tick reconciles the
                    # queue specification instead of creating a second job.
                    save_state(state_path,state)
                    result=frozen_call(package,'enqueue',python=python,queue=str(Path(queue_path).resolve()),job=stage['job'])
                    stage['status']=result['status'];state.update(status='waiting',waiting_for=stage['job_id'])
                save_state(state_path,state);return state
            assert_same_job(actual,stage['job'])
            stage['status']=actual['status']
            if actual['status'] not in TERMINAL:
                state.update(status='waiting',waiting_for=stage['job_id']);save_state(state_path,state);return state
            if actual['status']!='succeeded':raise ValueError('stage job ended '+actual['status']+': '+stage['job_id'])
            rows,evidence=harvest(package,stage,queue,python=python,output=output)
            for row in rows:
                if row['trial_id'] in state['records']:raise ValueError('duplicate executed trial: '+row['trial_id'])
                state['records'][row['trial_id']]=row
            stage.update(processed=True,evidence=evidence)
            save_state(state_path,state)
        decision=next_decision(package['manifest'],state['records'])
        decision_ref=write_once(output/f"decision-{len(state['decisions']):03d}.json",dict(decision,
            source_manifest=package['source_manifest'],tuning_manifest=package['tuning_manifest'],
            predecessor=stage['job_id'],input_receipts=[state['records'][identity]['receipt']
                for identity in sorted(state['records'])]))
        state['decisions'].append(decision_ref)
        if decision['kind']=='finalize':
            state['result']=freeze_result(package,state,decision,python=python,output=output)
            state['status']='needs_diagnosis' if decision['unqualified_rates'] else 'finished'
            if decision['unqualified_rates']:state['diagnosis']='no fully qualified candidate for one or more rates'
        else:
            next_stage=derived_stage(package['template'],package['base_group'],directory=output,
                index=stage['index']+1,kind=decision['kind'],trial_ids=decision['trial_ids'],predecessor=stage['job_id'])
            state['stages'].append(next_stage);state.update(status='prepared',waiting_for='explicit_enqueue')
            save_state(state_path,state)
            if enqueue:
                result=frozen_call(package,'enqueue',python=python,queue=str(Path(queue_path).resolve()),job=next_stage['job'])
                next_stage['status']=result['status'];state.update(status='waiting',waiting_for=next_stage['job_id'])
    except Exception as exc:
        state.update(status='needs_diagnosis',diagnosis=repr(exc))
    state['updated_s']=time.time();save_state(state_path,state)
    return state


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package',type=Path,required=True)
    parser.add_argument('--queue',type=Path,required=True)
    parser.add_argument('--state-dir',type=Path)
    parser.add_argument('--python',type=Path,default=Path(sys.executable))
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--once',action='store_true',help='check/prepare at most one next stage (default)')
    mode.add_argument('--watch',action='store_true',help='read-only CPU polling every five seconds')
    parser.add_argument('--enqueue',action='store_true',help='explicitly enable immutable follow-up job enqueueing')
    args=parser.parse_args()
    output=(args.state_dir or args.package/'driver').resolve();output.mkdir(parents=True,exist_ok=True)
    stop=output/'user-stop.json'
    scope_stop=args.package.parent/'matrix-handoff-request.json'
    def stopped():
        return stop.is_file() or (scope_stop.is_file() and
            json.loads(scope_stop.read_text()).get('new_low_m_qualification_authorized') is False)
    if stopped():
        print(json.dumps(dict(status='user_stopped',reason=str(stop if stop.is_file() else scope_stop))),flush=True)
        return 0
    package=load_package(args.package)
    prior=None
    while True:
        if stopped():
            print(json.dumps(dict(status='user_stopped',reason=str(stop if stop.is_file() else scope_stop))),flush=True)
            return 0
        with (output/'manager.lock').open('a+') as lock:
            fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
            if stopped():
                return 0
            state=tick(package,args.queue,output/'state.json',python=args.python,enqueue=args.enqueue)
        view=dict(status=state['status'],stage=state['stages'][-1]['index'],job_id=state['stages'][-1]['job_id'],
            accepted_trials=sum(row.get('accepted') is True for row in state['records'].values()),
            rejected_trials=sum(row.get('accepted') is not True for row in state['records'].values()),
            diagnosis=state.get('diagnosis'),result=state.get('result'))
        key=digest(view)
        if key!=prior:print(json.dumps(view,sort_keys=True),flush=True);prior=key
        if not args.watch or state['status'] in ('finished','needs_diagnosis') or (state['status']=='prepared' and not args.enqueue):
            break
        time.sleep(5.)
    return 1 if state['status']=='needs_diagnosis' else 0


if __name__=='__main__':raise SystemExit(main())
