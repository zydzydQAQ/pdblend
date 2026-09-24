#!/usr/bin/env python3
"""Wait for the complete repaired cohort, then drive predeclared narrow repeats.

Default is dry-run preparation plus CPU image preflight. Only --enqueue submits
leases, exactly one at a time. Failed attempts are retained; unsafe cleanup stops
the serializer. This process never executes GPU commands or changes old jobs.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parents[1]
PREPARER='2026-09-24_prepare_paired_energy_repeats.py'
SCHEMA='pdblend-paired-repeat-driver/v1'
TERMINAL={'succeeded','failed','cancelled','blocked'}
CONDITIONS={(m,d,s) for m in ('7B','14B','32B') for d in ('alpaca','sharegpt','longbench') for s in (.25,.5,.75,1.)}
SYSTEMS=('mixed','distserve','dynamollm','ecoserve')


def binding(path):
    path=Path(path).resolve();return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def bound(ref):
    if binding(ref['path'])!=ref:raise ValueError('immutable artifact changed: '+ref['path'])
    return json.loads(Path(ref['path']).read_text())


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix='.'+path.name,dir=path.parent)
    with os.fdopen(fd,'w') as stream:
        json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    os.replace(tmp,path)


def immutable(path,value):
    path=Path(path)
    if path.exists():
        if json.loads(path.read_text())!=value:raise ValueError('immutable output already differs: '+str(path))
    else:save(path,value)
    return binding(path)


def verify_program(ref):
    manifest=bound(ref);root=Path(ref['path']).parent
    for name,sha in manifest['files'].items():
        path=(root/name).resolve()
        if not path.is_relative_to(root) or binding(path)['sha256']!=sha:raise ValueError('frozen CPU program changed')
    return root


def freeze_program(out):
    root=Path(out)/'cpu-program'
    if (root/'program.json').exists():
        ref=binding(root/'program.json');verify_program(ref);return ref
    if root.exists():raise ValueError('partial CPU program freeze needs diagnosis; no files overwritten')
    shutil.copytree(ROOT/'src',root/'src',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    (root/'scripts').mkdir()
    for name in (PREPARER,'report_cohort_energy.py','cohort_energy_failures.py'):
        shutil.copyfile(ROOT/'scripts'/name,root/'scripts'/name)
    files={str(path.relative_to(root)):binding(path)['sha256'] for path in sorted(root.rglob('*')) if path.is_file()}
    return immutable(root/'program.json',dict(files=files,hardware_executed=False))


def run_program(program,args,*,python):
    root=verify_program(program)
    return subprocess.run([str(python),'-B',str(root/'scripts'/PREPARER),*map(str,args)],text=True,
        capture_output=True,timeout=3600,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(root/'src')))


def enqueue(program,queue_path,spec,*,python):
    root=verify_program(program)
    code="import json,sys;from pdblend.experimentation.lease import GPULeaseQueue;t=json.load(sys.stdin);j=t['job'];r=GPULeaseQueue(t['queue']).enqueue(j['job_id'],j['payload'],priority=j['priority'],max_attempts=j['max_attempts']);print(r.job_id)"
    result=subprocess.run([str(python),'-B','-c',code],input=json.dumps(dict(queue=str(queue_path),job=spec)),
        text=True,capture_output=True,timeout=60,cwd=root,env=dict(os.environ,PYTHONPATH=str(root/'src'),PYTHONDONTWRITEBYTECODE='1'))
    if result.returncode:raise RuntimeError('enqueue failed: '+result.stderr[-3000:])


def same_job(actual,spec):
    left=deepcopy(actual['payload']);right=deepcopy(spec['payload']);left.setdefault('depends_on',[]);right.setdefault('depends_on',[])
    if left!=right or actual.get('max_attempts')!=spec['max_attempts'] or actual.get('priority')!=spec['priority']:
        raise ValueError('existing queue payload/priority/attempt policy differs: '+spec['job_id'])


def released(queue,job_id):
    job=queue.get('jobs',{}).get(job_id)
    return bool(job and job['status'] in TERMINAL and job.get('lease_id') is None and not any(
        lease.get('job_id')==job_id and lease.get('status')=='active' for lease in queue.get('leases',{}).values()))


def energy_complete(row):
    values=[row.get(key) for key in ('total_energy_kj','service_energy_kj','tail_energy_kj')]
    return row.get('energy_measurement_complete') is True and all(type(v) in (int,float) and 0<=v<float('inf') for v in values) and abs(values[0]-values[1]-values[2])<1e-5


def cohort_ready(followup,queue,report_dir):
    if not followup.get('matrix') or not followup.get('queue_specs'):
        return None,dict(reason='final_standard_matrix_not_declared',followup_status=followup.get('status'))
    specs=bound(followup['queue_specs'])['jobs']
    if len(specs)!=7:raise ValueError('expected exactly three matrix and four supplement jobs')
    for spec in specs:
        actual=queue.get('jobs',{}).get(spec['job_id'])
        if actual:same_job(actual,spec)
    pending=[spec['job_id'] for spec in specs if not released(queue,spec['job_id'])]
    if pending:return None,dict(reason='parent_pipeline_not_terminal_and_released',pending_jobs=pending)
    matrix=bound(followup['matrix']['campaign']);source=bound(matrix['execution_source_manifest'])
    revision=source['source_sha256'];expected={p['name']:p for p in matrix['points']}
    if len(expected)!=36 or {p['revision'] for p in expected.values()}!={revision}:
        raise ValueError('followup matrix does not bind a unique final 36-point candidate revision')
    if matrix['execution_source_manifest']!=followup['input_identity']['low_m']['source_manifest']:
        raise ValueError('matrix revision differs from completed qualified low-M source')
    latest_path=Path(report_dir)/'latest.json';state_path=Path(report_dir)/'state.json'
    if not latest_path.exists() or not state_path.exists():return None,dict(reason='report_watcher_has_no_published_snapshot')
    latest=json.loads(latest_path.read_text());watch=json.loads(state_path.read_text())
    if not any(row==latest for row in watch.get('reports',[])):return None,dict(reason='report_publication_in_progress')
    snapshot=bound(latest['snapshot']);root=Path(latest['path']).resolve()
    if Path(latest['snapshot']['path']).resolve()!=root/'snapshot.json':raise ValueError('report snapshot escaped its published directory')
    points_ref=binding(root/'points.json');points=bound(points_ref)
    candidates=[row for row in points if row.get('system')=='pdblend' and row.get('point_id') in expected and row.get('revision')==revision]
    baselines=[row for row in points if row.get('system') in SYSTEMS]
    keys=lambda rows:{(r['model'],r['dataset'],float(r['rate_scale'])) for r in rows}
    candidate_ids=[row['point_id'] for row in candidates]
    baseline_slots={(row['system'],row['model'],row['dataset'],float(row['rate_scale'])) for row in baselines}
    if (len(candidates)!=36 or len(set(candidate_ids))!=36 or keys(candidates)!=CONDITIONS
            or len(baselines)!=144 or baseline_slots!={(s,*key) for s in SYSTEMS for key in CONDITIONS}
            or not all(energy_complete(row) for row in baselines)):
        return None,dict(reason='latest_report_does_not_cover_final_36_and_complete_144',candidate_points=len(candidates),
            baseline_points=len(baselines),baseline_complete=sum(energy_complete(row) for row in baselines),revision=revision)
    for row in candidates:
        point=expected[row['point_id']]
        if (row['model']!=point['model_id'].split('-')[1] or row['dataset']!=point['dataset']
                or row['rate_scale']!=point['scale'] or row['offered_rps']!=point['rate_rps']):raise ValueError('candidate report identity differs from final matrix')
    dominance_ref=binding(root/'dominance.json');dominance=bound(dominance_ref)
    return dict(revision=revision,rows=candidates+baselines,latest=latest,snapshot=latest['snapshot'],
        original_points=points_ref,dominance=dominance_ref,measurement_compatibility=dominance['measurement_compatibility'],
        matrix=followup['matrix']['campaign'],parent_job_ids=[spec['job_id'] for spec in specs]),None


def export_templates(cohort,queue):
    # Only actual jobs that produced the selected eligible observations can be
    # templates. No hand-written common engine is substituted for a baseline.
    comparisons=bound(cohort['dominance'])['comparisons']
    names={row['point_id'] for row in comparisons if row.get('revision')==cohort['revision']
        and row.get('observed_goal_met') is True and row.get('saving_vs_min_baseline_pct') is not None
        and 0<row['saving_vs_min_baseline_pct']<3}
    needed_keys={(r['model'],r['dataset'],r['rate_scale']) for r in cohort['rows'] if r['point_id'] in names}
    rows=[r for r in cohort['rows'] if (r['model'],r['dataset'],r['rate_scale']) in needed_keys]
    templates={};proofs=[]
    for row in rows:
        receipt=Path(row['receipt_path']).resolve()
        if binding(receipt)['sha256']!=row['receipt_sha256']:raise ValueError('published selected receipt changed')
        matches=[lease for lease in queue.get('leases',{}).values() if receipt.is_relative_to(Path(lease['attempt_dir']).resolve())]
        if len(matches)!=1:raise ValueError('selected observation lacks one actual lease/template: '+str(receipt))
        lease=matches[0];job=queue['jobs'][lease['job_id']]
        if lease.get('status')=='active':raise ValueError('initial observation still owns an active lease')
        claim_ref=binding(Path(lease['attempt_dir'])/'manifest.json');claim=bound(claim_ref)
        if claim['job_id']!=job['job_id'] or claim['payload']!=job['payload']:raise ValueError('actual initial job claim differs from queue template')
        templates[job['job_id']]={key:job[key] for key in ('job_id','payload','priority','max_attempts')}
        proofs.append(dict(receipt=dict(path=str(receipt),sha256=row['receipt_sha256']),claimed_execution=claim_ref))
    return [templates[key] for key in sorted(templates)],proofs


CHECK=r'''
import hashlib,importlib,json,os,sys
from pathlib import Path
from types import SimpleNamespace
from pdblend.bench.comparison_campaign import load_bound,binding
from pdblend.bench.resident_session import digest,engine_signature
from pdblend.bench.comparison_runtime import pdblend_window_resources
group=load_bound(json.loads(sys.argv[1]));point=group['points'][0]
assert len(group['points'])==1 and engine_signature(group['engine_identity'])==group['engine_signature']
source=Path(os.environ['PDBLEND_SOURCE_MANIFEST']);manifest=load_bound(binding(source))
assert digest(manifest['files'])==os.environ['PDBLEND_SOURCE_SHA256']==source.parent.name
for name,sha in manifest['files'].items():assert hashlib.sha256((source.parent/name).read_bytes()).hexdigest()==sha
assert Path(importlib.import_module('pdblend.bench.comparison_runtime').__file__).resolve().is_relative_to(Path('/opt/pdblend-src'))
trace=load_bound(point['trace'])
for key in ('model_id','dataset','slo','seed','rate_rps','duration_s'):assert trace[key]==point[key]
assert point['seed']==701 and point['duration_s']==150
assert all(0<=r['arrival_s']<150 and len(r['prompt'])+r['max_tokens']<=8192 for r in trace['requests'])
for key,value in group['engine_identity']['environment'].items():assert os.environ[key]==value
if point['system']=='pdblend':
 from pdblend.bench.comparison_pdblend_observation import validate_observation_inputs
 validate_observation_inputs(point,point['inputs'])
 specs=[SimpleNamespace(tp=r['tp'],pp=r['pp'],generation=0) for r in group['engine_identity']['instances']]
 pdblend_window_resources(point,specs)
elif point['system']=='dynamollm':
 from pdblend.bench.comparison_runtime import make_resident_adapter
 adapter=make_resident_adapter(group,Path('/output'),base_port=58000);adapter.identity=group['engine_identity']
 value,checked=adapter._prepare(point);adapter._check_inventory(value)
elif point.get('qualification_mode')=='ecoserve_native_bootstrap':
 module='comparison_ecoserve32_inputs' if point['model_id']=='Qwen2.5-32B-Instruct' else 'comparison_ecoserve_inputs'
 checked=importlib.import_module('pdblend.bench.'+module).validate_ecoserve_inputs(point,group['engine_identity'],source_manifest=binding(source))
 assert checked['preflight_ready'],checked
elif point.get('qualification_mode')=='distserve_native_bootstrap':
 from pdblend.bench.comparison_distserve_inputs import validate_distserve_inputs
 checked=validate_distserve_inputs(point,group['engine_identity'],source_manifest=binding(source),replay_search=True)
 assert checked['preflight_ready'],checked
elif point.get('observation_scope')=='baseline_profile_unqualified_evaluation/v1':
 from pdblend.bench.comparison_baseline_observation import validate_observation_inputs
 validate_observation_inputs(point,group['engine_identity'])
elif point['system']=='mixed' and point.get('qualification_mode')=='mixed_native_bootstrap':
 assert point['engine_identity']==group['engine_identity']
else:raise ValueError('no reviewed CPU-only preflight for this original baseline entrypoint')
print(json.dumps(dict(passed=True,point=point['name'],system=point['system'],hardware_executed=False,source_sha256=manifest['source_sha256'])))
'''


def cpu_command(spec,target):
    payload=spec['payload'];argv=payload['argv'];image=payload['image_digest'];boundary=argv.index(image)
    if argv[boundary+1:boundary+4]!=['-B','-m','pdblend.bench.comparison_runtime']:
        raise ValueError('CPU preflight has no reviewed adapter for the actual job entrypoint')
    prefix=['docker','run','--rm','--runtime','runc','--network','none','--cpus','1','--memory','4g',
        '--entrypoint','/opt/venv/bin/python']
    index=0
    while index<boundary:
        if argv[index] in ('-v','-e'):
            value=argv[index+1]
            if argv[index]=='-e' or value.endswith(':ro'):prefix.extend([argv[index],value])
            index+=2
        else:index+=1
    prefix+=['-v',str(Path(target).resolve())+':/output:rw','-e','NVIDIA_VISIBLE_DEVICES=void']
    group=binding(argv[argv.index('--group')+1])
    return prefix+[image,'-B','-c',CHECK,json.dumps(group)]


def preflight_jobs(specs,out):
    results=[]
    for spec in specs:
        target=Path(out)/'image-preflight'/spec['job_id'];target.mkdir(parents=True,exist_ok=True)
        path=target/'receipt.json';command=cpu_command(spec,target)
        if path.exists():
            receipt=json.loads(path.read_text())
            if receipt['command']!=command:raise ValueError('cached repeat preflight input differs')
        else:
            proc=subprocess.run(command,text=True,capture_output=True,timeout=1800)
            result=json.loads(proc.stdout.splitlines()[-1]) if proc.returncode==0 else None
            receipt=dict(command=command,returncode=proc.returncode,stdout=proc.stdout,stderr=proc.stderr,result=result,hardware_executed=False)
            immutable(path,receipt)
        if receipt['returncode']!=0 or (receipt.get('result') or {}).get('passed') is not True:
            raise ValueError('repeat image CPU preflight failed: '+spec['job_id'])
        results.append(binding(path))
    return immutable(Path(out)/'image-preflight.json',dict(passed=True,receipts=results,hardware_executed=False))


def safe_terminal(queue,spec):
    if not released(queue,spec['job_id']):return False
    leases=[r for r in queue.get('leases',{}).values() if r.get('job_id')==spec['job_id']]
    if len(leases)!=1 or leases[0].get('attempt')!=1:raise ValueError('predeclared first attempt missing or replaced')
    attempt=Path(leases[0]['attempt_dir']);complete_ref=binding(attempt/'session/completion.json');complete=bound(complete_ref)
    cleanup=complete.get('cleanup',{})
    if cleanup.get('passed') is not True or cleanup.get('process_cleanup_verified') is not True:
        raise ValueError('failed/blocked attempt lacks verified physical cleanup; successor prohibited')
    group=bound(binding(spec['payload']['argv'][spec['payload']['argv'].index('--group')+1]));name=group['points'][0]['name']
    receipt=bound(binding(attempt/'session/windows'/name/'receipt.json'))
    metrics=receipt.get('result',{}).get('metrics',{})
    if (receipt.get('cleanup_passed') is not True or metrics.get('unresolved_requests')!=0
            or type(metrics.get('offered_requests')) is not int or metrics['offered_requests']<=0
            or type(metrics.get('successful_requests')) is not int or type(metrics.get('failed_requests')) is not int
            or metrics['successful_requests']+metrics['failed_requests']!=metrics['offered_requests']):
        raise ValueError('failed/blocked attempt lacks all-request terminal and native cleanup evidence; successor prohibited')
    drain=bound(dict(path=str(attempt/'session/windows'/name/'run/native-drain.json'),
                     sha256=receipt['artifacts']['run/native-drain.json']))
    if drain.get('passed') is not True:raise ValueError('native drain failed; successor prohibited')
    return dict(completion=complete_ref,lease_id=leases[0].get('lease_id'),status=queue['jobs'][spec['job_id']]['status'])


def tick(args):
    out=Path(args.out).resolve();path=out/'state.json'
    identity=dict(script=binding(__file__),queue=str(Path(args.queue).resolve()),followup=str(Path(args.followup).resolve()),
        report_watch=str(Path(args.report_watch).resolve()),python=str(Path(args.python).resolve()))
    state=json.loads(path.read_text()) if path.exists() else dict(schema=SCHEMA,identity=identity,status='waiting',submitted=[],completed=[],goal_complete=False)
    if state.get('schema')!=SCHEMA or state['identity']!=identity:raise ValueError('repeat driver identity changed')
    if state['status'] in ('needs_diagnosis','evaluated_awaiting_review'):return state
    try:
        if not state.get('program'):
            state['program']=freeze_program(out);save(path,state)
        queue=json.loads(Path(args.queue).read_text())
        if not state.get('cohort'):
            if not Path(args.followup).exists():save(path,state);return state
            followup=json.loads(Path(args.followup).read_text())
            if followup.get('execution_paths',{}).get('queue')!=identity['queue']:raise ValueError('followup uses another lease queue')
            cohort,waiting=cohort_ready(followup,queue,args.report_watch)
            if cohort is None:
                state.update(status='waiting',waiting=waiting);save(path,state);return state
            templates,proofs=export_templates(cohort,queue)
            state['cohort']=immutable(out/'selected-cohort.json',cohort)
            state['initial_points']=immutable(out/'initial-points.json',cohort['rows'])
            state['templates']=immutable(out/'actual-job-templates.json',templates)
            state['template_proofs']=immutable(out/'template-proofs.json',proofs)
            state['followup_snapshot']=immutable(out/'followup-snapshot.json',followup)
            save(path,state)
        cohort=bound(state['cohort']);target=out/'repeats'
        if not state.get('preparation'):
            if target.exists() and not (target/'preparation.json').exists():raise ValueError('partial repeat declaration retained; needs diagnosis before any enqueue')
            if not target.exists():
                argv=['prepare','--points',state['initial_points']['path'],'--jobs',state['templates']['path'],
                    '--out',str(target),'--candidate-revision',cohort['revision'],'--driver-serialized']
                for ref in cohort['measurement_compatibility']:
                    bound(ref);argv.extend(['--measurement-compatibility',ref['path']])
                process=run_program(state['program'],argv,python=args.python)
                immutable(out/'prepare-invocation.json',dict(argv=argv,returncode=process.returncode,stdout=process.stdout,stderr=process.stderr))
                if process.returncode:raise RuntimeError('repeat preparation failed: '+process.stderr[-4000:])
            state['preparation']=binding(target/'preparation.json');save(path,state)
        preparation=bound(state['preparation']);specs=bound(preparation['jobs'])
        protocol=bound(preparation['protocol']);design=bound(protocol['design'])
        if design.get('queue_order')!='one_job_after_prior_released_lease_including_blocked':raise ValueError('repeats lack a predeclared serializer order')
        if any(spec['payload'].get('predeclared_driver_sequence')!=i or spec['payload'].get('after_terminal') for i,spec in enumerate(specs)):
            raise ValueError('serialized repeat jobs were changed')
        if len(protocol['planned_observations'])!=len(specs):raise ValueError('declared jobs and observation count differ')
        for plan,spec in zip(protocol['planned_observations'],specs):
            group=bound(plan['group']);point=bound(plan['point'])
            if (plan['job_id']!=spec['job_id'] or group['points']!=[point]
                    or spec['payload']['argv'][spec['payload']['argv'].index('--group')+1]!=plan['group']['path']):
                raise ValueError('predeclared point/group/job bindings differ')
        if not state.get('preflight'):
            state['preflight']=preflight_jobs(specs,out);save(path,state)
        else:bound(state['preflight'])
        for spec in specs:
            actual=queue.get('jobs',{}).get(spec['job_id'])
            if actual:same_job(actual,spec)
        for index,spec in enumerate(specs):
            actual=queue.get('jobs',{}).get(spec['job_id'])
            if actual is None:
                if any(other['job_id'] in queue.get('jobs',{}) for other in specs[index+1:]):raise ValueError('later repeat was submitted before its declared predecessor')
                if not args.enqueue:
                    state.update(status='prepared',next_job=spec['job_id']);save(path,state);return state
                save(path,state) # Immutable program/protocol/jobs/preflight are durable first.
                enqueue(state['program'],args.queue,spec,python=args.python)
                if spec['job_id'] not in state['submitted']:state['submitted'].append(spec['job_id'])
                state.update(status='running',next_job=spec['job_id']);save(path,state);return state
            evidence=safe_terminal(queue,spec)
            if not evidence:
                state.update(status='running',next_job=spec['job_id']);save(path,state);return state
            if spec['job_id'] not in state['completed']:
                immutable(out/'release-receipts'/(spec['job_id']+'.json'),evidence)
                state['completed'].append(spec['job_id']);save(path,state)
        result_dir=out/'acceptance'
        if not (result_dir/'acceptance.json').exists():
            process=run_program(state['program'],['evaluate','--package',str(target),'--queue',str(args.queue),'--out',str(result_dir)],python=args.python)
            immutable(out/'evaluate-invocation.json',dict(returncode=process.returncode,stdout=process.stdout,stderr=process.stderr))
            if process.returncode:raise RuntimeError('repeat evaluation failed: '+process.stderr[-4000:])
        state['acceptance']=binding(result_dir/'acceptance.json');state['status']='evaluated_awaiting_review'
    except Exception as exc:state.update(status='needs_diagnosis',diagnosis=repr(exc))
    state['updated_s']=time.time();save(path,state);return state


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--queue',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--followup',type=Path,default=ROOT/'results/2026-09-24/pdblend-energy-repair-v1/followup/followup-status.json')
    p.add_argument('--report-watch',type=Path,default=ROOT/'results/2026-09-24/pdblend-energy-repair-v1/report-watch')
    p.add_argument('--python',type=Path,default=Path(sys.executable));mode=p.add_mutually_exclusive_group()
    mode.add_argument('--once',action='store_true');mode.add_argument('--watch',action='store_true');p.add_argument('--enqueue',action='store_true')
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True);prior=None
    while True:
        with (args.out/'driver.lock').open('a+') as lock:
            fcntl.flock(lock.fileno(),fcntl.LOCK_EX);state=tick(args)
        view={key:state.get(key) for key in ('status','waiting','next_job','completed','diagnosis','acceptance')}
        if view!=prior:print(json.dumps(view,sort_keys=True),flush=True);prior=view
        if not args.watch or state['status'] in ('needs_diagnosis','evaluated_awaiting_review') or state['status']=='prepared' and not args.enqueue:break
        time.sleep(5.)
    return 1 if state['status']=='needs_diagnosis' else 0


if __name__=='__main__':raise SystemExit(main())
