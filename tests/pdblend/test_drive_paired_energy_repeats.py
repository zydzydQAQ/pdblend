from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def driver():
    path=Path(__file__).resolve().parents[2]/'scripts/2026-09-24_drive_paired_energy_repeats.py'
    spec=importlib.util.spec_from_file_location('narrow_repeat_driver_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))


def spec(identity,index=0,group='group.json'):
    return dict(job_id=identity,priority=600,max_attempts=1,payload=dict(depends_on=[],after_terminal=[],
        predeclared_driver_sequence=index,argv=['docker','run','--name',identity,'-v','/code:/opt/pdblend-src:ro',
          '-v','/clock:/clock:rw','--gpus','all','--cap-add','SYS_ADMIN','-e','VLLM_USE_V1=1','image','-B','-m',
          'pdblend.bench.comparison_runtime','--group',group,'--out','{attempt_dir}/session'],image_digest='image'))


@pytest.fixture
def cohort(driver,tmp_path):
    source=driver.immutable(tmp_path/'source.json',{'source_sha256':'final-revision'})
    points=[dict(name=f'{m}-{d}-{s}-final',revision='final-revision',model_id='Qwen2.5-'+m+'-Instruct',dataset=d,
        scale=s,rate_rps=2.) for m,d,s in sorted(driver.CONDITIONS)]
    matrix=driver.immutable(tmp_path/'matrix.json',dict(points=points,execution_source_manifest=source))
    jobs=[spec('parent-'+str(i)) for i in range(7)];queue=dict(jobs={j['job_id']:dict(j,status='succeeded',lease_id=None) for j in jobs},leases={})
    followup=dict(matrix={'campaign':matrix},queue_specs=driver.immutable(tmp_path/'specs.json',{'jobs':jobs}),
                  input_identity={'low_m':{'source_manifest':source}})
    rows=[]
    for p in points:
        key=dict(model=p['model_id'].split('-')[1],dataset=p['dataset'],rate_scale=p['scale'],offered_rps=2.)
        rows.append(dict(key,system='pdblend',point_id=p['name'],revision='final-revision'))
        for system in driver.SYSTEMS:
            rows.append(dict(key,system=system,point_id=p['name']+system,revision=system+'-revision',
                energy_measurement_complete=True,total_energy_kj=100,service_energy_kj=90,tail_energy_kj=10))
    report=tmp_path/'report';snapshot=driver.immutable(report/'snapshot.json',{'baseline_conditions':36})
    driver.immutable(report/'points.json',rows);driver.immutable(report/'dominance.json',{'measurement_compatibility':[],'comparisons':[]})
    latest=dict(path=str(report),snapshot=snapshot,input_sha256='published',finished_s=100.)
    watch=tmp_path/'watch';write(watch/'latest.json',latest);write(watch/'state.json',{'reports':[latest]})
    return followup,queue,watch,report,rows


def test_complete_final_candidate_revision_and_144_baselines_are_required(driver,cohort):
    followup,queue,watch,_,_=cohort
    ready,reason=driver.cohort_ready(followup,queue,watch)
    assert reason is None and ready['revision']=='final-revision' and len(ready['rows'])==180


@pytest.mark.parametrize('problem',['five_candidates','wrong_revision','23_energy_missing','missing_baseline','active_lease'])
def test_partial_reports_do_not_trigger_repeats_despite_36_baseline_conditions(driver,cohort,problem):
    followup,queue,watch,report,rows=cohort
    if problem=='five_candidates':
        candidates=[r for r in rows if r['system']=='pdblend'][:5];rows=[r for r in rows if r['system']!='pdblend']+candidates
    elif problem=='wrong_revision':
        for row in rows:
            if row['system']=='pdblend':row['revision']='old-candidate'
    elif problem=='23_energy_missing':
        for row in [r for r in rows if r['system']!='pdblend'][:23]:row['total_energy_kj']=None
    elif problem=='missing_baseline':rows.pop()
    else:queue['leases']['active']={'job_id':'parent-0','status':'active'}
    write(report/'points.json',rows)
    ready,reason=driver.cohort_ready(followup,queue,watch)
    assert ready is None and reason['reason'] in ('latest_report_does_not_cover_final_36_and_complete_144','parent_pipeline_not_terminal_and_released')


def test_preflight_uses_actual_frozen_imports_without_gpu_or_writable_code(driver,tmp_path):
    group=tmp_path/'group.json';write(group,{'points':[{'name':'test'}]})
    command=driver.cpu_command(spec('job',group=str(group)),tmp_path/'cpu-output')
    prefix=command[:command.index('image')]
    assert '--gpus' not in prefix and '--cap-add' not in prefix and '/clock:/clock:rw' not in prefix
    assert prefix[prefix.index('--runtime')+1]=='runc' and prefix[prefix.index('--network')+1]=='none'
    assert 'VLLM_USE_V1=1' in prefix and 'NVIDIA_VISIBLE_DEVICES=void' in prefix
    assert '/code:/opt/pdblend-src:ro' in prefix


def terminal_fixture(driver,tmp_path,*,status='blocked',cleanup=True,unresolved=0):
    group=tmp_path/'group.json';write(group,{'points':[{'name':'point'}]});job=spec('job',group=str(group))
    attempt=tmp_path/'attempt';window=attempt/'session/windows/point'
    write(attempt/'session/completion.json',dict(status='failed' if status=='blocked' else 'passed',
        complete=status=='succeeded',cleanup={'passed':cleanup,'process_cleanup_verified':cleanup}))
    drain=driver.immutable(window/'run/native-drain.json',{'passed':cleanup})
    write(window/'receipt.json',dict(cleanup_passed=cleanup,result={'metrics':dict(offered_requests=100,
        successful_requests=99,failed_requests=1,unresolved_requests=unresolved)},
        artifacts={'run/native-drain.json':drain['sha256']}))
    queue=dict(jobs={'job':dict(job,status=status,lease_id=None)},leases={'one':dict(job_id='job',attempt=1,status=status,attempt_dir=str(attempt))})
    return queue,job


def test_blocked_job_is_not_unblocked_but_verified_cleanup_can_release_serializer(driver,tmp_path):
    queue,job=terminal_fixture(driver,tmp_path);before=deepcopy(queue)
    assert driver.safe_terminal(queue,job)['status']=='blocked'
    assert queue==before


@pytest.mark.parametrize('cleanup,unresolved',[(False,0),(True,1)])
def test_unsafe_blocked_attempt_cannot_advance_to_next_repeat(driver,tmp_path,cleanup,unresolved):
    queue,job=terminal_fixture(driver,tmp_path,cleanup=cleanup,unresolved=unresolved)
    with pytest.raises(ValueError,match='successor prohibited'):driver.safe_terminal(queue,job)


@pytest.fixture
def prepared_state(driver,tmp_path,monkeypatch):
    out=tmp_path/'driver';queue=tmp_path/'queue.json';write(queue,{'jobs':{},'leases':{}})
    args=SimpleNamespace(out=out,queue=queue,followup=tmp_path/'followup.json',report_watch=tmp_path/'watch',python=Path('/python'),enqueue=False)
    root=out/'repeats';jobs=[];plans=[]
    for i in range(2):
        point={'name':'point-'+str(i)};point_ref=driver.immutable(root/f'point-{i}.json',point)
        group_ref=driver.immutable(root/f'group-{i}.json',{'points':[point]})
        job=spec('job-'+str(i),i,group_ref['path']);jobs.append(job)
        plans.append(dict(job_id=job['job_id'],group=group_ref,point=point_ref))
    design=driver.immutable(root/'design.json',{'queue_order':'one_job_after_prior_released_lease_including_blocked'})
    protocol=driver.immutable(root/'protocol.json',{'planned_observations':plans,'design':design})
    preparation=driver.immutable(root/'preparation.json',{'jobs':driver.immutable(root/'jobs.json',jobs),'protocol':protocol})
    identity=dict(script=driver.binding(driver.__file__),queue=str(queue.resolve()),followup=str(args.followup.resolve()),
        report_watch=str(args.report_watch.resolve()),python=str(args.python.resolve()))
    state=dict(schema=driver.SCHEMA,identity=identity,status='prepared',submitted=[],completed=[],goal_complete=False,
        program={'path':'cpu-program','sha256':'p'},cohort=driver.immutable(out/'selected-cohort.json',{'revision':'final'}),
        preparation=preparation,preflight=driver.immutable(out/'preflight.json',{'passed':True}))
    driver.save(out/'state.json',state)
    return args,jobs


def test_default_dry_run_never_submits(driver,prepared_state,monkeypatch):
    args,jobs=prepared_state
    monkeypatch.setattr(driver,'enqueue',lambda *a,**kw:pytest.fail('dry run enqueued'))
    state=driver.tick(args)
    assert state['status']=='prepared' and state['next_job']==jobs[0]['job_id']
    assert json.loads(args.queue.read_text())['jobs']=={}


def test_one_job_at_a_time_and_crash_after_enqueue_is_idempotent(driver,prepared_state,monkeypatch):
    args,jobs=prepared_state;args.enqueue=True;calls=[];crashed=[]
    def enqueue(_program,queue_path,spec,**kwargs):
        calls.append(spec['job_id']);queue=json.loads(queue_path.read_text())
        queue['jobs'][spec['job_id']]=dict(spec,status='queued',lease_id=None);write(queue_path,queue)
        if not crashed:crashed.append(True);raise KeyboardInterrupt('after durable enqueue')
    monkeypatch.setattr(driver,'enqueue',enqueue)
    monkeypatch.setattr(driver,'safe_terminal',lambda queue,spec:False if queue['jobs'][spec['job_id']]['status']=='queued' else {'released':True})
    with pytest.raises(KeyboardInterrupt):driver.tick(args)
    state=driver.tick(args)
    assert state['status']=='running' and calls==['job-0']
    queue=json.loads(args.queue.read_text());queue['jobs']['job-0']['status']='succeeded';write(args.queue,queue)
    driver.tick(args)
    assert calls==['job-0','job-1']


def test_cleanup_failure_stops_successor_even_when_lease_status_is_blocked(driver,prepared_state,monkeypatch):
    args,jobs=prepared_state;args.enqueue=True
    write(args.queue,{'jobs':{'job-0':dict(jobs[0],status='blocked',lease_id=None)},'leases':{}})
    def unsafe(*args):raise ValueError('request terminal or cleanup missing')
    monkeypatch.setattr(driver,'safe_terminal',unsafe)
    monkeypatch.setattr(driver,'enqueue',lambda *a,**kw:pytest.fail('unsafe blocked job advanced'))
    assert driver.tick(args)['status']=='needs_diagnosis'
