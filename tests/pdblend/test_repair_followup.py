"""Follow-up planning is CPU-only; fake queue files never reach real leases."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def followup():
    path=Path(__file__).resolve().parents[2]/'scripts/2026-09-24_follow_repair_campaign.py'
    spec=importlib.util.spec_from_file_location('repair_followup_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))


def job(identity,after=()):
    return dict(job_id=identity,priority=500,max_attempts=1,payload=dict(after_terminal=list(after),depends_on=[],
        gpu_count=8,exclusive=True,reserve_host=True,required_receipts=['session/completion.json'],
        source_sha256='source',image_digest='sha256:image',argv=['frozen-command']))


@pytest.fixture
def inputs():
    baselines=[job('base-'+str(i),['base-'+str(i-1)] if i else []) for i in range(4)]
    return dict(package=dict(identity={'package':'bound'},source_manifest={'path':'/frozen/source/manifest.json','sha256':'s'},
        tuning_manifest={'path':'/frozen/tuning/manifest.json','sha256':'t'},manifest={'identity':{'model_id':'7b'}}),
        baseline_jobs=baselines,baseline_refs={'jobs':{'path':'/original/jobs.json','sha256':'b'}},
        identity={'matrix_parent':{'path':'/parent.json','sha256':'p'}},parent=Path('/parent.json'))


@pytest.fixture
def terminal(followup,inputs,tmp_path,monkeypatch):
    state=dict(status='needs_diagnosis',package_identity=inputs['package']['identity'],stages=[{'job_id':'last-low-m'}])
    driver_path=tmp_path/'driver.json';queue_path=tmp_path/'queue.json';out=tmp_path/'followup'
    state['queue_path']=str(queue_path.resolve())
    write(driver_path,state);write(queue_path,dict(jobs={'last-low-m':dict(job('last-low-m'),status='succeeded',lease_id=None)},leases={}))
    monkeypatch.setattr(followup,'verify_inputs',lambda *_:None)
    return driver_path,queue_path,out


def test_serial_batch_links_only_its_own_jobs_and_preserves_originals(followup,inputs):
    baseline=deepcopy(inputs['baseline_jobs']);matrix=[job('pd-'+str(i)) for i in range(3)]
    specs=followup.serial_specs(matrix,baseline,'low-m-last')
    assert specs[0]['payload']['after_terminal']==['low-m-last']
    assert specs[3]['payload']['after_terminal']==['pd-2']
    assert all(specs[i]['payload']['after_terminal']==[specs[i-1]['job_id']] for i in range(1,7))
    assert baseline==inputs['baseline_jobs'] and all(not j['payload']['after_terminal'] for j in matrix)


@pytest.mark.parametrize('status,owned,expected',[
    ('succeeded',False,(True,'last-low-m')),('failed',False,(True,'last-low-m')),
    ('cancelled',False,(True,'last-low-m')),('blocked',False,(True,None)),
    ('running',True,(False,None)),('queued',False,(False,None)),('succeeded',True,(False,None))])
def test_release_gate_does_not_attach_an_impossible_blocked_edge(followup,status,owned,expected):
    queue=dict(jobs={'last-low-m':dict(status=status,lease_id='lease' if owned else None)},leases={})
    assert followup.released_predecessor(queue,{'stages':[{'job_id':'last-low-m'}]})==expected


def test_active_attempt_blocks_even_if_job_claims_terminal(followup):
    queue=dict(jobs={'last-low-m':dict(status='succeeded',lease_id=None)},
               leases={'active':dict(job_id='last-low-m',status='active')})
    assert followup.released_predecessor(queue,{'stages':[{'job_id':'last-low-m'}]})==(False,None)


def test_image_preflight_has_no_gpu_or_writable_project_access(followup,inputs):
    command=followup.image_command(inputs['package'],{'path':'/campaign.json','sha256':'c'},[job('pd')])
    prefix=command[:command.index('sha256:image')]
    assert '--gpus' not in prefix and '--cap-add' not in prefix and '--device' not in prefix
    assert prefix[prefix.index('--runtime')+1]=='runc' and prefix[prefix.index('--network')+1]=='none'
    assert 'NVIDIA_VISIBLE_DEVICES=void' in prefix
    assert all(prefix[i+1].endswith(':ro') for i,arg in enumerate(prefix) if arg=='-v')


def test_missing_or_running_driver_only_persists_wait_state(followup,inputs,tmp_path,monkeypatch):
    monkeypatch.setattr(followup,'verify_inputs',lambda *_:pytest.fail('active tuning reached handoff preparation'))
    state=followup.tick(inputs,tmp_path/'absent.json',tmp_path/'unused-queue.json',tmp_path/'out',python='unused',enqueue=True)
    assert state['status']=='waiting_low_m'
    assert (tmp_path/'out/followup-status.json').exists()


def test_state_cannot_switch_queue_or_upstream_driver(followup,inputs,tmp_path):
    followup.tick(inputs,tmp_path/'absent.json',tmp_path/'one.json',tmp_path/'out',python='unused')
    with pytest.raises(ValueError,match='another queue or driver path'):
        followup.tick(inputs,tmp_path/'absent.json',tmp_path/'two.json',tmp_path/'out',python='unused',enqueue=True)


def test_terminal_driver_from_another_queue_is_not_deployed(followup,inputs,terminal):
    driver_path,queue_path,out=terminal;state=json.loads(driver_path.read_text())
    state['queue_path']='/another-queue.json';write(driver_path,state)
    result=followup.tick(inputs,driver_path,queue_path,out,python='unused',enqueue=True)
    assert result['status']=='needs_diagnosis' and not result['submitted']


def test_failed_tuning_prepares_only_baseline_without_queue_mutation(followup,inputs,terminal,monkeypatch):
    driver_path,queue_path,out=terminal;before=queue_path.read_bytes()
    monkeypatch.setattr(followup,'prepare_matrix',lambda *a,**kw:pytest.fail('failed tuning deployed PD'))
    monkeypatch.setattr(followup.d,'frozen_call',lambda *a,**kw:pytest.fail('default preparation enqueued'))
    state=followup.tick(inputs,driver_path,queue_path,out,python='unused')
    assert state['status']=='prepared' and state['mode']=='baseline_facts_only'
    specs=followup.d.read_bound(state['queue_specs'])['jobs']
    assert len(specs)==4 and specs[0]['payload']['after_terminal']==['last-low-m']
    again=followup.tick(inputs,driver_path,queue_path,out,python='unused')
    assert again['queue_specs']==state['queue_specs'] and queue_path.read_bytes()==before


def test_finished_but_missing_floor_also_prepares_only_baseline(followup,inputs,terminal):
    driver_path,queue_path,out=terminal;state=json.loads(driver_path.read_text())
    state.update(status='finished',result={'floor':None,'unqualified_rates':[2]});write(driver_path,state)
    result=followup.tick(inputs,driver_path,queue_path,out,python='unused')
    assert result['mode']=='baseline_facts_only' and result['status']=='prepared'


def test_baseline_enqueue_is_idempotent_and_never_claims_goal_complete(followup,inputs,terminal,monkeypatch):
    driver_path,queue_path,out=terminal;calls=[]
    def frozen(_package,operation,**kwargs):
        assert operation=='enqueue';spec=kwargs['job'];calls.append(spec['job_id'])
        queue=json.loads(queue_path.read_text());queue['jobs'][spec['job_id']]=dict(spec,status='queued',lease_id=None)
        write(queue_path,queue);return {'status':'queued','job_id':spec['job_id']}
    monkeypatch.setattr(followup.d,'frozen_call',frozen)
    state=followup.tick(inputs,driver_path,queue_path,out,python='unused',enqueue=True)
    assert state['status']=='awaiting_pd_optimization' and state['goal_complete'] is False
    followup.tick(inputs,driver_path,queue_path,out,python='unused',enqueue=True)
    assert calls==['base-0','base-1','base-2','base-3']


def test_enqueue_crash_reconciles_prior_submission_without_duplicate(followup,inputs,terminal,monkeypatch):
    driver_path,queue_path,out=terminal;calls=[];crashed=[]
    def frozen(_package,operation,**kwargs):
        assert operation=='enqueue';spec=kwargs['job'];calls.append(spec['job_id'])
        queue=json.loads(queue_path.read_text());queue['jobs'][spec['job_id']]=dict(spec,status='queued',lease_id=None)
        write(queue_path,queue)
        if not crashed:crashed.append(True);raise KeyboardInterrupt('after durable enqueue')
        return {'status':'queued'}
    monkeypatch.setattr(followup.d,'frozen_call',frozen)
    with pytest.raises(KeyboardInterrupt):followup.tick(inputs,driver_path,queue_path,out,python='unused',enqueue=True)
    state=followup.tick(inputs,driver_path,queue_path,out,python='unused',enqueue=True)
    assert state['status']=='awaiting_pd_optimization' and calls==['base-0','base-1','base-2','base-3']


def configure_matrix(followup,inputs,terminal,monkeypatch):
    driver_path,queue_path,out=terminal;state=json.loads(driver_path.read_text())
    floor={'path':'/floor.json','sha256':'f'};state.update(status='finished',result={'floor':floor});write(driver_path,state)
    monkeypatch.setattr(followup,'validate_floor',lambda *a,**kw:dict(passed=True,floor=floor))
    refs={'inputs':[{'source':inputs['package']['source_manifest']}], 'campaign':{'path':'/matrix/campaign.json','sha256':'m'}}
    campaign=dict(execution_source_manifest=inputs['package']['source_manifest'],parent_campaign=inputs['identity']['matrix_parent'],
        repair_protocol=dict(capacity_floor=floor,capacity_points=['r2','r4'],frozen_source_reused=True,canonical_startup_for_capacity_v2=True))
    jobs=[job('pd-'+str(i),['pd-'+str(i-1)] if i else []) for i in range(3)]
    calls=[]
    def prepare(_package,parent,target,capacity,predecessor,**kwargs):
        assert capacity==floor and predecessor=='last-low-m';calls.append('prepare')
        for name in ('campaign.json','jobs.json','preparation.json'):write(target/name,{})
    monkeypatch.setattr(followup,'prepare_matrix',prepare)
    monkeypatch.setattr(followup,'package_jobs',lambda *a,**kw:(refs,campaign,jobs))
    monkeypatch.setattr(followup,'preflight_matrix',lambda *a,**kw:calls.append('preflight') or {'passed':True})
    return calls


def test_qualified_floor_prepares_three_matrix_then_four_baseline_jobs(followup,inputs,terminal,monkeypatch):
    calls=configure_matrix(followup,inputs,terminal,monkeypatch)
    driver_path,queue_path,out=terminal
    state=followup.tick(inputs,driver_path,queue_path,out,python='unused')
    assert state['status']=='prepared' and state['mode']=='qualified_matrix_then_baselines'
    specs=followup.d.read_bound(state['queue_specs'])['jobs']
    assert [spec['job_id'] for spec in specs]==['pd-0','pd-1','pd-2','base-0','base-1','base-2','base-3']
    assert calls==['prepare','preflight'] and specs[3]['payload']['after_terminal']==['pd-2']


def test_matrix_image_failure_prevents_the_entire_batch_enqueue(followup,inputs,terminal,monkeypatch):
    configure_matrix(followup,inputs,terminal,monkeypatch);driver_path,queue_path,out=terminal
    def fail(*args,**kwargs):raise ValueError('image preflight failed')
    monkeypatch.setattr(followup,'preflight_matrix',fail)
    monkeypatch.setattr(followup.d,'frozen_call',lambda *a,**kw:pytest.fail('image failure submitted a job'))
    state=followup.tick(inputs,driver_path,queue_path,out,python='unused',enqueue=True)
    assert state['status']=='needs_diagnosis' and not state['submitted']


def test_late_batch_conflict_is_detected_before_first_enqueue(followup,inputs,terminal,monkeypatch):
    configure_matrix(followup,inputs,terminal,monkeypatch);driver_path,queue_path,out=terminal
    queue=json.loads(queue_path.read_text());queue['jobs']['base-3']=dict(job('base-3'),status='queued',priority=999)
    write(queue_path,queue)
    monkeypatch.setattr(followup.d,'frozen_call',lambda *a,**kw:pytest.fail('conflicting batch partially submitted'))
    state=followup.tick(inputs,driver_path,queue_path,out,python='unused',enqueue=True)
    assert state['status']=='needs_diagnosis' and not state['submitted']


def test_full_floor_structure_and_raw_revalidation_both_required(followup,inputs,tmp_path,monkeypatch):
    package=inputs['package'];seeds=[8801,8802,8803];families=['nominal','burst25','short_output','long_prompt','initial_burst','tail_burst']
    package['manifest'].update(seeds=seeds,families=families,trials=[{'nominal_rate_rps':rate} for rate in (2,4)])
    rows=[];floors=[]
    for rate in (2,4):
        ids=[]
        for seed in seeds:
            for family in families:
                identity=f'{rate}-{seed}-{family}';ids.append(identity);path=tmp_path/(identity+'.json');write(path,{'trial_id':identity})
                rows.append(dict(trial_id=identity,seed=seed,family=family,receipt=followup.d.binding(path)))
        floors.append(dict(nominal_rate_rps=rate,trial_ids=ids))
    summary={'rows':rows,'selected':floors};write(tmp_path/'summary.json',summary)
    artifact=dict(kind='pdblend_capacity_floor_v2',formal_eligible=False,tuning_manifest=package['tuning_manifest'],
                  identity=package['manifest']['identity'],floors=floors,trial_receipts=[r['receipt'] for r in rows])
    write(tmp_path/'floor.json',artifact)
    state=dict(status='finished',package_identity=package['identity'],result=dict(unqualified_rates=[],qualified_rates=[2,4],
        floor=followup.d.binding(tmp_path/'floor.json'),summary=followup.d.binding(tmp_path/'summary.json')))
    monkeypatch.setattr(followup.d,'frozen_call',lambda *a,**kw:dict(summary,rejected=[]))
    assert followup.validate_floor(package,state,python='unused')['passed']
    monkeypatch.setattr(followup.d,'frozen_call',lambda *a,**kw:dict(summary,rejected=['broken raw receipt']))
    with pytest.raises(ValueError,match='raw validation'):followup.validate_floor(package,state,python='unused')
    artifact['floors'][1]['trial_ids'].pop();write(tmp_path/'floor.json',artifact)
    summary['selected']=artifact['floors'];write(tmp_path/'summary.json',summary)
    state['result'].update(floor=followup.d.binding(tmp_path/'floor.json'),summary=followup.d.binding(tmp_path/'summary.json'))
    with pytest.raises(ValueError,match='18 raw trials'):followup.floor_structure(package,state)
