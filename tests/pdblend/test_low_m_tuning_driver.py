"""CPU-only orchestration checks; all queue writes stay inside tmp_path."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys

import pytest


@pytest.fixture
def driver():
    path=Path(__file__).resolve().parents[2]/'scripts/2026-09-24_drive_low_m_tuning.py'
    spec=importlib.util.spec_from_file_location('low_m_tuning_driver',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))


@pytest.mark.parametrize('local_stop', [True, False])
def test_user_stop_precedes_loading_or_queuing_new_qualification(driver, tmp_path, monkeypatch, local_stop):
    package = tmp_path / 'new-low-m-version'
    stop = package / 'driver/user-stop.json' if local_stop else tmp_path / 'matrix-handoff-request.json'
    write(stop, dict(new_low_m_qualification_authorized=False))
    monkeypatch.setattr(sys, 'argv', ['driver', '--package', str(package),
        '--queue', str(tmp_path / 'queue.json'), '--enqueue'])
    monkeypatch.setattr(driver, 'load_package', lambda *a: pytest.fail('user stop reached package loading'))
    assert driver.main() == 0
    assert not (tmp_path / 'queue.json').exists()


def trial_id(rate,frequency,seed=8801,family='nominal'):
    return f'r{rate}-f{frequency}-s{seed}-{family}'


@pytest.fixture
def manifest():
    seeds=[8801,8802,8803]
    families=['nominal','burst25','short_output','long_prompt','initial_burst','tail_burst']
    return dict(identity={'model_id':'model'},selection_seed=8801,seeds=seeds,families=families,trials=[
        dict(id=trial_id(rate,freq,seed,family),nominal_rate_rps=rate,seed=seed,family=family,
             plan={'f_M':freq})
        for rate in (2,4) for freq in (2100,1800,1500,1200) for seed in seeds for family in families])


def record(identity,energy=100,accepted=True):
    return dict(trial_id=identity,accepted=accepted,energy_service_tail_j=energy,
                reason=None if accepted else 'zero SLO misses required',receipt={'path':identity,'sha256':'bound'})


def all_nominals(manifest,energies=(100,90,80,85)):
    return {trial['id']:record(trial['id'],energies[(2100,1800,1500,1200).index(trial['plan']['f_M'])])
            for trial in manifest['trials'] if trial['family']=='nominal' and trial['seed']==8801}


@pytest.fixture
def package(manifest):
    initial=[trial_id(rate,2100) for rate in (2,4)]
    source={'path':'/frozen/source/manifest.json','sha256':'source'}
    tuning={'path':'/frozen/tuning/manifest.json','sha256':'tuning'}
    group=dict(scope='independent_low_m_tuning/v2',source_manifest=source,tuning_manifest=tuning,
               session_id='initial-session',engine_identity={'model_id':'model'})
    payload=dict(argv=['docker','run','--name','initial-job','-v','/frozen/source:/opt/pdblend-src:ro',
        'image','-B','-m','pdblend.bench.low_m_tuning_runtime','--group','/initial-group.json',
        '--output','{attempt_dir}/session','--base-port','{lease_port}','--trial-ids',*initial],
        gpu_count=8,exclusive=True,reserve_host=True,required_receipts=['session/completion.json'],
        container_name='initial-job',session_id='initial-session',trial_ids=initial,
        source_sha256='source',timeout_s=7200,after_terminal=['previous-experiment'])
    return dict(identity={'package':'bound'},manifest=manifest,first_trials=initial,source_manifest=source,
                tuning_manifest=tuning,base_group=group,
                template=dict(job_id='initial-job',payload=payload,priority=800,max_attempts=1))


def test_screen_starts_at_highest_frequency_for_both_rates(driver,manifest):
    decision=driver.next_decision(manifest,{})
    assert decision['kind']=='screen'
    assert decision['trial_ids']==[trial_id(2,2100),trial_id(4,2100)]


def test_failed_nominal_stops_only_that_rates_descent(driver,manifest):
    records={trial_id(2,2100):record(trial_id(2,2100),accepted=False),
             trial_id(4,2100):record(trial_id(4,2100))}
    decision=driver.next_decision(manifest,records)
    assert decision['trial_ids']==[trial_id(4,1800)]
    assert decision['rate_decisions']['2']['nominal_stop']['frequency_mhz']==2100


def test_manifest_can_extend_descending_search_to_900_without_driver_changes(driver,manifest):
    added=[]
    for trial in manifest['trials']:
        if trial['plan']['f_M']==1200:
            row=deepcopy(trial);row['plan']['f_M']=900
            row['id']=trial_id(row['nominal_rate_rps'],900,row['seed'],row['family']);added.append(row)
    records=all_nominals(manifest);manifest['trials'].extend(added)
    decision=driver.next_decision(manifest,records)
    assert len(manifest['trials'])==180 and decision['kind']=='screen'
    assert decision['trial_ids']==[trial_id(2,900),trial_id(4,900)]


def test_missing_results_do_not_qualify_or_issue_floor(driver,manifest):
    records={trial_id(rate,2100):record(trial_id(rate,2100),accepted=False) for rate in (2,4)}
    decision=driver.next_decision(manifest,records)
    assert decision['kind']=='finalize'
    assert decision['unqualified_rates']==[2,4]


@pytest.mark.parametrize('energy',[None,0,-1,float('nan'),float('inf'),True])
def test_invalid_nominal_objective_stops_decision(driver,manifest,energy):
    with pytest.raises(ValueError,match='finite positive full energy'):
        driver.next_decision(manifest,{trial_id(2,2100):record(trial_id(2,2100),energy)})


def test_only_same_nominal_seed_ranks_energy_and_adjacent_candidate(driver,manifest):
    records=all_nominals(manifest)
    # A spectacular stress energy cannot change the nominal objective.
    identity=trial_id(2,1200,8802,'long_prompt');records[identity]=record(identity,.01)
    decision=driver.next_decision(manifest,records)
    assert decision['kind']=='qualify' and len(decision['trial_ids'])==34
    for info in decision['rate_decisions'].values():
        assert [row['frequency_mhz'] for row in info['candidates']]==[1500,1800]
    assert all('-f1500-' in identity for identity in decision['trial_ids'])


def test_rejected_best_qualifies_conservative_candidate(driver,manifest):
    records=all_nominals(manifest)
    for rate in (2,4):
        identity=trial_id(rate,1500,8802,'long_prompt');records[identity]=record(identity,accepted=False)
    decision=driver.next_decision(manifest,records)
    assert len(decision['trial_ids'])==34
    assert all('-f1800-' in identity for identity in decision['trial_ids'])
    assert all(info['candidates'][0]['status']=='rejected' for info in decision['rate_decisions'].values())


def test_complete_seed_family_domains_are_required(driver,manifest):
    records=all_nominals(manifest)
    for trial in manifest['trials']:
        if trial['plan']['f_M'] in (1500,1800):records.setdefault(trial['id'],record(trial['id']))
    decision=driver.next_decision(manifest,records)
    assert decision['kind']=='finalize' and not decision['unqualified_rates']
    assert all(info['qualified']==[1500,1800] for info in decision['rate_decisions'].values())
    # One missing family is not replaced by another seed/family pair.
    manifest['trials'].remove(next(t for t in manifest['trials'] if t['id']==trial_id(2,1500,8803,'tail_burst')))
    with pytest.raises(ValueError,match='exact complete'):
        driver.next_decision(manifest,records)


def test_stage_preserves_frozen_execution_and_is_immutable(driver,package,tmp_path):
    template=deepcopy(package['template']);base=deepcopy(package['base_group'])
    ids=[trial_id(2,1800),trial_id(4,1800)]
    kwargs=dict(directory=tmp_path,index=1,kind='screen',trial_ids=ids,predecessor='initial-job')
    stage=driver.derived_stage(template,base,**kwargs)
    assert stage==driver.derived_stage(template,base,**kwargs)
    assert template==package['template'] and base==package['base_group']
    payload=stage['job']['payload'];group=driver.read_bound(stage['group'])
    assert payload['after_terminal']==['initial-job']
    assert payload['source_sha256']=='source' and payload['required_receipts']==['session/completion.json']
    assert payload['argv'][-3:]==['--trial-ids',*ids]
    assert group['source_manifest']==package['source_manifest'] and group['tuning_manifest']==package['tuning_manifest']
    assert group['trial_ids']==ids and group['session_id']!='initial-session'
    assert stage['job']['priority']==800 and stage['job']['max_attempts']==1
    with pytest.raises(ValueError,match='immutable driver artifact'):
        driver.derived_stage(template,base,**dict(kwargs,trial_ids=[trial_id(2,1500)]))


def test_qualification_timeout_accounts_for_all_34_windows(driver,package,tmp_path):
    ids=[t['id'] for t in package['manifest']['trials'] if t['plan']['f_M']==1500 and
         not (t['seed']==8801 and t['family']=='nominal')]
    stage=driver.derived_stage(package['template'],package['base_group'],directory=tmp_path,index=1,
                              kind='qualify',trial_ids=ids,predecessor='previous')
    assert len(ids)==34 and stage['job']['payload']['timeout_s']>=34*450


def test_repeated_waits_cache_preflight_and_do_not_touch_queue(driver,package,tmp_path,monkeypatch):
    queue_path=tmp_path/'queue.json';state_path=tmp_path/'state.json'
    job=dict(package['template'],status='running',lease_id='active')
    write(queue_path,dict(jobs={job['job_id']:job},leases={}))
    before=queue_path.read_bytes();calls=[]
    def frozen(_package,operation,**kwargs):
        calls.append(operation);assert operation=='preflight';return {'validated':True}
    monkeypatch.setattr(driver,'frozen_call',frozen)
    for _ in range(3):assert driver.tick(package,queue_path,state_path,python='unused')['status']=='waiting'
    assert calls==['preflight'] and queue_path.read_bytes()==before


def test_state_cannot_be_reused_against_another_queue(driver,package,tmp_path,monkeypatch):
    queue_path=tmp_path/'one.json';state_path=tmp_path/'state.json'
    write(queue_path,dict(jobs={},leases={}))
    monkeypatch.setattr(driver,'frozen_call',lambda *a,**kw:{'validated':True})
    driver.tick(package,queue_path,state_path,python='unused')
    with pytest.raises(ValueError,match='different lease queue'):
        driver.tick(package,tmp_path/'two.json',state_path,python='unused',enqueue=True)


def setup_completed_initial(driver,package,tmp_path,monkeypatch):
    queue_path=tmp_path/'queue.json';state_path=tmp_path/'state.json'
    job=dict(package['template'],status='succeeded',lease_id=None)
    write(queue_path,dict(jobs={job['job_id']:job},leases={}))
    monkeypatch.setattr(driver,'harvest',lambda *args,**kwargs:
        ([record(identity) for identity in package['first_trials']],{'verified':True}))
    return queue_path,state_path


def test_default_prepares_stage_without_enqueue_then_restarts_idempotently(driver,package,tmp_path,monkeypatch):
    queue_path,state_path=setup_completed_initial(driver,package,tmp_path,monkeypatch);calls=[]
    def frozen(_package,operation,**kwargs):
        calls.append(operation);assert operation=='preflight';return {'validated':True}
    monkeypatch.setattr(driver,'frozen_call',frozen)
    first=driver.tick(package,queue_path,state_path,python='unused')
    assert first['status']=='prepared' and len(first['stages'])==2
    second=driver.tick(package,queue_path,state_path,python='unused')
    assert second['stages']==first['stages'] and len(second['decisions'])==1
    assert calls==['preflight']
    assert set(json.loads(queue_path.read_text())['jobs'])=={'initial-job'}


def test_crash_after_enqueue_reconciles_same_job_without_duplicate(driver,package,tmp_path,monkeypatch):
    queue_path,state_path=setup_completed_initial(driver,package,tmp_path,monkeypatch);calls=[]
    def frozen(_package,operation,**kwargs):
        calls.append(operation)
        if operation=='preflight':return {'validated':True}
        assert operation=='enqueue'
        queue=json.loads(queue_path.read_text());job=kwargs['job']
        queue['jobs'][job['job_id']]=dict(job,status='queued',lease_id=None);write(queue_path,queue)
        raise KeyboardInterrupt('simulated interruption after durable queue enqueue')
    monkeypatch.setattr(driver,'frozen_call',frozen)
    with pytest.raises(KeyboardInterrupt):driver.tick(package,queue_path,state_path,python='unused',enqueue=True)
    state=driver.tick(package,queue_path,state_path,python='unused',enqueue=True)
    assert state['status']=='waiting' and len(state['stages'])==2
    assert calls==['preflight','enqueue'] and len(json.loads(queue_path.read_text())['jobs'])==2


def test_crash_after_decision_file_replays_in_stable_receipt_order(driver,package,tmp_path,monkeypatch):
    queue_path,state_path=setup_completed_initial(driver,package,tmp_path,monkeypatch)
    monkeypatch.setattr(driver,'harvest',lambda *args,**kwargs:
        ([record(identity) for identity in reversed(package['first_trials'])],{'verified':True}))
    monkeypatch.setattr(driver,'frozen_call',lambda *args,**kwargs:{'validated':True})
    original=driver.write_once;crashed=[]
    def write_then_crash(path,value):
        result=original(path,value)
        if Path(path).name=='decision-000.json' and not crashed:
            crashed.append(True);raise KeyboardInterrupt('decision artifact committed before state')
        return result
    monkeypatch.setattr(driver,'write_once',write_then_crash)
    with pytest.raises(KeyboardInterrupt):driver.tick(package,queue_path,state_path,python='unused')
    before=(tmp_path/'decision-000.json').read_bytes()
    state=driver.tick(package,queue_path,state_path,python='unused')
    assert state['status']=='prepared' and len(state['decisions'])==1
    assert (tmp_path/'decision-000.json').read_bytes()==before


@pytest.mark.parametrize('status',['blocked','failed','cancelled'])
def test_failed_job_halts_without_new_stage(driver,package,tmp_path,monkeypatch,status):
    queue_path=tmp_path/'queue.json';state_path=tmp_path/'state.json'
    write(queue_path,dict(jobs={'initial-job':dict(package['template'],status=status)},leases={}))
    calls=[]
    monkeypatch.setattr(driver,'frozen_call',lambda _p,operation,**kw:calls.append(operation) or {'validated':True})
    state=driver.tick(package,queue_path,state_path,python='unused',enqueue=True)
    assert state['status']=='needs_diagnosis' and len(state['stages'])==1 and calls==['preflight']


def test_changed_queue_spec_halts_without_enqueue(driver,package,tmp_path,monkeypatch):
    queue_path=tmp_path/'queue.json';job=deepcopy(package['template']);job['payload']['source_sha256']='other'
    write(queue_path,dict(jobs={'initial-job':dict(job,status='queued')},leases={}))
    monkeypatch.setattr(driver,'frozen_call',lambda *args,**kwargs:{'validated':True})
    state=driver.tick(package,queue_path,tmp_path/'state.json',python='unused',enqueue=True)
    assert state['status']=='needs_diagnosis' and 'immutable stage' in state['diagnosis']


def test_success_requires_released_latest_attempt(driver,tmp_path):
    queue=dict(jobs={'job':{'lease_id':None}},leases={'one':dict(job_id='job',attempt=1,
               status='succeeded',attempt_dir=str(tmp_path))})
    assert driver.successful_attempt(queue,'job')==tmp_path
    queue['leases']['two']=dict(job_id='job',attempt=2,status='failed',attempt_dir=str(tmp_path/'failed'))
    with pytest.raises(ValueError,match='latest queue attempt'):driver.successful_attempt(queue,'job')
    queue['jobs']['job']['lease_id']='still-owned'
    with pytest.raises(ValueError,match='still owns'):driver.successful_attempt(queue,'job')


@pytest.mark.parametrize('count',[0,1,17,18])
def test_no_floor_until_frozen_summary_has_full_18(driver,package,tmp_path,monkeypatch,count):
    trials=driver.candidate_trials(package['manifest'],2,1500)[:count]
    rows=[dict(trial_id=t['id'],seed=t['seed'],family=t['family'],receipt=record(t['id'])['receipt']) for t in trials]
    records={row['trial_id']:dict(row,accepted=True) for row in rows}
    selected=[dict(nominal_rate_rps=2,frequency_mhz=1500,trial_ids=[row['trial_id'] for row in rows])] if count else []
    report=dict(rows=rows,selected=selected)
    decision=dict(unqualified_rates=[4],rate_decisions={'2':dict(rate_rps=2,qualified=[1500] if count else [])})
    monkeypatch.setattr(driver,'frozen_call',lambda *args,**kwargs:report)
    if count in (1,17):
        with pytest.raises(ValueError,match='incomplete capacity domain'):
            driver.freeze_result(package,{'records':records},decision,python='unused',output=tmp_path)
        assert not (tmp_path/'capacity-floor-v2.json').exists()
    else:
        result=driver.freeze_result(package,{'records':records},decision,python='unused',output=tmp_path)
        assert bool(result['floor'])==(count==18)


def test_final_revalidation_cannot_silently_lose_previously_accepted_trials(driver,package,tmp_path,monkeypatch):
    identity=trial_id(2,2100)
    monkeypatch.setattr(driver,'frozen_call',lambda *args,**kwargs:{'rows':[],'selected':[]})
    with pytest.raises(ValueError,match='final raw validation differs'):
        driver.freeze_result(package,{'records':{identity:record(identity)}},{},python='unused',output=tmp_path)
    assert not (tmp_path/'capacity-floor-v2.json').exists()


def test_source_verification_rejects_byte_change(driver,tmp_path):
    code=tmp_path/'pdblend/file.py';code.parent.mkdir();code.write_text('value = 1\n')
    files={'pdblend/file.py':driver.binding(code)['sha256']}
    write(tmp_path/'manifest.json',{'files':files,'source_sha256':driver.digest(files)})
    ref=driver.binding(tmp_path/'manifest.json')
    assert driver.verify_source(ref)==tmp_path
    code.write_text('value = 2\n')
    with pytest.raises(ValueError,match='source bytes changed'):driver.verify_source(ref)


@pytest.mark.parametrize('cleanup_passed',[True,False])
def test_harvest_requires_actual_cleanup_and_reaudits_trial_receipts(driver,package,tmp_path,monkeypatch,cleanup_passed):
    stage=driver.initialize_state(package)['stages'][0];attempt=tmp_path/'attempt';session=attempt/'session'
    cleanup=dict(passed=cleanup_passed,process_cleanup_verified=cleanup_passed)
    write(session/'cleanup.json',cleanup)
    write(session/'completion.json',dict(scope=driver.SCOPE,status='passed',complete=True,
        cleanup=cleanup,cleanup_receipt=driver.binding(session/'cleanup.json'),
        results={identity:{'accepted':False} for identity in stage['trial_ids']}))
    for identity in stage['trial_ids']:
        write(session/'trials'/identity/'trial-receipt.json',dict(trial_id=identity,manifest=package['tuning_manifest']))
    queue=dict(jobs={'initial-job':{'lease_id':None}},leases={'lease':dict(job_id='initial-job',attempt=1,
        status='succeeded',attempt_dir=str(attempt))})
    calls=[]
    def frozen(_package,operation,**kwargs):
        calls.append(operation);assert operation=='audit'
        assert len(kwargs['receipts'])==2
        return [dict(record(identity),receipt=ref) for identity,ref in zip(stage['trial_ids'],kwargs['receipts'])]
    monkeypatch.setattr(driver,'frozen_call',frozen)
    if cleanup_passed:
        rows,evidence=driver.harvest(package,stage,queue,python='unused',output=tmp_path/'driver')
        assert all(row['accepted'] for row in rows) and calls==['audit']
        assert driver.read_bound(evidence['audit'])['cleanup']==driver.binding(session/'cleanup.json')
    else:
        with pytest.raises(ValueError,match='cleanup is unverified'):
            driver.harvest(package,stage,queue,python='unused',output=tmp_path/'driver')
        assert not calls


def test_final_summary_order_survives_sorted_state_reload(driver,package,tmp_path,monkeypatch):
    records={identity:record(identity) for identity in reversed(package['first_trials'])}
    state={'records':records};decision=dict(unqualified_rates=[2,4],rate_decisions={})
    def frozen(_package,operation,**kwargs):
        assert operation=='summarize'
        return dict(rows=[dict(trial_id=ref['path'],receipt=ref) for ref in kwargs['receipts']],selected=[])
    monkeypatch.setattr(driver,'frozen_call',frozen)
    first=driver.freeze_result(package,state,decision,python='unused',output=tmp_path)
    reloaded=json.loads(json.dumps(state,sort_keys=True))
    assert driver.freeze_result(package,reloaded,decision,python='unused',output=tmp_path)==first
