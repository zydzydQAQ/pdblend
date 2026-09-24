from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest

from pdblend.bench.cohort_dominance import (BASELINE_SYSTEMS,STANDARD_CONDITIONS,
    declare_paired_repeats,evaluate_declared_repeats,identity)


def row(system,key,energy=100):
    model,dataset,scale=key;name=f'{model}-{dataset}-{scale}-{system}'
    reference=dict(trace_sha256='trace-'+str(key),seed=701,duration_s=150.,slo_ttft_s=5.,slo_tpot_s=.15,
        offered_requests=100,offered_rps=2.,measurement_protocol_version='native-comparison-150s/v1',
        model_hash=model,tokenizer_hash='tokens',image_digest='image',runtime_source_sha256='engine',
        measurement_source_sha256='meter',gpu_uuids=[f'gpu{i}' for i in range(8)])
    return dict(system=system,series=system,model=model,dataset=dataset,rate_scale=scale,offered_rps=2.,
        revision=system+'-revision',point_id=name,receipt_path=name+'/receipt.json',receipt_sha256=name,
        offered_requests=100,successful_requests=100,failed_requests=0,unresolved_requests=0,
        joint_slo_requests=100,all_requests_successful=True,ttft_p99_s=1.,tpot_p99_s=.1,
        slo_ttft_s=5.,slo_tpot_s=.15,service_energy_kj=energy-10,tail_energy_kj=10.,total_energy_kj=energy,
        energy_measurement_complete=True,measurement_evidence_valid=True,cleanup_passed=True,
        comparison_identity=reference,service_start_s=10.,formal_eligible=False)


@pytest.fixture
def snapshot():
    return [row(system,key,98 if key==STANDARD_CONDITIONS[0] and system=='pdblend' else 90 if system=='pdblend' else 100)
            for key in STANDARD_CONDITIONS for system in ('pdblend',*BASELINE_SYSTEMS)]


def declare(snapshot):
    protocol=declare_paired_repeats(snapshot,candidate_revision='pdblend-revision',declared_at_s=100.)
    observations=[]
    for case in protocol['conditions']:
        if not case['repeat_required']:continue
        for repeat in (1,2,3):
            for system in ('pdblend',*BASELINE_SYSTEMS):
                reference=case['candidate'] if system=='pdblend' else case['baselines'][system]
                name=reference['point_id']+f'-repeat-{repeat}'
                protocol['planned_observations'].append(dict(condition_id=case['condition_id'],repeat_id=repeat,
                    system=system,point_id=name,revision=reference['revision'],comparison_identity=reference['comparison_identity']))
                value=row(system,tuple(case['condition']),98 if system=='pdblend' else 100)
                value.update(point_id=name,repeat_id=repeat,receipt_path=name+'/receipt.json',receipt_sha256=name,service_start_s=200+repeat)
                observations.append(value)
    return protocol,observations


def test_exact_36_context_selects_only_strictly_positive_below_3_percent(snapshot):
    protocol,observations=declare(snapshot)
    assert len(protocol['conditions'])==36 and sum(c['repeat_required'] for c in protocol['conditions'])==1
    assert len(observations)==15 and protocol['repeat_ids']==[1,2,3]
    with pytest.raises(ValueError,match='36 original conditions'):
        declare_paired_repeats(snapshot[5:],candidate_revision='pdblend-revision',declared_at_s=100)


def test_three_predeclared_rounds_all_pass_before_stable(snapshot):
    protocol,observations=declare(snapshot)
    complete=evaluate_declared_repeats(protocol,observations)
    assert complete['conditions'][0]['status']=='stable_observed_win'
    assert complete['all_36_observed_criteria_met'] is True and complete['goal_complete'] is False
    assert complete['requires_result_review'] is True
    assert evaluate_declared_repeats(protocol,observations[:10])['conditions'][0]['status']=='incomplete'


@pytest.mark.parametrize('change',[
    dict(successful_requests=99,failed_requests=1,all_requests_successful=False),dict(unresolved_requests=1),
    dict(joint_slo_requests=99),dict(joint_slo_requests=89),dict(ttft_p99_s=5.01),dict(tpot_p99_s=.151),
    dict(total_energy_kj=100.,service_energy_kj=90.),dict(tail_energy_kj=None),
    dict(measurement_evidence_valid=False),dict(cleanup_passed=False)])
def test_one_bad_repeat_prevents_stable_and_diagnostic_cannot_replace_it(snapshot,change):
    protocol,observations=declare(snapshot);observations[5].update(change)
    diagnostic=deepcopy(observations[0]);diagnostic.update(point_id='diagnostic-round-4',repeat_id=4)
    result=evaluate_declared_repeats(protocol,[*observations,diagnostic])
    assert result['conditions'][0]['status']!='stable_observed_win'
    assert result['all_36_observed_criteria_met'] is False and result['diagnostics']


def test_failed_baseline_energy_is_still_an_absolute_target(snapshot):
    protocol,observations=declare(snapshot)
    baseline=next(r for r in observations if r['system']=='ecoserve' and r['repeat_id']==2)
    baseline.update(successful_requests=10,failed_requests=90,joint_slo_requests=10,all_requests_successful=False,
                    total_energy_kj=90.,service_energy_kj=80.)
    result=evaluate_declared_repeats(protocol,observations)
    assert result['conditions'][0]['status']=='failed'
    assert result['conditions'][0]['repeats'][1]['min_available_baseline_energy_kj']==90.


def test_repeat_cannot_lower_frozen_100_percent_attainment_target(snapshot):
    protocol,observations=declare(snapshot)
    for value in observations:
        if value['repeat_id']==2:value['joint_slo_requests']=99
    result=evaluate_declared_repeats(protocol,observations)
    repeat=result['conditions'][0]['repeats'][1]
    assert repeat['paired_repeat_pass'] is True and repeat['initial_target_pass'] is False
    assert repeat['frozen_initial_joint_slo_requests']==100 and result['conditions'][0]['status']=='failed'


def test_repeat_cannot_raise_frozen_initial_energy_target(snapshot):
    protocol,observations=declare(snapshot)
    for value in observations:
        if value['repeat_id']==2:
            value['total_energy_kj']=101. if value['system']=='pdblend' else 105.
            value['service_energy_kj']=value['total_energy_kj']-10
    result=evaluate_declared_repeats(protocol,observations)
    repeat=result['conditions'][0]['repeats'][1]
    assert repeat['paired_repeat_pass'] is True and repeat['initial_target_pass'] is False
    assert repeat['frozen_initial_min_energy_kj']==100 and repeat['saving_vs_frozen_initial_min_pct']<0
    assert result['conditions'][0]['status']=='failed'


def test_missing_baseline_in_initial_36_prevents_overall_win(snapshot):
    del snapshot[-1]
    protocol,observations=declare(snapshot);result=evaluate_declared_repeats(protocol,observations)
    assert result['conditions'][0]['status']=='stable_observed_win'
    assert result['conditions'][-1]['status']=='incomplete' and not result['all_36_observed_criteria_met']


@pytest.mark.parametrize('tamper',['hash','path','source','trace','timestamp','duplicate','repeat'])
def test_reused_replaced_or_unplanned_repeats_are_rejected(snapshot,tamper):
    protocol,observations=declare(snapshot)
    if tamper=='hash':observations[5]['receipt_sha256']=observations[0]['receipt_sha256']
    elif tamper=='path':observations[5]['receipt_path']=observations[0]['receipt_path']
    elif tamper=='source':observations[5]['revision']='new-unqualified-revision'
    elif tamper=='trace':observations[5]['comparison_identity']['trace_sha256']='another-trace'
    elif tamper=='timestamp':observations[5]['service_start_s']=99.
    elif tamper=='duplicate':observations.append(deepcopy(observations[5]))
    elif tamper=='repeat':observations[5]['repeat_id']=4
    result=evaluate_declared_repeats(protocol,observations)
    assert not result['all_36_observed_criteria_met'] and result['invalid_observations']


def test_protocol_cannot_change_round_count_or_skip_one_system(snapshot):
    protocol,observations=declare(snapshot);protocol['repeat_ids']=[1,2,3,4]
    with pytest.raises(ValueError,match='three predeclared'):evaluate_declared_repeats(protocol,observations)
    protocol['repeat_ids']=[1,2,3];protocol['planned_observations'].pop()
    with pytest.raises(ValueError,match='all 15 observations'):evaluate_declared_repeats(protocol,observations)


@pytest.fixture
def script():
    path=Path(__file__).resolve().parents[2]/'scripts/2026-09-24_prepare_paired_energy_repeats.py'
    spec=importlib.util.spec_from_file_location('paired_energy_script_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))


def test_clone_preserves_real_baseline_module_image_environment_and_startup(script,tmp_path,monkeypatch):
    config=tmp_path/'mixed-config.json';write(config,{'independent_policy':'mixed','startup':{'replicas':4,'freq':1800}})
    source={'path':'/frozen/mixed-revision/manifest.json','sha256':'source'}
    point=dict(name='mixed-case',revision='mixed-revision',system='mixed',source_manifest=source,
        engine_identity={'image_digest':'sha256:mixed-image'},inputs={'system_config':script.binding(config)},
        trace={'path':'same-trace','sha256':'same'},seed=701,duration_s=150,slo={'ttft_s':5.,'tpot_s':.15})
    group=dict(session_id='original-session',engine_identity=point['engine_identity'],points=[point])
    template=dict(point=point,group=group,jobs_ref={'path':'jobs','sha256':'j'},group_ref={'path':'group','sha256':'g'},
        job=dict(job_id='old-job',priority=1,max_attempts=1,payload=dict(source_sha256='mixed-revision',
            image_digest='sha256:mixed-image',required_receipts=['session/completion.json'],
            argv=['docker','run','--name','old-job','-e','PDBLEND_SOURCE_MANIFEST='+source['path'],
                  '-e','VLLM_USE_V1=0','sha256:mixed-image','-m','independent.mixed.runtime',
                  '--group','original-group','--out','{attempt_dir}/session'])))
    original=deepcopy(template);monkeypatch.setattr(script,'verify_source',lambda ref:None)
    claim=tmp_path/'old-attempt/manifest.json';write(claim,dict(schema=1,immutable=True,job_id='old-job',payload=template['job']['payload']))
    reference=dict(point_id='mixed-case',revision='mixed-revision',receipt_path=str(claim.parent/'session/windows/case/receipt.json'),comparison_identity={'trace_sha256':'same'})
    derived,new_group,new_job,plan=script.clone_observation(template,reference,design_ref={'path':'design','sha256':'design-sha'},
        condition_id='7B/alpaca/x0.25',repeat=1,output=tmp_path/'out',index=0,predecessor='last-job')
    assert template==original and derived['inputs']==point['inputs'] and derived['trace']==point['trace']
    assert derived['engine_identity']==point['engine_identity'] and derived['slo']==point['slo']
    assert 'independent.mixed.runtime' in new_job['payload']['argv'] and 'VLLM_USE_V1=0' in new_job['payload']['argv']
    assert new_job['payload']['image_digest']=='sha256:mixed-image' and new_job['max_attempts']==1
    assert new_job['payload']['after_terminal']==['last-job'] and plan['repeat_id']==1
    assert new_group['points']==[derived] and new_job['payload']['required_receipts']==['session/completion.json']


def test_clone_refuses_different_engine_source_or_image(script,tmp_path,monkeypatch):
    # A source mismatch is rejected before any immutable point/group is written.
    point=dict(name='case',revision='one',source_manifest={'path':'m','sha256':'m'},engine_identity={'image_digest':'image'})
    template=dict(point=point,group={'engine_identity':point['engine_identity']},job={'payload':{
        'source_sha256':'different','image_digest':'image','argv':[]}})
    monkeypatch.setattr(script,'verify_source',lambda _:None)
    monkeypatch.setattr(script,'verify_executed_template',lambda *a:{'path':'claim','sha256':'c'})
    with pytest.raises(ValueError,match='independent engine'):
        script.clone_observation(template,{'point_id':'case','revision':'one'},design_ref={'sha256':'d'},
            condition_id='7B/alpaca/x0.25',repeat=1,output=tmp_path,index=0,predecessor=None)
    assert list(tmp_path.iterdir())==[]


@pytest.fixture
def prepared(script,snapshot,tmp_path,monkeypatch):
    sources={};originals={};canonical={};templates=[]
    for system in ('pdblend',*BASELINE_SYSTEMS):
        text='system = '+repr(system)+'\n'
        import hashlib
        files={'independent.py':hashlib.sha256(text.encode()).hexdigest()};revision=script.digest(files)
        source=tmp_path/'sources'/revision;source.mkdir(parents=True);(source/'independent.py').write_text(text)
        write(source/'manifest.json',dict(source_sha256=revision,files=files));sources[system]=script.binding(source/'manifest.json')
    for index,value in enumerate(snapshot):
        system=value['system'];value['revision']=Path(sources[system]['path']).parent.name
        receipt=tmp_path/'old'/str(index)/'receipt.json';write(receipt,{'original':index})
        value.update(receipt_path=str(receipt),receipt_sha256=script.binding(receipt)['sha256'])
        config=tmp_path/'configs'/(str(index)+'.json');write(config,{'system':system,'frozen_startup':index})
        trace=tmp_path/'traces'/('-'.join(map(str,(value['model'],value['dataset'],value['rate_scale'])))+'.json')
        if not trace.exists():write(trace,{'seed':701,'requests':['immutable']})
        value['comparison_identity']['trace_sha256']=script.binding(trace)['sha256']
        point=dict(name=value['point_id'],revision=value['revision'],system=system,source_manifest=sources[system],
            engine_identity={'image_digest':'image'},inputs={'system_config':script.binding(config)},trace=script.binding(trace),
            seed=701,duration_s=150,slo={'ttft_s':5.,'tpot_s':.15},model_id='Qwen2.5-'+value['model']+'-Instruct')
        originals[point['name']]=point;canonical[str(receipt)]=(deepcopy(value),point,{'raw_checked':True})
        group=dict(session_id='old-session-'+str(index),engine_identity=point['engine_identity'],points=[point])
        group_path=tmp_path/'old-groups'/(str(index)+'.json');write(group_path,group)
        payload=dict(source_sha256=value['revision'],image_digest='image',session_id=group['session_id'],gpu_count=8,
            exclusive=True,reserve_host=True,required_receipts=['session/completion.json'],
            argv=['docker','run','--name','old-'+str(index),'-e','PDBLEND_SOURCE_MANIFEST='+sources[system]['path'],
                'image','-m','independent.'+system+'.runtime','--group',str(group_path),'--out','{attempt_dir}/session'])
        templates.append(dict(job_id='old-'+str(index),priority=1,max_attempts=1,payload=payload))
        write(receipt.parent/'manifest.json',dict(schema=1,immutable=True,job_id='old-'+str(index),payload=payload))
    points_path=tmp_path/'initial.json';jobs_path=tmp_path/'templates.json';write(points_path,snapshot);write(jobs_path,templates)
    monkeypatch.setattr(script,'canonical_receipt',lambda path:deepcopy(canonical[str(Path(path))]))
    out=tmp_path/'repeats';result=script.prepare(points_path,[jobs_path],out,
        candidate_revision=Path(sources['pdblend']['path']).parent.name,after_terminal='reviewed-matrix-last')
    return out,result,originals


def test_preparation_predeclares_15_serial_immutable_real_template_jobs(script,prepared):
    out,result,originals=prepared;protocol=script.load_bound(script.binding(out/'protocol.json'))
    jobs=script.load_bound(script.binding(out/'jobs.json'))
    assert result==dict(conditions=36,repeated_conditions=1,jobs=15,enqueued=False)
    assert jobs[0]['payload']['after_terminal']==['reviewed-matrix-last']
    assert all(jobs[i]['payload']['after_terminal']==[jobs[i-1]['job_id']] for i in range(1,15))
    assert len({job['job_id'] for job in jobs})==15 and all(job['max_attempts']==1 for job in jobs)
    assert [row['system'] for row in protocol['planned_observations'][:5]]==['pdblend',*BASELINE_SYSTEMS]
    for plan,job in zip(protocol['planned_observations'],jobs):
        point=script.load_bound(plan['point']);original=originals[point['name'].split('-paired-r')[0]]
        assert point['inputs']==original['inputs'] and point['source_manifest']==original['source_manifest']
        assert point['slo']==original['slo'] and point['trace']==original['trace']
        assert 'independent.'+point['system']+'.runtime' in job['payload']['argv']
    assert not (out/'acceptance.json').exists()


def test_opt_in_driver_serialization_is_part_of_immutable_design(script,prepared,tmp_path):
    package,_,_=prepared;protocol=script.load_bound(script.binding(package/'protocol.json'))
    out=tmp_path/'serialized'
    script.prepare(tmp_path/'initial.json',[tmp_path/'templates.json'],out,
        candidate_revision=protocol['candidate_revision'],after_terminal='parent-last',driver_serialized=True)
    jobs=script.load_bound(script.binding(out/'jobs.json'));design=script.load_bound(script.binding(out/'design.json'))
    assert design['queue_order']=='one_job_after_prior_released_lease_including_blocked'
    assert jobs[0]['payload']['after_terminal']==['parent-last']
    assert all(not job['payload']['after_terminal'] for job in jobs[1:])
    assert [job['payload']['predeclared_driver_sequence'] for job in jobs]==list(range(15))


@pytest.mark.parametrize('extra_attempt',[False,True])
def test_actual_queue_first_attempt_is_required_and_retry_cannot_replace_it(script,prepared,tmp_path,monkeypatch,extra_attempt):
    package,_,_=prepared;protocol=script.load_bound(script.binding(package/'protocol.json'))
    jobs=script.load_bound(script.binding(package/'jobs.json'));queue={'jobs':{},'leases':{}};canonical={}
    for index,(plan,spec) in enumerate(zip(protocol['planned_observations'],jobs)):
        point=script.load_bound(plan['point']);case=next(c for c in protocol['conditions'] if c['condition_id']==plan['condition_id'])
        value=row(plan['system'],tuple(case['condition']),98 if plan['system']=='pdblend' else 100)
        value.update(point_id=plan['point_id'],revision=plan['revision'],repeat_id=plan['repeat_id'],
                     comparison_identity=plan['comparison_identity'],service_start_s=protocol['declared_at_s']+index+100)
        attempt=tmp_path/'attempts'/spec['job_id'];session=attempt/'session';receipt=session/'windows'/plan['point_id']/'receipt.json'
        write(receipt,{'real_attempt':index});value.update(receipt_path=str(receipt),receipt_sha256=script.binding(receipt)['sha256'])
        canonical[str(receipt)]=(value,point,{'bound_raw':True})
        write(attempt/'manifest.json',dict(job_id=spec['job_id'],attempt=1,payload=spec['payload'],gpu_uuids=value['comparison_identity']['gpu_uuids']))
        write(session/'completion.json',dict(status='passed',complete=True,cleanup={'passed':True,'process_cleanup_verified':True}))
        queue['jobs'][spec['job_id']]=dict(spec,status='succeeded',lease_id=None)
        queue['leases'][str(index)]=dict(job_id=spec['job_id'],attempt=1,status='succeeded',attempt_dir=str(attempt))
    if extra_attempt:
        queue['leases']['retry']=dict(queue['leases']['0'],attempt=2)
    queue_path=tmp_path/'queue.json';write(queue_path,queue);before=queue_path.read_bytes()
    monkeypatch.setattr(script,'canonical_receipt',lambda path:deepcopy(canonical[str(Path(path))]))
    out=tmp_path/'acceptance';script.evaluate(package,queue_path,out)
    result=json.loads((out/'acceptance.json').read_text())
    assert result['conditions'][0]['status']==('incomplete' if extra_attempt else 'stable_observed_win')
    assert result['all_36_observed_criteria_met'] is (not extra_attempt)
    assert result['goal_complete'] is False and queue_path.read_bytes()==before
    assert script.load_bound(result['queue_snapshot'])==queue


@pytest.mark.parametrize('mutation',['env','module','mount','image','source'])
def test_template_must_match_actual_claimed_command_before_cloning(script,tmp_path,mutation):
    payload=dict(argv=['docker','run','-v','/original-source:/opt/pdblend-src:ro','-e','VLLM_USE_V1=1',
        'sha256:image','-m','original.mixed.runtime','--group','group.json'],image_digest='sha256:image',source_sha256='original',after_terminal=['old'])
    claim=tmp_path/'attempt/manifest.json';write(claim,dict(schema=1,immutable=True,job_id='original-job',payload=payload))
    template={'job':{'job_id':'original-job','payload':deepcopy(payload)}}
    reference={'receipt_path':str(claim.parent/'session/windows/case/receipt.json')}
    assert script.verify_executed_template(template,reference)==script.binding(claim)
    # A new scheduling dependency is allowed, an execution change is not.
    template['job']['payload']['after_terminal']=['new']
    assert script.verify_executed_template(template,reference)==script.binding(claim)
    changed=template['job']['payload']
    if mutation=='env':changed['argv'][changed['argv'].index('VLLM_USE_V1=1')]='VLLM_USE_V1=0'
    elif mutation=='module':changed['argv'][changed['argv'].index('original.mixed.runtime')]='pdblend.bench.run'
    elif mutation=='mount':changed['argv'][changed['argv'].index('/original-source:/opt/pdblend-src:ro')]='/other-source:/opt/pdblend-src:ro'
    elif mutation=='image':changed['image_digest']='sha256:other'
    else:changed['source_sha256']='other'
    with pytest.raises(ValueError,match='actual original execution'):
        script.verify_executed_template(template,reference)
