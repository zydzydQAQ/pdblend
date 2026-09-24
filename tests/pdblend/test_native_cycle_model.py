"""CPU fit/scope tests; physical raw audit is tested independently on TP1/TP2."""
from copy import deepcopy
import json
from pathlib import Path
import time
from types import SimpleNamespace

import pytest

from pdblend.profile.query import native_cycle_model as model
from pdblend.profile.collection.native_timing_plan import binding,digest
from pdblend.profile.collection.native_runtime_collect import write_new
from test_native_serving_cycles import plan_fixture


def phase(tmp_path,plan,purpose,monkeypatch,*,candidate=None,training=None,multiplier=1.):
    from pdblend.profile.collection.native_serving_cycles import cycle_trace
    start=time.time()-10000 if purpose=='training' else time.time()+1
    identity=dict(model_id=plan['model_id'],tp=plan['tp'],pp=1,model_hash='m',tokenizer_hash='t',
        image_digest='image',source_revision='source',engine_revision='engine')
    monkeypatch.setattr(model,'audit_cycle_window',lambda raw,p:raw['fixture_independent_audit'])
    rows=[]
    for index,point in enumerate(p for p in plan['points'] if p['purpose']==purpose):
        trace=cycle_trace(plan,point);power=(800+10*trace['rate_rps'])*multiplier
        audit=dict(passed=True,identity=identity,measured=dict(service_mean_power_w=power,energy_service_j=60*power),
                   occupancy_by_replica={'fixture':dict(mean_running_requests=.1)})
        raw=dict(point=point,trace=trace,service_started_s=start+index*70,tail_end_s=start+index*70+61,
                 candidate=candidate,fixture_independent_audit=audit)
        ref=write_new(tmp_path/purpose/f'{index}.json',raw);rows.append(dict(raw=ref,audit=audit))
    return write_new(tmp_path/purpose/'completion.json',dict(schema='pdblend-native-request-cycle-collection/v1',
        phase=purpose,plan_sha256=digest(plan),collection_complete=True,safe_restore_passed=True,cleanup_errors=[],
        windows=rows,started_s=start-1,finished_s=start+8*70,candidate=candidate,training=training))


def test_whole_cycle_candidate_can_freeze_then_pass_independent_inner_rates(tmp_path,monkeypatch):
    plan=plan_fixture(tmp_path,monkeypatch,2);training=phase(tmp_path,plan,'training',monkeypatch)
    candidate=model.fit_cycle_candidate(training,plan,tmp_path/'candidate.json')
    holdout=phase(tmp_path,plan,'holdout',monkeypatch,candidate=candidate,training=training)
    result=model.replay_cycle_component(training,candidate,holdout,plan)
    assert result['component_qualified'] and result['mean_relative_error']<1e-12
    assert not result['formal_eligible'] and not result['planner_automatically_wired']
    raw=json.loads(Path(candidate['path']).read_text());curve=model.RequestCycleModel(raw)
    args=dict(rate_rps=.1,frequency_mhz=1500,arrival_family='paced',model_id=plan['model_id'],tp=2,
        dataset='alpaca',parent_trace=plan['points'][0]['parent_trace'],duration_s=60.,initial_roles={'M':4})
    assert curve.predict_cycle_power_w(**args)==pytest.approx(801.)
    assert not hasattr(curve,'decode_power_w') and not hasattr(curve,'prefill_energy_j')
    for changes in [dict(duration_s=150),dict(initial_roles={'M':3,'off':1}),dict(arrival_family='unknown'),
                    dict(rate_rps=.01),dict(frequency_mhz=2100),dict(dataset='longbench'),dict(tp=1)]:
        with pytest.raises(ValueError,match='missing_profile'):curve.predict_cycle_power_w(**(args|changes))


def test_real_holdout_prediction_failure_is_preserved_in_scoped_component(tmp_path,monkeypatch):
    plan=plan_fixture(tmp_path,monkeypatch);training=phase(tmp_path,plan,'training',monkeypatch)
    candidate=model.fit_cycle_candidate(training,plan,tmp_path/'candidate.json')
    holdout=phase(tmp_path,plan,'holdout',monkeypatch,candidate=candidate,training=training,multiplier=1.3)
    result=model.replay_cycle_component(training,candidate,holdout,plan)
    assert not result['component_qualified'] and result['mean_relative_error']>.1


@pytest.mark.parametrize('damage',['candidate','freeze','raw_sha','duplicate_window'])
def test_frozen_flags_cannot_override_candidate_training_mutation(tmp_path,monkeypatch,damage):
    plan=plan_fixture(tmp_path,monkeypatch);training=phase(tmp_path,plan,'training',monkeypatch)
    candidate=model.fit_cycle_candidate(training,plan,tmp_path/'candidate.json')
    holdout=phase(tmp_path,plan,'holdout',monkeypatch,candidate=candidate,training=training)
    if damage in ('candidate','freeze'):
        path=Path(candidate['path']);value=json.loads(path.read_text())
        if damage=='candidate':value['nodes'][0]['power_w'][0]+=1
        else:value['frozen_s']=1.
        path.write_text(json.dumps(value));candidate=binding(path)
    else:
        path=Path(training['path']);value=json.loads(path.read_text())
        if damage=='duplicate_window':
            value['windows'][1]=value['windows'][0];path.write_text(json.dumps(value));training=binding(path)
        else:
            raw=Path(value['windows'][0]['raw']['path']);raw.write_text(raw.read_text()+' ')
    with pytest.raises(ValueError):model.replay_cycle_component(training,candidate,holdout,plan)


@pytest.mark.parametrize('cause',['success','fit_domain','holdout_error','operational','unsafe_restore',
                                  'training_frequency','holdout_frequency'])
def test_same_fleet_driver_continues_only_after_safe_nonoperational_model_gap(tmp_path,monkeypatch,cause):
    import asyncio
    from pdblend.profile.collection import native_serving_cycles as collector
    plan=plan_fixture(tmp_path,monkeypatch);calls=[]
    async def collect(specs,fleet,meter,sampler,out,**kwargs):
        phase=kwargs.get('phase','training');calls.append(phase)
        value=dict(collection_complete=cause!='operational',safe_restore_passed=cause!='unsafe_restore',
                   ready_for_timing=cause not in ('operational','unsafe_restore'))
        if cause==phase+'_frequency':value['qualification_gaps']=[dict(kind='observed_frequency_mismatch')]
        write_new(Path(out)/'completion.json',value);return value
    def fit(training,p,out):
        calls.append('fit')
        if cause=='fit_domain':raise ValueError('cycle endpoint mean power decreased: unsupported')
        return write_new(out,dict(candidate='synthetic'))
    monkeypatch.setattr(collector,'collect_serving_cycles',collect)
    monkeypatch.setattr(model,'fit_cycle_candidate',fit)
    monkeypatch.setattr(model,'replay_cycle_component',lambda *args:dict(component_qualified=cause!='holdout_error'))
    result=asyncio.run(collector.collect_and_fit_serving_cycles([],None,None,None,tmp_path/'driver',gpu_uuids=[],plan=plan))
    assert result['ready_for_timing']==(cause in ('success','fit_domain','holdout_error','training_frequency','holdout_frequency'))
    assert not result['formal_eligible']
    if cause in ('success','holdout_error'):assert calls==['training','fit','holdout']
    if cause=='fit_domain':assert calls==['training','fit']
    if cause=='training_frequency':
        assert calls==['training'] and result['holdout_not_collected'] and 'candidate' not in result
    if cause.endswith('_frequency'):
        assert result['status']=='frequency_qualification_failed' and not result['operational_failure']
        assert not result['component_qualified'] and 'component' not in result
