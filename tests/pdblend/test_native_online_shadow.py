from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import json

import pytest

from pdblend.planner.forecast import Forecast,Forecaster,InFlightWork
from pdblend.profile.collection.native_timing_plan import binding,digest
from pdblend.profile.query import native_online_shadow as shadow


def put(path,value):
    path.write_text(json.dumps(value));return binding(path)


def raw(phase='training'):
    point=dict(purpose=phase,dataset='alpaca',frequency_mhz=1500,duration_s=150.,arrival_family='poisson')
    trace=dict(schema='pdblend-native-layout-energy-trace/v1',model_id=shadow.MODEL,
        selection_split='calibration_'+phase,evaluation_used_for_selection=False,
        requests=[dict(req_id=0,arrival_s=1.,prompt=[1,2,3],max_tokens=4,source='alpaca'),
                  dict(req_id=1,arrival_s=20.,prompt=[5,6],max_tokens=2,source='alpaca')])
    launches=[dict(spec=dict(instance_id='i'+str(i),tp=2,pp=1,generation=4,pool_id='allM')) for i in range(4)]
    clients=[dict(req_id=0,request_id='r0',instance_id='i0',submitted_s=101.01,finished_s=130.,
        events=[dict(received_s=105.,token_ids=[8],token_index=1,finished=False),
                dict(received_s=125.,token_ids=[9,10,11],token_index=4,finished=True)],
        completion_tokens=4,terminal=True),
        dict(req_id=1,request_id='r1',instance_id='i1',submitted_s=120.01,finished_s=140.,
        events=[dict(received_s=121.,token_ids=[],token_index=0,finished=False),
                dict(received_s=135.,token_ids=[8,9],token_index=2,finished=True)],
        completion_tokens=2,terminal=True)]
    return dict(schema='pdblend-native-request-cycle-window/v1',system='pdblend',point=point,
        trace=trace,plan_sha256='plan',actual_launch=launches,client_requests=clients,
        service_started_s=100.,service_end_s=250.,routes=[
            dict(event='acquire',request_id='r0',instance_id='i0',at_s=101.),
            dict(event='acquire',request_id='r1',instance_id='i1',at_s=120.),
            dict(event='release',request_id='r0',instance_id='i0',at_s=130.1),
            dict(event='release',request_id='r1',instance_id='i1',at_s=140.1)],
        state_observations=[dict(instance_id='i'+str(i),received_s=t+.02,
            state=dict(native_at_s=t,running=[],all_queue=[])) for t in (99.5,109.5,119.5,124.5,129.5,134.5,139.5,149.5,229.5) for i in range(4)])


def prior(tmp_path):
    source=put(tmp_path/'prior-source.json',dict(evaluation_used=False,holdout_used=False,frozen_s=80.))
    fc=Forecast(.2,0.,3.,3.,4.,0,inputs=(3,),outputs=(4,),length_pairs=((3,4),))
    return dict(schema=shadow.PRIOR_SCHEMA,model_id=shadow.MODEL,dataset='alpaca',
        selection_split='calibration_training',evaluation_used=False,holdout_used=False,frozen_s=90.,
        inputs=[source],forecast=asdict(fc))


def test_exact_known_prior_matches_actual_forecaster_and_backlog(tmp_path):
    p=prior(tmp_path); result=shadow.replay_window(raw(),decision_offsets_s=[10.],prior=p)['queries'][0]
    f=Forecaster(initial=Forecast(**p['forecast']));f.arrive(3,101.,request_id='r0')
    f.set_backlog((InFlightWork('r0',3,3,0,4,'M','allM'),))
    expected=asdict(f.forecast(110.));backlog=expected.pop('backlog')
    assert result['forecast']==expected and result['backlog']==backlog
    assert result['backlog_summary']==dict(pending_prefill_tokens=0,remaining_decode_tokens=3,occupied_kv_tokens=4)
    assert not result['missing'] and not result['actual_action'] and result['energy_label'] is None


def test_missing_prior_masks_rate_output_pair_values_instead_of_cold_start():
    row=shadow.replay_window(raw(),decision_offsets_s=[10.])['queries'][0]
    assert row['forecast']['rate_rps'] is None and row['forecast']['trend_rps'] is None
    assert row['forecast']['output_mean'] is None and row['forecast']['length_pairs'] is None
    assert row['forecast']['input_mean']==3 and row['forecast']['recent_rate_rps']>0
    assert 'bootstrap_forecast_prior' in row['missing']
    assert row['native_states']['i0']['running']==[] and row['backlog'][0]['request_id']=='r0'


def test_future_results_events_and_queue_samples_do_not_leak_into_earlier_query(tmp_path):
    a=raw();b=deepcopy(a);p=prior(tmp_path)
    b['client_requests'][0]['completion_tokens']=9999
    b['client_requests'][0]['events'][1].update(token_ids=[111]*300,token_index=301)
    b['client_requests'][1]['error']='future timeout'
    b['trace']['requests'][1].update(prompt=[77]*700,max_tokens=400)
    b['state_observations'].append(dict(instance_id='i0',received_s=130.,state=dict(native_at_s=105.,running=['future'],all_queue=['future'])))
    assert shadow.replay_window(a,decision_offsets_s=[10.],prior=p)['queries']==shadow.replay_window(b,decision_offsets_s=[10.],prior=p)['queries']


def test_equal_timestamp_token_is_not_yet_observed_and_zero_token_event_is_not_first_token():
    value=shadow.replay_window(raw(),decision_offsets_s=[5.,22.])['queries']
    assert value[0]['backlog'][0]['waiting_prefill_tokens']==3
    later={w['request_id']:w for w in value[1]['backlog']}
    assert later['r1']['waiting_prefill_tokens']==2 and later['r1']['remaining_output_tokens']==2


def test_past_completion_not_requested_or_final_payload_count_enters_pairs(tmp_path):
    value=raw();p=prior(tmp_path)
    value['client_requests'][0]['completion_tokens']=9999
    result=shadow.replay_window(value,decision_offsets_s=[30.05],prior=p)['queries'][0]
    assert result['forecast']['inflight']==1 and len(result['backlog'])==2
    assert all(pair[1]!=9999 for pair in result['forecast']['length_pairs'])
    later=shadow.replay_window(value,decision_offsets_s=[31.],prior=p)['queries'][0]
    assert [w['request_id'] for w in later['backlog']]==['r1']


def test_late_and_stale_native_states_are_missing_not_forward_filled_zero():
    value=raw();value['state_observations']=[dict(instance_id='i0',received_s=115.,
        state=dict(native_at_s=109.5,running=['r0'],all_queue=['r0']))]
    row=shadow.replay_window(value,decision_offsets_s=[10.,20.])['queries']
    assert row[0]['native_states']['i0']['running'] is None
    assert 'no_received_state' in row[0]['native_states']['i0']['missing']
    assert 'native_state_stale' in row[1]['native_states']['i0']['missing']
    assert row[1]['native_states']['i0']['running'] is None


def test_error_release_without_native_cancel_ack_is_not_successful_retirement():
    value=raw();value['client_requests'][0]['error']='timeout'
    row=shadow.replay_window(value,decision_offsets_s=[35.])['queries'][0]
    assert row['backlog'] is None and row['backlog_summary'] is None
    assert 'native_cleanup_ack_history_missing:r0' in row['missing']
    assert 'r0' in {r['request_id'] for r in row['observed_owned_records']}


@pytest.mark.parametrize('damage',['token_ids','token_index'])
def test_missing_signals_stay_unknown(damage):
    value=raw()
    if damage=='token_ids':value['client_requests'][0]['events'][0].pop('token_ids')
    else:value['client_requests'][0]['events'][0]['token_index']=99
    row=shadow.replay_window(value,decision_offsets_s=[10.])['queries'][0]
    assert row['backlog'] is None and row['missing']
    assert row['forecast']['output_mean'] is None


def test_missing_future_finish_cannot_change_past_values_or_domain_availability(tmp_path):
    a=raw();b=deepcopy(a);b['client_requests'][0].pop('finished_s');p=prior(tmp_path)
    before=shadow.replay_window(a,decision_offsets_s=[10.,29.],prior=p)
    absent=shadow.replay_window(b,decision_offsets_s=[10.,29.],prior=p)
    assert before['queries']==absent['queries']
    assert 'missing_finish_notification:r0' in absent['offline_reconstruction_issues']
    later=shadow.replay_window(b,decision_offsets_s=[31.],prior=p)['queries'][0]
    assert later['backlog'] is None and 'native_cleanup_ack_history_missing:r0' in later['missing']


def test_offline_missing_journal_does_not_invent_a_past_causal_fault():
    value=raw();value['client_requests'][0].pop('events')
    result=shadow.replay_window(value,decision_offsets_s=[2.,31.])
    assert result['queries'][0]['backlog'][0]['waiting_prefill_tokens']==3
    assert 'missing_token_journal:r0' in result['offline_reconstruction_issues']
    assert result['queries'][1]['backlog'] is None


def test_unknown_prior_rate_influences_actual_EWMA_beyond_length_prior_window(tmp_path):
    p1=prior(tmp_path);p2=deepcopy(p1);p2['forecast']['rate_rps']=2.
    a=shadow.replay_window(raw(),decision_offsets_s=[0.,10.,130.],prior=p1)['queries']
    b=shadow.replay_window(raw(),decision_offsets_s=[0.,10.,130.],prior=p2)['queries']
    assert a[0]['forecast']['rate_rps']!=b[0]['forecast']['rate_rps']
    for x,y in zip(a[1:],b[1:]):
        assert x['forecast']['rate_rps']!=y['forecast']['rate_rps']
        assert x['forecast']['trend_rps']!=y['forecast']['trend_rps']
        assert x['forecast']['peak_rps']==y['forecast']['peak_rps']
        assert x['forecast']['completed_bins']==y['forecast']['completed_bins']


def test_prior_length_dependence_fades_only_after_actual_traffic_window(tmp_path):
    p1=prior(tmp_path);p2=deepcopy(p1)
    p2['forecast'].update(input_mean=900.,input_p95=900.,output_mean=100.,outputs=[100.],length_pairs=[[900,100]])
    a=shadow.replay_window(raw(),decision_offsets_s=[10.,50.,130.],prior=p1)['queries']
    b=shadow.replay_window(raw(),decision_offsets_s=[10.,50.,130.],prior=p2)['queries']
    assert a[0]['forecast']['input_mean']==b[0]['forecast']['input_mean']==3
    assert a[1]['forecast']['output_mean']!=b[1]['forecast']['output_mean']
    for field in ('output_mean','outputs','length_pairs'):
        assert a[2]['forecast'][field]==b[2]['forecast'][field]
    missing=shadow.replay_window(raw(),decision_offsets_s=[130.])['queries'][0]
    assert missing['forecast']['output_mean']==a[2]['forecast']['output_mean']
    assert missing['forecast']['rate_rps'] is None


def test_bin_clock_uses_known_bootstrap_presence_not_cold_start(tmp_path):
    value=raw();p=prior(tmp_path)
    for route in value['routes']:route['at_s']+=14.
    for client in value['client_requests']:
        client['finished_s']+=14.
        for event in client['events']:event['received_s']+=14.
    times=[0.,10.,70.,80.,130.]
    known=shadow.replay_window(value,decision_offsets_s=times,prior=p)['queries']
    unknown=shadow.replay_window(value,decision_offsets_s=times)['queries']
    for a,b in zip(known,unknown):
        for key in ('peak_rps','completed_bins','recent_rate_rps'):
            assert a['forecast'][key]==b['forecast'][key]
    assert unknown[-1]['forecast']['rate_rps'] is None


@pytest.mark.parametrize('damage',['late','holdout','source','fields'])
def test_prior_without_causal_training_provenance_rejected(tmp_path,damage):
    p=prior(tmp_path)
    if damage=='late':p['frozen_s']=101.
    elif damage=='holdout':p['holdout_used']=True
    elif damage=='source':p['inputs'][0]['sha256']='bad'
    else:p['forecast'].pop('outputs')
    with pytest.raises(ValueError):shadow.replay_window(raw(),prior=p)


def setup_ledger(tmp_path,monkeypatch):
    source={'source_sha256':'s'}
    monkeypatch.setattr(shadow,'_source',lambda reference:source)
    train,held=raw(),raw('holdout')
    plan=dict(schema='pdblend-native-layout-energy-plan/v1',model_id=shadow.MODEL,
        evaluation_used_for_selection=False,points=[train['point'],held['point']])
    for value in (train,held):
        value['plan_sha256']=digest(plan)
        value['capabilities']={str(i):dict(source_revision='s') for i in range(4)}
    return put(tmp_path/'plan.json',plan),put(tmp_path/'train.json',train),put(tmp_path/'held.json',held),{'path':'source','sha256':'s'}


def test_holdout_only_reports_seen_training_states_and_never_extends_or_rewrites(tmp_path,monkeypatch):
    plan,train,held,source=setup_ledger(tmp_path,monkeypatch)
    p=put(tmp_path/'prior.json',prior(tmp_path));kwargs=dict(source_manifest=source,prior_refs={'alpaca':p})
    frozen=shadow.build_ledger(plan,[train],out=tmp_path/'ledger.json',**kwargs)
    before=Path(frozen['path']).read_bytes()
    ref=shadow.build_ledger(plan,[held],out=tmp_path/'validation.json',phase='holdout',training_ledger=frozen,**kwargs)
    report=json.loads(Path(ref['path']).read_text())
    assert Path(frozen['path']).read_bytes()==before
    assert report['validation_only'] and not report['training_domain_modified']
    assert report['new_training_signatures_added']==0 and 'training_domain' not in report
    assert not report['formal_eligible'] and not report['model_selected']
    with pytest.raises(ValueError,match='raw split'):
        shadow.build_ledger(plan,[held],out=tmp_path/'bad.json',**kwargs)
    with pytest.raises(FileExistsError):shadow.build_ledger(plan,[train],out=tmp_path/'ledger.json',**kwargs)


def test_holdout_requires_same_observer_and_hash_bound_training(tmp_path,monkeypatch):
    plan,train,held,source=setup_ledger(tmp_path,monkeypatch)
    frozen=shadow.build_ledger(plan,[train],source_manifest=source,out=tmp_path/'ledger.json')
    with pytest.raises(ValueError,match='observer'):
        shadow.build_ledger(plan,[held],source_manifest=source,out=tmp_path/'validation.json',phase='holdout',training_ledger=frozen,period_s=5.)
    Path(frozen['path']).write_text('{}')
    with pytest.raises(ValueError,match='checksum'):
        shadow.build_ledger(plan,[held],source_manifest=source,out=tmp_path/'validation.json',phase='holdout',training_ledger=frozen)


def test_no_power_native_cuda_or_final_completion_metrics_are_read():
    value=raw();value['power']=object();value['samples']=object();value['metrics']=object()
    result=shadow.replay_window(value,decision_offsets_s=[10.])
    assert not result['energy_labels_created'] and not result['domain_qualified']


def test_unknown_bootstrap_never_becomes_a_qualified_exact_training_domain(tmp_path,monkeypatch):
    plan,train,_,source=setup_ledger(tmp_path,monkeypatch)
    ref=shadow.build_ledger(plan,[train],source_manifest=source,out=tmp_path/'ledger.json')
    report=json.loads(Path(ref['path']).read_text())
    assert report['training_domain']['exact_observed_state_signatures']==[]
    assert report['windows'][0]['queries'][1]['backlog']
    assert not report['domain_qualified']


def test_source_semantic_files_must_match_both_frozen_and_current(tmp_path):
    root=Path(shadow.__file__).parents[3];files={}
    for name in shadow.SEMANTIC_FILES:
        target=tmp_path/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((root/name).read_bytes())
        files[name]=binding(target)['sha256']
    ref=put(tmp_path/'manifest.json',dict(files=files,source_sha256=digest(files)))
    assert shadow._source(ref)['source_sha256']==digest(files)
    (tmp_path/shadow.SEMANTIC_FILES[0]).write_text('changed')
    with pytest.raises(ValueError,match='source differs'):shadow._source(ref)
