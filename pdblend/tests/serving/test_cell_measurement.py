import json

import pytest
from ecopadg.serving.measurement import summarize_cell, save_raw, power_evidence
from ecopadg.measure.backends import INSTANT_POWER_SOURCE_ID


def fixture_rows():
    trace={'requests':[dict(prompt_len=128,output_len=2),dict(prompt_len=128,output_len=2)]}
    rows=[dict(request_id=str(i),success=1,arrival_s=10+i,finish_s=12+i,
               prompt_len=128,output_len=2,input_tokens=128,generated_tokens=2,
               token_count_source='server_usage',token_ids_verified=1,latency_s=2,ttft_s=1,tpot_s=1,
               token_itl_exact=1,token_itl_s='[1.0]',error='') for i in range(2)]
    return trace,rows


def instant_evidence(power):
    source=dict(mode='instant',source_id=INSTANT_POWER_SOURCE_ID,field_id=186,scope_id=0,unit='W')
    metadata=[dict(t_s=t,gpus=list(range(8)),mode=['instant']*8,source_id=[INSTANT_POWER_SOURCE_ID]*8,
        field_id=[186]*8,scope_id=[0]*8,value_type=[1]*8,return_code=[0]*8,
        nvml_timestamp_us=[int(t*1e6)]*8,nvml_latency_us=[10]*8,
        read_started_s=[t]*8,read_finished_s=[t]*8) for t,_ in power]
    return source,metadata


def test_explicit_instant_evidence_preserves_energy_and_writes_raw_provenance(tmp_path):
    trace,rows=fixture_rows()
    power=[(9,[100]*8),(14,[100]*8)];util=[(t,[50]*8) for t,_ in power]
    source,metadata=instant_evidence(power)
    summary=summarize_cell(trace,rows,power,util,(5,2),power_source=source,
        power_metadata=metadata,require_power_mode='instant')
    assert summary['validity']=='ok' and summary['power_source_verified']
    assert summary['energy_j']==2400
    save_raw(tmp_path,rows,power,util,power_source=source,power_metadata=metadata)
    assert json.loads((tmp_path/'power_source.json').read_text())==source
    assert [json.loads(row) for row in (tmp_path/'power_metadata.jsonl').read_text().splitlines()]==metadata


@pytest.mark.parametrize('defect',['average','missing','partial','field','stale','gpu_order','unaligned'])
def test_unverified_power_never_satisfies_instant_measurement(defect):
    trace,rows=fixture_rows()
    power=[(9,[100]*8),(14,[100]*8)];util=[(t,[50]*8) for t,_ in power]
    source,metadata=instant_evidence(power)
    if defect=='average': source['mode']='average'
    if defect=='missing': metadata=[]
    if defect=='partial': metadata[1]['nvml_timestamp_us'].pop()
    if defect=='field': metadata[0]['field_id'][2]=185
    if defect=='stale': metadata[0]['nvml_timestamp_us'][2]-=1_000_000
    if defect=='gpu_order': metadata[0]['gpus'].reverse()
    if defect=='unaligned':
        metadata[1]['read_started_s']=[9]*8;metadata[1]['read_finished_s']=[9]*8
        metadata[1]['nvml_timestamp_us']=[9_000_000]*8
    summary=summarize_cell(trace,rows,power,util,(5,2),power_source=source,
        power_metadata=metadata,require_power_mode='instant')
    assert summary['validity']=='invalid_power_source'
    assert not summary['power_source_verified'] and summary['power_source_errors']


def test_legacy_call_is_unverified_even_when_work_is_complete():
    trace,rows=fixture_rows()
    summary=summarize_cell(trace,rows,[(9,[100]*8),(14,[100]*8)],
        [(9,[50]*8),(14,[50]*8)],(5,2))
    assert summary['validity']=='ok'
    assert not summary['power_source_verified'] and summary['power_mode']=='unspecified'


def test_empty_power_cannot_have_verified_source():
    source,_=instant_evidence([])
    assert not power_evidence([],source,[])['power_source_verified']


def test_raw_power_survives_a_utilization_read_failure(tmp_path):
    _,rows=fixture_rows()
    save_raw(tmp_path,rows,[(9,[100]*8)],[])
    assert '100' in (tmp_path/'power.csv').read_text()


def test_whole_node_energy_and_hardware_util_have_the_same_real_window():
    trace,rows=fixture_rows()
    summary=summarize_cell(trace,rows,[(9,[100]*8),(14,[100]*8)],
                           [(9,[50]*8),(14,[50]*8)],(5,2))
    assert summary['validity']=='ok'
    assert summary['energy_j']==2400
    assert summary['gpu_util']==.5
    assert summary['req_throughput']==pytest.approx(2/3)
    assert summary['token_itl_count']==2


def test_in_run_reconfiguration_tail_is_charged_without_inflating_request_time():
    trace,rows=fixture_rows()
    summary=summarize_cell(trace,rows,[(9,[100]*8),(20,[100]*8)],
                          [(9,[50]*8),(20,[50]*8)],(5,2),reconfiguration_end_s=15)
    assert summary['energy_j']==4000
    assert summary['req_throughput']==pytest.approx(2/3)
    assert summary['client_completion_end_s']==13
    assert summary['reconfiguration_tail_s']==2
    assert summary['measurement_end_s']==15


def test_same_completion_count_with_different_input_is_invalid_work():
    trace,rows=fixture_rows()
    rows[0]['input_tokens']=127
    summary=summarize_cell(trace,rows,[(9,[100]*8),(14,[100]*8)],
                           [(9,[50]*8),(14,[50]*8)],(5,2))
    assert summary['validity']=='invalid_work'


@pytest.mark.parametrize('change',[{'request_id':'unexpected'},{'token_ids_verified':0}])
def test_same_counts_with_unverified_request_or_token_identity_are_invalid(change):
    trace,rows=fixture_rows();rows[0].update(change)
    summary=summarize_cell(trace,rows,[(9,[100]*8),(14,[100]*8)],
                           [(9,[50]*8),(14,[50]*8)],(5,2))
    assert summary['validity']=='invalid_work'


def test_failed_run_keeps_denominator_and_serializes_unknown_latency_as_null():
    trace,rows=fixture_rows()
    for r in rows:
        r.update(success=0,error='timeout',ttft_s=None,tpot_s=None,generated_tokens=0,
                 token_itl_exact=0,token_itl_s='[]')
    summary=summarize_cell(trace,rows,[(9,[100]*8),(14,[100]*8)],
                           [(9,[50]*8),(14,[50]*8)],(5,2))
    assert summary['n_expected']==2 and summary['slo_attainment']==0
    assert summary['validity']=='invalid_work'
    json.dumps(summary,allow_nan=False)


@pytest.mark.parametrize('defect',[None,'unknown_http','unknown_code','some_tokens','wrong_request','sampling','power'])
def test_explicit_zero_work_refusal_can_only_support_a_capacity_upper_bound(defect):
    trace,rows=fixture_rows()
    rows[0].update(success=0,error='HTTP 429: admission refused',http_status=429,
        admission_rejection='admission_deadline',generated_tokens=0,input_tokens=0,
        token_count_source='missing',token_ids_verified=0,ttft_s=None,tpot_s=None,
        token_itl_exact=0,token_itl_s='[]',n_text_chunks=0)
    if defect=='unknown_http':rows[0]['http_status']=500
    if defect=='unknown_code':rows[0]['admission_rejection']='engine_failure'
    if defect=='some_tokens':rows[0]['generated_tokens']=1
    if defect=='wrong_request':rows[0]['request_id']='different'
    power=[(9,[100]*8),(14,[100]*8)];util=[(t,[50]*8) for t,_ in power]
    source,metadata=instant_evidence(power)
    if defect=='power':source['mode']='average'
    summary=summarize_cell(trace,rows,power,util,(5,2),power_source=source,
        power_metadata=metadata,require_power_mode='instant',sampling_error='sensor failed' if defect=='sampling' else None)
    assert summary['validity']!='ok' and summary['completed']==1 and summary['n_expected']==2
    assert summary['slo_attainment']==.5
    assert summary['capacity_observation_valid'] is (defect is None)
    if defect is None:
        assert summary['admission_rejections']==1 and summary['generated_tokens']==2
