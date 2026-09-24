"""Exporter attestation contract tests; these never claim a GPU raw replay."""
import csv
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest
from pdblend.bench import comparison_campaign as campaign

SCOPE='pdblend_profile_unqualified_evaluation/v1'
RAW=('trace','outcomes','power','native_result','canonical_requests','metering','startup_qualification',
     'reset','drain','controller','routes','native_cleanup','transition_measurements','frequencies')
GATES=({'raw.'+name for name in RAW}|{'binding.'+name for name in
        ('trace','native_result','startup_qualification','reset','metering','drain')}|{
    'pdblend.observation_inputs','pdblend.inventory','pdblend.full_physical_inventory','pdblend.actual_window',
    'pdblend.startup','pdblend.reset','pdblend.inventory_restoration','pdblend.controller_actions',
    'pdblend.physical_clocks','pdblend.request_routes','pdblend.published_route_roles',
    'pdblend.native_release_and_off','pdblend.canonical_metrics','metering.raw_eight_gpu_window'})


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,sort_keys=True))
    return dict(path=str(path.resolve()),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def load(path):return json.loads(Path(path).read_bytes())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def window_fixture(root,*,slo_pass=True):
    session=root/'session';window=session/'windows/pd'
    trace=write(root/'trace.json',{'evaluation':'fixed'})
    startup=write(session/'qualification.json',{'native_startup':'audited'})
    point=dict(name='pd',model_id='Qwen2.5-32B-Instruct',dataset='alpaca',system='pdblend',scale=.5,
        rate_rps=1.,seed=701,duration_s=150.,revision='legacy-observation-v1',status='prepared',blockers=[],
        trace=trace,observation_scope=SCOPE,measurement_protocol_version=campaign.PROTOCOL,slo={'ttft_s':1.,'tpot_s':.1})
    metrics=dict(duration_s=150.,slo_pass=slo_pass,offered_requests=120,successful_requests=100,failed_requests=20,
        energy_service_j=12345.67,energy_tail_j=890.,goodput_request_s=.5,goodput_token_s=12.,
        cohort_goodput_request_s=.6,ttft_p50_s=.2,ttft_p90_s=.4,ttft_p95_s=.6,ttft_p99_s=2.,
        tpot_p50_s=.01,tpot_p90_s=.03,tpot_p95_s=.1,tpot_p99_s=.2,gpu_util_mean_pct=77.)
    names=dict(outcomes='outcomes.json',power='power.json',native_result='native-result.json',
        canonical_requests='comparison-requests.json',metering='comparison-metering.json',drain='native-drain.json',
        controller='controller.jsonl',routes='routes.jsonl',native_cleanup='native-cleanup.json',
        transition_measurements='transition-measurements.json',frequencies='freq.jsonl')
    refs={name:write(window/'run'/filename,{'attestation_fixture':name}) for name,filename in names.items()}
    refs['metering']=write(window/'run/comparison-metering.json',dict(gpu_uuids=['GPU-'+str(i) for i in range(8)]))
    refs.update(trace=trace,startup_qualification=startup,reset=write(window/'reset.json',{'passed':True}))
    audit=dict(schema='pdblend-observation-acceptance/v1',scope=SCOPE,point_sha256=campaign.digest(point),
        metrics_sha256=campaign.digest(metrics),measurement_evidence_valid=True,profile_qualified=False,
        evidence_valid=False,formal_eligible=False,missing_gates=[],gate_failures={},checked_gates=sorted(GATES),
        profile_missing_gates=['native_timing_holdout','full_profile'],raw_refs=refs,evidence_sha256=campaign.digest(refs))
    result=dict(identity={},metrics=metrics,qualification=startup,observation_scope=SCOPE,
        measurement_evidence_valid=True,profile_qualified=False,evidence_valid=False,formal_eligible=False,
        observation_acceptance=audit)
    write(window/'point.json',point);write(window/'result.json',result);write(window/'run/observation-acceptance.json',audit)
    receipt=dict(point='pd',point_sha256=campaign.digest(point),result=result,session_id='session',engine_signature='engine',
        evidence_valid=False,cleanup_passed=True,baseline_frozen=False,artifacts={})
    bind_receipt(window,receipt)
    source=root/'campaign.json';write(source,dict(campaign_id='observed',points=[point]))
    return dict(session=session,window=window,campaign=source,output=root/'compare.csv',point=point)


def bind_receipt(window,receipt):
    receipt['artifacts']={str(path.relative_to(window)):sha(path) for path in window.rglob('*')
        if path.is_file() and path.name!='receipt.json'}
    write(window/'receipt.json',receipt)


def mutate(fixture,*,audit=None,result=None,receipt=None):
    window=fixture['window'];r=load(window/'receipt.json');value=r['result']
    if audit:
        proof=value['observation_acceptance'];audit(proof)
        write(window/'run/observation-acceptance.json',proof)
    if result:result(value)
    write(window/'result.json',value)
    if receipt:receipt(r)
    bind_receipt(window,r)


def export(fixture):
    summary=campaign.export(fixture['campaign'],fixture['output'],session_roots=[fixture['session']])
    return summary,list(csv.DictReader(fixture['output'].open()))


@pytest.mark.parametrize('slo_pass',[True,False])
def test_measured_unqualified_keeps_every_metric_and_never_ranks(tmp_path,slo_pass):
    f=window_fixture(tmp_path,slo_pass=slo_pass)
    summary,rows=export(f);row,=rows
    assert summary==dict(rows=1,measured=0,rank_eligible=0,measured_unqualified=1)
    assert row['status']=='measured_unqualified' and row['revision']=='legacy-observation-v1'
    for key,value in load(f['window']/'result.json')['metrics'].items():assert row[key]==str(value)
    assert all(row[key]=='False' for key in ('formal_eligible','evidence_valid','baseline_frozen','rank_eligible','observed_rank_eligible'))
    assert row['energy_rank']==row['observed_energy_rank']==''
    assert row['profile_qualified']=='False' and row['measurement_evidence_valid']=='True'
    assert row['observation_acceptance_sha256']==sha(f['window']/'run/observation-acceptance.json')


@pytest.mark.parametrize('edit,match',[
    (lambda x:x.update(checked_gates=[g for g in x['checked_gates'] if g!='pdblend.request_routes']),'required measurement'),
    (lambda x:x.update(missing_gates=['metering.raw_eight_gpu_window'],gate_failures={'metering.raw_eight_gpu_window':'gap'}),'verdict'),
    (lambda x:x.update(formal_eligible=True),'verdict'),
    (lambda x:x.update(metrics_sha256='a'*64),'metric binding'),
    (lambda x:x.update(point_sha256='b'*64),'point or metric'),
    (lambda x:x.update(evidence_sha256='c'*64),'evidence digest'),
    (lambda x:x.update(profile_missing_gates=[]),'qualification gaps'),
])
def test_rebound_receipt_still_rejects_missing_or_conflicting_proof(tmp_path,edit,match):
    f=window_fixture(tmp_path);mutate(f,audit=edit)
    with pytest.raises(ValueError,match=match):export(f)


def test_boolean_only_cannot_claim_observation(tmp_path):
    f=window_fixture(tmp_path);path=f['window']/'run/observation-acceptance.json';path.unlink()
    receipt=load(f['window']/'receipt.json');bind_receipt(f['window'],receipt)
    with pytest.raises(ValueError,match='hash-bound observation acceptance is missing'):export(f)


def test_metric_rewrite_without_acceptance_rebinding_rejected(tmp_path):
    f=window_fixture(tmp_path)
    mutate(f,result=lambda x:x['metrics'].update(energy_service_j=.001))
    with pytest.raises(ValueError,match='metric binding'):export(f)


def test_cross_window_raw_reference_rejected_even_when_hash_is_valid(tmp_path):
    f=window_fixture(tmp_path/'one');other=window_fixture(tmp_path/'two')
    foreign=load(other['window']/'run/observation-acceptance.json')['raw_refs']['outcomes']
    def edit(audit):
        audit['raw_refs']['outcomes']=foreign;audit['evidence_sha256']=campaign.digest(audit['raw_refs'])
    mutate(f,audit=edit)
    with pytest.raises(ValueError,match='another window'):export(f)


def test_data_invalid_window_retains_original_invalid_status_and_metrics(tmp_path):
    f=window_fixture(tmp_path)
    mutate(f,result=lambda x:x.update(measurement_evidence_valid=False))
    summary,rows=export(f)
    assert rows[0]['status']=='invalid_measurement' and rows[0]['energy_service_j']=='12345.67'
    assert 'measured_unqualified' not in summary


def test_frozen_baseline_fields_unchanged_and_tamper_cannot_replace_csv(tmp_path,monkeypatch):
    f=window_fixture(tmp_path)
    source=load(f['campaign'])
    baseline=dict(f['point'],name='mixed-frozen',system='mixed',revision='baseline')
    source['points'].append(baseline);write(f['campaign'],source)
    window=f['session']/'windows/mixed-frozen'
    result=dict(identity={},metrics={'slo_pass':False,'energy_service_j':54321.},evidence_valid=True,formal_eligible=True)
    write(window/'point.json',baseline);write(window/'result.json',result)
    receipt=dict(point=baseline['name'],point_sha256=campaign.digest(baseline),result=result,
        session_id='session',engine_signature='engine',evidence_valid=True,cleanup_passed=True,baseline_frozen=True)
    bind_receipt(window,receipt)
    # Default exporter has no new observation fields for the frozen baseline.
    saved_result=load(f['window']/'result.json')
    mutate(f,result=lambda x:x.pop('observation_scope'))
    _,original=export(f)
    original_baseline=next(row for row in original if row['system']=='mixed')
    mutate(f,result=lambda x:x.update(observation_scope=SCOPE))
    summary,rows=export(f)
    new_baseline=next(row for row in rows if row['system']=='mixed')
    assert all(new_baseline[k]==v for k,v in original_baseline.items())
    assert summary['measured']==1 and summary['measured_unqualified']==1
    assert load(f['window']/'result.json')==saved_result
    before=f['output'].read_bytes();replaced=[]
    (f['window']/'run/observation-acceptance.json').write_text('{}')
    monkeypatch.setattr(campaign.os,'replace',lambda *args:replaced.append(args))
    with pytest.raises(ValueError,match='window artifact changed'):export(f)
    assert f['output'].read_bytes()==before and replaced==[]


def test_failed_cleanup_remains_invalid_even_when_measurement_audit_passed(tmp_path):
    f=window_fixture(tmp_path)
    mutate(f,receipt=lambda x:x.update(cleanup_passed=False))
    summary,rows=export(f)
    assert rows[0]['status']=='invalid_measurement' and rows[0]['formal_eligible']=='False'
    assert rows[0]['energy_service_j']=='12345.67' and 'measured_unqualified' not in summary


def test_point_scope_required_and_in_process_artifacts_hashed_only_once(tmp_path,monkeypatch):
    f=window_fixture(tmp_path)
    reads=[];real=campaign.file_sha
    def tracked(path):
        reads.append(str(Path(path).resolve()));return real(path)
    monkeypatch.setattr(campaign,'file_sha',tracked)
    export(f)
    assert reads.count(str((f['window']/'run/outcomes.json').resolve()))==1
    assert reads.count(str((f['window']/'run/power.json').resolve()))==1
    # A caller cannot add the observation flag only to a result after the point
    # was frozen under a different execution contract.
    point=load(f['window']/'point.json');point.pop('observation_scope')
    write(f['window']/'point.json',point)
    source=load(f['campaign']);source['points'][0]=point;write(f['campaign'],source)
    mutate(f,audit=lambda x:x.update(point_sha256=campaign.digest(point)),
           receipt=lambda x:x.update(point_sha256=campaign.digest(point)))
    with pytest.raises(ValueError,match='system or measured duration'):export(f)
