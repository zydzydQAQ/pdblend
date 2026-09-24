import csv
from copy import deepcopy
import json

import pytest

from pdblend.bench import comparison_campaign as campaign
from pdblend.bench.comparison_recorded import POLICY, analyze, usable_metrics


def metrics(**changes):
    value = dict(duration_s=150., service_start_s=1000., service_end_s=1150.,
        offered_requests=100, successful_requests=100, failed_requests=0, unresolved_requests=0,
        joint_slo_requests=95, joint_slo_rate=.95, success_rate=1., slo_pass=True,
        ttft_p99_s=.5, tpot_p99_s=.05, ttft_samples=100, tpot_samples=100,
        ttft_p99_low_sample=False, tpot_p99_low_sample=False,
        energy_service_j=100., energy_tail_j=10., goodput_request_s=.6, gpu_util_mean_pct=80.)
    value.update(changes)
    return value


def row(system, revision='v1', **changes):
    value = dict(campaign_id='test', point_id=system, model_id='Qwen2.5-7B-Instruct', dataset='alpaca',
        system=system, rate_scale=.5, offered_rps=1., seed=701, revision=revision, trace_sha256='t'*64,
        measurement_protocol_version=campaign.PROTOCOL, model_hash='m'*64, tokenizer_hash='t'*64,
        gpu_uuids=['GPU-'+str(i) for i in range(8)], image_digest='image',
        runtime_source_sha256='runtime', measurement_source_sha256='measure', source_sha256=system+revision,
        slo_ttft_s=1., slo_tpot_s=.1, status='invalid_measurement', evidence_valid=False,
        formal_eligible=False, baseline_frozen=False, failure_reason='clock',
        receipt_path='/'+system+'/'+revision, receipt_sha256=system+revision,
        common_clock_evidence='fail', common_clock_scope='observed_requested_active_clock/v1', **metrics())
    value.update(changes)
    return value


def run(rows):
    records={r['receipt_path']:(dict(metrics={k:v for k,v in r.items() if k in metrics()},
        evidence_valid=r['evidence_valid'],formal_eligible=r['formal_eligible'],
        acceptance=dict(missing_gates=['clock'],gate_failures={'clock':'2490MHz'})),
        dict(evidence_valid=r['evidence_valid'],cleanup_passed=True)) for r in rows}
    campaign.rank_rows(rows)
    return analyze(rows, records)


def test_failed_qualification_is_usable_without_changing_metrics_or_audit():
    rows=[row('mixed'),row('pdblend',energy_service_j=80.)]
    before=deepcopy(rows);run(rows)
    for old,new in zip(before,rows):
        assert all(new[key]==value for key,value in old.items() if key not in ('status',))
        assert new['status']=='measured' and new['measurement_usable']
        assert new['evidence_valid'] is False and new['formal_eligible'] is False
        assert new['qualification_gate_failures']=={'clock':'2490MHz'}
        assert new['strict_rank_eligible'] is False
        assert new['analysis_baseline_frozen']==(new['system']!='pdblend')
    assert rows[1]['energy_rank']==1 and rows[0]['energy_rank']==2
    assert rows[1]['pdblend_saving_vs_best_feasible_baseline']==pytest.approx(.2)
    assert rows[1]['comparison_complete_five_systems'] is False


def test_request_failure_keeps_data_but_cannot_rank_and_original_slo_is_not_rewritten():
    failed=row('pdblend',successful_requests=90,failed_requests=10,slo_pass=True,energy_service_j=1.)
    run([row('mixed'),failed])
    assert failed['measurement_usable'] and failed['slo_pass'] is True
    assert not failed['analysis_slo_pass'] and not failed['rank_eligible']
    assert failed['energy_rank']=='' and failed['energy_service_j']==1.


@pytest.mark.parametrize('changes',[dict(joint_slo_requests=89),dict(ttft_p99_s=1.01),
    dict(tpot_p99_s=.101),dict(unresolved_requests=1),dict(joint_slo_requests=101)])
def test_each_slo_condition_is_required(changes):
    candidate=row('pdblend',**changes);run([row('mixed'),candidate])
    assert candidate['measurement_usable'] and not candidate['analysis_slo_pass']


def test_all_failed_window_retains_missing_latencies():
    candidate=row('pdblend',successful_requests=0,failed_requests=100,joint_slo_requests=0,
        ttft_p99_s=None,tpot_p99_s=None,ttft_samples=0,tpot_samples=0)
    run([candidate])
    assert candidate['measurement_usable'] and not candidate['rank_eligible']
    assert candidate['ttft_p99_s'] is None


@pytest.mark.parametrize('change',[dict(duration_s=149.),dict(service_end_s=1149.),
    dict(energy_service_j=None),dict(energy_service_j=float('nan')),dict(offered_requests=0),
    dict(ttft_p99_s=None),dict(successful_requests=None)])
def test_missing_real_measurement_is_not_filled(change):
    assert not usable_metrics(metrics(**change))[0]


def test_revisions_separate_duplicates_not_best_picked_tail_and_low_n_kept():
    rows=[row('mixed'),row('ecoserve','a',energy_service_j=110.),row('ecoserve','b',energy_service_j=120.),
          row('pdblend','a',energy_service_j=80.,energy_tail_j=100.,ttft_p99_low_sample=True),
          row('pdblend','b',energy_service_j=150.)]
    run(rows)
    assert rows[3]['energy_rank']==1 and rows[4]['energy_rank']==4
    assert rows[0]['energy_rank']=='' and rows[0]['energy_rank_by_revision']=={'a':2,'b':1}
    assert rows[3]['tail_reverses_saving'] and rows[3]['ttft_p99_low_sample']
    assert rows[3]['comparison_variant_count']==4 and rows[3]['comparison_system_count']==3
    dup=deepcopy(rows[3]);dup.update(receipt_path='/pdblend/a/repeat',receipt_sha256='dup',energy_service_j=1.)
    rows=[row('mixed'),row('pdblend','a',energy_service_j=80.),dup,row('pdblend','b',energy_service_j=150.)]
    run(rows)
    assert all(not r['rank_eligible'] and r['comparison_status']=='ambiguous_attempts' for r in rows[1:3])
    assert rows[3]['energy_rank']==2


def test_duplicate_baseline_variant_excluded_but_different_revision_retained():
    duplicate=row('mixed',receipt_path='/mixed/duplicate',energy_service_j=1.)
    rows=[row('mixed'),duplicate,row('mixed','other',energy_service_j=120.),row('pdblend',energy_service_j=100.)]
    run(rows)
    assert rows[0]['comparison_status']==rows[1]['comparison_status']=='ambiguous_attempts'
    assert rows[3]['best_feasible_baseline_revision']=='other'
    assert rows[3]['comparison_variant_count']==2


@pytest.mark.parametrize('key,value',[('trace_sha256','other'),('model_hash','other'),
    ('tokenizer_hash','other'),('seed',702),('rate_scale',.25),('gpu_uuids',['GPU-other'])])
def test_unmatched_identities_never_get_pd_savings(key,value):
    rows=[row('mixed'),row('pdblend',**{key:value})];run(rows)
    assert rows[1]['comparison_baseline_receipts']=={}
    assert rows[1]['pdblend_saving_vs_best_feasible_baseline'] is None
    assert rows[1]['comparison_system_count']==1


def test_qualified_complete_cohort_parity_with_strict_ranking():
    systems=['mixed','distserve','ecoserve','dynamollm','pdblend']
    rows=[row(system,evidence_valid=True,formal_eligible=True,baseline_frozen=system!='pdblend',
        common_clock_evidence='pass',energy_service_j=100.-index*5) for index,system in enumerate(systems)]
    run(rows)
    for value in rows:
        assert value['energy_rank']==value['strict_energy_rank']
        assert value['rank_eligible']==value['strict_rank_eligible']
    assert rows[-1]['pdblend_saving_vs_best_feasible_baseline']==rows[-1]['strict_pdblend_saving_vs_best_feasible_baseline']


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,sort_keys=True))


def exported_fixture(tmp_path):
    points=[];session=tmp_path/'session'
    for system in ('mixed','ecoserve','pdblend'):
        point=dict(name=system,model_id='Qwen2.5-7B-Instruct',dataset='alpaca',system=system,scale=.5,
            rate_rps=1.,seed=701,duration_s=150.,revision='v1',status='prepared',blockers=[],
            trace=dict(path='/immutable/trace',sha256='t'*64),slo=dict(ttft_s=1.,tpot_s=.1))
        points.append(point);window=session/'windows'/system
        identity={k:row(system)[k] for k in ('model_hash','tokenizer_hash','gpu_uuids','image_digest')}
        valid=system=='mixed'
        result=dict(identity=identity,metrics=metrics(energy_service_j=80. if system=='pdblend' else 100.),
            evidence_valid=valid,formal_eligible=valid,measurement_evidence_valid=False,profile_qualified=False,
            acceptance=dict(missing_gates=[] if valid else ['clock','freshness'],gate_failures={} if valid else
                {'clock':'active clock 2460MHz','freshness':'native receipt unproven'}))
        if system=='pdblend':
            result['observation_scope']='pdblend_profile_unqualified_evaluation/v1'
        write(window/'point.json',point);write(window/'result.json',result)
        receipt=dict(point=system,point_sha256=campaign.digest(point),result=result,evidence_valid=valid,
            cleanup_passed=True,baseline_frozen=valid,session_id='s',engine_signature='sig',
            artifacts={name:campaign.file_sha(window/name) for name in ('point.json','result.json')})
        write(window/'receipt.json',receipt)
    points.append(dict(points[-1],name='unrun',status='blocked',blockers=['missing_profile']))
    write(tmp_path/'campaign.json',dict(campaign_id='test',points=points))
    return session,tmp_path/'campaign.json',tmp_path/'compare.csv'


def test_real_export_opt_in_preserves_baseline_values_hashes_and_default(tmp_path):
    session,source,out=exported_fixture(tmp_path)
    default=campaign.export(source,out,session_roots=[session]);before=list(csv.DictReader(out.open()))
    assert default['measured']==1
    summary=campaign.export(source,out,session_roots=[session],analysis_policy=POLICY)
    rows=list(csv.DictReader(out.open()));assert summary['measured']==3 and summary['qualification_valid']==1
    for old,new in zip(before,rows):
        assert all(new[k]==old[k] for k in metrics() if k in old)
        assert new['receipt_sha256']==old['receipt_sha256'] and new['evidence_valid']==old['evidence_valid']
        assert new['formal_eligible']==old['formal_eligible'] and new['baseline_frozen']==old['baseline_frozen']
    assert rows[2]['status']=='measured' and rows[2]['energy_rank']=='1'
    assert rows[2]['original_result_profile_qualified']=='False'
    assert rows[3]['status']=='blocked' and rows[3]['measurement_usable']=='False'
    assert rows[1]['original_status']=='invalid_measurement'
    assert json.loads(rows[1]['qualification_gate_failures'])['clock']=='active clock 2460MHz'
    # Missing observation qualification artifact is diagnostic, not a new
    # gate in this mode. The immutable recorded result itself is still required.
    saved=out.read_bytes();(session/'windows/pdblend/result.json').write_text('{}')
    with pytest.raises(ValueError,match='window artifact changed'):
        campaign.export(source,out,session_roots=[session],analysis_policy=POLICY)
    assert out.read_bytes()==saved


def test_unknown_policy_rejected_before_replacing_output(tmp_path):
    session,source,out=exported_fixture(tmp_path);out.write_text('original')
    with pytest.raises(ValueError,match='unknown comparison analysis'):
        campaign.export(source,out,session_roots=[session],analysis_policy='typo')
    assert out.read_text()=='original'


def test_bound_receipt_without_metrics_is_failed_not_promoted(tmp_path):
    session,source,out=exported_fixture(tmp_path)
    window=session/'windows/pdblend'
    receipt=json.loads((window/'receipt.json').read_text())
    receipt['result']['metrics']={}
    write(window/'result.json',receipt['result'])
    receipt['artifacts']['result.json']=campaign.file_sha(window/'result.json')
    write(window/'receipt.json',receipt)
    campaign.export(source,out,session_roots=[session],analysis_policy=POLICY)
    value=next(r for r in csv.DictReader(out.open()) if r['system']=='pdblend' and r['receipt_path'])
    assert value['status']=='failed' and value['measurement_usable']=='False'
    assert value['energy_service_j']=='' and value['ttft_p99_s']=='' and value['energy_rank']==''
