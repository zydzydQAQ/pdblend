import importlib.util
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('live_status',ROOT/'scripts/2026-09-23_live_status.py')
status=importlib.util.module_from_spec(spec);spec.loader.exec_module(status)


def job(state='queued',**payload):
    return dict(status=state,created_at=1,lease_id=None,payload=payload)


def test_terminal_dependency_accepts_failed_but_not_owned_lease():
    q=dict(jobs={'a':job('failed'),'b':job(after_terminal=['a'])},leases={})
    assert status.blocked_by(q,'b')==[]
    q['leases']['lease']=dict(job_id='a',status='active')
    assert status.blocked_by(q,'b')[0]['active_lease'] is True
    q['leases']['lease']['status']='failed'
    q['jobs']['b']['payload']=dict(depends_on=['a'])
    assert status.blocked_by(q,'b')[0]['required']=='succeeded'


def test_recursive_blocking_and_supersession_do_not_redirect_immutable_deps():
    q=dict(jobs={'old':dict(job('cancelled'),superseded_by='new'),'new':job(),
                 'a':job(depends_on=['old']),'b':job(depends_on=['a'])},leases={})
    assert status.blocked_by(q,'b')[0]['upstream'][0]['superseded_by']=='new'
    value=status.build(q,{},now=1)
    assert not any(x['job_id']=='old' for x in value['active_jobs'])
    assert value['superseded_jobs']==[dict(job_id='old',superseded_by='new')]


def test_retry_waiting_is_latest_functional_not_old_failure(tmp_path):
    model=status.MODELS[-1]
    old=tmp_path/'old';old.mkdir();(old/'completion.json').write_text(json.dumps(dict(status='inconclusive',complete=False,seed=701)))
    q=dict(jobs={'dynamo-functional-32b-old':job('failed',model_id=model,required_receipts=['completion.json']),
        'dynamo-reroute-retry-32b-new':dict(job(model_id=model),created_at=2)},
        leases={'old':dict(job_id='dynamo-functional-32b-old',claimed_at=1,attempt_dir=str(old),status='failed')})
    value=status.build(q,{},now=1)
    row=next(r for r in value['matrix'] if r['model_id']==model and r['system']=='dynamollm')
    assert row['functional']['job_id']=='dynamo-reroute-retry-32b-new'
    assert row['functional']['status']=='queued'
    assert row['mechanism']['status']=='missing'
    assert len(value['matrix'])==15 and value['seeds']==[701]
    assert all(r['formal_status']=='inconclusive' for r in value['matrix'])


def test_passed_receipt_is_preserved_separately_during_retry(tmp_path):
    model=status.MODELS[0]
    (tmp_path/'completion.json').write_text(json.dumps(dict(status='passed',complete=True,seed=701,formal_eligible=False)))
    q=dict(jobs={'dynamo-functional-7b-old':job('succeeded',model_id=model,required_receipts=['completion.json']),
        'dynamo-reroute-retry-7b-new':dict(job(model_id=model),created_at=2)},
        leases={'old':dict(job_id='dynamo-functional-7b-old',claimed_at=1,attempt_dir=str(tmp_path),status='succeeded')})
    row=next(r for r in status.build(q,{})['matrix'] if r['model_id']==model and r['system']=='dynamollm')
    assert row['functional']['status']=='queued'
    assert row['functional']['last_passed']['job_id']=='dynamo-functional-7b-old'
    assert row['formal_eligible'] is False


def test_partial_receipts_and_lease_uuid_overlap_fail_visible(tmp_path):
    p=tmp_path/'completion.json';p.write_text('{')
    assert status.receipt(p)['status']=='unreadable_partial_receipt'
    q=dict(jobs={},leases={'a':dict(status='active',gpu_uuids=['GPU-x']),
                          'b':dict(status='active',gpu_uuids=['GPU-x'])})
    value=status.build(q,{},now=1)
    assert value['gpu_lease_occupancy']['mutual_exclusion'] is False
    assert '+08:00' in status.markdown(value)
    assert '局部机制证据' in status.markdown(value)


def test_sampling_progress_keeps_training_and_independent_holdout_separate(tmp_path):
    (tmp_path/'raw.json').write_text(json.dumps({'decode': [{}, {}]}))
    scope = {'measurement_scope': {'kind': 'followup', 'expected_training_points': 18,
                                  'expected_holdout_points': 36}}
    result = status.sampling_progress(scope, tmp_path)
    assert result['measured_points'] == 2 and result['expected_points'] == 54
    assert result['phases'][0]['measured_points'] == 2
    assert result['phases'][1]['measured_points'] == 0
    assert all(not p['calibration_qualified'] for p in result['phases'])
    directory = tmp_path/'long-holdout'; directory.mkdir()
    (directory/'raw.json').write_text('{')
    assert status.sampling_progress(scope,tmp_path)['phases'][1]['raw_readable'] is False


def test_resident_short_progress_counts_only_complete_repeats(tmp_path):
    (tmp_path/'short').mkdir()
    (tmp_path/'short/raw.json').write_text(json.dumps(dict(training={
        'done': dict(point=dict(repeats=3), repeats=[{}, {}, {}]),
        'partial': dict(point=dict(repeats=3), repeats=[{}])}, holdout={})))
    scope = dict(measurement_scope=dict(kind='experimental_short_and_optional_qualified_long_resident',
        long_holdout_points=24, short_training_points=30, short_holdout_points=36))
    result = status.sampling_progress(scope, tmp_path)
    assert result['measured_points'] == 1 and result['expected_points'] == 90
    assert not any(p['calibration_qualified'] for p in result['phases'])
