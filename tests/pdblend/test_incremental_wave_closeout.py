import importlib.util
import json
from pathlib import Path

import pytest

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('incremental_closeout',ROOT/'scripts/2026-09-23_incremental_wave_closeout.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)


def job(member,state='running',created=1,**extra):
    return dict(status=state,created_at=created,lease_id=None,
                payload=dict(cohort_id=audit.COHORT,cohort_member=member),**extra)


def test_succeeded_still_owned_is_not_released():
    j=job(audit.MEMBERS[0],'succeeded')
    q=dict(jobs={'j':j},leases={'l':dict(job_id='j',status='active')})
    assert not audit.released(q,'j',j)
    q['leases']['l']['status']='succeeded'
    assert audit.released(q,'j',j)
    j['lease_id']='l'
    assert not audit.released(q,'j',j)


def test_running_complete_counts_never_qualify(tmp_path,monkeypatch):
    (tmp_path/'raw.json').write_text(json.dumps(dict(decode=[{}]*36)))
    (tmp_path/'completion.json').write_text(json.dumps(dict(complete=True,status='passed')))
    member=audit.MEMBERS[3]
    q=dict(jobs={'j':job(member)},leases={'l':dict(job_id='j',status='active',claimed_at=1,
            attempt_dir=str(tmp_path),gpu_uuids=['GPU-a'])})
    monkeypatch.setattr(audit,'audit_entry',lambda _:pytest.fail('must wait for actual terminal release'))
    result=audit.build(q,audit.COHORT)
    assert result['all_jobs_released'] is False
    assert result['members'][member]['progress']['primary_decode']==36
    assert result['members'][member]['measurement_complete'] is False
    assert result['members'][member]['qualification_status']=='pending_terminal'


def test_measurement_completion_and_calibration_failure_stay_separate(tmp_path,monkeypatch):
    member=audit.MEMBERS[0]
    q=dict(jobs={'j':job(member,'succeeded')},leases={'l':dict(job_id='j',status='succeeded',claimed_at=1,
            attempt_dir=str(tmp_path),gpu_uuids=['GPU-a'])})
    monkeypatch.setattr(audit,'audit_entry',lambda _:dict(measurement_complete=True,
        qualification_status='failed',calibration_components_passed=False,version_creation_ready=False))
    row=audit.build(q,audit.COHORT)['members'][member]
    assert row['measurement_complete'] is True and row['version_creation_ready'] is False


def test_superseded_job_is_excluded_and_partial_receipt_failure_visible(tmp_path,monkeypatch):
    member=audit.MEMBERS[0]
    q=dict(jobs={'old':job(member,'failed',2,superseded_by='new'),'new':job(member,'failed',3)},
        leases={'l':dict(job_id='new',status='failed',claimed_at=3,attempt_dir=str(tmp_path),gpu_uuids=[])})
    assert audit.choose(q,audit.COHORT)[member][0]=='new'
    def invalid(_):raise ValueError('raw checksum mismatch')
    monkeypatch.setattr(audit,'audit_entry',invalid)
    row=audit.build(q,audit.COHORT)['members'][member]
    assert row['qualification_status']=='audit_error'
    assert row['version_creation_ready'] is False


def test_final_report_never_overwrites_changed_evidence(tmp_path):
    path=tmp_path/'audit.json';audit.write(path,dict(passed=False),immutable=True)
    with pytest.raises(ValueError,match='immutable closeout'):
        audit.write(path,dict(passed=True),immutable=True)
