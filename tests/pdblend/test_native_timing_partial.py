"""Terminal inventories preserve valid prefixes and never trigger a retry."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from pdblend.profile.collection.native_timing_partial import inventory, capture_inventory
from pdblend.profile.collection.native_timing_plan import read_bound, digest
from test_native_timing_single_pass import development
from test_native_timing_capacity import plans
from test_native_timing_replay import collected, put
from test_native_timing_replay_v2 import rebind_completion


def test_complete_development_inventory_is_not_a_qualification_or_retry(development):
    x=development;result=inventory(x.attempt,x.queue)
    assert result['counts']==dict(measured=len(x.plan['points']),unsupported_capacity=0,failed=0,invalid=0,missing=0)
    assert all(row['skip_remeasurement'] for row in result['points'])
    assert not result['automatic_retry_allowed'] and not result['full_job_retry_allowed']
    assert not result['component_qualified'] and not result['hardware_executed']


def test_partial_failure_keeps_prefix_and_distinguishes_failed_invalid_missing(development):
    x=development;report=deepcopy(x.report)
    refs=report['raw_bindings'];missing,failed,invalid=refs[-3:]
    Path(missing['path']).unlink()
    raw=json.loads(Path(failed['path']).read_text());raw.update(status='failed',error='clock acknowledgement failed')
    put(Path(failed['path']),raw)
    raw=json.loads(Path(invalid['path']).read_text());raw['frequency_samples']=[[t,[1400]] for t,_ in raw['frequency_samples']]
    invalid_ref=put(Path(invalid['path']),raw)
    report['raw_bindings']=refs[:-3]+[invalid_ref]
    report['window_owners']=report['window_owners'][:-3]+[dict(report['window_owners'][-1],raw=invalid_ref)]
    report.update(status='failed',complete=False,error='ValueError: observed frequency coverage differs')
    rebind_completion(x,report)
    queue=json.loads(x.queue.read_text());queue['jobs']['timing-job']['status']='failed';put(x.queue,queue)
    execution=json.loads((x.attempt/'execution.json').read_text())
    execution.update(status='failed',complete=False,returncode=2,error='RuntimeError: process exited 2')
    put(x.attempt/'execution.json',execution)
    result=inventory(x.attempt,x.queue)
    assert result['counts']==dict(measured=len(x.plan['points'])-3,unsupported_capacity=0,failed=1,invalid=1,missing=1)
    assert result['points'][-2]['raw_owner_identity_verified']
    assert 'frequency' in result['points'][-1]['error']
    assert all(row['skip_remeasurement'] for row in result['points'][:-3])
    assert not any(row['skip_remeasurement'] for row in result['points'][-3:])
    assert not result['full_job_retry_allowed']


@pytest.mark.parametrize('change',['running','owner','source','physical_owner'])
def test_terminal_owner_and_source_binding_cannot_be_bypassed(development,change):
    x=development
    if change=='running':
        queue=json.loads(x.queue.read_text());queue['jobs']['timing-job']['status']='running';put(x.queue,queue)
    elif change=='owner':
        report=deepcopy(x.report);report['window_owners'][0]['instance_id']='pd-timing-7';rebind_completion(x,report)
    elif change=='source':(x.tmp/'source/test.py').write_text('changed source')
    else:
        manifest=json.loads((x.attempt/'manifest.json').read_text());manifest['gpu_uuids'][-1]=manifest['gpu_uuids'][0]
        put(x.attempt/'manifest.json',manifest)
    with pytest.raises(ValueError):inventory(x.attempt,x.queue)


@pytest.fixture
def unbound_failure(development):
    x=development
    manifest=json.loads((x.attempt/'manifest.json').read_text());manifest['lease_id']='terminal-lease'
    put(x.attempt/'manifest.json',manifest)
    queue=json.loads(x.queue.read_text());queue['jobs']['timing-job']['status']='failed'
    queue['leases']={'terminal-lease':dict(lease_id='terminal-lease',job_id='timing-job',attempt=1,
        attempt_dir=str(x.attempt),status='failed',gpu_uuids=manifest['gpu_uuids'],claimed_at=899.)}
    put(x.queue,queue)
    report=deepcopy(x.report);report.update(status='failed',complete=False,error="ValueError('observed frequency coverage differs')")
    put(x.root/'completion.json',report)
    execution=json.loads((x.attempt/'execution.json').read_text())
    execution.update(status='failed',complete=False,returncode=2,error='RuntimeError: process exited 2',
                     receipt_sha256={},started_s=900.)
    put(x.attempt/'execution.json',execution)
    return x


def test_saved_failure_captures_new_snapshot_without_forging_worker_binding(unbound_failure):
    x=unbound_failure
    before=(x.attempt/'execution.json').read_bytes()
    ref=capture_inventory(x.attempt,x.queue,x.tmp/'terminal-capture')
    result=read_bound(ref)
    assert result['original_worker_completion_binding']==dict(present=False,sha256=None)
    assert result['completion_binding_scope']=='new_post_terminal_snapshot_not_original_worker_binding'
    assert result['terminal_lease_sha256']==digest(result['terminal_lease'])
    assert read_bound(result['queue_snapshot'])['jobs']['timing-job']==result['queue_job']
    assert result['counts']['measured']==len(x.plan['points'])
    assert (x.attempt/'execution.json').read_bytes()==before
    with pytest.raises(ValueError,match='immutable'):capture_inventory(x.attempt,x.queue,x.tmp/'terminal-capture')


@pytest.mark.parametrize('change',['wrong_hash','lease_status','lease_gpus','cleanup','exit','success'])
def test_post_terminal_capture_cannot_bypass_lease_worker_or_cleanup(unbound_failure,change):
    x=unbound_failure
    if change in ('lease_status','lease_gpus','success'):
        queue=json.loads(x.queue.read_text())
        if change=='lease_status':queue['leases']['terminal-lease']['status']='running'
        elif change=='lease_gpus':queue['leases']['terminal-lease']['gpu_uuids'].reverse()
        else:
            queue['jobs']['timing-job']['status']='succeeded';queue['leases']['terminal-lease']['status']='succeeded'
            execution=json.loads((x.attempt/'execution.json').read_text())
            execution.update(status='passed',complete=True,returncode=0,error=None)
            put(x.attempt/'execution.json',execution)
        put(x.queue,queue)
    elif change=='cleanup':
        report=json.loads((x.root/'completion.json').read_text())
        report['physical_cleanup']['observations'][-1]['devices'][0]['compute_pids']=[9001]
        put(x.root/'completion.json',report)
    else:
        execution=json.loads((x.attempt/'execution.json').read_text())
        if change=='wrong_hash':execution['receipt_sha256']={'native-timing/completion.json':'a'*64}
        else:execution['returncode']=137
        put(x.attempt/'execution.json',execution)
    with pytest.raises(ValueError):inventory(x.attempt,x.queue)
