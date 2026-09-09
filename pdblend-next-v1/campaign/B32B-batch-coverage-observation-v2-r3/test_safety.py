import asyncio
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

ROOT=Path(__file__).resolve().parent

def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
b=load('batch_v2_outer_test',ROOT/'run.py');child=load('batch_v2_child_test',ROOT/'child.py');old=load('batch_observation_spec_test',ROOT/'observation.frozen.py')

@pytest.fixture
def gate(tmp_path,monkeypatch):
 monkeypatch.setattr(b,'ROOT',tmp_path);g=b.Gate();yield g;g.log.close()

@pytest.mark.parametrize('route',['/control','/drain','/cancel','/v1/completions'])
def test_identity_failure_sends_no_write_or_dispatch_record(route):
 journal=io.StringIO();g=child.DispatchGuard(journal)
 with pytest.raises(RuntimeError,match='identity gate'):g.record('POST','http://127.0.0.1:33500'+route,{})
 assert journal.getvalue()==''
 g.record('GET','http://127.0.0.1:33500/runtime',{})


def test_guard_binds_target_and_durable_ownership():
 journal=io.StringIO();g=child.DispatchGuard(journal);g.identity_verified=True
 with pytest.raises(RuntimeError):g.record('POST','http://127.0.0.1:33501/v1/completions',{})
 with pytest.raises(RuntimeError):g.record('POST','http://127.0.0.1:33500/v1/completions',{})
 g.record('POST','http://127.0.0.1:33500/v1/completions',dict(headers={'X-Request-Id':'pdb-profile-owned'},json={'max_tokens':256}))
 assert journal.getvalue().endswith('\n') and json.loads(journal.getvalue())['request_id']=='pdb-profile-owned'


def test_full_outer_identity_failure_never_acquires_control(gate,monkeypatch):
 calls=[]
 async def fail(*args):raise ValueError('wrong full model identity')
 async def stop():calls.append('stop-child')
 async def http(*args,**kwargs):calls.append('http');raise AssertionError('no control allowed')
 monkeypatch.setattr(gate,'identity',fail);monkeypatch.setattr(gate,'stop_child',stop);monkeypatch.setattr(gate,'http',http)
 monkeypatch.setattr(b,'ClockOwner',lambda *a:pytest.fail('no clock acquisition before identity'))
 async def run():
  with pytest.raises(ValueError):await gate.work(None)
  await gate.cleanup()
 asyncio.run(run());assert not gate.verified and calls==['stop-child'] and gate.state['identity_failure_no_control']


def test_unknown_child_exit_is_fail_stop_before_native_or_clock(gate,monkeypatch):
 gate.verified=True;gate.child=SimpleNamespace(returncode=None)
 async def stop():raise TimeoutError('child exit not confirmed')
 async def restore(*args):pytest.fail('must not restore concurrently with child')
 monkeypatch.setattr(gate,'stop_child',stop);monkeypatch.setattr(gate,'restore_one',restore)
 monkeypatch.setattr(b,'ClockOwner',lambda *a:pytest.fail('must not reset concurrent child clocks'))
 asyncio.run(gate.cleanup());assert gate.state['cleanup_complete'] is False and gate.state['clock_release_skipped_child_may_be_alive']


def test_native_proof_failure_still_resumes_latest_generation(gate,monkeypatch):
 calls=[]
 async def idle(*a,**k):return dict(generation=5)
 async def http(*a,**k):return {} # Real proof is rejected.
 async def resume(port):calls.append(port);return dict(generation=7,accepting=True)
 monkeypatch.setattr(gate,'settled_idle',idle);monkeypatch.setattr(gate,'http',http);monkeypatch.setattr(gate,'resume',resume)
 with pytest.raises(RuntimeError):asyncio.run(gate.restore_one(33500))
 assert calls==[33500] and 'proof_error' in gate.state['cleanup']['33500'] and gate.state['cleanup']['33500']['restored']['accepting']


def test_both_replicas_finish_cleanup_before_outer_clock_reset(gate,monkeypatch):
 gate.verified=True;gate.child=SimpleNamespace(returncode=0);gate.hardware=object();done=[]
 async def stop():pass
 async def restore(port):
  if port==33501:await asyncio.sleep(.01)
  done.append(port)
  if port==33500:raise RuntimeError('retain first proof error')
 class Clock:
  def __init__(self,*a):assert set(done)=={33500,33501}
  async def close(self):done.append('clocks')
 monkeypatch.setattr(gate,'stop_child',stop);monkeypatch.setattr(gate,'restore_one',restore);monkeypatch.setattr(gate,'owned_ids',lambda:set())
 monkeypatch.setattr(b,'ClockOwner',Clock)
 asyncio.run(gate.cleanup());assert done[-1]=='clocks' and not gate.state['cleanup_complete'] and gate.state['clock_release_complete']


def test_ownership_parser_ignores_only_unsubmitted_partial_tail(gate):
 p=b.ROOT/'dispatch.jsonl';p.write_text(json.dumps(dict(route='/v1/completions',port=33500,request_id='pdb-profile-mine'))+'\n'+ '{"partial":')
 assert gate.owned_ids()=={(33500,'pdb-profile-mine')}
 p.write_text(json.dumps(dict(route='/v1/completions',port=33501,request_id='someone-else'))+'\n')
 with pytest.raises(RuntimeError):gate.owned_ids()


def test_secondary_event_error_is_captured_and_does_not_raise(gate,tmp_path,monkeypatch):
 monkeypatch.setattr(b,'OLD',tmp_path);gate.event_offsets={'nextv3b0':0,'nextv3b1':0}
 gate.capture_events();assert len(gate.state['event_capture_errors'])==2


def test_exact_natural_spec_stays_twelve_points_no_temporal():
 a=old.arguments();assert a.input_patterns==[[512]] and a.output_pattern==[256] and a.batches==[4,8]
 assert a.frequencies==[2520,1500] and a.budgets==[8192] and a.repeats==3 and a.target_gpus==[0,1]
 assert len(a.batches)*len(a.frequencies)*a.repeats==12 and a.arrival_offsets==[0.]


def test_tp2_requires_real_one_owner_and_two_transport_ranks():
 raw=dict(drain=dict(transfers=[dict(buffered_gpu_bytes=0)]*2),requests=[dict(request_id=str(i)) for i in range(4)],spec=dict(batch_size=4))
 state=dict(generation=1,acknowledged_generations=[1],scheduler_io=[{}]);raw.update(runtime_before=state,runtime_after_requests=state,runtime_drained=state)
 events=[dict(prefill=0,decode=4,request_ids=[str(i) for i in range(4)],started_s=1.,finished_s=2.)]*64
 assert old.validate_tp2_observation(raw,events)['full_batch_decode_steps']==64
 raw['runtime_drained']={**state,'scheduler_io':[{},{}]}
 with pytest.raises(ValueError):old.validate_tp2_observation(raw,events)


def native_proof():
 import time
 rank=dict(listener_alive=True,send_counters_observed=True,send_healthy=True,
  send_started=3,send_completed=3,send_failed=0,buffered_tensors=0,inflight_receives=0,
  inflight_sends=0,buffered_gpu_bytes=0,allocations={})
 return dict(drained=True,accepting=False,generation=6,drain_proof_type='synchronous_put_owner_barrier',
  transfer_observed_s=time.time(),send_counters_verified=True,transfers=[dict(rank),dict(rank)])

def test_missing_send_counter_cannot_pass_as_equal_none():
 proof=native_proof();b.check_drain(dict(generation=5),proof)
 proof['transfers'][0].pop('send_started');proof['transfers'][0].pop('send_completed')
 with pytest.raises(RuntimeError):b.check_drain(dict(generation=5),proof)

def test_missing_inflight_observation_cannot_pass_as_zero():
 proof=native_proof();b.check_drain(dict(generation=5),proof)
 proof['transfers'][1].pop('inflight_receives')
 with pytest.raises(RuntimeError):b.check_drain(dict(generation=5),proof)


def test_complete_model_reference_retains_all_original_hashes():
 expected=json.loads((ROOT/'expected-identity.json').read_text())
 source=json.loads((ROOT/'model-reference-source.json').read_text())
 assert source['old_22_all_equal'] and source['old_expected_count']==22
 assert len(expected['model_files_sha256'])==28 and expected['model_files_sha256']==source['model_files_sha256']
 old.validate_identity(expected,expected)

def test_exact_model_gate_still_rejects_missing_or_changed_file():
 import copy
 expected=json.loads((ROOT/'expected-identity.json').read_text());actual=copy.deepcopy(expected)
 actual['model_files_sha256'].pop(next(iter(actual['model_files_sha256'])))
 with pytest.raises(ValueError):old.validate_identity(actual,expected)
 actual=copy.deepcopy(expected);actual['model_files_sha256'][next(iter(actual['model_files_sha256']))]='changed'
 with pytest.raises(ValueError):old.validate_identity(actual,expected)
