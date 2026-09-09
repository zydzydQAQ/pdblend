import asyncio,copy,hashlib,importlib.util,json,sys,time
from pathlib import Path
from types import SimpleNamespace
import pytest
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
b=load('b_fixed_wrapper',ROOT/'run.py')
e=load('b_fixed_execution',ROOT/'execution.py')
from epoch import EpochGuard
from replay_contract import validate
def test_real_run_trace_late_epoch_zero_session(tmp_path,monkeypatch):
 import benchmarks.scripts.bench_vllm as bench
 calls=[];real=bench.evaluation_headers
 guard=EpochGuard(dict(issued_s=0,latest_arrival_epoch_s=1),tmp_path/'epoch.json',real)
 monkeypatch.setattr(bench,'evaluation_headers',guard)
 monkeypatch.setattr(bench.aiohttp,'TCPConnector',lambda **kw:calls.append('session'))
 with pytest.raises(RuntimeError,match='zero dispatch'):
  asyncio.run(bench.run_trace(dict(requests=[dict(arrival_s=0,prompt_len=1,output_len=1)],prompts=[[1]]),'http://127.0.0.1:18080','x',evaluation_protocol=bench.EVALUATION_V3))
 assert not calls and json.loads((tmp_path/'epoch.json').read_text())['accepted'] is False
def test_epoch_first_hook_requires_identical_epoch(tmp_path):
 g=EpochGuard(dict(issued_s=0,latest_arrival_epoch_s=100),tmp_path/'epoch.json',lambda *a:True)
 with pytest.raises(RuntimeError):g('url','p',10,10.1)
def test_epoch_hook_only_first_dispatch_is_gate(tmp_path):
 calls=[];g=EpochGuard(dict(issued_s=0,latest_arrival_epoch_s=10),tmp_path/'epoch.json',lambda *a:calls.append(a))
 g('url','p',10,10);g('url','p',20,20.3);assert len(calls)==2
 assert json.loads((tmp_path/'epoch.json').read_text())['actual_epoch_s']==10
def test_restore_always_resumes_after_bad_native_proof(monkeypatch):
 calls=[]
 async def idle(*a,**k):return dict(id='x',generation=7,role='mixed',mode='continuous',admit_prefill=True,admit_decode=True)
 async def http(session,index,path,body=None,**kw):
  calls.append((path,body));return dict(id='x',generation=6)
 def fail(*args):raise RuntimeError('bad rank proof')
 fake=SimpleNamespace(IDS=['x'],http=http,require=b.require,native_barrier=fail)
 monkeypatch.setattr(e,'wait_idle',idle)
 r=asyncio.run(e.restore_one(fake,None,0))
 assert not r['complete'] and any('bad rank proof' in x for x in r['errors'])
 assert any(path=='/control' and body['scheduler_budget']['max_num_batched_tokens']==8192 for path,body in calls)
@pytest.mark.parametrize('verified,mutated,child_failure',[(False,False,False),(True,True,True)])
def test_cleanup_never_mutates_unverified_or_unstopped_child(monkeypatch,verified,mutated,child_failure):
 calls=[]
 async def stop(child):
  if child_failure:raise RuntimeError('child still alive')
 async def http(*a,**kw):calls.append('write')
 fake=SimpleNamespace(IDS=['a','b'],PORTS=(33500,33501),http=http,write=lambda *a:None)
 obj=object.__new__(e.Cell);obj.b=fake;obj.verified=verified;obj.mutated=mutated;obj.child=None;obj.row={'cell_id':'test'};obj.receipt={}
 monkeypatch.setattr(e,'stop_child',stop)
 r=asyncio.run(obj.cleanup());assert not calls and not r['native']
 if child_failure:assert not r['complete'] and not r['clock_release_complete']
@pytest.mark.parametrize('counter',['send_started','send_completed','send_failed','buffered_gpu_bytes'])
def test_native_rank_missing_observation_rejected(counter):
 rank=dict(buffered_tensors=0,inflight_receives=0,inflight_sends=0,listener_alive=True,send_counters_observed=True,send_healthy=True,send_started=0,send_completed=0,send_failed=0,allocations={},buffered_gpu_bytes=0)
 rank.pop(counter);proof=dict(drained=True,accepting=False,generation=2,drain_proof_type='synchronous_put_owner_barrier',send_counters_verified=True,transfers=[rank,rank])
 with pytest.raises(RuntimeError):b.native_barrier(dict(generation=1),proof)
def test_epoch_guard_lives_in_exact_frozen_benchmark_module():
 import ecopadg.serving.cell as cell
 import benchmarks.scripts.bench_vllm as bench
 assert cell.run_trace is bench.run_trace and cell.run_trace.__globals__ is bench.__dict__

def test_three_original_traces_and_fixed_window_off():
 from ecopadg.serving.cell import configure_fixed_window
 child=load('b_replay_child',ROOT/'child.py')
 spec=b.package_check();refs=b.read(spec['source_replay_plan'])['cells']
 for row in spec['cells']:
  ref=next(r for r in refs if r['model']=='32b' and r['dataset']==row['dataset'])
  assert Path(row['trace']).read_bytes()==Path(ref['trace_path']).read_bytes()
  assert configure_fixed_window(child.cell_arguments(row),b.read(b.CONFIG),b.read(row['trace'])) is None
  assert row['seed']==11 and row['n_requests']==64

@pytest.mark.parametrize('dataset',['alpaca','sharegpt','longbench'])
def test_actual_replay_controller_slo_before_network(tmp_path,monkeypatch,dataset):
 from ecopadg.serving import cell
 child=load('b_replay_child_config',ROOT/'child.py')
 row=next(r for r in b.read(ROOT/'runspec.json')['cells'] if r['dataset']==dataset)
 args=child.cell_arguments(row);args.out=tmp_path/'out';seen=[]
 class BeforeNetwork(Exception):pass
 def constructor(cfg):seen.append(copy.deepcopy(cfg));raise BeforeNetwork()
 monkeypatch.setattr(cell,'Controller',constructor)
 with pytest.raises(BeforeNetwork):asyncio.run(cell.run_cell(args))
 assert (seen[0]['slo_ttft_s'],seen[0]['slo_tpot_s'])==(row['slo_ttft_s'],row['slo_tpot_s'])
 assert seen[0]['output_prior']==211 and 'measurement_window_protocol' not in seen[0]
 assert seen[0]['scheduler_budget_ablation']['max_num_batched_tokens']==8192

def test_deadline_cannot_begin_long_replay_late():
 from replay_queue import cell_limits
 row=dict(trace_duration_s=568.)
 assert cell_limits(dict(deadline_s=1000.),row,132.) is not None
 assert cell_limits(dict(deadline_s=1000.),row,133.) is None

def test_energy_sampler_failure_is_checked_during_child():
 source=(ROOT/'execution.py').read_text()
 assert "b.require(not sampler.error,'outer instantaneous power failed')" in source
