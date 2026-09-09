import copy,hashlib,importlib.util,json,subprocess,sys,time,types
from pathlib import Path
import pytest
P=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('isolated_sampler',P/'isolated_sampler.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
SOURCE='nvml:field:186:scope:0:mW'
SOURCE_DICT=dict(schema=1,mode='instant',source_id=SOURCE,field_id=186,scope_id=0,unit='W')
def ref(p):return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v));return ref(p)
@pytest.fixture
def fake(tmp_path):
 root=tmp_path/'host';d=root/'src/ecopadg/measure';d.mkdir(parents=True)
 for p in [root/'src/ecopadg/__init__.py',d/'__init__.py']:p.write_text('')
 original=P.parent.parent/'hosts/14b-capacity-p8/src/ecopadg/measure/power.py'
 (d/'power.py').write_bytes(original.read_bytes())
 (d/'backends.py').write_text("""import time
INSTANT_POWER_SOURCE_ID='nvml:field:186:scope:0:mW'
class GpuBackend:pass
class PynvmlBackend:
 def __init__(self,power_mode):self.power_source=dict(schema=1,mode=power_mode,source_id=INSTANT_POWER_SOURCE_ID,field_id=186,scope_id=0,unit='W')
 def power_reading(self,gpu):
  t=time.time();return dict(watts=100+gpu,mode='instant',source_id=INSTANT_POWER_SOURCE_ID,field_id=186,scope_id=0,value_type=1,return_code=0,nvml_timestamp_us=int(t*1e6),nvml_latency_us=0,read_started_s=t,read_finished_s=time.time())
 def utilization_pct(self,gpu):return 0
 def current_freq(self,gpu):return 1500
""")
 host=write(root/'manifest.json',dict(files={str(p.relative_to(root)):ref(p)['sha256'] for p in d.glob('*.py')}))
 adapter=write(tmp_path/'adapter.json',dict(files={str(p):ref(p)['sha256'] for p in [P/'isolated_sampler.py',P/'sampler_worker.py']}))
 m._CONTEXT=(tmp_path/'raw',host,adapter)
 sampler=m.IsolatedPowerSampler(range(8),backend=types.SimpleNamespace(power_source=SOURCE_DICT),sample_clocks=True)
 return sampler,tmp_path

def test_actual_spawn_original_sampler_full_snapshot_and_exit(fake):
 a,tmp=fake;a.start();assert a.wait_ready();time.sleep(.07);a.stop()
 assert a.error is None and not a._thread.is_alive() and len(a.samples)>=4
 assert len(a.samples)==len(a.power_metadata)==len(a.frequency_samples)==len(a.utilization_samples)
 receipt=json.loads((a._directory/'receipt.json').read_text());assert receipt['complete'] and receipt['child_exited']
 assert a._terminal['stream_sha256']==a._digest.hexdigest()

def test_parent_busy_does_not_stop_sampling_or_drop_rows(fake):
 a,tmp=fake;a.start();a.wait_ready();before=len(a.samples)
 # Deliberately stop the parent reader via a long GIL switch interval. The
 # independent worker's original sampler continues and retains its full log.
 old=sys.getswitchinterval();sys.setswitchinterval(.6)
 try:
  until=time.monotonic()+.4
  while time.monotonic()<until:pass
 finally:sys.setswitchinterval(old)
 a.stop();assert a.error is None and len(a.samples)>=before+12
 assert max(y[0]-x[0] for x,y in zip(a.samples,a.samples[1:]))<.25

def test_child_failure_is_explicit_and_no_successful_terminal(fake):
 a,tmp=fake;a.start();a.wait_ready();a._process.kill();a.stop();assert a.error and not a._thread.is_alive()

def test_ready_timeout_is_not_observation(fake,monkeypatch):
 a,tmp=fake;a._process=types.SimpleNamespace(poll=lambda:None)
 with pytest.raises(RuntimeError):a.wait_ready(timeout=.01)
 a._process=None

def packet(a,index=0):
 t=time.time();return dict(kind='row',index=index,power=[t,[100]*8],metadata=dict(t_s=t,gpus=list(range(8))),utilization=[t,[0]*8],frequency=[t,[1500]*8])
@pytest.mark.parametrize('change',['index','gpu','timestamp','width','partial_terminal','digest'])
def test_ipc_cannot_lose_relabel_or_truncate_raw(fake,change):
 a,tmp=fake;a._process=types.SimpleNamespace(pid=123);a._header=True;r=packet(a)
 if change=='index':r['index']=1
 elif change=='gpu':r['metadata']['gpus'][-1]=6
 elif change=='timestamp':r['metadata']['t_s']+=1
 elif change=='width':r['power'][1].pop()
 else:r=dict(kind='terminal',pid=123,counts={},emitted=1,stream_sha256='wrong',sampler_thread_stopped=True,error=None)
 with pytest.raises(RuntimeError):a._accept((json.dumps(r)+'\n').encode())
 a._process=None

def test_original_read_and_loop_are_not_modified():
 worker=(P/'sampler_worker.py').read_text()
 assert 'sampler=PowerSampler(' in worker and 'sampler.start()' in worker
 assert 'sampler._read(' not in worker and 'sampler._loop(' not in worker
 assert 'gc.disable()' in worker


def test_original_field_age_gate_still_rejects_stale_row(fake):
 a,tmp=fake;a.start();a.wait_ready();a.stop()
 host=P.parent.parent/'hosts/14b-capacity-p8';sys.path[:0]=[str(host/'src'),'/root/workspace/pdblend/.runtime-deps']
 from ecopadg.serving.measurement import power_evidence
 assert power_evidence(a.samples,a.power_source,a.power_metadata)['power_source_verified']
 changed=copy.deepcopy(a.power_metadata);changed[0]['read_finished_s'][0]-=.423
 assert not power_evidence(a.samples,a.power_source,changed)['power_source_verified']
 assert len(changed)==len(a.power_metadata)

def test_original_nvml_timestamp_gate_still_rejects_future_or_regression(fake):
 a,tmp=fake;a.start();a.wait_ready();a.stop()
 host=P.parent.parent/'hosts/14b-capacity-p8';sys.path[:0]=[str(host/'src'),'/root/workspace/pdblend/.runtime-deps']
 from ecopadg.serving.measurement import power_evidence
 for which in ('future','regressed'):
  changed=copy.deepcopy(a.power_metadata)
  changed[1]['nvml_timestamp_us'][0]=changed[0]['nvml_timestamp_us'][0]-1 if which=='regressed' else int((changed[1]['read_finished_s'][0]+1)*1e6)
  assert not power_evidence(a.samples,a.power_source,changed)['power_source_verified']

def test_stop_is_idempotent_and_retains_one_receipt(fake):
 a,tmp=fake;a.start();a.wait_ready();a.stop();digest=ref(a._directory/'receipt.json')['sha256'];a.stop();assert ref(a._directory/'receipt.json')['sha256']==digest
