import asyncio
import hashlib
import json

import pytest
pytest.importorskip('fastapi')

from pdblend_baselines.dynamollm.stationary_kv_probe import probe_on_resident, PREFIX
from pdblend_baselines.dynamollm.stationary_service import StationaryContext, StationaryCoordinator
from tests.independent_baselines.test_dynamo_stationary_service import NativeOracle


@pytest.fixture(autouse=True)
def no_http_in_protocol_oracles(monkeypatch):
    from pdblend_baselines.dynamollm import stationary_kv_probe as probe
    async def guard(session,url,transaction):
        return dict(cpu_oracle=True,http_status=409,body=dict(phase='released',transaction_id=transaction))
    monkeypatch.setattr(probe,'verify_public_fence',guard)


@pytest.mark.parametrize('status,phase',[(409,'released'),(200,'released'),(409,'idle')])
def test_actual_public_fence_http_contract_refuses_success_or_wrong_phase(monkeypatch,status,phase):
    # Import the original function again without the fixture's replacement.
    import importlib.util
    from pdblend_baselines.dynamollm import stationary_kv_probe as probe
    spec=importlib.util.spec_from_file_location('pdblend_baselines.dynamollm._probe_fence_test',probe.__file__)
    actual=importlib.util.module_from_spec(spec);spec.loader.exec_module(actual)
    class Response:
        async def __aenter__(self):return self
        async def __aexit__(self,*_):pass
        async def text(self):return json.dumps(dict(phase=phase,transaction_id='tx'))
    class Session:
        def post(self,url,json):
            assert url=='source/baseline/control' and json==dict(accepting=False)
            value=Response();value.status=status;return value
    if status==409 and phase=='released':
        assert asyncio.run(actual.verify_public_fence(Session(),'source','tx'))['http_status']==409
    else:
        with pytest.raises(ValueError,match='did not fence'):
            asyncio.run(actual.verify_public_fence(Session(),'source','tx'))


@pytest.mark.parametrize('fault', [None, 'golden', 'partial_rank', 'before_golden'])
def test_resident_probe_preserves_real_protocol_boundary_and_never_fakes_token_goldens(tmp_path, monkeypatch, fault):
    from pdblend_runtime import probe as native_probe
    native = NativeOracle()
    ctx = StationaryContext()
    coordinator = StationaryCoordinator(ctx, native)
    identity = dict(native.identity()['identity'], tp=2, pp=1)
    path = tmp_path/'identity.json'; path.write_text(json.dumps(identity))
    binding = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    original_ranks = native.ranks
    async def ranks(method, **kwargs):
        rows = await original_ranks(method, **kwargs)
        if kwargs.get('operation') == 'describe':
            for row in rows:
                row.update(shapes={'model.norm.weight':[4]}, geometry=dict(hidden_size=4,
                    num_attention_heads=2, num_key_value_heads=2, intermediate_size=8))
        return rows
    native.ranks = ranks
    if fault == 'partial_rank': native.fault = 'rpc'
    async def call(session, url, endpoint, body=None):
        assert url == 'http://owned-test-resident'
        if endpoint == '/baseline/capability':
            return dict(identity, supported=True, state=await native.state())
        if endpoint == '/baseline/drain': return await native.drain()
        assert endpoint.startswith(PREFIX)
        return await coordinator.execute(endpoint[len(PREFIX):], body)
    requests = []
    async def generate(session, url, payload):
        requests.append(payload)
        assert payload['seed'] == 9701 and payload['max_tokens'] == 16 and payload['ignore_eos']
        tokens = list(range(16))
        if fault == 'golden' and '-after-' in payload['request_id']: tokens[-1] = 17
        if fault == 'before_golden' and payload['request_id'].endswith('-before-1'): tokens[-1] = 17
        return dict(events=[dict(finished=True, token_ids=tokens)], token_ids=tokens)
    monkeypatch.setattr(native_probe, 'call', call)
    monkeypatch.setattr(native_probe, 'generate', generate)
    report = asyncio.run(probe_on_resident(None, 'http://owned-test-resident', tmp_path/'raw', identity_binding=binding))
    assert report['ready_for_next'] is (fault is None)
    assert report['safe_restore_passed'] is (fault != 'partial_rank')
    assert report['full_tp_switch_qualified'] is False and report['formal_eligible'] is False
    if fault == 'partial_rank':
        assert ctx.phase == 'quarantined' and 'restore_error' in report
        assert not any('-after-' in r['request_id'] for r in requests)
    else:
        assert ctx.phase == 'idle' and native.accepting is False
        assert report['final_native_drain']['generation'] == (5 if fault == 'before_golden' else 6)
    assert (tmp_path/'raw/completion.json').is_file()


@pytest.mark.parametrize('peer_fault', [None, 'token', 'epoch'])
def test_peer_really_serves_during_source_absence_and_peer_mismatch_cannot_qualify(tmp_path, monkeypatch, peer_fault):
    from pdblend_runtime import probe as native_probe
    source, peer = NativeOracle(), NativeOracle(tp=1)
    context = StationaryContext(); coordinator = StationaryCoordinator(context, source)
    source_identity = dict(source.identity()['identity'], tp=2, pp=1)
    peer_identity = dict(peer.identity()['identity'], tp=1, pp=1, gpu_uuids=['GPU-peer'])
    bindings = []
    for name, identity in [('source',source_identity),('peer',peer_identity)]:
        p = tmp_path/(name+'.json'); p.write_text(json.dumps(identity))
        bindings.append(dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    original = source.ranks
    async def ranks(method, **kwargs):
        rows = await original(method, **kwargs)
        if kwargs.get('operation') == 'describe':
            for row in rows: row.update(shapes={'model.norm.weight':[4]}, geometry=dict(hidden_size=4,
                num_attention_heads=2, num_key_value_heads=2, intermediate_size=8))
        return rows
    source.ranks = ranks
    async def call(session, url, endpoint, body=None):
        native = peer if url == 'peer' else source
        identity = peer_identity if url == 'peer' else source_identity
        if endpoint == '/baseline/capability': return dict(identity,supported=True,state=await native.state())
        if endpoint == '/baseline/state': return await native.state()
        if endpoint == '/baseline/drain': return await native.drain()
        assert url == 'source'
        return await coordinator.execute(endpoint[len(PREFIX):],body)
    completed_in_gap = []
    async def generate(session, url, payload):
        tokens = list(range(16))
        if url == 'peer' and 'source-KV-released-' in payload['request_id']:
            assert context.phase == 'released' and peer.accepting
            completed_in_gap.append(payload['request_id'])
            if peer_fault == 'token': tokens[-1] = 99
            if peer_fault == 'epoch': peer.generation += 1
        return dict(token_ids=tokens,events=[dict(finished=True,token_ids=tokens)])
    monkeypatch.setattr(native_probe,'call',call);monkeypatch.setattr(native_probe,'generate',generate)
    report = asyncio.run(probe_on_resident(None,'source',tmp_path/'probe',identity_binding=bindings[0],
        peer=dict(url='peer',identity_binding=bindings[1])))
    assert report['ready_for_next'] is (peer_fault is None)
    assert report['safe_restore_passed'] and source.generation == 6
    assert report['peer_isolation_exercised'] is (peer_fault is None)
    assert report['peer_slo_qualified'] is False
    assert len(completed_in_gap) == (1 if peer_fault == 'token' else 2)


@pytest.mark.parametrize('fault',[None,'startup','probe','cleanup'])
def test_owned_two_gpu_harness_uses_real_native_launch_shape_and_cleans_each_started_group(tmp_path,monkeypatch,fault):
    from types import SimpleNamespace as NS
    from pdblend_baselines.dynamollm import stationary_kv_probe as probe,stationary_probe
    from pdblend.engine import launcher
    from pdblend_runtime import probe as native_probe
    from tests.independent_baselines.test_dynamo_stationary_kv import drained
    config=dict(source_sha256='source',image_digest='image',model_id='Qwen2.5-7B-Instruct',
                model_path='/models/Qwen2.5-7B-Instruct')
    monkeypatch.setattr(probe,'preflight',lambda _:config)
    monkeypatch.setenv('PDBLEND_GPU_UUIDS','GPU-source,GPU-peer');monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0,1')
    starts=[];stops=[];groups=[];ready=[];instances={}
    class Instance:
        def __init__(self,s):self.spec=s;self.events=[];self.process=None
        def event(self,kind):self.events.append(dict(kind=kind,t_s=0,instance=self.spec.instance_id))
        def start(self):
            command=self.spec.command();environment=self.spec.environment()
            assert command[command.index('-m')+1]=='pdblend_baselines.dynamollm.stationary_service'
            assert '--enable-sleep-mode' not in command and '--kv-transfer-config' not in command
            assert '--enforce-eager' in command
            assert command[command.index('--max-num-seqs')+1]=='32'
            assert environment['CUDA_VISIBLE_DEVICES']==('GPU-source' if self.spec.gpus==(0,) else 'GPU-peer')
            self.process=NS(pid=90000+len(starts));starts.append(self.spec.instance_id);self.event('start')
        def wait_ready(self,timeout):
            assert timeout==900;ready.append(self.spec.instance_id)
            if fault=='startup' and self.spec.gpus==(0,):raise RuntimeError('source load failed')
            self.event('ready');return 0.
        def stop(self):stops.append(self.spec.instance_id);self.process=None;self.event('stop')
    class Fleet:
        def __init__(self,specs,log):instances.update({s.instance_id:Instance(s) for s in specs})
        def __getitem__(self,i):return instances[i]
        def events(self):return [e for i in instances.values() for e in i.events]
    monkeypatch.setattr(launcher,'Fleet',Fleet)
    empty_calls=[]
    def empty(uuid,timeout):
        empty_calls.append(uuid)
        return dict(passed=not(fault=='cleanup' and len(empty_calls)>2),observations=[dict(gpu_uuid=uuid)])
    monkeypatch.setattr(stationary_probe,'require_compute_empty',empty)
    def killpg(pid,sig):groups.append(pid);raise ProcessLookupError
    monkeypatch.setattr(probe.os,'killpg',killpg)
    async def call(session,url,path,body=None):
        gpu='GPU-source' if url.endswith(':29000') else 'GPU-peer'
        if path=='/baseline/capability':
            return dict(config,model_hash='m',tokenizer_hash='t',engine_revision='vllm',
                        source_revision=config['source_sha256'],gpu_uuids=[gpu],tp=1,pp=1)
        assert path=='/baseline/drain' and gpu=='GPU-peer'
        value=drained(1);value['generation']=0
        for rank in value['ranks']:rank['generation']=0
        return value
    monkeypatch.setattr(native_probe,'call',call)
    async def on_resident(session,url,out,*,identity_binding,peer):
        assert probe.read_bound(identity_binding)['gpu_uuids']==['GPU-source']
        assert probe.read_bound(peer['identity_binding'])['gpu_uuids']==['GPU-peer']
        return dict(ready_for_next=fault!='probe',peer_isolation_exercised=True)
    monkeypatch.setattr(probe,'probe_on_resident',on_resident)
    report=asyncio.run(probe.run_owned('config',tmp_path/'owned',29000))
    assert report['complete'] is (fault is None)
    assert len(starts)==len(stops)==len(ready)==2 and set(groups)=={90000,90001}
    assert report['model_loads']==2 and len(report['actual_engine_starts'])==2
    assert report['actual_model_ready_count']==(1 if fault=='startup' else 2)
    assert empty_calls==['GPU-source','GPU-peer','GPU-source','GPU-peer']
    assert (tmp_path/'owned/gpu-after.json').is_file() and (tmp_path/'owned/completion.json').is_file()
    assert not report['formal_eligible'] and not report['full_tp_switch_qualified']
