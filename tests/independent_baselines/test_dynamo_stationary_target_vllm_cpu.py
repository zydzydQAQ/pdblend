"""Pinned-image CPU ABI and failure contracts; no target GPU serving claim."""
import asyncio
from copy import deepcopy
from types import SimpleNamespace as NS

import pytest
import torch

pytest.importorskip('vllm')

from pdblend_baselines.dynamollm.stationary_target_loader import LOAD_FORMAT,StationaryTargetLoader,initialize_rope_buffers
from pdblend_baselines.dynamollm.stationary_target_bootstrap import TargetBootstrap
from pdblend_baselines.dynamollm.stationary_target_worker import SameTpTargetWorker,TargetAwareSourceExtension
from pdblend_baselines.dynamollm.stationary_target_service import TargetPublicFence,target_operation,TargetAwareSourceCoordinator
from tests.independent_baselines.test_dynamo_stationary_target_bootstrap import config,bound
from tests.independent_baselines.test_dynamo_stationary_service import NativeOracle
from tests.independent_baselines.test_dynamo_stationary_kv import drained


def test_actual_vllm_registry_selects_private_loader_and_dense_reload_download_are_forbidden():
    from vllm.config import LoadConfig
    from vllm.model_executor.model_loader import get_model_loader
    loader=get_model_loader(LoadConfig(load_format=LOAD_FORMAT,model_loader_extra_config={'bootstrap_ref':{}}))
    assert type(loader) is StationaryTargetLoader
    with pytest.raises(RuntimeError,match='checkpoint'):loader.download_model(None)
    with pytest.raises(RuntimeError,match='dense'):loader.load_weights(None,None)
    assert not torch.cuda.is_initialized()


def test_actual_pinned_rope_buffer_initialization_preserves_every_meta_weight():
    from vllm.config import VllmConfig,DeviceConfig,set_current_vllm_config
    from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding
    with set_current_vllm_config(VllmConfig(device_config=DeviceConfig(device='cpu'))):
        with torch.device('meta'):
            model=torch.nn.Module();model.rope=RotaryEmbedding(16,16,128,10000,True,torch.bfloat16)
            model.linear=torch.nn.Linear(16,16,dtype=torch.bfloat16)
        identities=[id(p) for p in model.parameters()]
        rows=initialize_rope_buffers(model,'cpu',cpu_oracle=True)
        assert rows and rows[0]['bytes']==128*16*2
        assert all(p.is_meta for p in model.parameters()) and [id(p) for p in model.parameters()]==identities
        expected=RotaryEmbedding(16,16,128,10000,True,torch.bfloat16)
        torch.testing.assert_close(model.rope.cos_sin_cache,expected.cos_sin_cache,rtol=0,atol=0)
        with pytest.raises(ValueError,match='before private'):initialize_rope_buffers(model,'cpu',cpu_oracle=True)
        assert not torch.cuda.is_initialized()


def test_unknown_meta_buffer_cannot_be_materialized_as_a_weight_copy():
    model=torch.nn.Module();model.register_buffer('unknown',torch.empty(8,device='meta'))
    with pytest.raises(ValueError,match='unreviewed'):initialize_rope_buffers(model,'cpu',cpu_oracle=True)
    assert model.unknown.is_meta


def test_target_public_mutations_remain_fenced_even_after_internal_model_initialization():
    async def run():
        calls=[];messages=[]
        async def app(scope,receive,send):calls.append(scope['path'])
        async def recv():return {'type':'http.request'}
        async def send(value):messages.append(value)
        fence=TargetPublicFence(app)
        await fence(dict(type='http',method='POST',path='/v1/completions'),recv,send)
        assert messages[0]['status']==409 and not calls
        await fence(dict(type='http',method='POST',path='/baseline/dynamollm/target/status'),recv,send)
        assert calls==['/baseline/dynamollm/target/status']
    asyncio.run(run())


@pytest.mark.parametrize('fault',[None,'tokens','no_native_forward','native_error','final_drain'])
def test_private_target_golden_runs_native_generator_and_requires_real_forward_counter(tmp_path,monkeypatch,fault):
    from pdblend_runtime import serve
    c=config(tmp_path)
    identity=dict(model_id='Qwen2.5-7B-Instruct',model_hash='m',tokenizer_hash='t',engine_revision='vllm',image_digest='image')
    raw=bound(tmp_path,'raw.json',dict(token_ids=[10,11],events=[dict(finished=True,token_ids=[10,11])]))
    c['golden_ref']=bound(tmp_path,'golden.json',dict(schema='dynamo-native-ordinary-golden/v1',**identity,
        tp=1,pp=1,prompt=[1,2,3],seed=9701,max_tokens=2,ignore_eos=True,token_ids=[10,11],raw_ref=raw))
    bootstrap=TargetBootstrap(bound(tmp_path,'bootstrap.json',c))
    state=NS(dynamo_target_bootstrap=bootstrap,dynamo_target_lock=asyncio.Lock(),
        dynamo_target_golden_started=False,native_identity=identity)
    request=NS(app=NS(state=state))
    async def payload():return dict(transaction_id='tx',expected_generation=0)
    request.json=payload
    count=[0];drains=[]
    async def drain(request,payload):
        if fault=='final_drain' and drains:raise RuntimeError('real final drain failed')
        value=drained(1);value['generation']=0
        for row in value['ranks']:row['generation']=0
        drains.append(value);return value
    async def workers(request,method,operation,payload):
        assert method=='dynamo_target_operation' and operation=='status'
        return [dict(rank=0,generation=0,gpu_uuid='GPU-0',KV_initialized=True,binding_closed=False,
                     native_execute_model_completed_calls=count[0])]
    async def generate(request,payload,*,private):
        assert private is True and payload['prompt']==[1,2,3] and payload['seed']==9701
        if fault=='native_error':raise RuntimeError('actual engine error')
        if fault!='no_native_forward':count[0]+=1
        yield dict(finished=True,token_ids=[10,12] if fault=='tokens' else [10,11])
    monkeypatch.setattr(serve,'drain_engine',drain);monkeypatch.setattr(serve,'workers',workers)
    monkeypatch.setattr(serve,'generate_events',generate)
    result=asyncio.run(target_operation(request,'golden'))
    assert result['private_native_output_match'] is (fault is None)
    assert not result['target_served'] and not result['formal_eligible']
    assert len(drains)==(1 if fault=='final_drain' else 2)
    assert result['process_isolation_required'] is (fault=='final_drain')
    assert (bootstrap.directory/'target-rank-0-native-golden.json').is_file()


def test_target_view_release_requires_real_fresh_full_native_drain_before_ack(monkeypatch):
    worker=object.__new__(SameTpTargetWorker);worker.rank=0;worker._native_generation=0
    closed=[]
    runtime=NS(bootstrap=NS(config=dict(transaction_id='tx')),codec=NS(uuid='GPU-0'),
               close=lambda:closed.append(True) or dict(real_ack=True))
    worker._target_runtime=runtime
    value=drained(1);value['generation']=0
    for row in value['ranks']:row['generation']=0
    p=dict(transaction_id='tx',expected_generation=0,native_scheduler_drain=value)
    stale=deepcopy(p);stale['native_scheduler_drain']['native_at_s']=0
    with pytest.raises(RuntimeError):worker.dynamo_target_operation('close',stale)
    assert not closed
    result=worker.dynamo_target_operation('close',p)
    assert closed==[True] and result['target_process_exit_required'] and not result['formal_eligible']


def test_actual_native_worker_load_hook_keeps_runner_guard_and_counts_successful_native_execution(monkeypatch):
    from pdblend_runtime.native_v1 import NativeWorker
    worker=object.__new__(SameTpTargetWorker);worker._target_kv_ready=False
    worker._target_native_executions=worker._target_scheduled_tokens=0
    runtime=NS(closed=False,bootstrap=NS(config=dict(target_generation=0)),inventory=NS(check=lambda:None))
    def original(self):
        self._native_generation=0
        self.model_runner=NS(model=NS(_dynamo_stationary_target_runtime=runtime),execute_model=lambda output:'native-result')
    monkeypatch.setattr(NativeWorker,'load_model',original)
    worker.load_model();output=NS(total_num_scheduled_tokens=3)
    with pytest.raises(ValueError,match='KV/binding'):worker.model_runner.execute_model(output)
    worker._target_kv_ready=True
    assert worker.model_runner.execute_model(output)=='native-result'
    assert worker._target_native_executions==1 and worker._target_scheduled_tokens==3
    runtime.closed=True
    with pytest.raises(ValueError,match='KV/binding'):worker.model_runner.execute_model(output)


def test_actual_attention_constructor_unregistered_meta_scalars_are_initialized_without_weights():
    from vllm.attention.layer import Attention
    from vllm.config import VllmConfig,DeviceConfig,set_current_vllm_config
    from pdblend_baselines.dynamollm.stationary_target_loader import initialize_attention_constants
    class BackendOracle:
        accept_output_buffer=True
        @staticmethod
        def get_name():return 'FLASH_ATTN'
        @staticmethod
        def get_impl_cls():return lambda *args,**kwargs:NS()
    with set_current_vllm_config(VllmConfig(device_config=DeviceConfig(device='cpu'))):
        with torch.device('meta'):
            root=torch.nn.Module()
            root.attn=Attention(8,64,.125,num_kv_heads=4,prefix='private-cpu-abi',attn_backend=BackendOracle)
            root.linear=torch.nn.Linear(16,16,dtype=torch.bfloat16)
        assert not list(root.attn.buffers()) and root.attn._k_scale.is_meta
        before=[id(p) for p in root.parameters()]
        rows=initialize_attention_constants(root,'cpu',cpu_oracle=True)
        assert len(rows)==7 and sum(row['bytes'] for row in rows)==28
        assert root.attn._q_scale.item()==root.attn._k_scale.item()==root.attn._v_scale.item()==root.attn._prob_scale.item()==1.
        assert all(p.is_meta for p in root.parameters()) and [id(p) for p in root.parameters()]==before
        assert root.attn.kv_cache[0].is_meta and root.attn.kv_cache[0].numel()==0
        assert not torch.cuda.is_initialized()


def test_unknown_unregistered_tensor_rejected_before_materializing_anything():
    from pdblend_baselines.dynamollm.stationary_target_loader import initialize_attention_constants
    root=torch.nn.Module();root.hidden=torch.empty(32,device='meta')
    with pytest.raises(ValueError,match='unregistered'):
        initialize_attention_constants(root,'cpu',cpu_oracle=True)
    assert root.hidden.is_meta
