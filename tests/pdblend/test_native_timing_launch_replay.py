"""Complete serialized launch evidence, including the real warmup epoch 1."""
from copy import deepcopy
from dataclasses import asdict
import json

import pytest

from pdblend_runtime.probe import NativeSpec
from pdblend.profile.collection.native_timing_replay import audit_native_launch


def fixture(tp=1):
    spec=NativeSpec('pd-timing-0',tuple(range(tp)),20000,'/models/Qwen2.5-7B-Instruct',
        tp=tp,max_num_seqs=32,generation=1,extra_args=('--enforce-eager','--worker-cls',
            'pdblend.profile.collection.native_timing_worker.PDNativeTimingWorker'))
    launch=dict(spec=json.loads(json.dumps(asdict(spec))),argv=spec.command(),environment=dict(
        CUDA_VISIBLE_DEVICES=','.join(map(str,range(tp))),VLLM_USE_V1='1',NCCL_CUMEM_ENABLE='0',
        NCCL_IB_DISABLE='1',NCCL_P2P_DISABLE='0'))
    capability=dict(supported=True,state=dict(max_num_seqs=32,max_model_len=8192,generation=1))
    args=dict(instance_id=spec.instance_id,gpus=range(tp),model_id='Qwen2.5-7B-Instruct')
    return launch,capability,args


@pytest.mark.parametrize('tp',[1,2])
def test_complete_native_launch_accepts_epoch_one_and_a_different_consumer_interpreter(tp):
    launch,capability,args=fixture(tp)
    launch['argv'][0]='/opt/venv/bin/python'
    audit_native_launch(launch,capability,**args)


@pytest.mark.parametrize('field,value',[
    ('max_num_seqs',128),('max_model_len',4096),('max_num_batched_tokens',4096),
    ('gpu_memory_utilization',.55),('kv_connector',None),('kv_role','kv_producer'),
    ('kv_port',44000),('side_channel_port',44001),('native_control',True),('pp',2),
    ('gpus',[1]),('extra_args',['--enforce-eager']),('generation',True)])
def test_self_reported_launch_spec_cannot_change_immutable_runtime_options(field,value):
    launch,capability,args=fixture();launch['spec'][field]=value
    with pytest.raises(ValueError):audit_native_launch(launch,capability,**args)


@pytest.mark.parametrize('change',['duplicate_flag','missing_eager','missing_worker','changed_module',
                                  'environment','missing_environment','capability_limit','generation_mismatch'])
def test_actual_command_environment_and_native_limits_are_independently_checked(change):
    launch,capability,args=fixture()
    if change=='duplicate_flag':launch['argv']+=['--max-num-seqs','128']
    elif change=='missing_eager':launch['argv'].remove('--enforce-eager')
    elif change=='missing_worker':
        i=launch['argv'].index('--worker-cls');del launch['argv'][i:i+2]
    elif change=='changed_module':launch['argv'][launch['argv'].index('pdblend_runtime.serve')]='vllm.entrypoints.openai.api_server'
    elif change=='environment':launch['environment']['NCCL_P2P_DISABLE']='1'
    elif change=='missing_environment':launch.pop('environment')
    elif change=='capability_limit':capability['state']['max_num_seqs']=128
    else:capability['state']['generation']=2
    with pytest.raises(ValueError):audit_native_launch(launch,capability,**args)
