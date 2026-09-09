import copy
import importlib.util
from pathlib import Path
import time
import pytest

spec=importlib.util.spec_from_file_location('b32_candidate',Path(__file__).with_name('run.py'))
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


@pytest.mark.parametrize('status',[{},dict(complete=True,phase='pdblend_development_cells'),
    dict(complete=True,phase='finished'),dict(complete=False,phase='finished',finished_s=1)])
def test_active_or_unproven_terminal_task_cannot_allow_deployment(status):
    assert not module.terminal(status)


def test_terminal_failed_screening_is_not_hidden():
    assert module.terminal(dict(complete=False,phase='failed',finished_s=1))


def test_b32_configs_preserve_tp2_model_and_allocate_disjoint_new_channels():
    cs=module.configs(dict(model='/models/Qwen2.5-32B-Instruct',tp=2,port=24300,kv_port=28600))
    assert [c['tp'] for c in cs]==[2,2]
    assert all(c['validated_tp_pairs']==[[2,2]] for c in cs)
    assert all(c['max_num_batched_tokens']==8192 and c['max_num_seqs']==32 for c in cs)
    assert all(c['retained_weights'] is None for c in cs)
    assert not set(module.PORTS)&set(range(24300,24304))
    assert set(range(module.KV_PORTS[0],module.KV_PORTS[0]+32)).isdisjoint(range(module.KV_PORTS[1],module.KV_PORTS[1]+32))
    assert module.NAMES==['pdb-v2-nextv3b0','pdb-v2-nextv3b1']


@pytest.mark.parametrize('template',[dict(model='/models/Qwen2.5-14B-Instruct',tp=2),dict(model='/models/Qwen2.5-32B-Instruct',tp=1)])
def test_a_model_or_tp1_copy_cannot_deploy(template):
    with pytest.raises(RuntimeError):module.configs(template)


def drain():
    return dict(drained=True,accepting=False,generation=8,drain_proof_type='synchronous_put_owner_barrier',
        transfers=[dict(buffered_tensors=0,inflight_receives=0,inflight_sends=0,buffered_gpu_bytes=0,allocations={},listener_alive=True) for _ in range(2)])


@pytest.mark.parametrize('change',['single_rank','residual','wrong_generation','wrong_proof'])
def test_real_two_rank_drain_required(change):
    p=drain();module.drain_proof(dict(generation=7),p)
    if change=='single_rank':p['transfers']=p['transfers'][:1]
    elif change=='residual':p['transfers'][1]['buffered_tensors']=1
    elif change=='wrong_generation':p['generation']=7
    else:p['drain_proof_type']='cached_telemetry'
    with pytest.raises(RuntimeError):module.drain_proof(dict(generation=7),p)


def test_missing_residual_key_is_not_idle():
    state=dict(active=0,running=0,waiting=0,kv_allocations={},transfer_allocations={},
        transfer_buffered_tensors=0,transfer_inflight_receives=0,transfer_inflight_sends=0,
        generation=3,acknowledged_generation=3,transport_healthy=True,timestamp=time.time())
    module.idle(state)
    del state['transfer_inflight_sends']
    with pytest.raises(RuntimeError,match='residual'):module.idle(state)


def inventory(extra=None):
    env=dict(NCCL_P2P_DISABLE='1',NCCL_SHM_DISABLE='1',NCCL_IB_DISABLE='1',
             NCCL_CUMEM_ENABLE='0',NCCL_DEBUG='WARN')
    env.update(extra or {})
    return [dict(Name='/'+name,Config=dict(Env=[k+'='+v for k,v in env.items()]+['CUDA_VISIBLE_DEVICES=0,1']))
            for name in module.OLD_NAMES[:2]]


@pytest.mark.parametrize('channels',[{},dict(NCCL_MIN_NCHANNELS='8',NCCL_MAX_NCHANNELS='8')])
def test_actual_nccl_channels_are_preserved_without_inventing_absent_values(channels):
    result=module.inherited_transport(inventory(channels))['nccl_environment']
    assert {k:v for k,v in result.items() if 'NCHANNELS' in k}==channels
    assert 'CUDA_VISIBLE_DEVICES' not in result


def test_different_old_replica_transports_fail_before_deployment():
    rows=inventory();rows[1]['Config']['Env'].append('NCCL_MIN_NCHANNELS=8')
    with pytest.raises(RuntimeError,match='different NCCL'):module.inherited_transport(rows)


def test_additional_actual_nccl_setting_is_retained():
    result=module.inherited_transport(inventory(dict(NCCL_SOCKET_IFNAME='lo')))['nccl_environment']
    assert result['NCCL_SOCKET_IFNAME']=='lo'


def test_missing_known_disable_setting_is_rejected():
    rows=inventory()
    for r in rows:r['Config']['Env']=[v for v in r['Config']['Env'] if not v.startswith('NCCL_P2P_DISABLE=')]
    with pytest.raises(RuntimeError,match='unexpected old transport'):module.inherited_transport(rows)


@pytest.mark.parametrize('wrong_tag',[False,True])
def test_actual_base_is_tagged_and_resolution_verified_before_build(monkeypatch,wrong_tag):
    import asyncio,json
    calls=[]
    async def command(*args,**kwargs):
        calls.append(args)
        if args[:3]==('docker','image','inspect'):
            image=module.A_IMAGE if wrong_tag and args[3]==module.BASE_TAG else module.BASE_IMAGE
            return json.dumps([{'Id':image}])
        return ''
    monkeypatch.setattr(module,'command',command)
    if wrong_tag:
        with pytest.raises(RuntimeError,match='local B base tag'):asyncio.run(module.tag_base_image())
    else:
        base,tagged=asyncio.run(module.tag_base_image())
        assert base['Id']==tagged['Id']==module.BASE_IMAGE
    assert calls==[('docker','image','inspect',module.BASE_IMAGE),
                   ('docker','tag',module.BASE_IMAGE,module.BASE_TAG),
                   ('docker','image','inspect',module.BASE_TAG)]
