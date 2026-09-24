"""Frozen primitive preparation and cleanup policy without touching CUDA/NVML."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from pdblend_baselines.dynamollm import stationary_probe as probe
from tests.independent_baselines.test_dynamo_stationary_tensors import narrow


def test_one_visible_leased_gpu_uses_only_explicitly_synthetic_unexecuted_rank():
    receipt=probe.lease_geometry('GPU-leased',[b'GPU-leased'])
    assert receipt['nvml_inventory_gpu_uuids']==['GPU-leased']
    assert receipt['unexecuted_logical_peer_uuid']==probe.UNKNOWN_UUID
    assert receipt['unexecuted_rank_uuid_source']=='synthetic_unknown_uuid_not_a_physical_peer'
    assert not receipt['logical_peer_is_observed_physical_gpu'] and receipt['peer_cuda_operations']==0
    plan=probe.tensor_plan(source_gpus=['GPU-leased'],target_gpus=['GPU-leased',probe.UNKNOWN_UUID],
        source_shapes=probe.shapes(1),target_shapes=probe.shapes(2),geometry=probe.GEOMETRY)
    assert plan['target_gpu_uuids'][0]=='GPU-leased'


@pytest.mark.parametrize('inventory',[[],['GPU-wrong'],['GPU-leased','GPU-other'],['GPU-leased','GPU-leased']])
def test_single_gpu_visibility_does_not_accept_an_unleased_or_extra_device(inventory):
    with pytest.raises(ValueError,match='exactly the one leased GPU'):
        probe.lease_geometry('GPU-leased',inventory)


def test_unknown_uuid_negative_guard_preserves_real_packet_and_never_imports_cuda(monkeypatch):
    from copy import deepcopy
    from pdblend_baselines.dynamollm.stationary_ipc import TorchCudaIpcCodec
    from tests.independent_baselines.test_dynamo_stationary_ipc import setup_owner,CONSUMER
    _,_,_,_,owner=setup_owner();packet=owner.export(CONSUMER,target_rank=0);original=deepcopy(packet)
    def forbidden(*args,**kwargs):raise AssertionError('negative metadata guard must never call CUDA import')
    monkeypatch.setattr(TorchCudaIpcCodec,'import_view',forbidden)
    result=probe.unknown_descriptor_rejection(packet,probe.UNKNOWN_UUID)
    assert result['passed'] and not result['cuda_import_attempted']
    assert not result['claimed_uuid_is_observed_physical_gpu'] and packet==original
    with pytest.raises(ValueError,match='unknown UUID test differs'):
        probe.unknown_descriptor_rejection(packet,'GPU-other')


def test_synthetic_oracle_covers_original_slices_without_copying_retained_parameter():
    from pdblend_baselines.dynamollm.stationary_tensors import tensor_plan
    plan=tensor_plan(source_gpus=['GPU-real-0'],target_gpus=['GPU-real-0','GPU-real-1'],
        source_shapes=probe.shapes(1),target_shapes=probe.shapes(2),geometry=probe.GEOMETRY)
    for piece in plan['pieces']:
        original=probe.expected_shard(piece['parameter'],1,0)
        actual=narrow(original,piece['axis'],piece['source_offset'],piece['length'])
        expected=narrow(probe.expected_shard(piece['parameter'],2,piece['target_rank']),
                        piece['axis'],piece['target_offset'],piece['length'])
        assert np.shares_memory(actual,original)
        np.testing.assert_array_equal(actual,expected)
    assert probe.PROBE_PLAN['gpu_count']==1 and probe.PROBE_PLAN['model_loads']==0
    assert probe.PROBE_PLAN['unexecuted_target_ranks']==[1]
    assert not probe.PROBE_PLAN['target_serving_activation_allowed']


def test_gpu_cleanup_observes_actual_empty_inventory_instead_of_trusting_child_exit(monkeypatch):
    states=iter([[dict(pid=7,used_gpu_memory_bytes=4096)],[]])
    monkeypatch.setattr(probe,'compute_processes',lambda _:next(states))
    monkeypatch.setattr(probe.time,'sleep',lambda _:None)
    receipt=probe.require_compute_empty('GPU-0',timeout=1)
    assert receipt['passed'] and len(receipt['observations'])==2
    assert receipt['observations'][0]['compute_processes'][0]['pid']==7
    monkeypatch.setattr(probe,'compute_processes',lambda _:[dict(pid=7)])
    receipt=probe.require_compute_empty('GPU-0',timeout=0)
    assert not receipt['passed'] and receipt['observations'][0]['compute_processes']


def test_child_error_or_timeout_is_not_a_successful_primitive_result():
    from types import SimpleNamespace
    with pytest.raises(ValueError,match='timeout'):
        probe.receive(SimpleNamespace(poll=lambda _:False))
    with pytest.raises(RuntimeError,match='real failure'):
        probe.receive(SimpleNamespace(poll=lambda _:True,recv=lambda:dict(error='real failure')))


def test_prepare_freezes_source_and_config_and_never_enqueues(tmp_path):
    root=Path(__file__).resolve().parents[2]
    path=root/'scripts/2026-09-24_prepare_dynamo_stationary_primitive.py'
    spec=importlib.util.spec_from_file_location('stationary_prepare_cpu_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    result=module.prepare(tmp_path/'new-only');assert not result['enqueued'] and not result['gpu_executed']
    jobs=json.loads(Path(result['jobs_path']).read_text());job=jobs[0];payload=job['payload']
    assert len(jobs)==1 and payload['gpu_count']==1 and not payload['exclusive']
    assert payload['timeout_s']==180 and payload['model_loads']==0
    assert not payload['formal_eligible'] and payload['prepare_only']
    assert payload['required_receipts']==['stationary-primitive/completion.json','stationary-primitive/gpu-after.json']
    assert '{attempt_dir}:{attempt_dir}:rw' in payload['argv']
    assert not any('/models' in arg for arg in payload['argv'])
    source=Path(payload['source_snapshot']);config=Path(payload['config_path'])
    env=dict(os.environ,PYTHONPATH=str(source),PYTHONDONTWRITEBYTECODE='1',PDBLEND_IMAGE_ID=module.IMAGE,
             PDBLEND_PRIMITIVE_CONFIG_SHA256=payload['config_sha256'])
    command=[sys.executable,'-B','-m','pdblend_baselines.dynamollm.stationary_probe','--config',str(config),'--preflight']
    # Configuration replacement must fail before importing Torch/CUDA.
    original=config.read_bytes();config.write_bytes(original+b' ')
    bad=subprocess.run(command,env=env,text=True,capture_output=True,timeout=20)
    assert bad.returncode!=0 and 'immutable primitive configuration differs' in bad.stderr
    config.write_bytes(original)
    # Mutating a dependency, rather than the entrypoint, must also be caught.
    dependency=source/'pdblend_baselines/dynamollm/gpu_weights.py'
    dependency.write_bytes(dependency.read_bytes()+b'\n# changed\n')
    bad=subprocess.run(command,env=env,text=True,capture_output=True,timeout=20)
    assert bad.returncode!=0 and 'frozen primitive source differs' in bad.stderr
