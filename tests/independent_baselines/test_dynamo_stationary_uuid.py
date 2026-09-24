"""Full physical identity normalization; CPU only, never calls real CUDA."""
from types import SimpleNamespace as NS
import json
import sys

import pytest

from pdblend_baselines.dynamollm.stationary_ipc import (
    CudaUuidIdentityError, TorchCudaIpcCodec, canonical_gpu_uuid,
    cuda_uuid_observation, verify_cuda_uuid)

GPU='GPU-d5135a74-255a-e68a-157d-dd669925844c'
OTHER='GPU-d5135a74-255a-e68a-157d-dd669925844d'


class CuUuidOracle:
    def __init__(self, text=GPU[4:], octets=None):
        self.text=text
        self.bytes=list(bytes.fromhex(GPU[4:].replace('-',''))) if octets is None else octets
    def __str__(self): return self.text


@pytest.mark.parametrize('raw',[GPU,GPU[4:],GPU.upper().replace('GPU-','GPU-'),GPU.encode(),GPU[4:].encode()])
def test_full_uuid_text_variants_keep_all_128_bits(raw):
    assert canonical_gpu_uuid(raw)==GPU


@pytest.mark.parametrize('raw',[None,0,True,'0','L20','GPU-0',GPU+' ', ' '+GPU,
    GPU.replace('-',''), 'MIG-'+GPU[4:], 'gpu-'+GPU[4:], b'\xff', bytes(16),
    'GPU-00000000-0000-0000-0000-000000000000'])
def test_partial_ordinal_mig_zero_or_ambiguous_identity_rejected(raw):
    with pytest.raises((ValueError,UnicodeError)):canonical_gpu_uuid(raw)


def test_pinned_cuuuid_bytes_and_string_are_independently_consistent():
    receipt=verify_cuda_uuid(GPU,CuUuidOracle(),torch_uuid_type=CuUuidOracle,device_index=0)
    assert receipt['passed'] and receipt['observed']['canonical']==GPU
    assert receipt['observed']['raw_text']==GPU[4:]
    assert len(receipt['observed']['raw_bytes'])==16
    assert json.loads(json.dumps(receipt))==receipt


@pytest.mark.parametrize('value',[
    CuUuidOracle(OTHER[4:]), CuUuidOracle(octets=[0]*16), CuUuidOracle(octets=[1]*15),
    CuUuidOracle(octets=[True]*16), CuUuidOracle(octets=[256]*16),
    CuUuidOracle(octets=bytes(16)), CuUuidOracle('MIG-'+GPU[4:])])
def test_cuuuid_corruption_keeps_evidence_and_fails_closed(value):
    with pytest.raises(CudaUuidIdentityError) as caught:
        verify_cuda_uuid(GPU,value,torch_uuid_type=CuUuidOracle,device_index=0)
    evidence=caught.value.uuid_identity
    assert not evidence['passed'] and evidence['observed']['raw_text']==str(value)
    assert evidence['expected']['canonical']==GPU
    assert 'raw_type' in evidence['observed'] and json.loads(json.dumps(evidence))==evidence


def test_arbitrary_object_string_must_not_supply_a_gpu_identity():
    with pytest.raises(CudaUuidIdentityError,match='unsupported CUDA UUID object'):
        verify_cuda_uuid(GPU,CuUuidOracle(),device_index=0)


@pytest.mark.parametrize('expected,actual',[(GPU,OTHER),(GPU,None),(GPU[4:],GPU),(GPU,OTHER.encode())])
def test_mismatch_or_noncanonical_lease_rejected(expected,actual):
    with pytest.raises(CudaUuidIdentityError) as caught:
        verify_cuda_uuid(expected,actual,device_index=0)
    assert not caught.value.uuid_identity['passed']


@pytest.mark.parametrize('actual,passed',[(CuUuidOracle(),True),(GPU[4:],True),(OTHER,False),(None,False)])
def test_codec_device_binding_happens_only_after_full_uuid_match(monkeypatch,actual,passed):
    calls=[]
    torch=NS(__version__='2.7.1+cu126',_C=NS(_CUuuid=CuUuidOracle),cuda=NS(
        get_device_properties=lambda device:NS(uuid=actual),set_device=lambda device:calls.append(device)))
    monkeypatch.setitem(sys.modules,'torch',torch)
    if passed:
        codec=TorchCudaIpcCodec(gpu_uuid=GPU,device_index=0)
        assert codec.uuid==GPU and codec.uuid_identity['passed'] and calls==[0]
    else:
        with pytest.raises(CudaUuidIdentityError):TorchCudaIpcCodec(gpu_uuid=GPU,device_index=0)
        assert calls==[]


def test_failure_pipe_retains_original_uuid_receipt_before_raising():
    from pdblend_baselines.dynamollm.stationary_probe import receive,RemotePrimitiveError,failure_event
    try:verify_cuda_uuid(GPU,OTHER,device_index=0)
    except CudaUuidIdentityError as error: event=failure_event(error)
    records=[]
    with pytest.raises(RemotePrimitiveError) as caught:
        receive(NS(poll=lambda _:True,recv=lambda:event),events=records)
    assert records==[event] and caught.value.uuid_identity==event['uuid_identity']
    assert caught.value.uuid_identity['observed']['canonical']==OTHER


def test_fixed_torch_exposes_pinned_cuuuid_abi_without_cuda_initialization():
    import torch
    before=torch.cuda.is_initialized()
    cls=torch._C._CUuuid
    assert cls.__module__=='torch._C' and cls.__name__=='_CUuuid'
    assert isinstance(cls.bytes,property) and callable(cls.__str__)
    assert torch.cuda.is_initialized()==before is False


def test_kv_describe_is_json_serializable_and_returns_the_observed_uuid(monkeypatch):
    from pdblend_baselines.dynamollm.stationary_kv_worker import DynamoStationaryKvWorkerExtension
    from pdblend_baselines.dynamollm.stationary_worker import DynamoStationaryWorkerExtension
    monkeypatch.setattr(DynamoStationaryWorkerExtension,'dynamo_stationary_operation',
                        lambda *_:dict(rank=0,generation=0))
    torch=NS(_C=NS(_CUuuid=CuUuidOracle),cuda=NS(current_device=lambda:0,
        get_device_properties=lambda _:NS(uuid=CuUuidOracle())))
    monkeypatch.setitem(sys.modules,'torch',torch)
    result=DynamoStationaryKvWorkerExtension().dynamo_stationary_operation('describe',dict(expected_generation=0))
    assert result['gpu_uuid']==GPU and result['cuda_uuid_observation']['raw_text']==GPU[4:]
    assert json.loads(json.dumps(result))==result
    torch.cuda.get_device_properties=lambda _:NS(uuid=OTHER)
    result=DynamoStationaryKvWorkerExtension().dynamo_stationary_operation('describe',dict(expected_generation=0))
    assert result['gpu_uuid']==OTHER  # service compares this independently to actual lease; never substitutes expected
