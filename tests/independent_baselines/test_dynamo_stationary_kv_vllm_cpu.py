"""Actual pinned vLLM KV allocation/reshape/bind ABI on CPU, no worker boot."""
from types import SimpleNamespace as NS
import weakref

import pytest
import torch

vllm = pytest.importorskip('vllm')

from pdblend_baselines.dynamollm.stationary_kv import SourceKvWorkspace
from tests.independent_baselines.test_dynamo_stationary_kv import setup, drained


@pytest.fixture(autouse=True)
def actual_cpu_config():
    from vllm.config import VllmConfig, DeviceConfig, set_current_vllm_config
    with set_current_vllm_config(VllmConfig(device_config=DeviceConfig(device='cpu'))):
        yield


def test_actual_vllm_kv_allocate_reshape_bind_detach_and_rebuild():
    assert vllm.__version__ == '0.10.1.1'
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend
    from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheTensor, KVCacheGroupSpec, FullAttentionSpec
    worker, owner, _ = setup()
    previous_runner = worker.model_runner
    runner = object.__new__(GPUModelRunner)  # Constructor/TP/CUDA worker boot explicitly out of scope.
    runner.model = previous_runner.model
    name = 'model.layers.0.self_attn.attn'
    spec = FullAttentionSpec(block_size=16, num_kv_heads=2, head_size=64, dtype=torch.bfloat16, use_mla=False)
    config = KVCacheConfig(num_blocks=2, kv_cache_tensors=[KVCacheTensor(size=2*spec.page_size_bytes,
        shared_by=[name])], kv_cache_groups=[KVCacheGroupSpec(layer_names=[name], kv_cache_spec=spec)])
    runner.kv_cache_config = config
    runner.compilation_config = NS(static_forward_context={name: NS(kv_cache=[])})
    runner.attn_groups = [[NS(backend=FlashAttentionBackend, layer_names=[name])]]
    runner.shared_kv_cache_layers = {}
    runner.encoder_cache = {}
    runner.kv_caches = []
    runner.device = torch.device('cpu')
    runner.initialize_kv_cache_tensors(config)
    assert runner.kv_caches[0] is runner.compilation_config.static_forward_context[name].kv_cache[0]
    assert runner.kv_caches[0].untyped_storage().nbytes() == 2*spec.page_size_bytes
    worker.model_runner = runner
    pointer = runner.kv_caches[0].untyped_storage().data_ptr()
    tensor_ref = weakref.ref(runner.kv_caches[0])
    class Backend:
        is_cpu_oracle = True
        def synchronize(self): pass
        def collect(self):
            import gc
            gc.collect()
        def observe(self):
            return dict(free_bytes=1024**3, cpu_oracle=True, blocks=[] if tensor_ref() is None else [
                dict(address=pointer, size=2*spec.page_size_bytes, state='active_allocated')])
    workspace = SourceKvWorkspace.cpu_oracle(worker, owner, Backend())
    workspace.release(drained())
    assert tensor_ref() is None and runner.kv_caches == []
    with pytest.raises(ValueError, match='execution blocked'): runner.execute_model(None)
    restored = workspace.restore(drained())
    assert restored['status'] == 'restored' and not restored['hardware_qualified']
    assert runner.kv_caches[0] is runner.compilation_config.static_forward_context[name].kv_cache[0]
    assert tuple(runner.kv_caches[0].shape) == FlashAttentionBackend.get_kv_cache_shape(2, 16, 2, 64)
    workspace.close()
    assert not torch.cuda.is_initialized()


def test_actual_native_worker_accepts_the_private_extension_mro_without_cuda_boot():
    from pdblend_runtime.native_v1 import NativeWorker
    from pdblend_baselines.dynamollm.stationary_kv_worker import DynamoStationaryKvWorkerExtension
    # This is the fixed WorkerWrapperBase's exact dynamic inheritance contract,
    # without calling its worker constructor or booting any CUDA context.
    original=NativeWorker.__bases__
    for name in dir(DynamoStationaryKvWorkerExtension):
        if not name.startswith('__'):assert not hasattr(NativeWorker,name),name
    try:
        NativeWorker.__bases__=original+(DynamoStationaryKvWorkerExtension,)
        assert NativeWorker.dynamo_stationary_operation is DynamoStationaryKvWorkerExtension.dynamo_stationary_operation
        assert hasattr(NativeWorker,'native_generation_set') and hasattr(NativeWorker,'dynamo_operation')
    finally:
        NativeWorker.__bases__=original
    assert not torch.cuda.is_initialized()
