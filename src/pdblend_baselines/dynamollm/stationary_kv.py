"""Dynamo-only, eager Qwen KV backing lifetime while original weights stay pinned.

This does not initialize or change a TP process group, copy model weights, or
activate a target. Native admission must remain closed for the transaction.
The source allocator's free blocks are not credited as driver free memory.
"""
from __future__ import annotations

from copy import deepcopy
from contextlib import nullcontext
from dataclasses import fields, is_dataclass
import gc
import time
import weakref

import torch

from .native_state import validate_state
from .stationary_ipc import TorchCudaIpcCodec, digest, need
from .stationary_tensors import _tensor_identity


def _config_value(value):
    if is_dataclass(value):
        return {f.name: _config_value(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, dict):
        return {str(k): _config_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_config_value(v) for v in value]
    if isinstance(value, torch.dtype):
        return str(value)
    need(value is None or type(value) in (bool, int, float, str),
         'KV configuration must contain metadata only, never tensor storage')
    return value


def allocation_blocks(snapshot, device):
    """Normalize the pinned Torch allocator snapshot without inferred bytes."""
    result = []
    for segment in snapshot:
        if segment['device'] != device:
            continue
        cursor = segment['address']
        for block in segment['blocks']:
            address = block.get('address', cursor)
            need(address == cursor and block['size'] > 0, 'allocator block boundaries differ')
            result.append(dict(address=address, size=block['size'], state=block['state'],
                segment_address=segment['address'], segment_bytes=segment['total_size']))
            cursor += block['size']
        need(cursor == segment['address'] + segment['total_size'], 'allocator segment coverage differs')
    return result


def _overlaps(address, size, other_address, other_size):
    return address < other_address + other_size and other_address < address + size


class TorchKvBackend:
    is_cpu_oracle = False

    def __init__(self, codec):
        need(isinstance(codec, TorchCudaIpcCodec), 'actual UUID-bound CUDA codec required')
        self.codec = codec

    def synchronize(self):
        self.codec.synchronize()

    def collect(self):
        # No source weight offload, allocator sleep, or IPC refcount collection.
        self.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        self.synchronize()

    def observe(self):
        import pynvml as nv
        free, total = torch.cuda.mem_get_info(self.codec.device)
        nv.nvmlInit()
        try:
            memory = nv.nvmlDeviceGetMemoryInfo(nv.nvmlDeviceGetHandleByUUID(self.codec.uuid))
            board = dict(at_s=time.time(), gpu_uuid=self.codec.uuid, free_bytes=int(memory.free),
                         used_bytes=int(memory.used), total_bytes=int(memory.total), source='NVML_device_memory_info')
        finally:
            nv.nvmlShutdown()
        return dict(at_s=time.time(), gpu_uuid=self.codec.uuid, device_index=self.codec.device,
            free_bytes=free, total_bytes=total,
            nvml_board_memory=board,
            allocated_bytes=torch.cuda.memory_allocated(self.codec.device),
            reserved_bytes=torch.cuda.memory_reserved(self.codec.device),
            blocks=allocation_blocks(torch.cuda.memory_snapshot(), self.codec.device))


class SourceKvWorkspace:
    """Detach both vLLM KV owners, verify allocation state, rebuild exact config.

    A failure after detachment keeps execution blocked and requires process
    isolation. Source weights and owner IPC exports are never freed here.
    """

    @classmethod
    def for_native_worker(cls, worker, owner):
        import vllm
        from vllm.distributed.kv_transfer import has_kv_transfer_group
        need(vllm.__version__ == '0.10.1.1', 'unreviewed native KV runner revision')
        need(not has_kv_transfer_group(), 'registered KV connector references cannot be detached')
        need(type(worker.model_runner).__name__ == 'GPUModelRunner'
             and type(worker.model_runner).__module__ == 'vllm.v1.worker.gpu_model_runner',
             'only pinned V1 GPUModelRunner is supported')
        return cls(worker, owner, TorchKvBackend(owner.codec), cpu_oracle=False)

    @classmethod
    def cpu_oracle(cls, worker, owner, backend):
        need(backend.is_cpu_oracle is True, 'explicit CPU oracle backend required')
        return cls(worker, owner, backend, cpu_oracle=True)

    def __init__(self, worker, owner, backend, *, cpu_oracle):
        self.worker, self.owner, self.backend = worker, owner, backend
        self.runner = worker.model_runner
        self.is_cpu_oracle, self.status = cpu_oracle, 'resident'
        self.generation = owner.lease.generation
        self.config = deepcopy(self.runner.kv_cache_config)
        self.config_sha256 = digest(_config_value(self.config))
        self._methods, self._model_guard = [], None
        self.observations = {}
        self._validate_supported()
        self.weight_identity = {n: _tensor_identity(t) for n, t in self.runner.model.named_parameters()}
        self.weight_storage_bytes = {n: t.untyped_storage().nbytes()
                                     for n, t in self.runner.model.named_parameters()}
        need(self.weight_identity, 'source must own actual original model parameters')
        self._check_weights()

    def _validate_supported(self):
        w, r = self.worker, self.runner
        need(w.model_config.enforce_eager is True and not w.model_config.enable_sleep_mode,
             'KV detach requires eager mode without tagged sleep allocator')
        need(w.model_config.hf_config.model_type == 'qwen2'
             and w.parallel_config.pipeline_parallel_size == 1
             and getattr(w.parallel_config, 'data_parallel_size', 1) == 1,
             'KV detach supports Qwen2 PP1 DP1 only')
        need(w.parallel_config.tensor_parallel_size == len(self.owner.plan['source_gpu_uuids'])
             and w.rank == self.owner.lease.source_rank
             and self.owner.plan['source_gpu_uuids'][w.rank] == self.owner.codec.uuid,
             'source owner physical rank/TP group differs')
        need(w.vllm_config.kv_transfer_config is None and w.vllm_config.speculative_config is None
             and w.vllm_config.lora_config is None and not r.shared_kv_cache_layers,
             'connector/speculative/LoRA/shared KV lifetime is not supported')
        need(not getattr(w, '_dynamo_weight_sessions', {}) and not getattr(w, '_native_scope', None)
             and not getattr(w, '_native_pending', []), 'concurrent weight transport or measurement is active')
        need(not getattr(r, 'encoder_cache', {}), 'encoder tensor lifetime is not supported')
        need(digest(_config_value(r.kv_cache_config)) == self.config_sha256, 'source KV configuration changed')
        need(all(type(g.kv_cache_spec).__name__ == 'FullAttentionSpec' for g in self.config.kv_cache_groups)
             and self.config.kv_cache_groups
             and all(len(t.shared_by) == 1 for t in self.config.kv_cache_tensors),
             'only unshared full-attention KV backing can be detached')
        need(not self.owner.closed and not self.owner.quarantined, 'original storage owner is unavailable')

    def _check_weights(self):
        self.owner.lease.receipt()
        current = {n: _tensor_identity(t) for n, t in self.runner.model.named_parameters()}
        need(current == self.weight_identity, 'source original parameter storage/version changed')

    def _drain(self, receipt):
        need(getattr(self.worker, '_native_generation', self.generation) == self.generation,
             'source native generation changed during KV transaction')
        validate_state(receipt, generation=self.generation,
            tp=self.worker.parallel_config.tensor_parallel_size, pp=1, drained=True,
            observed_after_s=time.time() - .5)
        need(receipt.get('accepting') is False and receipt.get('acknowledged') is True
             and receipt.get('drained') is True, 'source admission must remain closed and drained')
        self._validate_supported()
        self._check_weights()

    def _inventory(self):
        r = self.runner
        names = [name for g in self.config.kv_cache_groups for name in g.layer_names]
        need(len(names) == len(set(names)) and len(r.kv_caches) == len(names),
             'source KV layer inventory differs')
        rows, tensors = {}, []
        for name in names:
            layer = r.compilation_config.static_forward_context[name]
            need(isinstance(layer.kv_cache, list) and len(layer.kv_cache) == 1,
                 'source attention KV binding differs')
            tensor = layer.kv_cache[0]
            need(isinstance(tensor, torch.Tensor) and tensor.numel() > 0,
                 'source KV backing must be a real tensor')
            need(self.is_cpu_oracle or tensor.is_cuda and tensor.device.index == self.owner.codec.device,
                 'source KV belongs to another physical device')
            rows[name] = dict(shape=list(tensor.shape), stride=list(tensor.stride()), dtype=str(tensor.dtype),
                storage_ptr=tensor.untyped_storage().data_ptr(), storage_bytes=tensor.untyped_storage().nbytes())
            planned = [item.size for item in self.config.kv_cache_tensors if item.shared_by == [name]]
            need(planned == [rows[name]['storage_bytes']], 'KV backing differs from exact native configuration bytes')
            tensors.append(tensor)
        need(sorted(id(t) for t in tensors) == sorted(id(t) for t in r.kv_caches),
             'attention and runner do not own the exact same KV tensor inventory')
        need(len({(x['storage_ptr'], x['storage_bytes']) for x in rows.values()}) == len(rows),
             'KV layers share a backing allocation view')
        for row in rows.values():
            for parameter in r.model.parameters():
                need(not _overlaps(row['storage_ptr'], row['storage_bytes'],
                     parameter.untyped_storage().data_ptr(), parameter.untyped_storage().nbytes()),
                     'KV and retained parameter storage overlap')
        return rows, [weakref.ref(t) for t in tensors]

    def _install_guards(self):
        def guard(*args, **kwargs):
            need(self.status in ('resident', 'restored'), 'source execution blocked while KV backing is absent/uncertain')
        for name in ('execute_model', 'profile_run', 'capture_model', '_dummy_run'):
            if hasattr(self.runner, name):
                previous = getattr(self.runner, name)
                own = name in vars(self.runner)
                def wrapper(*args, _previous=previous, **kwargs):
                    guard()
                    return _previous(*args, **kwargs)
                self._methods.append((name, previous, own, wrapper))
                setattr(self.runner, name, wrapper)
        self._model_guard = self.runner.model.register_forward_pre_hook(lambda module, args: guard())

    def _clear_bindings(self):
        # Metadata/weakrefs only survive this call. Keeping the old tensor list
        # for rollback would prevent its CUDA backing from being released.
        for group in self.config.kv_cache_groups:
            for name in group.layer_names:
                self.runner.compilation_config.static_forward_context[name].kv_cache = []
        self.runner.kv_caches.clear()

    def release(self, native_scheduler_drain):
        need(self.status == 'resident', 'KV release is single-use')
        self._drain(native_scheduler_drain)
        self.backend.synchronize()
        self.original_kv, refs = self._inventory()
        before = self.backend.observe()
        for row in self.original_kv.values():
            matching = [b for b in before['blocks'] if b['state'] == 'active_allocated'
                and b['address'] <= row['storage_ptr']
                and b['address'] + b['size'] >= row['storage_ptr'] + row['storage_bytes']]
            need(len(matching) == 1, 'actual allocator cannot identify KV backing; release forbidden')
        self.observations['before_release'] = before
        self._install_guards()
        self.status = 'releasing'
        try:
            self._clear_bindings()
            self.backend.collect()
            need(all(r() is None for r in refs), 'another Python owner still references released KV tensors')
            after = self.backend.observe()
            self.observations['after_release'] = after
            need(all(not _overlaps(row['storage_ptr'], row['storage_bytes'], b['address'], b['size'])
                     for row in self.original_kv.values() for b in after['blocks']
                     if b['state'] != 'inactive'), 'old KV backing remains active or pending in CUDA allocator')
            self._check_weights()
            self.status = 'released'
        except BaseException as exc:
            self.status = 'failed'
            self.error = repr(exc)
            raise
        return self.receipt()

    def restore(self, native_scheduler_drain, *, guard_bytes=256 * 1024 * 1024):
        need(self.status == 'released', 'only a proven released KV workspace can be restored')
        self._drain(native_scheduler_drain)
        # Rollback cannot race a target still using this owner's allocations.
        need(all(row['release_ack'] is not None and not self.owner.alive(row['packet']['consumer'])
                 for row in self.owner.exports.values()), 'target consumers must cleanly release and exit before rollback')
        need(type(guard_bytes) is int and guard_bytes >= 0, 'explicit nonnegative KV restoration guard required')
        before = self.backend.observe()
        required = sum(t.size for t in self.config.kv_cache_tensors)
        self.observations['before_restore'] = before
        need(before['free_bytes'] >= required + guard_bytes,
             'actual driver free bytes cannot cover KV rebuild and guard; source KV release is not credited twice')
        self.status = 'restoring'
        try:
            if self.is_cpu_oracle:
                context = nullcontext()
            else:
                from vllm.config import set_current_vllm_config
                context = set_current_vllm_config(self.worker.vllm_config)
            with context:
                self.runner.initialize_kv_cache_tensors(deepcopy(self.config))
            self.backend.synchronize()
            restored, _ = self._inventory()
            need({n: {k: v for k, v in r.items() if k != 'storage_ptr'}
                  for n, r in restored.items()} ==
                 {n: {k: v for k, v in r.items() if k != 'storage_ptr'}
                  for n, r in self.original_kv.items()}, 'restored KV shape/stride/dtype differs')
            self._check_weights()
            self.observations['after_restore'] = self.backend.observe()
            self.status = 'restored'
        except BaseException as exc:
            self.status = 'failed'
            self.error = repr(exc)
            self._clear_bindings()
            raise
        return self.receipt()

    def receipt(self):
        return dict(schema='dynamo-source-kv-workspace/v1', status=self.status, cpu_oracle=self.is_cpu_oracle,
            gpu_uuid=self.owner.codec.uuid, generation=self.generation, config_sha256=self.config_sha256,
            source_owner_process=deepcopy(self.owner.identity), source_rank=self.owner.lease.source_rank,
            plan_sha256=self.owner.plan['plan_sha256'],
            kv_inventory=getattr(self, 'original_kv', None), observations=deepcopy(self.observations),
            original_weight_identity=self.weight_identity, original_weight_copy_bytes=0,
            original_weight_storage_bytes=self.weight_storage_bytes,
            host_weight_staging_bytes=0, driver_free_memory_credit_bytes=0,
            restoration_budget_scope='development_driver_free_plus_256MiB_guard_not_allocator_peak_qualification',
            source_execution_blocked=self.status not in ('resident', 'restored', 'closed'),
            source_admission_reopened=False, source_TP_group_reconfigured=False,
            target_TP_group_initialized=False, target_engine_activated=False,
            requires_process_isolation=self.status == 'failed', error=getattr(self, 'error', None),
            hardware_qualified=False, full_tp_switch_qualified=False, formal_eligible=False)

    def close(self):
        need(self.status == 'restored', 'cannot drop guards while KV backing is absent or uncertain')
        self._check_weights()
        for name, previous, own, wrapper in reversed(self._methods):
            need(getattr(self.runner, name) is wrapper, 'source runner execution guard was replaced')
            if own:
                setattr(self.runner, name, previous)
            else:
                delattr(self.runner, name)
        self._methods.clear()
        if self._model_guard is not None:
            self._model_guard.remove()
            self._model_guard = None
        self.status = 'closed'
