"""Opt-in same-TP real target worker, with unchanged native KV/forward methods."""
from __future__ import annotations

import os
import time
import torch

from pdblend_runtime.native_v1 import NativeWorker,NativeScheduler
from pdblend.online.native_control import validate_state
from .stationary_ipc import need,process_identity
from .stationary_kv import _config_value
from .stationary_kv_worker import DynamoStationaryKvWorkerExtension
from .stationary_target_bootstrap import TargetBootstrap,packet_from_worker_receipt
# Import registers the private format before the native runner asks for a loader.
from .stationary_target_loader import LOAD_FORMAT
from .stationary_tensors import _tensor_identity


class SameTpTargetScheduler(NativeScheduler):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.native_accepting=False
        self.native_admit_prefill=self.native_admit_decode=True
        self._native_event('private_stationary_target_only')


class SameTpTargetWorker(NativeWorker):
    def init_device(self):
        c=self.vllm_config
        need(c.load_config.load_format==LOAD_FORMAT
             and set(c.load_config.model_loader_extra_config)=={'bootstrap_ref'}, 'private stationary loader required')
        self._target_bootstrap=TargetBootstrap(c.load_config.model_loader_extra_config['bootstrap_ref'])
        b=self._target_bootstrap.config
        need(c.model_config.model==b['model_path'] and c.model_config.enforce_eager
             and not c.model_config.enable_sleep_mode and c.kv_transfer_config is None
             and c.parallel_config.tensor_parallel_size==c.parallel_config.pipeline_parallel_size==1
             and getattr(c.parallel_config,'data_parallel_size',1)==1,
             'private target launch must preserve bound same-TP eager/no-connector model')
        need(c.cache_config.gpu_memory_utilization==b['target_gpu_memory_utilization']
             and os.environ.get('PDBLEND_SOURCE_SHA256')==b['target_source_sha256']
             and os.environ.get('PDBLEND_IMAGE_ID')==b['image_digest'], 'private target source/image/native memory identity differs')
        # Keep the actual vLLM free-memory guard, independent process group,
        # runner construction and context allocation; never bypass .85 checks.
        super().init_device()
        self._target_native_executions=0;self._target_scheduled_tokens=0;self._target_kv_ready=False

    def load_model(self):
        # NativeWorker installs its normal runner observation wrapper after the
        # registered loader returns; no checkpoint/dummy loader is called.
        super().load_model()
        runtime=getattr(self.model_runner.model,'_dynamo_stationary_target_runtime',None)
        need(runtime is not None and not runtime.closed, 'native target did not use the original-storage loader')
        need(self._native_generation==runtime.bootstrap.config['target_generation'], 'target native worker epoch differs')
        self._target_runtime=runtime
        original=self.model_runner.execute_model
        def execute(scheduler_output,*args,**kwargs):
            need(not runtime.closed and self._target_kv_ready, 'target KV/binding is unavailable')
            runtime.inventory.check()
            value=original(scheduler_output,*args,**kwargs)
            if scheduler_output.total_num_scheduled_tokens:
                self._target_native_executions+=1
                self._target_scheduled_tokens+=scheduler_output.total_num_scheduled_tokens
            return value
        self.model_runner.execute_model=execute

    def determine_available_memory(self):
        # This remains development evidence. Other processes on this GPU must
        # keep their allocations fixed while the original native profiler runs.
        available=super().determine_available_memory()
        free,total=torch.cuda.mem_get_info()
        self._target_runtime.bootstrap.record('native-memory-profile',available_KV_bytes=available,
            actual_driver_free_bytes=free,actual_driver_total_bytes=total,
            native_requested_memory_bytes=self.requested_memory,
            model_load_observed_memory_bytes=self.model_runner.model_memory_usage,
            requires_other_process_allocations_stable=True,board_peak_qualified=False,formal_eligible=False)
        return available

    def initialize_from_config(self,kv_cache_config):
        super().initialize_from_config(kv_cache_config)
        need(self.model_runner.kv_caches and all(v.is_cuda and not v.is_meta and v.numel()>0
            for v in self.model_runner.kv_caches), 'actual native target KV buffers missing')
        self._target_kv_ready=True
        self._target_runtime.bootstrap.record('native-kv-ready',config=_config_value(kv_cache_config),
            tensors=[_tensor_identity(v) for v in self.model_runner.kv_caches],
            storage_bytes=[v.untyped_storage().nbytes() for v in self.model_runner.kv_caches],
            target_served=False,formal_eligible=False)

    def dynamo_target_operation(self,operation,payload=None):
        payload=dict(payload or {});runtime=self._target_runtime
        need(payload.get('transaction_id')==runtime.bootstrap.config['transaction_id']
             and payload.get('expected_generation')==self._native_generation, 'target transaction/epoch differs')
        if operation=='close':
            state=payload['native_scheduler_drain']
            validate_state(state,generation=self._native_generation,tp=1,pp=1,drained=True,
                           observed_after_s=time.time()-.5)
            need(state.get('acknowledged') is True and state.get('drained') is True
                 and state.get('accepting') is False, 'actual target drain must precede imported-view release')
            ack=runtime.close()
            return dict(rank=self.rank,generation=self._native_generation,process=process_identity(),
                gpu_uuid=runtime.codec.uuid,
                release_ack=ack,views_released=True,target_process_exit_required=True,formal_eligible=False)
        need(operation=='status','private target only supports status/close; public activation remains disabled')
        return dict(rank=self.rank,generation=self._native_generation,process=process_identity(),
            gpu_uuid=runtime.codec.uuid,KV_initialized=self._target_kv_ready,
            binding_closed=runtime.closed,native_execute_model_completed_calls=self._target_native_executions,
            native_scheduled_token_steps=self._target_scheduled_tokens,
            binding=runtime.binding.receipt() if not runtime.closed else None,
            target_public_admission=False,formal_eligible=False)


class TargetAwareSourceExtension(DynamoStationaryKvWorkerExtension):
    """New opt-in source extension; frozen source-only worker stays unchanged."""
    def dynamo_stationary_operation(self,operation,payload=None):
        row=super().dynamo_stationary_operation(operation,payload)
        if operation=='export' and row.get('participating') is not False:
            return dict(rank=self.rank,generation=row['generation'],transaction_id=row['transaction_id'],
                gpu_uuid=row['gpu_uuid'],packet=packet_from_worker_receipt(row),formal_eligible=False)
        if operation=='consumer_release_ack' and row.get('participating') is not False:
            row['gpu_uuid']=self._dynamo_stationary_owners[payload['transaction_id']].codec.uuid
        return row
