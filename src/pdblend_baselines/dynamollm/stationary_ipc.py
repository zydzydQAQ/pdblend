"""Dynamo-only CUDA IPC views with explicit producer/consumer lifetime.

This maps existing storage on the SAME physical GPU. It neither assembles a
target weight, copies a retained fragment, frees source KV, nor activates vLLM.
The source process remains a storage owner until every consumer has exited.
"""
from __future__ import annotations

import base64
from copy import deepcopy
import hashlib
import inspect
import json
import math
import os
import re
from pathlib import Path
import time

from .stationary_tensors import RetainedStorageLease, _tensor_identity


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def need(condition,message):
    if not condition:raise ValueError(message)


class CudaUuidIdentityError(ValueError):
    def __init__(self, receipt):
        self.uuid_identity = deepcopy(receipt)
        super().__init__('CUDA physical UUID differs from stationary lease or has invalid representation: '
                         +json.dumps(receipt,sort_keys=True,allow_nan=False))


def canonical_gpu_uuid(text):
    """Only complete 128-bit NVML / CUDA UUID text; no names, ordinals or MIG aliases."""
    if type(text) is bytes:
        text=text.decode('ascii',errors='strict')
    need(type(text) is str and re.fullmatch(
        r'(?:GPU-)?[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}',text),
        'complete physical GPU UUID text required')
    bare=text[4:] if text.startswith('GPU-') else text
    need(int(bare.replace('-',''),16)!=0,'zero UUID is not physical hardware identity')
    return 'GPU-'+bare.lower()


def cuda_uuid_observation(value, *, torch_uuid_type=None, _receipt=None):
    """Parse pinned Torch _CUuuid using BOTH its 16 bytes and text, not arbitrary str()."""
    result={} if _receipt is None else _receipt
    result.update(raw_type=type(value).__module__+'.'+type(value).__qualname__,raw_repr=repr(value)[:512])
    if type(value) in (str,bytes):
        result['raw_text']=value.decode('ascii',errors='strict') if type(value) is bytes else value
        result['canonical']=canonical_gpu_uuid(value)
    else:
        need(torch_uuid_type is not None and type(value) is torch_uuid_type,
             'unsupported CUDA UUID object type')
        octets=value.bytes
        result['raw_text']=str(value)
        if type(octets) is list and len(octets)<=16 and all(type(v) in (int,bool) for v in octets):
            result['raw_bytes']=octets
        need(type(octets) is list and len(octets)==16
             and all(type(v) is int and 0<=v<=255 for v in octets),'invalid Torch CUuuid bytes')
        canonical=canonical_gpu_uuid(result['raw_text'])
        need(bytes(octets).hex()==canonical[4:].replace('-',''),
             'Torch CUuuid text and original 16 bytes disagree')
        result['canonical']=canonical
    return result


def verify_cuda_uuid(expected, observed, *, torch_uuid_type=None, device_index):
    receipt=dict(schema='dynamo-cuda-uuid-identity/v1',device_index=device_index,
        expected=dict(raw_type=type(expected).__module__+'.'+type(expected).__qualname__,raw_repr=repr(expected)[:512]),
        observed=dict(raw_type=type(observed).__module__+'.'+type(observed).__qualname__,raw_repr=repr(observed)[:512]),
        passed=False,comparison='all_128_bits_no_ordinal_or_name_fallback')
    try:
        cuda_uuid_observation(expected,_receipt=receipt['expected'])
        need(type(expected) is str and expected==receipt['expected']['canonical'],
             'lease UUID must use canonical NVML GPU-prefixed text')
        cuda_uuid_observation(observed,torch_uuid_type=torch_uuid_type,_receipt=receipt['observed'])
        need(receipt['expected']['canonical']==receipt['observed']['canonical'],
             'actual CUDA UUID differs from leased physical UUID')
        receipt['passed']=True
    except (ValueError,TypeError,AttributeError) as error:
        receipt['error']=str(error)
        raise CudaUuidIdentityError(receipt) from error
    return receipt


def process_identity(pid=None):
    pid=os.getpid() if pid is None else pid
    need(type(pid) is int and pid>0,'positive process identity required')
    raw=Path('/proc',str(pid),'stat').read_text()
    fields=raw[raw.rfind(')')+2:].split()
    return dict(pid=pid,start_ticks=int(fields[19]),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())


def process_matches(identity):
    try:return process_identity(identity['pid'])==identity
    except (OSError,ValueError,KeyError,TypeError):return False


def _encode(value):
    return None if value is None else base64.b64encode(bytes(value)).decode('ascii')


def _decode(value):
    return None if value is None else base64.b64decode(value,validate=True)


def validate_descriptor(descriptor,*,gpu_uuid,piece=None,source=None):
    """Check byte bounds before handing an IPC descriptor to CUDA."""
    need(descriptor.get('schema')=='dynamo-cuda-ipc-descriptor/v1'
         and descriptor.get('gpu_uuid')==gpu_uuid and descriptor.get('dtype')=='bfloat16',
         'same physical GPU CUDA IPC required; peer-device imports forbidden')
    shape,stride=descriptor.get('shape'),descriptor.get('stride')
    need(isinstance(shape,list) and isinstance(stride,list) and len(shape)==len(stride)>0
         and all(type(v) is int and v>0 for v in shape)
         and all(type(v) is int and v>=0 for v in stride),'invalid CUDA IPC view dimensions')
    fields=('tensor_offset','storage_size_bytes','storage_offset_bytes','allocation_bytes','logical_bytes')
    need(all(type(descriptor.get(k)) is int and descriptor[k]>=0 for k in fields),
         'invalid CUDA IPC allocation bounds')
    last=descriptor['tensor_offset']+sum((n-1)*s for n,s in zip(shape,stride))
    need((last+1)*2<=descriptor['storage_size_bytes']
         and descriptor['storage_offset_bytes']+descriptor['storage_size_bytes']<=descriptor['allocation_bytes']
         and descriptor['logical_bytes']==math.prod(shape)*2,'CUDA IPC view exceeds original allocation')
    if piece is not None:
        expected=list(piece['source_shape'])
        if piece['axis'] is not None:expected[piece['axis']]=piece['length']
        need(shape==expected and descriptor['logical_bytes']==piece['bytes'],
             'CUDA IPC view differs from planned original fragment')
        if source is not None:
            offset=source['storage_offset']
            if piece['axis'] is not None:offset+=piece['source_offset']*source['stride'][piece['axis']]
            need(source['shape']==piece['source_shape'] and stride==source['stride']
                 and descriptor['tensor_offset']==offset
                 and descriptor['source_storage_ptr']==source['storage_ptr'],
                 'CUDA IPC view does not map the original source shard')
    return descriptor


def cuda_reducer_abi():
    """Read the pinned Python ABI without initializing CUDA."""
    import torch
    from torch.multiprocessing.reductions import rebuild_cuda_tensor
    names=list(inspect.signature(rebuild_cuda_tensor).parameters)
    expected=['tensor_cls','tensor_size','tensor_stride','tensor_offset','storage_cls','dtype',
              'storage_device','storage_handle','storage_size_bytes','storage_offset_bytes',
              'requires_grad','ref_counter_handle','ref_counter_offset','event_handle','event_sync_required']
    need(torch.__version__.split('+')[0].startswith('2.7.') and names==expected,
         'pinned PyTorch 2.7 CUDA IPC rebuild ABI differs')
    return dict(torch_version=torch.__version__,cuda_build=torch.version.cuda,rebuild_parameters=names,
                cuda_initialized=torch.cuda.is_initialized(),hardware_verified=False)


class TorchCudaIpcCodec:
    """Pinned PyTorch 2.7 CUDA reducer; JSON contains handles, never weights."""
    def __init__(self,*,gpu_uuid,device_index):
        import torch
        need(torch.__version__.split('+')[0].startswith('2.7.'),'pinned PyTorch 2.7 CUDA IPC ABI required')
        self.torch=torch;self.uuid=gpu_uuid;self.device=device_index
        props=torch.cuda.get_device_properties(device_index)
        actual=getattr(props,'uuid',None)
        self.uuid_identity=verify_cuda_uuid(gpu_uuid,actual,device_index=device_index,
            torch_uuid_type=getattr(torch._C,'_CUuuid',None))
        torch.cuda.set_device(device_index)

    def synchronize(self):self.torch.cuda.synchronize(self.device)

    def export_view(self,tensor):
        from torch.multiprocessing.reductions import reduce_tensor,rebuild_cuda_tensor
        torch=self.torch
        need(tensor.is_cuda and tensor.device.index==self.device and tensor.dtype==torch.bfloat16,
             'only local CUDA BF16 source views may be exported')
        view=tensor.detach()
        need(view.untyped_storage().data_ptr()==tensor.untyped_storage().data_ptr(),'detach changed storage')
        rebuild,args=reduce_tensor(view)
        need(rebuild is rebuild_cuda_tensor and len(args)==15,'CUDA IPC reducer ABI differs')
        (_,shape,stride,offset,_,dtype,device,handle,nbytes,storage_offset,requires_grad,
         ref_handle,ref_offset,event_handle,event_sync)=args
        need(dtype==torch.bfloat16 and not requires_grad and handle is not None,'invalid CUDA IPC tensor descriptor')
        # An exported slice pins the underlying cudaMalloc allocation. Count
        # that allocation separately from logical bytes and tensor storage.
        base=tensor.untyped_storage().data_ptr()-storage_offset
        allocations=[s for s in torch.cuda.memory_snapshot() if s.get('device')==self.device
                     and s.get('address')==base]
        need(len(allocations)==1,'allocator cannot identify retained cudaMalloc allocation; no budget claim allowed')
        return dict(schema='dynamo-cuda-ipc-descriptor/v1',gpu_uuid=self.uuid,
            shape=list(shape),stride=list(stride),tensor_offset=offset,dtype='bfloat16',
            handle=_encode(handle),storage_size_bytes=nbytes,storage_offset_bytes=storage_offset,
            ref_counter_handle=_encode(ref_handle),ref_counter_offset=ref_offset,
            event_handle=_encode(event_handle),event_sync_required=event_sync,
            source_device_index=device,allocation_bytes=allocations[0]['total_size'],
            allocation_identity=digest(dict(gpu_uuid=self.uuid,handle=_encode(handle))),
            source_storage_ptr=tensor.untyped_storage().data_ptr(),logical_bytes=tensor.numel()*2)

    def import_view(self,descriptor):
        from torch.multiprocessing.reductions import rebuild_cuda_tensor
        torch=self.torch
        validate_descriptor(descriptor,gpu_uuid=self.uuid)
        tensor=rebuild_cuda_tensor(torch.Tensor,tuple(descriptor['shape']),tuple(descriptor['stride']),
            descriptor['tensor_offset'],torch.storage.TypedStorage,torch.bfloat16,self.device,
            _decode(descriptor['handle']),descriptor['storage_size_bytes'],descriptor['storage_offset_bytes'],
            False,_decode(descriptor['ref_counter_handle']),descriptor['ref_counter_offset'],
            _decode(descriptor['event_handle']),descriptor['event_sync_required'])
        need(tensor.is_cuda and tensor.device.index==self.device and list(tensor.shape)==descriptor['shape']
             and list(tensor.stride())==descriptor['stride'] and tensor.storage_offset()==descriptor['tensor_offset'],
             'reconstructed IPC view geometry differs')
        return tensor

    def collect(self):
        import gc
        gc.collect()
        self.torch.cuda.ipc_collect()


class StationaryOwner:
    """Export each retained view once per consumer; never trust a timeout ACK."""
    def __init__(self,plan,*,source_rank,parameters,gpu_uuid,generation,codec,
                 owner_identity=None,process_alive=process_matches):
        need(codec.uuid==gpu_uuid,'owner codec UUID differs')
        self.plan=deepcopy(plan);self.codec=codec;self.alive=process_alive
        self.identity=process_identity() if owner_identity is None else deepcopy(owner_identity)
        self.lease=RetainedStorageLease(plan,source_rank=source_rank,parameters=parameters,
            gpu_uuid=gpu_uuid,generation=generation,expected_generation=generation)
        self.exports={};self.closed=False;self.quarantined=False

    def export(self,consumer,*,target_rank):
        need(not self.closed and not self.quarantined,'stationary owner is not exportable')
        need(consumer!=self.identity and self.alive(consumer),'a distinct live owned consumer is required')
        need(type(target_rank) is int and 0<=target_rank<len(self.plan['target_gpu_uuids'])
             and self.plan['target_gpu_uuids'][target_rank]==self.codec.uuid,
             'consumer target lies on another physical GPU')
        key=digest(dict(consumer=consumer,target_rank=target_rank))
        need(key not in self.exports,'consumer export already issued; CUDA refcounts must not be duplicated')
        before=self.lease.receipt();self.codec.synchronize()
        packet=dict(schema='dynamo-stationary-ipc-export/v1',owner=self.identity,consumer=deepcopy(consumer),
            plan_sha256=self.plan['plan_sha256'],source_rank=self.lease.source_rank,target_rank=target_rank,
            generation=self.lease.generation,gpu_uuid=self.codec.uuid,export_id=key,
            source_storage=before['source_storage'],views=[])
        # Keep a partial export pinned if a reducer fails after incrementing an
        # IPC refcounter; recovery then requires confirmed consumer isolation.
        self.exports[key]=dict(packet=packet,release_ack=None,issued_s=time.time(),complete=False)
        try:
            for piece,view in self.lease.views:
                if piece['target_rank']==target_rank:
                    descriptor=self.codec.export_view(view)
                    validate_descriptor(descriptor,gpu_uuid=self.codec.uuid,piece=piece,
                                        source=packet['source_storage'][piece['parameter']])
                    packet['views'].append(dict(piece=piece,descriptor=descriptor))
            need(packet['views'],'no original same-device fragments exist for consumer')
            packet['packet_sha256']=digest(packet)
            self.exports[key]['complete']=True
            return deepcopy(packet)
        except BaseException:
            self.quarantined=True
            raise

    def acknowledge_release(self,receipt):
        key=receipt.get('export_id');state=self.exports.get(key)
        need(state is not None and state['complete'],'unknown or incomplete export release')
        packet=state['packet']
        need(receipt.get('packet_sha256')==packet['packet_sha256'] and receipt.get('consumer')==packet['consumer']
             and receipt.get('gpu_uuid')==packet['gpu_uuid'] and receipt.get('cuda_synchronized') is True
             and receipt.get('views_released') is True,'consumer release ACK identity/synchronization differs')
        state['release_ack']=deepcopy(receipt)

    def release(self,*,consumer_processes_gone):
        need(not self.closed,'stationary owner already released')
        expected={digest(s['packet']['consumer']) for s in self.exports.values()}
        actual={digest(p) for p in consumer_processes_gone}
        need(actual==expected and all(not self.alive(p) for p in consumer_processes_gone),
             'source storage owner must outlive every consumer; PID reuse/liveness is checked')
        clean=all(s['release_ack'] is not None for s in self.exports.values())
        self.codec.synchronize();before=self.lease.receipt()
        retained_allocations={v['descriptor']['allocation_identity']:v['descriptor']['allocation_bytes']
            for s in self.exports.values() for v in s['packet']['views']}
        self.lease.release();self.closed=True
        # A killed consumer can leak its producer-side refcount. Do not turn
        # process isolation into a claim that the producer allocation is freed.
        self.quarantined=not clean
        return dict(released=True,owner=self.identity,consumer_processes_gone=deepcopy(consumer_processes_gone),
            clean_consumer_release=clean,owner_exit_required=not clean,
            original_storage_preserved=before,logical_retained_bytes=before['held_view_bytes'],
            pinned_cuda_allocations_bytes=sum(retained_allocations.values()),
            source_gpu_memory_reclaimed=False,target_engine_activated=False,formal_eligible=False)


class StationaryConsumer:
    """An inactive consumer owns imported views, not a routable native engine."""
    def __init__(self,packet,*,plan,codec,expected_generation,consumer_identity=None,process_alive=process_matches):
        self.identity=process_identity() if consumer_identity is None else deepcopy(consumer_identity)
        need(packet['packet_sha256']==digest({k:v for k,v in packet.items() if k!='packet_sha256'}),
             'stationary IPC packet changed')
        need(packet['consumer']==self.identity and process_alive(packet['owner']), 'IPC process owner/consumer differs')
        need(plan['plan_sha256']==digest({k:v for k,v in plan.items() if k!='plan_sha256'}),
             'stationary tensor plan changed')
        need(type(expected_generation) is int and packet['generation']==expected_generation,
             'stationary IPC generation differs')
        need(packet['plan_sha256']==plan['plan_sha256'] and packet['gpu_uuid']==codec.uuid
             and plan['source_gpu_uuids'][packet['source_rank']]==codec.uuid
             and plan['target_gpu_uuids'][packet['target_rank']]==codec.uuid,
             'IPC plan or target physical UUID differs')
        expected=[p for p in plan['pieces'] if p['source_rank']==packet['source_rank']
                  and p['target_rank']==packet['target_rank'] and p['kind']=='retain_on_gpu']
        need([v['piece'] for v in packet['views']]==expected,'export omitted or changed original fragments')
        self.packet=deepcopy(packet);self.codec=codec;self.alive=process_alive;self.views=[];self.closed=False
        try:
            for row in packet['views']:
                validate_descriptor(row['descriptor'],gpu_uuid=codec.uuid,piece=row['piece'],
                                    source=packet['source_storage'][row['piece']['parameter']])
                self.views.append((deepcopy(row['piece']),codec.import_view(row['descriptor'])))
            codec.synchronize()
        except BaseException:
            self.views.clear();codec.collect()
            raise

    def require_inactive(self):
        raise RuntimeError('stationary IPC views do not supply target TP runner, KV release/budget, or ordinary-output qualification')

    def receipt(self):
        need(not self.closed and self.alive(self.packet['owner']),'CUDA storage owner is absent; target must be isolated')
        return dict(export_id=self.packet['export_id'],packet_sha256=self.packet['packet_sha256'],
            owner=self.packet['owner'],consumer=self.identity,gpu_uuid=self.codec.uuid,
            views=[dict(piece=p,tensor=_tensor_identity(v)) for p,v in self.views],
            tensor_data_copied_bytes=0,host_weight_staging_bytes=0,
            source_storage_owner_must_remain_alive=True,target_engine_activated=False,formal_eligible=False)

    def close(self):
        need(not self.closed,'stationary consumer already closed')
        self.codec.synchronize();self.views.clear();self.codec.collect();self.closed=True
        return dict(export_id=self.packet['export_id'],packet_sha256=self.packet['packet_sha256'],
                    consumer=self.identity,gpu_uuid=self.codec.uuid,cuda_synchronized=True,views_released=True)
