"""Qwen2 BF16 weight shards transferred directly between unique GPU ranks.

The fixed execution bridge calls worker_operation between engine steps. No
checkpoint files or host weight tensors are created by this module. GPU-group
reuse and same-GPU multi-engine execution are deliberately not implied.
"""
from datetime import timedelta
import hashlib
import json
import math
import os
import time


class DynamoWorkerExtension:
    """vLLM 0.10.1.1 worker_extension_cls; no V0 engine monkey patch."""

    def dynamo_operation(self, operation, payload=None):
        payload = dict(payload or {})
        # NativeWorker can install a later generation on actual ranks. The
        # launch environment is only the bootstrap value, not a live ACK.
        generation = getattr(self, '_native_generation', int(os.environ.get('DYNAMO_GENERATION', '0')))
        if type(generation) is not int or generation < 0:
            raise ValueError('Dynamo worker generation is invalid')
        if (self.parallel_config.pipeline_parallel_size != 1 or
                getattr(self.parallel_config, 'data_parallel_size', 1) != 1):
            raise ValueError('Dynamo weight execution requires PP1 and DP1')
        expected = payload.get('expected_generation', generation)
        if type(expected) is not int or expected != generation:
            raise ValueError('Dynamo worker generation mismatch')
        if operation in ('open', 'transfer', 'close'):
            transaction = payload.get('transaction_id')
            if not isinstance(transaction, str) or not transaction or len(transaction) > 128:
                raise ValueError('bounded Dynamo transaction_id required')
        result = worker_operation(self, operation=operation, payload=payload)
        result.update(generation=generation, transaction_id=payload.get('transaction_id'),
                      engine_revision='vllm-0.10.1.1')
        return result


def _segments(name,shape,tp,rank,geometry):
    if tp not in (1,2,4,8) or not 0<=rank<tp:raise ValueError('invalid tensor parallel rank')
    if name.endswith(('input_layernorm.weight','post_attention_layernorm.weight','norm.weight')):
        return None,[(0,0,math.prod(shape))]
    if '.qkv_proj.' in name:
        h=geometry['hidden_size'];heads=geometry['num_attention_heads'];kv=geometry['num_key_value_heads']
        if heads%tp or kv%tp or h%heads:raise ValueError('unsupported Qwen GQA replication or TP geometry')
        sizes=(h,kv*(h//heads),kv*(h//heads));axis=0
    elif '.gate_up_proj.' in name:
        sizes=(geometry['intermediate_size'],)*2;axis=0
    elif '.o_proj.weight' in name:
        sizes=(geometry['hidden_size'],);axis=1
    elif '.down_proj.weight' in name:
        sizes=(geometry['intermediate_size'],);axis=1
    elif name.endswith(('embed_tokens.weight','lm_head.weight')):
        sizes=(shape[0]*tp,);axis=0
    else:raise ValueError('unqualified parameter layout: '+name)
    if any(size%tp for size in sizes) or shape[axis]!=sum(sizes)//tp:
        raise ValueError('parameter shape does not match Qwen partition geometry')
    result=[];local=global_start=0
    for size in sizes:
        width=size//tp
        result.append((local,global_start+rank*width,width));local+=width;global_start+=size
    return axis,result


def transfer_pieces(name,source_shape,target_shape,source_tp,source_rank,target_tp,target_rank,geometry):
    """Intersect canonical Q/K/V, gate/up, row or vocabulary shard intervals."""
    sa,source=_segments(name,source_shape,source_tp,source_rank,geometry)
    ta,target=_segments(name,target_shape,target_tp,target_rank,geometry)
    if sa!=ta or len(source_shape)!=len(target_shape):raise ValueError('source and target parameter axes differ')
    if sa is None:
        if tuple(source_shape)!=tuple(target_shape):raise ValueError('replicated weight shape changed')
        return [dict(axis=None,source_offset=0,target_offset=0,length=math.prod(source_shape))] if source_rank==0 else []
    if any(a!=b for index,(a,b) in enumerate(zip(source_shape,target_shape)) if index!=sa):
        raise ValueError('nonpartitioned weight dimensions changed')
    result=[]
    for sl,sg,size in source:
        for tl,tg,tsize in target:
            start=max(sg,tg);end=min(sg+size,tg+tsize)
            if start<end:result.append(dict(axis=sa,source_offset=sl+start-sg,target_offset=tl+start-tg,length=end-start))
    return result


def _metadata(worker):
    model=worker.model_runner.model
    parameters=dict(model.named_parameters())
    rows={name:list(p.shape) for name,p in sorted(parameters.items())}
    if any(str(p.dtype)!='torch.bfloat16' for p in parameters.values()):raise ValueError('only real BF16 parameters qualified')
    config=worker.model_config.hf_config
    geometry={key:int(getattr(config,key)) for key in
              ('hidden_size','num_attention_heads','num_key_value_heads','intermediate_size')}
    return parameters,rows,geometry


def worker_operation(worker,*,operation,payload):
    import torch
    import torch.distributed as distributed
    local_rank=worker.rank;tp=worker.parallel_config.tensor_parallel_size
    device=next(worker.model_runner.model.parameters()).device
    if operation == 'mark_ready':
        previous = getattr(worker, '_dynamo_target_transfers', {}).get(payload.get('operation_id'))
        if (payload.get('verified') is not True or not previous or not previous.get('target_complete')
                or payload.get('transaction_id') != previous.get('transaction_id')):
            raise ValueError('verified complete target transfer required before activation')
        worker._dynamo_weights_ready = True
        return dict(ok=True, rank=local_rank, weights_ready=True, operation_id=payload['operation_id'])
    sessions=getattr(worker,'_dynamo_weight_sessions',None)
    if sessions is None:sessions={};worker._dynamo_weight_sessions=sessions
    if operation=='drain':
        torch.cuda.synchronize(device)
        # This is the worker barrier only. The native owner must ALSO prove
        # scheduler requests/KV/connector allocations are empty before ACKing
        # aggregate drain. Dynamo instances run without a P/D KV connector.
        return dict(ok=True,rank=local_rank,drained=not sessions,at_s=time.time(),
            active_weight_sessions=len(sessions),cuda_synchronized=True,
            proof_scope='worker_weight_communication_only')
    if operation=='describe':
        parameters,rows,geometry=_metadata(worker)
        return dict(ok=True,rank=local_rank,tp=tp,device=str(device),parameters=rows,geometry=geometry,
            dtype='bfloat16',weight_bytes=sum(p.numel()*p.element_size() for p in parameters.values()),
            model_id=os.path.basename(str(worker.model_config.model).rstrip('/')),
            data_pointers_sha256=hashlib.sha256(json.dumps({n:p.data_ptr() for n,p in parameters.items()},sort_keys=True).encode()).hexdigest())
    session_id=payload.get('session_id')
    if not isinstance(session_id,str) or not session_id or len(session_id)>128:raise ValueError('bounded session ID required')
    if operation=='open':
        required=('store_host','store_port','world_size','rank_offset','gpu_ids')
        if any(key not in payload for key in required):raise ValueError('complete Dynamo GPU group geometry required')
        if payload['store_host'] not in ('127.0.0.1','localhost'):raise ValueError('Dynamo weight store must be local')
        world=payload['world_size'];offset=payload['rank_offset'];gpus=payload['gpu_ids']
        if (type(world) is not int or not 1<=world<=8 or type(offset) is not int or offset<0 or offset+tp>world
                or len(gpus)!=world or len(set(gpus))!=world or any(type(g) is not int or not 0<=g<8 for g in gpus)):
            raise ValueError('one unique physical GPU per Dynamo transfer rank required')
        identity={key:payload[key] for key in (*required,'transaction_id')}
        if session_id in sessions:
            if sessions[session_id]['identity']!=identity:raise ValueError('Dynamo session ID reused with different geometry')
            return dict(ok=True,rank=local_rank,session_id=session_id,reused=True)
        rank=offset+local_rank;timeout=timedelta(seconds=min(120,max(5,int(payload.get('timeout_s',60)))))
        store=distributed.TCPStore(payload['store_host'],payload['store_port'],world,rank==0,timeout,
                                   wait_for_workers=False)
        prefixed=distributed.PrefixStore(session_id,store)
        group=distributed.ProcessGroupNCCL(prefixed,rank,world,timeout)
        sessions[session_id]=dict(identity=identity,store=store,group=group,rank=rank,transfers={})
        return dict(ok=True,rank=local_rank,session_id=session_id,group_rank=rank,world_size=world)
    if session_id not in sessions:raise ValueError('Dynamo weight session not open')
    state=sessions[session_id]
    if payload.get('transaction_id') != state['identity']['transaction_id']:
        raise ValueError('Dynamo weight session transaction mismatch')
    if operation=='close':
        torch.cuda.synchronize(device)
        # Destruction must happen on all ranks after successful operations; an
        # uncertain NCCL failure instead requires process isolation/recovery.
        state['group'].shutdown()
        # Preserve an uncertain session if shutdown raises: a later drain must
        # not report zero active communicators before process isolation.
        del sessions[session_id]
        return dict(ok=True,rank=local_rank,session_id=session_id,closed=True)
    if operation!='transfer':raise ValueError('unsupported Dynamo worker action')
    operation_id=payload.get('operation_id')
    if not isinstance(operation_id,str) or not operation_id:raise ValueError('transfer operation ID required')
    identity=hashlib.sha256(json.dumps(payload,sort_keys=True,allow_nan=False).encode()).hexdigest()
    if operation_id in state['transfers']:
        before,result=state['transfers'][operation_id]
        if before!=identity:raise ValueError('transfer idempotency key reused with different parameters')
        return result
    source_ranks=payload['source_ranks'];target_ranks=payload['target_ranks']
    if (not source_ranks or not target_ranks or len(set(source_ranks+target_ranks))!=len(source_ranks)+len(target_ranks)
            or sorted(source_ranks+target_ranks)!=list(range(state['identity']['world_size']))):
        raise ValueError('transfer must explicitly partition all group ranks')
    source_tp=len(source_ranks);target_tp=len(target_ranks);rank=state['rank']
    is_source=rank in source_ranks
    if tp!=(source_tp if is_source else target_tp):raise ValueError('engine TP differs from transfer group partition')
    parameters,rows,geometry=_metadata(worker)
    if geometry!=payload['geometry']:raise ValueError('Dynamo source/target model geometry mismatch')
    source_shapes=payload['source_shapes'];target_shapes=payload['target_shapes']
    if sorted(parameters)!=sorted(source_shapes) or sorted(parameters)!=sorted(target_shapes):
        raise ValueError('Dynamo source/target parameter names mismatch')
    started=time.perf_counter();group=state['group'];sent=received=0;matches=True;compared=covered=0;parameter_count=0
    compare=payload.get('compare_target_before_copy',True)
    if type(compare) is not bool:raise ValueError('explicit target comparison mode required')
    warm=torch.zeros(1,device=device,dtype=torch.float32)
    group.allreduce([warm]).wait();torch.cuda.synchronize(device)
    warmup_s=time.perf_counter()-started
    with torch.no_grad():
        for name,tensor in sorted(parameters.items()):
            expected=source_shapes[name] if is_source else target_shapes[name]
            if list(tensor.shape)!=expected:raise ValueError('weight shape changed after transfer description')
            for sr,source_global in enumerate(source_ranks):
                for tr,target_global in enumerate(target_ranks):
                    pieces=transfer_pieces(name,source_shapes[name],target_shapes[name],source_tp,sr,target_tp,tr,geometry)
                    for piece in pieces:
                        if rank not in (source_global,target_global):continue
                        axis=piece['axis']
                        if rank==source_global:
                            buffer=(tensor if axis is None else tensor.narrow(axis,piece['source_offset'],piece['length'])).contiguous()
                            group.send([buffer],target_global,0).wait()
                            sent+=buffer.numel()*buffer.element_size()
                        else:
                            destination=tensor if axis is None else tensor.narrow(axis,piece['target_offset'],piece['length'])
                            buffer=torch.empty(destination.shape,device=device,dtype=tensor.dtype)
                            group.recv([buffer],source_global,0).wait()
                            if compare:
                                same=bool(torch.equal(buffer,destination));matches=matches and same
                                compared+=buffer.numel()
                            covered+=buffer.numel()
                            destination.copy_(buffer)
                            received+=buffer.numel()*buffer.element_size()
            parameter_count+=1
        torch.cuda.synchronize(device)
    result=dict(ok=True,rank=local_rank,session_id=session_id,operation_id=operation_id,
        source=is_source,group_rank=rank,sent_bytes=sent,received_bytes=received,
        golden_exact_match=matches if not is_source and compare else None,compared_elements=compared,
        covered_elements=covered,target_complete=(not is_source and covered==sum(p.numel() for p in parameters.values())),
        parameter_count=parameter_count,comm_warmup_s=warmup_s,duration_s=time.perf_counter()-started,
        weight_data_path='GPU NCCL send/recv; no host weight tensors',hardware_qualified=False)
    if not is_source and covered!=sum(p.numel() for p in parameters.values()):
        raise RuntimeError('Dynamo transfer did not cover every target parameter exactly once')
    if not is_source:
        worker._dynamo_weights_ready = False
        completed = getattr(worker, '_dynamo_target_transfers', {})
        completed[operation_id] = dict(result, transaction_id=payload['transaction_id'])
        worker._dynamo_target_transfers = completed
    state['transfers'][operation_id]=(identity,result)
    return result
