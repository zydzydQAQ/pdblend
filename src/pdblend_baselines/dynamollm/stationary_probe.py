"""Two CUDA processes on one leased GPU; no model loading or serving claim."""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import signal
import time
import traceback
from copy import deepcopy

from .stationary_ipc import StationaryOwner,StationaryConsumer,TorchCudaIpcCodec,cuda_reducer_abi,need,process_identity,validate_descriptor
from .stationary_tensors import tensor_plan

GEOMETRY=dict(hidden_size=16,num_attention_heads=8,num_key_value_heads=4,intermediate_size=32)
UNKNOWN_UUID='GPU-00000000-0000-0000-0000-000000000000'
PROBE_PLAN=dict(schema='dynamo-stationary-ipc-primitive-plan/v3',gpu_count=1,model_loads=0,
    cases=['clean_consumer_exit','consumer_crash'],start_method='spawn',case_timeout_s=60,
    source_tp=1,logical_target_tp=2,executed_target_ranks=[0],unexecuted_target_ranks=[1],
    unexecuted_rank_uuid_source='synthetic_unknown_uuid_not_a_physical_peer',
    unknown_uuid_negative_scope='descriptor_metadata_validation_before_cuda_import',
    actual_cuda_uuid_contract='dynamo-cuda-uuid-identity/v1',
    synthetic_parameter_dtype='bfloat16',synthetic_shape_geometry=GEOMETRY,
    retained_source_copy_allowed=False,host_weight_staging_allowed=False,
    target_serving_activation_allowed=False,energy_comparable=False,formal_eligible=False)


def write_new(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream:json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')


def shapes(tp):
    return {'model.layers.0.self_attn.qkv_proj.weight':[32//tp,16],
            'model.layers.0.self_attn.qkv_proj.bias':[32//tp],
            'model.layers.0.mlp.gate_up_proj.weight':[64//tp,16],
            'model.layers.0.self_attn.o_proj.weight':[16,16//tp],
            'model.layers.0.mlp.down_proj.weight':[16,32//tp],
            'model.embed_tokens.weight':[32//tp,16],'model.norm.weight':[16]}


def expected_shard(name,tp,rank):
    """Independent Qwen layout oracle; these are tiny synthetic values."""
    import numpy as np
    full=(np.arange(np.prod(shapes(1)[name]),dtype=np.float32)%32).reshape(shapes(1)[name])
    if name.endswith('norm.weight'):return full.copy()
    if '.qkv_proj.' in name:
        return np.concatenate([np.split(part,tp,axis=0)[rank] for part in np.split(full,[16,24],axis=0)])
    if '.gate_up_proj.' in name:
        return np.concatenate([np.split(part,tp,axis=0)[rank] for part in np.split(full,2,axis=0)])
    axis=1 if name.endswith(('o_proj.weight','down_proj.weight')) else 0
    return np.split(full,tp,axis=axis)[rank].copy()


class RemotePrimitiveError(RuntimeError):
    def __init__(self, event):
        self.remote_event=deepcopy(event)
        self.uuid_identity=event.get('uuid_identity')
        super().__init__(event['error'])


def failure_event(error):
    return dict(event='process_failure',error=traceback.format_exc(),
                uuid_identity=getattr(error,'uuid_identity',None))


def receive(connection,timeout=30,*,events=None):
    need(connection.poll(timeout),'owned IPC process response timeout')
    value=connection.recv()
    if events is not None:events.append(value)
    if isinstance(value,dict) and value.get('error'):raise RemotePrimitiveError(value)
    return value


def lease_geometry(uuid,inventory):
    """One visible leased GPU; rank 1 is geometry only, never a physical peer."""
    inventory=[v.decode() if isinstance(v,bytes) else v for v in inventory]
    need(inventory==[uuid], 'NVML visibility must contain exactly the one leased GPU')
    need(UNKNOWN_UUID not in inventory, 'synthetic unknown UUID collides with visible hardware')
    return dict(gpu_uuid=uuid,nvml_inventory_gpu_uuids=inventory,
        unexecuted_logical_peer_uuid=UNKNOWN_UUID,
        unexecuted_rank_uuid_source=PROBE_PLAN['unexecuted_rank_uuid_source'],
        logical_peer_is_observed_physical_gpu=False,peer_cuda_operations=0)


def unknown_descriptor_rejection(packet,unknown_uuid):
    """Negative metadata guard only; no foreign GPU is opened or asserted to exist."""
    need(unknown_uuid==UNKNOWN_UUID and unknown_uuid!=packet['gpu_uuid'], 'unknown UUID test differs')
    descriptor=deepcopy(packet['views'][0]['descriptor']);descriptor['gpu_uuid']=unknown_uuid
    try:validate_descriptor(descriptor,gpu_uuid=packet['gpu_uuid'])
    except ValueError as error:
        return dict(passed=True,claimed_gpu_uuid=unknown_uuid,actual_gpu_uuid=packet['gpu_uuid'],
            rejection=str(error),scope=PROBE_PLAN['unknown_uuid_negative_scope'],
            claimed_uuid_is_observed_physical_gpu=False,cuda_import_attempted=False)
    raise ValueError('unknown physical UUID descriptor was accepted')


def verify_views(consumer):
    import torch
    checked=[]
    for piece,view in consumer.views:
        # This independent synthetic oracle is not a copy of retained weights.
        expected=torch.tensor(expected_shard(piece['parameter'],2,0),dtype=torch.bfloat16,device='cuda:0')
        if piece['axis'] is not None:
            expected=expected.narrow(piece['axis'],piece['target_offset'],piece['length'])
        need(torch.equal(view,expected),'same-device original fragment differs from independent target shard')
        checked.append(dict(parameter=piece['parameter'],target_offset=piece['target_offset'],
                            bytes=piece['bytes'],equal=True,oracle='independent synthetic Qwen TP2 rank0'))
    return checked


def consumer_main(connection,uuid,plan):
    try:
        identity=process_identity();codec=TorchCudaIpcCodec(gpu_uuid=uuid,device_index=0)
        connection.send(dict(identity=identity,uuid_identity=codec.uuid_identity))
        packet=receive(connection)
        negative=unknown_descriptor_rejection(packet,UNKNOWN_UUID)
        consumer=StationaryConsumer(packet,plan=plan,codec=codec,expected_generation=0)
        checked=verify_views(consumer)
        connection.send(dict(imported=consumer.receipt(),checks=checked,unknown_uuid_rejection=negative))
        command=receive(connection)
        if command=='crash':os._exit(77)  # Intentional abnormal exit with imported views still live.
        need(command=='close','unknown qualification command')
        connection.send(dict(release_ack=consumer.close()))
    except BaseException as error:
        connection.send(failure_event(error));raise
    finally:connection.close()


def owner_main(connection,uuid,peer_uuid,case):
    child=None
    try:
        os.setsid();connection.send(dict(event='owner_started',identity=process_identity(),process_group=os.getpgrp()))
        import torch
        codec=TorchCudaIpcCodec(gpu_uuid=uuid,device_index=0)
        connection.send(dict(event='owner_uuid_identity',identity=process_identity(),uuid_identity=codec.uuid_identity))
        plan=tensor_plan(source_gpus=[uuid],target_gpus=[uuid,peer_uuid],source_shapes=shapes(1),
                         target_shapes=shapes(2),geometry=GEOMETRY)
        parameters={n:torch.tensor(expected_shard(n,1,0),dtype=torch.bfloat16,device='cuda:0') for n in shapes(1)}
        owner=StationaryOwner(plan,source_rank=0,parameters=parameters,gpu_uuid=uuid,generation=0,codec=codec)
        ctx=multiprocessing.get_context('spawn');parent_pipe,child_pipe=ctx.Pipe()
        child=ctx.Process(target=consumer_main,args=(child_pipe,uuid,plan));child.start();child_pipe.close()
        started=receive(parent_pipe);identity=started['identity']
        connection.send(dict(event='consumer_started',identity=identity,process_group=os.getpgrp(),
                             uuid_identity=started['uuid_identity']))
        packet=owner.export(identity,target_rank=0);parent_pipe.send(packet)
        imported=receive(parent_pipe)
        need(imported['checks'] and all(c['equal'] for c in imported['checks']),'no verified retained fragments')
        need(imported['unknown_uuid_rejection']['passed'], 'unknown UUID metadata guard failed')
        if case=='clean_consumer_exit':
            parent_pipe.send('close');ack=receive(parent_pipe)['release_ack'];owner.acknowledge_release(ack)
        else:parent_pipe.send('crash')
        child.join(10);need(not child.is_alive(),'consumer did not exit before owner release')
        need(child.exitcode==(0 if case=='clean_consumer_exit' else 77),'unexpected consumer exit')
        release=owner.release(consumer_processes_gone=[identity])
        need(release['owner_exit_required']==(case=='consumer_crash'),'owner quarantine policy differs')
        connection.send(dict(event='case_result',case=case,status='passed',plan=plan,export_packet=packet,
            imported=imported,release=release,consumer_exitcode=child.exitcode,
            owner_identity=process_identity(),owner_must_exit=True,at_s=time.time(),
            torch_version=torch.__version__,model_loads=0,hardware_primitive_only=True,
            full_tp_switch_qualified=False,target_engine_activated=False,formal_eligible=False))
        parent_pipe.close()
    except BaseException as error:
        connection.send(failure_event(error));raise
    finally:
        if child is not None and child.is_alive():child.terminate();child.join(3)
        if child is not None and child.is_alive():child.kill();child.join(3)
        connection.close()


def nvml_inventory():
    import pynvml as nv
    nv.nvmlInit()
    try:
        return [nv.nvmlDeviceGetUUID(nv.nvmlDeviceGetHandleByIndex(i)) for i in range(nv.nvmlDeviceGetCount())]
    finally:nv.nvmlShutdown()


def compute_processes(uuid):
    import pynvml as nv
    nv.nvmlInit()
    try:
        handle=nv.nvmlDeviceGetHandleByUUID(uuid)
        return [dict(pid=int(row.pid),used_gpu_memory_bytes=int(row.usedGpuMemory))
                for row in nv.nvmlDeviceGetComputeRunningProcesses(handle)]
    finally:nv.nvmlShutdown()


def require_compute_empty(uuid,timeout=5):
    deadline=time.monotonic()+timeout;observations=[]
    while True:
        rows=compute_processes(uuid);observations.append(dict(at_s=time.time(),gpu_uuid=uuid,compute_processes=rows))
        if not rows:return dict(passed=True,observations=observations)
        if time.monotonic()>=deadline:return dict(passed=False,observations=observations)
        time.sleep(.05)


def preflight(config_path):
    config_path=Path(config_path).resolve();config=json.loads(config_path.read_text())
    config_sha=hashlib.sha256(config_path.read_bytes()).hexdigest()
    need(os.environ.get('PDBLEND_PRIMITIVE_CONFIG_SHA256')==config_sha,'immutable primitive configuration differs')
    need(config['plan']==PROBE_PLAN,'fixed primitive plan changed')
    source=Path(config['source_snapshot']);manifest=json.loads((source/'manifest.json').read_text())
    expected=manifest['files']
    source_sha=hashlib.sha256(json.dumps(expected,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    need(config['source_sha256']==source_sha==manifest['source_sha256'],'source manifest identity differs')
    for name,sha in expected.items():
        path=source/name
        need(path.resolve().is_relative_to(source.resolve()) and hashlib.sha256(path.read_bytes()).hexdigest()==sha,
             'frozen primitive source differs: '+name)
    need(Path(__file__).resolve()==source/'pdblend_baselines/dynamollm/stationary_probe.py',
         'qualification must import the frozen source snapshot')
    need(os.environ.get('PDBLEND_IMAGE_ID')==config['image_id'],'pinned image identity missing')
    abi=cuda_reducer_abi();need(not abi['cuda_initialized'],'primitive coordinator must remain CPU-only')
    return dict(status='passed',config_sha256=config_sha,
        config_path=str(config_path),source_sha256=source_sha,source_snapshot=str(source),
        source_files_verified=len(expected),python_cuda_reducer_abi=abi,
        gpu_initialized=False,model_loads=0,formal_eligible=False)


def run(config_path,out):
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False)
    check=preflight(config_path);write_new(out/'preflight.json',check)
    uuids=os.environ.get('PDBLEND_GPU_UUIDS','').split(',');visible=os.environ.get('CUDA_VISIBLE_DEVICES','').split(',')
    need(len(uuids)==len(visible)==1 and uuids[0].startswith('GPU-') and visible[0].isdigit(),
         'one explicitly leased physical GPU required')
    uuid=uuids[0];geometry=lease_geometry(uuid,nvml_inventory());peer=geometry['unexecuted_logical_peer_uuid']
    concurrency_path=os.environ.get('PDBLEND_CONCURRENCY_ENVIRONMENT')
    concurrency=None
    if concurrency_path:
        path=Path(concurrency_path).resolve()
        concurrency=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    write_new(out/'lease.json',dict(**geometry,cuda_visible_devices=visible,
        concurrency_environment=concurrency,
        coordinator=process_identity(),coordinator_cuda_initialized=False))
    before=require_compute_empty(uuid,timeout=0);write_new(out/'gpu-before.json',before)
    need(before['passed'],'leased GPU is not empty; unowned work must not be interrupted')
    ctx=multiprocessing.get_context('spawn');results=[];failures=[];owners_started=0
    try:
        for case in PROBE_PLAN['cases']:
            parent_pipe,child_pipe=ctx.Pipe();owner=ctx.Process(target=owner_main,args=(child_pipe,uuid,peer,case))
            owner.start();owners_started+=1;child_pipe.close();events=[];group=None;result=None
            deadline=time.monotonic()+PROBE_PLAN['case_timeout_s']
            try:
                while result is None:
                    message=receive(parent_pipe,max(0,deadline-time.monotonic()),events=events)
                    if message.get('event')=='owner_started':
                        need(message['process_group']==owner.pid,'owned owner process group differs');group=owner.pid
                    if message.get('event')=='case_result':result=message
                owner.join(10);need(not owner.is_alive() and owner.exitcode==0,'source storage owner did not exit cleanly')
            finally:
                # Only the newly spawned owner group is signalled. The CPU
                # coordinator never signals any unrelated GPU process.
                if group is not None:
                    try:os.killpg(group,signal.SIGTERM)
                    except ProcessLookupError:pass
                elif owner.is_alive():owner.terminate()
                owner.join(3)
                if group is not None:
                    try:os.killpg(group,signal.SIGKILL)
                    except ProcessLookupError:pass
                elif owner.is_alive():owner.kill()
                owner.join(3);parent_pipe.close()
                write_new(out/(case+'-events.json'),dict(events=events,owner_pid=owner.pid,owner_exitcode=owner.exitcode))
                cleanup=require_compute_empty(uuid);write_new(out/(case+'-cleanup.json'),cleanup)
                need(cleanup['passed'],'leased GPU still owns CUDA processes after owner-group cleanup')
            need(result is not None,'case did not return a primitive result');results.append(result)
            write_new(out/(case+'.json'),result)
    except BaseException:failures.append(traceback.format_exc())
    finally:
        after=require_compute_empty(uuid);write_new(out/'gpu-after.json',after)
        complete=dict(schema='dynamo-stationary-ipc-primitive-completion/v3',
            status='passed' if len(results)==2 and not failures and after['passed'] else 'failed',
            complete=len(results)==2 and not failures and after['passed'],cases_completed=len(results),errors=failures,
            gpu_uuid=uuid,unexecuted_logical_peer_uuid=peer,source_sha256=check['source_sha256'],
            unexecuted_rank_uuid_source=geometry['unexecuted_rank_uuid_source'],
            logical_peer_is_observed_physical_gpu=False,peer_cuda_operations=0,
            model_loads=0,engine_starts=0,cuda_owner_processes_started=owners_started,
            same_gpu_hardware_primitive_qualified=len(results)==2 and not failures and after['passed'],
            gpu_cleanup_passed=after['passed'],full_tp_switch_qualified=False,
            original_dynamo_mechanism_qualified=False,target_engine_activated=False,
            energy_comparable=False,formal_eligible=False)
        write_new(out/'completion.json',complete)
    need(complete['complete'],'CUDA IPC primitive qualification failed; inspect raw case events')
    return complete


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True);parser.add_argument('--out',type=Path)
    parser.add_argument('--preflight',action='store_true');args=parser.parse_args()
    if args.preflight:print(json.dumps(preflight(args.config),sort_keys=True));return
    parser.error('--out required unless --preflight') if args.out is None else None
    print(json.dumps(run(args.config,args.out),sort_keys=True))


if __name__=='__main__':main()
