"""Metadata-only rendezvous for a not-yet-serving same-TP target worker.

Only JSON identities and CUDA IPC descriptors cross this directory, never
weights. The actual target worker publishes its process identity before model
load, because its ordinary HTTP endpoint is not ready at that point.
"""
from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import time
import uuid

from .stationary_ipc import digest,need,process_identity
from .stationary_owner_graph import read_bound


def publish(path,value):
    """Atomically publish once; no reader can see partial JSON or overwritten evidence."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.parent/('.'+path.name+'.'+uuid.uuid4().hex)
    try:
        with temporary.open('x') as stream:
            json.dump(value,stream,sort_keys=True,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
        os.link(temporary,path)
    finally:
        temporary.unlink(missing_ok=True)


class TargetBootstrap:
    def __init__(self,ref):
        self.ref=deepcopy(ref);self.config=read_bound(ref);c=self.config
        need(c.get('schema')=='dynamo-same-tp-target-bootstrap/v1', 'same-TP target bootstrap required')
        p=c['plan']
        need(p['plan_sha256']==digest({k:v for k,v in p.items() if k!='plan_sha256'})
             and len(p['source_gpu_uuids'])==len(p['target_gpu_uuids'])==1
             and p['source_gpu_uuids']==p['target_gpu_uuids'] and p['source_shapes']==p['target_shapes']
             and p['planned_transfer_bytes']==0 and all(r['kind']=='retain_on_gpu' for r in p['pieces']),
             'initial real target bridge supports same-GPU TP1 retained storage only')
        need(type(c.get('source_generation')) is int and c['source_generation']>=0
             and type(c.get('target_generation')) is int and c['target_generation']>=0,
             'source and target epochs must be explicit')
        need(isinstance(c.get('transaction_id'),str) and 0<len(c['transaction_id'])<=128,
             'bounded target transaction required')
        need(type(c.get('target_gpu_memory_utilization')) in (int,float)
             and 0<c['target_gpu_memory_utilization']<1, 'explicit target native memory request required')
        need(c.get('target_public_admission') is False, 'target must remain behind the private qualification fence')
        need(Path(c['rendezvous_dir']).is_absolute(), 'absolute private metadata rendezvous required')
        self.directory=Path(c['rendezvous_dir']).resolve()
        self.ready=None

    def publish_ready(self,*,gpu_uuid,device_index):
        need(gpu_uuid==self.config['plan']['target_gpu_uuids'][0]
             and type(device_index) is int and device_index>=0, 'target actual CUDA placement differs')
        self.ready=dict(schema='dynamo-target-worker-bootstrap-ready/v1',process=process_identity(),
            transaction_id=self.config['transaction_id'],bootstrap_ref=self.ref,
            plan_sha256=self.config['plan']['plan_sha256'],gpu_uuid=gpu_uuid,device_index=device_index,
            target_rank=0,target_generation=self.config['target_generation'],at_s=time.time(),
            target_model_loaded=False,target_KV_initialized=False,target_served=False,formal_eligible=False)
        publish(self.directory/'target-rank-0-ready.json',self.ready)
        return deepcopy(self.ready)

    def wait_packet(self,timeout_s=60.):
        need(self.ready is not None,'target must publish its real worker identity before requesting exports')
        path=self.directory/'target-rank-0-packet.json';deadline=time.monotonic()+timeout_s
        while not path.is_file():
            need(time.monotonic()<deadline,'original owner IPC export rendezvous timed out')
            time.sleep(.02)
        envelope=json.loads(path.read_text())
        need(envelope.get('bootstrap_ref')==self.ref
             and envelope.get('consumer')==self.ready['process'], 'packet bootstrap or target process differs')
        packet=read_bound(envelope['packet_ref'])
        need(packet['packet_sha256']==digest({k:v for k,v in packet.items() if k!='packet_sha256'})
             and packet['consumer']==self.ready['process'] and packet['target_rank']==0
             and packet['plan_sha256']==self.config['plan']['plan_sha256']
             and packet['generation']==self.config['source_generation']
             and packet['gpu_uuid']==self.ready['gpu_uuid'], 'original owner export packet differs')
        return packet

    def record(self,stage,**fields):
        publish(self.directory/('target-rank-0-'+stage+'.json'),dict(stage=stage,
            bootstrap_ref=self.ref,process=process_identity(),at_s=time.time(),**fields))


def packet_from_worker_receipt(row):
    """Unwrap transport metadata without adding fields to a hashed IPC packet."""
    fields=('schema','owner','consumer','plan_sha256','source_rank','target_rank','generation',
            'gpu_uuid','export_id','source_storage','views','packet_sha256')
    packet={key:deepcopy(row[key]) for key in fields}
    need(packet['packet_sha256']==digest({k:v for k,v in packet.items() if k!='packet_sha256'}),
         'worker RPC mutated the owner export packet')
    return packet
