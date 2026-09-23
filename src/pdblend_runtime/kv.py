"""Policy-free retained KV protocol for the native-v1 worker.

Tensor objects never leave this process through the protocol.  Workers provide
the CUDA/NCCL hooks; this module owns request identity, races, acknowledgements
and lifetime accounting.
"""
from __future__ import annotations
import threading
import time
import json
from types import MethodType
from dataclasses import dataclass, field
from typing import Any


class KVCapabilityError(RuntimeError): pass

@dataclass
class HeldKV:
    held_request_id: str
    source_address: str
    layers: dict[int, Any] = field(default_factory=dict)
    slots: dict[int, Any] = field(default_factory=dict)
    bytes: int = 0
    transferred: set[str] = field(default_factory=set)
    cancelled: bool = False

class NativeKV:
    def __init__(self, worker):
        self.worker=worker; self.lock=threading.RLock(); self.held={}; self.receiving={}; self.transactions={}; self.transaction_payloads={}; self.expected={}; self.loads={}

    def hold(self, request_id, source_address, layers, slots, *, bytes=0):
        with self.lock:
            if request_id in self.held: raise KVCapabilityError('duplicate retained request')
            if not layers: raise KVCapabilityError('cannot retain empty KV')
            value=HeldKV(request_id,source_address,dict(layers),dict(slots),int(bytes))
            self.held[request_id]=value
            return {'acknowledged':True,'held_request_id':request_id,'layers':len(layers),'bytes':value.bytes}

    def transfer(self, payload):
        required=('held_request_id','target_request_id','target_address','target_tp','generation','transaction_id')
        if any(k not in payload for k in required): raise KVCapabilityError('transfer payload missing field')
        with self.lock:
            value=self.held.get(payload['held_request_id'])
            if value is None or value.cancelled: raise KVCapabilityError('retained KV is unavailable')
            tx=payload['transaction_id']
            tp=int(payload['target_tp'])
            expected=int(self.worker.parallel_config.tensor_parallel_size)
            if tp != expected: raise KVCapabilityError('target TP does not match source worker TP')
            current=int(self.worker._native_generation)
            if int(payload['generation']) != current: raise KVCapabilityError('KV transfer generation is stale')
            identity=json.dumps({k:payload[k] for k in required},sort_keys=True)
            if tx in self.transaction_payloads:
                if identity != self.transaction_payloads[tx]: raise KVCapabilityError('transaction identity reused for another transfer')
                if tx not in self.transactions: raise KVCapabilityError('previous transfer has uncertain completion')
                return self.transactions[tx]
            expected_layers=int(self.worker.model_config.hf_config.num_hidden_layers)
            if len(value.layers) != expected_layers or set(value.layers) != set(value.slots):
                raise KVCapabilityError('retained source does not cover all attention layers')
            self.transaction_payloads[tx]=identity
            if not hasattr(self.worker,'send_kv_layer') or not hasattr(self.worker,'wait_for_sent'):
                raise KVCapabilityError('worker lacks native KV transfer hooks')
            ranks=[]
            for layer, tensor in value.layers.items():
                slot=value.slots.get(layer)
                ack=self.worker.send_kv_layer(tensor,slot,target_address=payload['target_address'],
                    target_tp=payload['target_tp'],request_id=payload['target_request_id'],generation=payload['generation'],transaction_id=tx,layer=layer,kv_digest=bool(payload.get('kv_digest')))
                if ack is not True and not (isinstance(ack,dict) and ack.get('submitted') is True):
                    raise KVCapabilityError('native send_tensor did not submit layer')
                ranks.append(dict(ack,layer=layer,submitted=True,tensor_id=payload['target_request_id']+'#'+str(layer)))
            done=self.worker.wait_for_sent(tx, ranks)
            if not isinstance(done,dict) or done.get('acknowledged') is not True:
                raise KVCapabilityError('KV transfer lacks complete rank ACK')
            receipt={'acknowledged':True,'transaction_id':tx,'held_request_id':value.held_request_id,
                     'target_request_id':payload['target_request_id'],'generation':current,'layers':len(ranks),'layer_receipts':ranks,'ranks':done.get('ranks',[])}
            value.transferred.add(tx); self.transactions[tx]=receipt; return receipt

    def expect_load(self, payload):
        required=('target_request_id','transaction_id','generation','source_tokens')
        if any(k not in payload for k in required): raise KVCapabilityError('expect_load missing field')
        generation=payload['generation']; tokens=payload['source_tokens']
        if type(generation) is not int or generation != self.worker._native_generation:
            raise KVCapabilityError('load generation is stale')
        if type(tokens) is not int or tokens < 1: raise KVCapabilityError('positive source token count required')
        layers=int(self.worker.model_config.hf_config.num_hidden_layers)
        if payload.get('expected_layers',layers)!=layers: raise KVCapabilityError('attention layer count differs from model')
        key=(payload['target_request_id'],payload['transaction_id'])
        if any(not isinstance(x,str) or not x for x in key): raise KVCapabilityError('nonempty load identity required')
        with self.lock:
            if key in self.expected:
                old=self.expected[key]
                if old['generation']!=generation or old['source_tokens']!=tokens:
                    raise KVCapabilityError('receive identity reused with different metadata')
                return dict(acknowledged=True,already_registered=True,generation=generation)
            if any(k[0]==key[0] for k in self.expected): raise KVCapabilityError('request already belongs to another transaction')
            self.expected[key]=dict(target_request_id=key[0],transaction_id=key[1],generation=generation,
                source_tokens=tokens,expected_layers=layers,loaded_layers=set(),failed=False,
                kv_digest=bool(payload.get('kv_digest')),digests={})
            self.loads[key]=self.expected[key]
        return dict(acknowledged=True,target_request_id=key[0],transaction_id=key[1],
                    generation=generation,expected_layers=layers,loaded_layers=0)

    def load_ack(self, payload):
        required=('target_request_id','transaction_id','layer','generation')
        if any(k not in payload for k in required): raise KVCapabilityError('load ACK missing field')
        with self.lock:
            if not hasattr(self.worker,'start_load_kv') or not hasattr(self.worker,'consume_loaded_kv'):
                raise KVCapabilityError('worker lacks native KV load hooks')
            key=(payload['target_request_id'],payload['transaction_id'])
            spec=self.expected.get(key)
            if spec is None: raise KVCapabilityError('load was not pre-registered with expect_load')
            if payload['generation']!=self.worker._native_generation or spec['generation']!=payload['generation']:
                raise KVCapabilityError('load acknowledgement generation differs')
            record=spec['loaded_layers']
            layer=payload['layer']
            if layer in record: return {'acknowledged':True,'duplicate':True,'layer':layer}
            try:
                loaded=self.worker.start_load_kv(request_id=payload['target_request_id'],transaction_id=payload['transaction_id'],layer=layer,generation=payload['generation'])
                if loaded is False: raise KVCapabilityError('native KV layer load failed')
                consumed=self.worker.consume_loaded_kv(request_id=payload['target_request_id'],transaction_id=payload['transaction_id'],layer=layer)
                if consumed is False: raise KVCapabilityError('native KV layer consume failed')
            except KVCapabilityError: raise
            except Exception as exc: raise KVCapabilityError('native KV load failed') from exc
            record.add(layer)
            self.loads[key]=spec
            return {'acknowledged':len(record)==spec['expected_layers'],'target_request_id':payload['target_request_id'],'transaction_id':payload['transaction_id'],'layer':layer,'loaded_layers':len(record),'expected_layers':spec['expected_layers']}

    def release(self, held_request_id):
        with self.lock:
            value=self.held.pop(held_request_id,None)
            if value is None: raise KVCapabilityError('unknown retained request')
            value.layers.clear(); value.slots.clear(); value.transferred.clear()
            return {'acknowledged':True,'held_request_id':held_request_id,'released':True}

    def cancel(self, request_id):
        with self.lock:
            uncertain=[spec for key,spec in self.expected.items() if key[0]==request_id and
                       (spec['failed'] or len(spec['loaded_layers'])!=spec['expected_layers'])]
            if uncertain:
                for spec in uncertain:spec['failed']=True
                return dict(acknowledged=False,request_id=request_id,quarantined=True,
                            reason='receive completion is uncertain')
            value=self.held.get(request_id)
            if value: value.cancelled=True; self.held.pop(request_id,None)
            for table in (self.receiving,self.expected,self.loads):
                for key in tuple(table):
                    if key[0]==request_id: table.pop(key,None)
            return {'acknowledged':True,'request_id':request_id}

    def query_load(self, payload):
        key=(payload.get('target_request_id'),payload.get('transaction_id'))
        spec=self.loads.get(key)
        if spec is None: return {'acknowledged':False,'loads':{},'target_request_id':key[0],'transaction_id':key[1]}
        loaded=len(spec['loaded_layers']); expected=spec['expected_layers']
        return {'acknowledged':loaded==expected and not spec['failed'] and spec['generation']==self.worker._native_generation,'loads':{str(x):True for x in spec['loaded_layers']},
                'target_request_id':spec['target_request_id'],'transaction_id':spec['transaction_id'],
                'generation':spec['generation'],'expected_layers':expected,'loaded_layers':loaded,
                'source_tokens':spec['source_tokens'],'digests':spec['digests'],'failed':spec['failed']}

    def state(self):
        with self.lock:
            return {'held_requests':list(self.held),'held_count':len(self.held),
                    'held_bytes':sum(v.bytes for v in self.held.values()),
                    'receiving_transactions':sum(len(v['loaded_layers']) != v['expected_layers'] or v['failed'] for v in self.expected.values()),'transaction_count':len(self.transactions)}

def operation(worker, name, payload):
    """Entry point for ``NativeWorker.native_kv_operation``."""
    manager=getattr(worker,'_pdblend_native_kv',None)
    if manager is None:
        manager=NativeKV(worker); worker._pdblend_native_kv=manager
    if name in ('hold','prefill_hold','save'):
        if payload['held_request_id'] in manager.held:
            value=manager.held[payload['held_request_id']]
            value.source_address=payload.get('source_address',value.source_address)
            value.layers.update(payload['layers']); value.slots.update(payload.get('slots',{})); value.bytes=max(value.bytes,int(payload.get('bytes',0)))
            return {'acknowledged':True,'held_request_id':value.held_request_id,'layers':len(value.layers),'bytes':value.bytes}
        return manager.hold(payload['held_request_id'],payload.get('source_address',''),payload['layers'],payload.get('slots',{}),bytes=payload.get('bytes',0))
    if name in ('transfer','send'): return manager.transfer(payload)
    if name=='expect_load': return manager.expect_load(payload)
    if name in ('load_ack','consume'): return manager.load_ack(payload)
    if name=='release': return manager.release(payload['held_request_id'])
    if name=='cancel': return manager.cancel(payload['request_id'])
    if name=='state': return manager.state()
    if name=='query_load': return manager.query_load(payload)
    raise KVCapabilityError('unsupported native KV operation: '+str(name))

def install(worker):
    """Bind the protocol to the vLLM P2P connector on a native worker.

    Installation is deliberately explicit: an unavailable transfer group or
    connector method raises during startup instead of silently becoming a
    local/fake KV path.
    """
    try:
        from vllm.distributed.kv_transfer import get_kv_transfer_group
        connector=get_kv_transfer_group()
    except Exception as exc: raise KVCapabilityError('P2P KV transfer group unavailable') from exc
    engine=getattr(connector,'p2p_nccl_engine',None)
    if engine is None or not all(hasattr(engine,n) for n in ('send_tensor','recv_tensor','wait_for_sent')):
        raise KVCapabilityError('P2P connector lacks native send/recv/wait methods')
    manager=NativeKV(worker); worker._pdblend_native_kv=manager; worker._pdblend_kv_connector=connector
    original_save=connector.save_kv_layer
    def save(self, layer_name, kv_layer, attn_metadata, **kwargs):
        metadata=self._get_connector_metadata()
        requests=getattr(metadata,'requests',())
        holds=[r for r in requests if str(getattr(r,'request_id','')).startswith('distserve-hold-')]
        normal=[r for r in requests if r not in holds]
        for req in holds:
            rid=str(req.request_id); value=manager.held.setdefault(rid,HeldKV(rid,'',{},{}))
            value.layers[layer_name]=kv_layer; value.slots[layer_name]=getattr(req,'slot_mapping',None)
        if holds:
            try: metadata.requests=normal; return original_save(layer_name,kv_layer,attn_metadata,**kwargs)
            finally: metadata.requests=requests
        return original_save(layer_name,kv_layer,attn_metadata,**kwargs)
    connector.save_kv_layer=MethodType(save,connector)
    original_load=connector.start_load_kv
    def load(self, forward_context, **kwargs):
        seen={}; original_recv=engine.recv_tensor
        metadata=self._get_connector_metadata()
        metas={r.request_id:r for r in getattr(metadata,'loads',())}
        layers={name:getattr(layer,'kv_cache',None) for name,layer in forward_context.no_compile_layers.items()
                if getattr(layer,'kv_cache',None) is not None}
        def recv(tensor_id, *args, **kw):
            value=original_recv(tensor_id,*args,**kw)
            if value is None: raise KVCapabilityError('native recv returned no KV for '+str(tensor_id))
            rid,layer=str(tensor_id).rsplit('#',1)
            entries=[v for k,v in manager.expected.items() if k[0]==rid]
            if entries:
                spec=entries[0]
                if spec['generation']!=worker._native_generation or spec['failed']:
                    raise KVCapabilityError('stale or failed receive transaction')
                if len(value.shape)!=3 or value.shape[0]!=2 or value.shape[1]!=spec['source_tokens']:
                    raise KVCapabilityError('received KV token layout differs from registered source')
                if layer not in layers: raise KVCapabilityError('received unknown attention layer')
                if spec['kv_digest']:
                    from .kv_digest import digest_tensor
                    spec['digests'].setdefault(layer,{})['received']=digest_tensor(value,
                        transaction_id=spec['transaction_id'],generation=spec['generation'],request_id=rid,
                        rank=worker.rank,layer=layer,stage='received')
            seen.setdefault(rid,set()).add(layer)
            return value
        engine.recv_tensor=recv
        try:
            result=original_load(forward_context,**kwargs)
            for rid,loaded in seen.items():
                entries=[v for k,v in manager.expected.items() if k[0]==rid]
                if not entries:continue
                spec=entries[0]
                if loaded!=set(layers) or len(loaded)!=spec['expected_layers']:
                    raise KVCapabilityError('native load completed with missing attention layers')
                if spec['kv_digest']:
                    from .kv_digest import digest_paged_kv
                    for layer in loaded:
                        slots=metas[rid].slot_mapping[:spec['source_tokens']]
                        spec['digests'][layer]['injected']=digest_paged_kv(layers[layer][forward_context.virtual_engine],slots,
                            transaction_id=spec['transaction_id'],generation=spec['generation'],request_id=rid,
                            rank=worker.rank,layer=layer,stage='injected')
                spec['loaded_layers'].update(loaded)
            return result
        except BaseException:
            for spec in manager.expected.values():
                if spec['target_request_id'] in metas:
                    spec['failed']=True;spec['loaded_layers'].clear()
            raise
        finally:engine.recv_tensor=original_recv
    connector.start_load_kv=MethodType(load,connector)
    def send(self,tensor,slot,**kw):
        tp=int(kw['target_tp']); expected=int(getattr(getattr(self,'parallel_config',None),'tensor_parallel_size',tp))
        if tp != expected: raise KVCapabilityError('target TP mismatch')
        address=str(kw['target_address']); host,port=address.rsplit(':',1)
        rank=int(getattr(self,'rank',0)); address=f'{host}:{int(port)+rank}'
        if hasattr(slot,'clone'): slot=slot.clone()
        digest=None
        if kw.get('kv_digest'):
            from .kv_digest import digest_paged_kv
            digest=digest_paged_kv(tensor,slot,transaction_id=kw['transaction_id'],generation=kw['generation'],
                request_id=kw['request_id'],rank=rank,layer=kw['layer'],stage='source')
        ok=engine.send_tensor(kw['request_id']+'#'+str(kw['layer']),tensor,address,slot,kw.get('is_mla',False))
        if ok is not True: raise KVCapabilityError('native send_tensor rejected layer')
        return {'submitted':True,'target_address':address,'layer':kw['layer'],'source_digest':digest}
    worker.send_kv_layer=MethodType(send,worker)
    def wait(self,tx,ranks):
        deadline=time.monotonic()+120
        with engine.send_queue_cv:
            while engine.send_queue:
                if not engine._send_thread.is_alive(): raise KVCapabilityError('native sender thread failed')
                if time.monotonic()>deadline: raise KVCapabilityError('native send completion timeout')
                engine.send_queue_cv.wait(timeout=.1)
        expected={r['tensor_id'] for r in ranks}
        actual=set().union(*engine.send_request_id_to_tensor_ids.values()) if engine.send_request_id_to_tensor_ids else set()
        if not expected <= actual:raise KVCapabilityError('native send queue emptied without complete successful sends')
        return {'acknowledged':True,'transaction_id':tx,'ranks':ranks,'sent_tensor_ids':sorted(expected)}
    worker.wait_for_sent=MethodType(wait,worker)
    return connector
