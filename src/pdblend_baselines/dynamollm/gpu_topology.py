"""Dynamo-owned acknowledged topology adapter using GPU shard transfer.

The first execution mapping uses a complete surviving source engine on GPUs
disjoint from each target. Redundant sources can be drained to make room while
the surviving source and unrelated pools still compute. CUDA IPC retention on
reused GPUs is not implemented and no stationary bytes are claimed for them.
"""
import asyncio
import inspect
import math
import time


class GpuTopologyHooks:
    hardware_qualified=False
    supports_native_transitions=True
    def __init__(self,transport,lifecycle,journal,*,goldens,store_port=19660,timeout_s=180):
        self.transport=transport;self.lifecycle=lifecycle;self.journal=journal
        self.goldens=goldens;self.store_port=store_port;self.timeout_s=timeout_s
        self.controller=None;self.transactions={}

    def bind(self,controller):self.controller=controller

    def plan_layout(self,source,target_tps,available):
        required=sum(target_tps)
        for candidate in sorted(source,key=lambda r:(r.tp,r.gpus,r.instance_id)):
            spare=[g for g in available if g not in candidate.gpus]
            if len(spare)>=required:
                targets=[];cursor=0
                for tp in target_tps:targets.append(tuple(spare[cursor:cursor+tp]));cursor+=tp
                return targets
        raise ValueError('Dynamo physical pool lacks disjoint complete source / target placement')

    async def emit(self,kind,**fields):
        value=self.journal(kind,**fields)
        if inspect.isawaitable(value):await value

    async def _gate(self,iid,accepting):
        if self.controller and iid in self.controller.replicas:
            self.controller.replicas[iid].accepting=accepting
        result=await self.transport.json(iid,'/baseline/dynamollm/'+('resume' if accepting else 'quiesce'),{})
        if bool(result.get('accepting'))!=accepting:raise RuntimeError('Dynamo HTTP admission gate ACK mismatch')

    async def _drain(self,iid):
        began=time.monotonic()
        while time.monotonic()-began<self.timeout_s:
            state=await self.transport.state(iid)
            if state.get('error') or state.get('runtime_error'):raise RuntimeError('Dynamo source unhealthy while draining')
            stamp=state.get('timestamp');fresh=type(stamp) in (int,float) and math.isfinite(stamp) and 0<=time.time()-stamp<=.5
            empty=(state.get('active')==0 and state.get('running')==0 and state.get('waiting')==0
                and state.get('kv_allocations')=={} and state.get('transfer_allocations')=={}
                and state.get('free_kv_tokens')==state.get('total_kv_tokens') and state.get('total_kv_tokens',0)>0
                and state.get('transport_healthy') is True and state.get('evidence_complete') is True
                and type(state.get('generation')) is int and state['generation']>=0
                and state.get('acknowledged_generation')==state.get('generation'))
            if fresh and empty:
                ack=await self.transport.json(iid,'/baseline/dynamollm/drain',{})
                ranks=ack.get('ranks',[]);tp=self.lifecycle.instances[iid]['tp']
                if (ack.get('drained') is True and ack.get('owner_ack') is True and len(ranks)==tp
                        and {r.get('rank') for r in ranks}==set(range(tp))
                        and all(r.get('ok') is True and r.get('drained') is True for r in ranks)):
                    await self.emit('dynamo_gpu_drained',instance_id=iid,runtime=state,ack=ack)
                    return
            await asyncio.sleep(.02)
        raise TimeoutError('Dynamo source request drain timed out')

    async def _stop_confirmed(self,iid):
        ack=await self.lifecycle.stop(iid)
        if not isinstance(ack,dict) or ack.get('absent') is not True:
            raise RuntimeError('Dynamo lifecycle lacks confirmed absent ACK; endpoint retained for isolation')
        self.transport.instances.pop(iid,None)
        self.lifecycle.instances.pop(iid,None)
        await self.emit('dynamo_gpu_instance_absent',instance_id=iid,ack=ack)
        return ack

    async def prepare(self,t,plan):
        state=self.transactions.setdefault(t.transaction_id,dict(targets=[],retired=[],sources={},sessions=[]))
        for iid,gpus in zip(t.source_ids,t.source_layout):
            state['sources'][iid]=dict(self.lifecycle.instances[iid])
        target_gpus={g for group in t.target_layout for g in group}
        survivors=[iid for iid,gpus in zip(t.source_ids,t.source_layout) if not set(gpus)&target_gpus]
        if not survivors:raise ValueError('GPU transfer adapter requires a complete source disjoint from target GPUs')
        source=min(survivors,key=lambda iid:(self.lifecycle.instances[iid]['tp'],iid))
        state['source']=source
        for iid,gpus in zip(t.source_ids,t.source_layout):
            if set(gpus)&target_gpus:
                await self._gate(iid,False);await self._drain(iid)
                await self._stop_confirmed(iid);state['retired'].append(iid)
        source_spec=state['sources'][source]
        source_desc=await self.transport.json(source,'/baseline/dynamollm/describe',{})
        state['source_description']=source_desc
        for index,gpus in enumerate(t.target_layout):
            tp=len(gpus);golden=self.goldens.get(tp)
            if not golden:raise ValueError('same-target-TP golden calibration absent')
            iid='dynamo-target-'+t.transaction_id[:12]+'-'+str(index)
            spec=dict(id=iid,instance_id=iid,gpus=list(gpus),tp=tp,role='mixed',
                port=self.lifecycle.target_port+index*2)
            spec['generation']=max(row.get('generation',0) for row in state['sources'].values())+1
            spec['url']='http://127.0.0.1:'+str(spec['port'])
            state['targets'].append(spec)
            self.transport.instances[iid]=dict(spec)
            began=time.time();ready=await self.lifecycle.start(spec,dummy=True,golden=golden)
            if not isinstance(ready,dict) or ready.get('ready') is not True:
                raise RuntimeError('Dynamo target lifecycle missing ready ACK')
            target_desc=await self.transport.json(iid,'/baseline/dynamollm/describe',{})
            source_tp=source_spec['tp'];session=t.transaction_id+'-'+str(index)
            common=dict(transaction_id=t.transaction_id,session_id=session,store_host='127.0.0.1',store_port=self.store_port,
                world_size=source_tp+tp,gpu_ids=source_spec['gpus']+list(gpus),timeout_s=120)
            participants=[source,iid];state['sessions'].append(dict(session_id=session,participants=participants))
            await asyncio.gather(
                self.transport.json(source,'/baseline/dynamollm/open',dict(common,rank_offset=0)),
                self.transport.json(iid,'/baseline/dynamollm/open',dict(common,rank_offset=source_tp)))
            body=dict(transaction_id=t.transaction_id,session_id=session,operation_id=t.transaction_id,source_ranks=list(range(source_tp)),
                target_ranks=list(range(source_tp,source_tp+tp)),
                source_shapes=source_desc['ranks'][0]['parameters'],target_shapes=target_desc['ranks'][0]['parameters'],
                geometry=source_desc['ranks'][0]['geometry'],timeout_s=120,compare_target_before_copy=False)
            transfer_started=time.time()
            acks=await asyncio.gather(*(self.transport.json(participant,'/baseline/dynamollm/transfer',body)
                                       for participant in participants))
            transfer_finished=time.time()
            sent=sum(r['sent_bytes'] for r in acks[0]['ranks']);received=sum(r['received_bytes'] for r in acks[1]['ranks'])
            if sent!=received or not all(r.get('target_complete') is True for r in acks[1]['ranks']):
                raise RuntimeError('Dynamo automatic transfer lacks byte conservation / full target coverage')
            await asyncio.gather(*(self.transport.json(participant,'/baseline/dynamollm/close',
                {'session_id':session,'transaction_id':t.transaction_id})
                                   for participant in participants))
            state['sessions'].remove(state['sessions'][-1])
            await self.emit('dynamo_gpu_target_prepared',transaction_id=t.transaction_id,source=source,target=iid,
                started_s=began,finished_s=time.time(),transfer_started_s=transfer_started,
                transfer_finished_s=transfer_finished,rank_acks=acks,source_description=source_desc,
                target_description=target_desc,golden_source_sha256=golden['source_sha256'],
                stationary_weight_bytes=0,abstract_stationary_units=plan['retained_units'],
                execution_adaptation='reused GPUs lose retired-process weights; full surviving source supplies all target shards')
        return dict(instances=state['targets'],source=source)

    async def verify(self,t,prepared):
        for spec in prepared['instances']:
            result=await self.transport.json(spec['id'],'/baseline/dynamollm/verify',
                dict(operation_id=t.transaction_id,transaction_id=t.transaction_id))
            if not result.get('verified'):raise RuntimeError('Dynamo private target probe did not verify')
            state=await self.transport.state(spec['id'])
            spec['free_kv_tokens']=state['free_kv_tokens']
            await self.emit('dynamo_gpu_target_verified',transaction_id=t.transaction_id,instance_id=spec['id'],result=result)
        return dict(ready=True,outputs_valid=True)

    async def freeze(self,t):
        state=self.transactions.get(t.transaction_id,{})
        for iid in t.source_ids:
            if iid not in state.get('retired',[]):await self._gate(iid,False)

    async def drain(self,t):
        state=self.transactions.get(t.transaction_id,{})
        for iid in t.source_ids:
            if iid not in state.get('retired',[]):await self._drain(iid)

    async def release(self,t):
        # Keep one complete source model for GPU transfer. Admission is already
        # closed and requests drained; target preparation retires overlaps.
        return None

    async def activate(self,t,prepared):
        for spec in prepared['instances']:
            result=await self.transport.json(spec['id'],'/baseline/dynamollm/activate',
                dict(operation_id=t.transaction_id,transaction_id=t.transaction_id))
            if not result.get('activated'):raise RuntimeError('Dynamo target activation unacknowledged')
        return dict(activated=True)

    async def retire(self,t):
        state=self.transactions[t.transaction_id]
        for iid in t.source_ids:
            if iid not in state['retired']:
                await self._stop_confirmed(iid);state['retired'].append(iid)

    async def prepare_retirement(self,t):
        self.transactions[t.transaction_id]=dict(targets=[],retired=[],sessions=[],
            sources={iid:dict(self.lifecycle.instances[iid]) for iid in t.source_ids})

    async def park_retired(self,t):
        gpus=sorted({g for group in t.source_layout for g in group})
        if any(set(gpus)&set(row['gpus']) for row in self.lifecycle.instances.values()):
            raise RuntimeError('cannot park a GPU still owned by an active Dynamo instance')
        await self.transport.clock(gpus,210)
        await self.emit('dynamo_empty_gpu_parked',transaction_id=t.transaction_id,gpus=gpus,frequency_mhz=210)

    async def abort(self,t,prepared):
        state=self.transactions.get(t.transaction_id,{})
        for spec in reversed(state.get('targets',[])):await self._stop_confirmed(spec['id'])
        # An incomplete communicator may poison its surviving source. Do not
        # reuse it merely because the target process disappeared.
        for session in state.get('sessions',[]):
            for iid in session['participants']:
                if iid in state.get('sources',{}):
                    await self._stop_confirmed(iid)
                    if iid not in state['retired']:state['retired'].append(iid)
        return dict(target_stopped=True,source_restore_required=bool(state.get('sources')))

    async def restore(self,t):
        state=self.transactions.get(t.transaction_id,{})
        for iid in state.get('retired',[]):
            row=state['sources'][iid]
            ack=await self.lifecycle.start(row,dummy=False,golden=None)
            if not isinstance(ack,dict) or ack.get('ready') is not True:
                raise RuntimeError('Dynamo source restore lacks ready ACK')
            self.transport.instances[iid]=dict(row);self.lifecycle.instances[iid]=dict(row)
        for iid in t.source_ids:await self._gate(iid,True)
        await self.emit('dynamo_gpu_recovery',transaction_id=t.transaction_id,
            rebuilt_sources=state.get('retired',[]),mechanism='isolated checkpoint rebuild; excluded from fast transfer timings')

    async def isolate(self,t):
        state=self.transactions.get(t.transaction_id,{})
        for iid in set(t.source_ids)|{s['id'] for s in state.get('targets',[])}:await self._stop_confirmed(iid)
