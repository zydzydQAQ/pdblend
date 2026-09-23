"""Complete-donor stages for an occupied multi-instance GPU node.

No model cache or PDblend resident pool is used. Every stage retires only GPUs
outside a complete donor, receives genuine weights into a new dummy engine,
checks a fixed same-TP output, then exposes the intermediate measured capacity.
The single TP8-on-eight-GPUs geometry cannot start this disjoint protocol.
"""
from dataclasses import replace
import hashlib
import time

from .gpu_topology import GpuTopologyHooks
from .policy import Replica,SHAPES
from .reconfiguration import Transition,weight_transfer_plan


def plan_relay(sources,target_layout):
    live={row['id']:dict(row) for row in sources};flat=[g for row in sources for g in row['gpus']]
    targets=[tuple(g) for g in target_layout];target_flat=[g for group in targets for g in group]
    if (len(live)!=len(sources) or len(flat)!=len(set(flat)) or len(target_flat)!=len(set(target_flat))
            or any(len(g) not in (1,2,4,8) for g in targets)
            or any(row['tp']!=len(row['gpus']) for row in sources)):
        raise ValueError('unique physical source and final target layouts required')
    reserved=set(live);labels={}
    for index in range(len(targets)):
        label='target-'+str(index)
        while label in reserved:label='_'+label
        reserved.add(label);labels[index]=label
    stages=[];pending=set(range(len(targets)))
    while pending:
        candidates=[]
        for index in sorted(pending):
            gpus=set(targets[index]);donors=[r for r in live.values() if gpus.isdisjoint(r['gpus'])]
            if not donors:continue
            donor=min(donors,key=lambda r:(r['tp'],r['id']))
            victims=sorted(iid for iid,row in live.items() if gpus.intersection(row['gpus']))
            candidates.append((sum(live[i]['tp'] for i in victims),index,donor,victims))
        if not candidates:raise ValueError('no complete disjoint donor for the next physical stage')
        _,index,donor,victims=min(candidates,key=lambda row:row[:2]);pending.remove(index)
        stages.append(dict(target_index=index,target_id=labels[index],target_gpus=list(targets[index]),
            donor=donor['id'],donor_gpus=list(donor['gpus']),retire_ids=victims,
            surviving_ids=sorted(set(live)-set(victims))))
        for iid in victims:del live[iid]
        live[labels[index]]=dict(id=labels[index],gpus=list(targets[index]),tp=len(targets[index]))
    return stages


class _PortView:
    def __init__(self,lifecycle,offset):self.base=lifecycle;self.offset=offset
    @property
    def target_port(self):
        port=self.base.target_port+self.offset
        used={r['port'] for r in self.base.instances.values()}
        while port in used:port+=2
        if port >= self.base.base_port+80:
            raise RuntimeError('Dynamo relay exhausted its owned HTTP port window')
        return port
    def __getattr__(self,name):return getattr(self.base,name)


class StagedGpuTopologyHooks(GpuTopologyHooks):
    hardware_qualified=False
    def plan_layout(self,source,target_tps,available):
        try:return super().plan_layout(source,target_tps,available)
        except ValueError:
            result=[];cursor=0
            for tp in target_tps:result.append(tuple(available[cursor:cursor+tp]));cursor+=tp
            if cursor>len(available):raise ValueError('relay target exceeds owned physical GPU budget')
            plan_relay([dict(id=i.instance_id,gpus=i.gpus,tp=i.tp) for i in source],result)
            return result

    def _stage_hook(self,index):
        hook=GpuTopologyHooks(self.transport,_PortView(self.lifecycle,index*2),self.journal,
            goldens=self.goldens,store_port=self.store_port+index,timeout_s=self.timeout_s)
        hook.bind(self.controller);return hook

    async def _layout_event(self,t,phase,index,**fields):
        rows=[]
        for i in self.controller.replicas.values():
            if not i.accepting:continue
            # This view is explicitly reduced to the engines still accepting.
            # It does not inherit the capacity of drained/retired old instances.
            rows.append(dict(id=i.instance_id,gpus=list(i.gpus),tp=i.tp,shape=i.shape,
                active_requests=len(i.requests),max_num_seqs=i.max_num_seqs,free_kv_tokens=i.free_kv_tokens))
        await self.emit('dynamo_relay_layout',transaction_id=t.transaction_id,phase=phase,stage_index=index,
            active_instances=rows,active_gpu_count=sum(r['tp'] for r in rows),
            preserves_previous_capacity_or_slo=False,hardware_qualified=False,**fields)

    async def prepare(self,t,plan):
        if self.controller is None:raise ValueError('relay requires the independent live controller')
        if len(t.target_shapes)!=len(t.target_layout) or any(s not in SHAPES for s in t.target_shapes):
            raise ValueError('relay targets require frozen class assignments')
        originals={iid:dict(self.lifecycle.instances[iid]) for iid in t.source_ids}
        stages=plan_relay([dict(row,id=iid) for iid,row in originals.items()],t.target_layout)
        state=dict(targets=[],retired=[],sources=originals,sessions=[],stages=[],
            original_replicas={iid:self.controller.replicas[iid] for iid in t.source_ids},stage_plan=stages)
        self.transactions[t.transaction_id]=state;identities={iid:iid for iid in t.source_ids}
        for index,item in enumerate(stages):
            started=time.time();donor=identities[item['donor']];victims=[identities[i] for i in item['retire_ids']]
            serving=[iid for iid in [donor,*victims] if iid in self.lifecycle.instances]
            stage_id=hashlib.sha256((t.transaction_id+':'+str(index)).encode()).hexdigest()[:32]
            st=Transition(stage_id,tuple(serving),tuple(tuple(self.lifecycle.instances[i]['gpus']) for i in serving),
                (tuple(item['target_gpus']),),overlap_memory_qualified=True,timeout_s=t.timeout_s,
                target_shapes=(t.target_shapes[item['target_index']],))
            hook=self._stage_hook(item['target_index']);entry=dict(hook=hook,transition=st,prepared=None)
            state['stages'].append(entry)
            await self._layout_event(t,'before_quiesce',index,planned_retire_ids=victims,donor=donor,started_s=started)
            # Quiesce and retire only overlapping old engines. State is kept in
            # the outer transaction so partial failure can restore all originals.
            for iid in victims:
                await self._gate(iid,False);await self._drain(iid);await self._stop_confirmed(iid)
                if iid in originals:state['retired'].append(iid)
                async with self.controller.lock:self.controller.replicas.pop(iid,None)
            await self._layout_event(t,'capacity_reduced',index,donor=donor,started_s=started,confirmed_s=time.time())
            # Victims are already proven absent; the stage copies only from its
            # complete surviving donor and never resurrects stale endpoints.
            st=replace(st,source_ids=(donor,),source_layout=(tuple(self.lifecycle.instances[donor]['gpus']),))
            entry['transition']=st
            prepared=await hook.prepare(st,weight_transfer_plan(st.source_layout,st.target_layout));entry['prepared']=prepared
            state['targets'].extend(prepared['instances'])
            checked=await hook.verify(st,prepared)
            if not checked.get('ready') or not checked.get('outputs_valid'):raise RuntimeError('relay private output ACK missing')
            activated=await hook.activate(st,prepared)
            if not activated.get('activated'):raise RuntimeError('relay target activation unconfirmed')
            spec=prepared['instances'][0];iid=spec['id'];shape=t.target_shapes[item['target_index']]
            replica=Replica(iid,tuple(spec['gpus']),spec['tp'],shape,
                max(self.controller.profiles.frequencies(spec['tp'])),free_kv_tokens=spec['free_kv_tokens'],
                max_num_seqs=self.controller.config.get('max_num_seqs',16),generation=spec['generation'])
            before=time.perf_counter()
            async with self.controller.lock:
                if iid in self.controller.replicas:raise RuntimeError('relay target instance identity collision')
                self.controller.replicas[iid]=replica
            pause=time.perf_counter()-before;identities[item['target_id']]=iid
            await self._layout_event(t,'intermediate_target_serving',index,donor=donor,target=iid,
                started_s=started,confirmed_s=time.time(),stage_duration_s=time.time()-started,commit_pause_s=pause)
        ordered={tuple(spec['gpus']):spec for spec in state['targets']}
        return dict(instances=[ordered[tuple(g)] for g in t.target_layout],intermediate_serving=True)

    async def verify(self,t,prepared):
        if not prepared.get('intermediate_serving'):raise RuntimeError('staged targets not verified/activated')
        for row in prepared['instances']:
            current=await self.transport.state(row['id'])
            if not current.get('accepting') or current.get('error') or not current.get('dynamo_weights_ready'):
                raise RuntimeError('intermediate relay target is no longer healthy')
        return dict(ready=True,outputs_valid=True)

    async def activate(self,t,prepared):
        # Every target already has explicit GPU output + activation ACK. The
        # outer commit preserves its current requests and generation objects.
        return dict(activated=True,intermediate_serving=True)

    async def abort(self,t,prepared):
        state=self.transactions.get(t.transaction_id)
        if state is None:return dict(target_stopped=True,source_restore_required=False)
        target_ids={spec['id'] for spec in state.get('targets',[])}
        for entry in state.get('stages',[]):
            hs=entry['hook'].transactions.get(entry['transition'].transaction_id,{})
            target_ids.update(spec['id'] for spec in hs.get('targets',[]))
        # Execution is revoked and physical process absence confirmed before
        # any source restoration. Timeout never grants a replacement authority.
        for iid in target_ids:
            replica=self.controller.replicas.get(iid)
            if replica:replica.accepting=False
        for iid in target_ids:
            if iid in self.lifecycle.instances or iid in self.transport.instances:await self._stop_confirmed(iid)
            async with self.controller.lock:self.controller.replicas.pop(iid,None)
        # An incomplete NCCL group can poison an otherwise surviving donor.
        for entry in state.get('stages',[]):
            hs=entry['hook'].transactions.get(entry['transition'].transaction_id,{})
            for session in hs.get('sessions',[]):
                for iid in session['participants']:
                    if iid in state.get('sources',{}) and iid in self.lifecycle.instances:
                        await self._stop_confirmed(iid)
                        if iid not in state['retired']:state['retired'].append(iid)
        state['targets_confirmed_stopped']=True
        await self._layout_event(t,'targets_confirmed_stopped',-1,target_ids=sorted(target_ids))
        return dict(target_stopped=True,source_restore_required=True)

    async def restore(self,t):
        state=self.transactions[t.transaction_id]
        if not state.get('targets_confirmed_stopped'):
            raise RuntimeError('relay target stop confirmation required before source restore')
        for iid,row in state['sources'].items():
            if iid not in self.lifecycle.instances:
                ack=await self.lifecycle.start(row,dummy=False,golden=None)
                if ack.get('ready') is not True:raise RuntimeError('relay source restore lacks explicit ready ACK')
                self.transport.instances[iid]=dict(row);self.lifecycle.instances[iid]=dict(row)
            await self.transport.clock(row['gpus'],state['original_replicas'][iid].frequency_mhz)
            result=await self.transport.json(iid,'/baseline/dynamollm/resume',{})
            if result.get('accepting') is not True:raise RuntimeError('relay source restore admission unacknowledged')
        async with self.controller.lock:
            for iid,replica in state['original_replicas'].items():
                replica.accepting=True;self.controller.replicas[iid]=replica
        await self._layout_event(t,'original_layout_restored',-1,recovery='confirmed checkpoint rebuild; not fast relay')


    async def isolate(self,t):
        state=self.transactions.get(t.transaction_id,{})
        identities=set(t.source_ids)|{spec['id'] for spec in state.get('targets',[])}
        for entry in state.get('stages',[]):
            hs=entry['hook'].transactions.get(entry['transition'].transaction_id,{})
            identities.update(spec['id'] for spec in hs.get('targets',[]))
        errors=[]
        for iid in identities:
            replica=self.controller.replicas.get(iid)
            if replica:replica.accepting=False
        for iid in identities:
            if iid in self.lifecycle.instances or iid in self.transport.instances:
                try:await self._stop_confirmed(iid)
                except BaseException as exc:errors.append(dict(instance_id=iid,error=repr(exc)))
        await self._layout_event(t,'isolated',-1,unconfirmed_stops=errors)
        if errors:raise RuntimeError('relay isolation retains unconfirmed GPU endpoint: '+repr(errors))
