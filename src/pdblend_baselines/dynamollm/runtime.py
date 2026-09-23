"""Standalone DynamoLLM serving controller over the neutral engine transport."""
import asyncio
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from functools import partial
import inspect
import hashlib
import json
import time
import uuid
from pathlib import Path

from .policy import (Request, Replica, DynamoPolicy, Epochs, PERIODS,
                     WeeklyLoadTemplate, classify, dominates, allocate_pools, shard_milp)
from .profiles import PaperProfiles, CoverageError
from .reconfiguration import Reconfiguration, Transition, weight_transfer_plan


class DynamoController:
    def __init__(self,config,transport,journal):
        self.config=config;self.transport=transport;self.journal=journal
        self.profiles=PaperProfiles.load(config['profiles'])
        self.policy=DynamoPolicy(self.profiles)
        assignments=config.get('dynamo_assignments',{})
        self.replicas={}
        for row in config['instances']:
            iid=row.get('instance_id',row.get('id'))
            if not iid or iid in self.replicas:raise ValueError('unique explicit Dynamo instance IDs required')
            if row.get('role','mixed')!='mixed':raise ValueError('Dynamo baseline requires colocated P/D instances')
            shape=assignments.get(iid,row.get('shape'))
            if shape is None:raise ValueError('explicit Dynamo instance-to-shape assignments required')
            frequency=row.get('frequency_mhz',max(self.profiles.frequencies(row['tp'])))
            self.replicas[iid]=Replica(iid,tuple(row['gpus']),row['tp'],shape,frequency,
                max_num_seqs=config.get('max_num_seqs',16))
        gpus=[g for i in self.replicas.values() for g in i.gpus]
        if len(gpus)!=len(set(gpus)):raise ValueError('Dynamo active GPU placements overlap')
        self.lock=asyncio.Lock();self.pending={};self.tasks=set();self.priorities={};self.exclusions={};self.rejected=set()
        self.stop=asyncio.Event();self.control_task=None;self.topology_task=None;self.predictor=None;self.tokenizer=None
        self.emergency_stages={};self.history=None;self.epochs=None;self.requests={};self.closed=False
        self.inflight_ids=set();self.recent_ids=deque();self.recent_set=set()
        self.cpu_executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='dynamo-predictor')
        self.control_executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='dynamo-control')
        self.kv_reservations={}
        self.topology_hooks=getattr(transport,'dynamo_topology',None)
        self.transitions=Reconfiguration(self.topology_hooks,journal) if self.topology_hooks else None
        self.shapes=config.get('dynamo_shape_demands',{})
        if self.topology_hooks and hasattr(self.topology_hooks,'bind'):self.topology_hooks.bind(self)

    async def emit(self,event,**fields):
        value=self.journal(event,**fields)
        if inspect.isawaitable(value):await value

    async def cpu(self,function,*args,**kwargs):
        future=asyncio.get_running_loop().run_in_executor(self.cpu_executor,partial(function,*args,**kwargs))
        try:
            while not future.done():await asyncio.wait((future,),timeout=.05)
            return future.result()
        except asyncio.CancelledError:
            future.cancel();raise

    async def control_cpu(self,function,*args,**kwargs):
        future=asyncio.get_running_loop().run_in_executor(self.control_executor,partial(function,*args,**kwargs))
        try:
            while not future.done():await asyncio.wait((future,),timeout=.05)
            return future.result()
        except asyncio.CancelledError:
            future.cancel();raise

    async def startup(self):
        from .predictor import BertLengthPredictor,model_identity
        injected=self.config.get('development_predictor')
        if injected is not None:
            if not self.config.get('development_allow_predictor_injection'):
                raise ValueError('predictor injection is restricted to explicit development tests')
            self.predictor=injected;self.tokenizer=self.config.get('development_tokenizer')
        else:
            directory=self.config.get('dynamo_predictor_dir');tokenizer=self.config.get('tokenizer')
            if not directory or not tokenizer:raise ValueError('all Dynamo entrypoints require verified BERT and Qwen tokenizer')
            identity=await self.cpu(model_identity,tokenizer)
            self.predictor=await self.cpu(BertLengthPredictor,directory,device='cpu',
                expected_model=identity['model'],expected_tokenizer_sha256=identity['tokenizer_sha256'])
            from transformers import AutoTokenizer
            self.tokenizer=await self.cpu(AutoTokenizer.from_pretrained,tokenizer,local_files_only=True)
        warmup=self.config.get('warmup')
        if warmup:
            work=warmup.get('requests')
            raw=(json.dumps(work,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)+'\n').encode()
            if (warmup.get('source_split')!='calibration' or not isinstance(work,list) or not work
                    or warmup.get('workload_sha256')!=hashlib.sha256(raw).hexdigest()):
                raise ValueError('Dynamo predictor warmup requires hash-bound calibration work')
            for index,row in enumerate(work):
                prompt=row.get('prompt')
                if not isinstance(prompt,(list,str)) or not prompt:raise ValueError('visible calibration warmup prompt required')
                before=time.perf_counter()
                text=prompt if isinstance(prompt,str) else await self.cpu(self.tokenizer.decode,prompt,skip_special_tokens=False)
                predicted=await self.cpu(self.predictor.predict_text,text)
                if type(predicted) is not int or predicted<=0:raise RuntimeError('invalid warmup BERT prediction')
                await self.emit('dynamo_predictor_warmup',index=index,predicted_output=predicted,
                    prompt_sha256=hashlib.sha256(text.encode()).hexdigest(),elapsed_s=time.perf_counter()-before,
                    workload_sha256=warmup['workload_sha256'],source_split='calibration')
        history_path=self.config.get('dynamo_weekly_history')
        if history_path:
            from .validation import verified_history
            value,self.history,receipt=await self.cpu(verified_history,history_path)
            await self.emit('dynamo_verified_weekly_history',**receipt)
            if self.config.get('dynamo_history_mapping'):
                from .history_mapping import MappedWeeklyLoad
                self.history=MappedWeeklyLoad(self.history,value,self.config['dynamo_history_mapping'])
                await self.emit('dynamo_history_workload_mapping',**self.history.receipt())
        if self.config.get('dynamo_require_full_mechanisms',False) and (
                self.history is None or self.transitions is None or not self.shapes
                or not getattr(self.topology_hooks,'supports_native_transitions',False)):
            raise ValueError('full Dynamo requires weekly history, shape profiling, and qualified topology adapter')
        for replica in self.replicas.values():
            await self.refresh(replica)
            await self.transport.clock(replica.gpus,replica.frequency_mhz)
        origin=time.time()
        if self.config.get('dynamo_history_reference_start_s') is not None:
            from .history_mapping import ReferenceClock
            if self.history is None:raise ValueError('reference calendar requires verified actual history')
            reference=self.config['dynamo_history_reference_start_s']
            self.history=ReferenceClock(self.history,reference_start_s=reference,wall_start_s=origin)
            await self.emit('dynamo_history_reference_clock',reference_start_s=reference,wall_start_s=origin,
                            seconds_per_real_second=1,production_prediction_accuracy_evaluation=False)
        self.epochs=Epochs(origin)
        self.control_task=asyncio.create_task(self.control())
        await self.emit('dynamo_startup',implementation='paper-reimplementation',
            hardware_qualified=False,periods_s=PERIODS,predictor_injected=injected is not None,
            weekly_template=self.history is not None,topology_adapter=self.topology_hooks is not None,
            profile_sha256=self.profiles.fingerprint)

    async def refresh(self,replica):
        state=await self.transport.state(replica.instance_id)
        if state.get('error') or state.get('runtime_error'):raise RuntimeError('Dynamo engine reports unhealthy state')
        if state.get('role','mixed')!='mixed':raise RuntimeError('Dynamo engine is not mixed')
        # A target may be excluded by the independent transaction even though
        # its engine reports accepting; do not reopen it on telemetry refresh.
        allocated=state.get('kv_allocations',{})
        pending=sum(self.kv_reservations.get(r.request_id,0) for r in replica.requests
                    if r.request_id not in allocated)
        replica.free_kv_tokens=max(0,int(state.get('free_kv_tokens',0))-pending)
        replica.generation=int(state.get('generation',replica.generation))
        return state

    async def handle(self,payload,request_id):
        if request_id in self.inflight_ids or request_id in self.recent_set:
            raise ValueError('duplicate Dynamo request ID')
        if len(self.inflight_ids)>=self.config.get('max_pending',256):raise RuntimeError('Dynamo bounded queue is full')
        self.inflight_ids.add(request_id)
        stream=self._handle(payload,request_id)
        try:
            async for event in stream:yield event
        finally:
            await stream.aclose()
            self.inflight_ids.discard(request_id);self.requests.pop(request_id,None)
            self.recent_ids.append(request_id);self.recent_set.add(request_id)
            if len(self.recent_ids)>4096:self.recent_set.discard(self.recent_ids.popleft())

    async def _handle(self,payload,request_id):
        if self.closed or self.predictor is None:raise RuntimeError('Dynamo controller is not started')
        if request_id in self.requests:raise ValueError('duplicate Dynamo request ID')
        prompt=payload.get('prompt');arrived=time.time()
        ttft=float(self.config.get('slo_ttft_s',1.));tpot=float(self.config.get('slo_tpot_s',.1))
        if isinstance(prompt,list):
            if not prompt or any(type(v) is not int or v<0 for v in prompt):raise ValueError('valid prompt tokens required')
            if self.tokenizer is None:raise ValueError('Qwen tokenizer required for visible token prompt prediction')
            tokens=len(prompt);text=await self.cpu(self.tokenizer.decode,prompt,skip_special_tokens=False)
        elif isinstance(prompt,str) and prompt:
            if self.tokenizer is None:raise ValueError('Qwen tokenizer required for input length')
            tokens=len(await self.cpu(self.tokenizer.encode,prompt,add_special_tokens=False));text=prompt
        else:raise ValueError('Dynamo prompt must be visible text or token IDs')
        prediction_started=time.perf_counter()
        predicted=await asyncio.wait_for(self.cpu(self.predictor.predict_text,text),
                                        max(.001,arrived+ttft-time.time()))
        if type(predicted) is not int or predicted<=0:raise RuntimeError('invalid Dynamo output prediction')
        # max_tokens is execution/KV storage limit, never the predictor label.
        limit=payload.get('max_tokens')
        if type(limit) is not int or limit<1:raise ValueError('positive execution token limit required')
        request=Request(request_id,tokens,predicted,arrived,ttft,tpot)
        self.requests[request_id]=request;self.pending[request_id]=request
        await self.emit('dynamo_prediction',request_id=request_id,predicted_output=predicted,
                        prediction_s=time.perf_counter()-prediction_started,shape=classify(tokens,predicted))
        chosen=None;finished=False;stream=None
        try:
            while chosen is None:
                if self.closed or request_id in self.rejected:raise RuntimeError('Dynamo request rejected; retry elsewhere')
                if time.time()>request.deadline():raise asyncio.TimeoutError('Dynamo admission SLO expired')
                replicas=tuple(self.replicas.values())
                states=await asyncio.gather(*(self.refresh(i) for i in replicas),return_exceptions=True)
                eligible=[i for i,s in zip(replicas,states) if not isinstance(s,BaseException)
                          and s.get('accepting',False)
                          and i.free_kv_tokens>=tokens+limit]
                async with self.lock:
                    same_class=[rid for rid,q in self.pending.items() if classify(q.input_tokens,q.predicted_output)==classify(tokens,predicted)]
                    head=min(same_class,key=lambda rid:(self.priorities.get(rid,float('inf')),
                        self.pending[rid].arrival_s,rid),default=None)
                    if head==request_id:
                        excluded=self.exclusions.get(request_id,set())
                        route=self.policy.route(request,[i for i in eligible
                            if i.instance_id not in excluded],time.time())
                        # Emergency rerouting is a preference for another
                        # replica, not a permanent ban. If that destination
                        # becomes unavailable, reconsider healthy replicas
                        # through the unchanged capacity/profile/SLO policy.
                        if route is None and excluded:
                            route=self.policy.route(request,eligible,time.time())
                            if route:self.exclusions.pop(request_id,None)
                        if route:
                            chosen=self.replicas[route.instance_id]
                            chosen.requests.append(request);self.pending.pop(request_id,None)
                            self.kv_reservations[request_id]=tokens+limit
                            chosen.free_kv_tokens-=tokens+limit
                if chosen is None:await asyncio.sleep(.005)
            request.started=True
            engine_payload=dict(payload,request_id=request_id,stream=True)
            await self.emit('dynamo_route',request_id=request_id,instance_id=chosen.instance_id,
                            shape=chosen.shape,frequency_mhz=chosen.frequency_mhz)
            async def next_event(iterator):
                return await asyncio.wait_for(anext(iterator),self.config.get('engine_request_timeout_s',120.))
            stream=self.transport.stream(chosen.instance_id,engine_payload).__aiter__()
            while True:
                try:event=await next_event(stream)
                except StopAsyncIteration:break
                if event.get('error'):raise RuntimeError(str(event['error']))
                new_ids=event.get('token_ids',[])
                if new_ids or event.get('token_index',0)>request.emitted:
                    now=time.time();request.first_token_s=request.first_token_s or now
                    request.last_token_s=now
                    request.emitted=max(request.emitted+len(new_ids),event.get('token_index',0))
                if any(c.get('finish_reason') is not None for c in event.get('choices',[])):finished=True
                yield event
            if not finished:raise RuntimeError('Dynamo engine stream ended without terminal completion')
        finally:
            self.pending.pop(request_id,None);self.priorities.pop(request_id,None);self.exclusions.pop(request_id,None)
            self.rejected.discard(request_id)
            if chosen:
                try:
                    if stream is not None and hasattr(stream,'aclose'):await stream.aclose()
                    if not finished:await self.transport.cancel(chosen.instance_id,request_id)
                finally:
                    async with self.lock:
                        if request in chosen.requests:chosen.requests.remove(request)
                        self.kv_reservations.pop(request_id,None)
            await self.emit('dynamo_request_done',request_id=request_id,finished=finished,
                            emitted=request.emitted,elapsed_s=time.time()-arrived)

    async def emergency_tick(self,now):
        for instance in tuple(self.replicas.values()):
            if not instance.accepting:continue
            pending=[r for r in self.pending.values()
                     if instance.instance_id not in self.exclusions.get(r.request_id,set())
                     and dominates(instance.shape,classify(r.input_tokens,r.predicted_output))]
            view=replace(instance,requests=[*instance.requests,*pending])
            decision=self.policy.emergency(view,now,previous_stage=self.emergency_stages.get(instance.instance_id,0))
            self.emergency_stages[instance.instance_id]=decision['stage']
            for rid in decision['reorder']:self.priorities[rid]=self.requests[rid].deadline()
            if decision['frequency'] is not None and instance.frequency_mhz!=decision['frequency']:
                await self.transport.clock(instance.gpus,decision['frequency'])
                instance.frequency_mhz=decision['frequency']
            for rid in decision['reroute']:
                # Exclude a source only when a distinct, nonexcluded target
                # passes the ordinary routing policy. Testing mere existence
                # let two busy replicas exclude each other forever.
                async with self.lock:
                    request=self.pending.get(rid)
                    excluded=self.exclusions.get(rid,set())
                    targets=[other for other in self.replicas.values()
                             if other.instance_id!=instance.instance_id
                             and other.instance_id not in excluded
                             and other.shape==instance.shape and other.accepting]
                    if request is not None and self.policy.route(request,targets,now):
                        self.exclusions.setdefault(rid,set()).add(instance.instance_id)
            self.rejected.update(decision['reject'])
            if decision['stage']:await self.emit('dynamo_emergency',instance_id=instance.instance_id,**decision)

    async def reconfigure(self,operation,now):
        if self.history is None or self.transitions is None or not self.shapes:
            await self.emit('dynamo_control_epoch',operation=operation,executed=False,
                reason='independent weekly history, shape coverage, or GPU topology adapter unavailable')
            return
        rates=self.history.forecast(now,PERIODS[operation])
        assignments={}
        for i in self.replicas.values():assignments.setdefault(i.shape,[]).append(i)
        if operation=='ScaleInst':
            reference_tp=self.config.get('dynamo_reference_tp',8)
            allocation_tp=self.config.get('dynamo_allocation_tp',reference_tp)
            capacities={}
            for shape,demand in self.shapes.items():
                options=await self.control_cpu(self.profiles.configurations,**demand)
                capacities[shape]=max((o.capacity_rps for o in options if o.tp==allocation_tp),default=0.)
            pools=allocate_pools(rates,capacities,reference_tp=allocation_tp,
                gpu_budget=len(self.config.get('node_gpus',range(8))))
            await self._apply_cluster(pools)
            return
        else:
            from .cluster import pool_forecasts
            assigned_rates=pool_forecasts(rates,assignments)
            pools={shape:dict(gpus=sum(i.tp for i in instances),rate_rps=assigned_rates[shape])
                   for shape,instances in assignments.items()}
        await self._apply_pools(pools,operation=operation)

    async def _apply_cluster(self,pools):
        """Reallocate nine pools together; preserve every unchanged replica."""
        from .cluster import plan_reallocation,admission_reason
        from .transition_costs import validate_cost
        desired={};options_by_shape={}
        for shape,pool in pools.items():
            demand=self.shapes.get(shape)
            if not demand:raise CoverageError('ScaleInst target lacks frozen calibration shape')
            options=await self.control_cpu(self.profiles.configurations,**demand)
            options_by_shape[shape]=options
            desired[shape]=await self.control_cpu(shard_milp,options,pool['gpus'],pool['rate_rps'])
        current=list(self.replicas.values());plan=plan_reallocation(current,desired)
        if not plan['changed']:
            await self.emit('dynamo_control_epoch',operation='ScaleInst',executed=False,reason='global inventory unchanged')
            return
        source=[i for i in current if i.instance_id in plan['source_ids']]
        if not source:raise CoverageError('ScaleInst new pool has no allocated source model for GPU transfer')
        demand=dict(input_tokens=max(self.shapes[i.shape]['input_tokens'] for i in source),
                    output_tokens=max(self.shapes[i.shape]['output_tokens'] for i in source),
                    batch=max(i.max_num_seqs for i in source))
        for target in plan['targets']:
            shape=self.shapes[target['shape']]
            demand['input_tokens']=max(demand['input_tokens'],shape['input_tokens'])
            demand['output_tokens']=max(demand['output_tokens'],shape['output_tokens'])
        candidates=[c for c in self.config.get('dynamo_transition_costs',[])
                    if c.get('source_tps')==plan['source_tps'] and c.get('target_tps')==plan['target_tps']]
        if not candidates:raise CoverageError('ScaleInst global source/target layout has no measured transition cost')
        for cost in candidates:await self.control_cpu(validate_cost,cost,demand,model_id=self.config['model_id'])
        cost=max(candidates,key=lambda c:c['energy_j'])
        capacity=0.;old_power=0.;new_power=0.;required=sum(pool['rate_rps'] for pool in pools.values())
        for instance in current:
            options=options_by_shape.get(instance.shape)
            if options is None:options=await self.control_cpu(self.profiles.configurations,**self.shapes[instance.shape])
            choices=[c for c in options if c.tp==instance.tp]
            if not choices:raise CoverageError('ScaleInst current capacity lacks independent measured profile')
            capacity+=max(c.capacity_rps for c in choices);old_power+=max(c.power_w for c in choices)
        for choices in desired.values():new_power+=sum(c.power_w*n for c,n in choices)
        savings=(old_power-new_power)*max(0.,PERIODS['ScaleInst']-cost['duration_s'])
        reason=admission_reason(required_rate=required,current_capacity=capacity,savings_j=savings,overhead_j=cost['energy_j'])
        # A class with no compatible current pool requires capacity even if an
        # unrelated specialised pool contributes to the aggregate total.
        for shape,pool in pools.items():
            compatible=[i for i in current if dominates(i.shape,shape)]
            compatible_capacity=0.
            for instance in compatible:
                choices=[c for c in options_by_shape[shape] if c.tp==instance.tp]
                compatible_capacity+=max((c.capacity_rps for c in choices),default=0.)
            if pool['rate_rps']>compatible_capacity+1e-9:reason='required_capacity'
        if reason is None:
            await self.emit('dynamo_control_epoch',operation='ScaleInst',executed=False,reason='global change does not amortize cost',plan=plan)
            return
        occupied={g for i in current if i.instance_id in plan['retained_ids'] for g in i.gpus}
        available=[g for g in self.config.get('node_gpus',range(8)) if g not in occupied]
        target_tps=[r['tp'] for r in plan['targets']]
        targets=self.topology_hooks.plan_layout(source,target_tps,available) if target_tps else []
        if targets:targets=weight_transfer_plan(tuple(i.gpus for i in source),tuple(targets))['target_rank_gpus']
        transition=Transition(uuid.uuid4().hex,tuple(i.instance_id for i in source),tuple(i.gpus for i in source),tuple(targets),
            overlap_memory_qualified=bool(cost.get('overlap_memory_qualified')),
            savings_j=savings if reason=='amortized_energy' else None,
            overhead_j=cost['energy_j'] if reason=='amortized_energy' else None,timeout_s=240.,
            target_shapes=tuple(r['shape'] for r in plan['targets']))
        async def commit(prepared):
            rows=prepared.get('instances',[])
            if len(rows)!=len(plan['targets']):raise RuntimeError('ScaleInst target count ACK differs from global plan')
            replacements={}
            for row,selection,gpus in zip(rows,plan['targets'],targets):
                iid=row.get('instance_id',row.get('id'))
                if (not iid or iid in replacements or tuple(row['gpus'])!=tuple(gpus)
                        or row['tp']!=selection['tp']):raise RuntimeError('ScaleInst physical target ACK differs')
                existing=self.replicas.get(iid)
                if existing is not None:
                    if (not prepared.get('intermediate_serving') or existing.gpus!=tuple(gpus)
                            or existing.tp!=selection['tp'] or existing.shape!=selection['shape']):
                        raise RuntimeError('ScaleInst intermediate target identity differs')
                    replacements[iid]=existing
                    continue
                replacements[iid]=Replica(iid,tuple(gpus),selection['tp'],selection['shape'],
                    row.get('frequency_mhz',max(self.profiles.frequencies(selection['tp']))),
                    free_kv_tokens=row.get('free_kv_tokens',0),max_num_seqs=self.config.get('max_num_seqs',16),
                    generation=row['generation'])
            async with self.lock:
                for iid in plan['source_ids']:self.replicas.pop(iid,None)
                self.replicas.update(replacements)
        if plan['retire_only']:
            result=await self.transitions.retire_only(transition,commit=commit)
        else:result=await self.transitions.execute(transition,commit=commit)
        await self.emit('dynamo_control_epoch',operation='ScaleInst',executed=True,result=result,plan=plan,
            weekly_predictor_used=True,admission_reason=reason,predicted_savings_j=savings,measured_cost_j=cost['energy_j'])

    async def initialize_pool(self,shape,rate_rps,gpu_budget):
        """Physical initial ScaleShard setup; never a periodic/weekly action.

        Uses the independent declared average demand, measured shape coverage
        and the same MILP as later epochs. Measured transition overhead is
        collected here, so no existing cost model or week trace is invented.
        """
        if self.transitions is None:raise ValueError('initial topology requires a physical adapter')
        if rate_rps<=0 or type(gpu_budget) is not int:raise ValueError('positive independent initialization demand required')
        return await self._apply_pools({shape:dict(gpus=gpu_budget,rate_rps=rate_rps)},operation='ScaleShard',initialization=True)

    async def _apply_pools(self,pools,*,operation,initialization=False):
        assignments={}
        for i in self.replicas.values():assignments.setdefault(i.shape,[]).append(i)
        for shape,pool in pools.items():
            demand=self.shapes.get(shape)
            if not demand:raise CoverageError('Dynamo target pool lacks frozen shape demand')
            options=await self.control_cpu(self.profiles.configurations,**demand)
            counts=await self.control_cpu(shard_milp,options,pool['gpus'],pool['rate_rps'])
            target_tps=tuple(sorted(c.tp for c,n in counts for _ in range(n)))
            source=assignments.get(shape,[])
            if target_tps==tuple(sorted(i.tp for i in source)):continue
            source_tps=tuple(sorted(i.tp for i in source))
            costs=[c for c in self.config.get('dynamo_transition_costs',[])
                   if tuple(c['source_tps'])==source_tps and tuple(c['target_tps'])==target_tps
                   and c.get('measurement')=='hardware' and c.get('source_sha256')]
            if not costs and not initialization:raise CoverageError('Dynamo target transition lacks independent measured overhead')
            if not initialization:
                from .transition_costs import validate_cost
                cost_demand=dict(demand,batch=max(i.max_num_seqs for i in source))
                for value in costs:await self.control_cpu(validate_cost,value,cost_demand,model_id=self.config['model_id'])
            cost=max(costs,key=lambda c:c['energy_j']) if costs else None
            old_power=0.
            for i in source:
                candidates=[c for c in options if c.tp==i.tp]
                if not candidates:raise CoverageError('Dynamo source power lacks measured coverage')
                old_power+=max(c.power_w for c in candidates)
            new_power=sum(c.power_w*n for c,n in counts)
            savings=(old_power-new_power)*max(0.,PERIODS[operation]-cost['duration_s']) if cost else None
            current_capacity=sum(max(c.capacity_rps for c in options if c.tp==i.tp) for i in source)
            required_capacity=pool['rate_rps']>current_capacity+1e-9
            if cost and not required_capacity and savings<=cost['energy_j']:continue
            occupied={g for instances in assignments.values() for i in instances if i not in source for g in i.gpus}
            available=[g for g in self.config.get('node_gpus',range(8)) if g not in occupied]
            if sum(target_tps)>len(available):raise CoverageError('Dynamo cannot borrow another pool GPU budget')
            targets=[];cursor=0
            for tp in target_tps:targets.append(tuple(available[cursor:cursor+tp]));cursor+=tp
            if hasattr(self.topology_hooks,'plan_layout'):
                targets=self.topology_hooks.plan_layout(source,target_tps,available)
            targets=weight_transfer_plan(tuple(i.gpus for i in source),tuple(targets))['target_rank_gpus']
            t=Transition(uuid.uuid4().hex,tuple(i.instance_id for i in source),tuple(i.gpus for i in source),
                tuple(targets),overlap_memory_qualified=bool(cost and cost.get('overlap_memory_qualified')) or bool(
                    initialization and self.config.get('development_disjoint_prepare')),
                savings_j=None if required_capacity else savings,
                overhead_j=cost['energy_j'] if cost and not required_capacity else None,timeout_s=240.,
                target_shapes=tuple(shape for _ in targets))
            async def commit(prepared):
                rows=prepared.get('instances')
                if not rows:raise RuntimeError('Dynamo topology adapter omitted target instance endpoints')
                new={}
                for row in rows:
                    iid=row.get('instance_id',row.get('id'))
                    if not iid or iid in new:raise RuntimeError('Dynamo target identities invalid')
                    existing=self.replicas.get(iid)
                    if existing is not None:
                        if (not prepared.get('intermediate_serving') or existing.gpus!=tuple(row['gpus'])
                                or existing.tp!=row['tp'] or existing.shape!=shape):
                            raise RuntimeError('ScaleShard intermediate target identity differs')
                        new[iid]=existing
                        continue
                    new[iid]=Replica(iid,tuple(row['gpus']),row['tp'],shape,
                        row.get('frequency_mhz',max(self.profiles.frequencies(row['tp']))),
                        free_kv_tokens=row.get('free_kv_tokens',0),max_num_seqs=self.config.get('max_num_seqs',16),
                        generation=row['generation'])
                if sorted(tuple(sorted(i.gpus)) for i in new.values())!=sorted(tuple(sorted(g)) for g in targets):
                    raise RuntimeError('Dynamo target ACK does not match planned GPU layout')
                async with self.lock:
                    for i in source:self.replicas.pop(i.instance_id,None)
                    self.replicas.update(new)
            try:result=await self.transitions.execute(t,commit=commit)
            except BaseException:
                # Only acknowledged restore permits routing to the source.
                for i in source:
                    if not set(i.gpus)&self.transitions.quarantined:
                        state=await self.transport.state(i.instance_id)
                        i.accepting=bool(state.get('accepting'))
                raise
            await self.emit('dynamo_initialization_topology' if initialization else 'dynamo_control_epoch',
                operation=operation,executed=True,result=result,initialization=initialization,
                weekly_predictor_used=not initialization)
            # Stagger physical changes: refreshed state/forecast for next epoch.
            return result
        await self.emit('dynamo_control_epoch',operation=operation,executed=False,reason='no amortized configuration change')

    async def control(self):
        try:
            while not self.stop.is_set():
                now=time.time()
                try:
                    await self.emergency_tick(now)
                    for operation in self.epochs.due(now):
                        if operation=='ScaleFreq':
                            changed=[]
                            for instance in tuple(self.replicas.values()):
                                if not instance.accepting:continue
                                frequency=self.policy.frequency(instance,now)
                                if frequency!=instance.frequency_mhz:
                                    await self.transport.clock(instance.gpus,frequency)
                                    instance.frequency_mhz=frequency;changed.append(instance.instance_id)
                            await self.emit('dynamo_control_epoch',operation=operation,executed=bool(changed),instances=changed)
                        elif self.topology_task is None or self.topology_task.done():
                            if self.topology_task is not None:
                                error=self.topology_task.exception()
                                if error:await self.emit('dynamo_topology_failure',error=repr(error))
                            self.topology_task=asyncio.create_task(self.reconfigure(operation,now))
                        else:await self.emit('dynamo_control_epoch',operation=operation,executed=False,reason='staggered transition active')
                except Exception as exc:await self.emit('dynamo_control_failure',error=repr(exc))
                try:await asyncio.wait_for(self.stop.wait(),.1)
                except asyncio.TimeoutError:pass
        except asyncio.CancelledError:pass

    async def close(self):
        self.closed=True;self.stop.set()
        if self.control_task:await self.control_task
        if self.topology_task:
            self.topology_task.cancel()
            await asyncio.gather(self.topology_task,return_exceptions=True)
        await asyncio.gather(*(self.transport.cancel(i.instance_id,r.request_id)
                              for i in self.replicas.values() for r in i.requests),return_exceptions=True)
        self.cpu_executor.shutdown(wait=True,cancel_futures=True)
        self.control_executor.shutdown(wait=True,cancel_futures=True)
