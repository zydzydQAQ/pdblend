"""Official DistServe SimPy core with explicit local model/profile adaptation.

Worker/Request/Scheduler/DisaggCluster are source-identical apart from import
names. Adapter changes: Qwen measured per-stage timing, calibration histories,
and mapping API total outputs to decode steps. This is simulation, not GPU proof.
"""
import math
import random
import re
import threading

import numpy as np

from .policy import positive_int

_RNG_LOCK=threading.Lock()


class MeasuredLatency:
    """Independent immutable measured surface; no PDblend O(1) index or fallback."""
    def __init__(self, points):
        checked=[]
        for point in points:
            point=dict(point)
            if point.get('role') not in ('prefill','decode'):raise ValueError('P/D measurement required')
            for key in ('tp','pp','batch','max_input_tokens','max_context_tokens'):
                positive_int(point.get(key),key)
            delay=point.get('stage_latency_ms')
            if not isinstance(delay,(int,float)) or not math.isfinite(delay) or delay<=0:
                raise ValueError('positive measured stage_latency_ms required')
            if not re.fullmatch('[0-9a-f]{64}',point.get('source_sha256','')):
                raise ValueError('raw measured source SHA256 required')
            if point['pp']==1:point.setdefault('stage_index',0)
            if type(point.get('stage_index')) is not int or not 0<=point['stage_index']<point['pp']:
                raise ValueError('measured physical pipeline stage_index required')
            if point['pp']>1 and point.get('timing_scope')!='stage_service_including_pp_and_host':
                raise ValueError('PP wire and host stage service timing required; rank-local CUDA is insufficient')
            checked.append(point)
        # Match stable first-table ordering, never borrow a different TP or PP.
        self.points=tuple(sorted(checked,key=lambda p:(p['batch'],p['max_context_tokens'],p['max_input_tokens'])))

    def __call__(self, role, tp, pp, batch, inputs, contexts):
        if pp!=1:raise ValueError('measured latency coverage requires an explicit pipeline stage identity')
        return self.stage_latency(role,tp,pp,0,batch,inputs,contexts)

    def stage_latency(self,role,tp,pp,stage,batch,inputs,contexts):
        max_input=max(inputs,default=0);max_context=max(contexts,default=0)
        for point in self.points:
            if (point['role'],point['tp'],point['pp'],point['stage_index'])!=(role,tp,pp,stage):continue
            if point['batch']<batch or point['max_input_tokens']<max_input or point['max_context_tokens']<max_context:continue
            return float(point['stage_latency_ms'])
        raise ValueError(f'measured latency coverage missing: role={role}, TP={tp}, PP={pp}, batch={batch}, input={max_input}, context={max_context}')


class OfficialSimulator:
    def __init__(self, records, *, latency, capacities, seed=701, sample_size=None,
                 coefficient_of_variation=1., max_events=1000000, provenance=None):
        self.records=tuple((positive_int(r[0],'input_tokens'),positive_int(r[1],'output_tokens')) for r in records)
        if not self.records:raise ValueError('calibration history is empty')
        if provenance is not None:
            if provenance.get('split')!='calibration' or not re.fullmatch('[0-9a-f]{64}',provenance.get('source_sha256','')):
                raise ValueError('independent calibration provenance required')
        self.provenance=provenance
        self.latency=latency;self.capacities={tuple(k):positive_int(v,'capacity') for k,v in capacities.items()}
        self.seed=seed;self.sample_size=len(self.records) if sample_size is None else positive_int(sample_size,'sample_size')
        if self.sample_size>len(self.records):raise ValueError('official sampling is without replacement')
        if not math.isfinite(coefficient_of_variation) or coefficient_of_variation<=0:
            raise ValueError('positive coefficient of variation required')
        self.cv=coefficient_of_variation;self.max_events=positive_int(max_events,'max_events')

    def __call__(self, config, rate):
        if not math.isfinite(rate) or rate<=0:raise ValueError('positive simulated rate required')
        try:import simpy
        except ImportError as exc:raise RuntimeError('install baseline simulator dependency simpy==4.1.1') from exc
        from ._sim.base.request import Request
        from ._sim.base.scheduler import put_requests_with_interarrivals
        from ._sim.clusters.disagg import DisaggCluster
        from ._sim.estimators.time_estimator import LATENCY
        cross,tp,pp,td,pd=config;pp*=cross;pd*=cross
        for pair in ((tp,pp),(td,pd)):
            if pair not in self.capacities:raise ValueError(f'measured capacity absent for TP/PP={pair}')
        with _RNG_LOCK:
            old_rng=random.getstate();random.seed(self.seed)
            token=LATENCY.set(self.latency)
            try:
                records=random.sample(self.records,self.sample_size)
                if any(n>self.capacities[(tp,pp)] for n,_ in records):
                    raise ValueError('calibration prompt exceeds prefill capacity; author FCFS would not advance')
                max_events=self.max_events
                class BoundedEnvironment(simpy.Environment):
                    steps=0
                    def step(self):
                        self.steps+=1
                        if self.steps>max_events:raise RuntimeError('official simulator event budget exhausted (no progress or overload)')
                        return super().step()
                env=BoundedEnvironment()
                requests=[Request(env=env,req_id=i,prefill_length=n,output_lens=out-1) for i,(n,out) in enumerate(records)]
                # Original arrival generator uses milliseconds and a zero-delay first arrival.
                rng=np.random.RandomState(self.seed)
                delays=[0.]+list(rng.gamma(1/(self.cv*self.cv),self.cv*self.cv/rate,size=len(records)-1)*1000)
                cluster=DisaggCluster(env,PP_prefill=pp,PP_decode=pd,worker_configs=dict(
                    model_type='local_measured_model',TP=tp,TP_Prefill=tp,TP_Decode=td,
                    prefill_max_batch_size=10**7,decode_max_batch_size=10**7,
                    prefill_max_tokens=self.capacities[(tp,pp)],decode_max_tokens=self.capacities[(td,pd)],
                    enable_chunked_prefill=False,engine_type='distserve'))
                from .simulator_stage import bind_worker_stages
                bind_worker_stages(cluster)
                cluster.run();put_requests_with_interarrivals(env,cluster.scheduler,delays,requests);env.run()
                ttft=[];tpot=[]
                for request in requests:
                    initial=[t for t,event,_ in request.log if event=='init']
                    first=[t for t,event,_ in request.log if event=='wait_decode']
                    last=[t for t,event,_ in request.log if event=='exit_system']
                    if not initial or not first or not last:raise RuntimeError('simulation ended with incomplete offered request')
                    ttft.append((min(first)-min(initial))/1000)
                    tpot.append((max(last)-min(first))/max(request.output_lens,1)/1000)
                return dict(ttft_s=ttft,tpot_s=tpot,request_events=[dict(request_id=r.req_id,events=r.log) for r in requests],
                    worker_events=[dict(worker_id=w.wid,events=w.log) for w in cluster.get_all_workers()],
                    raw_event_unit='milliseconds',sampled_shapes=records,seed=self.seed,
                    upstream_revision='82831f1604cc6b10bebd360f6c437a07790dde9f',gpu_qualified=False,
                    output_convention='API total outputs mapped to upstream decode steps = max_tokens - 1',
                    retained_upstream_behavior=['hardcoded 50000 decode context budget','2ms + 0.0001ms/token Ray overhead',
                        'prefill/decode simulator flow is not evidence for physical KV transfer'],
                    profile_adaptation='measured per-stage delay replaces OPT/A100 time estimator; no PP extrapolation')
            finally:
                LATENCY.reset(token);random.setstate(old_rng)
