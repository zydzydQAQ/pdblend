"""DynamoLLM's independent hierarchy on resident single-node instances.

The 1800/300/5 second clocks are real control periods. A physical topology
change must be acknowledged by the optional topology backend; without it the
strategy is explicitly a resident-pool variant, never a full ScaleShard claim.
"""
from collections import defaultdict, deque
from dataclasses import replace

from .baselines import DynamoLLMPolicy
from .planner import JointPlanner
from .frequency import FrequencyCost
from .types import ControlPlan, FrequencyAction

SHAPES=tuple(a+b for a in 'SML' for b in 'SML')


def dominates(larger, smaller):
    return all('SML'.index(a)>='SML'.index(b) for a,b in zip(larger,smaller))


class DynamoScheduler:
    def __init__(self,profiles,assignments,*,costs=(),input_cuts=(255,1023),
                 output_cuts=(99,349),clock_settle_s=.3,frequency_costs=(),max_frequency=2520):
        if not assignments or any(v not in SHAPES for v in assignments.values()):
            raise ValueError('explicit instance-to-shape pool assignments required')
        self.hierarchy=DynamoLLMPolicy(costs,input_cuts,output_cuts)
        self.assignments=dict(assignments)
        self.profiles=profiles
        self.estimator=JointPlanner(profiles,max_frequency=max_frequency,allow_pd=False,dvfs=True,clock_settle_s=clock_settle_s,
                                    frequency_costs=frequency_costs)
        self.history=defaultdict(lambda:deque(maxlen=8192))
        self.frequency_costs=tuple(FrequencyCost(**c) for c in frequency_costs)
        self.version=0

    def arrival(self,request):
        shape=self.hierarchy.classify(request.input_tokens,request.predicted_output)
        self.hierarchy.observe_arrival(request.arrival_s,request.input_tokens,request.predicted_output)
        self.history[shape].append(request)
        return shape

    def forecast(self,now,window=300,bucket=10):
        # The upper observed bucket rate is a causal load forecast, with no
        # future trace, realized output lengths, or compressed control clock.
        result={}
        for shape,history in self.history.items():
            samples=[r for r in history if now-window<=r.arrival_s<=now]
            if not samples:
                continue
            bins=defaultdict(int)
            for r in samples:
                bins[int((now-r.arrival_s)//bucket)]+=1
            result[shape]=dict(rate=max(bins.values())/bucket,
                input_tokens=max(r.input_tokens for r in samples),
                output_tokens=max(r.predicted_output for r in samples),
                ttft_s=min(r.ttft_s for r in samples),tpot_s=min(r.tpot_s for r in samples))
        return result

    def plan(self,snapshot,pending,*,now):
        request=pending[0]
        shape=self.hierarchy.classify(request.input_tokens,request.predicted_output)
        # Preserve the type pool; spill only to pools for larger shapes.
        eligible=[i for i in snapshot.instances if i.role=='mixed' and
                  dominates(self.assignments.get(i.instance_id,'SS'),shape)]
        full=self.estimator.candidates(replace(snapshot,instances=tuple(eligible)),request,now)
        # The instance frequency manager, not each arriving request, selects
        # clocks. Idle parking is a shared backend feature and wakes to the
        # last manager command.
        frequencies={i.instance_id:i.frequency_mhz for i in eligible}
        candidates=[p for p in full if all(a.frequency_mhz==frequencies[a.instance_id]
                                          for a in p.frequencies)]
        if not candidates:
            # Emergency capacity recovery is separate from the five-second
            # optimizing clock. It never downclocks to conceal saturation.
            return ControlPlan(snapshot.version,now,now+1,feasible=False,
                frequencies=tuple(FrequencyAction(i.instance_id,self.estimator.max_frequency) for i in eligible),
                reason='DynamoLLM emergency: larger pools full, unsafe or unprofiled')
        native=[p for p in candidates if self.assignments[p.routes[0].decode_id]==shape]
        return replace((native or candidates)[0],reason='DynamoLLM shape pool, feasible minimum-energy instance')

    def frequency_plan(self,snapshot,now):
        actions=[]
        for instance in snapshot.instances:
            if instance.instance_id not in self.assignments or not instance.requests or not instance.accepting:
                continue
            requests=instance.requests
            batch=max(1,instance.running+instance.waiting,len(requests))
            context=max(r.input_tokens+max(r.predicted_output,r.emitted) for r in requests)
            input_tokens=max(r.input_tokens for r in requests)
            candidates=[]
            for frequency in self.profiles.frequencies('mixed',instance.tp):
                p=self.profiles.lookup('mixed',instance.tp,frequency,input_tokens,context,batch)
                if p is None:
                    continue
                pending=[self.profiles.lookup('mixed',instance.tp,frequency,r.input_tokens,r.input_tokens+1,1)
                         for r in requests if not r.emitted]
                if not all(pending): continue
                prefill=sum(q.phase_time_bound('prefill') for q in pending)
                switches=[c for c in self.frequency_costs if c.tp==instance.tp
                          and (c.source_mhz,c.target_mhz)==(instance.frequency_mhz,frequency)]
                settling=max((c.duration_upper_s for c in switches),default=(
                    self.estimator.clock_settle_s if frequency>instance.frequency_mhz else 0))
                latency=p.iteration_s*p.bound
                if (now-instance.timestamp_s>1 or any(latency>r.tpot_s or
                        r.next_token_remaining(now)<latency+prefill+settling for r in requests)):
                    continue
                remaining=max(max(r.predicted_output-r.emitted,1) for r in requests)
                # Whole-instance work/energy, charged once for the active batch.
                energy=p.phase_power('prefill')*prefill+p.phase_power('decode')*remaining*p.iteration_s
                energy+=max((c.energy_upper_j for c in switches),default=0.)
                candidates.append((energy,frequency))
            chosen=min(candidates)[1] if candidates and instance.dvfs_allowed else self.estimator.max_frequency
            actions.append(FrequencyAction(instance.instance_id,chosen))
        return ControlPlan(snapshot.version,now,now+1,frequencies=tuple(actions),
                           reason='DynamoLLM ScaleFreq five-second epoch')

    def resident_reassignment(self,snapshot,now):
        """Single-node fragmentation handling, with explicit idle-only commits.

        Whole instances serve native demand. Fractional demand is carried to a
        componentwise larger shape; LL absorbs the final rounding. The physical
        instance set is unchanged and this does not claim node scale-out.
        """
        forecasts=self.forecast(now)
        idle=[i for i in snapshot.instances if not i.requests and not i.running and not i.waiting
              and i.accepting and now-i.timestamp_s<=1]
        assignments=dict(self.assignments)
        carry={}
        for shape in SHAPES:
            demand=forecasts.get(shape)
            leftover=carry.pop(shape,None)
            if leftover:
                demand=dict(leftover) if not demand else dict(rate=demand['rate']+leftover['rate'],
                    input_tokens=max(demand['input_tokens'],leftover['input_tokens']),
                    output_tokens=max(demand['output_tokens'],leftover['output_tokens']),
                    ttft_s=min(demand['ttft_s'],leftover['ttft_s']),
                    tpot_s=min(demand['tpot_s'],leftover['tpot_s']))
            if not demand:
                continue
            usable=[]
            for instance in idle:
                points=[p for p in self.profiles.points if p.role=='mixed' and p.tp==instance.tp
                        and p.frequency_mhz==self.estimator.max_frequency and p.input_tokens>=demand['input_tokens']
                        and p.context_tokens>=demand['input_tokens']+demand['output_tokens']
                        and p.phase_time_bound('prefill')<=demand['ttft_s'] and p.iteration_s*p.bound<=demand['tpot_s']]
                if points:
                    capacity=max(p.batch/p.batch_service_s(demand['output_tokens']) for p in points)
                    usable.append((instance,capacity))
            rate=demand['rate']
            # Avoid unmeasured capacity; keep old assignments if no measured fit.
            for instance,capacity in usable:
                if rate<capacity and shape!='LL':
                    continue
                assignments[instance.instance_id]=shape
                idle.remove(instance)
                rate=max(0,rate-capacity)
                if not rate:
                    break
            larger=next((s for s in SHAPES if s!=shape and dominates(s,shape)),None)
            if rate and larger:
                previous=carry.get(larger)
                carry[larger]=dict(demand,rate=rate)
                if previous:
                    carry[larger]=dict(rate=rate+previous['rate'],
                        input_tokens=max(demand['input_tokens'],previous['input_tokens']),
                        output_tokens=max(demand['output_tokens'],previous['output_tokens']),
                        ttft_s=min(demand['ttft_s'],previous['ttft_s']),
                        tpot_s=min(demand['tpot_s'],previous['tpot_s']))
        # A catch-all larger pool prevents shape prediction errors from leaving
        # requests without any route. It stays within the prepared GPU budget.
        for instance in idle:
            assignments[instance.instance_id]='LL'
        if 'LL' not in assignments.values():
            return dict(self.assignments)
        return assignments

    def commit_assignments(self,assignments):
        if set(assignments)!=set(self.assignments) or any(v not in SHAPES for v in assignments.values()):
            raise ValueError('resident reassignment cannot create physical instances')
        if assignments!=self.assignments:
            self.assignments=dict(assignments)
            self.version+=1
