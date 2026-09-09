"""Slow per-pool configuration search for DynamoLLM's hierarchy.

All TP points and switch costs must be measured. The single-node adaptation
scales GPU allocations inside one eight-card host, not physical host count.
"""
from dataclasses import dataclass
from collections import Counter
import itertools
import math

from .dynamo import SHAPES,dominates


@dataclass(frozen=True)
class TopologyCost:
    source_tps: tuple[int,...]
    target_tps: tuple[int,...]
    duration_upper_s: float
    energy_upper_j: float
    source_sha256: str
    cached_weights: bool=False

    def __post_init__(self):
        if (not self.source_tps and not self.target_tps or
                any(tp not in (1,2,4,8) for tp in (*self.source_tps,*self.target_tps)) or
                any(not math.isfinite(v) or v<0 for v in (self.duration_upper_s,self.energy_upper_j))):
            raise ValueError('invalid measured physical transition cost')


@dataclass(frozen=True)
class PoolConfiguration:
    shape: str
    tps: tuple[int,...]
    capacity_rps: float
    power_w: float
    batch_sizes: tuple[int,...]


class DynamoTopologyPlanner:
    def __init__(self,profiles,costs,*,error_fraction=.3,gpu_count=8,park_grace_s=.5):
        self.profiles=profiles;self.costs=tuple(costs)
        self.error_fraction=error_fraction;self.gpu_count=gpu_count
        self.park_grace_s=max(0.,park_grace_s)

    @staticmethod
    def reusable(current,target_tps):
        """Keep matching resident replicas, preferring those serving requests."""
        needed=Counter(target_tps);retained=[];remove=[]
        for instance in sorted(current,key=lambda i:(-bool(i.requests or i.running or i.waiting),
                -max(len(i.requests),i.running+i.waiting),i.instance_id)):
            if needed[instance.tp]:
                retained.append(instance);needed[instance.tp]-=1
            else:
                remove.append(instance)
        return tuple(retained),tuple(remove),tuple(sorted(needed.elements()))

    def pool_demand(self,forecasts,assignments):
        pools=set(assignments.values());result={}
        for shape,demand in forecasts.items():
            candidates=[p for p in SHAPES if p in pools and dominates(p,shape)]
            if not candidates:
                return None
            pool=candidates[0]
            old=result.get(pool)
            result[pool]=dict(demand) if old is None else dict(rate=old['rate']+demand['rate'],
                input_tokens=max(old['input_tokens'],demand['input_tokens']),
                output_tokens=max(old['output_tokens'],demand['output_tokens']),
                ttft_s=min(old['ttft_s'],demand['ttft_s']),tpot_s=min(old['tpot_s'],demand['tpot_s']))
        return result

    def configurations(self,shape,demand,budget):
        points={}
        for tp,batch in sorted({(p.tp,p.batch) for p in self.profiles.points if p.role=='mixed'}):
            p=self.profiles.lookup('mixed',tp,2520,demand['input_tokens'],
                                  demand['input_tokens']+demand['output_tokens'],batch)
            if p is None or p.phase_time_bound('prefill')>demand['ttft_s'] or p.iteration_s*p.bound>demand['tpot_s']:
                continue
            duration=p.batch_service_s(demand['output_tokens'])
            if duration>0:
                points.setdefault(p.tp,[]).append((p,p.batch/duration))
        result=[]
        tps=tuple(sorted(points))
        for counts in itertools.product(*(range(budget//tp+1) for tp in tps)):
            used=sum(tp*n for tp,n in zip(tps,counts))
            if not used or used>budget:
                continue
            # Independent stage full-frequency configurations; frequency is
            # optimized only by the lower five-second instance controller.
            # Replicas of the same TP are interchangeable in this model.
            # Enumerate batch multisets once, not every replica permutation.
            variants=itertools.product(*(itertools.combinations_with_replacement(points[tp],n)
                                          for tp,n in zip(tps,counts)))
            for groups in variants:
                variant=tuple(itertools.chain.from_iterable(groups))
                capacity=sum(c for _,c in variant)
                if capacity<demand['rate']:
                    continue
                watts=0.
                for p,c in variant:
                    parked=self.profiles.parked_residency(p.tp)
                    arrival_rate=demand['rate']*c/capacity
                    single=self.profiles.lookup('mixed',p.tp,2520,demand['input_tokens'],
                        demand['input_tokens']+demand['output_tokens'],1)
                    if single is None: break
                    # The measured batch is a capacity ceiling, not a batch
                    # that must always be full. Requiring >97% utilization
                    # for batch 32 left holes between measured batch sizes.
                    busy=min(1.,arrival_rate*single.batch_service_s(demand['output_tokens']))
                    power_point=single if busy<1 else p
                    prefill_work=power_point.prefill_s*power_point.batch
                    decode_work=max(demand['output_tokens']-1,0)*power_point.iteration_s
                    active_power=(power_point.phase_power('prefill')*prefill_work+
                                  power_point.phase_power('decode')*decode_work)/max(prefill_work+decode_work,1e-9)
                    warm=min(1.,busy+arrival_rate*self.park_grace_s)
                    watts+=(parked+(p.residency_w-parked)*warm+
                            (active_power-p.residency_w)*busy)
                else:
                    watts+=(budget-used)*self.profiles.idle_unallocated_gpu_w
                    result.append(PoolConfiguration(shape,tuple(p.tp for p,_ in variant),capacity,watts,
                                                    tuple(p.batch for p,_ in variant)))
        # Dominated choices of the same physical layout need not enter the
        # expensive lifecycle path. Enumeration is only on the slow clock.
        best={}
        for c in result:
            if (c.tps not in best or c.power_w<best[c.tps].power_w or
                    (c.power_w==best[c.tps].power_w and c.capacity_rps>best[c.tps].capacity_rps)):
                best[c.tps]=c
        return sorted(best.values(),key=lambda c:c.power_w)

    def choose(self,snapshot,assignments,forecasts,operation,*,cached_weights=False,has_unrouted_requests=False):
        demands=self.pool_demand(forecasts,assignments)
        if demands is None:
            return None
        horizon={'ScaleInst':1800.,'ScaleShard':300.}[operation]
        proposals=[]
        if operation=='ScaleInst' and not has_unrouted_requests:
            # A causal forecast omits dormant pools. They still consume loaded
            # model residency: retain a warm replica but allow an idle duplicate
            # to leave through the same measured-cost physical transaction.
            for shape in sorted(set(assignments.values())-set(demands)):
                current=[i for i in snapshot.instances if assignments.get(i.instance_id)==shape]
                if len(current)<2 or any(i.requests or i.running or i.waiting or not i.accepting
                        or not 0<=snapshot.timestamp_s-i.timestamp_s<=1
                        or i.reserved_kv_tokens or i.reserved_transfer_bytes
                        or i.kv_allocations or i.transfer_allocations for i in current):
                    continue
                for remove in current:
                    if sum(i.tp==remove.tp for i in current)<2:
                        continue
                    costs=[c for c in self.costs if tuple(c.source_tps)==(remove.tp,)
                           and not c.target_tps and c.source_sha256
                           and (cached_weights or not c.cached_weights)]
                    if not costs:
                        continue
                    cost=max(costs,key=lambda c:c.energy_upper_j)
                    saved_w=self.profiles.parked_residency(remove.tp)-remove.tp*self.profiles.idle_unallocated_gpu_w
                    lower=saved_w*(1-self.error_fraction)*max(0,horizon-cost.duration_upper_s)
                    if lower<=cost.energy_upper_j:
                        continue
                    keep=tuple(i for i in current if i.instance_id!=remove.instance_id)
                    proposals.append(dict(shape=shape,remove_ids=(remove.instance_id,),
                        retained_ids=tuple(i.instance_id for i in keep),
                        target_tps=tuple(sorted(i.tp for i in keep)),add_tps=(),
                        savings_lower_j=lower,cost_upper_j=cost.energy_upper_j,
                        forecast=dict(rate=0.,reason='no demand in the causal forecast window'),
                        source_cost=cost,operation=operation,capacity_recovery=None))
        for shape,demand in demands.items():
            current=[i for i in snapshot.instances if assignments.get(i.instance_id)==shape]
            if not current:
                continue
            source=tuple(sorted(i.tp for i in current))
            current_gpus=sum(source)
            others=sum(i.tp for i in snapshot.instances if assignments.get(i.instance_id)!=shape)
            budget=self.gpu_count-others if operation=='ScaleInst' else current_gpus
            choices=self.configurations(shape,demand,budget)
            if operation=='ScaleShard':
                # The pool manager repartitions its existing GPU allocation.
                # Changing the allocation belongs to the 30-minute controller.
                choices=[c for c in choices if sum(c.tps)==current_gpus]
            old=next((c for c in choices if c.tps==source),None)
            current_capacity=(old.capacity_rps if old else max((c.capacity_rps for c in
                self.configurations(shape,dict(demand,rate=0),budget) if c.tps==source),default=0))
            for choice in choices:
                if choice.tps==source or (old is not None and choice.power_w>=old.power_w):
                    continue
                retained,removed,added=self.reusable(current,choice.tps)
                variants=[(retained,removed,added)]
                # A measured whole-pool rebuild remains an explicit fallback;
                # it cannot be relabeled as a free incremental add/remove.
                if retained: variants.append(((),tuple(current),choice.tps))
                for keep,remove,add in variants:
                    changed_source=tuple(sorted(i.tp for i in remove))
                    if not changed_source and not cached_weights: continue
                    costs=[c for c in self.costs if tuple(sorted(c.source_tps))==changed_source
                           and tuple(sorted(c.target_tps))==add and c.source_sha256
                           and (cached_weights or not c.cached_weights)]
                    if not costs: continue
                    cost=max(costs,key=lambda c:c.energy_upper_j)
                    lower=((old.power_w-choice.power_w)*(1-self.error_fraction)*max(0,horizon-cost.duration_upper_s)
                           if old else 0.)
                    recovery=(dict(current_rps=current_capacity,required_rps=demand['rate'],
                        target_rps=choice.capacity_rps,source_sha256=cost.source_sha256) if old is None else None)
                    if lower>cost.energy_upper_j or recovery:
                        proposals.append(dict(shape=shape,remove_ids=tuple(i.instance_id for i in remove),
                            retained_ids=tuple(i.instance_id for i in keep),target_tps=choice.tps,add_tps=add,
                            savings_lower_j=lower,cost_upper_j=cost.energy_upper_j,
                            forecast=demand,source_cost=cost,operation=operation,capacity_recovery=recovery))
        return max(proposals,key=lambda p:(bool(p['capacity_recovery']),p['savings_lower_j']-p['cost_upper_j'],
                   len(p['retained_ids']))) if proposals else None
