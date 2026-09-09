"""Independent baseline control mechanisms on the shared EngineBackend.

Mechanism code is not validation evidence. The campaign still requires executed
traces, correctness and independent calibration for every claimed mechanism.
"""
from dataclasses import dataclass
from collections import defaultdict, deque
from functools import lru_cache
import itertools
import math

from .transfers import TransferStore


def online_profile_points(profiles,role,input_tokens,context_tokens,frequency_mhz=2520):
    """Only buckets the online lookup can select for this request shape.

    Enumerating raw points can bypass a more specific bucket or the stable
    first-table priority for duplicate measurements. TP and batch candidates
    come from measurements; lookup resolves their actual input/context bucket.
    """
    queries=sorted({(p.tp,p.batch) for p in profiles.points
                    if p.role==role and p.frequency_mhz==frequency_mhz})
    resolved=(profiles.lookup(role,tp,frequency_mhz,input_tokens,context_tokens,batch)
              for tp,batch in queries)
    return tuple(dict.fromkeys(p for p in resolved if p is not None))


@dataclass(frozen=True)
class Placement:
    prefill_tp: int
    decode_tp: int
    prefill_count: int
    decode_count: int
    prefill_batch: int
    decode_batch: int
    capacity_rps: float
    gpus: tuple[tuple[int,...],...]
    transfer_upper_s: float


class DistServeSearch:
    """Independent stage batching, TP and instance search under KV/SLO limits."""
    def __init__(self,profiles,transfers,gpu_count=8,topology=None):
        self.profiles,self.transfers,self.gpu_count=profiles,tuple(transfers),gpu_count
        self.topology=topology
        self.transfer_point=lru_cache(maxsize=4096)(TransferStore(self.transfers,topology).lookup)

    def placements(self,p_tp,d_tp,np,nd):
        # Preserve the measured within-TP rank locality, while searching P/D
        # placement across PCIe switches and NUMA nodes independently.
        pblocks=[tuple(range(g,g+p_tp)) for g in range(0,self.gpu_count-p_tp+1,p_tp)]
        dblocks=[tuple(range(g,g+d_tp)) for g in range(0,self.gpu_count-d_tp+1,d_tp)]
        for ps in itertools.combinations(pblocks,np):
            used={g for group in ps for g in group}
            for ds in itertools.combinations([d for d in dblocks if not used.intersection(d)],nd):
                yield ps+ds

    def search(self,input_tokens,predicted_output,ttft_s,tpot_s,rate,kv_capacity):
        ppoints=online_profile_points(self.profiles,'prefill',input_tokens,input_tokens+1)
        dpoints=online_profile_points(self.profiles,'decode',input_tokens,input_tokens+predicted_output)
        candidates=[]
        for p,d in itertools.product(ppoints,dpoints):
            links=[t for t in self.transfers if (t.source_tp,t.target_tp)==(p.tp,d.tp)
                   and t.max_input_tokens>=input_tokens and t.profile_batch>=p.batch
                   and t.source_sha256 and t.validated]
            if not links:
                continue
            if p.phase_time_bound('prefill')+d.iteration_s*d.bound>ttft_s or d.iteration_s*d.bound>tpot_s:
                continue
            if kv_capacity.get(d.tp,0)<d.batch*(input_tokens+predicted_output):
                continue
            for np in range(1,self.gpu_count//p.tp+1):
                for nd in range(1,self.gpu_count//d.tp+1):
                    used=np*p.tp+nd*d.tp
                    if used>self.gpu_count:
                        continue
                    capacity=min(np*p.batch/p.phase_time_bound('prefill'),
                        nd*d.batch/(d.iteration_s*d.bound*max(predicted_output,1)))
                    if capacity<rate:
                        continue
                    for groups in self.placements(p.tp,d.tp,np,nd):
                        measured=[]
                        for source,target in itertools.product(groups[:np],groups[np:]):
                            link=self.transfer_point(p.tp,d.tp,source,target,input_tokens,p.batch,
                                                     decode_frequency_mhz=2520)
                            if link is None: break
                            measured.append(link.seconds_upper)
                        else:
                            bound=max(measured)
                            capacity=min(np*p.batch/(p.phase_time_bound('prefill')+bound),
                                nd*d.batch/(d.iteration_s*d.bound*max(predicted_output,1)))
                            if capacity>=rate and p.phase_time_bound('prefill')+bound+d.iteration_s*d.bound<=ttft_s:
                                candidates.append(Placement(p.tp,d.tp,np,nd,p.batch,d.batch,
                                                            capacity,groups,bound))
        return sorted(candidates,key=lambda c:(sum(map(len,c.gpus)),c.transfer_upper_s,-c.capacity_rps))


@dataclass(frozen=True)
class ReconfigurationCost:
    operation: str
    duration_upper_s: float
    energy_upper_j: float
    source_sha256: str


class DynamoLLMPolicy:
    """Nine logical shape classes; independent 30min/5min/5s control clocks.

    Instance and shard actions require measured costs and a backend transaction;
    returning a due action does not falsely count as an executed reconfiguration.
    """
    PERIODS={'ScaleInst':1800.,'ScaleShard':300.,'ScaleFreq':5.}
    def __init__(self,costs=(),input_cuts=(255,1023),output_cuts=(99,349)):
        # Table IV's strict <256/<1024 and <100/<350 boundaries. A workload
        # adaptation may instead freeze 33rd/66th percentiles from calibration.
        if any(len(c)!=2 or c[0]<0 or c[1]<=c[0] for c in (input_cuts,output_cuts)):
            raise ValueError('two increasing nonnegative class cutoffs required')
        self.costs={c.operation:c for c in costs}
        self.input_cuts,self.output_cuts=input_cuts,output_cuts
        self.last={name:None for name in self.PERIODS}
        self.arrivals=defaultdict(lambda:deque(maxlen=4096))

    @staticmethod
    def length_class(length,cuts):
        return 'S' if length<=cuts[0] else 'M' if length<=cuts[1] else 'L'

    def classify(self,input_tokens,predicted_output):
        return self.length_class(input_tokens,self.input_cuts)+self.length_class(predicted_output,self.output_cuts)

    def observe_arrival(self,at_s,input_tokens,predicted_output):
        self.arrivals[self.classify(input_tokens,predicted_output)].append(at_s)

    def due(self,now):
        actions=[]
        for name,period in self.PERIODS.items():
            if self.last[name] is None:
                self.last[name]=now
            elif now-self.last[name]>=period:
                actions.append(name)
                # Avoid catch-up storms; preserve the original cycle grid.
                self.last[name]+=math.floor((now-self.last[name])/period)*period
        return tuple(actions)

    def forecast(self,now,window_s=300):
        return {name:sum(now-window_s<=t<=now for t in samples)/window_s
                for name,samples in self.arrivals.items()}

    def amortizes(self,operation,saving_w,horizon_s,lost_capacity_j=0):
        cost=self.costs.get(operation)
        return bool(cost and cost.source_sha256 and
                    saving_w*max(horizon_s-cost.duration_upper_s,0)>
                    cost.energy_upper_j+lost_capacity_j)

    async def stagger(self,backend,actions):
        # Never drain multiple serving groups together. The backend transaction
        # must retain capacity and validate the replacement before committing.
        for action in actions:
            await backend.execute(action)
            if not await backend.confirm(action):
                raise RuntimeError('DynamoLLM reconfiguration was not confirmed')
