"""Independent DynamoLLM scheduling, weekly prediction, and hierarchy."""
from dataclasses import dataclass,field
import math
from .profiles import CoverageError

SHAPES=tuple(a+b for a in 'SML' for b in 'SML')
PERIODS={'ScaleInst':1800.,'ScaleShard':300.,'ScaleFreq':5.}


def classify(n,o):
    def bucket(v,a,b):return 'S' if v<a else 'M' if v<b else 'L'
    return bucket(n,256,1024)+bucket(o,100,350)


def dominates(a,b):
    return all('SML'.index(x)>='SML'.index(y) for x,y in zip(a,b))


class Epochs:
    def __init__(self,started_s):self.last={key:started_s for key in PERIODS}
    def due(self,now):
        result=[]
        for operation,period in PERIODS.items():
            if now-self.last[operation]>=period:
                self.last[operation]+=math.floor((now-self.last[operation])/period)*period
                result.append(operation)
        return tuple(result)


@dataclass
class Request:
    request_id: str
    input_tokens: int
    predicted_output: int
    arrival_s: float
    ttft_s: float
    tpot_s: float
    emitted: int=0
    first_token_s: float | None=None
    last_token_s: float | None=None
    started: bool=False
    def remaining(self):return max(1,self.predicted_output-self.emitted)
    def deadline(self):
        return (self.arrival_s+self.ttft_s if self.first_token_s is None else
                (self.last_token_s or self.first_token_s)+self.tpot_s)


@dataclass
class Replica:
    instance_id: str
    gpus: tuple[int,...]
    tp: int
    shape: str
    frequency_mhz: int
    accepting: bool=True
    free_kv_tokens: int=0
    max_num_seqs: int=16
    requests: list[Request]=field(default_factory=list)
    generation: int=0


@dataclass(frozen=True)
class Route:
    instance_id: str
    energy_j: float


@dataclass(frozen=True)
class Configuration:
    tp: int
    batch: int
    capacity_rps: float
    power_w: float


def shard_milp(options,gpu_budget,rate_rps):
    """Paper pool optimization with fixed GPU budget and integer replica counts."""
    import pulp
    if not options or gpu_budget<1 or not math.isfinite(rate_rps) or rate_rps<0:
        raise CoverageError('measured shard choices and finite demand required')
    solver=pulp.PULP_CBC_CMD(msg=False,threads=1,timeLimit=30)
    if not solver.available():raise RuntimeError('Dynamo requires an available CBC MILP solver')
    problem=pulp.LpProblem('dynamo_pool_sharding',pulp.LpMinimize)
    counts=[pulp.LpVariable('replica_'+str(i),lowBound=0,upBound=gpu_budget//c.tp,cat='Integer')
            for i,c in enumerate(options)]
    problem+=pulp.lpSum(c.power_w*n for c,n in zip(options,counts))
    problem+=pulp.lpSum(c.tp*n for c,n in zip(options,counts))==gpu_budget
    problem+=pulp.lpSum(c.capacity_rps*n for c,n in zip(options,counts))>=rate_rps
    status=problem.solve(solver)
    if status!=pulp.LpStatusOptimal:raise CoverageError('Dynamo measured MILP has no proven optimal feasible solution')
    result=tuple((c,round(n.value())) for c,n in zip(options,counts) if n.value()>.5)
    if sum(c.tp*n for c,n in result)!=gpu_budget or sum(c.capacity_rps*n for c,n in result)+1e-9<rate_rps:
        raise RuntimeError('Dynamo MILP integer solution violates capacity/budget')
    return result


def allocate_pools(rates,reference_capacities,*,reference_tp,gpu_budget):
    """Cluster floor allocation plus componentwise larger-pool fragmentation.

    HPCA'25 DynamoLLM ScaleInst sizing: every pool receives floor(rate/cap)
    full-power reference replicas (the largest pool is ceil, so only it may be
    overprovisioned) and the fractional remainder is promoted to the next
    componentwise-larger pool that dominates the shape.

    A fixed host may be smaller than the paper's reference demand. Instead of
    raising, each pool is sized down to the remaining budget and one replica is
    reserved for the largest (LL) pool, which is the universal sink for promoted
    load and must exist so every request class stays routable. Unserved load
    keeps promoting; what remains after LL is the initialization shortfall the
    caller records and the replay measures. SLO bounds are never weakened: every
    allocated pool still uses its measured full-power reference capacity.
    """
    pending=dict(rates);result={};used=0
    reserve=min(reference_tp,gpu_budget)
    for shape in SHAPES:
        rate=pending.get(shape,0.)
        if rate<=0:continue
        capacity=reference_capacities.get(shape)
        if not capacity or not math.isfinite(capacity):raise CoverageError('missing full-performance reference '+shape)
        count=math.ceil(rate/capacity) if shape=='LL' else math.floor(rate/capacity)
        room=gpu_budget-used if shape=='LL' else gpu_budget-used-reserve
        count=min(count,max(0,room//reference_tp))
        if count:
            used+=reference_tp*count
            result[shape]=dict(gpus=reference_tp*count,rate_rps=min(rate,count*capacity))
        leftover=rate-count*capacity
        if leftover>1e-12 and shape!='LL':
            larger=next(s for s in SHAPES if s!=shape and dominates(s,shape))
            pending[larger]=pending.get(larger,0)+leftover
    return result


class WeeklyLoadTemplate:
    """Prior-week weekday/weekend daily median, adapted from SmartOClock.

    UTC defines day boundaries. The one-week coverage gate is this project's
    qualification rule; DynamoLLM only specifies historical weekly patterns.
    The referenced template predicts power; here daily observations are RPS.
    """
    WEEK=604800
    def __init__(self,table,slot_s,trained_until):
        self.table=table;self.slot_s=slot_s;self.trained_until=trained_until
    @classmethod
    def fit(cls,records,*,start_s,end_s,slot_s=300):
        from statistics import median
        if (type(slot_s) is not int or slot_s<=0 or 86400%slot_s
                or not math.isfinite(start_s) or not math.isfinite(end_s) or end_s-start_s<cls.WEEK):
            raise ValueError('at least one complete prior week and integral weekly slots required')
        exposure={};counts={}
        window_start=end_s-cls.WEEK;cursor=window_start
        while cursor<end_s:
            edge=min(end_s,(math.floor(cursor/slot_s)+1)*slot_s)
            slot=(int(cursor//86400),int(cursor%86400//slot_s))
            exposure[slot]=exposure.get(slot,0.)+edge-cursor
            cursor=edge
        for row in records:
            t=row['at_s']
            if not start_s<=t<end_s or row.get('split','calibration')!='calibration':
                raise ValueError('weekly predictor forbids future/evaluation training rows')
            if t<window_start:continue
            count=row.get('count',1)
            if (type(count) is not int or count<1 or 'count' in row and
                    row.get('record_type')!='aggregated_verified_arrivals'):
                raise ValueError('positive actual aggregate count and explicit record identity required')
            key=(int(t//86400),int(t%86400//slot_s),classify(row['input_tokens'],row['output_tokens']))
            counts[key]=counts.get(key,0)+count
        daily={}
        for (day,slot),duration in exposure.items():
            if duration!=slot_s:continue
            weekend=(day+3)%7>=5 # Unix epoch was Thursday; Monday is zero.
            for shape in SHAPES:
                daily.setdefault((weekend,slot,shape),[]).append(counts.get((day,slot,shape),0)/duration)
        table={key:median(values) for key,values in daily.items()}
        return cls(table,slot_s,end_s)
    def forecast(self,at_s,horizon_s):
        if at_s<self.trained_until or horizon_s<=0:raise ValueError('forecast must follow training and have positive horizon')
        result={shape:0. for shape in SHAPES};cursor=at_s
        while cursor<at_s+horizon_s:
            slot=int(cursor%86400//self.slot_s);weekend=(int(cursor//86400)+3)%7>=5
            for shape in SHAPES:
                if (weekend,slot,shape) not in self.table:raise CoverageError('missing prior-week template slot')
                result[shape]=max(result[shape],self.table[weekend,slot,shape])
            cursor=(math.floor(cursor/self.slot_s)+1)*self.slot_s
        return result


class DynamoPolicy:
    def __init__(self,profiles):self.profiles=profiles
    def estimate(self,replica,requests,frequency=None):
        return self.profiles.query(replica.tp,frequency or replica.frequency_mhz,
            max(r.input_tokens for r in requests),
            max(r.input_tokens+max(r.predicted_output,r.emitted+1) for r in requests),len(requests))
    def feasible(self,replica,requests,now,frequency=None):
        try:p=self.estimate(replica,requests,frequency)
        except CoverageError:return None
        prefills=sum(not r.started for r in requests)*p.prefill_s
        if any(p.decode_s>r.tpot_s or prefills+p.decode_s>r.deadline()-now for r in requests):return None
        return p
    def route(self,request,replicas,now):
        shape=classify(request.input_tokens,request.predicted_output);native=[];spill=[]
        for i in replicas:
            if (not i.accepting or not dominates(i.shape,shape) or len(i.requests)>=i.max_num_seqs
                    or i.free_kv_tokens<request.input_tokens+request.predicted_output):continue
            work=[*i.requests,request];p=self.feasible(i,work,now)
            if p is None:continue
            energy=sum(not r.started for r in work)*p.prefill_s*p.prefill_power_w
            energy+=max(r.remaining() for r in work)*p.decode_s*p.decode_power_w
            (native if i.shape==shape else spill).append(Route(i.instance_id,energy))
        return min(native or spill,key=lambda r:(r.energy_j,r.instance_id),default=None)
    def frequency(self,replica,now):
        if not replica.requests:return replica.frequency_mhz
        options=[]
        for f in self.profiles.frequencies(replica.tp):
            p=self.feasible(replica,replica.requests,now,f)
            if p:
                energy=sum(not r.started for r in replica.requests)*p.prefill_s*p.prefill_power_w
                energy+=max(r.remaining() for r in replica.requests)*p.decode_s*p.decode_power_w
                options.append((energy,f))
        return min(options)[1] if options else max(self.profiles.frequencies(replica.tp))
    def emergency(self,replica,now,*,previous_stage=0):
        ordered=tuple(r.request_id for r in sorted(replica.requests,key=lambda r:(r.deadline(),r.arrival_s)) if not r.started)
        if not replica.requests or self.feasible(replica,replica.requests,now):
            return dict(stage=0,reorder=(),frequency=None,reroute=(),reject=())
        highest=max(self.profiles.frequencies(replica.tp))
        # Escalation is acknowledged between ticks; never claim a frequency RPC
        # or queue operation succeeded merely because this decision was emitted.
        stage=min(4,previous_stage+1)
        return dict(stage=stage,reorder=ordered,frequency=highest if stage>=2 else None,
                    reroute=ordered if stage>=3 else (),
                    reject=tuple(r.request_id for r in replica.requests if not r.started and r.deadline()<=now) if stage>=4 else ())
