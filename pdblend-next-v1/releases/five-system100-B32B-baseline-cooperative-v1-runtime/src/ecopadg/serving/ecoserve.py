"""EcoServe paper-mechanism policy on the shared execution backend.

Algorithm 1 and section 3.4 of https://www.usenix.org/system/files/osdi26-du.pdf.
The paper's mean accumulated TPOT credit is deliberately distinct from
PDBlend's per-request feasibility rule. Client metrics remain shared.
"""
from dataclasses import replace
import statistics

from .types import ControlPlan,FrequencyAction,RouteAction,WindowAction


class EcoServeScheduler:
    def __init__(self,profiles,instances,*,lower=2,upper=3):
        if not 1<=lower<=upper or upper<2*lower-1:
            raise ValueError('invalid macro bounds')
        self.profiles=profiles
        self.lower,self.upper=lower,upper
        self.groups=[]
        self.selected={}
        self.phase_started={}
        self.version=0
        for instance_id in instances:
            self.add_instance(instance_id)

    def add_instance(self,instance_id):
        if any(instance_id in g for g in self.groups):
            raise ValueError('instance already belongs to a macro')
        index=next((j for j,g in enumerate(self.groups) if len(g)<self.upper),0)
        if not self.groups:
            self.groups=[(instance_id,)]
        else:
            old=self.groups[index]
            self.groups[index]=old+(instance_id,)
            self.selected[self.groups[index]]=self.selected.pop(old,old[0])
            if len(self.groups[index])>self.upper:
                self.split(index)
        self.version+=1

    def split(self,index):
        old=self.groups[index]
        if len(old)<2*self.lower:
            return False
        a,b=old[:-self.lower],old[-self.lower:]
        current=self.selected.pop(old,old[0])
        self.groups[index:index+1]=[a,b]
        self.selected[a]=current if current in a else a[0]
        self.selected[b]=current if current in b else b[0]
        self.version+=1
        return True

    def remove_idle_instance(self,snapshot):
        states={i.instance_id:i for i in snapshot.instances}
        # Drain is a prerequisite; handles move without migrating any KV.
        order=sorted(range(len(self.groups)),key=lambda j:(len(self.groups[j])<=self.lower,len(self.groups[j])))
        if sum(map(len,self.groups))<=self.lower:
            return None
        for j in order:
            old=self.groups[j]
            for rid in reversed(old):
                i=states[rid]
                if i.requests or i.running or i.waiting or i.reserved_kv_tokens:
                    continue
                remaining=tuple(x for x in old if x!=rid)
                current=self.selected.pop(old,old[0])
                self.groups[j]=remaining
                if remaining:
                    self.selected[remaining]=current if current in remaining else remaining[0]
                else:
                    self.groups.pop(j)
                for a in range(len(self.groups)):
                    for b in range(a+1,len(self.groups)):
                        if len(self.groups[a])+len(self.groups[b])<=self.upper:
                            merged=self.groups[a]+self.groups[b]
                            chosen=self.selected.pop(self.groups[a],self.groups[a][0])
                            self.selected.pop(self.groups[b],None)
                            self.groups[a]=merged;self.groups.pop(b)
                            self.selected[merged]=chosen
                            self.version+=1
                            return rid
                self.version+=1
                return rid
        return None

    def prefill(self,i,r):
        p=self.profiles.lookup('mixed',i.tp,2520,r.input_tokens,r.input_tokens+1,1)
        return p.phase_time_bound('prefill') if p else None

    def feasible(self,i,r,now,switch):
        new=self.prefill(i,r)
        if new is None:
            return None
        durations=[self.prefill(i,q) for q in i.requests if q.arrival_s>=switch]
        if any(t is None for t in durations):
            return None
        total=new+sum(durations)
        credits=[q.emitted*q.tpot_s-(now-q.first_token_s) for q in i.requests
                 if q.arrival_s<switch and q.first_token_s is not None]
        reserve=((r.input_tokens+(r.output_limit or r.predicted_output)+15)//16)*16
        if (total>r.ttft_remaining(now) or (credits and statistics.mean(credits)<total)
                or i.free_kv_tokens-i.reserved_kv_tokens<reserve):
            return None
        return total,reserve

    def plan(self,snapshot,pending,*,now):
        r=pending[0]
        states={i.instance_id:i for i in snapshot.instances}
        groups=sorted(self.groups,key=lambda g:sum(len(states[x].requests) for x in g))
        for group in groups:
            current=self.selected.get(group,group[0])
            start=group.index(current)
            for offset in range(len(group)):
                rid=group[(start+offset)%len(group)]
                i=states[rid]
                if i.role!='mixed' or not i.accepting or not 0<=now-i.timestamp_s<=1:
                    continue
                switch=self.phase_started.get(rid,now) if rid==current else now
                choice=self.feasible(i,r,now,switch)
                if choice is None:
                    continue
                windows=tuple(WindowAction(x,states[x].generation,x==rid) for x in group
                    if states[x].mode!='temporal' or states[x].admit_prefill!=(x==rid))
                total,reserve=choice
                return ControlPlan(snapshot.version,now,now+1,
                    routes=(RouteAction(r.request_id,rid,rid,reserve,total,0,0),),
                    frequencies=(FrequencyAction(rid,2520),),windows=windows,
                    reason='EcoServe: current-then-cyclic window, accumulated prefill, mean saved TPOT and KV')
        return ControlPlan(snapshot.version,now,now+1,feasible=False,
                           reason='EcoServe: no macro instance satisfies constraints')

    def committed(self,plan,now):
        rid=plan.routes[0].decode_id
        group=next(g for g in self.groups if rid in g)
        if self.selected.get(group)!=rid or rid not in self.phase_started:
            self.phase_started[rid]=now
        self.selected[group]=rid

    def membership_plan(self,snapshot,now):
        """Reconcile physical windows immediately after split/merge changes."""
        chosen={self.selected.get(group,group[0]) for group in self.groups}
        windows=tuple(WindowAction(i.instance_id,i.generation,i.instance_id in chosen)
            for i in snapshot.instances if i.mode!='temporal' or
            i.admit_prefill!=(i.instance_id in chosen))
        return ControlPlan(snapshot.version,now,now+1,windows=windows,
            reason='EcoServe macro membership commit: one selected prefill window per group')
