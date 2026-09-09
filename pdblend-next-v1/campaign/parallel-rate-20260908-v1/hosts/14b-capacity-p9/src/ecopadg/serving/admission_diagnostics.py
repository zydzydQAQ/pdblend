"""Describe actual admission decisions without inventing rejected candidates.

Only the actual planning snapshot for the selected request is observed. Beam
search's hypothetical successor snapshots are excluded. Reason counters can
overlap; they are not a partition of requests or failed planning attempts.
"""
from collections import Counter
from contextlib import contextmanager
import json
import os
from pathlib import Path
import threading
import time

class AdmissionDiagnostics:
    def __init__(self):
        self.local=threading.local()
        self.requests={}
        self.reasons=Counter()
        self.profile_misses=Counter()
        self.first_witness={}

    @contextmanager
    def planning(self,snapshot,request):
        previous=getattr(self.local,'planning',None)
        self.local.planning=(snapshot,request)
        start=time.perf_counter();at=time.time()
        row=self.requests.setdefault(request.request_id,dict(arrival_s=request.arrival_s,
            first_planning_s=at,first_planning_wait_s=max(0.,at-request.arrival_s),
            input_tokens=request.input_tokens,output_limit=request.output_limit,
            attempts=0,planning_total_s=0.,first_planning_duration_s=None,errors=0))
        row['attempts']+=1
        try:
            yield
        except BaseException:
            row['errors']+=1
            raise
        finally:
            duration=time.perf_counter()-start
            row['planning_total_s']+=duration
            if row['first_planning_duration_s'] is None:row['first_planning_duration_s']=duration
            row['repeat_planning_s']=row['planning_total_s']-row['first_planning_duration_s']
            self.local.planning=previous

    @contextmanager
    def candidate(self,snapshot,request):
        previous=getattr(self.local,'actual',None)
        current=getattr(self.local,'planning',None)
        self.local.actual=(snapshot,request) if current and current[0] is snapshot and current[1].request_id==request.request_id else None
        try:yield
        finally:self.local.actual=previous

    def note(self,category,**details):
        actual=getattr(self.local,'actual',None)
        if actual is None:return
        snapshot,request=actual
        self.reasons[(request.request_id,category)]+=1
        key=(request.request_id,category)
        if key not in self.first_witness:
            self.first_witness[key]=dict(request_id=request.request_id,category=category,
                snapshot_version=snapshot.version,snapshot_timestamp_s=snapshot.timestamp_s,at_s=time.time(),**details)

    def lookup_miss(self,args,kwargs):
        actual=getattr(self.local,'actual',None)
        if actual is None:return
        names=('role','tp','frequency_mhz','input_tokens','context_tokens','batch')
        values=dict(zip(names,args));values.update(kwargs)
        self.profile_misses[tuple(values.get(name) for name in names)]+=1
        self.note('profile_missing',lookup=values)

    def mark(self,request_id,category,**details):
        self.reasons[(request_id,category)]+=1
        self.first_witness.setdefault((request_id,category),dict(request_id=request_id,
            category=category,at_s=time.time(),source='actual_controller_check',**details))

    def dump(self,path):
        # Called only after the sole planning worker has joined.
        result=dict(schema='admission-diagnostics-v1',actual_snapshot_only=True,
            hypothetical_beam_candidates_excluded=True,
            reason_counts_overlap=True,reason_counts_are_not_request_failures=True,
            requests=self.requests,
            reasons=[dict(request_id=rid,category=reason,count=count) for (rid,reason),count in self.reasons.items()],
            first_witness=list(self.first_witness.values()),
            profile_lookup_misses=[dict(zip(('role','tp','frequency_mhz','input_tokens','context_tokens','batch','count'),(*key,count)))
                                   for key,count in self.profile_misses.items()])
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        temporary=path.with_name(path.name+'.tmp')
        with temporary.open('w') as handle:
            json.dump(result,handle,allow_nan=False);handle.write('\n');handle.flush();os.fsync(handle.fileno())
        temporary.replace(path)

def attach(planner,enabled):
    if not enabled:return None
    audit=AdmissionDiagnostics();planner._admission_diagnostics=audit
    candidate=planner.candidates;lookup=planner.profiles.lookup
    def observed_candidates(snapshot,request,now):
        with audit.candidate(snapshot,request):return candidate(snapshot,request,now)
    def observed_lookup(*args,**kwargs):
        value=lookup(*args,**kwargs)
        if value is None:audit.lookup_miss(args,kwargs)
        return value
    planner.candidates=observed_candidates
    planner.profiles.lookup=observed_lookup
    return audit

def reject(planner,category,**details):
    audit=getattr(planner,'_admission_diagnostics',None)
    if audit is not None:audit.note(category,**details)

def observed_plan(function,planner,snapshot,pending,**kwargs):
    audit=getattr(planner,'_admission_diagnostics',None)
    if audit is None or not pending:return function(planner,snapshot,pending,**kwargs)
    with audit.planning(snapshot,pending[0]):return function(planner,snapshot,pending,**kwargs)
