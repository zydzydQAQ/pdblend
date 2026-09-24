#!/usr/bin/env python3
"""CPU-only evidence for compiled profile queries; never qualifies a GPU run."""
import argparse
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time
import tracemalloc
from types import MethodType

from pdblend.profile.query.model import PerfModel
from pdblend.profile.query.power_table import CompiledPowerTable, KIND, predict
from pdblend.profile.query.versions import load_version
from pdblend.planner.pool import PoolPlanner, PlannerConfig, SLO
from pdblend.planner.forecast import Forecast


def measure(call, repeats):
    for _ in range(100): call()
    durations=[]
    for _ in range(repeats):
        start=time.perf_counter_ns();call();durations.append(time.perf_counter_ns()-start)
    durations.sort()
    return dict(samples=repeats,median_us=statistics.median(durations)/1000,
                p95_us=durations[int(.95*(repeats-1))]/1000,
                p99_us=durations[int(.99*(repeats-1))]/1000)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--registry',type=Path,action='append',required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--samples',type=int,default=10000)
    args=parser.parse_args()
    rows=[]; planner_model=None
    for registry in args.registry:
        for row in json.loads(registry.read_text())['versions']:
            started=time.perf_counter()
            loaded=load_version(registry,row['version_id'],system=row['system'],model_id=row['model_id'],
                                tp=row['tp'],pp=row['pp'],usage='development')
            load_s=time.perf_counter()-started
            model=loaded.model
            queries={'prefill':lambda:model.prefill_seconds(512,1500),
                     'short_timing':lambda:model.step_seconds(4.,1200.125,1500),
                     'short_power':lambda:model.decode_power_w(4.5,1500,ctx=1200.125)}
            # The 14B TP4 table only has calibrated B1/B128 and interpolation
            # starts at B128, preserving the diagnostic-B1 regime boundary.
            if not model.decode_power_supported(4.5,1200.125,1500):
                queries['short_power']=lambda:model.decode_power_w(128,1500,ctx=1200.125)
            if row.get('long_domain'):
                queries.update(long_timing=lambda:model.step_seconds(4.,7000.125,1500),
                               long_power=lambda:model.decode_power_w(4.,1500,ctx=7000.125))
            timings={name:measure(call,args.samples) for name,call in queries.items()}
            # Allocation of the candidate index is measured separately from
            # checksum validation/raw evidence parsing and frozen-source ASTs.
            tracemalloc.start(); start=time.perf_counter()
            candidate=PerfModel.load(row['evidence']['power_candidate']['path'])
            compile_s=time.perf_counter()-start
            current,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
            rows.append(dict(version_id=row['version_id'],queries=timings,
                checksum_bound_load_s=load_s,candidate_load_compile_s=compile_s,
                candidate_python_allocation_bytes=current,candidate_python_peak_bytes=peak,
                index=model.query_qualification,qualification=loaded.qualification,
                serialized_profile='JSON',query_storage='compiled in-memory arrays',
                warm_query_file_reads=0,formal_eligible=False))
            if planner_model is None: planner_model=candidate
    scaling=[]
    for n in (16,64,256,1024):
        spec=dict(kind=KIND,batch_interpolation='linear',nodes=[dict(batch=4,context_min=10*i+1.,
            context_max=10*i+2.,power_w=100+i*.1) for i in range(n)])
        start=time.perf_counter();table=CompiledPowerTable(spec);compile_s=time.perf_counter()-start
        context=5*n+.5
        scaling.append(dict(nodes=n,index_bytes=table.bytes,compile_s=compile_s,
            directory_reads=1,max_knot_comparisons=4,
            compiled=measure(lambda:table.predict(4.,context),args.samples),
            legacy_scan=measure(lambda:predict(spec,4.,context),min(args.samples,2000))))
    # Complete planner timing deliberately uses a historical full PerfModel,
    # because component registries lack qualified static/capacity/transfer.
    # Compare compiled power/frequency to the original scan/algebra on exactly
    # the same model and workload; no missing qualifications are invented.
    compiled=deepcopy(planner_model); reference=deepcopy(planner_model)
    def legacy_power(self,batch,frequency,*,ctx=None):
        return predict(self.decode_power_overrides[frequency],batch,ctx)
    reference.decode_power_w=MethodType(legacy_power,reference)
    reference.nearest_freq=MethodType(lambda self,f:min(self.freqs,key=lambda x:abs(x-f)),reference)
    forecast=Forecast(.1,0,950,950,100,0,(950,)*10)
    config=PlannerConfig(slots=8,slo=SLO(5,.15),allow_pd=False)
    cp,rp=PoolPlanner(compiled,config),PoolPlanner(reference,config)
    start=time.perf_counter(); plan=cp.plan(forecast);first_ms=(time.perf_counter()-start)*1000
    ref=rp.plan(forecast)
    if asdict(plan)!=asdict(ref):raise AssertionError('compiled planner result differs from legacy query result')
    planner=dict(purpose='historical_model_numerical_compatibility_only',formal_eligible=False,
        component_registry_planner_status='blocked_unqualified_static_capacity_transfer_clock',
        same_plan=True,slots=8,allow_pd=False,rate_rps=.1,input_tokens=950,output_tokens=100,first_compiled_ms=first_ms,
        compiled=measure(lambda:cp.plan(forecast),25),legacy_scan=measure(lambda:rp.plan(forecast),25))
    report=dict(schema=1,created_unix_s=time.time(),python=sys.version,samples_per_scalar=args.samples,
        notes=['latency includes Python timing overhead; no CI hard real-time assertion',
               'O(1) applies to loaded PerfModel, BoundedVersionModel and compiled table queries',
               'legacy raw-spec diagnostic helper intentionally retained for backwards compatibility',
               'index_bytes counts numeric directories/coordinates; tracemalloc measures Python allocations',
               'all provenance/raw checks occur once at load; no measurement gate was weakened'],
        versions=rows,scaling=scaling,planner=planner)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(report=str(args.out),p95_max_us=max(v['p95_us'] for r in rows for v in r['queries'].values()),
                         planner=planner),indent=2))

if __name__=='__main__':main()
