"""Materialize two declared C LongBench rates using the original frozen generator."""
import copy
from decimal import Decimal
import hashlib
import importlib.util
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]

def load(path,name):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def main():
    oldpath=REPO/'campaign/five-system-fixed-window-v1/generate.py'
    g=load(oldpath,'window100');parent=g.parent_generator()
    source=g.read(REPO/'campaign/five-system-fixed-window-v1/sources.json')['models']['7b']['source_spec']
    spec=g.read(g.frozen(source));groups,sampling_seed=parent.load_sources(spec)
    out=HERE/'new-rates-p1';assert not out.exists();(out/'source300').mkdir(parents=True);(out/'traces').mkdir()
    old=g.read(HERE/'work-declaration.json');template=next(c['source_row'] for c in old['cells'] if c['dataset']=='longbench')
    workloads=[]
    for rate in (Decimal('2'),Decimal('2.5')):
        _,trace300=parent.build_trace(spec,'7b','longbench',rate,701,groups[('7b','longbench')],sampling_seed)
        path300=out/'source300'/('7b-longbench-r'+g.number(rate)+'-s701-w300.json');path300.write_bytes(parent.encode(trace300))
        trace=g.prefix_trace(trace300,g.ref(path300));wid='7b-longbench-r'+g.number(rate)+'-s701-w100'
        path=out/'traces'/(wid+'.json');path.write_bytes(g.encode(trace))
        r=copy.deepcopy(template);r.update(workload_id=wid,cell_id=wid,rate_rps=float(rate),rate_rps_decimal=g.number(rate),
            n_requests=trace['n_requests'],trace=str(path),trace_path=str(path),trace_sha256=g.sha(path),
            trace_bytes=path.stat().st_size,content_pairing_sha256=trace['content_pairing_sha256'],
            planned_arrival_span_s=trace['planned_arrival_span_s'],source_300s_trace=g.ref(path300),
            source_indices_sha256=g.digest(trace['source_pool_indices']),
            unique_selected_pool_records=len(set(trace['source_pool_indices'])),within_trace_resampling=trace['within_trace_resampling'])
        r.pop('source_manifest',None);r.pop('source_manifest_sha256',None)
        r['new_rate_source_spec']=source;r['sampling_seed']=sampling_seed
        r['expected_generated_tokens']=sum(q['output_len'] for q in trace['requests']);workloads.append(r)
    cells=[]
    for system in g.SYSTEMS:
        for repeat in (1,2):
            for r in workloads:
                q=copy.deepcopy(r);q.update(system=system,cell_id='parallel-rate-p1-new-'+r['workload_id']+'-'+system+'-slo1-repeat'+str(repeat),
                                            repeat=repeat,sequence=len(cells)+1,phase='main',part='main')
                cells.append(q)
    assert len(cells)==20
    for r in workloads:
        group=[c for c in cells if c['workload_id']==r['workload_id']]
        assert len(group)==10 and {c['system'] for c in group}==set(g.SYSTEMS)
        assert len({(c['trace_sha256'],c['content_pairing_sha256'],c['n_requests'],c['expected_generated_tokens']) for c in group})==1
    result=dict(schema='parallel-rate-five-system-new-rates-v1',protocol_id=g.PROTOCOL,model='7b',dataset='longbench',
                rates=[2.,2.5],arrival_window_s=100.,seed=701,repeats=2,independent_arrival_seeds=False,
                deadline_s=1788872770.0400891,request_hard_timeout_s=120.,drain_after_arrival_window_s=120.,
                original_snapshots_unchanged=True,all_five_actual_bindings_required=True,baseline_restoration_and_mechanism_gate_required=True,
                runnable=False,source_spec=source,generators=[g.ref(oldpath),g.ref(g.PARENT_GENERATOR)],workloads=workloads,cells=cells)
    (out/'declaration.json').write_bytes(g.encode(result))
    print(json.dumps(dict(path=str(out/'declaration.json'),sha256=g.sha(out/'declaration.json'),cells=20,
                         workloads=[{k:r[k] for k in ['rate_rps','n_requests','expected_generated_tokens','trace_sha256']} for r in workloads])))

if __name__=='__main__':main()
