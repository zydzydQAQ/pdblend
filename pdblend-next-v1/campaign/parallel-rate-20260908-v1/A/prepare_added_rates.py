"""CPU-only same-generator, same-content five-system rate declarations."""
from pathlib import Path
from decimal import Decimal
import importlib.util
import json
import time
HERE=Path(__file__).resolve().parent
REPO=Path('/root/workspace/pdblend-next-v1')

def main():
    generator=REPO/'campaign/five-system-fixed-window-v1/generate.py'
    s=importlib.util.spec_from_file_location('fixed100_source_generator',generator)
    g=importlib.util.module_from_spec(s);s.loader.exec_module(g)
    old=g.parent_generator()
    entry=g.read(REPO/'campaign/five-system-fixed-window-v1/sources.json')['models']['14b']
    source=g.frozen(entry['source_spec']);spec=g.read(source)
    groups,sampling_seed=old.load_sources(spec)
    out=HERE/'added-rates-001';out.mkdir(exist_ok=False)
    (out/'source300').mkdir();(out/'traces').mkdir()
    workloads=[];cells=[]
    for dataset,rate in [('sharegpt','1.75'),('longbench','1.125')]:
        _,parent=old.build_trace(spec,'14b',dataset,Decimal(rate),701,groups[('14b',dataset)],sampling_seed)
        parent_path=out/'source300'/('14b-'+dataset+'-r'+rate+'-s701-w300.json')
        parent_path.write_bytes(old.encode(parent))
        trace=g.prefix_trace(parent,g.ref(parent_path))
        wid='14b-'+dataset+'-r'+rate+'-s701-w100'
        trace_path=out/'traces'/(wid+'.json');trace_path.write_bytes(g.encode(trace))
        workload=dict(workload_id=wid,model='14b',dataset=dataset,protocol_id=g.PROTOCOL,
            measurement_schema=3,split='development',load='declared_absolute_rate',seed=701,
            arrival_seed=701,rate_rps=float(rate),rate_rps_decimal=rate,arrival_window_s=100.,
            trace_duration_s=100.,n_requests=trace['n_requests'],trace=str(trace_path),trace_path=str(trace_path),
            trace_sha256=g.sha(trace_path),trace_bytes=trace_path.stat().st_size,materialized=True,
            content_pairing_sha256=trace['content_pairing_sha256'],slo=trace['slo'],slo_protocol='per-dataset-slo-v1',
            sampling_seed=sampling_seed,source_300s_trace=g.ref(parent_path),source_indices_sha256=g.digest(trace['source_pool_indices']),
            expected_generated_tokens=sum(r['output_len'] for r in trace['requests']),
            source_spec=g.ref(source),generator=g.ref(generator),parent_generator=g.ref(g.PARENT_GENERATOR),
            output_lengths_modified=False,prompts_truncated=False,formal_eligible=False)
        workloads.append(workload)
        for system in g.SYSTEMS:
            for repeat in (1,2):
                cells.append(dict(workload,cell_id='parallel-rate-p1-added-'+wid+'-'+system+'-slo1-repeat'+str(repeat),
                    system=system,repeat=repeat,phase='main',part='main',sequence=len(cells)+1,
                    slo_scale=1.,slo_ttft_s=trace['slo']['ttft_s'],slo_tpot_s=trace['slo']['tpot_s'],slo_attainment_target=.9,
                    reuse_main_cell_id=None,baseline_binding_pending=system!='pdblend',execution_status='not_run'))
    old.load_sources(spec)
    for workload in workloads:
        rows=[r for r in cells if r['workload_id']==workload['workload_id']]
        assert len(rows)==10 and len({r['trace_sha256'] for r in rows})==1
        assert {r['system'] for r in rows}==set(g.SYSTEMS)
    result=dict(schema='parallel-rate-added-five-system-A-v1',created_s=time.time(),workloads=workloads,cells=cells,
        deadline_s=1788872770.0400891,request_hard_timeout_s=120.,drain_after_arrival_window_s=120.,
        arrivals_derived_from_frozen_generator=True,source_workload_pool_unchanged=True,
        new_rates_explicitly_authorized=True,original_450_unchanged=True,execution_started=False)
    (out/'declaration.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(path=str(out),cells=len(cells),workloads=[dict(dataset=w['dataset'],rate=w['rate_rps'],requests=w['n_requests'],outputs=w['expected_generated_tokens']) for w in workloads])))

if __name__=='__main__':main()
