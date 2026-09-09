"""Declare the next 1.25x rate for all five systems; release only a PDB exploration."""
import argparse
import copy
from decimal import Decimal
import hashlib
import importlib.util
import json
from pathlib import Path
import time

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
def load(p,name):
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--dataset',choices=['alpaca','sharegpt','longbench'],required=True)
    parser.add_argument('--rate',required=True);parser.add_argument('--predecessor',type=Path,required=True);a=parser.parse_args()
    rate=Decimal(a.rate);prior_rate=rate/Decimal('1.25')
    generator=REPO/'campaign/five-system-fixed-window-v1/generate.py';g=load(generator,'c_p4_explore_generator');parent=g.parent_generator()
    tag=a.dataset+'-r'+g.number(rate);out=HERE/('explore-p4-'+tag+'-input');workflow=HERE/('workflow-explore-p4-'+tag)
    assert not out.exists() and not workflow.exists();(out/'source300').mkdir(parents=True);(out/'traces').mkdir();workflow.mkdir()
    source=g.read(REPO/'campaign/five-system-fixed-window-v1/sources.json')['models']['7b']['source_spec']
    spec=g.read(g.frozen(source));groups,sampling_seed=parent.load_sources(spec)
    _,trace300=parent.build_trace(spec,'7b',a.dataset,rate,701,groups[('7b',a.dataset)],sampling_seed)
    path300=out/'source300'/('7b-'+tag+'-s701-w300.json');path300.write_bytes(parent.encode(trace300))
    trace=g.prefix_trace(trace300,g.ref(path300));wid='7b-'+tag+'-s701-w100';path=out/'traces'/(wid+'.json');path.write_bytes(g.encode(trace))
    original=g.read(HERE/'work-declaration.json');row=copy.deepcopy(next(c['source_row'] for c in original['cells'] if c['dataset']==a.dataset))
    row.update(workload_id=wid,cell_id=wid,rate_rps=float(rate),rate_rps_decimal=g.number(rate),n_requests=trace['n_requests'],
        trace=str(path),trace_path=str(path),trace_sha256=g.sha(path),trace_bytes=path.stat().st_size,
        content_pairing_sha256=trace['content_pairing_sha256'],planned_arrival_span_s=trace['planned_arrival_span_s'],source_300s_trace=g.ref(path300),
        source_indices_sha256=g.digest(trace['source_pool_indices']),unique_selected_pool_records=len(set(trace['source_pool_indices'])),
        within_trace_resampling=trace['within_trace_resampling'],new_rate_source_spec=source,sampling_seed=sampling_seed,
        expected_generated_tokens=sum(q['output_len'] for q in trace['requests']))
    row.pop('source_manifest',None);row.pop('source_manifest_sha256',None)
    cells=[]
    for system in g.SYSTEMS:
        r=copy.deepcopy(row);r.update(system=system,cell_id='parallel-rate-p4-explore-'+wid+'-'+system+'-slo1-repeat1',repeat=1,sequence=len(cells)+1)
        cells.append(r)
    five=dict(schema='parallel-rate-five-system-exploration-p4',created_s=time.time(),model='7b',dataset=a.dataset,rate_rps=float(rate),
        rate_multiplier=1.25,prior_rate_rps=float(prior_rate),predecessor=str(a.predecessor.resolve()),cells=cells,
        five_system_same_trace_declared=True,baseline_measurements_pending=True,strict_pair_eligible=False,
        strict_pair_requires_both_complete_slo90_lower_energy_and_jgood=True,arrival_window_s=100,seed=701,
        request_hard_timeout_s=120,drain_after_arrival_window_s=120,all8gpu_power=True,
        stop_after_first_complete_slo_below_90=True,any_request_failure_stops_node_expansion=True,
        generators=[g.ref(generator),g.ref(g.PARENT_GENERATOR)],source_spec=source)
    (out/'declaration.json').write_bytes(g.encode(five))
    pdb=next(c for c in cells if c['system']=='pdblend')
    work=dict(schema='parallel-rate-C-exploration-p4',created_s=time.time(),deadline_s=1788872770.0400891,
        five_system_declaration=g.ref(out/'declaration.json'),predecessor=str(a.predecessor.resolve()),prior_rate_rps=float(prior_rate),
        baseline_pairing_pending=True,strict_pair_eligible=False,any_request_failure_stops_further_measurement=True,
        cells=[dict(schema=1,model='7b',dataset=a.dataset,arm='fixed2',repeat=1,stage='screen_fixed2',source_row=pdb,
            trace=g.ref(path),original_cell_id=wid+'-pdblend-slo1',cell_id=pdb['cell_id'],new_rate=True,exploration_only=True)])
    (workflow/'work-declaration.json').write_bytes(g.encode(work))
    template=HERE/'workflow-p4-strict'
    for name in ['protocol.py','prepare_release.py','operate.py']:(workflow/name).write_bytes((template/name).read_bytes())
    preparation=(workflow/'prepare_release.py').read_text()
    marker="    release = dict(schema='main-slo-improvement-release-v1'"
    replacement="""    work=p.read(p.ROOT/'work-declaration.json')
    paired=p.checked(work['five_system_declaration'])
    references=[work['five_system_declaration'],*paired['generators'],paired['source_spec'],work['cells'][0]['source_row']['source_300s_trace']]
    for reference in references:
        p.need(p.sha(reference['path'])==reference['sha256'],'exploration provenance changed')
        files[reference['path']]=reference['sha256']
    release = dict(schema='main-slo-improvement-release-v1'"""
    assert preparation.count(marker)==1
    (workflow/'prepare_release.py').write_text(preparation.replace(marker,replacement))
    runner=(template/'runner.py').read_text().replace(g.sha(template/'work-declaration.json'),g.sha(workflow/'work-declaration.json'))
    before="    with node_lease():\n        state['node_lease_held'] = True"
    after="""    with node_lease():
        predecessor=p.read(declaration['predecessor']);prior=predecessor['row']
        prior_binding=p.read(predecessor['binding']);prior_receipt=p.read(predecessor['receipt'])
        p.need(prior_binding['host_release']==release['host_release'], 'exploration requires same active shared PDB source')
        p.need(p.sha(predecessor['binding'])==predecessor['binding_sha256']
               and p.sha(predecessor['receipt'])==predecessor['receipt_sha256'],'predecessor binding/receipt changed')
        for file,digest in predecessor['artifacts'].items():p.need(p.sha(file)==digest,'predecessor raw evidence changed')
        ps=prior_receipt['summary'];row=declaration['cells'][0]['source_row']
        p.need(ps.get('work_complete') is True and ps.get('failed_requests')==0 and ps.get('request_timeouts')==0
               and ps.get('slo_attainment',0)>=.9,'higher rate forbidden after first complete miss or any request failure')
        p.need(prior_receipt.get('measurement_valid') is True and prior_receipt.get('child_stopped') is True
               and prior_receipt.get('clock_restore_complete') is True and not prior_receipt.get('outer_cleanup_errors'),
               'predecessor measured cleanup missing')
        p.need(prior['dataset']==row['dataset'] and prior['model']=='7b' and prior['system']=='pdblend'
               and abs(prior['rate_rps']*1.25-row['rate_rps'])<1e-8,'exact next 1.25x rate required')
        first=p.read(p.ROOT.parent/'screen-p4/status.json')
        p.need(first.get('phase')=='stopped_at_boundary' and len(first.get('completed',[]))==6
               and not first.get('failed') and not first.get('engineering_gate_failed') and first.get('node_lease_held') is False,
               'first six shared-source control/problem cells must be cleanly complete')
        p.need(not Path('/proc/'+str(first['pid'])).exists(),'previous screen still active')
        state['node_lease_held'] = True"""
    assert runner.count(before)==1;runner=runner.replace(before,after)
    (workflow/'runner.py').write_text(runner)
    print(json.dumps(dict(workflow=str(workflow),declaration=g.ref(workflow/'work-declaration.json'),five_system_declaration=g.ref(out/'declaration.json'),
        dataset=a.dataset,rate=float(rate),n_requests=row['n_requests'],expected_output=row['expected_generated_tokens'],exploration_only_until_endpoint_baselines=True)))

if __name__=='__main__':main()
