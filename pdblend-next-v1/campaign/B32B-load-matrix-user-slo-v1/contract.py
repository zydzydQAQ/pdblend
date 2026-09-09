"""Pure validation of a frozen PDB-only development workload and user SLOs."""
import math

DATASET_SLOS={'alpaca':{'slo_ttft_s':1.,'slo_tpot_s':.1},
              'sharegpt':{'slo_ttft_s':5.,'slo_tpot_s':.15},
              'longbench':{'slo_ttft_s':15.,'slo_tpot_s':.2}}


def require(ok,reason):
    if not ok:raise ValueError(reason)


def validate(spec,source,contract,read_trace,hash_trace):
    require(contract['schema']==1 and contract['slo_protocol']=='per-dataset-slo-v1'
        and contract['dataset_slos']==DATASET_SLOS and contract['slo_attainment_target']==.9,'wrong user SLO contract')
    require(spec['slo_protocol']==contract['slo_protocol'] and spec['slo_attainment_target']==.9,'wrong spec SLO protocol')
    selection=contract['source_selection'];seeds=selection['arrival_seeds'];variants=selection['variants']
    require(selection['model']=='32b' and seeds and len(seeds)==len(set(seeds))
        and all(type(s) is int and s>=0 for s in seeds),'wrong model/arrival seeds')
    require(variants and all(v=='candidate' or v.startswith('pdblend-') for v in variants),'baseline variant not allowed')
    require(contract['split']=='development' and contract['formal_eligible'] is False
        and contract['saturation_verified'] is False,'must keep development evidence scope')
    minimum=contract['requests_min'];exact=contract.get('requests_exact');span=contract['duration_min_s']
    require(type(minimum) is int and minimum>0 and (exact is None or type(exact) is int and exact>=minimum)
        and isinstance(span,(int,float)) and math.isfinite(span) and span>=0,'invalid workload minimum')
    selected={c['cell_id']:c for c in source['cells'] if c['model']==selection['model']
        and c['arrival_seed'] in seeds and c['variant'] in variants}
    if 'cell_ids' in selection:
        ids=selection['cell_ids']
        require(ids and len(ids)==len(set(ids)) and set(ids)<=set(selected),'invalid exact source cell selection')
        selected={key:selected[key] for key in ids}
    rows=spec['cells'];require(len(rows)==len(selected)==contract['expected_cells'] and rows,'wrong frozen selection size')
    require(len({r['cell_id'] for r in rows})==len(rows),'duplicate run cell')
    require({r['seed'] for r in rows}==set(seeds),'missing selected arrival seed')
    for row in rows:
        original=selected.get(row['cell_id']);require(original and row['source_cell']==original,'source cell metadata changed')
        require(row['system']==original['system']=='pdblend' and original['strategy'].startswith('pdblend'), 'baseline execution denied')
        require(row['controller_config']==rows[0]['controller_config'],'cross-dataset policy config changed')
        trace=read_trace(row['trace']);n=len(trace['requests'])
        require(hash_trace(row['trace'])==row['trace_sha256']==original['trace_sha256'],'trace bytes changed')
        require(trace['split']==row['split']==original['split']=='development'
            and trace['model']==selection['model'] and trace['arrival_seed']==row['seed']==original['arrival_seed'],'split/model/seed changed')
        require(n==row['n_requests']==original['n_requests'] and n>=minimum and (exact is None or n==exact),'workload size violates source contract')
        duration=trace['duration_s']
        require(isinstance(duration,(int,float)) and math.isfinite(duration) and duration>=span
            and duration==row['trace_duration_s']==original['trace_duration_s'],'workload span violates source contract')
        require(trace['dataset']==row['dataset']==original['dataset'] and trace['load']==row['load']==original['load']
            and trace['rate']==row['rate_rps']==original['rate_rps'] and trace['rate']>0,'trace labels/rate changed')
        arrivals=[r['arrival_s'] for r in trace['requests']]
        require(all(isinstance(a,(int,float)) and math.isfinite(a) and 0<=a<=duration for a in arrivals)
            and arrivals==sorted(arrivals),'invalid absolute arrival offsets')
        require(trace.get('formal_eligible') is False and trace.get('saturation_verified') is False,'unverified formal/capacity claim')
        require(row['slo_protocol']==contract['slo_protocol'] and all(row[k]==v for k,v in DATASET_SLOS[row['dataset']].items()),'dataset SLO mismatch')
    return rows
