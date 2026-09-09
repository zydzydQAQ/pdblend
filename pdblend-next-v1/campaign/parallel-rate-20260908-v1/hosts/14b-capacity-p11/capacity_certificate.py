"""Recompute empirical capacity bounds from three retained physical repetitions.

No hardware operations. A group is an immutable JSON reference containing kind,
capacity_binding and three members. Layout members are result.json references;
savings members are {source, target} pairs of the same low trace; transition
members are {result, inventory} references. Bounds are regenerated, not trusted.
"""
import argparse
import csv
import math
from pathlib import Path

from capacity_executor import durable, fixed, require, sha


def ref(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path))


def canonical_groups(values):
    return sorted(sorted(int(g) for g in group) for group in values)


def close(a, b):
    return type(a) in (int, float) and type(b) in (int, float) and math.isfinite(a) and math.isfinite(b) \
        and math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-7)


def raw_measurement(reference):
    raw = fixed(reference)
    require(raw.get('measurement_valid') is True and raw.get('gpu_indices') == list(range(8))
            and raw.get('power_evidence', {}).get('power_source_verified') is True
            and not any(raw.get(k) for k in ('sampling_error','window_error','memory_sampling_error')),
            'actual whole-node power measurement invalid')
    require(raw.get('artifacts') and all(sha(p) == h for p,h in raw['artifacts'].items()),
            'raw measurement artifact changed')
    require(raw.get('energy_j',0) > 0 and raw['measurement_end_s'] > raw['measurement_start_s'],
            'positive measured energy/window required')
    import json
    clocks=[Path(p) for p in raw['artifacts'] if Path(p).name=='clocks.json']
    require(len(clocks)==1,'actual clock samples must be retained for calibration')
    clock_rows=json.loads(clocks[0].read_text())
    require(len(clock_rows)==raw.get('clock_samples_observed') and len(clock_rows)>=2
        and all(len(v)==8 and all(type(f) in (int,float) and math.isfinite(f) and f>0 for f in v)
                for t,v in clock_rows) and clock_rows[0][0]<=raw['measurement_start_s']
        and clock_rows[-1][0]>=raw['measurement_end_s'], 'eight-GPU clock observation incomplete')
    powers = [Path(p) for p in raw['artifacts'] if Path(p).name == 'power.csv']
    require(len(powers) == 1, 'one eight-GPU raw power CSV required')
    rows = []
    with powers[0].open() as stream:
        for row in csv.DictReader(stream):
            watts = [float(row[f'gpu{g}_w']) for g in range(8)]
            require(all(math.isfinite(w) and w >= 0 for w in watts), 'raw per-GPU power invalid')
            rows.append((float(row['t_s']),sum(watts)))
    require(rows and all(math.isfinite(t) for t,_ in rows)
            and all(a[0] < b[0] for a,b in zip(rows,rows[1:])), 'raw power timebase invalid')
    raw['_power'] = rows
    require(close(window_energy(raw,raw['measurement_start_s'],raw['measurement_end_s']),raw['energy_j']),
            'measurement energy differs from raw eight-GPU integral')
    return raw


def window_energy(raw, start, end):
    rows = raw['_power']
    require(rows[0][0] <= start < end <= rows[-1][0], 'exact energy window is not bracketed')
    total = 0.
    for (t0,p0),(t1,p1) in zip(rows, rows[1:]):
        a,b=max(start,t0),min(end,t1)
        if b <= a:
            continue
        pa=p0+(p1-p0)*(a-t0)/(t1-t0);pb=p0+(p1-p0)*(b-t0)/(t1-t0)
        total+=(pa+pb)*(b-a)/2
    return total


def source_identity(reference, identity):
    binding = fixed(reference)
    require(binding.get('identity') == identity and binding.get('files')
            and all(sha(p) == h for p,h in binding['files'].items()), 'physical source identity changed')
    return binding


def load_result(reference, identity):
    result = fixed(reference)
    require(result.get('schema') == 'capacity-load-measurement-v1'
            and result.get('complete') is True and result.get('work_complete') is True
            and result.get('native_idle') is True and result.get('resident_groups'),
            'complete unchanged-layout actual load measurement required')
    require(result.get('artifacts') and all(sha(p) == h for p,h in result['artifacts'].items()),
            'request artifact changed')
    source_identity(result['source']['capacity_binding'], identity)
    for key in ('original_binding','config','host_manifest'):
        fixed(result['source'][key])
    config=fixed(result['source']['config'])
    trace = fixed(result['trace'])
    require(trace.get('schema') == 'capacity-development-trace-v1' and trace.get('split') == 'development'
            and trace.get('formal_eligible') is False and trace['duration_s'] >= 60
            and trace['demand_domain_sha256'] == result['demand_domain_sha256'],
            'same independently declared development domain required')
    paths=[p for p in result['artifacts'] if Path(p).name=='requests.json']
    require(len(paths)==1,'actual request rows required')
    import json
    rows=json.loads(Path(paths[0]).read_text())
    expected=trace['n_requests']
    require(expected == len(trace['requests']) == len(rows) == result['n_expected'] == result['n_rows']
        and all(r.get('success') == 1 and r.get('token_ids_verified') == 1
                and r.get('generated_tokens') == t['output_len'] == r.get('output_len')
                and r.get('input_tokens') == t['prompt_len'] == r.get('prompt_len')
                for r,t in zip(rows,trace['requests'])), 'request work/token denominator differs')
    require(len({r['request_id'] for r in rows})==expected,'distinct actual request identities required')
    recomputed=[bool(r['ttft_s'] is not None and r['tpot_s'] is not None
        and r['ttft_s'] < config['slo_ttft_s'] and r['tpot_s'] < config['slo_tpot_s']) for r in rows]
    require(all(r['slo_ok']==int(ok) for r,ok in zip(rows,recomputed)),'request SLO differs from actual timing')
    good=sum(recomputed)
    require(good==result['n_good'] and close(good/expected,result['slo_attainment']) and good/expected >= .9,
            'all sustained calibration repetitions require SLO >= .9')
    raw=raw_measurement(result['raw_measurement'])
    duration=raw['measurement_end_s']-raw['measurement_start_s']
    require(close(result['energy_j'],raw['energy_j']) and close(result['offered_rate_rps'],expected/trace['duration_s']),
            'reported load energy/rate differs from raw work')
    result['_sustainable']=min(expected/trace['duration_s'],expected/duration)
    result['_trace']=trace;result['_raw']=raw
    return result


def verify_idle_native(path, result, inventory, start, end):
    import json
    before=result['resident_before'];after=result['resident_after']
    require(before==after and canonical_groups(v['gpus'] for v in before)==canonical_groups(result['resident_groups']),
            'idle identity or GPU layout changed')
    expected={v['id']:sorted(v['gpus']) for v in before}
    require(len(expected)==len(before) and inventory.get('complete') is True
            and inventory.get('transition_inflight') is False,'final physical inventory incomplete')
    for iid,gpus in expected.items():
        known=inventory['known_instances'][iid]
        require(known.get('verified') is True and sorted(known['gpus'])==gpus,
                'idle owner differs from the physically verified GPU inventory')
        require(iid in inventory['initial_ids'] or known.get('physical_proof'),
                'new idle replica has no physical ordinary/native proof')
    records=[json.loads(line) for line in Path(path).read_text().splitlines() if line]
    require(len(records)>=2 and records[0]['at_s']<=start and records[-1]['at_s']>=end
            and all(0 < b['at_s']-a['at_s'] <= 1. for a,b in zip(records,records[1:])),
            'native samples do not continuously cover the exact declared idle window')
    zeros=('active','running','waiting','kv_allocations','transfer_allocations',
           'transfer_buffered_tensors','transfer_inflight_receives','transfer_inflight_sends')
    for record in records:
        now=record['at_s'];raws=record['raw']
        require(len(raws)==len(expected) and {r.get('id') for r in raws}==set(expected),
                'native idle sample has missing, duplicate or foreign owner identity')
        for raw in raws:
            require(all(k in raw and ((isinstance(raw[k],dict) and not raw[k])
                    if k in ('kv_allocations','transfer_allocations') else type(raw[k]) is int and raw[k]==0)
                    for k in zeros), 'native idle sample contains work or missing counters')
            require(all(type(raw.get(k)) in (int,float) and math.isfinite(raw[k]) and 0<=now-raw[k]<=1.
                        for k in ('timestamp','transfer_observed_s')), 'native idle sample is stale')
            require(raw.get('transport_healthy') is True and not raw.get('error') and not raw.get('runtime_error')
                and type(raw.get('generation')) is int and raw['generation']==raw.get('acknowledged_generation')
                and raw.get('scheduler_budget_pending') is None, 'native idle control/transport not settled')


def idle_result(reference, identity, inventory):
    result=fixed(reference)
    require(result.get('schema')=='capacity-load-measurement-v1' and result.get('phase_kind')=='idle'
        and result.get('complete') is True and result.get('native_idle') is True
        and result.get('work_complete') is True and result.get('resident_groups')
        and result.get('n_expected')==result.get('n_rows')==result.get('n_good')==0
        and result.get('slo_attainment') is None, 'real empty-work idle measurement required')
    source_identity(result['source']['capacity_binding'],identity)
    for key in ('original_binding','config','host_manifest'):
        fixed(result['source'][key])
    require(result.get('artifacts') and all(sha(p)==h for p,h in result['artifacts'].items()),
            'idle raw artifacts changed')
    paths=[p for p in result['artifacts'] if Path(p).name=='requests.json']
    import json
    require(len(paths)==1 and json.loads(Path(paths[0]).read_text())==[], 'idle observation contains requests')
    start,observed_end=result['actual_idle_start_s'],result['actual_idle_end_s']
    duration=result['declared_idle_duration_s'];end=start+duration
    require(duration>=60 and observed_end>=end
            and close(result['measured_idle_duration_s'],observed_end-start),
            'sustained matched idle observation required')
    logs=[p for p in result['artifacts'] if Path(p).name=='idle-native.jsonl']
    require(len(logs)==1,'actual native idle log required')
    verify_idle_native(logs[0],result,inventory,start,end)
    raw=raw_measurement(result['raw_measurement'])
    result['_idle_w']=window_energy(raw,start,end)/(end-start)
    return result


def derive_group(reference, identity):
    group=fixed(reference)
    require(group.get('schema')=='capacity-evidence-group-v1' and group.get('identity')==identity,
            'same physical identity evidence group required')
    source_identity(group['capacity_binding'],identity)
    members=group['members']
    require(len(members)==3,'exactly three independently retained repetitions required')
    kind=group['kind'];raw_refs=[]
    if kind=='layout':
        results=[load_result(r,identity) for r in members]
        require(len({r['trace']['sha256'] for r in results})==3
                and len({r['_trace']['seed'] for r in results})==3,'independent trace seeds required')
        layouts=[canonical_groups(r['resident_groups']) for r in results]
        domains={r['demand_domain_sha256'] for r in results}
        require(all(v==layouts[0] for v in layouts) and len(domains)==1,'layout/domain changes among repetitions')
        bound=dict(resident_groups=layouts[0],demand_domain_sha256=domains.pop(),
                   sustainable_rate_lower_rps=min(r['_sustainable'] for r in results))
        raw_refs=[r['raw_measurement'] for r in results]
    elif kind=='savings':
        pairs=[(load_result(r['source'],identity),load_result(r['target'],identity)) for r in members]
        require(len({s['trace']['sha256'] for s,t in pairs})==3
                and len({s['_trace']['seed'] for s,t in pairs})==3,'independent low-load trace seeds required')
        first_source,first_target=pairs[0]
        domains=set();rates=[];savings=[]
        source_groups=canonical_groups(first_source['resident_groups'])
        target_groups=canonical_groups(first_target['resident_groups'])
        require(len(source_groups)==len(target_groups)+1 and all(g in source_groups for g in target_groups),
                'savings must describe measured one-replica removal')
        for source,target in pairs:
            require(source['trace']==target['trace'] and source['source']['config']==target['source']['config']
                    and canonical_groups(source['resident_groups'])==source_groups
                    and canonical_groups(target['resident_groups'])==target_groups,
                    'low source/target must use exactly matching trace/config/layout')
            domains.update((source['demand_domain_sha256'],target['demand_domain_sha256']))
            span=source['_trace']['duration_s']
            # Compare equal actual arrival windows, not averages over unequal drain tails.
            watts=[]
            for result in (source,target):
                start=result['actual_arrival_epoch_s']
                watts.append(window_energy(result['_raw'],start,start+span)/span)
                raw_refs.append(result['raw_measurement'])
            savings.append(watts[0]-watts[1]);rates.append(source['offered_rate_rps'])
        require(len(domains)==1 and min(savings)>0,'no positive matched measured saving in every repetition')
        bound=dict(source_groups=source_groups,target_groups=target_groups,demand_domain_sha256=domains.pop(),
            rate_lower_rps=min(rates),rate_upper_rps=max(rates),whole_node_saving_lower_w=min(savings))
    elif kind=='idle_savings':
        inventory=fixed(group['inventory'])
        require(inventory.get('identity')==identity,'idle physical inventory belongs to another source')
        pairs=[(idle_result(r['source'],identity,inventory),idle_result(r['target'],identity,inventory)) for r in members]
        require(len({r['source']['sha256'] for r in members})==len({r['target']['sha256'] for r in members})==3,
                'three distinct measured idle pairs required')
        first_source,first_target=pairs[0]
        source_groups=canonical_groups(first_source['resident_groups'])
        target_groups=canonical_groups(first_target['resident_groups'])
        require(len(source_groups)==len(target_groups)+1 and all(g in source_groups for g in target_groups),
                'idle saving must describe one physical replica removal')
        savings=[]
        for source,target in pairs:
            require(source['source']['config']==target['source']['config']
                    and canonical_groups(source['resident_groups'])==source_groups
                    and canonical_groups(target['resident_groups'])==target_groups
                    and source['declared_idle_duration_s']==target['declared_idle_duration_s'],
                    'matched empty work/config/layout/duration required')
            savings.append(source['_idle_w']-target['_idle_w'])
            raw_refs.extend([source['raw_measurement'],target['raw_measurement']])
        require(min(savings)>0,'idle removal has no positive saving in every pair')
        bound=dict(source_groups=source_groups,target_groups=target_groups,demand_domain_sha256=None,
            rate_lower_rps=0.,rate_upper_rps=0.,whole_node_saving_lower_w=min(savings))
    elif kind=='savings_grid':
        parts=[derive_group(member,identity) for member in members]
        require([k for k,b,r in parts].count('idle_savings')==1
                and [k for k,b,r in parts].count('savings')==2,
                'one real idle and two real low-rate three-repeat groups required')
        require(all(fixed(member)['kind'] in ('idle_savings','savings') for member in members),
                'savings grid cannot recursively recycle another grid')
        first=parts[0][1];domain=group['demand_domain_sha256']
        require(all(b['source_groups']==first['source_groups'] and b['target_groups']==first['target_groups']
                    and (k=='idle_savings' or b['demand_domain_sha256']==domain) for k,b,r in parts),
                'grid changes physical layout or measured shape domain')
        loaded=[b for k,b,r in parts if k=='savings']
        require(loaded[0]['rate_upper_rps'] < loaded[1]['rate_lower_rps']
                or loaded[1]['rate_upper_rps'] < loaded[0]['rate_lower_rps'],
                'two distinct measured low-rate endpoints required')
        bound=dict(source_groups=first['source_groups'],target_groups=first['target_groups'],
            demand_domain_sha256=domain,rate_lower_rps=0.,
            rate_upper_rps=max(b['rate_upper_rps'] for k,b,r in parts),
            whole_node_saving_lower_w=min(b['whole_node_saving_lower_w'] for k,b,r in parts))
        raw_refs=[ref for k,b,refs in parts for ref in refs]
        kind='savings'
    elif kind=='transition':
        operation=group['operation'];gpus=sorted(group['gpus'])
        require(len(gpus)==identity['tp'] and len(set(gpus))==len(gpus)
                and all(type(g) is int and 0 <= g < 8 for g in gpus),'actual TP group required')
        require(operation in ('restore_cold','remove'),'only physically measured cold/stop supported')
        records=[];transactions=[];instance_ids=[]
        for member in members:
            record=fixed(member['result'])
            if record.get('schema')=='capacity-load-measurement-v1':
                source_identity(record['source']['capacity_binding'],identity)
                record=record['physical']
            require(record.get('execution_verified') is True
                    and record['operation']==('restore' if operation=='restore_cold' else 'remove'),
                    'actual successful physical operation required')
            inventory=fixed(member['inventory'])
            require(inventory['identity']==identity and inventory['complete'] is True
                    and inventory['transition_inflight'] is False,'physical cleanup inventory incomplete')
            known=inventory['known_instances'][record['instance_id']]
            require(sorted(known['gpus'])==gpus and known.get('physical_proof'),
                    'actual GPU allocation and ordinary/native proof required')
            transaction=record['transaction']
            require(any(e['kind']=='physical_commit' and e.get('transaction')==transaction
                        and e.get('execution_verified') is True and e.get('instance_id')==record['instance_id']
                        and e.get('operation')==record['operation'] for e in inventory['events']),
                    'physical transaction commit is absent')
            require(any(e['kind']=='transition_measurement' and e.get('transaction')==transaction
                        and e.get('receipt')==record['measurement']['receipt'] for e in inventory['events']),
                    'physical measurement is not linked to its actual transaction')
            raw=raw_measurement(record['measurement']['receipt'])
            require(close(raw['energy_j'],record['measurement']['energy_j'])
                    and record['finished_s']>record['started_s'],'physical timing/energy differs')
            records.append((record,raw));transactions.append(transaction);instance_ids.append(record['instance_id'])
            raw_refs.append(record['measurement']['receipt'])
        require(len(set(transactions))==len(set(instance_ids))==3,'three separate physical instances/transactions required')
        bound=dict(operation=operation,gpus=gpus,
            duration_upper_s=max(max(r['finished_s']-r['started_s'],raw['duration_s']) for r,raw in records),
            energy_upper_j=max(raw['energy_j'] for r,raw in records),
            peak_memory_per_gpu_upper_bytes=max(raw['peak_memory_per_gpu_bytes'][str(g)] for r,raw in records for g in gpus))
    else:
        raise ValueError('unknown capacity evidence group')
    bound['raw_sha256']=reference['sha256']
    return kind,bound,raw_refs


def build(identity, groups, out):
    require(not Path(out).exists(),'fresh immutable certificate output required')
    certificate=dict(schema='capacity-measured-bounds-v1',identity=identity,measurement_verified=True,
        bounds_are_empirical_not_hard_guarantees=True,three_independent_repetitions_per_item=True,
        evidence_groups=groups,layouts=[],transitions=[],savings=[])
    raw={}
    for group in groups:
        require(fixed(group)['kind'] in ('layout','transition','savings_grid'),
                'released savings must include the actual idle and two-low-rate grid')
        kind,bound,refs=derive_group(group,identity)
        certificate[{'layout':'layouts','transition':'transitions','savings':'savings'}[kind]].append(bound)
        raw.update({r['sha256']:r for r in refs})
    require(certificate['layouts'] and certificate['savings']
            and {b['operation'] for b in certificate['transitions']}=={'restore_cold','remove'},
            'actual layouts, matched savings and both physical directions are required')
    certificate['raw_measurements']=list(raw.values())
    durable(out,certificate)
    return ref(out)


def validate(certificate, identity):
    require(certificate.get('three_independent_repetitions_per_item') is True
            and certificate.get('evidence_groups'),'three-repetition source groups required')
    expected=dict(layouts=[],transitions=[],savings=[]);raw={}
    for group in certificate['evidence_groups']:
        require(fixed(group)['kind'] in ('layout','transition','savings_grid'),
                'released savings must include the actual idle and two-low-rate grid')
        kind,bound,refs=derive_group(group,identity)
        expected[{'layout':'layouts','transition':'transitions','savings':'savings'}[kind]].append(bound)
        raw.update({r['sha256']:r for r in refs})
    require(all(certificate[k]==v for k,v in expected.items()),'claimed bounds differ from actual three-repeat derivation')
    require({r['sha256']:r for r in certificate['raw_measurements']}==raw,'raw measurement membership differs')
    return {g['sha256'] for g in certificate['evidence_groups']}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--declaration',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    import json
    declaration=json.loads(args.declaration.read_text())
    print(build(declaration['identity'],declaration['evidence_groups'],args.out))


if __name__=='__main__':
    main()
