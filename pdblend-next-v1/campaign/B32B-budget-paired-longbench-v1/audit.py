"""CPU audit of original work, unchanged SLO and eight-GPU measurement integration."""
import csv
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def power_rows(path):
    with path.open() as f:
        reader=csv.reader(f);header=next(reader)
        assert header[:9]==['t_s']+[f'gpu{i}_w' for i in range(8)]
        return [(float(r[0]),list(map(float,r[1:9]))) for r in reader]


def integrate(rows,start,end):
    assert rows[0][0]<=start<end<=rows[-1][0]
    values=[0.]*8
    for (a,va),(b,vb) in zip(rows,rows[1:]):
        assert b>a
        lo,hi=max(a,start),min(b,end)
        if lo>=hi:continue
        for i in range(8):
            left=va[i]+(vb[i]-va[i])*(lo-a)/(b-a);right=va[i]+(vb[i]-va[i])*(hi-a)/(b-a)
            values[i]+=(left+right)*.5*(hi-lo)
    return values


def audit_cell(tokens,entry,trace):
    out=ROOT/f'cell-longbench-budget{tokens}';op=ROOT/f'budget{tokens}-operation'
    summary=json.loads((out/'summary.json').read_text())
    with (out/'bench.csv').open() as f:bench=list(csv.DictReader(f))
    assert len(bench)==len(trace['requests'])==64 and {int(r['idx']) for r in bench}==set(range(64))
    good=completed=generated=0;work=[]
    for row in bench:
        i=int(row['idx']);request=trace['requests'][i]
        assert int(row['prompt_len'])==request['prompt_len'] and int(row['output_len'])==request['output_len']
        success=row['success']=='1';completed+=success
        observed=int(row['generated_tokens']) if row['generated_tokens'] else 0;generated+=observed
        ok=success and float(row['ttft_s'])<=5 and float(row['tpot_s'])<=.1
        assert (row['slo_ok']=='1')==ok;good+=ok
        if success:assert int(row['input_tokens'])==request['prompt_len'] and observed==request['output_len']
        work.append(dict(idx=i,success=success,input_tokens=row['input_tokens'],generated_tokens=observed,
            original_input_tokens=request['prompt_len'],original_output_tokens=request['output_len'],slo_ok=ok,
            ttft_s=row['ttft_s'],tpot_s=row['tpot_s'],output_token_sha256=row['output_token_sha256']))
    assert completed==summary['completed'] and good/64==summary['slo_attainment']
    primary=power_rows(out/'power.csv');per_gpu=integrate(primary,summary['measurement_start_s'],summary['measurement_end_s'])
    assert abs(sum(per_gpu)-summary['energy_j'])<1e-5
    if good:assert abs(sum(per_gpu)/good-summary['energy_per_good_request_j'])<1e-5
    outer=power_rows(op/'power/power.csv');outer_gpu=integrate(outer,entry['observation_start_s'],entry['observation_end_s'])
    assert abs(sum(outer_gpu)-entry['full_operation_energy_j'])<1e-5
    simultaneous=sum(integrate(outer,summary['measurement_start_s'],summary['measurement_end_s']))
    events=[]
    for name in ('nextv3b0','nextv3b1'):
        raw=(op/(name+'.events.jsonl')).read_bytes();assert not raw or raw.endswith(b'\n')
        events.extend(json.loads(x) for x in raw.splitlines() if x)
    model=[x for x in events if x.get('tokens',0)]
    assert model and all(x['mode']=='continuous' and x['role']=='mixed' and x['tokens']<=tokens for x in model)
    before=json.loads((op/'budget-before.json').read_text());after=json.loads((op/'budget-after.json').read_text())
    for port,row in before.items():
        b=row['after'];a=after[port]
        for r in (b,a):
            assert r['generation']==r['acknowledged_generation'] and r['scheduler_budget_pending'] is None
            assert r['scheduler_budget_effective']==dict(max_num_batched_tokens=tokens,max_num_seqs=32)
            owners=[x.get('controls',{}).get('runtime') for x in r['scheduler_io']]
            assert owners and all(x and x['generation']==r['generation'] and not x.get('error') for x in owners)
    cleanup=json.loads((op/'outer-cleanup.json').read_text())
    assert cleanup['complete'] and summary['post_measurement_cleanup']['cleanup_complete']
    for row in cleanup['native'].values():
        proof=row['drain'];assert proof['send_counters_verified'] and len(proof['transfers'])==2
        for r in proof['transfers']:
            assert r['send_counters_observed'] and r['send_healthy'] and r['send_started']==r['send_completed'] and not r['send_failed']
            assert not any(r.get(k) for k in ('buffered_tensors','inflight_sends','inflight_receives','allocations','buffered_gpu_bytes'))
        r=row['restored'];assert r['accepting'] and r['generation']==r['acknowledged_generation']
        assert r['scheduler_budget_effective']==dict(max_num_batched_tokens=8192,max_num_seqs=32)
    return dict(tokens=tokens,completed=completed,n_expected=64,good_requests=good,slo_attainment=good/64,
        input_tokens=sum(int(x['input_tokens'] or 0) for x in bench),generated_tokens=generated,
        work_complete=summary['work_complete'],measurement_valid=summary['measurement_valid'],
        energy_j=sum(per_gpu),per_gpu_energy_j=per_gpu,energy_per_good_request_j=sum(per_gpu)/good if good else None,
        duration_s=summary['measurement_end_s']-summary['measurement_start_s'],
        whole_operation_energy_j=sum(outer_gpu),whole_operation_per_gpu_j=outer_gpu,
        outer_same_window_energy_j=simultaneous,outer_same_window_relative_difference=simultaneous/sum(per_gpu)-1,
        max_power_sample_gap_s=max(b[0]-a[0] for a,b in zip(primary,primary[1:])),
        model_steps=len(model),temporal_model_steps=0,max_actual_scheduled_tokens=max(x['tokens'] for x in model),
        post_controller_cleanup=summary['post_measurement_cleanup'],outer_cleanup_elapsed_s=cleanup['elapsed_s'],
        ttft_avg_s=summary.get('ttft_avg_s'),ttft_p99_s=summary.get('ttft_p99_s'),
        tpot_avg_s=summary.get('tpot_avg_s'),tpot_p99_s=summary.get('tpot_p99_s'),work=work)


def main():
    status=json.loads((ROOT/'status.json').read_text());trace=json.loads((ROOT.parent/'B32B-io-v1/longbench.trace.json').read_text())
    records=[]
    for tokens in (8192,2048):
        entry=status['cells'].get(str(tokens))
        if entry and entry.get('complete') and (ROOT/f'cell-longbench-budget{tokens}/summary.json').exists():
            records.append(audit_cell(tokens,entry,trace))
    result=dict(schema_version=1,scope='paired developer budget ablation; inherited profiles uncertified',cells=records,
        trace_sha256=hashlib.sha256((ROOT.parent/'B32B-io-v1/longbench.trace.json').read_bytes()).hexdigest(),
        original_full_runtime_gate='failed',temporal_repaired=False,primary_energy_scope='schema3 all-eight-GPU actual work and drain',
        whole_operation_energy_scope='separately measured setup, primary cell, all failures, controller cleanup and native final restoration')
    if len(records)==2:
        a,b=records;result['comparison']=dict(same_completed_work=a['work_complete'] and b['work_complete']
            and a['input_tokens']==b['input_tokens']==391052 and a['generated_tokens']==b['generated_tokens']==2711,
            energy_relative_change=b['energy_j']/a['energy_j']-1,
            energy_per_good_relative_change=b['energy_per_good_request_j']/a['energy_per_good_request_j']-1,
            slo_percentage_point_change=100*(b['slo_attainment']-a['slo_attainment']),
            exact_output_hash_matches=sum(x['output_token_sha256']==y['output_token_sha256'] for x,y in zip(a['work'],b['work'])))
    (ROOT/'independent-audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='cells'}))
    for r in records:print(json.dumps({k:v for k,v in r.items() if k not in ('work','post_controller_cleanup')}))


if __name__=='__main__':main()
