"""Read-only CPU recomputation of this gate's output, scheduling and energy evidence."""
import csv
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def integral(rows,start,end):
    assert rows[0][0]<=start<end<=rows[-1][0]
    total=[0.]*8
    for (a,va),(b,vb) in zip(rows,rows[1:]):
        assert b>a and len(va)==len(vb)==8
        lo,hi=max(a,start),min(b,end)
        if hi<=lo:continue
        for i in range(8):
            x=va[i]+(vb[i]-va[i])*(lo-a)/(b-a)
            y=va[i]+(vb[i]-va[i])*(hi-a)/(b-a)
            total[i]+=(x+y)*.5*(hi-lo)
    return total


def main():
    status=json.loads((ROOT/'status.json').read_text())
    result=json.loads((ROOT/'validation/result.json').read_text()) if (ROOT/'validation/result.json').exists() else {}
    manifest=json.loads((ROOT/'manifest.json').read_text())
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    assert all(sha(ROOT/p)==v for p,v in manifest['files'].items())
    frozen={p:sha(p)==v for p,v in manifest['frozen_inputs'].items()}; assert all(frozen.values())
    with (ROOT/'power/power.csv').open() as f:
        reader=csv.reader(f);header=next(reader)
        assert header[:9]==['t_s']+[f'gpu{i}_w' for i in range(8)]
        rows=[(float(x[0]),list(map(float,x[1:9]))) for x in reader]
    energy=integral(rows,status['measurement_start_s'],status['measurement_end_s'])
    error=sum(energy)-status['total_node_energy_j'];assert abs(error)<1e-6
    http=[json.loads(x) for x in (ROOT/'validation/http.jsonl').read_text().splitlines() if x] if (ROOT/'validation/http.jsonl').exists() else []
    complete=[r for r in http if r['route']=='/v1/completions' and r.get('status')==200]
    cancelled=[r for r in http if r['route']=='/v1/completions' and r.get('status',0)>=400]
    negative=[r for r in http if r.get('label')=='reject-short-budget-temporal']
    assert all(r.get('status') in (400,409) for r in negative)
    events={}
    for name in ('nextv3b0','nextv3b1'):
        path=ROOT/(name+'.events.jsonl')
        if not path.exists():continue
        raw=path.read_bytes(); assert not raw or raw.endswith(b'\n')
        events[name]=[json.loads(x) for x in raw.splitlines() if x]
    temporal_steps=sum(bool(e.get('tokens') and e.get('mode')=='temporal') for es in events.values() for e in es)
    assert temporal_steps==0
    records=[]
    for port,item in result.get('instances',{}).items():
        checks=item.get('checks',{}); budget_rows=[]; reference=checks.get('budget_8192',{}).get('outputs')
        for tokens in (8192,1024,2048):
            value=checks.get('budget_'+str(tokens))
            if value:
                outputs=value['outputs'];ack=value['ack']
                exact=outputs==reference and all(len(ids)==64 for ids in outputs.values())
                assert exact
                assert ack['generation']==ack['acknowledged_generation']
                assert ack['scheduler_budget_effective']==dict(max_num_batched_tokens=tokens,max_num_seqs=32)
                budget_rows.append(dict(tokens=tokens,exact_to_8192=exact,outputs=outputs,actual_events=value['events']))
        records.append(dict(port=port,passed=item.get('passed'),phase=item.get('phase'),error=item.get('error'),
            budgets=budget_rows,sequence_shrink=checks.get('sequence_shrink'),invalid_controls=checks.get('invalid_controls'),
            cancellation=checks.get('cancellation')))
    same_resident=[]
    for b,a in zip(status.get('identity_before',[]),status.get('identity_after',[])):
        same=b['container']['Id']==a['container']['Id'] and b['container']['State']['StartedAt']==a['container']['State']['StartedAt']
        assert same and b['provenance']==a['provenance']
        same_resident.append(dict(id=a['runtime']['id'],same_container_source=same,generation=a['runtime']['generation'],
            acknowledged_generation=a['runtime']['acknowledged_generation'],accepting=a['runtime']['accepting'],
            final_budget=a['runtime']['scheduler_budget_effective']))
    report=dict(schema_version=1,scope=status['scope'],scoped_gate_passed=status['passed'],
        total_runtime_gate='failed',temporal_correctness='unfixed',validator_unchanged_sha256=manifest['validator_sha256'],
        instances=records,cross_replica_outputs=status.get('cross_replica_outputs'),
        successful_completion_responses=len(complete),successful_output_tokens=sum(len(r['body'].get('token_ids',[])) for r in complete),
        cancelled_completion_responses=len(cancelled),
        cancellation_owner_steps={r['request_id']:sum(r['request_id'] in e.get('request_ids',[]) and bool(e.get('tokens'))
            for es in events.values() for e in es) for r in cancelled},
        temporal_negative_controls=len(negative),temporal_negative_statuses=[r['status'] for r in negative],
        actual_temporal_model_steps=temporal_steps,per_gpu_energy_j=energy,total_node_energy_j=sum(energy),
        reported_energy_difference_j=error,duration_s=status['measurement_end_s']-status['measurement_start_s'],
        max_power_sample_gap_s=max(b[0]-a[0] for a,b in zip(rows,rows[1:])),
        measurement_valid=status.get('measurement_valid'),cleanup_complete=status.get('cleanup_complete'),
        cleanup_elapsed_s=status.get('cleanup_elapsed_s'),same_resident=same_resident,frozen_inputs_unchanged=frozen,
        limits=['Independent continuous budget scope only; original total runtime gate remains failed.',
            'All eight GPU energy includes cancelled requests, failures and both inner and outer cleanup.',
            'No performance, SLO or energy improvement claim is made.'])
    (ROOT/'independent-audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('scoped_gate_passed','successful_completion_responses','successful_output_tokens',
        'cancelled_completion_responses','temporal_negative_statuses','actual_temporal_model_steps','total_node_energy_j',
        'reported_energy_difference_j','duration_s','cleanup_complete','cleanup_elapsed_s')}))


if __name__=='__main__':main()
