"""Read-only CPU correlation of retained controller plans, owner steps and actual SM clocks."""
import bisect,csv,json,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def jl(path):return [json.loads(s) for s in path.read_text().splitlines() if s]
def cs(rows,times,start,end,gpus):
    a,b=bisect.bisect_left(times,start),bisect.bisect_right(times,end)
    vals={str(g):[float(r[f'gpu{g}_sm_mhz']) for r in rows[a:b]] for g in gpus}
    return {g:dict(samples=len(v),min_mhz=min(v) if v else None,median_mhz=statistics.median(v) if v else None,
                   max_mhz=max(v) if v else None,fraction_2520=sum(x==2520 for x in v)/len(v) if v else None)
            for g,v in vals.items()}
def event_small(e):
    return {k:e.get(k) for k in ('started_s','finished_s','tokens','prefill','decode','request_ids','generation','role','mode')}
def cell(budget):
    out=ROOT/f'cell-longbench-budget{budget}';op=ROOT/f'budget{budget}-operation'
    control=jl(out/'control.jsonl');bench=list(csv.DictReader((out/'bench.csv').open()))
    timings={x['client_request_id']:x for x in control if x.get('kind')=='request_timing'}
    adm={x['client_request_id']:x for x in control if x.get('kind')=='admission'}
    snaps=[x for x in control if 'snapshot' in x and x.get('kind')=='pdb_stale_tail_fallback_snapshot']
    clocks=list(csv.DictReader((op/'power/clocks.csv').open()));times=[float(r['t_s']) for r in clocks]
    events={n:jl(op/(n+'.events.jsonl')) for n in ['nextv3b0','nextv3b1']}
    result=[]
    for r in bench:
        idx=r['idx'];t=timings[idx];a=adm[idx];rid=t['request_id'];route=next(x for x in a['plan']['routes'] if x['request_id']==rid)
        instance=route['prefill_id'];gpus=[0,1] if instance=='nextv3b0' else [2,3]
        own=[e for e in events[instance] if rid in e.get('request_ids',[]) and e['finished_s']<=t['first_token_s']]
        pre=[e for e in own if e['prefill']>0]
        f=t['forward_started_s'];end=t['first_token_s']
        exact=[x for x in snaps if x['request_id']==rid and x['snapshot']['version']==a['plan']['snapshot_version']]
        cache=exact[-1]['snapshot'] if exact else None
        snapshot_ages={x['instance_id']:f-x['timestamp_s'] for x in cache['instances']} if cache else None
        step_details=[]
        for e in pre:
            item=event_small(e);item['duration_s']=e['finished_s']-e['started_s'];item['actual_clocks']=cs(clocks,times,e['started_s'],e['finished_s'],gpus)
            step_details.append(item)
        busy={n:[event_small(e) for e in ev if e['started_s']<=f<e['finished_s']] for n,ev in events.items()}
        result.append(dict(idx=int(idx),request_id=rid,input_tokens=int(r['prompt_len']),output_tokens=int(r['output_len']),
            slo_ok=r['slo_ok']=='1',ttft_s=float(r['ttft_s']),tpot_s=float(r['tpot_s']),
            planned_to_first_planning_s=t['first_planning_s']-t['planned_arrival_s'],
            first_planning_to_forward_s=f-t['first_planning_s'],forward_to_first_token_s=end-f,
            action_lock_wait_s=t['action_acquired_s']-t['action_wait_started_s'],
            backend_confirm_s=t['backend_confirmed_s']-t['reserved_s'],
            timing=t,route=route,plan_frequencies=a['plan']['frequencies'],clock_outcomes=a['clock_outcomes'],plan_reason=a['plan']['reason'],
            first_planning_actual_observation_ages_s=t['first_instance_ages_s'],
            exact_admission_snapshot_retained=cache is not None,exact_admission_snapshot=cache,
            exact_admission_observation_ages_at_forward_s=snapshot_ages,
            actual_owner_steps_in_progress_at_forward=busy,
            completed_prefill_steps_containing_request_before_first_token=step_details,
            completed_prefill_step_duration_sum_s=sum(e['finished_s']-e['started_s'] for e in pre),
            forward_to_first_own_step_s=own[0]['started_s']-f if own else None,
            completed_prefill_shared_with_decode=any(e['decode']>0 for e in pre),
            forward_to_first_actual_clocks=cs(clocks,times,f,end,gpus),
            output_token_sha256=r['output_token_sha256']))
    bad=[r for r in result if not r['slo_ok']]
    return dict(budget=budget,requests=result,bad_request_indices=[r['idx'] for r in bad],
       bad_request_plan_frequencies=sorted({x['frequency_mhz'] for r in bad for x in r['plan_frequencies']}),
       bad_request_shared_prefill_indices=[r['idx'] for r in bad if r['completed_prefill_shared_with_decode']],
       exact_admission_snapshots_retained=sum(r['exact_admission_snapshot_retained'] for r in result))

def main():
    cells=[cell(b) for b in (8192,2048)]
    result=dict(schema_version=1,scope='CPU raw evidence correlation; no replay and no source changes',cells=cells,
      caveats=['Actual SM-clock samples are contained in each owner step wall-time interval; they do not certify every microsecond.',
      'request_timing.first_instance_ages_s is age of real cached owner state at first planning, not at dispatch.',
      'Admission journals normally retain a snapshot version but not its full state. Exact dispatch-time observation age is null unless a retained fallback snapshot matches the executed admission version.',
      'An owner event spanning forward time proves actual scheduled work in flight, not the exact cached controller observation timestamp.',
      'No heartbeat timestamp is treated as a fresh owner observation. Token count equality is not token ID equality.',
      'A prefill event containing a request can also include decode for that request; raw events retain batch counts and IDs but not per-request stage. Only events completed by observed HTTP first token are counted in the before-first-token section.'])
    a,b=cells
    result['cross_budget_output_hash_difference_indices']=[x['idx'] for x,y in zip(a['requests'],b['requests']) if x['output_token_sha256']!=y['output_token_sha256']]
    result['good_to_bad_indices']=[x['idx'] for x,y in zip(a['requests'],b['requests']) if x['slo_ok'] and not y['slo_ok']]
    result['bad_to_good_indices']=[x['idx'] for x,y in zip(a['requests'],b['requests']) if not x['slo_ok'] and y['slo_ok']]
    (ROOT/'slo-owner-clock-audit.json').write_text(json.dumps(result,indent=2)+'\n')
    for c in cells:
        print('BUDGET',c['budget'],'bad',c['bad_request_indices'],'freq',c['bad_request_plan_frequencies'],'shared',c['bad_request_shared_prefill_indices'],'fullsnapshotcount',c['exact_admission_snapshots_retained'])
        for r in c['requests']:
            if r['slo_ok']:continue
            print(r['idx'],r['route']['prefill_id'],'ttft',round(r['ttft_s'],3),'planwait',round(r['first_planning_to_forward_s'],3),'engine',round(r['forward_to_first_token_s'],3),'prefill',round(r['completed_prefill_step_duration_sum_s'],3),'steps',len(r['completed_prefill_steps_containing_request_before_first_token']),'queue',round(r['forward_to_first_own_step_s'],4),'shared',r['completed_prefill_shared_with_decode'],'clocks',[(x['min_mhz'],x['median_mhz'],x['max_mhz'],round(x['fraction_2520'],3)) for e in r['completed_prefill_steps_containing_request_before_first_token'] for x in e['actual_clocks'].values()],'fresh0', {k:round(v,3) for k,v in r['first_planning_actual_observation_ages_s'].items()},'exactdispatch',r['exact_admission_observation_ages_at_forward_s'])
    print({k:v for k,v in result.items() if k not in ('cells','caveats')})
if __name__=='__main__':main()
