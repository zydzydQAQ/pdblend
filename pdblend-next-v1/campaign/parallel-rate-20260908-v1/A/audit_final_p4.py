"""Read-only independently recomputed A P4 point and pair acceptance."""
from pathlib import Path
import collections,csv,hashlib,json,math,statistics,time
ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[2]
BASE=REPO/'campaign/five-system-results-v4/actual-snapshot-006/results.json'
SYSTEMS=('mixed','distserve','dynamollm','ecoserve')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
def require(v,s):
    if not v:raise ValueError(s)
def energy(path,start,end):
    with path.open() as f:rows=[(float(r['t_s']),sum(float(r['gpu'+str(g)+'_w']) for g in range(8))) for r in csv.DictReader(f)]
    require(rows[0][0]<=start<end<=rows[-1][0],'eight-board power fails to bracket measurement')
    total=0.
    for (t,a),(u,b) in zip(rows,rows[1:]):
        require(u>t and all(math.isfinite(x) and x>=0 for x in (a,b)),'invalid power sample')
        l,h=max(t,start),min(u,end)
        if l<h:total+=(h-l)*(a+(b-a)*(l-t)/(u-t)+a+(b-a)*(h-t)/(u-t))/2
    return total

def collect():
    require(sha(BASE)=='1ddd027859c026ebbea6dce212fe8369098de1505f18d489af2a8f0be3e3febb','original comparison snapshot changed')
    baseline=read(BASE)['points'];points=[];stages=[]
    for stage in (ROOT/'p4/fixed-screen-001',ROOT/'p4-minimal/fixed-screen-001'):
        status=read(stage/'status.json')
        require(status['node_lease_held'] is False and not Path('/proc',str(status['pid'])).exists(),'actual stage still owns/runs work')
        stages.append(dict(path=str(stage),phase=status['phase'],completed=status['completed'],skipped=status.get('skipped_saturated',[]),
                           reported_complete=status['complete'],engineering_fault_cell=status.get('engineering_fault_cell')))
        for cp in sorted((stage/'results/checkpoints').glob('*.json')):
            c=read(cp);require(sha(c['receipt'])==c['receipt_sha256'] and sha(c['binding'])==c['binding_sha256'],'checkpoint source changed')
            for path,h in c['artifacts'].items():require(sha(path)==h,'raw artifact changed '+path)
            receipt=read(c['receipt']);summary=receipt['summary'];row=c['row'];cell=stage/'results/cells'/row['cell_id']
            with (cell/'bench.csv').open() as f:bench=list(csv.DictReader(f))
            trace=read(row['trace']);require(sha(row['trace'])==row['trace_sha256'],'trace changed')
            expected=sum(x['output_len'] for x in trace['requests']);generated=sum(int(x['generated_tokens']) for x in bench)
            full=sum(x['success'] in ('1','true','True') and int(x['generated_tokens'])==int(x['output_len']) for x in bench)
            good=sum(x['slo_ok'] in ('1','true','True') for x in bench)
            measured=energy(cell/'power.csv',summary['measurement_start_s'],summary['measurement_end_s'])
            require(math.isclose(measured,summary['energy_j'],rel_tol=1e-9,abs_tol=1e-5),'all8 independent energy differs')
            require(len(bench)==summary['n_expected']==len(trace['requests']) and expected==summary['expected_generated_tokens']
                and generated==summary['generated_tokens'] and full==summary['completed_work_requests']
                and math.isclose(good/len(bench),summary['slo_attainment'],abs_tol=1e-12),'raw requests differ from summary')
            cleanup=bool(receipt['child_stopped'] and receipt['clock_restore_complete'] and not receipt['outer_cleanup_errors']
                         and summary['post_measurement_cleanup']['cleanup_complete'])
            require(cleanup and receipt['measurement_valid'] and summary['measurement_valid'] and summary['gpu_count']==8,'measurement/cleanup invalid')
            gate=read(stage/'results/operations'/row['cell_id']/'engineering-gate.json')
            work=full==len(bench) and generated==expected
            point=dict(cell_id=row['cell_id'],dataset=row['dataset'],rate_rps=row['rate_rps'],seed=row['seed'],
                checkpoint=str(cp),checkpoint_sha256=sha(cp),measurement_valid=True,all8_independent_energy_verified=True,
                work_complete=work,completed=full,n_expected=len(bench),generated_tokens=generated,expected_generated_tokens=expected,
                slo_attainment=good/len(bench),energy_j=measured,failed_requests=summary['failed_requests'],request_timeouts=summary['request_timeouts'],
                engine_gate_passed=gate['passed'],cleanup_passed=cleanup,ttft_avg_s=summary['ttft_avg_s'],tpot_avg_s=summary['tpot_avg_s'],pairs=[])
            for system in SYSTEMS:
                candidates=[p for p in baseline if p['model']=='14b' and p['phase']=='main' and p['slo_scale']==1 and p['dataset']==row['dataset'] and p['rate_rps']==row['rate_rps'] and p['system']==system]
                require(len(candidates)==1,'baseline not unique');b=candidates[0]
                require(b['trace_sha256']==row['trace_sha256'] and b['content_pairing_sha256']==row['content_pairing_sha256'] and b['seed']==row['seed']
                    and b['slo_ttft_s']==row['slo_ttft_s'] and b['slo_tpot_s']==row['slo_tpot_s'] and b['n_expected']==len(bench)
                    and b['expected_generated_tokens']==expected,'not exact workload comparison')
                relative=work and measured<=b['energy_j'] and point['slo_attainment']>=min(.9,b['slo_attainment'])
                strict=relative and b['work_complete'] and point['slo_attainment']>=.9 and b['slo_attainment']>=.9
                point['pairs'].append(dict(system=system,baseline_cell_id=b['cell_id'],baseline_slo=b['slo_attainment'],baseline_energy_j=b['energy_j'],
                    required_slo=min(.9,b['slo_attainment']),energy_saving_pct=100*(1-measured/b['energy_j']),
                    campaign_criterion_passed=relative,both_complete_and90_energy_win=strict))
            if row['dataset']=='alpaca' and row['rate_rps']==12:
                events=[json.loads(l) for l in (cell/'control.jsonl').open()]
                timing=[e for e in events if e['kind']=='request_timing' and e.get('completed')]
                point['latency_decomposition_s']={k:statistics.mean(e[b]-e[a] for e in timing if a in e and b in e)
                  for k,a,b in [('queue_to_reserve','queued_s','reserved_s'),('reserve_to_confirmation','reserved_s','backend_confirmed_s'),('confirmation_to_forward','backend_confirmed_s','forward_started_s')]}
                point['latency_failure_counts']=dict(ttft=sum(float(b['ttft_s'])>row['slo_ttft_s'] for b in bench if b['ttft_s']),
                    tpot=sum(float(b['tpot_s'])>row['slo_tpot_s'] for b in bench if b['tpot_s']))
            points.append(point)
    require(len({p['cell_id'] for p in points})==len(points),'duplicate P4 cell')
    points.sort(key=lambda p:({'alpaca':0,'sharegpt':1,'longbench':2}[p['dataset']],p['rate_rps']))
    return dict(schema='A-p4-independent-point-pair-audit-v1',created_s=time.time(),stages=stages,points=points,
        observed_points=len(points),work_complete_points=sum(p['work_complete'] for p in points),
        campaign_pairs_passed=sum(q['campaign_criterion_passed'] for p in points for q in p['pairs']),
        strict_pairs_passed=sum(q['both_complete_and90_energy_win'] for p in points for q in p['pairs']),
        baseline_snapshot=dict(path=str(BASE),sha256=sha(BASE)),profile_2400_measured=False,new_rates_measured=False,
        first_failure_slo90_is_rate_search_endpoint_only=True)
if __name__=='__main__':
    result=collect();out=ROOT/'final-p4-audit.json';require(not out.exists(),'new immutable audit output required')
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('points','stages')},indent=2))
    for p in result['points']:print(p['dataset'],p['rate_rps'],str(p['completed'])+'/'+str(p['n_expected']),p['slo_attainment'],p['energy_j'],[(q['system'],q['campaign_criterion_passed'],q['energy_saving_pct']) for q in p['pairs']])
