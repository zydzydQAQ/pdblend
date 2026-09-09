"""Independent CPU-only audit of the completed native reference and attempt003."""
import csv,hashlib,json,math,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent;C=ROOT.parent
N=C/'B32B-native-default-reference-attempt-001';T=C/'B32B-temporal-observation-attempt-003'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def integrate(rows,start,end):
    assert rows[0][0]<=start<end<=rows[-1][0]
    total=[0.]*8
    for (x,a),(y,b) in zip(rows,rows[1:]):
        assert y>x
        left,right=max(start,x),min(end,y)
        if left>=right:continue
        for i in range(8):
            vl=a[i]+(b[i]-a[i])*(left-x)/(y-x);vr=a[i]+(b[i]-a[i])*(right-x)/(y-x)
            total[i]+=(vl+vr)*(right-left)/2
    return total
def diff(a,b):
    assert len(a)==len(b)==64
    return next((dict(position_one_based=i+1,reference=x,observed=y) for i,(x,y) in enumerate(zip(a,b)) if x!=y),None)
def capture(p):
    return {(r['request_id'],r['rank'],r['output_index']):r
        for f in (p/'results/capture-frozen').glob('*.jsonl') for r in map(json.loads,f.read_text().splitlines())}

status=read(N/'results/status.json');child=read(N/'results/child/status.json');terminal=read(ROOT/'actual-terminal.json')
assert status['complete'] and status['observation_completed'] and status['measurement_valid'] and status['all_original_restored']
assert status['capture_complete'] and child['complete'] and child['completed_requests']==4 and child['cleanup_complete']
assert not any(terminal['processes_live'].values())
files=[N/'spec.json',N/'observation-spec.json',N/'results/status.json',N/'results/child/status.json',
       N/'results/child/full-outputs.json',N/'results/power/power.csv',N/'results/power/clocks.csv',
       N/'results/diagnostic-owner.events.jsonl',N/'results/restored-bootstrap.binding.json',
       N/'results/runtime-prefix-after-restore.json',T/'results/child/full-outputs.json',T/'results/status.json',ROOT/'actual-terminal.json']
files += list((N/'results/capture-frozen').glob('*'))+list((T/'results/capture-frozen').glob('*'))
before={str(p):sha(p) for p in files if p.is_file()}
with (N/'results/power/power.csv').open() as f:
    rows=[(float(r['t_s']),[float(r[f'gpu{i}_w']) for i in range(8)]) for r in csv.DictReader(f)]
assert all(len(v)==8 and all(math.isfinite(x) and x>=0 for x in v) for _,v in rows)
start,end=status['operation_start_s'],status['operation_end_s'];per_gpu=integrate(rows,start,end);energy=sum(per_gpu)
assert abs(energy-status['full_operation_energy_j'])<1e-6
phases=[]
for stage in status['stages']:
    e=sum(integrate(rows,stage['started_s'],stage['ended_s']));assert abs(e-stage['energy_j'])<1e-6
    phases.append(dict(phase=stage['phase'],duration_s=stage['ended_s']-stage['started_s'],energy_j=e))
assert abs(sum(x['energy_j'] for x in phases)-energy)<1e-6
ids=[r['request_uuid'] for r in read(N/'observation-spec.json')['requests']]
n=read(N/'results/child/full-outputs.json')['token_ids_by_request_uuid'];old=read(T/'results/child/full-outputs.json')['token_ids_by_request_uuid']
parity={rid:dict(exact=n[rid]==old[rid],first_difference=diff(old[rid],n[rid]),different_positions=sum(a!=b for a,b in zip(old[rid],n[rid]))) for rid in ids}
nc,tc=capture(N),capture(T);assert len(nc)==28
keys=['actual','block_size','computed_before','driver_block_table','expected_query_len','input_sequence_row',
      'input_token_offset','last_input_token','logits_row','num_decode_tokens','num_prefill_tokens','num_prefills',
      'ordered_request_seq_ids','output_len_before','prompt_len','raw_logits_dtype','raw_logits_shape','seq_id','sequence_len']
comparison=[]
for rid in (ids[1],ids[3]):
    for k in range(28,35):
        same_prefix=n[rid][:k-1]==old[rid][:k-1]
        for rank in (0,1):
            a,b=tc[rid,rank,k],nc[rid,rank,k]
            comparison.append(dict(request_id=rid,output_index=k,rank=rank,same_input_prefix=same_prefix,
                metadata_same={key:a.get(key)==b.get(key) for key in keys},
                old_logits=a.get('raw_model_logits_pre_sampler'),native_logits=b.get('raw_model_logits_pre_sampler')))
step32={rid:{'old':tc[rid,0,32],'native':nc[rid,0,32]} for rid in (ids[1],ids[3])}
report=dict(schema=1,checked_s=time.time(),actual_terminal_observed_s=terminal['observed_s'],performance_evidence=False,
    native_reference_complete=True,original_temporal_gate_changed=False,ecoserve_released=False,
    full_operation_energy_j=energy,reported_energy_j=status['full_operation_energy_j'],energy_difference_j=energy-status['full_operation_energy_j'],
    operation_duration_s=end-start,per_gpu_energy_j=per_gpu,power_frames=len(rows),max_power_gap_s=max(b[0]-a[0] for a,b in zip(rows,rows[1:])),phases=phases,
    all256_equal_original003=False,per_request_full64_parity=parity,native_pair_vs_own_solos=[diff(n[ids[j]],n[ids[j+2]]) for j in range(2)],
    native_pair_steps69=status['owner_evidence']['pair_steps']==69,total_actual_steps=status['owner_evidence']['steps'],
    step32=step32,same_prefix_capture_comparison=comparison,fresh_binding_sha256=sha(N/'results/restored-bootstrap.binding.json'),
    kv_content_correctness_proven=False,numerical_root_cause_proven=False,files=before)
assert before=={p:sha(Path(p)) for p in before}
(ROOT/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:report[k] for k in ['full_operation_energy_j','energy_difference_j','operation_duration_s','power_frames','max_power_gap_s','native_pair_vs_own_solos','total_actual_steps']}))
