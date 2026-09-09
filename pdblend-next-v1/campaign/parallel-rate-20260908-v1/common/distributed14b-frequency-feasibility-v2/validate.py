"""Read-only evidence for bounded target-host profile checks, never a new profile."""
import hashlib,json,math,statistics
from pathlib import Path

def require(ok,message):
    if not ok:raise ValueError(message)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def ref(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def fixed(r):
    require(sha(r['path'])==r['sha256'],'reference changed: '+r['path']);return read(r['path'])
def source_check(spec):
    require(spec['schema']=='distributed14b-frequency-feasibility-probes-v1' and spec['authorized'] is True,'explicit probe declaration required')
    require(spec['profile_publication_allowed'] is False and spec['request_timeout_s']==120 and spec['cleanup_timeout_s']==120,'unchanged bounded request/cleanup and no fabricated profile required')
    require(spec['points'] and len({p['point_id'] for p in spec['points']})==len(spec['points']),'unique prescribed probes required')
    require(all(sha(p)==h for p,h in spec['files'].items()),'probe source changed')
    jobs=fixed(spec['jobs']);profile=fixed(spec['profile_reference']);binding=fixed(spec['binding']);ordinary=fixed(spec['ordinary'])
    require(jobs['node']==spec['node'] and jobs['model']=='14b' and binding['model']=='14b' and binding['hostname']==spec['hostname'],'target/model mismatch')
    require(jobs['profile_reference']==spec['profile_reference'] and profile['model']=='Qwen2.5-14B-Instruct','original reference profile changed')
    require(ordinary['passed'] and ordinary['complete'] and ordinary['measurement_valid'] and not ordinary['errors'],'actual native ordinary prerequisite failed')
    require(fixed(spec['ordinary_invocation'])['binding']==spec['binding'],'ordinary bound to another deployment')
    instances={i['id']:i for i in binding['instances']}
    for p in spec['points']:
        require(p['frequency_mhz']==2400 and spec.get('candidate_unprofiled_frequency_mhz')==2400,'only explicitly unprofiled2400 feasibility allowed')
        require(p['instance_id'] in instances and instances[p['instance_id']]['tp']==1 and instances[p['instance_id']]['gpus']==[p['gpu']],'probe hardware identity mismatch')
        require(all(type(p[k]) is int and p[k]>0 for k in ('frequency_mhz','input_tokens','output_tokens','batch')) and p['input_tokens']+p['output_tokens']<=8192 and p['batch']<=32,'invalid probe geometry')
        candidates=[x for x in profile['points'] if x['role']=='mixed' and x['tp']==1 and x['input_tokens']==p['input_tokens'] and x['batch']==p['batch'] and x['context_tokens']>=p['input_tokens']+p['output_tokens']]
        require(candidates,'probe has no declared input/batch and sufficient-context geometry; this check supplies no2400performance reference')
    return binding

def native_saved(raw,expected_id,tokens):
    gen=raw['generation']
    require(raw['id']==expected_id and type(gen) is int and gen==raw['acknowledged_generation'] and raw['acknowledged_generations']==[gen] and raw['observed_control_generation']==gen,'saved native ACK invalid')
    require(raw['scheduler_budget_pending'] is None and raw['scheduler_budget_effective']==dict(max_num_batched_tokens=tokens,max_num_seqs=32),'saved applied budget invalid')
    require(raw['transport_healthy'] is True and raw['transfer_send_counters_observed'] is True and raw['transfer_inflight_sends_observed'] is True and raw['transfer_send_healthy'] is True,'saved native sender unobserved')
    counts=[raw[k] for k in ('transfer_send_started','transfer_send_completed','transfer_send_failed')]
    require(all(type(x) is int and x>=0 for x in counts) and counts[0]==counts[1] and counts[2]==0,'saved pending or failed send')
    owners=[x.get('controls',{}).get('runtime') for x in raw['scheduler_io']]
    require(len(owners)==1 and owners[0]['generation']==gen and owners[0]['error'] is None,'saved TP1 owner ACK differs')
    require(raw.get('runtime_error') is None and raw.get('error') is None and raw['role']=='mixed' and raw['mode']=='continuous' and raw['accepting'] is True,'saved native mode unhealthy')
    require(all(k in raw and not raw[k] for k in ('active','running','waiting','kv_allocations','transfer_allocations','transfer_buffered_tensors','transfer_inflight_receives','transfer_inflight_sends')),'saved native residue')

def cleanup_saved(cleanup,expected_id):
    require(cleanup['complete'] is True and not cleanup['errors'],'saved native cleanup failure')
    before,proof,resumed=cleanup['before'],cleanup['proof'],cleanup['resumed'];native_saved(before,expected_id,2048)
    require(proof['drained'] is True and proof['accepting'] is False and proof['generation']==before['generation']+1 and proof['drain_proof_type']=='synchronous_put_owner_barrier' and proof['send_counters_verified'] is True,'saved native barrier invalid')
    require(len(proof['transfers'])==1,'saved native TP rank absent')
    rank=proof['transfers'][0]
    require(rank['send_counters_observed'] is True and rank['send_healthy'] is True and rank['listener_alive'] is True and all(not rank[k] for k in ('buffered_tensors','inflight_sends','send_failed','inflight_receives','buffered_gpu_bytes','allocations')) and rank['send_started']==rank['send_completed'],'saved native barrier residue')
    native_saved(resumed['after'],expected_id,8192)
    require(resumed['after']['generation']==proof['generation']+1==resumed['control']['generation'],'saved native restore generation differs')

def validate_point(raw,events,clock_samples):
    p=raw['point']; n=p['batch']; ids={r.get('request_id') for r in raw['requests']};length=p['input_tokens'];out=p['output_tokens']
    require(raw.get('error') is None and raw['cleanup']['complete'] is True,'probe/cleanup failed')
    require(len(ids)==len(raw['requests'])==n and all(isinstance(r,str) and r.startswith('pdb-a-batch-') for r in ids),'owned batch incomplete')
    prompt=([9707,1879,13]*(length//3+1))[:length]
    for r in raw['requests']:
        require(r.get('success') is True and r.get('done_marker') is True and r.get('http_status')==200 and r.get('prompt_token_ids')==prompt and r.get('requested_output_tokens')==out,'failed/changed request')
        require(len(r.get('output_token_ids',[]))==len(r.get('token_received_s',[]))==out and all(type(x) is int for x in r['output_token_ids']),'output token work missing')
        require(r.get('usage',{}).get('prompt_tokens')==length and r['usage'].get('completion_tokens')==out,'terminal token usage differs')
        times=r['token_received_s']; require(all(type(t) in (int,float) and math.isfinite(t) for t in times) and times==sorted(times),'token times invalid')
        require(raw['started_s']<=r['dispatch_s']<=times[0]<=times[-1]<=raw['finished_s'],'request escapes probe window')
    require(all(r['output_token_ids']==raw['requests'][0]['output_token_ids'] for r in raw['requests']),'same-prompt batch numerical outputs diverge')
    before=raw['runtime_before']; generation=before['generation']; after=raw['runtime_after_requests']
    cleanup_saved(raw['cleanup'],p['instance_id'])
    for state in (before,after):
        native_saved(state,p['instance_id'],2048)
        require(state['id']==p['instance_id'] and state['generation']==generation==state['acknowledged_generation'] and state['scheduler_budget_effective']['max_num_batched_tokens']==2048 and state.get('runtime_error') is None and state.get('error') is None,'actual service budget/ACK changed')
        require(all(k in state and not state[k] for k in ('active','running','waiting','kv_allocations','transfer_allocations','transfer_buffered_tensors','transfer_inflight_receives','transfer_inflight_sends')),'native work residue')
    pf=dc=0; decode=[]; owned=[];last=None
    for e in events:
        a,b=e['started_s'],e['finished_s'];rs=e['request_ids']
        require(math.isfinite(a) and math.isfinite(b) and raw['started_s']<=a<b<=raw['finished_s'] and (last is None or a>=last-1e-6),'native owner timeline invalid');last=b
        require(e['role']=='mixed' and e['mode']=='continuous' and e['generation']==generation and len(rs)==len(set(rs)) and set(rs)<=ids,'native owner/gen/mode differs')
        require(all(type(e[k]) is int and e[k]>=0 for k in ('prefill','decode','tokens')) and e['prefill']+e['decode']==len(rs) and e['tokens']<=2048 and len(rs)<=32,'native applied phase/budget invalid')
        require((e['prefill']>0 and e['tokens']>e['decode']) or (e['prefill']==0 and e['tokens']==e['decode']),'native token accounting differs')
        pf+=e['tokens']-e['decode'];dc+=e['decode']
        if rs:owned.append(e)
        if not e['prefill'] and e['decode']==n and set(rs)==ids:decode.append(e)
    require(pf==n*length and dc==n*(out-1),'native prefill/decode token accounting incomplete')
    require(len(decode)>=16,'representative full-batch decode absent')
    require(clock_samples and all(len(v)==8 and all(type(x) in (int,float) and math.isfinite(x) and x>0 for x in v) for t,v in clock_samples) and all(b[0]>a[0] for a,b in zip(clock_samples,clock_samples[1:])),'actual all8 clock stream invalid')
    active=[v[p['gpu']] for t,v in clock_samples if any(e['started_s']<=t<=e['finished_s'] for e in owned)]
    require(len(active)>=2 and all(abs(x-p['frequency_mhz'])<=15 for x in active),'loaded SM outside original +/-15 MHz command band')
    ttft=statistics.mean(r['token_received_s'][0]-r['dispatch_s'] for r in raw['requests'])
    tpot=statistics.mean((r['token_received_s'][-1]-r['token_received_s'][0])/(out-1) for r in raw['requests'])
    return dict(passed=True,actual_prefill_tokens=pf,actual_decode_tokens=dc,observed_full_batch_decode_steps=len(decode),owner_full_batch_iteration_s=statistics.mean(e['finished_s']-e['started_s'] for e in decode),client_ttft_s=ttft,client_tpot_s=tpot,loaded_clock=dict(samples=len(active),min_mhz=min(active),max_mhz=max(active),tolerance_mhz=15),output_sha256=hashlib.sha256(json.dumps(raw['requests'][0]['output_token_ids']).encode()).hexdigest(),profile_publication_allowed=False,entire_context_domain_remeasured=False)

def prediction_error(result,p,profile):
    require(p['frequency_mhz']==2400,'only2400 feasibility')
    return dict(reference_points=[],iteration_relative_errors=[],client_ttft_to_reference_prefill_ratios=[],not_new_target_profile=True,same_frequency_reference_available=False,comparison='No prior2400 reference; only real finite workload, native cleanup and loaded-clock feasibility; no profile is published or inferred from2520')
