"""Actual two-card 2100-MHz same-shape empirical profile evidence."""
import bisect,hashlib,importlib.util,json,math,statistics,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent
PARENT=HERE.parent/'distributed14b-frequency-feasibility-v2/validate.py'
PARENT_SHA='997c57f55c65490c78db3133157db2dc90a25e41b4f2b4b751647381c3a77234'

def require(value,message):
    if not value:raise ValueError(message)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def ref(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def fixed(r):
    require(sha(r['path'])==r['sha256'],'changed reference '+r['path']);return read(r['path'])
def load(path,name):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m

def geometry(profile):
    groups={}
    for row in profile['points']:
        if row['role']=='mixed' and row['tp']==1 and row['frequency_mhz'] in (900,1500,2100,2520):
            key=row['input_tokens'],row['batch'];groups[key]=max(groups.get(key,0),row['context_tokens'])
    expected={(128,b) for b in (1,2,4,8,16,32)}|{(2048,b) for b in (1,4,8,16)}|{(6144,6)}|{(7168,b) for b in (1,4,6)}
    require(set(groups)==expected and len(groups)==14,'actual original profile geometry differs')
    priority={(2048,16):0,(7168,6):1,(128,32):2}
    return [dict(input_tokens=n,batch=b,context_tokens=groups[n,b]) for n,b in sorted(groups,key=lambda x:(priority.get(x,3),x))]

def declared_points(profile,binding):
    instances=sorted(binding['instances'],key=lambda i:i['gpus'])
    require([(i['tp'],i['gpus']) for i in instances]==[(1,[6]),(1,[7])],'two actual TP1 GPU6/7 required')
    result=[]
    for shape in geometry(profile):
        for repeat,instance in enumerate(instances,1):
            result.append(dict(shape,point_id=f"in{shape['input_tokens']}-b{shape['batch']}-ctx{shape['context_tokens']}-f2100-gpu{instance['gpus'][0]}",
                frequency_mhz=2100,output_tokens=1024,instance_id=instance['id'],gpu=instance['gpus'][0],repeat=repeat,
                arrival_offsets_s=[0.]*(shape['batch']-1)+([2.] if shape['batch']>1 else [0.]),
                independent_arrival_seed=False,full_workload_output_extended_only_for_microprofile=True))
    return result

def source_check(spec):
    require(spec['schema']=='distributed14b-actual2100-profile-input-v1' and spec['authorized'] is True,'explicit measured-profile declaration required')
    require(spec.get('registration_frequency_mhz')==spec.get('max_service_frequency_mhz')==2100 and spec['node']=='B','explicit B 2100 successor required')
    negative=fixed(spec['prior_domain_negative'])
    require(negative['classification']=='complete_native_work_failed_loaded_clock_qualification' and negative['native_cleanup_complete'] and negative['clock_restore_complete'] and not negative['profile_published'],'prior 2400 negative not preserved with cleanup')
    require(spec['request_timeout_s']==120 and spec['cleanup_timeout_s']==120 and spec['frequency_tolerance_mhz']==15,
            'original request/native/clock limits changed')
    require(all(sha(p)==h for p,h in spec['files'].items()),'measurement source changed')
    jobs=fixed(spec['jobs']);profile=fixed(spec['profile_reference']);binding=fixed(spec['binding'])
    require(jobs['node']==spec['node'] and spec['node'] in ('B','C') and jobs['model']==binding['model']=='14b'
            and binding['hostname']==spec['hostname'],'actual assigned node/model differs')
    require(jobs['profile_reference']==spec['profile_reference'] and profile['model']=='Qwen2.5-14B-Instruct','original reference changed')
    require(spec['points']==declared_points(profile,binding) and len(spec['points'])==28,'all fourteen shapes on both cards required')
    require(spec['transition_gpus']==[6,7] and spec['transition_pairs']==[[a,b] for a in (900,1500,2100) for b in (900,1500,2100) if a!=b],
            'all six directed transitions on both cards required')
    require(spec['samples_per_shape']==2 and spec['samples_per_transition']==2,'exact two-card empirical samples required')
    feasible=fixed(spec['feasibility']);fspec=fixed(spec['feasibility_spec'])
    require(feasible['passed'] and feasible['complete'] and feasible['measurement_valid'] and not feasible['errors']
            and fspec['binding']==spec['binding'],'actual prior low-frequency qualification did not pass on this deployment')
    require({p['instance_id'] for p in fspec['points'] if p['frequency_mhz']==2100}=={i['id'] for i in binding['instances']},'both actual cards require prior 2100 qualification')
    require(spec['host_manifest']==ref(Path(binding['host_release'])/'manifest.json'),'actual sampler host source differs')
    require(spec['source_order_contract']==ref(HERE/'source-order-contract.json'),'source-order proof changed')
    require(all(p['input_tokens']+p['output_tokens']<=8192 for p in spec['points']),'micro context exceeds actual model limit')
    return binding

def phase_clock(clock_samples,gpu,frequency,start,end):
    require(clock_samples and all(math.isfinite(t) and len(v)==8 for t,v in clock_samples)
            and all(b[0]>a[0] for a,b in zip(clock_samples,clock_samples[1:])), 'complete original clock order required')
    ts=[r[0] for r in clock_samples];lo=bisect.bisect_right(ts,start)-1;hi=bisect.bisect_left(ts,end)
    require(0<=lo<hi<len(ts),'actual clock stream must bracket each phase')
    rows=clock_samples[lo:hi+1]
    require(max(b[0]-a[0] for a,b in zip(rows,rows[1:]))<=.25,'phase clock sampling gap exceeds original .25s')
    values=[v[gpu] for _,v in rows]
    require(all(type(x) in (int,float) and math.isfinite(x) and abs(x-frequency)<=15 for x in values),'actual phase frequency outside original +/-15MHz')
    return dict(samples=len(rows),first_s=rows[0][0],last_s=rows[-1][0],min_mhz=min(values),max_mhz=max(values),tolerance_mhz=15)

def validate_profile_point(raw,events,power,clock_samples,source_order_verified=False):
    from ecopadg.metrics import clip_power_window
    from ecopadg.measure.power import trapezoid_energy
    original=load(PARENT,'actual2100_original_native_validator')
    p=raw['point'];start,end=raw['started_s'],raw['finished_s'];gpu=p['gpu'];n=p['batch']
    clocks=[r for r in clock_samples if start-.5<=r[0]<=end+.5]
    native=original.validate_point(raw,events,clocks)
    require(source_order_verified is True,'actual scheduler metadata order not verified')
    require(sorted(r['arrival_offset_s'] for r in raw['requests'])==sorted(p['arrival_offsets_s']), 'declared staggered mixed interference differs')
    compatible=dict(raw,spec=dict(batch_size=n,input_tokens=p['input_tokens'],output_tokens=p['output_tokens']),
                    measurement_start_s=start,measurement_end_s=end)
    selected=events;terminal=None
    if any(not e.get('request_ids') for e in events):
        selected,terminal=load(HERE/'terminal.py','actual2100_terminal_empty').trailing_empty(compatible,events)
    context=load(HERE/'context_export.py','actual2100_logical_context')
    late=context.reconstruct(compatible,selected,generation=raw['runtime_before']['generation'],token_budget=2048,source_order_verified=True)
    late=context.attach_power(late,power,clocks,target_gpu=gpu,frequency_mhz=2100)
    require(min(v['attention_after_min'] for v in late['per_request'].values())>=p['context_tokens'],
            'full simultaneous late-decode window does not reach the registered context')
    ids={r['request_id'] for r in raw['requests']};prefills=[];runs=[];run=[]
    def energy(a,b):
        w=clip_power_window(power,a,b,pad_s=0)
        require(all(len(v)==8 and all(math.isfinite(x) and x>=0 for x in v) for _,v in w),'actual all8 phase power incomplete')
        return trapezoid_energy(w),trapezoid_energy([(t,[v[gpu]]) for t,v in w])
    for rid in sorted(ids):
        pf=[e for e in selected if rid in e['request_ids'][:e['prefill']]]
        require(pf,'actual request prefill missing');a,b=pf[0]['started_s'],pf[-1]['finished_s'];whole,target=energy(a,b)
        prefills.append(dict(request_id=rid,start_s=a,end_s=b,duration_s=b-a,target_gpu_power_w=target/(b-a),
            all8_energy_j=whole,background_decode_max=max(e['decode'] for e in pf),clock=phase_clock(clocks,gpu,2100,a,b)))
    require(max(r['background_decode_max'] for r in prefills)==n-1,'same-shape new prefill beside all other decodes absent')
    for e in selected:
        if e['prefill']==0 and e['decode']==n and set(e['request_ids'])==ids:run.append(e)
        else:
            if run:runs.append(run)
            run=[]
    if run:runs.append(run)
    decode=[]
    for run in runs:
        if len(run)<2:continue
        a,b=run[0]['started_s'],run[-1]['finished_s'];whole,target=energy(a,b)
        decode.append(dict(start_s=a,end_s=b,steps=len(run),finish_spacing_values_s=[y['finished_s']-x['finished_s'] for x,y in zip(run,run[1:])],
                           target_gpu_mean_power_w=target/(b-a),all8_energy_j=whole,clock=phase_clock(clocks,gpu,2100,a,b)))
    require(decode,'actual full-batch decode is absent')
    idle=raw['idle_residency'];whole,target=energy(idle['start_s'],idle['end_s'])
    require(idle['end_s']-idle['start_s']>=2 and not idle['native']['active'] and not idle['native']['waiting'],'actual idle residency interval absent')
    original.native_saved(idle['native'],p['instance_id'],2048)
    return dict(passed=True,point=p,native=native,late_context=late,prefill_phases=prefills,full_decode_runs=decode,
        idle_residency=dict(start_s=idle['start_s'],end_s=idle['end_s'],all8_energy_j=whole,target_gpu_power_w=target/(idle['end_s']-idle['start_s']),
                            commanded_frequency_mhz=2100,lower_idle_Pstate_is_legal=True),
        terminal_empty_event=terminal,same_shape_interference_only=True,new_cross_context_interference_points=0,
        empirical_not_future_guarantee=True,output_sha256=native['output_sha256'])
