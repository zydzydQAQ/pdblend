"""Recompute target-local 2400 MHz evidence; never infer it from 2520 MHz.

verify(registration_ref, profile_ref=None) returns the complete derived_profile,
frequency_costs, measured_domains and sources. Optional profile_ref must match
the full derived object. This module makes no hardware or current-PID queries.
"""
import argparse,copy,hashlib,importlib.util,json,math,os,statistics,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent.parent
DRIVERS={str(HERE.parent/'distributed14b-frequency-profile-v1/manifest.json'):'f1296cbecd65a5636bf1ea42b5c0ad433f50c29ef6633d71213f2746bb120c8a',
         str(HERE.parent/'distributed14b-frequency-profile-v2/manifest.json'):'6df9fbe80b903ea5a9f167a3343f393420834f5bf5ac2d994a43326e4b948436'}
SAMPLER_AUDIT=ROOT/'isolated_measurement_audit_v1.py'
SAMPLER_AUDIT_SHA='7b7ae5d55b0dfed733ab514a80e7177fd4ce7cc9d12e89c67a7cf740b147b9d8'

def need(value,message):
    if not value:raise ValueError(message)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def ref(path):return dict(path=str(Path(path).resolve()),sha256=sha(path))
def fixed(reference):
    need(sha(reference['path'])==reference['sha256'],'changed actual evidence '+reference['path']);return read(reference['path'])
def load(path,name):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def full_files(files):
    need(files,'empty frozen evidence closure')
    for path,digest in files.items():need(sha(path)==digest,'changed frozen input '+path)
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')

def identity(before,after,binding):
    first={x['provenance']['instance_id']:x for x in before};last={x['provenance']['instance_id']:x for x in after}
    need(len(first)==len(before)==len(last)==len(after)==2 and set(first)==set(last)=={i['id'] for i in binding['instances']},'both actual TP1 process identities required')
    for instance in binding['instances']:
        a,b=first[instance['id']],last[instance['id']]
        need(a['provenance']==b['provenance'] and all(a['provenance'].get(k)==v for k,v in instance['provenance'].items()),'measured engine provenance changed')
        need(a['container']['Id']==b['container']['Id']==instance['container']['id']
             and a['container']['Image']==b['container']['Image']==instance['container']['image']
             and a['container']['State']['StartedAt']==b['container']['State']['StartedAt']==instance['container']['StartedAt']
             and type(a['container']['State']['Pid']) is int and a['container']['State']['Pid']>0
             and a['container']['State']['Pid']==b['container']['State']['Pid'], 'actual measured container/process changed')
    return dict(passed=True,instances=sorted(first),no_live_identity_queries=True)

def aggregate(profile,rows,transitions,*,source_reference,hostname,node,measurement_reference):
    """Only actual repeated phase envelopes become new 2400 ProfilePoints."""
    from ecopadg.serving.profiles import ProfilePoint,ProfileStore
    result=copy.deepcopy(profile)
    removed=[p for p in profile['points'] if p['frequency_mhz']>2400]
    result['points']=[p for p in result['points'] if p['frequency_mhz']<=2100]
    result['interference_points']=[p for p in result.get('interference_points',[]) if p['frequency_mhz']<=2100]
    groups={}
    for row in rows:
        p=row['point'];key=p['input_tokens'],p['batch'],p['context_tokens'];groups.setdefault(key,[]).append(row)
    need(len(groups)==14 and len(rows)==28,'all fourteen actual shapes need two complete observations')
    resident=max(row['idle_residency']['target_gpu_power_w'] for row in rows)
    need(math.isfinite(resident) and resident>0,'actual residency power must be finite and positive')
    expected_geometry={(p['input_tokens'],p['batch']):max(q['context_tokens'] for q in profile['points'] if q['role']=='mixed' and q['tp']==1 and q['input_tokens']==p['input_tokens'] and q['batch']==p['batch']) for p in profile['points'] if p['role']=='mixed' and p['tp']==1}
    need(set(groups)=={(n,b,c) for (n,b),c in expected_geometry.items()},'registered shape set must exactly match original declared lookup domain')
    domains=[];added=[]
    for (length,batch,context),members in sorted(groups.items()):
        need(len(members)==2 and {r['point']['gpu'] for r in members}=={6,7}
             and {r['point']['repeat'] for r in members}=={1,2}
             and len({r['output_sha256'] for r in members})==1,'each geometry requires both actual cards and identical prescribed outputs')
        need(all(r['passed'] is True and r['same_shape_interference_only'] is True for r in members),'unqualified actual shape')
        for r in members:
            need(min(p['attention_after_min'] for p in r['late_context']['per_request'].values())>=context,
                 'late full-batch logical context does not support registered edge')
        need(all(r['full_decode_runs'] and all(run['finish_spacing_values_s'] for run in r['full_decode_runs']) for r in members),'each actual observation must contain complete decode spacings')
        gaps=[g for r in members for run in r['full_decode_runs'] for g in run['finish_spacing_values_s']]
        need(gaps and all(math.isfinite(g) and g>0 for g in gaps),'actual complete decode spacings missing')
        iteration=statistics.mean(gaps);upper=max(gaps);error=max(0.,upper/iteration-1.)
        pf=max(p['duration_s'] for r in members for p in r['prefill_phases'])
        pfp=max(p['target_gpu_power_w'] for r in members for p in r['prefill_phases'])
        dcp=max(p['target_gpu_mean_power_w'] for r in members for p in r['full_decode_runs'])
        need(all(math.isfinite(x) and x>0 for x in (pf,pfp,dcp)), 'actual phase durations and power must be finite and positive')
        need(all(r['point']['frequency_mhz']==2400 and r['point']['output_tokens']==1024 for r in members), 'actual frequency and full microprofile outputs changed')
        provenance=dict(input_tokens=length,batch=batch,context_tokens=context,frequency_mhz=2400,
            measured_points=[r['measurement_references'] for r in members],samples=2,actual_gpus=[6,7],
            iteration_mean_s=iteration,iteration_sample_max_s=upper,prefill_sample_max_s=pf,
            prefill_power_sample_max_w=pfp,decode_power_sample_max_w=dcp,residency_sample_max_w=resident,
            empirical_not_guaranteed=True,same_shape_mixed_only=True,new_cross_context_interference_points=0)
        digest=hashlib.sha256(json.dumps(provenance,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        for role in ('mixed','decode'):
            point=ProfilePoint(role=role,tp=1,frequency_mhz=2400,input_tokens=length,context_tokens=context,batch=batch,
                prefill_s=pf if role=='mixed' else 0.,iteration_s=iteration,power_w=max(resident,pfp,dcp),residency_w=resident,
                error_fraction=error,samples=2,source_sha256=digest,interference_s=pf if role=='mixed' else 0.,
                prefill_power_w=max(resident,pfp) if role=='mixed' else 0.,decode_power_w=max(resident,dcp),energy_error_fraction=0.,
                prefill_power_upper_w=max(resident,pfp) if role=='mixed' else 0.,prefill_duration_upper_s=pf if role=='mixed' else 0.)
            added.append(point.__dict__)
        domains.append(dict(provenance,source_sha256=digest))
    result['points'].extend(added)
    frequency_costs=[]
    expected={(a,b) for low in (900,1500,2100) for a,b in ((low,2400),(2400,low))}
    pairs={}
    for row in transitions:pairs.setdefault((row['source_mhz'],row['target_mhz']),[]).append(row)
    need(set(pairs)==expected and len(transitions)==12,'six actual directed transitions on both cards required')
    for (source,target),members in sorted(pairs.items()):
        need(len(members)==2 and {r['gpu'] for r in members}=={6,7},'both actual cards required per frequency direction')
        need(all(math.isfinite(r['actual_duration_s']) and r['actual_duration_s']>0 and math.isfinite(r['all8_energy_j']) and r['all8_energy_j']>0 for r in members),'actual directed transition measurement invalid')
        digest=hashlib.sha256(json.dumps(members,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        frequency_costs.append(dict(tp=1,source_mhz=source,target_mhz=target,
            duration_upper_s=max(r['actual_duration_s'] for r in members),energy_upper_j=max(r['all8_energy_j'] for r in members),source_sha256=digest))
    result['status']='target-local development empirical 2400MHz profile; fresh serving qualification remains required'
    result['heldout_calibration_complete']=False;result['instant_heldout_calibration_complete']=False
    result['frequency2400_registration']=dict(schema='distributed14b-measured2400-profile-registration-v1',hostname=hostname,node=node,
        registration=source_reference,measurement=measurement_reference,original_profile_retained=True,
        original_profile_reference_only_below_or_equal_2100=True,original_2520_points_removed=len(removed),
        removed_frequency_domain=[2520],new_frequency_mhz=2400,added_points=len(added),measured_domains=domains,
        samples_per_shape=2,each_actual_card_once=True,frequency_costs=frequency_costs,
        new_prefill_role_points=0,new_cross_context_interference_points=0,
        lower_frequency_cross_host_qualification_separate=True,empirical_estimates_not_hard_guarantees=True,
        development_profile_and_independent_serving_validation_required=True)
    ProfileStore([ProfilePoint(**p) for p in result['points']],interference_points=result.get('interference_points',()))
    return result,domains,frequency_costs

def _verify(registration_reference,profile_reference=None):
    declaration=fixed(registration_reference)
    need(declaration['schema']=='distributed14b-actual2400-registration-input-v1' and declaration['approved'] is True,
         'explicit immutable target-local registration required')
    need(DRIVERS.get(declaration['driver']['path'])==declaration['driver']['sha256'],'actual measurement source changed')
    driver=Path(declaration['driver']['path']).parent
    package=fixed(declaration['driver']);full_files(package['files'])
    own=fixed(declaration['registration_package']);full_files(own['files'])
    need(own['files'].get(str(Path(__file__).resolve()))==sha(__file__),'registration verifier not frozen')
    measurement=fixed(declaration['measurement']);full_files(measurement['files'])
    out=Path(declaration['measurement']['path']).parent;status=read(out/'status.json');spec=read(out/'spec.json')
    need(measurement['spec']==ref(out/'spec.json') and measurement['source']==declaration['driver']
         and measurement['passed'] and status['passed'] and status['complete'] and status['measurement_valid']
         and not status['errors'] and status.get('clock_restore_complete') is True
         and all(x['complete'] and not x.get('errors') for x in status['native_cleanup']), 'actual complete microprofile/clock/native evidence required')
    sys.path.insert(0,str(driver));v=load(driver/'validate.py','validate');binding=v.source_check(spec)
    need(declaration['node']==spec['node'] and declaration['hostname']==spec['hostname']
         and declaration['binding']==spec['binding'] and declaration['profile_reference']==spec['profile_reference'], 'registration refers to another host/profile/binding')
    need(spec['hostname']=={'B':'iZwz9i5bte3xkpmcoes3t2Z','C':'iZwz9gfq11hx1sbob59yrgZ'}[spec['node']], 'actual physical target hostname differs')
    host=Path(binding['host_release']);sys.path[:0]=[str(host/'src'),str(host),'/root/workspace/pdblend/.runtime-deps']
    from ecopadg.serving.measurement import power_evidence
    from ecopadg.metrics import clip_power_window
    from ecopadg.measure.power import trapezoid_energy
    host_manifest=fixed(spec['host_manifest'])
    for module_name in ('ecopadg.serving.measurement','ecopadg.metrics','ecopadg.measure.power','ecopadg.serving.profiles'):
        module=__import__(module_name,fromlist=['__name__']);source=Path(module.__file__).resolve()
        relative=str(source.relative_to(host.resolve()))
        need(host_manifest['files'].get(relative)==sha(source),'actual profile/measurement validator source differs')
    actual_identity=identity(read(out/'identity.before.json'),read(out/'identity.after.json'),binding)
    need(sha(SAMPLER_AUDIT)==SAMPLER_AUDIT_SHA,'saved observer verifier changed')
    sampler=load(SAMPLER_AUDIT,'actual2400_saved_measurement');audited=sampler.audit_samplers(status['isolated_samplers'],spec['host_manifest'],artifacts=measurement['files'])
    match=sampler.match_power_directory(out/'power',audited,artifacts=measurement['files'])
    raw_power=audited['raw_values'][match['isolated_directory']];power=raw_power['samples'];clocks=raw_power['frequency']
    actual_power=power_evidence(power,raw_power['power_source'],raw_power['metadata'])
    need(actual_power['power_source_verified'] and actual_power==status['power_evidence'],'original power source/age/gap evidence differs')
    actual_energy=trapezoid_energy(clip_power_window(power,status['operation_start_s'],status['operation_end_s'],pad_s=0))
    need(abs(actual_energy-status['all8_operation_energy_j'])<=max(1e-6,abs(actual_energy)*1e-12),'actual all8 operation integral differs')
    before,after=read(out/'source-order.before.json'),read(out/'source-order.after.json')
    order=load(driver/'source_order.py','actual2400_saved_source_order')
    need([p['point_id'] for p in status['points']]==[p['point_id'] for p in spec['points']],'all declared microbatches must remain in original order')
    rows=[];ids=set();previous_end=None
    for declared,saved in zip(spec['points'],status['points']):
        raw=fixed(saved['raw']);path=Path(saved['raw']['path']).parent
        need(raw['point']==declared and (previous_end is None or raw['started_s']>=previous_end),'changed or overlapping actual microbatch')
        previous_end=raw['finished_s'];owned={r['request_id'] for r in raw['requests']}
        need(not ids&owned,'actual requests were reused across microbatches');ids.update(owned)
        source=order.validate_pair(before,after,declared['instance_id'],measurement_start_s=raw['started_s'],measurement_end_s=raw['finished_s'],contract_path=driver/'source-order-contract.json')
        events=[json.loads(line) for line in (path/'events.jsonl').read_text().splitlines() if line]
        result=v.validate_profile_point(raw,events,power,clocks,source_order_verified=source['verified'])
        need(result==fixed(saved['validation']),'saved empirical point differs from independent complete reconstruction')
        result['measurement_references']=dict(raw=saved['raw'],validation=saved['validation'],events=ref(path/'events.jsonl'),source_order=source)
        rows.append(result)
    transition_rows=[];switcher=load(driver/'frequency_switch.py','actual2400_saved_switch')
    need([s['gpu'] for s in status['switches']]==[6,7],'both actual transition runs required')
    for saved in status['switches']:
        raw=fixed(saved['raw'])
        order.validate_pair(before,after,raw['instance_id'],measurement_start_s=raw['started_s'],measurement_end_s=raw['finished_s'],contract_path=driver/'source-order-contract.json')
        result=switcher.derive(raw,power,clocks);need(result==fixed(saved['validation']),'actual transition reconstruction differs')
        transition_rows.extend(result['switches'])
    profile,domains,costs=aggregate(fixed(declaration['profile_reference']),rows,transition_rows,
        source_reference=registration_reference,hostname=spec['hostname'],node=spec['node'],measurement_reference=declaration['measurement'])
    if profile_reference is not None:need(fixed(profile_reference)==profile,'published profile differs from complete raw reconstruction')
    sources={**measurement['files'],**spec['files'],**package['files'],**own['files']}
    for reference in (registration_reference,declaration['registration_package'],declaration['measurement'],declaration['driver'],declaration['profile_reference']):sources[reference['path']]=reference['sha256']
    full_files(sources)
    return dict(passed=True,derived_profile=profile,frequency_costs=costs,measured_domains=domains,sources=sources,
        actual_identity=actual_identity,whole_operation_energy_j=actual_energy,measurement_adapter=spec['measurement_adapter'],
        empirical_estimates_not_guaranteed=True,all_28_actual_points_used=True,all_12_actual_switches_used=True,
        source_order_and_logical_context_recomputed=True,physical_KV_context_claimed=False)

def verify(registration_reference,profile_reference=None):
    """Replay in a fresh CPU process, independent of ambient host modules."""
    payload=dict(registration_reference=registration_reference,profile_reference=profile_reference)
    environment=dict(os.environ);environment.pop('PYTHONPATH',None)
    result=subprocess.run([sys.executable,'-I',str(Path(__file__).resolve()),'--verify-input-json'],
        input=json.dumps(payload),text=True,capture_output=True,timeout=300,env=environment)
    need(result.returncode==0,'independent saved-evidence verifier rejected: '+result.stderr[-12000:])
    return json.loads(result.stdout)

def prepare(measurement_path,out):
    measurement_path=Path(measurement_path).resolve();out=Path(out).resolve();need(not out.exists(),'new registration declaration required')
    measurement=read(measurement_path);spec=read(measurement_path.parent/'spec.json')
    declaration=dict(schema='distributed14b-actual2400-registration-input-v1',approved=True,node=spec['node'],hostname=spec['hostname'],
        driver=measurement['source'],registration_package=ref(HERE/'manifest.json'),
        measurement=ref(measurement_path),binding=spec['binding'],profile_reference=spec['profile_reference'])
    write(out,declaration);return ref(out)

if __name__=='__main__':
    if sys.argv[1:]==['--verify-input-json']:
        payload=json.load(sys.stdin);print(json.dumps(_verify(**payload),allow_nan=False));raise SystemExit(0)
    parser=argparse.ArgumentParser();parser.add_argument('--measurement',type=Path);parser.add_argument('--declaration',type=Path,required=True);parser.add_argument('--out',type=Path)
    args=parser.parse_args()
    reference=prepare(args.measurement,args.declaration) if args.measurement else ref(args.declaration)
    result=verify(reference)
    if args.out:
        need(not args.out.exists(),'new profile publication directory required');args.out.mkdir(parents=True)
        write(args.out/'profiles.development.json',result['derived_profile']);write(args.out/'frequency-costs.json',result['frequency_costs'])
        proof={k:v for k,v in result.items() if k!='derived_profile'};proof['profile']=ref(args.out/'profiles.development.json');proof['registration']=reference
        write(args.out/'qualification.json',proof)
    print(json.dumps(dict(passed=True,registered_points=28,frequency_costs=len(result['frequency_costs']),profile_published=bool(args.out))))
