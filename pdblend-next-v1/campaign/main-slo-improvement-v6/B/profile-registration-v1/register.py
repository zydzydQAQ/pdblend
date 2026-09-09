"""Append-only B decode-only empirical qualification from complete measured raws."""
import argparse,copy,hashlib,importlib.util,json,statistics,sys
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[3]
HOST=ROOT/'main-slo-improvement-v1/hosts/32b-fixed-v6'
sys.path[:0]=[str(HOST/'src'),str(HOST),'/root/workspace/pdblend/.runtime-deps']
from ecopadg.serving.profiles import ProfileStore,ProfilePoint

def need(ok,why):
 if not ok:raise ValueError(why)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def ref(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def load(name,p):
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def register(short,long,base,out):
 short,long,base,out=map(lambda p:Path(p).resolve(),(short,long,base,out));need(not out.exists(),'new output required')
 sources={};used=lambda p:(sources.setdefault(str(Path(p).resolve()),sha(p)),Path(p))[1]
 used(__file__);used(base);used(HOST/'manifest.json')
 original=load('b_original_measured_evidence',used(ROOT/'budget-profiling-v2-candidate/evidence.py'))
 reports=[];allids=set();groups=[]
 for micro,batch,inputlen,outputlen,edge in ((short,16,512,256,768),(long,6,7168,512,7680)):
  manifest=read(used(micro/'manifest.json'))
  for n,h in manifest['files'].items():need(sha(used(micro/n))==h,'micro source changed '+n)
  for n,h in manifest['frozen_references'].items():need(sha(used(n))==h,'frozen predecessor changed '+n)
  state=read(used(micro/'status.json'))
  need(all(state.get(k) is True for k in ('complete','passed','measurement_valid','evidence_complete','baseline_preservation_verified','clock_release_complete')),'complete actual source/native/full8 measurement cleanup required')
  report=read(used(micro/'late-context-evidence.json'))
  need(sha(micro/'late-context-evidence.json')==state['late_context_evidence_sha256'] and report['complete'] and len(report['points'])==6,'actual complete six-point receipt differs')
  for p,h in report['source_receipts'].items():need(sha(used(p))==h,'live-source receipt changed')
  declaration=read(used(micro/'declaration.json'));campaign=read(used(micro/'results/campaign.json'))
  need(campaign['complete'] and len(campaign['points'])==6 and campaign['clock_locks_reset'],'original helper not restored/complete')
  context=load('context_export',micro/'context_export.py');late=load('b_late_context_'+str(batch),micro/'late_context.py')
  capacity=load('b_capacity_'+str(batch),micro/'capacity.py')
  observation=load('b_observation_'+str(batch),micro/'observation.frozen.py')
  before,after=[read(used(p)) for p in sorted((micro/'source-order').glob('*.json'))]
  expected=read(used(micro/'expected-identity.json'))
  rederived=[]
  for index,item in enumerate(report['points']):
   path=micro/'results'/item['point_id'];saved=read(used(path/'late-context.json'))
   need(sha(path/'late-context.json')==item['late_context_sha256'] and saved['valid'],'saved raw qualification differs')
   for p,h in saved['source_sha256'].items():need(sha(used(p))==h,'measured raw changed')
   spec=campaign['points'][index]['spec'];raw=read(path/'raw.json')
   need(spec['batch_size']==batch and spec['input_lengths']==[inputlen]*batch and spec['output_lengths']==[outputlen]*batch
    and spec['clock_command_mhz']==(1500 if index<3 else 2520) and spec['budget_tokens']==8192 and spec['max_num_seqs']==32
    and spec['target_gpus']==[0,1] and spec['tp']==2 and spec['seed']==0 and spec['temperature']==0,'original declaration shape/frequency differs')
   actual=late.audit_point(path,before,after,expected=expected,original_evidence=original,
    tp2_validator=observation.validate_tp2_observation,contract_path=micro/'source-order-contract.json',expected_spec=spec)
   need(actual['valid'] and actual==saved,'recomputed raw/native/power/clocks/source qualification differs '+str(actual['errors'])+' fields '+str([k for k in actual if actual[k]!=saved.get(k)]))
   ids={r['request_id'] for r in raw['requests']};need(len(ids)==batch and not ids&allids,'actual independent request IDs reused');allids|=ids
   contexts=actual['late_context']['per_request'].values()
   need(all(r['complete_work_attention_upper']==edge-1 and r['declared_bucket_edge']==edge for r in contexts),'actual complete context does not reach declared discrete bucket edge')
   rederived.append(dict(point_id=item['point_id'],raw=ref(path/'raw.json'),qualification=ref(path/'late-context.json'),
    spec=spec,full_decode_windows=actual['full_decode_windows'],late_window=actual['late_context']['late_window']))
  warm=late.warmup_boundaries(micro,state['measurement_start_s'],state['measurement_end_s'])
  need(warm['valid'] and warm==state['warmup_energy_boundaries'],'original full8 outer warmups not qualified')
  for freq in (1500,2520):
   rows=[r for r in rederived if r['spec']['clock_command_mhz']==freq];need(len(rows)==3,'three independent executions required')
   gaps=[gap for r in rows for w in r['full_decode_windows'] for gap in w['finish_spacing_values_s']]
   center=statistics.mean(gaps);upper=max(gaps);power=max(w['target_gpus_mean_power_w'] for r in rows for w in r['full_decode_windows'])
   groups.append(dict(batch=batch,input_tokens=inputlen,context_tokens=edge,frequency_mhz=freq,rows=rows,
    iteration_s=center,iteration_upper_s=upper,power_upper_w=power,samples=3,actual_max_attention=edge-1,
    all_full_decode_gaps_retained=True,empty_steps_preserved_in_time_and_energy=True,
    source_sha256=hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(',',':')).encode()).hexdigest()))
  reports.append(dict(root=str(micro),status=ref(micro/'status.json'),late_report=ref(micro/'late-context-evidence.json'),
    independent_executions=6,raw_qualification_recomputed=True,unchanged_strict_temporal_failure=state.get('temporal_correctness'),warmup=warm))
 profile=copy.deepcopy(read(base));added=[]
 def key(p):return tuple(p[k] for k in ('role','tp','frequency_mhz','input_tokens','context_tokens','batch'))
 for g in groups:
  residents=[p['residency_w'] for p in profile['points'] if p['role']=='decode' and p['tp']==2 and p['frequency_mhz']==g['frequency_mhz']]
  resident=max(residents);power=max(resident,g['power_upper_w'])
  point=ProfilePoint(role='decode',tp=2,frequency_mhz=g['frequency_mhz'],input_tokens=g['input_tokens'],context_tokens=g['context_tokens'],batch=g['batch'],prefill_s=0,
   iteration_s=g['iteration_s'],power_w=power,residency_w=resident,error_fraction=max(0,g['iteration_upper_s']/g['iteration_s']-1),
   samples=3,source_sha256=g['source_sha256'],decode_power_w=power).__dict__
  need(all(key(p)!=key(point) for p in profile['points']),'new point would overwrite original profile key');added.append(point)
 profile['points'].extend(added)
 flags=('heldout_calibration_complete','instant_heldout_calibration_complete','heldout_validation')
 profile['base_profile_qualification']={k:copy.deepcopy(profile.get(k)) for k in flags}
 profile.update(heldout_calibration_complete=False,instant_heldout_calibration_complete=False,formal_eligible=False,
  status='development empirical decode-only TP2 extension; fresh serving validation pending')
 profile['empirical_registration']=dict(measured_role='decode',actual_owner_role='mixed',new_mixed_points=0,new_interference_points=0,
  groups=[{k:v for k,v in g.items() if k!='rows'} for g in groups],hard_future_bounds_claimed=False,
  fixed_prompt_repetitions_not_arrival_seeds=True,context_edge_semantics='complete attention reaches bucket edge minus one; no work beyond edge claimed',
  measured_native_budget=dict(max_num_batched_tokens=8192,max_num_seqs=32),source_sha256=sources)
 ProfileStore([ProfilePoint(**p) for p in profile['points']],interference_points=profile['interference_points'],mixed_decode_phase_fallback=profile['mixed_decode_phase_fallback'])
 need(all(sha(p)==h for p,h in sources.items()),'sources changed during raw qualification')
 out.mkdir(parents=True)
 def save(n,v):(out/n).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
 save('profiles.development.json',profile)
 save('qualification.json',dict(passed=True,all_twelve_valid=True,profile=ref(out/'profiles.development.json'),reports=reports,groups=groups,
  source_sha256=sources,serving_validation_pending=True,no_gpu_executed_by_registration=True,
  limits='Empirical decode-only TP2 bounds from three fixed-prompt executions per point. Existing strict temporal failure unchanged; no new mixed/interference/KV-content guarantee.'))
 save('manifest.json',dict(passed=True,files={p.name:sha(p) for p in out.iterdir()},source_sha256=sources))
 return dict(passed=True,added_points=len(added),profile=ref(out/'profiles.development.json'),qualification=ref(out/'qualification.json'))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--short',required=True);p.add_argument('--long',required=True);p.add_argument('--base',required=True);p.add_argument('--out',required=True);a=p.parse_args();print(json.dumps(register(a.short,a.long,a.base,a.out),indent=2))
