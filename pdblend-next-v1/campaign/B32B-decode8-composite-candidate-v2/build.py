"""Add only successfully measured long4096 decode phases to the same uniform profile."""
import hashlib,importlib.util,json,statistics,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent/'B32B-batch-coverage-long4096-v1'
PRIOR=ROOT.parent/'B32B-decode8-composite-candidate-v1/profiles.json'
HOST=ROOT.parents[1]/'releases/io-v1.2.1-fixed-window-decode8-v1-runtime'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def main():
 assert not (ROOT/'profiles.json').exists()
 status=read(BASE/'status.json');assert status['passed'] and status['measurement_valid'] and status['cleanup_complete'] and status['clock_release_complete']
 m=importlib.util.spec_from_file_location('integral',ROOT.parent/'B32B-budget-paired-longbench-v1/audit.py');integ=importlib.util.module_from_spec(m);m.loader.exec_module(integ)
 original=read(PRIOR);profile=read(PRIOR);proofs=[]
 for frequency in (1500,2520):
  observations=[];rawrefs={};durations=[]
  for point in sorted((BASE/'results').glob('point-*')):
   raw=read(point/'raw.json')
   if raw['spec']['clock_command_mhz']!=frequency:continue
   assert raw['spec']['batch_size']==8 and raw['spec']['input_lengths']==[4096]*8 and raw['spec']['output_lengths']==[256]*8
   assert not raw.get('error') and raw['identity_before']==raw['identity_after']
   ids={r['request_id'] for r in raw['requests']};assert len(ids)==8
   events=[json.loads(x) for x in (point/'events.jsonl').read_text().splitlines()]
   steps=[e for e in events if e['prefill']==0 and e['decode']==8 and set(e['request_ids'])==ids]
   assert len(steps)>=64 and all(e['tokens']==8 and e['mode']=='continuous' for e in steps)
   start,end=steps[0]['started_s'],steps[-1]['finished_s']
   window=[e for e in events if start<=e['started_s'] and e['finished_s']<=end]
   assert window==steps,'interleaved non-pure work cannot enter decode phase energy'
   times=[e['finished_s']-e['started_s'] for e in steps];durations+=times
   energy=integ.integrate(integ.power_rows(point/'power.csv'),start,end)
   clocks=[v for t,v in read(point/'clocks.json') if start<=t<=end]
   assert clocks and all(abs(v[g]-frequency)<=15 for v in clocks for g in (0,1)),'actual decode clocks outside frozen band'
   observations.append(dict(point=str(point),decode_steps=len(steps),start_s=start,end_s=end,
    mean_owner_step_s=statistics.mean(times),max_owner_step_s=max(times),tp2_power_w=sum(energy[:2])/(end-start),per_gpu_energy_j=energy))
   for n in ('raw.json','events.jsonl','power.csv','clocks.json','profile.json'):rawrefs[str(point/n)]=sha(point/n)
  assert len(observations)==3
  iteration=max(x['mean_owner_step_s'] for x in observations);error=max(.05,max(durations)/iteration-1)
  power=max(x['tp2_power_w'] for x in observations);residency=max(p['residency_w'] for p in profile['points'] if p['tp']==2 and p['frequency_mhz']==frequency)
  assert power>=residency
  proof=dict(role='decode',tp=2,batch=8,input_tokens=4096,context_tokens=4352,frequency_mhz=frequency,
   observations=observations,raw_sha256=rawrefs,iteration_s=iteration,error_fraction=error,time_bound_s=iteration*(1+error),
   semantics='Pure decode common to all eight actual streams after long prefills. Actual owner CPU wall and TP2 instantaneous energy. No fabricated mixed-prefill/background interference observation; no context above4352.')
  file=ROOT/f'decode4096-{frequency}.json';write(file,proof);proofs.append(proof)
  profile['points'].append(dict(role='decode',tp=2,frequency_mhz=frequency,input_tokens=4096,context_tokens=4352,batch=8,
   prefill_s=0.,iteration_s=iteration,power_w=power,residency_w=residency,error_fraction=error,samples=3,
   source_sha256=sha(file),interference_s=0.,prefill_power_w=0.,decode_power_w=power,energy_error_fraction=.05,
   prefill_power_upper_w=0.,prefill_duration_upper_s=0.))
 assert profile['points'][:-2]==original['points']
 profile['long_decode_extension']=dict(prior_profile=str(PRIOR),prior_profile_sha256=sha(PRIOR),observed_operation=str(BASE),
  old_points_unchanged=True,added_roles=['decode'],added_points=2,batch=8,input_upper=4096,context_upper=4352,
  unmeasured_larger_context_rejected=True,engine_unchanged=True,formal_eligible=False,
  mixed_interference_certified=False,original_temporal_gate_passed=False)
 write(ROOT/'profiles.json',profile)
 write(ROOT/'manifest.json',dict(created_s=time.time(),host_release=str(HOST),host_manifest_sha256=sha(HOST/'manifest.json'),
  original_long_observation_manifest_sha256=sha(BASE/'manifest.json'),original_profile_sha256=sha(PRIOR),
  files={n:sha(ROOT/n) for n in ('build.py','profiles.json','decode4096-1500.json','decode4096-2520.json')},
  scope='Uniform two-TP2 developer composite profile; only two measured long-context decode points added',performance_validation='pending'))
 print(json.dumps(dict(profile_sha256=sha(ROOT/'profiles.json'),bounds_s=[p['time_bound_s'] for p in proofs])))
if __name__=='__main__':main()
