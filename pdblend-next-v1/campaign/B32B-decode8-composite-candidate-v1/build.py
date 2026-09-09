"""Derive only observed decode phases; retain original mixed/prefill measurements."""
import hashlib,importlib.util,json,statistics,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent/'B32B-batch-coverage-observation-v2-r4'
PARENT=ROOT.parents[1]/'releases/io-v1.2.1-fixed-window-v2-runtime'
RELEASE=ROOT.parents[1]/'releases/io-v1.2.1-fixed-window-decode8-v1-runtime'
OLD=ROOT.parent/'baseline-raw-mirror-v1/B/root/workspace/pdblend/new-results/campaigns/node-b-v9/quick32-v1/profiles.provisional-tp2.json'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def main():
 assert not RELEASE.exists() and not (ROOT/'profiles.json').exists()
 status=read(BASE/'status.json');assert status['passed'] and status['measurement_valid']
 module=importlib.util.spec_from_file_location('integral',ROOT.parent/'B32B-budget-paired-longbench-v1/audit.py')
 integral=importlib.util.module_from_spec(module);module.loader.exec_module(integral)
 original=read(OLD);profile=read(OLD);proof=[]
 for freq in (1500,2520):
  observations=[];files={};all_steps=[]
  for point in sorted((BASE/'results').glob('point-*')):
   raw=read(point/'raw.json')
   if raw['spec']['batch_size']!=8 or raw['spec']['clock_command_mhz']!=freq:continue
   assert not raw.get('error') and raw['identity_before']==raw['identity_after']
   events=[json.loads(x) for x in (point/'events.jsonl').read_text().splitlines()]
   ids={r['request_id'] for r in raw['requests']}
   steps=[e for e in events if e['prefill']==0 and e['decode']==8 and set(e['request_ids'])==ids]
   assert len(steps)==255 and all(e['tokens']==8 and e['mode']=='continuous' for e in steps)
   start,end=steps[0]['started_s'],steps[-1]['finished_s']
   durations=[e['finished_s']-e['started_s'] for e in steps];all_steps.extend(durations)
   energy=integral.integrate(integral.power_rows(point/'power.csv'),start,end)
   clocks=[values for t,values in read(point/'clocks.json') if start<=t<=end]
   assert clocks and all(abs(values[g]-freq)<=15 for values in clocks for g in (0,1))
   observations.append(dict(point=str(point),decode_start_s=start,decode_end_s=end,
    decode_steps=len(steps),decode_power_tp2_w=sum(energy[:2])/(end-start),
    mean_owner_step_s=statistics.mean(durations),max_owner_step_s=max(durations),energy_per_gpu_j=energy))
   for n in ('raw.json','events.jsonl','power.csv','clocks.json','profile.json'):files[str(point/n)]=sha(point/n)
  assert len(observations)==3
  iteration=max(x['mean_owner_step_s'] for x in observations)
  uncertainty=max(.05,max(all_steps)/iteration-1)
  power=max(x['decode_power_tp2_w'] for x in observations)
  resident=max(p['residency_w'] for p in original['points'] if p['tp']==2 and p['frequency_mhz']==freq)
  assert power>=resident
  phase=dict(frequency_mhz=freq,role='decode',tp=2,batch=8,input_tokens=512,context_tokens=768,
    observations=observations,source_sha256=files,iteration_s=iteration,error_fraction=uncertainty,
    time_bound_s=iteration*(1+uncertainty),decode_power_w=power,
    semantics='Actual decode-only owner CPU wall and TP2 instantaneous power, after one natural eight-request prefill; no mixed interference observation; context spans513..767 and bucket upper768.')
  phase_path=ROOT/f'decode-{freq}.json';write(phase_path,phase);proof.append(phase)
  profile['points'].append(dict(role='decode',tp=2,frequency_mhz=freq,input_tokens=512,context_tokens=768,batch=8,
   prefill_s=0.,iteration_s=iteration,power_w=power,residency_w=resident,error_fraction=uncertainty,
   samples=3,source_sha256=sha(phase_path),interference_s=0.,prefill_power_w=0.,decode_power_w=power,
   energy_error_fraction=.05,prefill_power_upper_w=0.,prefill_duration_upper_s=0.))
 profile.update(mixed_decode_phase_fallback=True,formal_eligible=False,status='developer_composite_phase_candidate',
  composite_provenance=dict(original_path=str(OLD),original_sha256=sha(OLD),original_points_unchanged=True,
   added_roles=['decode'],added_points=2,host_opt_in='lookup_execution_phase',
   prefill='Original measured mixed batch1 lookup, unchanged',
   interference='Original measured lookup when covered; otherwise existing conservative max(prefill,dp.interference) fallback, not newly certified.',
   limitation='No measured preexisting-decode interference at batch8. No extrapolation above batch8/input512/context768. Original temporal failure retained. Batch4/8 outputs differ at position11; within each batch all requests/frequencies/repeats exact.'))
 write(ROOT/'profiles.json',profile)
 parent=read(PARENT/'manifest.json');RELEASE.mkdir()
 for name,h in parent['files'].items():
  assert sha(PARENT/name)==h
  target=RELEASE/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((PARENT/name).read_bytes())
 p=RELEASE/'src/ecopadg/serving/profiles.py';s=p.read_text()
 s=s.replace('parked_residency_w_by_tp=None,interference_points=()):','parked_residency_w_by_tp=None,interference_points=(),mixed_decode_phase_fallback=False):')
 s=s.replace('self.points = tuple(points)','self.mixed_decode_phase_fallback = mixed_decode_phase_fallback is True\n        self.points = tuple(points)')
 s=s.replace("interference_points=data.get('interference_points',()))","interference_points=data.get('interference_points',()),\n                   mixed_decode_phase_fallback=data.get('mixed_decode_phase_fallback',False))")
 at='    def frequencies(self, role, tp):'
 method='''    def lookup_execution_phase(self, role, tp, frequency, input_tokens, context, batch):
        """Explicit phase composition; pure decode evidence is never relabeled mixed.

        Existing covered mixed points retain precedence. The opt-in candidate
        only fills a missing decode estimate; callers must separately require
        measured mixed prefill and preserve interference/deadline checks.
        """
        point = self.lookup(role,tp,frequency,input_tokens,context,batch)
        if point is None and role == 'mixed' and self.mixed_decode_phase_fallback:
            return self.lookup('decode',tp,frequency,input_tokens,context,batch)
        return point

'''
 assert at in s;s=s.replace(at,method+at);p.write_text(s)
 p=RELEASE/'src/ecopadg/serving/planner.py';s=p.read_text();old='return self.profiles.lookup(instance.role, instance.tp, freq,';assert s.count(old)==1;p.write_text(s.replace(old,'return self.profiles.lookup_execution_phase(instance.role, instance.tp, freq,'))
 p=RELEASE/'src/ecopadg/serving/tails.py';s=p.read_text();old='self.estimator.profiles.lookup(*key)';assert s.count(old)==1;p.write_text(s.replace(old,'self.estimator.profiles.lookup_execution_phase(*key)'))
 changed=[n for n,h in parent['files'].items() if sha(RELEASE/n)!=h]
 assert changed==['src/ecopadg/serving/planner.py','src/ecopadg/serving/profiles.py','src/ecopadg/serving/tails.py']
 write(RELEASE/'manifest.json',dict(created_s=time.time(),release=str(RELEASE),source_release=str(PARENT),source_manifest_sha256=sha(PARENT/'manifest.json'),changed_files=changed,scope='Opt-in developer decode-phase composition; pure decode role preserved, original mixed prefill and deadline/interference checks remain. No engine change.',files={n:sha(RELEASE/n) for n in parent['files']}))
 write(ROOT/'manifest.json',dict(created_s=time.time(),release=str(RELEASE),release_manifest_sha256=sha(RELEASE/'manifest.json'),files={n:sha(ROOT/n) for n in ['build.py','profiles.json','decode-1500.json','decode-2520.json']},runtime_validation='pending',original_temporal_gate='failed'))
 print(json.dumps(dict(release=str(RELEASE),manifest_sha256=sha(RELEASE/'manifest.json'),profiles_sha256=sha(ROOT/'profiles.json'),bounds=[p['time_bound_s'] for p in proof])))
if __name__=='__main__':main()
