"""Independent reconstruction of every legacy candidate and actual rank clock."""
import math
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'qualification-v2'))
import frequency as q
p=q.p

def verify(out,binding_ref,profile_ref):
 out=Path(out);binding=p.checked(binding_ref);profile=p.checked(profile_ref);s=p.read(out/'status.json')
 p.need(not s['passed'] and not s['complete'] and s['finished_s'] and not s['node_lease_held'] and not s['cleanup_errors'] and s.get('measurement_valid') is True,'clean terminal export-only failure required')
 p.need(s.get('error','').startswith('ContentTypeError(') and '/retain_weights' in s['error'] and 'status=404' in s['error'],'only proven final route-spelling failure may be composed')
 p.need(s.get('retained_weights') and 'response' not in s['retained_weights'] and s['retained_weights']['started_s']>=max(r['token_received_s'][-1] for v in s['idle_wakeup'] for r in v['requests']),'failure must occur strictly after every measured load/idle window')
 p.need(s['binding']==binding_ref and s['profile']==profile_ref and not s['cost_values_recalibrated'],'candidate input changed')
 q.paths(binding)
 from ecopadg.serving.measurement import power_evidence
 audit=p.load(q.ROOT.parent/'AC-baseline-binding-v2/gate_evidence.py','legacy_candidate_readonly')
 audit.identities(out,binding['instances']);physical=audit.power(out,s,power_evidence)
 import csv
 with (out/'power/clocks.csv').open() as stream:
  clocks=[(float(r['t_s']),[float(r[f'gpu{i}_sm_mhz']) for i in range(8)]) for r in csv.DictReader(stream)]
 expected={(i['id'],*shape) for i in binding['instances'] for shape in q.shapes(profile,i['tp'])}
 actual={(c['instance_id'],c['frequency'],c['input_length'],c['batch']) for c in s['cases']}
 p.need(expected==actual and len(s['cases'])==len(expected),'candidate set incomplete or duplicated')
 references={};all_requests={}
 for i in binding['instances']:
  requests=audit.lines(out/(i['id']+'.requests.jsonl'));events=audit.lines(out/(i['id']+'.stream.jsonl'))
  by={r['request_id']:r for r in requests};p.need(len(by)==len(requests),'duplicate request journal rows')
  all_requests[i['id']]=by
  for r in requests:
   ev=[e for e in events if e['request_id']==r['request_id']]
   p.need([dict(received_s=e['received_s'],event=e['event']) for e in ev]==r['stream_events'],'raw SSE differs from request')
   ids=[];times=[]
   for e in ev:
    tokens=e['event'].get('token_ids',[]);ids.extend(tokens);times.extend([e['received_s']]*len(tokens));p.need(e['event'].get('token_index')==len(ids),'raw token index gap')
   p.need(ids==r['output_token_ids'] and times==r['token_received_s'],'raw token/timing differs')
  own=[r for r in s['references'] if r['instance_id']==i['id']]
  p.need({r['input_length'] for r in own}=={v[1] for v in q.shapes(profile,i['tp'])},'missing length oracle')
  for r in own:
   row=r['request'];q.check(row,r['input_length'],row['output_token_ids']);p.need(by[row['request_id']]==row,'oracle lacks raw request');references[i['id'],r['input_length']]=row['output_token_ids']
  for c in [v for v in s['cases'] if v['instance_id']==i['id']]:
   p.need(p.read(q.case_path(out,c))==q.case_record(c),'per-case durable evidence differs from terminal snapshot/raw request journal')
   p.need(c['tp']==i['tp'] and c['gpus']==i['gpus'] and len(c['requests'])==c['batch'],'actual case layout changed');audit.ack(c['native_after'],i)
   loaded=[]
   for row in c['requests']:
    p.need(by[row['request_id']]==row,'candidate lacks exact raw request');q.check(row,c['input_length'],references[i['id'],c['input_length']]);loaded.append(q.clock_window(clocks,i['gpus'],c['frequency'],row['token_received_s'][0],row['token_received_s'][-1]))
   p.need(len(c['loaded_clocks'])==len(loaded),'stored clock windows missing')
   for online,final in zip(c['loaded_clocks'],loaded):
    p.need(all(online[k]==final[k] for k in ('gpus','target_mhz','start_s','end_s')) and final['samples']>=online['samples'],'final isolated flush changed declared clock window')
    for gpu in i['gpus']:
     old=online['actual_min_max_mhz'][str(gpu)];new=final['actual_min_max_mhz'][str(gpu)]
     p.need(new[0]<=old[0]<=old[1]<=new[1] and all(abs(v-c['frequency'])<=15 for v in old),'final clock extrema do not contain original observed samples')
 p.need([v['frequency'] for v in s['idle_wakeup']]==[900,1500,2100],'idle domain incomplete')
 for v in s['idle_wakeup']:
  polls=v['polls'];p.need(polls and polls[0]['at_s']-v['start_s']<1 and polls[-1]['at_s']-v['start_s']>=2.8,'declared natural idle interval incomplete')
  p.need(max(y['at_s']-x['at_s'] for x,y in zip(polls,polls[1:]))<=1,'native idle polling gap')
  for poll in polls:
   p.need(len(poll['native'])==len(binding['instances']),'native idle member missing')
   for i,raw in zip(binding['instances'],poll['native']):audit.ack(raw,i)
  p.need(len(v['requests'])==len(binding['instances']),'wake missing member')
  for i,row in zip(binding['instances'],v['requests']):
   p.need(all_requests[i['id']][row['request_id']]==row and row['dispatch_s']-v['start_s']>=3,'wake raw/idle timing differs');q.check(row,128,references[i['id'],128]);q.clock_window(clocks,i['gpus'],v['frequency'],row['token_received_s'][0],row['token_received_s'][-1])
 for i in binding['instances']:
  r=s['restoration'][i['id']];p.need(r['complete'],'native restoration incomplete');audit.ack(r['resumed']['after'],i)
 p.need(p.sha(s['topology']['path'])==s['topology']['sha256'] and all('GPU'+str(i) in Path(s['topology']['path']).read_text() for i in range(8)),'fresh physical topology differs')
 retained=None  # This separate proof grants no retained-weight capability.
 if retained:
  i=next(i for i in binding['instances'] if i['id']==retained['instance_id']);audit.ack(retained['native_before'],i)
  value=p.checked(retained['manifest']);response=retained['response'];payload=retained['payload']
  p.need(retained['http_status']==200 and response['manifest']==value and response['generation']==payload['expected_generation']==retained['native_before']['generation'] and response['accepting'] is False,'fresh weight export native identity differs')
  cache=Path(response['retained_weights']);p.need(cache==Path(p.read(i['engine_config'])['weight_cache_root'])/payload['transaction'] and retained['manifest']['path']==str(cache/'manifest.json'),'cache outside this new owner')
  p.need(value['complete'] and value['tp']==i['tp'] and len(value['ranks'])==i['tp'] and {r['rank'] for r in value['ranks']}==set(range(i['tp'])),'actual retained TP ranks differ')
  common=q.paths(binding)
  for rank in value['ranks']:
   path=cache/rank['file'];proof=retained['rank_files'][str(path)];p.need(path.parent==cache and proof['sha256']==rank['sha256'] and common.stat_identity(path)==proof['stat'],'retained rank immutable stat differs')
  audit.ack(retained['resumed']['after'],i)
 for file,digest in s['isolated_artifacts'].items():p.need(p.sha(file)==digest,'isolated measurement evidence changed')
 hooks=p.load(q.U/'meter-runtime/sampler_hooks.py','legacy_candidate_independent_sampler')
 frozen=hooks.completed_artifacts(hooks.directories(out/'isolated-samplers'),p.ref(Path(binding['host_release'])/'manifest.json'),p.ref(q.U/'isolated-power/manifest.json'))
 p.need(frozen==s['isolated_artifacts'],'sampler terminal closure differs')
 return dict(passed=True,independently_recomputed=True,original_failed_status=p.ref(out/'status.json'),original_export_error=s['error'],retained_weights_qualified=False,cases=len(expected),requests=sum(c['batch'] for c in s['cases']),native_idle_wakeup_cycles=3,frequency_domain_mhz=[900,1500,2100],loaded_tolerance_mhz=15,physical=physical,historical_profile_costs_recalibrated=False,files=audit.files(out))
