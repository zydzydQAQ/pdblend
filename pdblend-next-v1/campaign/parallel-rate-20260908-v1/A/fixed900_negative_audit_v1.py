"""Read-only audit of the retained fixed-two 900 s capacity-negative reference.

A passed audit is not successful qualification or an equal-work energy result.
The original strict failed status and every original request remain unchanged.
"""
from pathlib import Path
import collections
import json
import math
import sys
A=Path(__file__).resolve().parent
sys.path.insert(0,str(A))
import continue_p6_qualification_v3 as prior
require, fixed, ref, sha, close = prior.require, prior.fixed, prior.ref, prior.sha, prior.close


def audit_rows(r,trace,config,raw,rows,timings):
 n=trace['n_requests'];require(n>0 and n==len(trace['requests'])==len(trace['prompts'])==len(rows)==r['n_expected']==r['n_rows'],'complete original request denominator required')
 require(len({str(x['request_id']) for x in rows})==n,'duplicate request identity')
 epoch=r['actual_arrival_epoch_s'];require(close(r['measured_arrival_duration_s'],900),'original 900s window required')
 timing={str(x['client_request_id']):x for x in timings}
 require(len(timing)==len(timings),'duplicate controller timing identity')
 good=failed=timeouts=rejected=recorded=0;late=[];timeout_ids=[]
 for idx,(x,t,prompt) in enumerate(zip(rows,trace['requests'],trace['prompts'])):
  require(x['idx']==idx and str(x['request_id'])==str(idx),'request order/identity changed')
  require(x['prompt_len']==t['prompt_len']==len(prompt) and x['output_len']==t['output_len'],'original prompt/output demand changed')
  planned=epoch+t['arrival_s'];deadline=x['request_deadline_s']
  require(math.isclose(x['planned_arrival_s'],planned,rel_tol=0,abs_tol=1e-6) and math.isclose(x['arrival_s'],planned,rel_tol=0,abs_tol=1e-6) and close(deadline-planned,120),'original open-loop arrival or 120s deadline changed')
  require(x['open_loop_independent'] is True and x['evaluation_protocol']=='evaluation-v3','independent original protocol required')
  delay=x['actual_dispatch_s']-planned;late.append(delay)
  require(-1e-6<=delay and close(delay,x['dispatch_delay_s']) and x['finish_s']>=x['actual_dispatch_s'],'invalid actual dispatch/finish timing')
  # This conservative diagnostic screen uses the already-declared TTFT budget.
  # It does not change performance SLO or discard any request from the denominator.
  require(delay<config['slo_ttft_s'],'dispatcher late by a full original TTFT budget; engineering diagnosis required')
  if x['success']==1:
   z=timing.get(str(idx));require(z is not None and z['completed'] is True and close(z['hard_deadline_s'],deadline),'successful native terminal/deadline differs')
   require(not x['error'] and x['request_timeout'] is False and not x['admission_rejection'] and x['token_ids_verified']==1 and x['generated_tokens']==t['output_len'] and x['input_tokens']==t['prompt_len'],'successful work/token evidence incomplete')
   require(all(type(x[k]) in (int,float) and math.isfinite(x[k]) and x[k]>=0 for k in ('ttft_s','tpot_s')),'successful timing invalid')
   ok=x['ttft_s']<config['slo_ttft_s'] and x['tpot_s']<config['slo_tpot_s'];require(x['slo_ok']==int(ok),'successful SLO differs');good+=ok;recorded+=x['generated_tokens']
  else:
   failed+=1;require(x['success']==0 and x['slo_ok']==0,'failure must count in denominator and not good requests')
   require(x['generated_tokens']==0 and x['token_ids_verified']==0 and x['token_count_source']=='missing','unreviewed failed token accounting')
   if x['request_timeout'] is True:
    require(x['error']=='request_hard_timeout' and x['http_status'] is None and not x['admission_rejection'],'unexpected timeout error')
    require(0<=x['finish_s']-deadline<1,'timeout did not terminate at original hard deadline')
    z=timing.get(str(idx));require(z is not None and z['completed'] is False,'timeout lacks original native/controller unsuccessful terminal record')
    require(close(z['planned_arrival_s'],planned) and close(z['hard_deadline_s'],deadline) and close(z['actual_dispatch_s'],x['actual_dispatch_s']) and z['queued_s']<deadline and z['first_planning_s']<deadline and z['cleanup_end_s']>=deadline,'timeout was not original queued capacity wait')
    require((x['n_text_chunks']==0 and x['first_token_s'] is None) or (x['n_text_chunks']>0 and planned<=x['first_token_s']<=x['last_token_s']<=x['finish_s'] and len(json.loads(x['chunk_itl_s']))==x['n_text_chunks']-1),'partial timeout chunk timing/count inconsistent')
    timeouts+=1;timeout_ids.append(idx)
   else:
    require(x['http_status']==429 and x['admission_rejection']=='admission_queue_full','unexpected non-timeout failure')
    prefix='RuntimeError: HTTP 429: ';require(x['error'].startswith(prefix),'named rejection has no original HTTP error JSON')
    e=json.loads(x['error'][len(prefix):])['error'];require(e.get('type')=='admission_rejection' and e.get('code')=='admission_queue_full' and x['finish_s']<deadline,'rejection code/deadline mismatch')
    require(str(idx) not in timing and x['n_text_chunks']==0 and x['first_token_s'] is None,'rejected-before-admission row unexpectedly has output or a dispatched timing');rejected+=1
 require(failed>0 and failed==timeouts+rejected==r['failed_requests'] and timeouts==r['request_timeouts'] and r['work_complete'] is False,'capacity failure counts differ')
 require(good==r['n_good'] and close(good/n,r['slo_attainment']),'original all-request SLO differs')
 require(close(r['energy_j'],raw['energy_j']) and close(r['offered_rate_rps'],n/900),'raw energy/rate differs')
 require(raw['measurement_start_s']<=epoch and raw['measurement_end_s']>=max(epoch+900,max(x['finish_s'] for x in rows)),'raw power does not cover arrival and completed drain')
 require(set(timing)=={str(i) for i,x in enumerate(rows) if not x['admission_rejection']},'controller terminal ledger and raw work differ')
 return dict(n_expected=n,n_completed=n-failed,n_good=good,slo_attainment=good/n,energy_j=raw['energy_j'],failed_requests=failed,request_timeouts=timeouts,admission_rejections=rejected,successful_truncated_requests=0,max_actual_dispatch_delay_s=max(late),timeout_request_indices=timeout_ids,recorded_generated_tokens=recorded,generated_token_count_complete=False,generated_tokens_semantics='verified successful output count; missing timeout usage is not proof of true zero output',unverified_partial_output_requests=timeouts,observed_partial_timeout_requests=sum(bool(x['request_timeout'] and x['n_text_chunks']>0) for x in rows),observed_partial_timeout_chunks=sum(x['n_text_chunks'] for x in rows if x['request_timeout']))


def clock_audit(rows):
 pending={};writes=parks=0
 for x in rows:
  require(not any(x.get(k) for k in ('error','physical_command_uncertainty','sticky_failure','pending_physical_commands')),'physical command uncertainty/error requires engineering stop')
  kind=x['kind']
  require(kind in ('idle_first_admission_clock','physical_clock_observation','frequency_coverage_limited','physical_clock_write','physical_clock_park'),'unreviewed clock event kind')
  if kind in ('physical_clock_write','physical_clock_park'):
   key=(kind,x['gpu'],x['started_s'],x['target_mhz']);require(x['gpu'] in (6,7),'clock write outside original layout')
   if x['stage']=='intent':require(key not in pending and x['completed'] is False,'duplicate physical intent');pending[key]=x
   else:
    require(x['stage']=='command_completed' and x['completed'] is True and key in pending and x['finished_s']>=x['started_s'],'physical command missing/mismatched terminal');pending.pop(key)
    if kind=='physical_clock_write':writes+=1
    else:parks+=1
 require(not pending,'unterminated physical command')
 return dict(physical_writes=writes,physical_parks=parks,all_physical_commands_terminal=True)


def identity_records(records,instances):
 require(len(records)==len(instances)==2,'exact original two identities required')
 actual={x['runtime']['id']:x for x in records};require(set(actual)=={i['id'] for i in instances},'foreign/missing instance')
 for i in instances:
  x=actual[i['id']];c=x['container'];s=c['State'];expected=i['container']
  require(c['Id']==expected['id'] and c['Image']==expected['image'] and s['StartedAt']==expected['StartedAt'] and s['Pid']==i['host_pid'] and s['Running'] is True,'actual original container process changed')
  require(all(x['provenance'].get(k)==v for k,v in i['provenance'].items()),'actual imported engine provenance changed')
 return actual


def audit(out,spec_reference,declaration,terminal_proof_reference=None):
 pair_ref=ref(A/'p6-qualification900-inputs-002/declaration.json');require(fixed(pair_ref)==declaration,'original paired declaration changed')
 out=Path(out).resolve();status=fixed(ref(out/'status.json'))
 require(status.get('complete') is False and status.get('phase')=='needs_attention' and status.get('error')=="ValueError('900s request failure/incomplete work prevents any successor')",'original strict failed gate must be retained')
 require(status.get('cleanup_complete') is True and not status.get('cleanup_errors') and status.get('finished_s') and not prior.alive(status['pid']),'predecessor not terminal/clean')
 require(len(status['completed'])==1,'exactly one original fixed900 result required')
 require(json.loads((out/'spec-reference.json').read_text())==spec_reference==declaration['specs']['fixed2'],'executed fixed2 specification differs')
 sp=fixed(spec_reference);require(sp['mode']=='qualification900' and sp['arm']=='fixed2' and sp['deadline_s'] is None and sp['campaign_lifecycle']=='until_declared_complete_v1','original fixed2 lifecycle changed')
 require(sp['trace']==declaration['trace'] and sp['profiles']==declaration['profile'] and sp['capacity_binding']==declaration['capacity_binding'] and sp['actual_certificate']==declaration['certificate'] and ref(Path(sp['host_release'])/'manifest.json')==declaration['source'],'original paired source/profile/trace/certificate changed')
 require(sp['files'] and all(sha(p)==h for p,h in sp['files'].items()),'original frozen source files changed')
 cap=fixed(sp['capacity_binding']);prior.source_identity(sp['capacity_binding'],cap['identity']);config=fixed(sp['config']);fixed(sp['profiles']);fixed(sp['actual_certificate'])
 require(config['profiles']==sp['profiles']['path'] and config['capacity_integration_v1'] is False and config['capacity_binding_path']==sp['capacity_binding']['path'] and config['capacity_binding_sha256']==sp['capacity_binding']['sha256'],'fixed2 runtime source/config differs')
 actual_config=fixed(ref(out/'runtime-config.json'));expected=dict(config,journal=str(out/'control.jsonl'),capacity_inventory_path=str(out/'inventory.json'));require(actual_config==expected,'actual runtime policy differs from frozen config')
 binding=fixed(sp['original_binding']);require(fixed(ref(out/'execution-binding.json'))==binding,'actual execution binding differs')
 identity_records(fixed(ref(out/'identity.before.json')),binding['instances'])
 r=fixed(status['completed'][0]);require(status['completed'][0]==ref(out/'qualification900/result.json'),'foreign phase result')
 expected_source=dict(original_binding=sp['original_binding'],capacity_binding=sp['capacity_binding'],config=sp['config'],host_manifest=ref(Path(sp['host_release'])/'manifest.json'))
 require(r['source']==expected_source and r['schema']=='capacity-load-measurement-v1' and r['complete'] is True and r['native_idle'] is True and r['trace']==sp['trace'] and r['demand_domain_sha256']==sp['demand_domain_sha256'],'fixed900 source/complete measurement differs')
 require(r['resident_before']==r['resident_after']==[dict(gpus=[6],id='nextv3a6'),dict(gpus=[7],id='nextv3a7')] and r['resident_groups']==[[6],[7]],'fixed2 physical layout changed')
 drain=r['dynamic_drain'];require(drain['drain_complete'] is True and not drain['residual'] and not drain['error'] and set(drain['drain_barriers'])=={'nextv3a6','nextv3a7'},'original drain incomplete')
 for d in drain['drain_barriers'].values():
  require(d['drained'] is True and d['mode']=='continuous' and d['role']=='mixed' and d['send_counters_verified'] is True,'native drain barrier invalid')
  require(d['transfers'] and all(x['send_healthy'] is True and not any(x.get(k) for k in ('inflight_receives','inflight_sends','buffered_gpu_bytes','buffered_tensors','send_failed','send_last_error')) for x in d['transfers']),'native transfer residual/failure')
 trace=fixed(r['trace']);prior.validate_trace(trace,sp['demand_domain_sha256'],duration=900)
 require([p['name'] for p in trace['phases']]==['low','high','low'] and all(p['duration_s']==300 for p in trace['phases']),'original low/high/low phase windows differ')
 paths=[p for p in r['artifacts'] if Path(p).name=='requests.json'];require(paths==[str(out/'qualification900/requests.json')] and all(sha(p)==h for p,h in r['artifacts'].items()),'original request artifact identity/hash changed')
 control=[json.loads(l) for l in (out/'control.jsonl').read_text().splitlines()];timings=[x for x in control if x['kind']=='request_timing']
 require(not any(x['kind'] in ('error','request_error','physical_failure','rollback_failed') for x in control),'engineering control event')
 raw=prior.raw_measurement(r['raw_measurement']);fullraw=prior.raw_measurement(status['full_operation_measurement'])
 derived=audit_rows(r,trace,config,raw,json.loads(Path(paths[0]).read_text()),timings)
 clocks=clock_audit([json.loads(l) for l in (out/'control.clock-guard.jsonl').read_text().splitlines()])
 dispatch=[json.loads(l) for l in (out/'engine-dispatch.jsonl').read_text().splitlines()];require(len(dispatch)==status['dispatched_native_requests'] and all(x['instance_id'] in ('nextv3a6','nextv3a7') for x in dispatch),'native dispatched ledger/layout changed')
 # A separate, timestamped post-failure identity proof is mandatory. The producer
 # skipped its happy-path after file at the strict failure; never synthesize it.
 if terminal_proof_reference is None:terminal_proof_reference=ref(A/'p6-fixed900-negative-diagnosis-001/fresh-terminal-snapshot.json')
 proof=fixed(terminal_proof_reference)
 validate_terminal_proof(proof,out,status,binding)
 files={str(p):sha(p) for p in out.rglob('*') if p.is_file()};files.update(sp['files']);files.update(cap['files']);files.update(proof['files'])
 for rr in (pair_ref,spec_reference,r['trace'],sp['profiles'],sp['config'],sp['original_binding'],sp['capacity_binding'],sp['actual_certificate'],declaration['source'],terminal_proof_reference):files[rr['path']]=rr['sha256']
 for module in (Path(__file__),Path(prior.__file__),Path(prior.raw_measurement.__code__.co_filename),Path(prior.validate_trace.__code__.co_filename)):files[str(module.resolve())]=sha(module)
 return dict(schema='fixed900-capacity-negative-independent-audit-v1',passed=True,result=status['completed'][0],status=ref(out/'status.json'),spec=spec_reference,original_pair_declaration=pair_ref,declaration=declaration,output=str(out),full_work=False,measurement_valid=True,classification='fixed2_capacity_negative',raw_energy_recomputed=True,equal_work_energy_comparison_eligible=False,strict_qualification_passed=False,original_strict_failed_status_retained=True,terminal_identity_proof=terminal_proof_reference,auditor_source=ref(__file__),files=files,full_operation_energy_j=fullraw['energy_j'],**derived,**clocks)


def validate_terminal_proof(proof,out,status,binding):
 require(proof.get('schema')=='A-fixed900-fresh-readonly-terminal-snapshot-v1' and proof.get('passed') is True and proof['original_status']==ref(out/'status.json') and proof['original_before']==ref(out/'identity.before.json') and fixed(proof['original_binding'])==binding,'terminal physical proof must pin this original status/before/binding')
 require(proof['started_s']>=status['finished_s'] and proof['finished_s']>=proof['started_s'] and proof['old_owner_exited'] is True and proof['fresh_post_termination_evidence'] is True and proof['not_part_of_original_measurement'] is True and proof['old_identity_after_not_created'] is True and proof['read_only'] is True and proof['node_lease_held_during_snapshot'] is True and proof['node_lease_released'] is True and proof['hostname']==binding['hostname'],'terminal proof predates cleanup or wrong owner/lifecycle')
 require(proof['files'] and all(sha(p)==h for p,h in proof['files'].items()),'terminal identity raw proof changed')
 require({int(x.split(',')[0]) for x in proof['all_eight_clock_snapshot_csv'].strip().splitlines()}==set(range(8)),'fresh snapshot does not include eight GPUs')
 actual=identity_records(proof['actual_identity'],binding['instances'])
 before=identity_records(fixed(proof['original_before']),binding['instances'])
 require(all(actual[k]['provenance']==before[k]['provenance'] for k in actual),'full before/after process provenance differs')
 for x in actual.values():
  r=x['runtime'];require(r['active']==r['running']==r['waiting']==0 and r['mode']=='continuous' and r['role']=='mixed' and r['accepting'] is True and r['admit_prefill'] is True and r['admit_decode'] is True and not r.get('error') and not r.get('runtime_error'),'original two not restored to ordinary idle')
  require(r['transport_healthy'] is True and r['transfer_send_healthy'] is True and not any(r.get(k) for k in ('transfer_inflight_receives','transfer_inflight_sends','transfer_buffered_tensors','transfer_send_failed','kv_allocations','transfer_allocations','scheduler_budget_pending')),'terminal native residual/error')


def validate_evidence(reference):
 previous=fixed(reference);require(previous['auditor_source']==ref(__file__) and previous['files'] and all(sha(p)==h for p,h in previous['files'].items()),'fixed reference auditor/evidence changed')
 current=audit(previous['output'],previous['spec'],previous['declaration'],previous['terminal_identity_proof'])
 require(current==previous,'fixed reference independent audit changed')
 return current
