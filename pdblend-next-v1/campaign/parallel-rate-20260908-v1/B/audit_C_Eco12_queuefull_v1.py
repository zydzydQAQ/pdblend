"""Independent CPU diagnosis of C's first corrected EcoServe rate observation."""
import csv,hashlib,json,os,socket,time
from pathlib import Path
import audit_original_AB_Eco60_window_v1 as w
B=Path(__file__).resolve().parent; INPUT=B/'C-Eco12-queuefull-inputs-v1.json';OUT=B/'C-Eco12-queuefull-independent-audit-v1.json'
def main():
 data=w.read(INPUT)
 for p,h in data['files'].items():assert w.sha(p)==h,p
 cp=w.read(data['checkpoint']);binding=w.read(cp['binding']);receipt=w.read(cp['receipt']);cell=Path(data['checkpoint']).parents[1]/'cells'/cp['row']['cell_id'];summary=w.read(cell/'summary.json');bench=w.rows(cell/'bench.csv');controls=[json.loads(l) for l in (cell/'control.jsonl').read_text().splitlines()];power=w.rows(cell/'power.csv')
 print(json.dumps(dict(pid=os.getpid(),host=socket.gethostname(),started_s=time.time())),flush=True)
 for p,h in cp['artifacts'].items():assert w.sha(p)==h,p
 assert w.sha(cp['binding'])==cp['binding_sha256'];assert w.sha(cp['receipt'])==cp['receipt_sha256']
 admissions={x['request_id']:x for x in controls if x['kind']=='admission'};timing={x['request_id']:x for x in controls if x['kind']=='request_timing'};byclient={a['client_request_id']:a for a in admissions.values()};bybench={r['idx']:r for r in bench};ends={x['request_id']:x for x in controls if x['kind']=='request_end'}
 eng={};cfg={}
 for i in binding['instances']:
  ec=Path(i['engine_config']);assert binding['files'][str(ec)]==w.sha(ec);cfg[i['id']]=w.read(ec)
  for path,h in i['provenance']['source_files_at_import'].items():assert w.sha(path)==h;eng[path]=h
 assert len(eng)==1
 code=Path(next(iter(eng))).read_text();assert 'if len(self.streams) >= self.config.get("max_pending", 128):' in code and 'raise web.HTTPTooManyRequests(text="bounded admission queue full")' in code
 failed=[]
 for r in bench:
  if r['success']=='1':continue
  assert r['error']=='RuntimeError: HTTP 503: decode failed: bounded admission queue full' and r['request_timeout']=='False' and not r['first_token_s']
  a=byclient[r['idx']];rid=a['request_id'];t=timing[rid];target=a['plan']['routes'][0]['decode_id'];at=t['forward_started_s'];active=[]
  for other,ot in timing.items():
   if other==rid or ot.get('forward_started_s',float('inf'))>at:continue
   oa=admissions[other];br=bybench[oa['client_request_id']]
   if oa['plan']['routes'][0]['decode_id']!=target:continue
   if br['success']=='1' and float(br['last_token_s'])>at:active.append(oa['client_request_id'])
  gpus=next(i['gpus'] for i in binding['instances'] if i['id']==target);start=max(float(summary['measurement_start_s']),at-3);finish=min(float(summary['measurement_end_s']),at+3)
  failed.append(dict(client_id=r['idx'],server_request_id=rid,target=target,actual_native_pending_limit=cfg[target].get('max_pending',128),actual_native_batch_seq_limit=cfg[target].get('max_num_seqs'),independently_proven_other_native_streams_alive=len(active),proven_active_clients=active,dispatch_lateness_s=float(r['actual_dispatch_s'])-float(r['planned_arrival_s']),handler_delay_s=t['handler_arrival_s']-t['actual_dispatch_s'],queue_to_forward_s=at-t['queued_s'],forward_to_terminal_s=ends[rid]['at_s']-at,native_rejection_no_false_completion=ends[rid]['completed'] is False,error=r['error'],request_timing=t,target_util_near_rejection_pct=w.mean_util(power,gpus,start,finish)))
 lat=[float(r['actual_dispatch_s'])-float(r['planned_arrival_s']) for r in bench];handler=[t['handler_arrival_s']-t['actual_dispatch_s'] for t in timing.values()]
 assert len(failed)==42 and len(bench)==1190 and summary['completed_work_requests']==1148 and summary['request_timeouts']==0 and summary['runtime_error'] is None and summary['drain_complete'] is True
 # Recompute original all-eight measurement integral without importing an executor.
 energy=0;s=float(summary['measurement_start_s']);e=float(summary['measurement_end_s'])
 for a,b in zip(power,power[1:]):
  t0=float(a['t_s']);t1=float(b['t_s']);l=max(t0,s);h=min(t1,e)
  if h<=l:continue
  v0=sum(float(a[f'gpu{g}_w']) for g in range(8));v1=sum(float(b[f'gpu{g}_w']) for g in range(8));energy+=(v0+(v1-v0)*(l-t0)/(t1-t0)+v0+(v1-v0)*(h-t0)/(t1-t0))*(h-l)/2
 assert abs(energy-summary['energy_j'])<1e-6
 native_before=w.read(Path(cp['receipt']).parent/'identity.before.json');native_after=w.read(Path(cp['receipt']).parent/'identity.after.json')
 result=dict(schema='C-Eco12-corrected-window-native-queue-rejection-independent-audit-v1',host=socket.gethostname(),pid=os.getpid(),created_s=time.time(),source=w.ref(Path(__file__).resolve()),inputs=w.ref(INPUT),checkpoint=w.ref(data['checkpoint']),binding=w.ref(cp['binding']),receipt=w.ref(cp['receipt']),engine_sources=eng,failed_request_evidence=failed,observed_count=1190,completed_count=1148,failed_count=42,timeouts=0,energy_j=energy,slo_attainment=summary['slo_attainment'],work_complete=False,raw_all8_energy_valid=True,actual_dispatch_max_s=max(lat),actual_dispatch_p99_s=w.pct(lat,.99),handler_delay_max_s=max(handler),native_pending_limits=sorted({v['actual_native_pending_limit'] for v in failed}),native_batch_seq_limits=sorted({v['actual_native_batch_seq_limit'] for v in failed}),proven_active_streams_min=min(v['independently_proven_other_native_streams_alive'] for v in failed),proven_active_streams_max=max(v['independently_proven_other_native_streams_alive'] for v in failed),max_forward_rejection_latency_s=max(v['forward_to_terminal_s'] for v in failed),min_target_util_near_rejection_pct=min(v['target_util_near_rejection_pct'] for v in failed),drain_complete=summary['drain_complete'],runtime_error=summary['runtime_error'],limitations=['Native HTTP streams threshold is128, separate from max_num_seqs32 batch. Source error proves threshold was hit; reconstructed alive successful streams are a lower bound at forward time, not a new direct native-stream counter.','This analysis preserves policy and measurement; no expansion or new source is executed. Additional guard timing causality is reviewed separately.'])
 assert not OUT.exists()
 with OUT.open('x') as f:json.dump(result,f,indent=2)
 print(json.dumps({k:v for k,v in result.items() if k not in ('failed_request_evidence','engine_sources')}),flush=True)
if __name__=='__main__':main()
