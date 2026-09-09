"""Read raw fixed window and complete planned-work denominator, never filter failures."""
import csv,json,math
def validate_window(b,row,summary,epoch,limits,out):
 w=summary.get('fixed_window',{})
 b.require(summary.get('measurement_schema')==3 and summary.get('measurement_window_protocol')=='per-dataset-slo-fixed-window-v2'
  and summary.get('fixed_window_valid') is True,'missing actual fixed-window evidence')
 b.require(epoch.get('accepted') is True and epoch.get('before_any_request_worker') is True
  and epoch['actual_epoch_s']==w.get('arrival_epoch_s') and epoch['actual_epoch_s']<=limits['latest_arrival_epoch_s'],'real epoch gate missing/late')
 b.require(w.get('arrival_window_s')==300 and w.get('arrival_window_end_s')==epoch['actual_epoch_s']+300
  and w.get('window_observed_complete') is True and w.get('clock_consistent') is True,'unobserved idle suffix or clock changed')
 start,end=summary['measurement_start_s'],summary['measurement_end_s']
 b.require(abs(start-epoch['actual_epoch_s'])<1e-5 and end>=epoch['actual_epoch_s']+300
  and end<=limits['cell_execution_deadline_s'],'energy window differs from admitted fixed window')
 b.require(summary.get('n_expected')==row['n_requests'] and summary.get('trace_sha256')==row['trace_sha256']
  and summary.get('slo_scale')==row['slo_scale'],'work/SLO hash differs')
 trace=b.read(row['trace']);records=list(csv.DictReader((out/'bench.csv').open()))
 b.require(len(records)==len(trace['requests'])==row['n_requests'],'missing failed/successful request rows')
 ids=[int(x['idx']) for x in records];b.require(set(ids)==set(range(len(records))) and len(set(ids))==len(ids),'request denominator changed')
 for record in records:
  request=trace['requests'][int(record['idx'])]
  b.require(int(record['prompt_len'])==request['prompt_len'] and int(record['output_len'])==request['output_len'],'planned input/output workload changed')
  b.require(abs(float(record['planned_arrival_s'])-epoch['actual_epoch_s']-request['arrival_s'])<1e-5,'planned arrival offset changed')
  if record['success']=='1':
   b.require(int(record['input_tokens'])==request['prompt_len'] and int(record['generated_tokens'])==request['output_len']
    and record['token_ids_verified']=='1','successful stream work/token verification differs')
 # Whole-work and SLO failures remain observations; no favorable-point gate.
 return True
