"""Variable-N fixed300 source binding. No historical minimum-N/span gate."""
import math
PROTOCOL='per-dataset-slo-fixed-window-v2'
SLOS={'alpaca':(1.,.1),'sharegpt':(5.,.15),'longbench':(15.,.2)}
def require(ok,reason):
 if not ok:raise ValueError(reason)
def validate(spec,source,contract,read_trace,hash_trace):
 require(spec['model']=='32b' and spec['protocol_id']==PROTOCOL and spec['execute_baselines'] is False,'PDB32B scope')
 require(contract['protocol_id']==PROTOCOL and contract['minimum_requests'] is None and contract['arrival_window_s']==300,'fixed window contract')
 sources={x['cell_id']:x for x in source['cells']}
 rows=spec['cells'];require(rows and len({r['cell_id'] for r in rows})==len(rows),'unique cells')
 for row in rows:
  original=sources.get(row['source_cell_id']);require(original==row['source_cell'],'source metadata changed')
  require(row['controller_config']==rows[0]['controller_config'] and row['system']=='pdblend' and row['model']=='32b','uniform PDB policy')
  trace=read_trace(row['trace']);requests=trace['requests'];arrivals=[r['arrival_s'] for r in requests]
  require(hash_trace(row['trace'])==row['trace_sha256']==original['trace_sha256'],'trace bytes changed')
  require(trace['protocol_id']==PROTOCOL and trace['measurement_schema']==3 and trace['duration_s']==trace['arrival_window_s']==300,'old window protocol')
  require(trace['split']==row['split']==original['split']=='development' and trace['model']=='32b','split/model changed')
  require(row['seed']==row['arrival_seed']==trace['seed']==original['arrival_seed'] and row['seed'] in (701,1701),'seed changed')
  require(len(requests)==len(trace['prompts'])==trace['n_requests']==row['n_requests']==original['n_requests'] and requests,'full variable request count')
  require(all(type(a) in (int,float) and math.isfinite(a) and 0<=a<300 for a in arrivals) and arrivals[0]==0 and arrivals==sorted(arrivals),'invalid arrival offsets')
  require(arrivals[-1]==row['planned_arrival_span_s']==original['planned_arrival_span_s'],'planned span changed')
  require(trace['dataset']==row['dataset']==original['dataset'] and trace['rate']==row['rate_rps']==original['rate_rps'],'labels/rate changed')
  require(row['slo_scale'] in (.5,1.,2.) and row['slo_ttft_s']==SLOS[row['dataset']][0]*row['slo_scale'] and row['slo_tpot_s']==SLOS[row['dataset']][1]*row['slo_scale'],'scaled user SLO differs')
  require(trace['formal_eligible'] is False and row['formal_eligible'] is False,'unverified formal claim')
 return rows
