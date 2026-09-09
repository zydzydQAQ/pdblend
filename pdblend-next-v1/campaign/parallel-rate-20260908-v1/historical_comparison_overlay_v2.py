"""Derived historical eligibility: diagnosed arrival stalls and temporal window stranding.
Original 720 records and their raw energy/count fields are never edited.
"""
import copy
from pathlib import Path
import historical_comparison_overlay_v1 as legacy
sha=legacy.sha;need=legacy.need;checked=legacy.checked
ROOT=Path(__file__).resolve().parent
ECO_AUDITS={
 str(ROOT/'B/original-AB-Eco60-window-audit-v1.json'):'b30da7db17f5b80c9744ff9faccd97a45642490ea5bc47cc0ecb82981f8b27b3',
 str(ROOT/'B/original-C-Eco6-window-audit-v1.json'):'bfc304a3fe2395f527ff1457ffd101ae4cc27ed2d963c662afd7dd8fc79bc984',
}
ECO_QUARANTINED=frozenset({
 '14b-alpaca-r12-s701-w100-ecoserve-slo1','14b-alpaca-r9-s701-w100-ecoserve-slo1','14b-alpaca-r4.8-s701-w100-ecoserve-slo1',
 '14b-longbench-r1.25-s701-w100-ecoserve-slo1','14b-longbench-r1-s701-w100-ecoserve-slo1','14b-longbench-r0.75-s701-w100-ecoserve-slo1','14b-longbench-r0.4-s701-w100-ecoserve-slo1',
 '32b-longbench-r0.25-s701-w100-ecoserve-slo1','32b-longbench-r0.125-s701-w100-ecoserve-slo1','32b-longbench-r0.3-s701-w100-ecoserve-slo1','32b-longbench-r0.2-s701-w100-ecoserve-slo1',
 '7b-longbench-r1.5-s701-w100-ecoserve-slo1','7b-alpaca-r12-s701-w100-ecoserve-slo1','7b-longbench-r3-s701-w100-ecoserve-slo1','7b-longbench-r1.25-s701-w100-ecoserve-slo1','7b-alpaca-r7.5-s701-w100-ecoserve-slo1','7b-longbench-r1-s701-w100-ecoserve-slo1'})
QUARANTINED=legacy.QUARANTINED|ECO_QUARANTINED

def verify_eco(points,references,sources):
 need({r['path']:r['sha256'] for r in references}==ECO_AUDITS and len(references)==2,'exact independently reviewed Eco window audit set required')
 by={p['cell_id']:p for p in points};need(len(by)==len(points),'duplicate historical points')
 bad=[];all_audited=[]
 for reference in references:
  audit=checked(reference,sources);need(audit['original_720_unchanged'] is True,'raw archive mutation not allowed')
  for key in ('source','replay_source'):
   if key in audit:
    source=audit[key];need(sha(source['path'])==source['sha256'],'window auditor changed');sources[source['path']]=source['sha256']
  inputs=checked(audit['inputs'],sources)
  need(len(audit['points'])==audit['point_count'] and audit['point_count'] in (60,6),'wrong original Eco audit coverage')
  for raw in audit['points']:
   cid=raw['cell_id'];need(cid in by,'window audit outside original720');point=by[cid];all_audited.append(cid)
   need(point['system']=='ecoserve' and point['phase']=='main' and point['metrics_verified'] is True,'original raw-verified Eco main point required')
   if not raw['scientific_quarantine_recommended']:continue
   scheduler=raw['original_scheduler'];need(sha(scheduler['path'])==scheduler['sha256'],'original membership source changed');sources[scheduler['path']]=scheduler['sha256']
   cp=checked(raw['checkpoint'],sources);receipt=checked(raw['receipt'],sources)
   need(point['checkpoint_path']==raw['checkpoint']['path'] and point['receipt_path']==raw['receipt']['path'],'window audit target execution changed')
   need(point['executed_source']['binding_sha256']==raw['source']['binding_sha256'],'window audit source binding changed')
   need(all(point[k]==raw[k] for k in ('n_expected','failed_requests','request_timeouts','energy_j','slo_attainment','work_complete')),'window audit metrics differ')
   for path,digest in raw['raw_files'].items():
    need(cp['artifacts'].get(path)==digest and sha(path)==digest,'window audit raw changed: '+path);sources[path]=digest
   evidence=[r for r in raw['failed_request_evidence'] if r.get('window_stranding_demonstrated')]
   need(len(evidence)==raw['demonstrated_window_stranding_count']>0,'no direct window-stranding victim')
   for row in evidence:
    need(row['admitted'] and row['no_client_or_server_first_token'] and row['error']=='request_hard_timeout' and row['n_text_chunks']=='0','window victim is not an admitted no-token timeout')
    off=row['final_off_without_reopen'];need(off and off['on'] is False,'actual/reconstructed OFF absent')
    following=[v for v in row['window_events_before_request_end'] if v['index']>off['index']]
    need(not any(v['on'] for v in following) and row['target_gpu_final30_util_pct']<1,'window reopened or target was still busy')
   bad.append(dict(cell_id=cid,checkpoint=raw['checkpoint'],receipt=raw['receipt'],audit=reference,
    raw_arithmetic_verified=True,scientific_comparison_eligible=False,
    reason='admitted prefill requests had no first token, their final temporal window was OFF without reopening, and target GPU final30s utilization was below1%; exact original membership replay included',
    classification='ecoserve_temporal_prefill_window_liveness_defect',energy_j=point['energy_j'],energy_preserved=True,demonstrated_victims=len(evidence),failed_request_evidence=evidence))
 need(len(all_audited)==66 and len(set(all_audited))==66,'duplicate or omitted original Eco audit rows')
 need(len(bad)==17 and {r['cell_id'] for r in bad}==ECO_QUARANTINED,'exact seventeen demonstrated original Eco cells required')
 return bad

def verify(points,audit_reference,sources,eco_references=None):
 old=legacy.verify(points,audit_reference,sources)
 refs=eco_references if eco_references is not None else [dict(path=p,sha256=h) for p,h in ECO_AUDITS.items()]
 eco=verify_eco(points,refs,sources)
 return dict(schema='historical-scientific-comparison-overlay-v2',original_raw_records_unchanged=True,timing_audit=audit_reference,eco_window_audits=refs,explicit_diagnosed_cells_only=True,quarantined=old['quarantined']+eco,raw_energy_not_erased=True,raw_energy_not_reintegrated_by_this_overlay=True,applies_to_comparisons_and_figures=True,scope='Two actual arrival stalls plus seventeen original Eco window-liveness defects; incomplete/low-SLO observations without these engineering defects are retained.')

def comparison_view(points,overlay):
 need(overlay['schema']=='historical-scientific-comparison-overlay-v2' and len(overlay['quarantined'])==19 and {p['cell_id'] for p in overlay['quarantined']}==QUARANTINED,'exact nineteen historical engineering exclusions required')
 excluded={p['cell_id']:p for p in overlay['quarantined']};view=copy.deepcopy(points)
 need(len({p['cell_id'] for p in view})==len(view) and QUARANTINED<={p['cell_id'] for p in view},'historical identities differ')
 for p in view:
  p['raw_arithmetic_verified']=p['metrics_verified'];p['scientific_comparison_eligible']=bool(p['metrics_verified'] and p['cell_id'] not in excluded)
  if p['cell_id'] in excluded:
   q=excluded[p['cell_id']];need(q['scientific_comparison_eligible'] is False and q['raw_arithmetic_verified'] is True and q['energy_j']==p['energy_j'],'exclusion erased/changed energy or eligibility')
   p.update(metrics_verified=False,status='engineering_invalid',scientific_exclusion_reason=q['reason'])
 return view
