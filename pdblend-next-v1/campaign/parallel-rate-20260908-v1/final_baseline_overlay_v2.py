"""Explicit engineering quarantine and whole-group baseline replacement.

Raw arithmetic and scientific eligibility are separate facts. This module never
rewrites original checkpoints, metrics, or the historical 450-point snapshot.
"""
from pathlib import Path
import source_identity_v2
import eco_baseline_replacement_v1 as eco_replacement
import eco_baseline_replacement_b_v1 as eco_replacement_b
import eco_baseline_replacement_a_v1 as eco_replacement_a


def validate(p,selection,originals):
 ref=selection['baseline_overlay'];doc=p.checked(ref)
 p.need(doc['schema']=='final-baseline-scientific-overlay-v2' and doc['authorized'] is True,'unapproved baseline overlay')
 sources={ref['path']:ref['sha256']};quarantine={};groups=[];retired={};fresh={};original_by_id={x['cell_id']:x for x in originals}
 def pin(r):
  sources[r['path']]=r['sha256']
  p.need(p.sha(r['path'])==r['sha256'],'overlay reference changed')
  return p.read(r['path']) if Path(r['path']).suffix=='.json' else None
 for entry in doc['engineering_quarantines']:
  evidence=pin(entry['evidence']);records=evidence.get('points',[evidence]);matches=[x for x in records if x.get('cell_id')==entry['cell_id']]
  p.need(len(matches)==1,'quarantine evidence must name one exact point');record=matches[0];cp=pin(record['checkpoint']);pin(record['receipt'])
  p.need(cp['row']['cell_id']==entry['cell_id'] and entry['cell_id'] not in quarantine,'quarantine checkpoint/ID mismatch or duplicate')
  raw=record.get('raw_timing_evidence',record.get('raw_files'))
  p.need(raw and all(p.sha(path)==digest for path,digest in raw.items()),'quarantine raw timing evidence changed');sources.update(raw)
  if record.get('classification')=='ecoserve_temporal_prefill_window_liveness_defect':
   p.need(evidence['schema']=='baseline-engineering-quarantine-v1' and evidence['no_original_raw_modified'] is True
          and record['scientific_eligible_false'] is True and record['scientific_comparison_eligible'] is False,
          'EcoServe quarantine lacks an explicit engineering diagnosis')
   if evidence.get('diagnosis'):
    pin(evidence['diagnosis']);pin(evidence['replacement_declaration'])
   elif evidence.get('audit'):
    audit=pin(evidence['audit'])
    p.need(audit['original_720_unchanged'] is True and audit['raw_energy_preserved'] is True
           and entry['cell_id'] in audit['quarantined_cells'], 'EcoServe raw audit does not quarantine this point')
    failures=record['failed_request_evidence']
    p.need(failures and all(x['window_stranding_demonstrated'] is True
           and x['no_client_or_server_first_token'] is True
           and x['target_gpu_final30_util_pct'] == 0 for x in failures),
           'EcoServe AB diagnosis lacks stranded work evidence')
    pin(audit['source'])
   else:
    pin(evidence['source']);pin(evidence['original_snapshot'])
    # C's successor contains both the original six and its prior new-rate
    # diagnosis. An independent full membership replay is pinned in the entry.
    proof=pin(entry['membership_replay'])
    p.need(proof['schema']=='original-C-Eco6-full-window-replay-readonly-audit-v1'
           and proof['original_720_unchanged'] is True
           and entry['cell_id'] in proof['quarantined_cells'], 'C window replay authority differs')
    matched=[x for x in proof['points'] if x['cell_id']==entry['cell_id']]
    p.need(len(matched)==1 and matched[0]['checkpoint']==record['checkpoint']
           and matched[0]['scientific_quarantine_recommended'] is True
           and matched[0]['demonstrated_window_stranding_count']==matched[0]['failed_requests']>0
           and matched[0]['raw_energy_preserved'] is True
           and record['raw_energy_retained'] is True, 'C full window replay does not prove the exact failure')
    for key in ('source','replay_source','inputs'):pin(proof[key])
  else:
   p.need(record.get('actual_dispatch_lateness_max_s',0)>10,'quarantine lacks observed dispatch starvation diagnosis')
  quarantine[entry['cell_id']]=dict(entry,checkpoint=record['checkpoint'],receipt=record['receipt'],record=record)
 for reference in doc['replacement_groups']:
  d=pin(reference)
  if d['schema']=='A-Eco-whole-source-logical31-v1':
   additions,old_group=eco_replacement_a.validate(p,reference,d,originals,sources)
   p.need(not set(additions)&set(fresh) and not set(old_group)&set(retired),'overlapping whole-group replacement')
   fresh.update(additions);retired.update(old_group);groups.append(reference);continue
  if d['schema']=='B-Eco-window-drain-whole-group-v1':
   additions,old_group=eco_replacement_b.validate(p,reference,d,originals,sources)
   p.need(not set(additions)&set(fresh) and not set(old_group)&set(retired),'overlapping whole-group replacement')
   fresh.update(additions);retired.update(old_group);groups.append(reference);continue
  if d['schema']=='C-EcoServe-whole-source-replacement-37-v1':
   additions,old_group=eco_replacement.validate(p,reference,d,originals,sources)
   p.need(not set(additions)&set(fresh) and not set(old_group)&set(retired),'overlapping whole-group replacement')
   fresh.update(additions);retired.update(old_group);groups.append(reference);continue
  p.need(d['schema']=='B-whole-group-cooperative-Dynamo-eight-v1','unrecognized whole-group replacement')
  retired_before=set(retired)
  parent=pin(d['parent_declaration']);host=pin(d['host_manifest']);cpu=pin(d['cpu_validation']);pin(d['original_alp4_quarantine'])
  p.need(cpu['passed'] is True and d['host_manifest']['path']==str(Path(d['host_release'])/'manifest.json'),'replacement source/CPU declaration differs')
  for path,digest in d['source_files'].items():p.need(p.sha(path)==digest,'replacement declaration source changed');sources[path]=digest
  for relative,digest in host['files'].items():p.need(p.sha(Path(d['host_release'])/relative)==digest,'replacement serving source changed')
  p.need(d['deadline_s'] is None and d['arrival_window_s']==100 and d['request_hard_timeout_s']==120 and d['drain_after_arrival_window_s']==120 and d['cleanup_local_budget_s']==90 and d['energy_gpu_indices']==list(range(8)),'replacement changed experiment budget')
  old={x['cell_id']:x for x in parent['cells']};rows=d['cells'];expected={(ds,rate,rep) for ds,rate in [('alpaca',4.),('alpaca',5.),('sharegpt',1.25),('sharegpt',1.5)] for rep in (1,2)}
  p.need(len(rows)==8 and {(x['dataset'],x['rate_rps'],x['repeat']) for x in rows}==expected and d['new_rate_count']==6 and d['original_alp4_repair_count']==2,'whole group cannot select individual outcomes or expand rate scope')
  expected_retired={x['cell_id'] for x in d['original_newrate_group_rows']};p.need(len(expected_retired)==6,'complete old new-rate group required')
  for row in rows:
   source=old[row['original_cell_id']];p.need(row['source_row']==source and row['system']=='dynamollm' and row['model']=='32b' and row['engineering_replacement'] is True,'replacement changed parent row/system')
   for field in (*p.PAIR_FIELDS,'repeat','system','trace_path','arrival_window_s','slo_scale'):
    a=row['n_requests'] if field=='n_expected' else row[field];b=source['n_requests'] if field=='n_expected' else source[field];p.need(a==b,'replacement changed exact scientific pair: '+field)
   p.need(p.sha(row['trace_path'])==row['trace_sha256'],'replacement trace changed');sources[row['trace_path']]=row['trace_sha256']
   cp=pin(row['paired_pdb_checkpoint']);r=cp['row'];p.need(cp['work_complete'] and r['system']=='pdblend' and r['repeat']==row['repeat'],'replacement paired PDB is not complete exact repeat')
   for field in p.PAIR_FIELDS:
    p.need((r['n_requests'] if field=='n_expected' else r[field])==(row['n_requests'] if field=='n_expected' else row[field]),'replacement PDB trace/SLO/denominator differs')
   cid=row['cell_id'];p.need(cid not in fresh,'duplicate replacement execution');fresh[cid]=dict(row=row,declaration=reference,host_release=d['host_release'],host_manifest=d['host_manifest'])
   if row.get('replaces_original_historical_id'):
    hid=row['replaces_original_historical_id'];p.need(hid in quarantine and hid in original_by_id,'historical replacement has no exact quarantine authority')
    p.need(p.pair_identity(dict(row,n_expected=row['n_requests']))==p.pair_identity(original_by_id[hid]),'historical replacement changed workload')
   else:
    p.need(source['cell_id'] in expected_retired and source['cell_id'] not in retired,'replacement omitted/duplicated old group member');retired[source['cell_id']]=cid
  p.need(expected_retired==set(retired)-retired_before,'replacement did not replace complete original group')
  p.need(set(d['retired_unexecuted_original_newrate_repeat2'])=={x['cell_id'] for x in d['original_newrate_group_rows'] if x['repeat']==2},'retired original repetition list differs')
  groups.append(reference)
 return dict(quarantine=quarantine,retired=retired,fresh=fresh,groups=groups,sources=sources)


def annotate(point,overlay):
 cid=point['cell_id'];valid=point.get('measurement_valid') is True
 point.update(raw_arithmetic_verified=valid,scientific_comparison_eligible=valid,selected_for_final_comparison=True,required_baseline_execution=True)
 if cid in overlay['retired']:
  point.update(selected_for_final_comparison=False,required_baseline_execution=False,replacement_cell_id=overlay['retired'][cid],baseline_scope_status='archived_whole_group_replacement')
 if cid in overlay['quarantine']:
  point.update(scientific_comparison_eligible=False,selected_for_final_comparison=False,required_baseline_execution=False,engineering_quarantine=overlay['quarantine'][cid]['evidence'],baseline_scope_status='engineering_quarantined',original_observation_status=point.get('status'))
 if cid in overlay['fresh']:
  item=overlay['fresh'][cid]
  original_rate=item.get('existing_original_rate',item['row'].get('existing_original_rate',False))
  point.update(baseline_scope_status='required_whole_group_replacement',replacement_declaration=item['declaration'],
      existing_original_rate=original_rate,new_rate=not original_rate,
      reuse_original_rate_first_repeat=item.get('reuse_original_rate_first_repeat',False))
 return point


def eligible(point):
 return point.get('measurement_valid') is True and point.get('scientific_comparison_eligible') is True and point.get('selected_for_final_comparison') is True


def audit_quarantined_original(audit,point,entry):
 p=audit.p;cp=p.checked(entry['checkpoint']);receipt=p.checked(entry['receipt']);summary=receipt['summary'];row=cp['row'];directory=Path(entry['receipt']['path']).parents[2]/'cells'/point['cell_id']
 for path,digest in cp['artifacts'].items():p.need(p.sha(path)==digest,'quarantined original artifact changed')
 trace=dict(path=row['trace_path'],sha256=row['trace_sha256']);proof=audit.audit_raw(summary,directory,dict(trace=trace,original_point=point))
 additional=audit.raw_metrics.audit_additional_metrics(summary,directory)
 p.need(point['energy_j']==summary['energy_j'] and point['slo_attainment']==summary['slo_attainment'] and point['work_complete']==summary['work_complete'],'original quarantine metrics changed')
 point.update(raw_arithmetic_verified=True,quarantine_raw_verification=proof)
 point.update(additional['normalized_metrics'])
 return {**cp['artifacts'],entry['checkpoint']['path']:entry['checkpoint']['sha256'],entry['receipt']['path']:entry['receipt']['sha256']}


def verify_fresh_identity(p,point,overlay,originals):
 if point['cell_id'] not in overlay['fresh'] or not point['measurement_valid']:return
 expected=overlay['fresh'][point['cell_id']]
 p.need(point['implementation_id']==expected['host_release'] and point['host_manifest_sha256']==expected['host_manifest']['sha256'],'replacement ran another source release')
 candidates=[x for x in originals if x['model']==point['model'] and x['system']==point['system'] and x['dataset']==point['dataset']]
 identities={(x['profile_sha256'],x['policy_sha256']) for x in candidates}
 p.need(len(identities)==1 and (point['profile_sha256'],point['policy_sha256']) in identities,'cooperative replacement changed original policy/profile')


def required_baselines(points):
 return [x for x in points if x.get('required_baseline_execution',True) and x.get('selected_for_final_comparison',True)]
