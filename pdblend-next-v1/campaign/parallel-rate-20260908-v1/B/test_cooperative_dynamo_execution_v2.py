"""Meaningful predeclared timing failures, complete-low-SLO allowance and source checks."""
import csv,copy,json,sys
from pathlib import Path
import pytest
B=Path(__file__).resolve().parent;sys.path.insert(0,str(B));import cooperative_dynamo_runner_v2 as r

def values(delays):
 rows=[dict(planned_arrival_s=str(i),actual_dispatch_s=str(i+d),dispatch_delay_s=str(d)) for i,d in enumerate(delays)]
 actual=sorted(float(x['actual_dispatch_s'])-float(x['planned_arrival_s']) for x in rows);pos=.99*(len(actual)-1);lo=int(pos);hi=min(lo+1,len(actual)-1)
 summary=dict(n_expected=len(rows),dispatch_delay_max_s=actual[-1],dispatch_delay_p99_s=actual[lo]+(actual[hi]-actual[lo])*(pos-lo),slo_attainment=.01)
 return rows,summary

def test_complete_low_slo_has_no_timing_exclusion():
 rows,s=values([.001]*100);assert r.timing_values(rows,s)['passed']
def test_max_limit_and_percentile_have_independent_effect():
 rows,s=values([0.]*99+[1.]);assert r.timing_values(rows,s)['passed']
 rows,s=values([0.]*99+[1.001]);g=r.timing_values(rows,s);assert not g['passed'] and len(g['errors'])==1
 rows,s=values([.11]*100);g=r.timing_values(rows,s);assert not g['passed'] and len(g['errors'])==1
def test_old_starvation_max_is_rejected():
 rows,s=values([0.]*99+[101.68]);assert not r.timing_values(rows,s)['passed']
@pytest.mark.parametrize('mutation',[lambda x:x[0].update(actual_dispatch_s=''),lambda x:x[0].update(actual_dispatch_s='nan'),lambda x:x[0].update(actual_dispatch_s='-1'),lambda x:x[0].update(dispatch_delay_s='99'),lambda x:x.pop()])
def test_missing_nonfinite_early_changed_or_omitted_raw_rejected(mutation):
 rows,s=values([.001]*100);mutation(rows)
 with pytest.raises((ValueError,KeyError)):r.timing_values(rows,s)
def test_summary_cannot_cover_changed_actual_raw():
 rows,s=values([.001]*100);s['dispatch_delay_p99_s']=0
 with pytest.raises(ValueError):r.timing_values(rows,s)
def test_actual_retained_good_raw_recomputes_independently():
 cp=next((B/'baselines-reconciled-006/results/checkpoints').glob('*alpaca-r5*'))
 d=json.loads(cp.read_text());rec=json.loads(Path(d['receipt']['path']).read_text());path=Path(d['receipt']['path']).parents[2]/'cells'/rec['cell_id']/'bench.csv'
 assert r.timing_values(list(csv.DictReader(path.open())),rec['summary'])['passed']
def test_host_policy_profile_source_scope_still_exact():
 d,files=r.qualification.host_contract();assert len(d['cells'])==8 and len(files)==134
 assert {(x['dataset'],x['rate_rps_decimal'],x['repeat']) for x in d['cells']}=={(dataset,rate,repeat) for dataset,rate in [('alpaca','4'),('alpaca','5'),('sharegpt','1.25'),('sharegpt','1.5')] for repeat in (1,2)}

@pytest.mark.parametrize('mutation',['none','missing_freeze','wrong_identity','changed_original'])
def test_original_fresh_binding_lineage_is_frozen_and_exact(tmp_path,mutation):
 path=tmp_path/'original.json';path.write_text(json.dumps(dict(instances=[dict(id='b0',host_pid=123)])))
 ref=r.p.ref(path);boot=dict(instances=[dict(id='b0',host_pid=123)],files={ref['path']:ref['sha256']},cooperative_qualification=dict(original_fresh_binding=ref))
 if mutation=='missing_freeze':boot['files'].clear()
 if mutation=='wrong_identity':boot['instances'][0]['host_pid']=124
 if mutation=='changed_original':path.write_text(json.dumps(dict(instances=[dict(id='b0',host_pid=124)])))
 if mutation=='none':assert r.qualification.original_binding_lineage(boot)['instances']==boot['instances']
 else:
  with pytest.raises(ValueError):r.qualification.original_binding_lineage(boot)

@pytest.mark.parametrize('mutation',['none','historical_reintroduced','remaining','observation_count','retired_count','live_owner'])
def test_fixed_slo_handoff_requires_terminal_complete_non_dynamo_scope(tmp_path,monkeypatch,mutation):
 import os
 import baseline_reconciliation_v3 as ledger
 q=r.qualification
 scope=dict(historical_scale11_required_for_this_task=False,historical_scale11_unmeasured=True,cooperative_group=r.p.ref(r.DECL))
 if mutation=='historical_reintroduced':scope['historical_scale11_required_for_this_task']=True
 sp=tmp_path/'scope.json';sp.write_text(json.dumps(scope));monkeypatch.setattr(q,'SCOPE',sp);monkeypatch.setattr(q,'SCOPE_SHA',r.p.sha(sp))
 ld=tmp_path/'ledger.json';ld.write_text('{}')
 state=tmp_path/'status.json';state.write_text(json.dumps(dict(finished_s=1,node_lease_held=False,pid=os.getpid() if mutation=='live_owner' else 99999999)))
 checked=dict(remaining_cells=[] if mutation!='remaining' else ['unmeasured'],already_observed_count=21 if mutation!='observation_count' else 20,retired_unexecuted_cells=[1,2,3] if mutation!='retired_count' else [],prior_stages=[dict(status=r.p.ref(state))])
 declaration=dict(schema='B-cooperative-after-original-nonDyn-fixedSLO-v2',current_scope=r.p.ref(sp),completed_non_dynamo_ledger=r.p.ref(ld))
 reference=q.B/'cooperative-dynamo8-001/qualification-prerequisite.json';original_read=r.p.read;original_ref=r.p.ref
 monkeypatch.setattr(r.p,'read',lambda path:declaration if Path(path)==reference else original_read(path))
 monkeypatch.setattr(r.p,'ref',lambda path:dict(path=str(reference),sha256='cpu-only') if Path(path)==reference else original_ref(path))
 monkeypatch.setattr(ledger,'audit',lambda path:checked)
 if mutation=='none':assert q.prerequisite()['sha256']=='cpu-only'
 else:
  with pytest.raises(ValueError):q.prerequisite()
