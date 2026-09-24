"""Synthetic new-domain observations; never transform existing GPU evidence."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from pdblend.profile.collection import native_layout_energy as collect
from pdblend.profile.query import native_layout_model as model,native_layout_profile as profile
from pdblend.profile.collection.native_frequency_domain import make_domain,domain_fields,with_domain,validate_collection_inputs
from pdblend.profile.collection.native_runtime_collect import write_new,build_runtime_plan,NativeRuntimeCollector
from pdblend.profile.collection.native_timing_plan import digest,read_bound
from pdblend.planner.native_layout import NativeLayoutPlanner
from pdblend.planner.pool import PlannerConfig,SLO
from test_native_layout_energy import plan_fixture,layout_raw,candidate,Timing,forecast,runtime_scope_fixture


def domain(tmp_path):
    value=make_domain(model_id=collect.MODEL,high_mhz=2100,revision='pd32-frequency-1500-2100-v1')
    return value,write_new(tmp_path/'pd32-domain.json',value)


def new_plan(tmp_path,monkeypatch):
    old=plan_fixture(tmp_path,monkeypatch);d,ref=domain(tmp_path)
    plan=collect.build_layout_plan(old['query_ledger'],old['query_provenance'],frequency_domain_ref=ref)
    return plan,d,ref


def energy_and_timing(tmp_path):
    d,_=domain(tmp_path);c=candidate();c.update(tp=2,pp=1,revision=collect.DOMAIN_REVISION,**domain_fields(d))
    c['identity']=with_domain(dict(model_id=collect.MODEL,tp=2,pp=1),d)
    for node in c['nodes']:
        if node['frequency_mhz']==2520:node['frequency_mhz']=2100
    timing=Timing();timing.freqs=(1500,2100);timing.calibration_identity=c['identity']
    return model.NativeLayoutEnergyModel(c),timing


def test_32_domain_tp_owned_static_plan_and_no_qualification(tmp_path,monkeypatch):
    plan,d,ref=new_plan(tmp_path,monkeypatch);collect.validate_layout_plan(plan)
    assert d['tp']==2 and plan['revision']==collect.DOMAIN_REVISION
    assert len(plan['points'])==36 and plan['training_service_s']==720 and plan['holdout_service_s']==3600
    assert {p['frequency_mhz'] for p in plan['points']}=={1500,2100}
    assert all(p['frequency_domain_sha256']==digest(d) for p in plan['points'])
    assert plan['frequency_semantics']['observed_tolerance_mhz']==30
    runtime=build_runtime_plan(ref)
    inputs=dict(plan,timing_first=True,runtime_include_transfer=False,runtime_plan=runtime,runtime_scope=runtime['scope'])
    assert validate_collection_inputs(plan,inputs,collect_runtime=True,layout_energy=True)==(1500,2100)
    assert not plan['formal_eligible'] and not d['hardware_qualified']
    for flag in ('power_pilot','request_cycles'):
        with pytest.raises(ValueError):validate_collection_inputs(plan,inputs,collect_runtime=True,layout_energy=True,**{flag:True})
    for tp in (1,4):
        bad=deepcopy(d);bad['tp']=tp
        from pdblend.profile.collection.native_frequency_domain import validate_domain
        with pytest.raises(ValueError):validate_domain(bad)
    with pytest.raises(ValueError):make_domain(model_id=collect.MODEL,high_mhz=2400,revision='unregistered-endpoint')


def test_tp2_runtime_raw_new_domain_replays_every_rank(tmp_path):
    from tests.test_native_runtime_frequency_domain import runtime_fixture,save_rows
    from tests.test_native_runtime_tp2 import tp2_inventory
    from pdblend.profile.collection.native_runtime_audit import replay_runtime
    report,power,rows=runtime_fixture(tmp_path,collect.MODEL);audit=replay_runtime(report)
    assert audit['raw_components_complete'],audit['errors']
    assert len(audit['capacity'])==4 and audit['scoped_runtime_qualified']
    assert not audit['full_profile_qualified'] and not audit['handoff_prediction_qualified']
    specs,fleet,meter,sampler=tp2_inventory()
    runner=NativeRuntimeCollector(specs,fleet,meter,sampler,tmp_path,gpu_uuids=power['gpu_uuids'],
        frequency_domain_ref=report['frequency_domain_ref'])
    assert runner.restore_frequency==2100 and runner.domain['tp']==2
    row=next(r for r in rows if r.get('state')=='active_idle@2100')
    for stamp,values in power['frequency_samples']:
        if row['started_s']<=stamp<row['finished_s']:values[1]=2520
    report['power']=write_new(tmp_path/'wrong-rank-frequency.json',power)
    assert not replay_runtime(report)['raw_components_complete']


@pytest.mark.parametrize('damage',[None,'old_actual_clock','missing_domain_point','old_plan_sha','domain_file_tamper'])
def test_layout_raw_requires_new_bound_domain_and_real_clocks(tmp_path,monkeypatch,damage):
    raw,old=layout_raw(tmp_path,monkeypatch);d,ref=domain(tmp_path)
    plan=collect.build_layout_plan(old['query_ledger'],old['query_provenance'],frequency_domain_ref=ref)
    point=next(p for p in plan['points'] if p['purpose']=='training' and p['dataset']=='alpaca' and p['rate_scale']==.25 and p['frequency_mhz']==2100)
    raw.update(point=point,plan_sha256=digest(plan),trace=collect.layout_trace(plan,point))
    for receipt in raw['clocks'].values():
        receipt['ack']['requested_frequency_mhz']=2100
        for row in receipt['observations']:row['frequencies_mhz']=[2100,2100]
    for _,values in raw['power']['frequency_samples']:values[:]=[2100]*8
    if damage=='old_actual_clock':raw['power']['frequency_samples'][50][1][7]=2520
    elif damage=='missing_domain_point':raw['point']=dict(point);raw['point'].pop('frequency_domain_sha256')
    elif damage=='old_plan_sha':raw['plan_sha256']=digest(old)
    elif damage=='domain_file_tamper':Path(ref['path']).write_text('{}')
    audit=collect.audit_layout_window(raw,plan)
    assert audit['passed'] is (damage is None),audit
    if damage is None:
        assert audit['identity']['frequency_domain_sha256']==digest(d)
        assert audit['revision']==collect.DOMAIN_REVISION and not audit['formal_eligible']


def test_new_planner_full_counts_and_fallback_use_only_new_endpoints(tmp_path):
    energy,timing=energy_and_timing(tmp_path)
    config=PlannerConfig(slots=4,slo=SLO(10,1),freqs=(1500,2100),max_num_seqs=32,min_m_instances=4)
    scope=dict(dataset='alpaca',parent_trace={'path':'alpaca','sha256':'alpaca'},arrival_family='poisson',service_duration_s=150.)
    planner=NativeLayoutPlanner(timing,config,energy,workload_scope=scope)
    plans=planner.candidates(forecast())
    assert len(plans)==2 and {p.f_M for p in plans}=={1500,2100} and not planner.unsupported
    assert all(p.detail['revision']==collect.DOMAIN_REVISION for p in plans)
    assert planner.fallback(forecast()).f_M==2100
    with pytest.raises(ValueError):NativeLayoutPlanner(Timing(),config,energy,workload_scope=scope)
    args=dict(model_id=collect.MODEL,tp=2,pp=1,counts={'M':4},rate_rps=.5,frequency_mhz=2520,**scope)
    with pytest.raises(ValueError):energy.predict_layout_mean_w(**args)
    fc=forecast();fc.backlog=(object(),)
    assert planner.evaluate({'M':4},2100,2100,2100,0,fc) is None


def test_scoped_runtime_loader_requires_component_domain_and_new_clock_holdout(tmp_path,monkeypatch):
    _,resolver,ident,audit=runtime_scope_fixture(monkeypatch);d,ref=domain(tmp_path);ident=with_domain(ident,d)
    report=resolver.read({});report.update(model_id=collect.MODEL,tp=2,pp=1,**domain_fields(d))
    audit['component_identity']=dict(ident,source_revision='s')
    for row in audit['holdout_comparisons']:row['component']=row['component'].replace('2520','2100')
    result=profile.replay_layout_runtime_component({},resolver,ident,['s'])
    base=profile._RuntimeTimingBase(ident,result)
    assert base.freqs==(1500,2100) and base.freq_switch_s==1.
    assert not result['original_global_holdout_passed'] and result['scoped_runtime_qualified']
    audit['component_identity'].pop('frequency_domain_sha256')
    with pytest.raises(ValueError):profile.replay_layout_runtime_component({},resolver,ident,['s'])


def test_new_domain_full_candidate_freeze_replay_and_independent_holdout(tmp_path,monkeypatch):
    from test_native_layout_energy import component_fixture
    _,ref=domain(tmp_path)
    plan,candidate_ref,selection_ref,timing,holder=component_fixture(tmp_path,monkeypatch,ref)
    selected=read_bound(selection_ref);c=read_bound(candidate_ref)
    assert selected['revision']==c['revision']==collect.DOMAIN_REVISION
    assert len(selected['groups'])==12
    assert all({p['f_M'] for p in row['candidates']}=={1500,2100} for row in selected['groups'])
    result=model.replay_layout_component(plan,candidate_ref,selection_ref,{},timing)
    assert result['component_qualified'] and len(result['comparisons'])==24
    assert not result['formal_eligible'] and not result['full_profile_qualified']
    timing.calibration_identity.pop('frequency_domain_sha256')
    with pytest.raises(ValueError):model.replay_layout_component(plan,candidate_ref,selection_ref,{},timing)


def test_collector_restores_new_high_endpoint_on_all_four_tp2_replicas(tmp_path,monkeypatch):
    import asyncio
    from tests.test_native_runtime_tp2 import tp2_inventory
    plan,_,ref=new_plan(tmp_path,monkeypatch);specs,fleet,meter,sampler=tp2_inventory();calls=[]
    class Runner:
        def __init__(self,*args,**kw):assert kw['frequency_domain_ref']==ref
        async def stop_measurement(self,spec):return {}
        async def drain(self,spec):return {}
        async def resume(self,spec):return {}
        async def clock(self,spec,frequency):calls.append((spec.instance_id,spec.tp,frequency));return {}
    async def window(runner,specs,plan,point,path,**kw):
        raw=dict(point=point);write_new(path,raw);return raw
    monkeypatch.setattr(collect,'NativeRuntimeCollector',Runner)
    monkeypatch.setattr(collect,'validate_inventory',lambda *a:{})
    monkeypatch.setattr(collect,'collect_cycle_window',window)
    monkeypatch.setattr(collect,'audit_layout_window',lambda *a:dict(passed=True))
    result=asyncio.run(collect.collect_layout_energy(specs,fleet,meter,sampler,tmp_path/'capture',
        gpu_uuids=['unused']*8,plan=plan))
    assert result['collection_complete'] and result['safe_restore_passed']
    assert len(calls)==4 and all(tp==2 and f==2100 for _,tp,f in calls)
    assert not result['formal_eligible']
