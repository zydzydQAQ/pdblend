"""CPU-only synthetic frequency revisions; no synthetic hardware qualification."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from pdblend.profile.collection import native_frequency_domain as domain
from pdblend.profile.collection.native_timing_plan import binding
from pdblend.profile.collection.native_timing_plan_v2 import build_plan,validate_plan
from pdblend.profile.collection.native_timing_audit import audit_window,fit_component
from pdblend.profile.collection.native_timing_replay_v2 import _plan
from pdblend.profile.collection.native_timing_replay import Resolver
from pdblend.profile.query.native_timing import NativeTimingOverlay
from pdblend.profile.query.native_composition import _identity
from test_native_timing_capacity import plans,put
from test_pdblend_native_timing import component_rows,raw_fixture


def design():
    return domain.make_domain(model_id='Qwen2.5-7B-Instruct',high_mhz=2100,revision='pd-timing-frequency-2100-v1')


@pytest.mark.parametrize('change',[
    lambda d:d.update(model_id='Qwen2.5-32B-Instruct'),
    lambda d:d.update(observed_tolerance_mhz=120),
    lambda d:d.update(maximum_observation_gap_s=5),
    lambda d:d.update(hardware_qualified=True),
    lambda d:d.update(min_m_instances=2),
    lambda d:d.update(frequencies_mhz=[1500,2520]),
])
def test_domain_cannot_relax_frozen_policy_or_claim_hardware_qualification(change):
    value=design();change(value)
    with pytest.raises(ValueError):domain.validate_domain(value)


def test_legacy_default_and_explicit_bound_identity_do_not_mix():
    old=dict(model_id=design()['model_id'],tp=1,pp=1)
    new=domain.with_domain(old,design())
    assert domain.identity_frequencies(old)==(1500,2520)
    assert domain.identity_frequencies(new)==(1500,2100)
    domain.require_same_domain(new,deepcopy(new))
    with pytest.raises(ValueError,match='cannot be mixed'):domain.require_same_domain(old,new)
    with pytest.raises(ValueError,match='cannot be mixed'):_identity(new,old)
    new.pop('frequency_domain')
    with pytest.raises(ValueError,match='partial'):domain.identity_frequencies(new)


def new_plan(tmp_path,plans):
    # Synthetic query calls are test fixtures only. The production script
    # re-executes actual PoolPlanner calls on bound independent tuning traces.
    model=design()['model_id'];ref=put(tmp_path/'domain.json',design())
    ledger=json.loads(open(plans['ledger']['path']).read())
    ledger['ledgers']=[r for r in ledger['ledgers'] if r['model_id']==model]
    ledger.update(frequency_domain_ref=ref,**domain.domain_fields(design()))
    for row in ledger['ledgers']:
        row.update(frequency_scope=[1500,2100],frequency_domain_sha256=ledger['frequency_domain_sha256'])
        for q in row['queries']:
            if q['args'][-1]==2520:q['args'][-1]=2100
    ledger_ref=put(tmp_path/'new-ledger.json',ledger)
    provenance=put(tmp_path/'new-provenance.json',dict(evaluation_read=False,outputs={'ledger':ledger_ref},
        frequency_domain_ref=ref,**domain.domain_fields(design())))
    return build_plan(ledger_ref,provenance,plans['corpora'][model],model_id=model,frequency_domain_ref=ref)


def test_new_plan_requires_own_bound_actual_query_domain(tmp_path,plans):
    p=new_plan(tmp_path,plans)
    assert validate_plan(p)==p
    assert domain.plan_frequencies(p)==(1500,2100)
    assert {r['frequency_mhz'] for r in p['points']}=={1500,2100}
    assert all(r['frequency_domain_sha256']==p['frequency_domain_sha256'] for r in p['points'])
    with pytest.raises(ValueError,match='freshly executed'):
        build_plan(plans['ledger'],plans['provenance'],plans['corpora'][p['model_id']],
            model_id=p['model_id'],frequency_domain_ref=p['frequency_domain_ref'])
    bad=deepcopy(p);bad['points'][0]['frequency_domain_sha256']='old'
    with pytest.raises(ValueError):validate_plan(bad)


@pytest.mark.parametrize('supplement',['collect_runtime','power_pilot','request_cycles','layout_energy'])
def test_unintegrated_new_domain_supplements_fail_before_loading(tmp_path,plans,supplement):
    p=new_plan(tmp_path,plans);inputs=dict(p,timing_first=True)
    assert domain.validate_collection_inputs(p,inputs)==(1500,2100)
    with pytest.raises(ValueError,match='before GPU load'):
        domain.validate_collection_inputs(p,inputs,**{supplement:True})


def test_replay_rebuild_binds_inputs_and_report_domain(tmp_path,plans):
    p=new_plan(tmp_path,plans);inputs=dict(p,point_plan=put(tmp_path/'point-plan.json',p),timing_first=True)
    assert _plan(inputs,Resolver())==p
    inputs.pop('frequency_domain_sha256')
    with pytest.raises(ValueError,match='partial'):_plan(inputs,Resolver())


def test_new_frequency_raw_keeps_same_strict_clock_and_shape_audit():
    physical,raw=raw_fixture();raw['point'].update(frequency_mhz=2100,**domain.point_fields(domain.with_domain(physical,design())))
    raw['clock_receipt']['requested_frequency_mhz']=2100
    raw['frequency_samples']=[[t,[2100]] for t,_ in raw['frequency_samples']]
    rows=audit_window(raw,identity=physical)
    assert domain.check_rows(rows,domain.with_domain(physical,design()))==(1500,2100)
    raw['frequency_samples'][2][1]=[2055]
    with pytest.raises(ValueError,match='frequency coverage'):audit_window(raw,identity=physical)


def test_fit_and_consumer_refuse_old_rows_or_old_runtime_power_identity():
    train,held=component_rows();identity=dict(system='pdblend',model_id=design()['model_id'],tp=1,pp=1)
    identity=domain.with_domain(identity,design())
    kwargs=dict(identity=identity,raw_bindings=[],measurement_qualification=dict(qualified=True),
        limits=dict(mean_relative_error=.1,p95_relative_error=.2,max_relative_error=.25))
    with pytest.raises(ValueError):fit_component(train,held,**kwargs)
    for row in train+held:
        if row['frequency_mhz']==2520:row['frequency_mhz']=2100
        row['frequency_domain_sha256']=identity['frequency_domain_sha256']
    component=fit_component(train,held,**kwargs)
    assert component['component_qualified'] and not component['formal_eligible']
    replay=dict(component=component,formal_eligible=False)
    base=SimpleNamespace(system='pdblend',model=identity['model_id'],tp=1,pp=1)
    with pytest.raises(ValueError,match='cannot be mixed'):NativeTimingOverlay(base,replay)
    base.calibration_identity=identity
    assert NativeTimingOverlay(base,replay).freqs==(1500,2100)
    held[0].pop('frequency_domain_sha256')
    with pytest.raises(ValueError,match='cannot be relabeled'):fit_component(train,held,**kwargs)


@pytest.mark.parametrize('location,key',[('plan','frequency_domain_ref'),('inputs','frequency_domain_ref'),
    ('inputs','frequency_domain'),('inputs','frequency_domain_sha256')])
def test_legacy_preflight_rejects_partial_or_unbound_domain(location,key):
    plan={};inputs={};(plan if location=='plan' else inputs)[key]='unbound'
    with pytest.raises(ValueError,match='unbound new domain'):domain.validate_collection_inputs(plan,inputs)
