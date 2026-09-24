import asyncio
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path

import pytest

from pdblend.bench import comparison_pdblend_observation as obs
from pdblend.bench.comparison_campaign import binding
from pdblend.bench.comparison_runtime import pdblend_window_resources
from pdblend.bench.independent_dispatch import Resources, execute
from pdblend.bench.resident_session import ResidentGroupSession, digest, write_new
from test_comparison_runtime_inputs import prepared, change_choice, FIXTURE
from test_comparison_pdblend_acceptance import fixture, raw, write
from test_resident_comparison import Adapter, group


def observation_point(root):
    point, specs = prepared(root, qualified=False)
    point.update(seed=701,duration_s=150,dataset='alpaca',rate_rps=1.,slo=dict(ttft_s=1.,tpot_s=.1),
        observation_scope=obs.SCOPE,qualification_mode=obs.SCOPE)
    inputs=point['inputs']
    profile=Path(inputs['profiles'][0]['path']);pv=json.loads(profile.read_text())
    pv.update(system='pdblend',model='/models/'+point['model_id'])
    profile.write_text(json.dumps(pv));inputs['profiles']=[binding(profile)]
    change_choice(point,lambda c:c.update(profile_sha256=inputs['profiles'][0]['sha256']))
    trace=dict(model_id=point['model_id'],dataset='alpaca',seed=701,duration_s=150,rate_rps=1.,
        slo=point['slo'],selection_split='evaluation',requests=[dict(idx=0,arrival_s=1.,prompt=[1,2],max_tokens=16)])
    for name,value in [('trace',trace),('planning_trace',dict(trace,selection_split='tuning',seed=9702))]:
        p=root/(name+'.json');p.write_text(json.dumps(value));inputs[name]=binding(p)
    point['trace']=inputs['trace'];inputs['qualifications']=[]
    cfg=Path(inputs['system_config']['path']);value=json.loads(cfg.read_text())
    value.update(profile=inputs['profiles'][0],profile_usage='development',observation_scope=obs.SCOPE)
    cfg.write_text(json.dumps(value));inputs['system_config']=binding(cfg)
    change_choice(point,lambda c:c.update(planning_trace_sha256=inputs['planning_trace']['sha256']))
    return point,specs


def test_original_profile_numbers_and_flags_unchanged(tmp_path):
    point,specs=observation_point(tmp_path);ref=point['inputs']['profiles'][0]
    original=Path(ref['path']).read_bytes()
    loaded,plan=pdblend_window_resources(point,specs)
    from pdblend.profile.query.model import PerfModel
    expected=PerfModel.from_json(original.decode())
    assert loaded.model.step_seconds(8,512,2520)==expected.step_seconds(8,512,2520)
    assert loaded.qualification['usage']=='development' and not loaded.qualification['formal_eligible']
    assert Path(ref['path']).read_bytes()==original and plan.generation==5
    assert not obs.validate_observation_inputs(point,point['inputs'])['formal_eligible']


@pytest.mark.parametrize('fault',['scope','mode','config','prior','eval','foreign','profile_sha'])
def test_optin_does_not_allow_substitution(tmp_path,fault):
    point,_=observation_point(tmp_path)
    if fault=='scope':point.pop('observation_scope')
    elif fault=='mode':point.pop('qualification_mode')
    elif fault=='config':
        p=Path(point['inputs']['system_config']['path']);v=json.loads(p.read_text());v.pop('profile_usage');p.write_text(json.dumps(v));point['inputs']['system_config']=binding(p)
    elif fault=='prior':change_choice(point,lambda c:c.update(planning_trace_sha256='wrong'))
    elif fault=='eval':change_choice(point,lambda c:c.update(selection_split='evaluation'))
    elif fault=='profile_sha':change_choice(point,lambda c:c.update(profile_sha256='wrong'))
    else:point['system']='mixed'
    with pytest.raises(ValueError):obs.validate_observation_inputs(point,point['inputs'])


def test_unqualified_observation_dispatch_keeps_original_online_policy(tmp_path):
    point,specs=observation_point(tmp_path);loaded,plan=pdblend_window_resources(point,specs)
    seen={}
    async def runner(*args,**kw):seen.update(args=args,kw=kw);return {'ran':True}
    resources=Resources(specs=specs,pd_model=loaded.model,pd_plan=plan,comparison_record_tokens=True)
    result=asyncio.run(execute(point,point['inputs'],resources,tmp_path/'run',runner_overrides={'pdblend':runner}))
    assert result['ran'] and seen['kw']['initial_plan'] is plan
    assert seen['args'][3].name=='pdblend'
    assert seen['kw']['observation_duration_s']==150 and seen['kw']['comparison_record_tokens']
    assert seen['kw']['comparison_wait_initial_plan'] is True
    assert 'fixed_plan' not in seen['kw'] and 'freeze_controller' not in seen['kw']
    assert seen['kw']['planning_trace'][0].prompt==[1,2]


def test_infinite_fallback_is_preserved_but_not_nan(tmp_path):
    point,specs=observation_point(tmp_path)
    change_choice(point,lambda c:c['plan'].update(ttft_s=float('inf'),tpot_s=float('inf')))
    _,plan=pdblend_window_resources(point,specs)
    assert plan.ttft_s==float('inf') and plan.tpot_s==float('inf')
    change_choice(point,lambda c:c['plan'].update(ttft_s=float('nan')))
    with pytest.raises(ValueError,match='estimate'):pdblend_window_resources(point,specs)


@pytest.mark.parametrize('parked,failed,slow',[(False,False,False),(True,False,False),(False,True,False),(False,False,True)])
def test_raw_observation_is_valid_without_promoting_profile(tmp_path,monkeypatch,parked,failed,slow):
    args=fixture(tmp_path,monkeypatch,parked=parked,failed=failed,slow=slow)
    from pdblend.bench import comparison_pdblend_acceptance as formal
    monkeypatch.setattr(obs,'observation_selection',formal._inputs)
    result=obs.audit_observation_window(**args)
    assert result['measurement_evidence_valid'],result['gate_failures']
    assert not result['evidence_valid'] and not result['formal_eligible'] and not result['profile_qualified']
    assert result['slo_pass'] is (not failed and not slow)
    assert obs.REQUIRED_MEASUREMENT_GATES<=set(result['checked_gates'])


@pytest.mark.parametrize('fault',['missing_outcome','missing_token','clock','route','off_pid'])
def test_raw_measurement_failures_remain_invalid(tmp_path,monkeypatch,fault):
    args=fixture(tmp_path,monkeypatch,parked=True)
    from pdblend.bench import comparison_pdblend_acceptance as formal
    monkeypatch.setattr(obs,'observation_selection',formal._inputs)
    if fault=='missing_outcome':write(args,'outcomes',[])
    elif fault=='missing_token':
        rows=raw(args,'outcomes');rows[0]['token_events'].pop();write(args,'outcomes',rows)
    elif fault=='clock':
        rows=raw(args,'frequencies');rows[50][1][0]=900;write(args,'frequencies',rows)
    elif fault=='route':write(args,'routes',[])
    else:
        args['drain']['off_instances'][0]['physical_gpus'][0]['compute_pids']=[123]
        from test_comparison_acceptance import save_change
        save_change(args,'drain',args['drain'])
    result=obs.audit_observation_window(**args)
    assert not result['measurement_evidence_valid'] and result['missing_gates']


def observed_result(point,out):
    write_new(out/'raw.json',{'real_fixture':True})
    refs={name:binding(out/'raw.json') for name in obs.RAW_REFS}
    metrics={'slo_pass':False,'duration_s':150.}
    audit=dict(schema=obs.SCHEMA,scope=obs.SCOPE,point_sha256=digest(point),metrics_sha256=digest(metrics),
        measurement_evidence_valid=True,evidence_valid=False,formal_eligible=False,profile_qualified=False,
        missing_gates=[],gate_failures={},checked_gates=sorted(obs.REQUIRED_MEASUREMENT_GATES),
        profile_missing_gates=list(obs.PROFILE_GAPS),raw_refs=refs,evidence_sha256=digest(refs))
    write_new(out/'observation-acceptance.json',audit)
    return dict(observation_scope=obs.SCOPE,measurement_evidence_valid=True,evidence_valid=False,
        formal_eligible=False,profile_qualified=False,observation_acceptance=audit,metrics=metrics)


def test_session_continues_and_skips_complete_observations_even_slo_failure(tmp_path):
    g=group()
    for p in g['points']:p.update(system='pdblend',observation_scope=obs.SCOPE,qualification_mode=obs.SCOPE)
    class Observed(Adapter):
        async def execute(self,p,out):self.runs+=1;return observed_result(p,out)
    a=Observed();first=tmp_path/'first'
    report=asyncio.run(ResidentGroupSession(g,a,first).run())
    assert report['complete'] and not report['quarantined'] and a.runs==3
    assert report['all_observations_valid'] and report['invalid_observations']==0
    for row in report['windows']:
        receipt=json.loads(Path(row['path']).read_text())
        assert receipt['measurement_evidence_valid'] and not receipt['evidence_valid'] and not receipt['baseline_frozen']
    b=Observed();second=asyncio.run(ResidentGroupSession(g,b,tmp_path/'second',previous=[first]).run())
    assert second['complete'] and len(second['skipped'])==3 and b.runs==b.starts==0
    (first/'windows'/g['points'][0]['name']/'run/raw.json').write_text('{}')
    with pytest.raises(ValueError,match='changed'):
        asyncio.run(ResidentGroupSession(g,Observed(),tmp_path/'third',previous=[first]).run())


@pytest.mark.parametrize('fault',['gate','hash','scope','drain'])
def test_session_observation_failure_quarantines(tmp_path,fault):
    g=group()
    for p in g['points']:p.update(system='pdblend',observation_scope=obs.SCOPE,qualification_mode=obs.SCOPE)
    class Bad(Adapter):
        async def execute(self,p,out):
            self.runs+=1;v=observed_result(p,out)
            if fault=='gate':v['observation_acceptance']['checked_gates'].pop()
            elif fault=='hash':v['observation_acceptance']['metrics_sha256']='wrong'
            elif fault=='scope':v['observation_scope']='other'
            return v
        async def drain(self,p):return {'passed':fault!='drain'}
    a=Bad();report=asyncio.run(ResidentGroupSession(g,a,tmp_path/'bad').run())
    assert not report['complete'] and report['quarantined'] and a.runs==1


@pytest.mark.parametrize('limit',[32,256])
def test_pd_controller_uses_actual_engine_batch_limit(tmp_path,limit):
    from types import SimpleNamespace
    from pdblend.bench.run import _make_controller
    from pdblend.bench.client import Request
    from pdblend.control.policies import get_policy
    from pdblend.planner.pool import Plan,SLO
    from pdblend.profile.query.model import PerfModel
    from pdblend.online.router import Router
    fleet=SimpleNamespace(instances={str(i):SimpleNamespace(spec=SimpleNamespace(max_num_seqs=limit)) for i in range(8)})
    model=PerfModel.from_json(FIXTURE.read_text());router=Router(list(fleet.instances))
    plan=Plan({'M':8},2520,2520,2520,0,100.,1.,.1)
    ctl=_make_controller(fleet,router,None,model,get_policy('pdblend'),SLO(1.,.1),
        [Request(0,0.,[1,2],16),Request(1,1.,[1,2],16)],tmp_path,10.,initial_plan=plan)
    assert ctl.planner.cfg.max_num_seqs==limit and ctl.initial_plan is plan and not ctl.freeze


def test_initial_wait_requires_real_complete_receipt_and_propagates_failure():
    from types import SimpleNamespace
    from pdblend.bench.run import _await_initial_transition
    async def scenario(fail):
        controller=SimpleNamespace(plan_now=None,_log=[])
        done=asyncio.Event()
        async def worker():
            await asyncio.sleep(.01)
            if fail:raise RuntimeError('native drain failed')
            controller.plan_now=object()
            await asyncio.sleep(.06)
            controller._log.append(dict(kind='transition_complete',transition_id='actual',finished_s=12.))
            await done.wait()
        task=asyncio.create_task(worker())
        try:
            value=await _await_initial_transition(controller,task,1.)
            assert value['transition_id']=='actual' and value['transition_finished_s']==12.
        finally:
            done.set();await asyncio.gather(task,return_exceptions=True)
    asyncio.run(scenario(False))
    with pytest.raises(RuntimeError,match='native drain failed'):asyncio.run(scenario(True))


def test_legacy_weight_path_is_normalized_only_for_explicit_observation(tmp_path):
    from pdblend.bench.independent_dispatch import validate
    point,specs=observation_point(tmp_path)
    profile_path=Path(point['inputs']['profiles'][0]['path'])
    original=profile_path.read_bytes()
    assert json.loads(original)['model']=='/models/Qwen2.5-7B-Instruct'
    assert validate(point,point['inputs'])['system']=='pdblend'
    loaded,_=pdblend_window_resources(point,specs)
    assert loaded.model.model=='/models/Qwen2.5-7B-Instruct'
    assert profile_path.read_bytes()==original
    for field in ('observation_scope','qualification_mode'):
        other=deepcopy(point);other.pop(field)
        with pytest.raises(ValueError,match='cross-system or cross-model'):
            validate(other,other['inputs'])
    other=deepcopy(point);other['model_id']='Qwen2.5-14B-Instruct'
    # Normalize only the path spelling, never substitute a model identity.
    data=json.loads(original);data['model']='/models/Qwen2.5-14B-Instruct'
    profile_path.write_text(json.dumps(data));new_ref=binding(profile_path)
    point['inputs']['profiles']=[new_ref]
    with pytest.raises(ValueError,match='cross-system or cross-model'):
        validate(point,point['inputs'])


def recorded_group():
    g=group()
    for p in g['points']:
        p.update(system='pdblend',duration_s=150,result_policy='all_recorded_windows/v1',
                 observation_scope=obs.SCOPE,qualification_mode=obs.SCOPE)
    return g


class RecordedAdapter(Adapter):
    def __init__(self,fault=None):super().__init__();self.fault=fault
    async def execute(self,p,out):
        self.runs+=1
        if self.fault=='execute':raise RuntimeError('native runner failed')
        value=observed_result(p,out)
        value['metrics'].update(service_started_s=100.,service_finished_s=250.,offered_requests=3,
            successful_requests=1,failed_requests=2,energy_service_j=1234.,slo_pass=False)
        value.update(evidence_valid=False,measurement_evidence_valid=False,missing_gates=['pdblend.physical_clocks','pdblend.canonical_metrics'])
        value['acceptance']=dict(evidence_valid=False,formal_eligible=False,missing_gates=value['missing_gates'],
            gate_failures={k:'strict raw diagnostic' for k in value['missing_gates']})
        # The strict observation receipt intentionally still disagrees: the new
        # user policy must keep this diagnosis instead of rerunning the window.
        if self.fault=='metrics':value['metrics'].pop('service_finished_s')
        return value
    async def drain(self,p):return {'passed':self.fault!='drain'}


def test_explicit_recorded_policy_continues_all_audit_failures_and_skips(tmp_path):
    g=recorded_group();a=RecordedAdapter();first=tmp_path/'first'
    report=asyncio.run(ResidentGroupSession(g,a,first).run())
    assert report['complete'] and not report['quarantined'] and a.runs==3 and report['recorded_windows']==3
    assert report['invalid_observations']==3 and not report['all_observations_valid']
    for item in report['windows']:
        row=json.loads(Path(item['path']).read_text());result=row['result']
        assert row['recorded_window_complete'] and row['cleanup_passed']
        assert not row['evidence_valid'] and not row['measurement_evidence_valid'] and not row['baseline_frozen']
        assert result['metrics']['energy_service_j']==1234. and result['metrics']['failed_requests']==2
        assert result['metrics']['slo_pass'] is False and not result['formal_eligible']
        assert row['diagnostics']['strict_audit_missing_gates']==['pdblend.physical_clocks','pdblend.canonical_metrics']
    b=RecordedAdapter();resumed=asyncio.run(ResidentGroupSession(g,b,tmp_path/'second',previous=[first]).run())
    assert resumed['complete'] and len(resumed['skipped'])==3 and b.runs==0
    (first/'windows'/g['points'][0]['name']/'run/raw.json').write_text('{}')
    with pytest.raises(ValueError,match='changed'):
        asyncio.run(ResidentGroupSession(g,RecordedAdapter(),tmp_path/'third',previous=[first]).run())


@pytest.mark.parametrize('fault',['execute','drain','metrics','no_optin'])
def test_recorded_policy_still_requires_execution_window_and_drain(tmp_path,fault):
    g=recorded_group()
    if fault=='no_optin':
        for p in g['points']:p.pop('result_policy')
    adapter=RecordedAdapter(fault);report=asyncio.run(ResidentGroupSession(g,adapter,tmp_path/'failed').run())
    assert not report['complete'] and report['quarantined'] and adapter.runs==1
