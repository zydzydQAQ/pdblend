import asyncio
import json
from types import SimpleNamespace

import pytest

from pdblend.bench.independent_dispatch import Resources, REQUIRED_GATES, execute, sha, validate


def prepared(root, system):
    def bind(name, value):
        path = root/name; path.write_text(json.dumps(value))
        return dict(path=str(path), sha256=sha(path))
    point = dict(system=system, seed=701, duration_s=300, model_id='Qwen2.5-7B-Instruct',
        dataset='alpaca', rate_rps=1., slo=dict(ttft_s=1.,tpot_s=.1))
    trace = dict(point, selection_split='evaluation', requests=[dict(idx=0, arrival_s=1., prompt=[1,2], max_tokens=16)])
    inputs = dict(trace=bind('trace.json', trace), system_config=bind('config.json',
        dict(system=system, model_id=point['model_id'], max_batch_size=8)), profiles=[])
    if system != 'mixed':
        inputs['profiles'] = [bind('profile.json',dict(system=system,model_id=point['model_id']))]
    if system in ('pdblend','distserve'):
        inputs['offline_choice'] = bind('choice.json', dict(system=system,model_id=point['model_id'],
            selection_split='tuning',evaluation_used_for_selection=False,
            deployment=dict(pairs=[dict(prefill='p',decode='d')])))
    if system == 'pdblend':
        inputs['planning_trace'] = bind('prior.json',dict(trace, selection_split='tuning'))
    inputs['qualifications'] = [bind(gate+'.json',dict(gate=gate,system=system,
        model_id=point['model_id'], formal_eligible=True, trace_sha256=inputs['trace']['sha256']))
        for gate in REQUIRED_GATES]
    specs = [SimpleNamespace(instance_id=name,base_url='http://'+name,tp=1,pp=1,
                             gpus=(i,),zmq_address='localhost:'+str(1000+i)) for i,name in enumerate(('p','d'))]
    return point,inputs,Resources(specs=specs)


@pytest.mark.parametrize('system',['mixed','ecoserve','distserve'])
def test_baseline_dispatch_never_constructs_shared_legacy_policy(tmp_path, monkeypatch, system):
    point,inputs,resources = prepared(tmp_path,system)
    monkeypatch.setattr('pdblend.control.policies.get_policy',lambda *a:pytest.fail('legacy policy used'))
    calls = []
    async def native(*args,**kwargs):
        calls.append((args,kwargs));return dict(native=system)
    result = asyncio.run(execute(point,inputs,resources,tmp_path/'out',runner_overrides={system:native}))
    assert result['native']==system and len(calls)==1


def test_missing_formal_gate_prevents_any_native_execution(tmp_path):
    point,inputs,resources = prepared(tmp_path,'mixed')
    inputs['qualifications'].pop()
    async def native(*args,**kwargs):pytest.fail('unqualified hardware was invoked')
    with pytest.raises(ValueError,match='inconclusive'):
        asyncio.run(execute(point,inputs,resources,tmp_path/'out',runner_overrides={'mixed':native}))


def test_cross_system_profile_and_evaluation_selected_layout_are_rejected(tmp_path):
    point,inputs,resources = prepared(tmp_path,'pdblend')
    from pathlib import Path
    path=Path(inputs['profiles'][0]['path'])
    path.write_text(json.dumps(dict(system='dynamollm',model_id=point['model_id'])))
    inputs['profiles'][0]['sha256']=sha(path)
    with pytest.raises(ValueError,match='cross-system'):
        validate(point,inputs)
    path.write_text(json.dumps(dict(system='pdblend',model_id=point['model_id'])))
    inputs['profiles'][0]['sha256']=sha(path)
    choice=Path(inputs['offline_choice']['path']); value=json.loads(choice.read_text())
    value['selection_split']='evaluation';choice.write_text(json.dumps(value))
    inputs['offline_choice']['sha256']=sha(choice)
    with pytest.raises(ValueError,match='calibration/tuning'):
        validate(point,inputs)
