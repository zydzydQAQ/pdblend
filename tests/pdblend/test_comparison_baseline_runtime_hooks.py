import asyncio
from copy import deepcopy
from types import SimpleNamespace

import pytest
from pdblend.bench import comparison_baseline_observation as obs
from pdblend.bench import comparison_runtime as runtime
from pdblend.bench import comparison_dynamo_runtime as dynamo
from test_comparison_baseline_observation import prepared


def group(point, identity):
    return dict(engine_identity=identity,points=[point])


def test_observation_dynamo_factory_retains_native_resident_owner(tmp_path,monkeypatch):
    point,identity,_=prepared(tmp_path);calls=[];owner=object()
    def make(out,*,base_port):calls.append((out,base_port));return owner
    monkeypatch.setattr(obs,'make_dynamo_adapter',make)
    assert runtime.make_resident_adapter(group(point,identity),tmp_path/'out',base_port=18000) is owner
    assert calls==[(tmp_path/'out',18000)]


def test_default_dynamo_and_pd_adapter_selection_is_unchanged(tmp_path,monkeypatch):
    point,identity,_=prepared(tmp_path);point.pop('observation_scope')
    sentinel=object();monkeypatch.setattr(dynamo,'DynamoResidentAdapter',lambda *a,**k:sentinel)
    assert runtime.make_resident_adapter(group(point,identity),tmp_path,base_port=18000) is sentinel
    point.update(system='pdblend',observation_scope='pdblend_profile_unqualified_evaluation/v1')
    identity['entrypoint']='pdblend_runtime.serve'
    assert type(runtime.make_resident_adapter(group(point,identity),tmp_path,base_port=18000)) is runtime.NativeResidentAdapter


@pytest.mark.parametrize('fault',['mixed_scope','qualification_mode','policy','foreign_system'])
def test_mixed_or_incomplete_baseline_observation_identity_rejected(tmp_path,fault):
    point,identity,_=prepared(tmp_path);g=group(point,identity)
    if fault=='mixed_scope':g['points'].append(dict(point,observation_scope='formal'))
    elif fault=='foreign_system':point['system']='pdblend'
    elif fault=='policy':point.pop('result_policy')
    else:point.pop('qualification_mode')
    with pytest.raises(ValueError):runtime.make_resident_adapter(g,tmp_path,base_port=18000)


def test_distserve_uses_original_deployment_then_real_drain_and_shared_snapshot(tmp_path,monkeypatch):
    point,identity,_=prepared(tmp_path,system='distserve');calls=[];specs=[object()]
    adapter=runtime.make_resident_adapter(group(point,identity),tmp_path/'session',base_port=18000)
    assert type(adapter) is runtime.NativeResidentAdapter
    adapter.identity=identity;adapter.specs=specs;adapter.qualification={'startup':'bound'};adapter.reset_receipt={'passed':True}
    snapshot={'actual_sampler':'same-session'}
    adapter.monitor=SimpleNamespace(snapshot=lambda:snapshot)
    raw={'system':'distserve','actual':'native'};states=[{'native':'drained'}]
    async def run(p,i,*,out,specs):calls.append(('execute',specs));assert p is point and i is identity;return raw
    async def drain(s):calls.append(('drain',s));return states
    def final(p,i,**kwargs):
        calls.append(('finalize',kwargs));assert kwargs['native_result'] is raw and kwargs['snapshot'] is snapshot
        assert kwargs['drain']['passed'] and kwargs['drain']['states'] is states
        assert kwargs['startup'] is adapter.qualification and kwargs['reset'] is adapter.reset_receipt
        return {'evidence_valid':False,'formal_eligible':False,'metrics':{'duration_s':150}}
    monkeypatch.setattr(obs,'execute_native_observation',run)
    monkeypatch.setattr(obs,'finalize_observation',final)
    monkeypatch.setattr(runtime,'drain_endpoints',drain)
    monkeypatch.setattr(runtime,'dispatch',lambda *a:pytest.fail('formal dispatcher must not intercept observation'))
    result=asyncio.run(adapter._execute_window(point,tmp_path/'run'))
    assert not result['formal_eligible'] and [c[0] for c in calls]==['execute','drain','finalize']
    assert adapter.last_drain['states'] is states
