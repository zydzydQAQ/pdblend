from copy import deepcopy
import asyncio
import itertools
import json
import time
from types import SimpleNamespace

import pytest
from ecopadg.serving.role_profiling import costs,verify_change,ROLES,PURPOSE,ENERGY_BOUND
from test_transfer_power import instant


def state(role,generation):
    return dict(role=role,generation=generation,acknowledged_generation=generation,mode='continuous',
        admit_prefill=True,admit_decode=True,accepting=True,running=0,waiting=0,active=0,
        kv_allocations={},transfer_allocations={})


def fixture():
    raw=dict(instant([(1,[50]*8),(2,[50]*8)]),complete=True,passed=True,purpose=PURPOSE,errors=[],
        power_limit_w=[350]*8,energy_upper_method=ENERGY_BOUND,
        instance=dict(id='i0',tp=1,gpus=[0]),provenance_before=[dict(pid=1,image_id='frozen',instance_id='i0',tp=1,cuda_visible_devices='0')],
        provenance_after=[dict(pid=1,image_id='frozen',instance_id='i0',tp=1,cuda_visible_devices='0')],original_control=dict(role='mixed',mode='continuous',admit_prefill=True,admit_decode=True),
        switches=[],restoration=dict(before=state('decode',50),after=state('mixed',51)))
    for index,(source,target) in enumerate(itertools.permutations(ROLES,2)):
        for trial in (0,1):
            started=1.1+index*.1+trial*.02;finished=started+.001
            raw['switches'].append(dict(tp=1,source_role=source,target_role=target,trial=trial,
                before=state(source,index*4+trial*2),after=state(target,index*4+trial*2+1),
                started_s=started,finished_s=finished,energy_j=400*(finished-started)))
    return raw


def test_role_costs_use_instant_integral_plus_explicit_short_window_limit_bound():
    raw=fixture();result=costs(raw,'raw-digest')
    assert len(result)==6
    assert {(c['source_role'],c['target_role']) for c in result}==set(itertools.permutations(ROLES,2))
    for cost in result:
        assert cost['time_upper_s']==pytest.approx(.0012)
        assert cost['energy_upper_j']==pytest.approx(3.36)
        assert cost['energy_upper_j']>1.2*max(s['energy_j'] for s in raw['switches'])
        assert cost['source_sha256']=='raw-digest'


@pytest.mark.parametrize('defect',['average','metadata','missing_trial','duplicate_trial','generation','kv','source','tp','integral','limits','restoration'])
def test_invalid_role_cost_evidence_never_builds_an_online_cost(defect):
    raw=fixture()
    if defect=='average': raw['power_source']['mode']='average'
    if defect=='metadata': raw['power_metadata'].pop()
    if defect=='missing_trial': raw['switches'].pop()
    if defect=='duplicate_trial': raw['switches'][1]['trial']=0
    if defect=='generation': raw['switches'][0]['after']['acknowledged_generation']=-1
    if defect=='kv': raw['switches'][0]['before']['transfer_allocations']={'pending':128}
    if defect=='source': raw['provenance_after'][0]['pid']=2
    if defect=='tp': raw['provenance_before'][0]['tp']=raw['provenance_after'][0]['tp']=2
    if defect=='integral': raw['switches'][0]['energy_j']=0
    if defect=='limits': raw['power_limit_w'].pop()
    if defect=='restoration': raw['restoration']['after']['role']='decode'
    with pytest.raises((ValueError,RuntimeError)): costs(raw,'raw-digest')


def test_acknowledgement_does_not_allow_rewinding_a_generation():
    before=state('decode',10);after=state('mixed',9)
    with pytest.raises(RuntimeError,match='acknowledgement'):
        verify_change(before,after,dict(role='mixed',mode='continuous',admit_prefill=True,admit_decode=True))


@pytest.mark.parametrize('failure',[False,True])
def test_light_measurement_sends_no_generation_and_restores_control_after_failure(tmp_path,monkeypatch,failure):
    from ecopadg.serving import role_profiling as role
    live=state('decode',5);calls=[]
    class Hardware:
        def __init__(self,**kwargs): assert kwargs['power_mode']=='instant'
        def power_limit_w(self,gpu): return 350
    class Clocks:
        def __init__(self,*a): pass
        async def set(self,*a,**k): calls.append('clock_set')
        async def close(self): calls.append('clock_restore')
    class Sampler:
        def __init__(self,*a,**k):
            self.error=None;self.frequency_samples=[[1,[2520]*8]]
        def start(self):
            at=time.time();self.samples=[(at-.1,[50]*8),(at,[50]*8)]
        def stop(self): self.samples.append((time.time()+.1,[50]*8))
        @property
        def power_source(self): return instant(self.samples)['power_source']
        @property
        def power_metadata(self): return instant(self.samples)['power_metadata']
    class Profiler:
        def __init__(self,*a): self.count=0
        async def call(self,instance,path):
            assert path=='/runtime'  # This stage must never generate reference tokens.
            return dict(live)
        async def control(self,instance,**changes):
            self.count+=1
            if failure and self.count==3: raise RuntimeError('injected role failure')
            live.update(changes,generation=live['generation']+1)
            live['acknowledged_generation']=live['generation']
        async def provenance(self): return [dict(image_id='frozen',pid=1,instance_id='i0',tp=1,cuda_visible_devices='0')]
    for name,value in [('PynvmlBackend',Hardware),('ClockOwner',Clocks),('PowerSampler',Sampler),('HardwareProfiler',Profiler)]:
        monkeypatch.setattr(role,name,value)
    topology=tmp_path/'topology.json';topology.write_text(json.dumps(dict(prefill=dict(id='i0',tp=1,gpus=[0]))))
    args=SimpleNamespace(topology=topology,instance=None,out=tmp_path/'out',runtime_dir=tmp_path)
    if failure:
        with pytest.raises(ValueError,match='complete instant resident role'):asyncio.run(role.measure(args))
        assert not (args.out/'role_costs.json').exists()
    else:
        assert asyncio.run(role.measure(args))['passed']
        assert len(json.loads((args.out/'role_costs.json').read_text()))==6
    assert live['role']=='decode' and live['generation']>5
    assert live['acknowledged_generation']==live['generation']
    assert calls[-1]=='clock_restore'
    assert json.loads((args.out/'raw.json').read_text())['passed'] is (not failure)
