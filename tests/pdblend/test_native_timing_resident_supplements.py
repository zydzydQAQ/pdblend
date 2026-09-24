"""NativeSpec epoch adoption and all-device cleanup before lease release."""
import asyncio
import json
from types import SimpleNamespace

import pytest

from pdblend_runtime.probe import NativeSpec
from pdblend.profile.collection.native_timing_collect import adopt_warmup_generations, verify_compute_empty, resident_specs


def inventory():
    specs=[NativeSpec(f'i{i}',(i,),17000+i,'Qwen2.5-7B-Instruct',max_num_seqs=32) for i in range(8)]
    fleet={s.instance_id:SimpleNamespace(spec=s) for s in specs}
    receipts=[dict(instance_id=s.instance_id,generation=1,control=dict(acknowledged=True,generation=1)) for s in specs]
    return specs,fleet,receipts


def test_real_warmup_ack_updates_runtime_epoch_without_rewriting_launch_command():
    specs,fleet,receipts=inventory()
    commands=[s.command() for s in specs]
    updated=adopt_warmup_generations(specs,fleet,receipts)
    assert all(s.generation==1 and fleet[s.instance_id].spec==s for s in updated)
    assert [s.command() for s in updated]==commands
    assert all(s.generation==0 for s in specs)


@pytest.mark.parametrize('fault',['missing','duplicate','ack','mixed_epoch','nonadvancing'])
def test_bad_warmup_cannot_partially_update_any_resident_spec(fault):
    specs,fleet,receipts=inventory()
    if fault=='missing':receipts.pop()
    elif fault=='duplicate':receipts[-1]=receipts[0]
    elif fault=='ack':receipts[-1]['control']['acknowledged']=False
    elif fault=='mixed_epoch':receipts[-1].update(generation=2,control=dict(acknowledged=True,generation=2))
    elif fault=='nonadvancing':receipts[-1].update(generation=0,control=dict(acknowledged=True,generation=0))
    with pytest.raises(ValueError):adopt_warmup_generations(specs,fleet,receipts)
    assert all(fleet[s.instance_id].spec==s for s in specs)


@pytest.mark.parametrize('fault',[None,'orphan','missing_uuid','wrong_uuid','server_alive'])
def test_cleanup_requires_eight_verified_physical_devices_without_processes(fault):
    calls=[]
    def processes(gpu):
        calls.append(gpu)
        return [SimpleNamespace(pid=123)] if fault=='orphan' and gpu==7 else []
    backend=SimpleNamespace(gpu_uuid=lambda g:f'GPU-{g}',_handle=lambda g:g,
        _nvml=SimpleNamespace(nvmlDeviceGetComputeRunningProcesses=processes))
    meter=SimpleNamespace(gpus=list(range(8)),backend=backend)
    fleet=SimpleNamespace(instances={'server':SimpleNamespace(alive=lambda:fault=='server_alive')})
    uuids=[f'GPU-{g}' for g in range(8)]
    if fault=='missing_uuid':uuids.pop()
    elif fault=='wrong_uuid':uuids[-1]='GPU-other'
    result=asyncio.run(verify_compute_empty(fleet,meter,uuids,timeout_s=0))
    assert result['passed'] is (fault is None)
    if fault is None:assert calls==list(range(8))
    else:assert result['error']


@pytest.mark.parametrize('model,tp',[('7B',1),('14B',1),('32B',2)])
def test_model_owned_resident_inventory_partitions_all_physical_devices(model,tp):
    plan=dict(schema='pdblend-native-timing-plan/v2',model_id='Qwen2.5-'+model+'-Instruct',tp=tp,pp=1,
              resident_instances=8//tp)
    args=SimpleNamespace(gpus=[7,5,3,1,6,4,2,0],model='/models/'+plan['model_id'],base_port=17000)
    specs=resident_specs(args,plan)
    assert [g for spec in specs for g in spec.gpus]==args.gpus
    assert len(specs)==8//tp and all(s.tp==tp and s.max_num_seqs==32 for s in specs)
    assert [s.instance_id for s in specs]==[f'pd-timing-{i}' for i in range(8//tp)]
    plan['tp']=3-tp
    with pytest.raises(ValueError):resident_specs(args,plan)


from test_native_timing_capacity import plans,capacity_fixture


@pytest.mark.parametrize('kind',['unsupported','supported','timeout','busy','stale'])
def test_collector_capacity_check_never_submits_or_masks_operational_errors(tmp_path,monkeypatch,plans,kind):
    from pdblend.profile.collection import native_timing_collect as module
    from pdblend_runtime import probe
    plan=plans['plans']['Qwen2.5-32B-Instruct']
    point,receipts=capacity_fixture(plan,total=10**7 if kind=='supported' else 1024)
    calls=[]
    async def call(session,url,route,payload=None):
        calls.append(route)
        if kind=='timeout':raise TimeoutError('real drain timeout')
        value=receipts['drain'] if route=='/baseline/drain' else receipts['capability']
        if kind=='busy' and route=='/baseline/capability':value['state']['running']=['uncompleted']
        return value
    monkeypatch.setattr(probe,'call',call)
    monkeypatch.setattr(module.time,'time',lambda:110. if kind=='stale' else 100.2)
    path=tmp_path/'raw.json';spec=SimpleNamespace(base_url='http://local')
    invocation=module.capacity_before_window(spec,point,path,plan,receipts['identity'])
    if kind in ('timeout','busy','stale'):
        with pytest.raises((ValueError,TimeoutError)):asyncio.run(invocation)
        assert not path.exists()
    else:
        result=asyncio.run(invocation)
        assert (result is None)==(kind=='supported')
        assert path.exists()==(kind=='unsupported')
        if result is not None:assert result['client_requests']==[] and not result['measurement_started']
    assert set(calls)<= {'/baseline/drain','/baseline/capability'}


@pytest.mark.parametrize('fault',[None,'different_binding','missing_flag','model','topology','ledger','provenance'])
def test_request_cycle_invocation_binds_exact_timing_inventory(tmp_path,monkeypatch,fault):
    from pdblend.profile.collection import native_serving_cycles as cycles
    from pdblend.profile.collection.native_timing_collect import validate_cycle_invocation
    from pdblend.profile.collection.native_timing_plan import binding
    plan=dict(model_id='Qwen2.5-7B-Instruct',tp=1,pp=1,query_ledger={'sha256':'ledger'},
              query_bindings={'sha256':'provenance'})
    cycle=dict(plan,query_provenance=plan['query_bindings'])
    if fault=='model':cycle['model_id']='Qwen2.5-14B-Instruct'
    if fault=='topology':cycle['tp']=2
    if fault=='ledger':cycle['query_ledger']={'sha256':'other'}
    if fault=='provenance':cycle['query_provenance']={'sha256':'other'}
    path=tmp_path/'plan.json';path.write_text(json.dumps(cycle))
    args=SimpleNamespace(request_cycle_plan=None if fault=='missing_flag' else path)
    expected=dict(request_cycle_plan=binding(path))
    if fault=='different_binding':expected['request_cycle_plan']['sha256']='0'*64
    monkeypatch.setattr(cycles,'validate_cycle_plan',lambda value:value)
    if fault:
        with pytest.raises(ValueError):validate_cycle_invocation(args,plan,expected)
    else:assert validate_cycle_invocation(args,plan,expected)==cycle


@pytest.mark.parametrize('kind',['qualified','model_gap','unsafe','operational','contradiction','drain_failure'])
def test_cycle_supplement_preserves_fleet_and_requires_fresh_safe_boundary(tmp_path,monkeypatch,kind):
    from pdblend.profile.collection import native_serving_cycles as cycles
    from pdblend.profile.collection.native_timing_collect import collect_cycle_supplement
    from pdblend_baselines import resident_campaign
    path=tmp_path/'plan.json';path.write_text('{}')
    args=SimpleNamespace(out=tmp_path,request_cycle_plan=path)
    specs,fleet,meter,sampler=object(),object(),object(),object();calls=[]
    result=dict(ready_for_timing=True,safe_restore_passed=True,operational_failure=False,
                component_qualified=kind=='qualified')
    if kind=='unsafe':result.update(ready_for_timing=False,safe_restore_passed=False)
    if kind=='operational':result.update(ready_for_timing=False,operational_failure=True)
    if kind=='contradiction':result['operational_failure']=True
    async def collect(got_specs,got_fleet,got_meter,got_sampler,out,**kwargs):
        assert (got_specs,got_fleet,got_meter,got_sampler)==(specs,fleet,meter,sampler)
        assert kwargs['gpu_uuids']==[f'GPU-{i}' for i in range(8)]
        calls.append('collect');out.mkdir();(out/'completion.json').write_text(json.dumps(result))
        return result
    async def capabilities(got_specs):
        assert got_specs is specs;calls.append('capability');return {'fresh':True}
    async def drains(got_specs):
        assert got_specs is specs;calls.append('drain')
        if kind=='drain_failure':raise RuntimeError('rank not empty')
        return {'fresh':True}
    monkeypatch.setattr(cycles,'collect_and_fit_serving_cycles',collect)
    monkeypatch.setattr(resident_campaign,'verify_endpoints',capabilities)
    monkeypatch.setattr(resident_campaign,'drain_endpoints',drains)
    monkeypatch.setenv('PDBLEND_GPU_UUIDS',','.join(f'GPU-{i}' for i in range(8)))
    report={};operation=collect_cycle_supplement(args,specs,fleet,meter,sampler,report)
    if kind in ('qualified','model_gap'):
        asyncio.run(operation)
        assert calls==['collect','capability','drain']
        assert report['post_cycle_drains']=={'fresh':True}
    else:
        with pytest.raises((ValueError,RuntimeError)):asyncio.run(operation)
        assert 'post_cycle_drains' not in report
    assert report['resident_request_cycles']


@pytest.mark.parametrize('fault',[None,'binding','missing_flag','runtime','v1','pilot','cycle','model','tp','ledger','provenance'])
def test_layout_invocation_is_explicit_and_shares_exact_timing_inputs(tmp_path,monkeypatch,fault):
    from pdblend.profile.collection import native_layout_energy
    from pdblend.profile.collection.native_timing_collect import validate_layout_invocation
    from pdblend.profile.collection.native_timing_plan import binding
    plan=dict(schema='pdblend-native-timing-plan/v2',model_id='Qwen2.5-32B-Instruct',tp=2,pp=1,
              query_ledger={'sha256':'ledger'},query_provenance={'sha256':'provenance'})
    layout=dict(plan)
    if fault=='model':layout['model_id']='Qwen2.5-7B-Instruct'
    if fault=='tp':layout['tp']=1
    if fault=='ledger':layout['query_ledger']={'sha256':'other'}
    if fault=='provenance':layout['query_provenance']={'sha256':'other'}
    if fault=='v1':plan['schema']='pdblend-native-timing-plan-v1'
    path=tmp_path/'plan.json';path.write_text(json.dumps(layout))
    args=SimpleNamespace(layout_energy_plan=None if fault=='missing_flag' else path,
        collect_runtime=fault!='runtime',power_pilot_plan='pilot' if fault=='pilot' else None,
        request_cycle_plan='cycle' if fault=='cycle' else None)
    expected=dict(layout_energy_plan=binding(path))
    if fault=='binding':expected['layout_energy_plan']['sha256']='0'*64
    monkeypatch.setattr(native_layout_energy,'validate_layout_plan',lambda value:value)
    if fault:
        with pytest.raises(ValueError):validate_layout_invocation(args,plan,expected)
    else:assert validate_layout_invocation(args,plan,expected)==layout


@pytest.mark.parametrize('kind',['qualified','model_gap','timing_first','timing_holdout','clock_holdout','unsafe','operational','stage_tamper','source_tamper','drain_failure'])
def test_layout_bridge_reuses_inventory_and_never_invents_completed_lease(tmp_path,monkeypatch,kind):
    from pdblend.profile.collection import native_layout_stage,native_layout_energy
    from pdblend.profile.query import native_layout_profile
    from pdblend.profile.query.native_composition import IDENTITY
    from pdblend.profile.collection.native_timing_collect import collect_layout_supplement
    from pdblend.profile.collection.native_timing_plan import binding
    from pdblend_baselines import resident_campaign
    out=tmp_path/'native-timing';out.mkdir()
    def put(path,value):path.write_text(json.dumps(value));return binding(path)
    manifest=put(tmp_path/'manifest.json',{'immutable':True})
    inputs=put(tmp_path/'inputs.json',{'source_manifest':{'path':'source','sha256':'source'}})
    identity={k:k for k in IDENTITY};component=put(out/'timing-component.json',{'component':{'identity':identity}})
    layout=put(tmp_path/'layout.json',{})
    args=SimpleNamespace(out=out,input_manifest=tmp_path/'inputs.json',layout_energy_plan=tmp_path/'layout.json')
    args.timing_first=kind=='timing_first'
    specs,fleet,meter,sampler=object(),object(),object(),object();calls=[]
    result=dict(status='passed' if kind=='qualified' else 'holdout_failed',ready_for_next=True,
        safe_restore_passed=kind!='unsafe',operational_failure=kind=='operational')
    stage={'path':'stage','sha256':'stage'};selection={'path':'selection','sha256':'selection'}
    report=dict(component_qualified=kind!='timing_holdout',timing_component=component,resident_runtime={'sha256':'runtime'},
                final_drains=['real previous drain'])
    if kind=='timing_first':report['resident_timing_stage']={'path':'independent-timing-stage','sha256':'kept'}
    def capture(got,**kwargs):
        calls.append('stage');assert got is report and kwargs['specs'] is specs and kwargs['fleet'] is fleet
        assert kwargs['attempt_manifest_ref']==manifest and kwargs['input_manifest_ref']==inputs
        assert 'physical_cleanup' not in got
        if kind=='stage_tamper':raise ValueError('stage bytes differ')
        return stage
    def freeze(**kwargs):
        calls.append('selection');assert kwargs['timing_ref']==stage and kwargs['identity']==identity
        assert kwargs['runtime_ref']==report['resident_runtime']
        if kind=='clock_holdout':raise native_layout_profile.LayoutQualificationUnavailable(
            'scoped_clock_holdout','required clock holdout failed',evidence={'runtime':report['resident_runtime']})
        if kind=='source_tamper':raise ValueError('source bytes differ')
        return selection
    async def collect(gs,gf,gm,gp,path,**kwargs):
        calls.append('collect');assert (gs,gf,gm,gp)==(specs,fleet,meter,sampler)
        assert kwargs['timing_profile_ref']==selection and kwargs['gpu_uuids']==[f'GPU-{i}' for i in range(8)]
        path.mkdir();put(path/'completion.json',result);return result
    async def capabilities(gs):assert gs is specs;calls.append('capability');return {'fresh':True}
    async def drains(gs):
        assert gs is specs;calls.append('drain')
        if kind=='drain_failure':raise RuntimeError('rank not empty')
        return {'fresh':True}
    monkeypatch.setattr(native_layout_stage,'capture_resident_timing',capture)
    monkeypatch.setattr(native_layout_profile,'freeze_layout_timing_selection',freeze)
    monkeypatch.setattr(native_layout_energy,'collect_and_replay_layout_energy',collect)
    monkeypatch.setattr(resident_campaign,'verify_endpoints',capabilities)
    monkeypatch.setattr(resident_campaign,'drain_endpoints',drains)
    monkeypatch.setenv('PDBLEND_GPU_UUIDS',','.join(f'GPU-{i}' for i in range(8)))
    op=collect_layout_supplement(args,specs,fleet,meter,sampler,report)
    if kind=='timing_holdout':
        asyncio.run(op);assert not calls and report['resident_layout_energy_status']['status']=='blocked'
    elif kind=='clock_holdout':
        asyncio.run(op);assert calls==['stage','selection']
        assert report['resident_layout_energy_status']['gate']=='scoped_clock_holdout'
        assert 'resident_layout_energy' not in report
    elif kind in ('qualified','model_gap','timing_first'):
        asyncio.run(op);assert calls==['stage','selection','collect','capability','drain']
        assert report['post_layout_drains']=={'fresh':True}
        if kind=='timing_first':
            assert report['resident_timing_stage']=={'path':'independent-timing-stage','sha256':'kept'}
            assert report['resident_layout_timing_stage']==stage
    else:
        with pytest.raises((ValueError,RuntimeError)):asyncio.run(op)
        assert 'post_layout_drains' not in report
