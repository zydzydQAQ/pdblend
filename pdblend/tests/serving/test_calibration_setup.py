import asyncio
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ecopadg.serving import calibration_setup as setup
from ecopadg.serving.topology import InstanceSpec


def fixture_inputs(tmp_path, monkeypatch):
    image = 'sha256:'+'a'*64
    campaign = tmp_path/'campaign'; campaign.mkdir()
    setup.write(campaign/'budget.json',dict(started_s=100,limit_s=86400))
    monkeypatch.setattr(setup.time,'time',lambda:200)
    corpus = tmp_path/'corpus'; corpus.mkdir()
    datasets = {}; artifacts = {}
    for dataset in setup.DATASETS:
        path = corpus/(dataset+'.json')
        setup.write(path,dict(calibration=[dict(input_tokens=20,output_tokens=60)]*128,
                             development='must never enter a decision',formal='must never enter a decision'))
        artifacts[str(path)] = setup.sha256(path)
        mixed = dict(tp=1,instance_count=2,batch=8,capacity_rps=2,gpus=[[0],[1]])
        pd = dict(prefill_tp=1,decode_tp=1,prefill_count=1,decode_count=1,
                  prefill_batch=4,decode_batch=8,capacity_rps=2,gpus=[[0],[1]])
        datasets[dataset] = dict(mixed=[mixed,dict(mixed,instance_count=1,gpus=[[0]],capacity_rps=1)],distserve=[pd])
    search = dict(datasets=datasets,artifacts=artifacts)
    profiles = dict(points=[])
    transfers = dict(instant_power_costs_verified=True,receiver_transfer_energy_included=True,
                     links=[dict(source_tp=1,target_tp=1)])
    template = dict(model=setup.MODEL,max_model_len=8192)
    monkeypatch.setattr(setup,'validated_inputs',lambda m:(profiles,transfers,search,template,[],[],[],dict(artifacts)))
    monkeypatch.setattr(setup,'verify_frozen_costs',lambda *args:None)
    manifest = dict(image=image,campaign_root=str(campaign),profiles=str(tmp_path/'profiles.json'),
        transfers=str(tmp_path/'transfers.json'),interconnect=str(tmp_path/'interconnect.txt'),
        retained_weights=str(tmp_path/'weights'),corpus=str(corpus),calibration_budget_s=10000,
        initial_instances=[],entries=[dict(system=s,dataset='alpaca',candidate_index=0) for s in setup.BASELINES])
    return manifest


def test_generator_groups_physical_engines_but_calibrates_each_policy_independently(tmp_path,monkeypatch):
    manifest = fixture_inputs(tmp_path,monkeypatch)
    result = setup.generate(manifest,tmp_path/'generated')
    stages = setup.read(tmp_path/'generated/campaign.json')['stages']
    assert len(stages)==7  # one prepare, five calibrations, one CPU summary
    assert result['status']=='prepared_not_calibrated'
    assert result['measurement_gpus']==list(range(8))
    assert len(result['selected'])==5 and result['unmeasured']
    assert stages[-2]['name'].startswith('calibrate-dynamollm')
    configurations = {x['system']:setup.read(x['config']) for x in result['selected']}
    assert configurations['distserve']['instances'][0]['role']=='prefill'
    assert configurations['distserve']['instances'][1]['role']=='decode'
    assert all(i['role']=='mixed' for i in configurations['ecoserve']['instances'])
    assert all(x['idle_unallocated_gpus']==list(range(2,8)) for x in result['selected'])
    dyn = configurations['dynamollm']
    assert dyn['topology']['image']==manifest['image']
    assert dyn['dynamo_assignments']=={'cal0_0':'LL','cal0_1':'SS'}
    assert result['selected'][-1]['dynamo_initialization']['periods_s']=={'ScaleInst':1800,'ScaleShard':300,'ScaleFreq':5}
    assert all(config['node_gpus']==list(range(8)) for config in configurations.values())
    assert all(config['power_mode']=='instant' for config in configurations.values())
    evidence = setup.read(tmp_path/'generated/input-evidence.json')['artifacts']
    setup.verify_artifacts(evidence)


def test_generator_does_not_use_development_or_formal_shapes(tmp_path,monkeypatch):
    manifest = fixture_inputs(tmp_path,monkeypatch)
    result = setup.generate(manifest,tmp_path/'generated')
    assert all(setup.read(x['config'])['output_prior']==60 for x in result['selected'])
    for item in result['selected']:
        source=setup.read(Path(item['config']).with_name(Path(item['config']).name.replace('.config.json','.calibration.json')))
        assert source['entries'][0]['initial_rate']==1.2
        assert 'development' not in source and 'formal' not in source


@pytest.mark.parametrize('change,match',[
    ({'calibration_budget_s':100},'upper bounds'),
    ({'calibration_budget_s':86400},'effective authorized campaign'),
    ({'target':.95},'99% SLO'),
    ({'max_trials':80},'seven probes'),
])
def test_generator_preserves_original_deadline_and_calibration_protocol(tmp_path,monkeypatch,change,match):
    manifest = fixture_inputs(tmp_path,monkeypatch);manifest.update(change)
    with pytest.raises(ValueError,match=match): setup.generate(manifest,tmp_path/'generated')
    assert not (tmp_path/'generated').exists()


def test_generator_rejects_pdb_derived_baseline_and_unavailable_candidate(tmp_path,monkeypatch):
    manifest = fixture_inputs(tmp_path,monkeypatch)
    manifest['entries'][0]['system']='pdblend-joint'
    with pytest.raises(ValueError,match='independent baselines'): setup.generate(manifest,tmp_path/'generated')
    manifest['entries'][0].update(system='mixed',candidate_index=999)
    with pytest.raises(ValueError,match='unavailable'): setup.generate(manifest,tmp_path/'generated')


def test_initial_dynamo_fragmentation_keeps_all_nine_types_and_catch_all():
    records=[dict(input_tokens=15,output_tokens=10)]*20+[dict(input_tokens=800,output_tokens=200)]*7
    assignments,evidence=setup.dynamo_pools(records,['a','b','c'])
    assert assignments=={'a':'LL','b':'SS','c':'SS'}
    assert set(evidence['counts'])==set(setup.SHAPES)
    assert evidence['initial_spill']['MM']=='LL'
    assert all(setup.dominates(target,shape) for shape,target in evidence['initial_spill'].items())


def inspected(tmp_path,root='owned',instance_id='dynabc'):
    path=tmp_path/root/(instance_id+'.json')
    value=InstanceSpec(instance_id,1,(0,),24000,28000)
    setup.write(path,dict(id=instance_id,tp=1,gpus=[0],port=24000,kv_port=28000))
    image='sha256:'+'a'*64
    raw=dict(Name='/pdb-v2-'+instance_id,Image=image,State=dict(Running=True),
             Config=dict(Cmd=['python3','-m','ecopadg.serving.engine','--config',str(path)],
                         Env=['CUDA_VISIBLE_DEVICES=0']))
    manifest=dict(image=image,ownership_root=str(tmp_path/'owned'),instances=[],initial_instances=[])
    return raw,manifest,value


def test_restore_accepts_only_owned_or_explicitly_declared_instances(tmp_path):
    raw,manifest,value=inspected(tmp_path)
    assert setup.inspected_instance(raw,manifest)==value
    raw,manifest,value=inspected(tmp_path,root='someone-else')
    with pytest.raises(ValueError,match='outside calibration ownership'): setup.inspected_instance(raw,manifest)
    manifest['initial_instances']=[asdict(value)]
    assert setup.inspected_instance(raw,manifest)==value
    raw['Config']['Env']=['CUDA_VISIBLE_DEVICES=8']
    with pytest.raises(ValueError,match='GPU environment'): setup.inspected_instance(raw,manifest)


def test_restore_rejects_wrong_image_or_nonengine_container(tmp_path):
    raw,manifest,_=inspected(tmp_path)
    raw['Image']='sha256:'+'b'*64
    with pytest.raises(ValueError,match='image'): setup.inspected_instance(raw,manifest)
    raw['Image']=manifest['image'];raw['Config']['Cmd']=['unrelated','process']
    with pytest.raises(ValueError,match='not a declared'): setup.inspected_instance(raw,manifest)


def test_changed_evidence_cannot_start_calibration(tmp_path):
    path=tmp_path/'proof.json';setup.write(path,dict(passed=True))
    evidence={str(path):setup.sha256(path)}
    setup.write(path,dict(passed=False))
    with pytest.raises(ValueError,match='changed measurement'): setup.verify_artifacts(evidence)


def test_uncertified_profiles_fail_before_cpu_configuration_generation(tmp_path):
    path=tmp_path/'profiles.json';setup.write(path,dict(status='development_only'))
    with pytest.raises(ValueError,match='certified hardware profiles'):
        setup.validated_inputs(dict(image='sha256:'+'a'*64,profiles=str(path)))


@pytest.mark.parametrize('missing',['instant_prefill_calibration_complete','instant_heldout_calibration_complete'])
def test_average_power_profile_cannot_enter_calibration_by_relabeling_status(tmp_path,missing):
    image='sha256:'+'a'*64
    data=dict(engine_image=image,model=setup.MODEL_NAME,status='validated_envelope',
        frequency_commands_verified=True,heldout_calibration_complete=True,
        mixed_interference_measured=True,resident_idle_measured=True,
        instant_prefill_calibration_complete=True,instant_heldout_calibration_complete=True)
    data.pop(missing);path=tmp_path/'profiles.json';setup.write(path,data)
    with pytest.raises(ValueError,match='certified hardware profiles'):
        setup.validated_inputs(dict(image=image,profiles=str(path)))


@pytest.mark.parametrize('receiver_proof',[None,False,1,'true'])
def test_independent_calibration_rejects_sender_only_transfer_costs(tmp_path,receiver_proof):
    image='sha256:'+'a'*64
    raw=tmp_path/'profile-raw.json';setup.write(raw,dict(test_only=True))
    profile=tmp_path/'profiles.json'
    setup.write(profile,dict(schema=2,measurement='hardware',points=[],engine_image=image,
        model=setup.MODEL_NAME,status='validated_envelope',frequency_commands_verified=True,
        heldout_calibration_complete=True,mixed_interference_measured=True,resident_idle_measured=True,
        instant_prefill_calibration_complete=True,instant_heldout_calibration_complete=True,
        certification_artifacts={str(raw):setup.sha256(raw)}))
    transfer=tmp_path/'transfers.json'
    data=dict(certified=True,instant_power_costs_verified=True,engine_image=image,
              links=[dict(source_tp=1,target_tp=1)],certification_artifacts={})
    if receiver_proof is not None:data['receiver_transfer_energy_included']=receiver_proof
    setup.write(transfer,data)
    with pytest.raises(ValueError,match='certified same-image transport'):
        setup.validated_inputs(dict(image=image,profiles=str(profile),transfers=str(transfer)))


def test_restore_matching_layout_retains_processes_and_leaves_roles_for_runtime(tmp_path,monkeypatch):
    raw,manifest,value=inspected(tmp_path)
    manifest.update(instances=[dict(asdict(value),role='decode')],engine_template=raw['Config']['Cmd'][-1],retained_weights='unused')
    async def docker(*argv):
        return 'pdb-v2-dynabc\n' if argv[0]=='ps' else json.dumps([raw])
    monkeypatch.setattr(setup,'docker',docker)
    monkeypatch.setattr(setup,'unallocated_gpu_processes',lambda gpus:{g:[] for g in gpus})
    class Response:
        status=200
        async def __aenter__(self):return self
        async def __aexit__(self,*args):pass
        async def json(self):return dict(role='prefill',running=0,waiting=0,active={},transfer_allocations={},
                                        source_files_at_import=setup.current_engine_sources())
    class Session:
        def __init__(self,**kwargs):assert kwargs['trust_env'] is False
        async def __aenter__(self):return self
        async def __aexit__(self,*args):pass
        def get(self,url):return Response()
    import aiohttp
    monkeypatch.setattr(aiohttp,'ClientSession',Session)
    result=asyncio.run(setup.restore_layout(manifest,tmp_path/'preparation'))
    assert result['complete'] and result['changed'] is False and result['energy_j'] is None
    saved=setup.read(tmp_path/'preparation/restore.json')
    assert not saved['add'] and not saved['remove'] and saved['keep'][0]['role']=='decode'


def test_restore_reads_real_engine_cuda_environment_when_config_omits_gpus(tmp_path):
    raw,manifest,value=inspected(tmp_path)
    path=Path(raw['Config']['Cmd'][-1]);config=setup.read(path);config.pop('gpus');setup.write(path,config)
    assert setup.inspected_instance(raw,manifest)==value


@pytest.mark.parametrize('reason',['source','forced','template'])
def test_restore_restarts_stale_imports_templates_or_explicitly_requested_processes(tmp_path,monkeypatch,reason):
    raw,manifest,value=inspected(tmp_path)
    template=tmp_path/'template.json';setup.write(template,{} if reason!='template' else dict(transfer_buffer_bytes=4*1024**3))
    manifest.update(instances=[asdict(value)],engine_template=str(template),retained_weights=None,
                    force_restart=reason=='forced')
    async def docker(*argv):return 'pdb-v2-dynabc\n' if argv[0]=='ps' else json.dumps([raw])
    monkeypatch.setattr(setup,'docker',docker)
    monkeypatch.setattr(setup,'unallocated_gpu_processes',lambda gpus:{g:[] for g in gpus})
    prepared=[]
    async def prepare(config,out):prepared.append(config);return dict(complete=True)
    from ecopadg.serving import prepare as prepare_module
    monkeypatch.setattr(prepare_module,'prepare',prepare)
    class Response:
        status=200
        def __init__(self,url):self.url=url
        async def __aenter__(self):return self
        async def __aexit__(self,*args):pass
        async def json(self):
            if self.url.endswith('/runtime'):return dict(running=0,waiting=0,active={},transfer_allocations={})
            return dict(source_files_at_import={'obsolete.py':'old'} if reason=='source' and not prepared else setup.current_engine_sources())
    class Session:
        def __init__(self,**kwargs):pass
        async def __aenter__(self):return self
        async def __aexit__(self,*args):pass
        def get(self,url):return Response(url)
    import aiohttp
    monkeypatch.setattr(aiohttp,'ClientSession',Session)
    asyncio.run(setup.restore_layout(manifest,tmp_path/'preparation'))
    assert len(prepared)==1 and not prepared[0]['keep']
    assert prepared[0]['remove'][0]['instance_id']==value.instance_id
    assert prepared[0]['add'][0]['instance_id']==value.instance_id


def test_idle_gpu_check_uses_actual_processes_and_rejects_resident_work(monkeypatch):
    import sys
    calls=[]
    nvml=SimpleNamespace(nvmlInit=lambda:calls.append('init'),nvmlShutdown=lambda:calls.append('shutdown'),
        nvmlDeviceGetHandleByIndex=lambda gpu:gpu,
        nvmlDeviceGetComputeRunningProcesses=lambda gpu:[SimpleNamespace(pid=99)] if gpu==6 else [],
        nvmlDeviceGetGraphicsRunningProcesses=lambda gpu:[])
    monkeypatch.setitem(sys.modules,'pynvml',nvml)
    with pytest.raises(RuntimeError,match='still host processes'):
        setup.unallocated_gpu_processes([5,6,7])
    assert calls==['init','shutdown']


def test_physical_cost_gate_accepts_pure_shrink_without_inventing_new_output_checks():
    shrink=dict(target_tps=[],output_checks=[],result=dict(committed=True),unaffected_requests_completed=2)
    assert setup.valid_transition_output(shrink)
    assert not setup.valid_transition_output(dict(shrink,unaffected_requests_completed=0))
    assert setup.valid_transition_output(dict(shrink,unaffected_requests_completed=0,
                                             unaffected_requests_crossing_interval=1))
    assert not setup.valid_transition_output(dict(shrink,result=dict(committed=False)))
    added=dict(shrink,target_tps=[1],output_checks=[dict(tp=1,matches=True)])
    assert setup.valid_transition_output(added)
    assert not setup.valid_transition_output(dict(added,output_checks=[]))
    assert not setup.valid_transition_output(dict(added,output_checks=[dict(tp=2,matches=True)]))


def test_summary_requires_all_independent_datasets_and_same_execution_sources(tmp_path,monkeypatch):
    manifest=fixture_inputs(tmp_path,monkeypatch)
    manifest['calibration_budget_s']=30000
    manifest['entries']=[dict(system=s,dataset=d,candidate_index=0) for d in setup.DATASETS for s in setup.BASELINES]
    selection=setup.generate(manifest,tmp_path/'generated')
    code=tmp_path/'source.py';code.write_text('original')
    from ecopadg.serving import calibration
    monkeypatch.setattr(calibration,'implementation_sources',lambda:[code])
    source={str(code):setup.sha256(code)}
    for selected in selection['selected']:
        path=Path(selected['result'])
        row=dict(system=selected['system'],dataset=selected['dataset'],config=selected['config'],passed=True,
            capacity_rps=1,infeasible_upper_rps=2,confirmation=dict(rate=1,validity='ok',split='calibration',
                completed=128,n_expected=128,slo_attainment=1))
        setup.write(path,dict(passed=True,source_unchanged=True,results=[row]))
        setup.write(path.parent/'source.before.json',source)
    result=setup.summarize(selection,tmp_path/'combined.json')
    assert result['passed'] and result['common_capacity']==dict.fromkeys(setup.DATASETS,1)
    partial=dict(selection,selected=[s for s in selection['selected'] if s['dataset']=='alpaca'])
    assert not setup.summarize(partial,tmp_path/'partial.json')['passed']
    code.write_text('changed measurement')
    assert not setup.summarize(selection,tmp_path/'changed.json')['source_unchanged']
