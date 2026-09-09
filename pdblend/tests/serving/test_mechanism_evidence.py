from pathlib import Path
import pytest

from ecopadg.serving import mechanism_evidence as proof
from ecopadg.serving.campaign_followup_setup import read,write


def test_actual_cpu_contracts_archive_test_output_and_full_source_inventory(tmp_path):
    result=proof.cpu_contracts(tmp_path/'cpu')
    assert result['passed'] and result['source_unchanged']
    assert set(result['cases'])==set(proof.CONTRACTS) and all(result['cases'].values())
    assert any(p.endswith('/runtime.py') for p in result['source_files'])
    assert any(p.endswith('/test_mechanism_contracts.py') for p in result['source_files'])
    assert '5 passed' in (tmp_path/'cpu/output.txt').read_text()
    assert proof.validate_freeze(result['artifacts'])==[]


def test_old_or_partial_execution_source_inventory_is_rejected():
    with pytest.raises(ValueError,match='incomplete or changed'):proof.current_sources({})


def test_hardware_provenance_requires_current_import_and_exact_model(monkeypatch):
    from ecopadg.serving import calibration_setup
    monkeypatch.setattr(calibration_setup,'current_engine_sources',lambda:{'engine':'current'})
    row=dict(image_id='image',model='/models/Qwen2.5-14B-Instruct',engine_version='0.9.2',source_files_at_import={'engine':'current'})
    proof.engine_provenance([row],'image')
    with pytest.raises(ValueError,match='hardware image'):
        proof.engine_provenance([dict(row,source_files_at_import={'engine':'old'})],'image')
    with pytest.raises(ValueError,match='hardware image'):
        proof.engine_provenance([dict(row,model='another-model')],'image')


def test_actual_power_reader_rejects_missing_field_provenance(tmp_path):
    names=['t_s']+[f'gpu{i}_w' for i in range(8)]
    (tmp_path/'power.csv').write_text(','.join(names)+'\n'+','.join(['1']+['100']*8)+'\n')
    write(tmp_path/'power_source.json',dict(mode='average'))
    (tmp_path/'power_metadata.jsonl').write_text('')
    with pytest.raises(ValueError,match='verified instantaneous'):proof.checked_power(tmp_path)


def fixture_collect(tmp_path,monkeypatch):
    plan=dict(followup=str(tmp_path/'followup.json'),eco_result=str(tmp_path/'eco/raw.json'),
        dynamo_setup=str(tmp_path/'dynamo'),smoke_out=str(tmp_path/'smoke'),
        mechanisms_out=str(tmp_path/'mechanisms.certified-v2.json'))
    write(plan['followup'],dict(calibration_template=str(tmp_path/'cal.json'),calibration_out=str(tmp_path/'calibration')))
    write(tmp_path/'cal.json',{})
    def contracts(out):
        result=dict(passed=True,cases={k:True for k in proof.CONTRACTS});write(out/'summary.json',result);return result
    monkeypatch.setattr(proof,'cpu_contracts',contracts)
    monkeypatch.setattr(proof,'sources',lambda:{})
    return plan


def test_cpu_contracts_cannot_fill_hardware_batching_kv_or_outputs(tmp_path,monkeypatch):
    plan=fixture_collect(tmp_path,monkeypatch)
    old=tmp_path/'mechanisms.json';write(old,{'old':'untouched'})
    result=proof.collect(plan,tmp_path/'out');registry=read(result['registry'])
    assert result['complete'] and not result['baseline_mechanisms_complete'] and not result['formal_eligible']
    assert result['registry_sha256']==proof.sha256(result['registry'])
    assert read(tmp_path/'out/summary.json')['registry_sha256']==result['registry_sha256']
    assert registry['dynamollm']['length_prediction']['passed']
    assert registry['dynamollm']['nine_logical_pools']['passed']
    assert registry['ecoserve']['unified_constraints']['passed']
    for field in ('phase_batching','kv_admission','output_correctness','independent_calibration'):
        assert not registry['distserve'][field]['passed']
    assert not registry['dynamollm']['fragmentation']['passed']
    assert read(old)=={'old':'untouched'}


def test_collection_never_overwrites_an_existing_registry(tmp_path,monkeypatch):
    plan=fixture_collect(tmp_path,monkeypatch);write(plan['mechanisms_out'],{'before':True})
    with pytest.raises(ValueError,match='overwrite existing'):proof.collect(plan,tmp_path/'out')
    assert read(plan['mechanisms_out'])=={'before':True}


def eco_fixture(tmp_path,monkeypatch):
    monkeypatch.setattr(proof,'current_sources',lambda _:None)
    monkeypatch.setattr(proof,'engine_provenance',lambda *args:None)
    profile=tmp_path/'profiles.json';write(profile,{})
    admissions=[dict(request_id='r'+str(i),client_request_id=str(i),input_tokens=128,
        plan=dict(routes=[dict(decode_id='i'+str(i))],windows=[dict(admit_prefill=False),dict(admit_prefill=True)])) for i in range(4)]
    events=[dict(tokens=1,prefill=p,decode=not p,mode='temporal',role='mixed',request_ids=['r'+str(i)],instance='i'+str(i))
        for i in range(4) for p in (False,True)]
    raw=dict(complete=True,passed=True,errors=[],cleanup_errors=[],source_files={},config=dict(profiles=str(profile)),
        profile_sha256=proof.sha256(profile),admissions=admissions,events=events,
        kv_observations=[dict(owners={'i'+str(i):['r'+str(i)] for i in range(4)})],
        reference={'i'+str(i):{'128':[1,2]} for i in range(4)},
        outputs=[dict(success=True,token_ids=[1,2],generated_tokens=2) for _ in range(4)],
        checks={k:True for k in ('engine_temporal_exclusion','rolling_activation','macro_split_merge','output_correctness')},
        changes=[dict(operation=op,after=[['0','1'],['2','3']] if op=='split' else [['0','2','3']],
            before_kv=dict(owners={'i0':['r0']}),acknowledgements=[dict(generation=1,acknowledged_generation=1)],
            removed='1' if op=='merge' else None,removed_state={}) for op in ('split','merge')],
        engine_provenance=[{'same':True}],engine_provenance_after=[{'same':True}])
    path=tmp_path/'raw.json';write(path,raw);return path,raw


@pytest.mark.parametrize('corruption',['empty_outputs','overlap','wrong_token','missing_merge'])
def test_eco_rechecks_hardware_events_and_outputs_instead_of_trusting_passed(tmp_path,monkeypatch,corruption):
    path,raw=eco_fixture(tmp_path,monkeypatch)
    assert proof.checked_eco(path,'image')['engine_temporal_exclusion']
    if corruption=='empty_outputs':raw['outputs']=[]
    elif corruption=='overlap':raw['events'][0]['prefill']=raw['events'][0]['decode']=True
    elif corruption=='wrong_token':raw['outputs'][0]['token_ids']=[9,9]
    else:raw['changes'].pop()
    write(path,raw)
    with pytest.raises((ValueError,RuntimeError)):proof.checked_eco(path,'image')


def test_dynamo_missing_actual_actions_cannot_inherit_old_passed_flag(tmp_path,monkeypatch):
    from ecopadg.serving import calibration_setup,dynamo_validation_setup
    monkeypatch.setattr(calibration_setup,'verify_artifacts',lambda _:None)
    monkeypatch.setattr(proof,'current_sources',lambda _:None)
    monkeypatch.setattr(proof,'engine_provenance',lambda *args:None)
    monkeypatch.setattr(proof,'checked_power',lambda _:[])
    monkeypatch.setattr(proof,'events',lambda _:[])
    monkeypatch.setattr(proof,'rows',lambda _:[])
    monkeypatch.setattr(dynamo_validation_setup,'audit',lambda *args:dict(passed=False,proposed_mechanism_fields={}))
    write(tmp_path/'mechanisms.json',dict(passed=True,artifacts={},proposed_mechanism_fields={}))
    write(tmp_path/'raw.json',dict(source_files={},errors=[],provenance_before=[],provenance_after=[]));write(tmp_path/'summary.json',{})
    with pytest.raises(ValueError,match='actual cycles/actions'):proof.checked_dynamo(tmp_path,'image')


def test_failed_cpu_contract_cannot_mark_algorithm_mechanisms_passed(tmp_path,monkeypatch):
    plan=fixture_collect(tmp_path,monkeypatch)
    monkeypatch.setattr(proof,'cpu_contracts',lambda out:dict(passed=False))
    result=proof.collect(plan,tmp_path/'out');registry=read(result['registry'])
    assert not registry['dynamollm']['length_prediction']['passed']
    assert not registry['ecoserve']['unified_constraints']['passed']


def batching_fixture(tmp_path,monkeypatch):
    from ecopadg.serving import measurement
    monkeypatch.setattr(measurement,'power_evidence',lambda *args:dict(power_source_verified=True))
    from ecopadg.serving import engine
    source_hash=proof.sha256(engine.__file__)
    provenance=[dict(image_id='image',engine_version='0.9.2',model='/models/Qwen2.5-14B-Instruct',
        source_files_at_import={str(Path(engine.__file__).resolve()):source_hash})]
    topology=dict(prefill=dict(id='p',tp=1,gpus=[0]),decode=dict(id='d',tp=1,gpus=[1]))
    common=dict(complete=True,runs=[],sampling_error=None,frequency_samples=[[1,[2520]*8]],
        topology=topology,engine_provenance=provenance,commanded_frequencies={'0':2520,'1':2520})
    producer=dict(common,runs=[dict(step=dict(role='prefill',prefill=2,decode=0,tokens=256,
        request_ids=['p1','p2']),batch=2,output_matches=True,send=[{}],receive=[{},{}])])
    consumer=dict(common,runs=[dict(layout='pd',batch=2,output_tokens=3,held_kv_tokens={'d1':128,'d2':128},
        token_ids=[[1,2,3],[1,2,3]],usages=[dict(completion_tokens=3)]*2,events=[
            dict(instance='d',role='decode',decode=2,prefill=0,tokens=2,request_ids=['d1','d2']) for _ in range(2)])])
    pp=tmp_path/'p/raw.json';dp=tmp_path/'d/raw.json'
    write(pp,producer);write(dp,consumer)
    config=tmp_path/'config.json';write(config,dict(instances=[dict(role='prefill',tp=1),dict(role='decode',tp=1)]))
    calibration=tmp_path/'calibration.json';write(calibration,dict(results=[dict(system='distserve',passed=True,config=str(config))]))
    profiles=tmp_path/'profiles.json';transfers=tmp_path/'transfers.json'
    def refresh():
        write(profiles,dict(certification_artifacts={str(dp):proof.sha256(dp)}))
        write(transfers,dict(instant_power_costs_verified=True,receiver_transfer_energy_included=True,
            certification_artifacts={str(pp):proof.sha256(pp)}))
    refresh()
    return dict(profiles=str(profiles),transfers=str(transfers),image='image'),calibration,pp,dp,refresh


def test_independent_real_p_and_d_steps_certify_batching(tmp_path,monkeypatch):
    cal,path,*_=batching_fixture(tmp_path,monkeypatch)
    result=proof.checked_batches(cal,path)
    assert result['passed'] and result['examples']['prefill'][1]['batch']==2
    assert result['examples']['decode'][1]['executed_decode_steps']==2


@pytest.mark.parametrize('receiver_proof',[None,False,1,'true'])
def test_batching_certificate_does_not_use_old_transfer_bundle(tmp_path,monkeypatch,receiver_proof):
    cal,path,*_=batching_fixture(tmp_path,monkeypatch)
    value=read(cal['transfers'])
    if receiver_proof is None:value.pop('receiver_transfer_energy_included')
    else:value['receiver_transfer_energy_included']=receiver_proof
    write(cal['transfers'],value)
    with pytest.raises(ValueError,match='sender and receiver costs'):
        proof.checked_batches(cal,path)


@pytest.mark.parametrize('receiver_proof',[None,False,1,'true',True])
def test_search_certificate_rechecks_receiver_cost_basis(tmp_path,monkeypatch,receiver_proof):
    from ecopadg.serving import calibration_setup
    bundle=dict(instant_power_costs_verified=True,links=[dict(source_tp=1,target_tp=1,
        source_gpus=[0],target_gpus=[1],max_input_tokens=128,seconds_upper=.1,incremental_j=2,
        validated=True,source_sha256='synthetic')])
    if receiver_proof is not None:bundle['receiver_transfer_energy_included']=receiver_proof
    choice=dict(prefill_count=1,decode_count=1,prefill_tp=1,decode_tp=1,
                prefill_batch=1,decode_batch=1,gpus=[[0],[1]])
    search=dict(datasets={dataset:dict(shape=dict(input_tokens=128),distserve=[choice])
                         for dataset in ('alpaca','sharegpt','longbench')})
    monkeypatch.setattr(calibration_setup,'validated_inputs',lambda cal:({},bundle,search,None,None,None,None,{}))
    topology=tmp_path/'topology.txt';topology.write_text('GPU0 X PIX\nGPU1 PIX X\n')
    if receiver_proof is True:
        assert proof.checked_search(dict(interconnect=str(topology)))['candidate_counts']==dict(
            alpaca=1,sharegpt=1,longbench=1)
    else:
        with pytest.raises(ValueError,match='sender and receiver costs'):
            proof.checked_search(dict(interconnect=str(topology)))


@pytest.mark.parametrize('change',('prefill_label_only','decode_label_only','missing_output','wrong_clock','wrong_engine'))
def test_batch_labels_cannot_replace_actual_multi_request_steps_and_work(change,tmp_path,monkeypatch):
    cal,path,pp,dp,refresh=batching_fixture(tmp_path,monkeypatch)
    p=read(pp);d=read(dp)
    if change=='prefill_label_only':p['runs'][0]['step']['prefill']=1
    elif change=='decode_label_only':d['runs'][0]['events'][0]['decode']=1
    elif change=='missing_output':d['runs'][0]['token_ids'][0].pop()
    elif change=='wrong_clock':d['commanded_frequencies']['1']=900
    else:p['engine_provenance'][0]['source_files_at_import']={'engine':'old'}
    write(pp,p);write(dp,d);refresh()
    with pytest.raises(ValueError,match='missing real multi-request'):proof.checked_batches(cal,path)
