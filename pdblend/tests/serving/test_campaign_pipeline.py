import time
from pathlib import Path

import pytest

from ecopadg.serving import campaign_pipeline as pipe


def instance(name,gpu,port):
    return dict(id=name,tp=1,gpus=[gpu],port=port,kv_port=port+100,role='mixed')


def fixture_plan(tmp_path):
    from ecopadg.serving.evidence import sha256
    root=tmp_path/'campaign';path=root/'pipeline/pending.json'
    plan=pipe.template(root,path)
    old=[instance('old'+str(g),g,18000+g) for g in range(4)]
    desired=[instance('dvr'+str(g),g,24000+g) for g in range(3)]
    pipe.write(plan['initial_layout_evidence'],dict(complete=True,passed=True,live_instances=old))
    pipe.write(plan['active_campaign'],dict(stages=[dict(name='last')]))
    pipe.write(Path(plan['active_campaign']).with_suffix('.execution-result.json'),dict(
        exit_code=0,complete=True,error=None,manifest_sha256=sha256(plan['active_campaign'])))
    pipe.write(plan['legacy_search_completion'],dict(status='predicted_candidates_only'))
    pipe.write(root/'budget.json',dict(started_s=time.time()-10,limit_s=86400,stage='last',
        last_started_s=1,last_finished_s=2,last_exit_code=0,last_error=None,last_interrupted=False))
    pipe.write(root/'calibration.json',dict(image='sha256:'+'a'*64,retained_weights=str(root/'weights')))
    pipe.write(plan['followup'],dict(calibration_template=str(root/'calibration.json'),
        calibration_out=str(root/'calibration'),method_out=str(root/'methods')))
    setup=Path(plan['dynamo_setup'])
    pipe.write(setup/'engine-template.json',dict(model='Qwen'))
    restoration=dict(instances=desired,initial_instances=old+desired,ownership_root=str(setup),
        image='sha256:'+'a'*64,engine_template=str(setup/'engine-template.json'),retained_weights=str(root/'weights'))
    pipe.write(setup/'restoration.json',restoration)
    pipe.write(setup/'input-evidence.json',dict(artifacts={str(setup/'restoration.json'):sha256(setup/'restoration.json')}))
    return plan,path,old,desired


def test_cpu_template_never_creates_or_resets_budget(tmp_path):
    plan=pipe.template(tmp_path/'root',tmp_path/'pending.json')
    assert plan['status']=='conditional_template_not_executed'
    assert not (tmp_path/'root/budget.json').exists()
    with pytest.raises(FileNotFoundError):pipe.existing_budget(plan['campaign_root'])


def test_original_queue_completion_is_a_hashed_checkpoint_not_profile_certification(tmp_path):
    plan,_,old,_=fixture_plan(tmp_path)
    budget=pipe.existing_budget(plan['campaign_root'])
    checkpoint=pipe.queue_completion(plan,budget)
    assert checkpoint['checkpoint_only'] and not checkpoint['formal_eligible']
    assert len(checkpoint['artifacts'])==4 and len(checkpoint['initial_instances'])==len(old)
    with pytest.raises(ValueError,match='last completed'):
        pipe.queue_completion(plan,dict(budget,last_finished_s=0))


def test_after_dynamo_restores_exact_proven_layout_with_dedicated_ownership(tmp_path):
    from ecopadg.serving.calibration_setup import spec,physical
    plan,_,old,desired=fixture_plan(tmp_path)
    result=pipe.prepare_return(plan)
    assert {physical(spec(i)) for i in result['instances']}=={physical(spec(i)) for i in old}
    assert {i['id'] for i in result['initial_instances']}=={i['id'] for i in old+desired}
    assert result['ownership_root']==plan['dynamo_setup']
    assert pipe.read(Path(plan['return_manifest']).with_suffix('.evidence.json'))['status']=='prepared_not_executed'


@pytest.mark.parametrize('kind',['foreign_previous','broad_owner','out_of_budget_gpu','changed_source'])
def test_after_dynamo_rejects_unowned_engines_and_changed_setup(tmp_path,kind):
    from ecopadg.serving.evidence import sha256
    plan,_,_,_=fixture_plan(tmp_path);setup=Path(plan['dynamo_setup'])
    restoration=pipe.read(setup/'restoration.json')
    if kind=='foreign_previous':restoration['initial_instances'].append(instance('foreign',7,26000))
    elif kind=='broad_owner':restoration['ownership_root']=plan['campaign_root']
    elif kind=='out_of_budget_gpu':restoration['instances'][0]['gpus']=[8]
    else:restoration['instances'][0]['port']+=1
    pipe.write(setup/'restoration.json',restoration)
    if kind!='changed_source':
        pipe.write(setup/'input-evidence.json',dict(artifacts={str(setup/'restoration.json'):sha256(setup/'restoration.json')}))
    with pytest.raises(ValueError):pipe.prepare_return(plan)
    assert not Path(plan['return_manifest']).exists()


def test_stage_list_cannot_change_campaign_or_open_formal_work(tmp_path):
    root=tmp_path/'root';manifest=tmp_path/'stages.json'
    stages=[dict(name='cpu',gpu=False,limit_s=1,argv=['python3']),
        dict(name='gpu',requires=['cpu'],limit_s=2,argv=['python3'])]
    pipe.write(manifest,dict(output=str(root),budget_s=86400,stages=stages))
    assert pipe.stage_list(manifest,root)==stages
    with pytest.raises(ValueError):pipe.stage_list(manifest,tmp_path/'other')
    stages[1]['formal']=True;pipe.write(manifest,dict(output=str(root),stages=stages))
    with pytest.raises(ValueError,match='formal'):pipe.stage_list(manifest,root)


def mock_execution(tmp_path,monkeypatch,*,dynamo_passed):
    from ecopadg.serving import campaign,calibration_setup
    plan,path,_,_=fixture_plan(tmp_path);root=Path(plan['campaign_root']);calls=[]
    pipe.write(plan['post_campaign'],dict(output=str(root),stages=[dict(name='post',argv=['post'],limit_s=100)]))
    pipe.write(plan['eco_result'],dict(complete=True,passed=True))
    pipe.write(Path(plan['smoke_out'])/'campaign.json',dict(output=str(root),stages=[dict(name='smoke',argv=['smoke'],limit_s=900)]))
    pipe.write(Path(plan['smoke_out'])/'run/summary.json',dict(passed=True,status='controller_smoke_passed'))
    pipe.write(Path(plan['admission_out'])/'raw.json',dict(complete=True,passed=True))
    pipe.write(plan['dynamo_campaign'],dict(output=str(root),stages=[
        dict(name='dyn-gen',gpu=False,argv=['generate'],limit_s=60),
        dict(name='dyn-prepare',requires=['dyn-gen'],argv=['prepare'],limit_s=600),
        dict(name='dyn-run',requires=['dyn-prepare'],argv=['run'],limit_s=3000)]))
    pipe.write(Path(plan['dynamo_setup'])/'run/mechanisms.json',dict(passed=dynamo_passed))
    class FakeCampaign:
        def __init__(self,root,limit):
            assert limit==86400
            self.state=pipe.read(Path(root)/'budget.json');self.remaining_s=10000
        def run(self,name,argv,limit,*,gpu=True):calls.append((name,limit,gpu))
        def close(self):calls.append(('closed',0,False))
    monkeypatch.setattr(campaign,'Campaign',FakeCampaign)
    monkeypatch.setattr(calibration_setup,'validated_inputs',lambda m:None)
    pipe.prepare_return(plan)
    return plan,path,calls


def test_failed_real_dynamo_gate_restores_layout_but_never_starts_calibration(tmp_path,monkeypatch):
    plan,path,calls=mock_execution(tmp_path,monkeypatch,dynamo_passed=False)
    with pytest.raises(RuntimeError,match='did not pass'):pipe.execute(plan,path)
    names=[c[0] for c in calls]
    assert names[-2:]==['restore-proven-layout-after-dynamo','closed']
    assert 'prepare-followup-calibration' not in names
    assert not pipe.read(path.with_suffix('.status.json'))['complete']
    assert pipe.read(path.with_suffix('.queue-completion.json'))['complete']


def test_failed_calibration_never_opens_development(tmp_path,monkeypatch):
    plan,path,calls=mock_execution(tmp_path,monkeypatch,dynamo_passed=True)
    followup=pipe.read(plan['followup']);root=Path(plan['campaign_root'])
    pipe.write(Path(followup['calibration_out'])/'campaign.json',dict(output=str(root),stages=[
        dict(name='calibration',argv=['calibrate'],limit_s=1200)]))
    pipe.write(Path(followup['calibration_out'])/'summary.json',dict(passed=False))
    with pytest.raises(RuntimeError,match='did not pass'):pipe.execute(plan,path)
    assert 'prepare-followup-methods' not in [c[0] for c in calls]
    assert pipe.read(root/'budget.json')['started_s']<time.time()


def test_smoke_reuses_proven_ids_accepted_by_dynamo_initial_layout(tmp_path,monkeypatch):
    from ecopadg.serving import controller_smoke
    plan,_,old,_=fixture_plan(tmp_path)
    followup=pipe.read(plan['followup']);cal=pipe.read(followup['calibration_template'])
    cal.update(profiles='profiles',transfers='transfers',interconnect='interconnect',frequency_costs=['clocks'],engine_template='engine')
    pipe.write(followup['calibration_template'],cal)
    monkeypatch.setattr(controller_smoke,'generate',lambda manifest,out:manifest)
    result=pipe.prepare_smoke(plan)
    assert [i['id'] for i in result['instances']]==[i['id'] for i in old]
    assert all(i['role']=='mixed' for i in result['instances'])
    assert result['ownership_root']==plan['smoke_out']


def test_mechanism_collection_is_cpu_only_before_development_and_missing_stays_explicit(tmp_path,monkeypatch):
    from ecopadg.serving import method_selection
    plan,path,calls=mock_execution(tmp_path,monkeypatch,dynamo_passed=True)
    followup=pipe.read(plan['followup']);root=Path(plan['campaign_root'])
    pipe.write(Path(followup['calibration_out'])/'campaign.json',dict(output=str(root),stages=[]))
    pipe.write(Path(followup['calibration_out'])/'summary.json',dict(passed=True))
    pipe.write(Path(followup['method_out'])/'paired/campaign.json',dict(output=str(root),stages=[]))
    missing=dict(distserve=['phase_batching','kv_admission'])
    pipe.write(Path(plan['mechanism_collection_out'])/'summary.json',dict(complete=True,
        baseline_mechanisms_complete=False,missing=missing))
    monkeypatch.setattr(method_selection,'summarize',lambda _:dict(selected_variant='pdblend-greedy'))
    result=pipe.execute(plan,path);names=[c[0] for c in calls]
    assert names.index('collect-baseline-mechanism-evidence')<names.index('prepare-followup-methods')
    assert next(c for c in calls if c[0]=='collect-baseline-mechanism-evidence')[2] is False
    assert result['complete'] and not result['formal_eligible']
    assert not result['baseline_mechanisms_complete'] and result['missing_baseline_mechanisms']==missing


@pytest.mark.parametrize('damage',['missing','error','interrupted','child_failed','receipt_failed','receipt_changed','bool_exit'])
def test_queue_completion_requires_real_successful_receipt_and_stage(tmp_path,damage):
    plan,_,_,_=fixture_plan(tmp_path)
    value=pipe.existing_budget(plan['campaign_root'])
    receipt=Path(plan['active_campaign']).with_suffix('.execution-result.json')
    if damage=='missing':receipt.unlink()
    elif damage=='error':value['last_error']='KeyboardInterrupt'
    elif damage=='interrupted':value['last_interrupted']=True
    elif damage=='child_failed':value['last_exit_code']=7
    elif damage=='bool_exit':value['last_exit_code']=False
    else:
        raw=pipe.read(receipt)
        if damage=='receipt_failed':raw['exit_code']=1
        else:raw['manifest_sha256']='0'*64
        pipe.write(receipt,raw)
    with pytest.raises((ValueError,FileNotFoundError)):pipe.queue_completion(plan,value)
