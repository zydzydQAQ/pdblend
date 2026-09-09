import asyncio
from copy import deepcopy
from pathlib import Path
import time
from types import SimpleNamespace

import pytest

from ecopadg.serving import formal_setup as formal
from ecopadg.serving import evidence, provenance
from ecopadg.serving import method_selection
from ecopadg.serving.calibration import implementation_sources


def save(path, value):
    formal.write(path, value)
    return path


@pytest.fixture
def prepared_inputs(tmp_path, monkeypatch):
    from test_transfer_power import instant
    image = 'sha256:'+'a'*64
    proof = save(tmp_path/'profile-proof.json', dict(complete=True))
    clock = save(tmp_path/'clock.json', dict(instant([(1,[100]*8),(2,[100]*8)]), complete=True, prefix_matches_reference=True,
        sampling_error=None, frequency_samples=[[1, [1500]*8]], engine_provenance=[dict(image_id=image)],
        switches=[dict(tp=1, source_mhz=900, target_mhz=1500, started_s=1, finished_s=1.01, energy_j=5)]))
    frequency_costs = [dict(tp=1, source_mhz=900, target_mhz=1500,
        duration_upper_s=.02, energy_upper_j=10, source_sha256=evidence.sha256(clock))]
    profiles = save(tmp_path/'profiles.json', dict(status='validated_envelope', model='Qwen2.5-14B-Instruct',
        engine_image=image, frequency_commands_verified=True, heldout_calibration_complete=True,
        mixed_interference_measured=True, resident_idle_measured=True,
        instant_prefill_calibration_complete=True,instant_heldout_calibration_complete=True,
        points=[dict(tp=1, frequency_mhz=2520)], certification_artifacts=evidence.freeze_files([proof])))
    template = save(tmp_path/'engine.json', dict(model='/models/Qwen2.5-14B-Instruct', max_model_len=8192))
    common = dict(slo_ttft_s=5., slo_tpot_s=.1, output_prior=2, profiles=str(profiles), node_gpus=list(range(8)),power_mode='instant',
                  frequency_costs=frequency_costs, frequency_evidence=[str(clock)])
    instances = [dict(id='i1', tp=1, gpus=[0], port=18101, kv_port=19101, role='mixed')]
    configs = {}
    for system in formal.SYSTEMS:
        strategy = 'pdblend-joint' if system == 'pdblend' else system
        configs[system] = save(tmp_path/(system+'.config.json'), dict(common, strategy=strategy, instances=instances))
    calibrated = []
    for dataset in formal.DATASETS:
        for system in evidence.REQUIRED_MECHANISMS:
            calibrated.append(dict(system=system, dataset=dataset, config=str(configs[system]), passed=True,
                capacity_rps=.1, infeasible_upper_rps=.2, confirmation={}))
    calibration = save(tmp_path/'calibration.json', {})
    method_prepared = save(tmp_path/'method-prepared.json', {})
    selection = save(tmp_path/'method-selection.json', {})
    monkeypatch.setattr(formal, 'checked_calibration', lambda path: (calibrated, {d:.1 for d in formal.DATASETS}, {calibration}))
    monkeypatch.setattr(formal, 'checked_selection', lambda *paths: ('pdblend-joint', formal.read(configs['pdblend']), {method_prepared,selection}))
    mechanisms = {}
    for baseline, names in evidence.REQUIRED_MECHANISMS.items():
        mechanisms[baseline] = {name:dict(passed=True,artifact=str(proof),sha256=evidence.sha256(proof)) for name in names}
    mechanisms_path = save(tmp_path/'mechanisms.json', mechanisms)
    collection_path = save(tmp_path/'collection.json',dict(complete=True,baseline_mechanisms_complete=True,
        missing={},registry=str(mechanisms_path),registry_sha256=evidence.sha256(mechanisms_path),
        components={k:dict(passed=True) for k in evidence.COLLECTION_COMPONENTS},
        source_files={str(proof):evidence.sha256(proof)}))
    corpus_fingerprints = {}
    for dataset in formal.DATASETS:
        records = [dict(prompt=[i, 7], input_tokens=2, output_tokens=2, request_shape_sha256=f'{dataset}-{i}') for i in range(500)]
        path = save(tmp_path/'corpus'/(dataset+'.json'), dict(dataset=dataset, development=[],
            calibration=[dict(records[0],request_shape_sha256='calibration-'+dataset)],
            formal={str(seed):records for seed in formal.FORMAL_SEEDS}))
        corpus_fingerprints[dataset] = dict(sha256=evidence.sha256(path))
    corpus_manifest = save(tmp_path/'corpus/manifest.json', dict(datasets=corpus_fingerprints))
    priors, sources = method_selection.calibration_priors(tmp_path/'corpus',formal.DATASETS,calibrated)
    save(configs['pdblend'],dict(formal.read(configs['pdblend']),output_priors=priors,output_prior_sources=sources))
    protocol = dict(model='Qwen2.5-14B-Instruct', static_requests_per_run=500, formal_paired_seeds=[101,202,303],
        datasets=list(formal.DATASETS), load_fractions=[.3,.6,.9], dynamic_duration_seconds=3600,
        energy_savings_ci95_lower_target=.05, joint_slo_difference_ci95_lower_target=-.01,
        formal_baselines=list(evidence.REQUIRED_MECHANISMS), max_model_len=8192,gpu=dict(count=8),
        slo_ttft_s=5., slo_tpot_s=.1, corpus_manifest=str(corpus_manifest),
        dynamic_phases=[dict(start_s=i*900,end_s=(i+1)*900,length_mix=mix,capacity_fraction=fraction)
            for i,(mix,fraction) in enumerate((([.7,.2,.1],.3),([.2,.6,.2],.6),([.1,.2,.7],.9),([.7,.2,.1],.3)))])
    protocol_path = save(tmp_path/'protocol.json', protocol)
    model = tmp_path/'model';model.mkdir()
    for name in ('config.json','tokenizer.json','tokenizer_config.json','model-1.safetensors','model-2.safetensors'):
        (model/name).write_text('fixture')
    save(model/'model.safetensors.index.json', dict(weight_map=dict(a='model-1.safetensors',b='model-2.safetensors')))
    monkeypatch.setattr(provenance, 'MODEL_ROOT', model)
    inspect = save(tmp_path/'image.json', [dict(Id=image)])
    mappings = {dataset:{load:{s:str(p) for s,p in configs.items()} for load in formal.LOADS} for dataset in formal.DATASETS}
    mappings['dynamic'] = dict(changing={s:str(p) for s,p in configs.items()})
    save(tmp_path/'campaign/budget.json', dict(started_s=time.time()-60,limit_s=86400))
    manifest = dict(mechanisms=str(mechanisms_path),mechanism_collection=str(collection_path),
        calibration=str(calibration),method_prepared=str(method_prepared),
        method_selection=str(selection),protocol=str(protocol_path),corpus=str(tmp_path/'corpus'),model_dir=str(model),
        image_inspect=str(inspect),engine_image=image,source_files=[],profile_files=[str(proof),str(clock)],
        campaign_root=str(tmp_path/'campaign'),restore=dict(initial_instances=instances,ownership_root=str(tmp_path),
        image=image,engine_template=str(template),retained_weights=None), configs=mappings,
        dynamic_prior_dataset='sharegpt',
        execute_groups=[dict(dataset='alpaca',load='low',seed=101)],formal_budget_s=1000,cell_limit_s=100)
    path = save(tmp_path/'manifest.json', manifest)
    return path, tmp_path/'formal'


def test_generate_freezes_full_target_while_scheduling_one_complete_six_system_group(prepared_inputs):
    result = formal.generate(*prepared_inputs)
    cells = formal.read(result['expected_cells'])
    assert len(cells) == 30 and not evidence.matrix_gaps(cells)
    assert len(result['jobs']) == 180 and result['selected_groups'] == 1
    campaign = formal.read(result['campaign'])
    assert len(campaign['stages']) == 1 and campaign['stages'][0]['formal'] is True
    group = formal.read(next(g['path'] for g in result['groups'] if g['selected']))
    assert len(group['jobs']) == 6
    jobs = [formal.read(p) for p in group['jobs']]
    assert {j['system'] for j in jobs} == set(formal.SYSTEMS)
    assert len({j['trace'] for j in jobs}) == 1
    freeze = formal.read(result['freeze'])
    assert not evidence.formal_freeze_gaps(freeze)
    assert len(freeze['groups']['traces']) == 30
    assert freeze['formal_evidence']['expected_cells']==result['expected_cells']
    assert freeze['formal_evidence']['mechanism_collection']==result['mechanism_collection']
    assert result['mechanism_collection'] in freeze['groups']['protocol']
    assert {Path(p).name for p in freeze['groups']['model']} >= {'model-1.safetensors','model-2.safetensors','tokenizer.json'}
    assert str(Path(formal.__file__).resolve()) in freeze['groups']['source']
    for path in freeze['groups']['traces']:
        trace = formal.read(path)
        assert trace['split'] == 'formal' and trace['seed'] in (101,202,303)
        if trace['dataset'] == 'dynamic':
            assert trace['duration_s'] == 3600 and trace['requests'][-1]['arrival_s'] == 3600
        else: assert len(trace['requests']) == 500
    report = formal.report(Path(result['out'])/'prepared.json',Path(result['out'])/'empty-report')
    assert report['verdict']['verdict'] == 'evidence_insufficient'
    assert len(report['verdict']['comparisons']['mixed']['invalid_pairs']) == 30


@pytest.mark.parametrize('mutation', [
    dict(execute_groups=[dict(dataset='alpaca',load='low',seed=11)]),
    dict(formal_budget_s=500), dict(formal_budget_s=90000),
    dict(execute_groups=[dict(dataset='dynamic',load='changing',seed=101)],dynamic_cell_limit_s=100),
])
def test_no_development_seed_partial_group_or_budget_extension(prepared_inputs,mutation):
    path,out = prepared_inputs
    save(path,dict(formal.read(path),**mutation))
    with pytest.raises(ValueError): formal.generate(path,out)


def test_changed_protocol_or_incomplete_model_cannot_open_formal_gate(prepared_inputs):
    path,out=prepared_inputs;manifest=formal.read(path)
    protocol_path=Path(manifest['protocol'])
    save(protocol_path,dict(formal.read(protocol_path),static_requests_per_run=100))
    with pytest.raises(ValueError,match='fixed first-round target'): formal.generate(path,out)
    save(protocol_path,dict(formal.read(protocol_path),static_requests_per_run=500))
    (Path(manifest['model_dir'])/'model-2.safetensors').unlink()
    with pytest.raises(ValueError,match='missing'): formal.generate(path,out)


def test_unverified_mechanism_and_changed_corpus_are_rejected(prepared_inputs):
    path,out=prepared_inputs;manifest=formal.read(path)
    mechanism_path=Path(manifest['mechanisms']);mechanisms=formal.read(mechanism_path)
    original=deepcopy(mechanisms)
    mechanisms['distserve']['kv_admission']['passed']=False;save(mechanism_path,mechanisms)
    with pytest.raises(ValueError,match='mechanism gate'): formal.generate(path,out)
    save(mechanism_path,original)
    corpus_path=Path(manifest['corpus'])/'alpaca.json'
    save(corpus_path,dict(formal.read(corpus_path),changed=True))
    with pytest.raises(ValueError,match='corpus differs'): formal.generate(path,out)


def test_formal_job_restores_before_shared_cell_and_preserves_formal_arguments(prepared_inputs,monkeypatch):
    from ecopadg.serving import calibration_setup,cell
    result=formal.generate(*prepared_inputs);job_path=result['jobs'][0];job=formal.read(job_path)
    calls=[]
    async def restore(layout,out):
        calls.append('restore');assert layout==job['restore']
        assert str(out).endswith('.preparation')
    async def measure(options):
        calls.append('measure');assert calls==['restore','measure']
        assert options.split=='formal' and options.seed in formal.FORMAL_SEEDS
        assert options.freeze==Path(result['freeze']) and options.strategy is None
        return dict(passed=True)
    monkeypatch.setattr(calibration_setup,'restore_layout',restore)
    monkeypatch.setattr(cell,'run_cell',measure)
    assert asyncio.run(formal.run_job(job_path))==dict(passed=True)
    assert calls==['restore','measure']


def test_changed_group_or_insufficient_remaining_budget_starts_no_partial_pair(prepared_inputs,monkeypatch):
    result=formal.generate(*prepared_inputs)
    path=next(g['path'] for g in result['groups'] if g['selected'])
    group=formal.read(path);calls=[]
    async def job(path):calls.append(path)
    monkeypatch.setattr(formal,'run_job',job)
    save(Path(group['campaign_root'])/'budget.json',dict(started_s=time.time()-86000,limit_s=86400))
    with pytest.raises(ValueError,match='complete paired group'):asyncio.run(formal.run_group(path))
    assert not calls
    save(Path(path),dict(group,jobs=group['jobs'][:1]))
    with pytest.raises(ValueError,match='gate closed'):asyncio.run(formal.run_group(path))
    assert not calls


@pytest.fixture
def calibrated_evidence(tmp_path):
    results=[]
    for dataset in formal.DATASETS:
        for system in evidence.REQUIRED_MECHANISMS:
            root=tmp_path/(dataset+'-'+system)
            config_path=save(root/'config.json',dict(strategy=system,profiles='unchanged'))
            summary=dict(system=system,dataset=dataset,split='calibration',slo_attainment=1.,
                validity='ok',completed=128,n_expected=128,gpu_count=8,generated_tokens=256,expected_generated_tokens=256)
            trace=save(root/'trace.json',dict(split='calibration',dataset=dataset,rate=1.,requests=[{}]*128))
            summary['trace_sha256']=evidence.sha256(trace)
            artifact=save(root/'confirmation/summary.json',summary)
            save(root/'confirmation/runtime_config.json',dict(strategy=system,profiles='unchanged',journal='output-only'))
            save(root/'source.before.json',evidence.freeze_files(implementation_sources()))
            upper_trace=save(root/'upper.trace.json',dict(split='calibration',dataset=dataset,rate=2.,requests=[{}]*128))
            upper=dict(summary,slo_attainment=.9,trace_sha256=evidence.sha256(upper_trace))
            upper_artifact=save(root/'upper/summary.json',upper)
            save(root/'upper/runtime_config.json',dict(strategy=system,profiles='unchanged',journal='upper-output'))
            results.append(dict(system=system,dataset=dataset,passed=True,capacity_rps=1.,infeasible_upper_rps=2.,
                config=str(config_path),confirmation=dict(summary,rate=1.,artifact=str(artifact),trace=str(trace)),
                infeasible_upper_observation=dict(upper,rate=2.,artifact=str(upper_artifact),trace=str(upper_trace))))
    path=save(tmp_path/'calibration.json',dict(passed=True,source_unchanged=True,results=results))
    return path,results


def test_checked_calibration_requires_unchanged_source_and_actual_runtime_configuration(calibrated_evidence):
    path,results=calibrated_evidence
    checked,capacities,files=formal.checked_calibration(path)
    assert len(checked)==15 and capacities=={d:1. for d in formal.DATASETS}
    config=Path(results[0]['config']);save(config,dict(strategy=results[0]['system'],profiles='edited-after-calibration'))
    with pytest.raises(ValueError,match='configuration changed'):formal.checked_calibration(path)
    save(config,dict(strategy=results[0]['system'],profiles='unchanged'))
    source=Path(results[0]['confirmation']['artifact']).parent.parent/'source.before.json'
    recorded=formal.read(source);recorded[next(iter(recorded))]='changed';save(source,recorded)
    with pytest.raises(ValueError,match='implementation fingerprints'):formal.checked_calibration(path)


@pytest.mark.parametrize('mutation',['missing','invalid_runtime','invalid_power_source','invalid_work',
    'successful_slo','different_rate','changed_raw','changed_runtime'])
def test_formal_requires_actual_valid_slo_failure_for_capacity_upper(calibrated_evidence,mutation):
    path,results=calibrated_evidence;row=results[0];upper=row['infeasible_upper_observation']
    if mutation=='missing':del row['infeasible_upper_observation']
    elif mutation.startswith('invalid_') or mutation=='successful_slo':
        change=dict(validity=mutation) if mutation.startswith('invalid_') else dict(slo_attainment=1.)
        upper.update(change)
        artifact=Path(upper['artifact']);save(artifact,dict(formal.read(artifact),**change))
    elif mutation=='different_rate':upper['rate']=3.
    elif mutation=='changed_raw':
        artifact=Path(upper['artifact']);save(artifact,dict(formal.read(artifact),slo_attainment=.8))
    else:
        save(Path(upper['artifact']).parent/'runtime_config.json',dict(strategy='mixed',profiles='different'))
    save(path,dict(passed=True,source_unchanged=True,results=results))
    with pytest.raises(ValueError):formal.checked_calibration(path)


def test_formal_accepts_verified_admission_boundary_only_as_upper(calibrated_evidence):
    path,results=calibrated_evidence;row=results[0];upper=row['infeasible_upper_observation']
    change=dict(validity='invalid_work',capacity_observation_valid=True,admission_rejections=1,
        completed=127,generated_tokens=254,slo_attainment=127/128,power_mode='instant',power_source_verified=True)
    upper.update(change);artifact=Path(upper['artifact']);save(artifact,dict(formal.read(artifact),**change))
    save(path,dict(passed=True,source_unchanged=True,results=results))
    assert formal.checked_calibration(path)[1]=={d:1. for d in formal.DATASETS}
    upper['capacity_observation_valid']=False
    save(artifact,dict(formal.read(artifact),capacity_observation_valid=False))
    save(path,dict(passed=True,source_unchanged=True,results=results))
    with pytest.raises(ValueError,match='invalid work'):formal.checked_calibration(path)


def test_cost_sources_must_be_complete_and_successful(tmp_path):
    raw=save(tmp_path/'role.json',dict(complete=True,passed=False))
    config=dict(role_costs=[dict(source_sha256=evidence.sha256(raw))])
    with pytest.raises(ValueError,match='failed measurement'):formal.certified_profile_files([config],[raw])
    save(raw,dict(complete=True,passed=True))
    config['role_costs'][0]['source_sha256']=evidence.sha256(raw)
    assert formal.certified_profile_files([config],[raw])=={raw}


def test_formal_cannot_change_prior_or_choose_different_dynamic_calibration_histories(prepared_inputs):
    path,out = prepared_inputs; manifest = formal.read(path)
    config_path = Path(manifest['configs']['alpaca']['low']['pdblend'])
    config = formal.read(config_path)
    save(config_path,dict(config,output_prior=3))
    with pytest.raises(ValueError,match='policy changed'):
        formal.generate(path,out)
    save(config_path,config)
    manifest.pop('dynamic_prior_dataset'); save(path,manifest)
    with pytest.raises(ValueError,match='dynamic_prior_dataset'):
        formal.generate(path,out)


def test_method_selection_winner_is_rechecked_from_explicit_measurements(tmp_path,monkeypatch):
    from test_method_selection import experiment,complete
    manifest,out=experiment.__wrapped__(tmp_path)
    prepared=method_selection.prepare(manifest,out)
    complete(prepared)
    selection_path=out/'selection.json'
    method_selection.summarize(out/'prepared.json')
    winner,config,files=formal.checked_selection(out/'prepared.json',selection_path)
    assert winner=='pdblend-joint' and config['strategy']==winner
    selected=formal.read(selection_path);selected['selected_variant']='pdblend-greedy';save(selection_path,selected)
    with pytest.raises(ValueError,match='winner was changed'):formal.checked_selection(out/'prepared.json',selection_path)


def test_formal_requires_the_executed_collection_summary(prepared_inputs):
    path,out=prepared_inputs;manifest=formal.read(path);manifest.pop('mechanism_collection')
    save(path,manifest)
    with pytest.raises(ValueError,match='collector summary'):formal.generate(path,out)


def test_formal_rejects_collection_without_the_registry_digest(prepared_inputs):
    path,out=prepared_inputs;manifest=formal.read(path);collection=Path(manifest['mechanism_collection'])
    value=formal.read(collection);value.pop('registry_sha256');save(collection,value)
    with pytest.raises(ValueError,match='collector result'):formal.generate(path,out)


def authorize_extended_budget(tmp_path,campaign,*,elapsed=90000,limit=200000):
    from ecopadg.serving.budget import append_extension
    now=time.time()-1;started=now-elapsed
    save(campaign/'budget.json',dict(started_s=started,limit_s=86400))
    authority=save(tmp_path/'synthetic-authorization.json',dict(
        instruction='SYNTHETIC TEST ONLY: extend this test campaign',scope='unit test',
        authorized_at_s=now,original_started_s=started,original_limit_s=86400,
        original_deadline_s=started+86400))
    return append_extension(campaign,authorization_path=authority,limit_s=limit,
        reason='synthetic integration test',now=now)


def test_formal_can_prepare_after_original_deadline_only_with_a_real_ledger_revision(prepared_inputs,tmp_path):
    path,out=prepared_inputs;manifest=formal.read(path);campaign=Path(manifest['campaign_root'])
    approved=authorize_extended_budget(tmp_path,campaign)
    result=formal.generate(path,out)
    assert result['budget_at_preparation']['revision_seq']==1
    assert result['budget_at_preparation']['original_limit_s']==86400
    assert result['budget_at_preparation']['deadline_s']==approved['deadline_s']
    assert formal.read(result['campaign'])['budget_s']==200000
    assert len(result['jobs'])==180 and result['full_target_groups']==30
    freeze=formal.read(result['freeze'])
    assert not any(p.endswith('/budget.json') for p in freeze['files'])
    assert set(approved['revision_artifacts'])<=set(freeze['groups']['protocol'])
    assert all(freeze['files'][p]==digest for p,digest in approved['revision_artifacts'].items())
    assert result['budget_at_preparation']['revision_artifacts']==approved['revision_artifacts']


def test_formal_rejects_unapproved_mutable_budget_increase(prepared_inputs):
    path,out=prepared_inputs;manifest=formal.read(path);campaign=Path(manifest['campaign_root'])
    save(campaign/'budget.json',dict(started_s=time.time()-90000,limit_s=200000))
    with pytest.raises(ValueError,match='authorized revisions'):formal.generate(path,out)


def test_group_rechecks_extension_and_records_revision_without_freeze_drift(prepared_inputs,tmp_path,monkeypatch):
    path,out=prepared_inputs;manifest=formal.read(path);campaign=Path(manifest['campaign_root'])
    approved=authorize_extended_budget(tmp_path,campaign)
    result=formal.generate(path,out);group_path=next(g['path'] for g in result['groups'] if g['selected'])
    from ecopadg.serving.budget import append_extension
    authority=tmp_path/'synthetic-authorization.json'
    changed=append_extension(campaign,authorization_path=authority,limit_s=240000,reason='second synthetic envelope')
    calls=[]
    async def fake_job(path):calls.append(path)
    monkeypatch.setattr(formal,'run_job',fake_job)
    asyncio.run(formal.run_group(group_path))
    assert len(calls)==6
    records=list((out/'budget-checks').glob('*.json'));assert len(records)==1
    audit=formal.read(records[0])
    assert audit['passed'] is True and audit['revision_seq']==2 and audit['limit_s']==240000
    assert audit['authorization_sha256']==changed['authorization_sha256']
    assert audit['revision_artifacts']==changed['revision_artifacts']
    assert evidence.formal_freeze_gaps(formal.read(result['freeze']))==[]
