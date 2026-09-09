from dataclasses import asdict
import json
from pathlib import Path
import time

import pytest

from ecopadg.serving import campaign_followup_setup as follow


def test_templates_need_no_measurements_and_declare_all_independent_work(tmp_path):
    plan=follow.templates(tmp_path/'missing-campaign',tmp_path/'templates')
    cal=follow.read(plan['calibration_template']);methods=follow.read(plan['methods_template'])
    assert plan['status']=='template_only_no_measurements_or_calibration'
    assert len(cal['entries'])==15 and all(e['candidate_index']==0 for e in cal['entries'])
    assert all('-instant/frequency_costs.json' in p for p in cal['frequency_costs'])
    assert Path(cal['transfers']).name=='transfers.certified-v2.json'
    assert methods['layout']==dict(tp=1,prefill=1,decode=4,mixed=3)
    assert methods['seeds']==[11,22] and {p['dataset'] for p in methods['points']}==set(follow.DATASETS)
    assert methods['requests']==64
    assert not (Path(plan['calibration_out'])/'summary.json').exists()


def test_initial_layout_requires_passed_evidence_and_stays_inside_node(tmp_path):
    proof=tmp_path/'raw.json'
    item=dict(instance_id='prior',tp=1,gpus=[0],port=18000,kv_port=19000)
    follow.write(proof,dict(passed=False,live_instances=[item]))
    with pytest.raises(ValueError,match='complete and passed'):
        follow.initial_layout(dict(initial_layout_evidence=str(proof)))
    follow.write(proof,dict(passed=True,complete=True,live_instances=[item]))
    assert follow.initial_layout(dict(initial_layout_evidence=str(proof)))[0]['id']=='prior'
    with pytest.raises(ValueError):follow.initial_layout(dict(initial_instances=[dict(item,gpus=[8])]))


def test_baseline_continuation_preserves_explicit_initial_rate_without_pdb_selection(tmp_path,monkeypatch):
    plan=follow.templates(tmp_path/'campaign',tmp_path/'templates')
    cal=follow.read(plan['calibration_template'])
    cal['initial_instances']=[dict(id='prior',tp=1,gpus=[0],port=18000,kv_port=19000)]
    cal['entries'][0]['initial_rate']=.375
    follow.write(plan['calibration_template'],cal)
    from ecopadg.serving import calibration_setup
    monkeypatch.setattr(calibration_setup,'generate',lambda manifest,out:manifest)
    assert follow.prepare_calibration(plan)['entries'][0]['initial_rate']==.375
    cal['entries'].pop();follow.write(plan['calibration_template'],cal)
    with pytest.raises(ValueError,match='fifteen'):follow.prepare_calibration(plan)


def test_role_costs_require_all_directions_and_instant_calibration_separate_from_correctness(tmp_path):
    from test_role_profiling import fixture as role_fixture
    from ecopadg.serving.role_profiling import costs as verified_costs
    from ecopadg.serving.evidence import sha256
    root=tmp_path/'role';root.mkdir();image='sha256:'+'a'*64
    raw=role_fixture()
    for phase in ('provenance_before','provenance_after'): raw[phase][0]['image_id']=image
    follow.write(root/'raw.json',raw)
    costs=verified_costs(raw,sha256(root/'raw.json'))
    follow.write(root/'role_costs.json',costs)
    assert len(follow.role_evidence([root/'role_costs.json'],image,{1})[0])==6
    with pytest.raises(ValueError,match='no measured cost'):
        follow.role_evidence([root/'role_costs.json'],image,{1,2})
    costs[0]['energy_upper_j']=1;follow.write(root/'role_costs.json',costs)
    with pytest.raises(ValueError,match='differs from its measured'):follow.role_evidence([root/'role_costs.json'],image,{1})
    raw['power_source']['mode']='average';follow.write(root/'raw.json',raw)
    with pytest.raises(ValueError,match='instant resident role'):follow.role_evidence([root/'role_costs.json'],image,{1})


def fixture_methods(tmp_path,monkeypatch):
    from ecopadg.serving import calibration_setup,calibration
    from ecopadg.serving.evidence import sha256
    from ecopadg.serving.profiles import ProfilePoint
    from ecopadg.serving.interconnect import InterconnectTopology
    plan=follow.templates(tmp_path/'campaign',tmp_path/'templates')
    follow.write(tmp_path/'campaign/budget.json',dict(started_s=time.time()-10,limit_s=86400))
    cal=follow.read(plan['calibration_template']);methods=follow.read(plan['methods_template'])
    cal['initial_instances']=[dict(id='prior',tp=1,gpus=[0],port=18000,kv_port=19000)]
    follow.write(plan['calibration_template'],cal)
    methods.update(method_budget_s=60000,requests=4);follow.write(plan['methods_template'],methods)
    code=tmp_path/'source.py';code.write_text('original execution')
    monkeypatch.setattr(calibration,'implementation_sources',lambda:[code])
    source={str(code):sha256(code)}
    dataset_priors=dict(zip(follow.DATASETS,(2,4,8)))
    for dataset in follow.DATASETS:
        records=[dict(input_tokens=16,output_tokens=dataset_priors[dataset],prompt=[13]*16,request_shape_sha256=f'{dataset}-{i}') for i in range(4)]
        follow.write(Path(cal['corpus'])/(dataset+'.json'),dict(calibration=records,development=records,formal='unused'))
    topology_text='\n'.join('GPU'+str(i)+' '+' '.join('X' if i==j else 'PIX' for j in range(8)) for i in range(8))
    Path(cal['interconnect']).write_text(topology_text);topology=InterconnectTopology.parse(topology_text)
    points=[asdict(ProfilePoint(role,tp,f,16,17 if role=='prefill' else 32,1,.01,.01,100,30,0,3,'test-only'))
            for role in ('prefill','decode','mixed') for tp in (1,2) for f in (900,1500,2100,2520)]
    profiles=dict(schema=2,measurement='hardware',points=points)
    follow.write(cal['profiles'],profiles)
    link=dict(source_tp=1,target_tp=1,max_input_tokens=16,seconds_upper=.01,incremental_j=1,
        source_sha256='test-only',validated=True,source_gpus=[0],target_gpus=[1],
        interconnect_class='PIX',topology_sha256=topology.source_sha256)
    transfers=dict(links=[link]);follow.write(cal['transfers'],transfers)
    template=dict(model='/models/Qwen2.5-14B-Instruct',max_model_len=8192);follow.write(cal['engine_template'],template)
    topo=[dict(source_tps=[1],target_tps=[2],duration_upper_s=10,energy_upper_j=100,source_sha256='test-only')]
    proof=tmp_path/'proof.json';follow.write(proof,dict(test_only=True))
    def inputs(manifest):return profiles,transfers,{},template,[dict(test_only=True)],[str(proof)],topo,{}
    monkeypatch.setattr(calibration_setup,'validated_inputs',inputs)
    monkeypatch.setattr(follow,'role_evidence',lambda *args:([dict(test_only=True)],[str(proof)]))
    monkeypatch.setattr(follow,'measured_capacities',lambda *args:([dict(test_only=True)],[str(proof)]))
    cal['topology_costs']=str(tmp_path/'topology/topology_costs.json');follow.write(Path(cal['topology_costs']).parent/'raw.json',{})
    cal['weights_evidence']=str(proof);cal['retained_weights']=str(tmp_path/'weights');follow.write(Path(cal['retained_weights'])/'manifest.json',{})
    follow.write(plan['calibration_template'],cal)
    results=[]
    for dataset in follow.DATASETS:
        for system in follow.BASELINES:
            root=tmp_path/'results'/(system+'-'+dataset)
            config=dict(strategy=system,profiles=cal['profiles'],slo_ttft_s=5,slo_tpot_s=.1,
                output_prior=dataset_priors[dataset],
                instances=[dict(id='control',tp=1,gpus=[0],port=22000,kv_port=23000,role='mixed',url='http://127.0.0.1:22000')])
            if system in ('mixed_dvfs','dynamollm'):config['frequency_costs']=[dict(test_only=True)]
            follow.write(root/'config.json',config);follow.write(root/'source.before.json',source)
            result=dict(system=system,dataset=dataset,config=str(root/'config.json'),passed=True,
                capacity_rps=1000,infeasible_upper_rps=2000,confirmation=dict(rate=1000,validity='ok',split='calibration',
                    completed=128,n_expected=128,slo_attainment=1,artifact=str(root/'confirm/summary.json')))
            follow.write(root/'confirm/summary.json',result['confirmation']);results.append(result)
    follow.write(methods['calibration'],dict(passed=True,source_unchanged=True,results=results))
    follow.write(Path(plan['calibration_out'])/'input-evidence.json',dict(artifacts={cal['profiles']:sha256(cal['profiles'])}))
    captured={}
    from ecopadg.serving import method_selection
    def prepare(path,out):
        captured['manifest']=follow.read(path);captured['out']=str(out)
        return dict(campaign=str(Path(out)/'campaign.json'))
    monkeypatch.setattr(method_selection,'prepare',prepare)
    return plan,captured,profiles


def test_method_setup_has_all_78_paired_cells_fixed_samples_and_measured_cost_inputs(tmp_path,monkeypatch):
    plan,captured,_=fixture_methods(tmp_path,monkeypatch)
    follow.prepare_methods(plan)
    manifest=captured['manifest'];setup=follow.read(Path(plan['method_out'])/'setup.json')
    assert setup['expected_cells']==78 and manifest['requests']==4 and manifest['seeds']==[11,22]
    base=follow.read(manifest['base_config']);control=follow.read(manifest['control_config'])
    assert [i['role'] for i in base['instances']]==['prefill']+['decode']*4+['mixed']*3
    assert base['slow_topology'] is True and base['topology']['image']==follow.IMAGE
    assert all(base[k] for k in ('frequency_costs','role_costs','topology_costs','measured_capacities','retained_weights'))
    assert base['output_prior']==control['output_prior'] and base['node_gpus']==list(range(8))
    assert base['output_priors']==control['output_priors']==dict(zip(follow.DATASETS,(2,4,8)))
    assert base['output_prior_sources']==control['output_prior_sources']==setup['output_prior_sources']
    assert all(p['split']=='calibration' for p in base['output_prior_sources'].values())
    assert base['power_mode']==control['power_mode']=='instant'
    assert setup['layout_scope'].startswith('predeclared development candidate')


def test_method_setup_does_not_trim_groups_when_budget_cannot_fit(tmp_path,monkeypatch):
    plan,_,_=fixture_methods(tmp_path,monkeypatch)
    manifest=follow.read(plan['methods_template']);manifest['method_budget_s']=10
    follow.write(plan['methods_template'],manifest)
    with pytest.raises(ValueError,match='complete development paired groups'):follow.prepare_methods(plan)
    assert not Path(plan['method_out']).exists()


def test_method_setup_rejects_missing_physical_route_or_changed_calibration_source(tmp_path,monkeypatch):
    plan,_,_=fixture_methods(tmp_path,monkeypatch)
    from ecopadg.serving import calibration_setup
    original=calibration_setup.validated_inputs
    def missing_link(manifest):
        items=list(original(manifest));items[1]=dict(links=[]);return tuple(items)
    monkeypatch.setattr(calibration_setup,'validated_inputs',missing_link)
    with pytest.raises(ValueError,match='no certified transfer'):follow.prepare_methods(plan)
    monkeypatch.setattr(calibration_setup,'validated_inputs',original)
    (tmp_path/'source.py').write_text('changed benchmark')
    with pytest.raises(ValueError,match='execution sources changed'):follow.prepare_methods(plan)


def test_development_shapes_cannot_silently_extrapolate_an_unmeasured_tp(tmp_path,monkeypatch):
    plan,_,_=fixture_methods(tmp_path,monkeypatch)
    manifest=follow.read(plan['methods_template']);path=Path(manifest['corpus'])/'longbench.json'
    corpus=follow.read(path);corpus['development'][0]['input_tokens']=8000
    follow.write(path,corpus)
    with pytest.raises(ValueError,match='exceeds a reachable measured'):follow.prepare_methods(plan)


def test_methods_emit_the_actual_common_runner_sealed_78_cell_campaign(tmp_path,monkeypatch):
    from ecopadg.serving import method_selection
    real_prepare=method_selection.prepare
    plan,_,_=fixture_methods(tmp_path,monkeypatch)
    monkeypatch.setattr(method_selection,'prepare',real_prepare)
    methods=follow.read(plan['methods_template']);cal=follow.read(methods['calibration'])
    for row in cal['results']:
        if row['dataset']=='longbench':
            row.update(capacity_rps=.1,infeasible_upper_rps=.2)
            row['confirmation']['rate']=.1
            follow.write(row['confirmation']['artifact'],row['confirmation'])
    follow.write(methods['calibration'],cal)
    prepared=follow.prepare_methods(plan)
    assert len(prepared['jobs'])==78 and len(prepared['groups'])==6
    assert len(prepared['configurations'])==13
    assert prepared['formal_eligible'] is False
    assert not method_selection.check_preparation(prepared)
    expected=sum(group['cell_limit_s']*len(group['jobs']) for group in prepared['groups'])
    assert prepared['stage_upper_bound_s']==expected
    stages=follow.read(prepared['campaign'])['stages']
    assert len({stage['limit_s'] for stage in stages})==2
    assert expected<78*max(stage['limit_s'] for stage in stages)


def test_execute_never_opens_development_stage_after_incomplete_calibration(tmp_path,monkeypatch):
    plan=follow.templates(tmp_path/'campaign',tmp_path/'templates')
    follow.write(tmp_path/'campaign/budget.json',dict(started_s=time.time()-10,limit_s=86400))
    calls=[]
    class Campaign:
        def __init__(self,root,limit):assert root==plan['campaign_root'] and limit==86400
        def run(self,name,argv,limit_s,*,gpu):
            calls.append((name,argv,gpu))
            if name=='prepare-followup-calibration':
                follow.write(Path(plan['calibration_out'])/'campaign.json',dict(stages=[dict(
                    name='independent-calibration',argv=['test-only-no-execution'],limit_s=10,gpu=True)]))
            elif name=='independent-calibration':
                follow.write(Path(plan['calibration_out'])/'summary.json',dict(passed=False))
        def close(self):calls.append(('closed',[],False))
    from ecopadg.serving import campaign
    monkeypatch.setattr(campaign,'Campaign',Campaign)
    with pytest.raises(RuntimeError,match='dependent method stage remains closed'):
        follow.execute(plan,tmp_path/'templates/followup.json')
    assert [c[0] for c in calls]==['prepare-followup-calibration','independent-calibration','closed']
    assert calls[0][1][2]=='ecopadg.serving.campaign_followup_setup'
    assert not follow.read(tmp_path/'templates/execution.status.json')['complete']


def test_execute_does_not_call_failed_method_selection_complete(tmp_path,monkeypatch):
    plan=follow.templates(tmp_path/'campaign',tmp_path/'templates')
    follow.write(tmp_path/'campaign/budget.json',dict(started_s=time.time()-10,limit_s=86400))
    follow.write(Path(plan['calibration_out'])/'campaign.json',dict(stages=[]))
    follow.write(Path(plan['calibration_out'])/'summary.json',dict(passed=True))
    follow.write(Path(plan['method_out'])/'paired/campaign.json',dict(stages=[]))
    class Campaign:
        def __init__(self,*args):pass
        def run(self,*args,**kwargs):pass
        def close(self):pass
    from ecopadg.serving import campaign,method_selection
    monkeypatch.setattr(campaign,'Campaign',Campaign)
    monkeypatch.setattr(method_selection,'summarize',lambda _:dict(selected_variant=None,status='evidence_insufficient'))
    with pytest.raises(RuntimeError,match='no method passed'):
        follow.execute(plan,tmp_path/'templates/followup.json')
    assert not follow.read(tmp_path/'templates/execution.status.json')['complete']
