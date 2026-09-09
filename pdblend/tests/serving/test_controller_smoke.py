import asyncio
from copy import deepcopy
import csv
import itertools
import json
from pathlib import Path
import time

import pytest
from ecopadg.serving import controller_smoke as smoke
from test_transfer_power import instant
from test_role_profiling import fixture as role_fixture
from ecopadg.serving.role_profiling import costs as role_costs


def save(path,value):
    smoke.write(path,value)
    return str(path)


@pytest.fixture
def manifest(tmp_path):
    image='sha256:'+'a'*64
    proof=save(tmp_path/'proof.json',dict(complete=True))
    profiles=save(tmp_path/'profiles.json',dict(schema=2,measurement='hardware',
        status='validated_envelope',engine_image=image,model='Qwen2.5-14B-Instruct',
        frequency_commands_verified=True,heldout_calibration_complete=True,mixed_interference_measured=True,
        resident_idle_measured=True,instant_prefill_calibration_complete=True,instant_heldout_calibration_complete=True,
        certification_artifacts=smoke.freeze_files([proof]),points=[dict(role=role,tp=1,frequency_mhz=2520,
            input_tokens=8192,context_tokens=8192,batch=8,prefill_s=.05,iteration_s=.02,power_w=140,
            residency_w=30,error_fraction=.1,samples=3,source_sha256=smoke.sha256(proof))
            for role in ('mixed','prefill','decode')]))
    transfers=save(tmp_path/'transfers.json',dict(certified=True,instant_power_costs_verified=True,
        receiver_transfer_energy_included=True,engine_image=image,
        certification_artifacts=smoke.freeze_files([proof]),links=[dict(source_tp=1,target_tp=1,
        max_input_tokens=8192,seconds_upper=.1,incremental_j=2,validated=True,
        source_sha256=smoke.sha256(proof))]))
    raw_clock=save(tmp_path/'clock/raw.json',dict(instant([(1,[100]*8),(2,[100]*8)]),complete=True,prefix_matches_reference=True,
        sampling_error=None,frequency_samples=[[1,[2520]*8]],engine_provenance=[dict(image_id=image)],
        switches=[dict(tp=1,source_mhz=900,target_mhz=2520,started_s=1,finished_s=1.01,energy_j=1)]))
    costs=save(tmp_path/'clock/costs.json',[dict(tp=1,source_mhz=900,target_mhz=2520,
        duration_upper_s=.02,energy_upper_j=2,source_sha256=smoke.sha256(raw_clock))])
    role_value=role_fixture()
    for phase in ('provenance_before','provenance_after'): role_value[phase][0]['image_id']=image
    role_raw=save(tmp_path/'role/raw.json',role_value)
    roles=save(tmp_path/'role/costs.json',role_costs(role_value,smoke.sha256(role_raw)))
    template=save(tmp_path/'engine.json',dict(model='/models/Qwen2.5-14B-Instruct',max_model_len=8192))
    save(tmp_path/'campaign/budget.json',dict(started_s=time.time()-20,limit_s=86400))
    return dict(profiles=profiles,transfers=transfers,frequency_costs=[costs],role_costs=roles,
        image=image,engine_template=template,campaign_root=str(tmp_path/'campaign'),
        ownership_root=str(tmp_path),runtime_dir=str(tmp_path/'runtime'),
        instances=[dict(id=f'i{g}',tp=1,gpus=[g],port=18010+g,kv_port=19010+g,role='mixed') for g in range(4)])


def rows_and_summary():
    reference=dict(tokens={str(n):[n]*64 for n in set(smoke.INPUTS)})
    rows=[dict(request_id=str(i),success=1,input_tokens=n,generated_tokens=64,token_ids_verified=1,
        output_token_sha256=smoke.token_hash(reference['tokens'][str(n)]),token_itl_exact=1,
        token_itl_s=json.dumps([.02]*63),slo_ok=1) for i,n in enumerate(smoke.INPUTS)]
    summary=dict(n_expected=8,completed=8,generated_tokens=512,gpu_count=8,validity='ok',
        split='development',formal_eligible=False,power_mode='instant',power_field_id=186,
        power_source_id='nvml:field:186:scope:0:mW',power_source_verified=True,slo_attainment=1)
    return reference,rows,summary


def test_cpu_generation_has_six_fixed_cells_and_one_bounded_campaign_stage(manifest,tmp_path,monkeypatch):
    monkeypatch.setattr(smoke,'PynvmlBackend',lambda **k:pytest.fail('CPU generate accessed GPU'))
    setup=smoke.generate(manifest,tmp_path/'setup')
    trace=smoke.read(setup['trace'])
    assert len(trace['requests'])==8 and [r['prompt_len'] for r in trace['requests']]==list(smoke.INPUTS)
    assert [r['arrival_s'] for r in trace['requests']]==[i*5 for i in range(8)]
    assert all(r['output_len']==64 for r in trace['requests'])
    assert trace['seed']==77 and trace['split']=='development'
    expected=dict(mixed=['mixed']*4,mixed_dvfs=['mixed']*4,distserve=['prefill']+['decode']*3)
    for strategy,path in setup['configs'].items():
        config=smoke.read(path)
        assert config['power_mode']=='instant' and config['node_gpus']==list(range(8))
        assert config['slow_topology'] is False and 'topology' not in config
        assert [i['role'] for i in config['instances']]==expected.get(strategy,['prefill','decode','decode','mixed'])
    stages=smoke.read(tmp_path/'setup/campaign.json')['stages']
    assert len(stages)==1 and stages[0]['limit_s']==900
    assert setup['status']=='prepared_not_executed'


@pytest.mark.parametrize('defect',['average_profile','unproven_transfer','unmeasured_placement','too_many','tp2','outside_owner','deadline'])
def test_incomplete_prerequisites_never_generate_gpu_campaign(manifest,tmp_path,defect):
    if defect=='average_profile':
        path=Path(manifest['profiles']);value=smoke.read(path);value.pop('instant_prefill_calibration_complete');save(path,value)
    if defect=='unproven_transfer':
        path=Path(manifest['transfers']);value=smoke.read(path);value['certified']=False;save(path,value)
    if defect=='unmeasured_placement':
        path=Path(manifest['transfers']);value=smoke.read(path)
        value['links'][0].update(source_gpus=[0],target_gpus=[1]);save(path,value)
    if defect=='too_many': manifest['instances']=manifest['instances'][:3]
    if defect=='tp2': manifest['instances'][0].update(tp=2,gpus=[0,7])
    if defect=='outside_owner': manifest['runtime_dir']='/somewhere-else'
    if defect=='deadline': manifest['limit_s']=901
    with pytest.raises(ValueError): smoke.generate(manifest,tmp_path/'setup')
    assert not (tmp_path/'setup').exists()


def test_prefill_coverage_queries_input_plus_one_not_decode_tail(manifest,tmp_path):
    path=Path(manifest['profiles']);value=smoke.read(path)
    for point in value['points']:
        if point['role']=='prefill': point.update(input_tokens=7168,context_tokens=7169)
    save(path,value)
    assert smoke.generate(manifest,tmp_path/'setup')['status']=='prepared_not_executed'


@pytest.mark.parametrize('receiver_proof',[None,False,1,'true'])
def test_smoke_rejects_transfer_costs_without_explicit_receiver_energy(manifest,tmp_path,receiver_proof):
    path=Path(manifest['transfers']);value=smoke.read(path)
    if receiver_proof is None: value.pop('receiver_transfer_energy_included')
    else: value['receiver_transfer_energy_included']=receiver_proof
    save(path,value)
    with pytest.raises(ValueError,match='certified TP1 transfer'):
        smoke.generate(manifest,tmp_path/'setup')
    assert not (tmp_path/'setup').exists()


class Profiler:
    def __init__(self,stale=False):
        self.stale=stale
        self.states={f'i{g}':dict(generation=10,acknowledged_generation=10,role='decode',mode='continuous',
            admit_prefill=True,admit_decode=True,accepting=True) for g in range(4)}
    async def call(self,instance,path): return dict(self.states[instance['id']])
    async def control(self,instance,**changes):
        value=self.states[instance['id']];value.update(changes,generation=value['generation']+1)
        if not self.stale: value['acknowledged_generation']=value['generation']


def test_role_reset_preserves_monotonic_versions_and_requires_engine_ack(manifest):
    profiler=Profiler()
    result=asyncio.run(smoke.reset_roles(profiler,manifest['instances']))
    assert len(result)==4 and all(r['after']['generation']==11 for r in result)
    assert all(r['after']['acknowledged_generation']==11 and r['after']['role']=='mixed' for r in result)
    with pytest.raises(RuntimeError,match='unconfirmed'):
        asyncio.run(smoke.reset_roles(Profiler(stale=True),manifest['instances']))


@pytest.mark.parametrize('defect',[None,'tokens','duplicate','average','missing_itl','failure_denominator'])
def test_audit_requires_reference_identity_exact_tokens_and_failure_denominator(defect):
    reference,rows,summary=rows_and_summary()
    if defect=='tokens': rows[0]['output_token_sha256']='other'
    if defect=='duplicate': rows[1]['request_id']='0'
    if defect=='average': summary['power_mode']='average'
    if defect=='missing_itl': rows[0]['token_itl_exact']=0
    if defect=='failure_denominator':
        rows[0].update(success=0,slo_ok=0);summary['completed']=7
        # Incorrectly omitting the failed request produces 1.0 instead of 7/8.
    result=smoke.audit_cell(summary,rows,reference,[],[[1,[2520]*8],[2,[1500]*8]])
    assert result['passed'] is (defect is None)
    if defect=='failure_denominator': assert not result['failure_denominator_verified']
    assert len(result['actual_clock_actions']['observed_changes'])==1


@pytest.mark.parametrize('fail_strategy',[None,'distserve'])
def test_run_uses_existing_cells_and_restores_mixed_even_on_failure(manifest,tmp_path,monkeypatch,fail_strategy):
    smoke.generate(manifest,tmp_path/'setup')
    reference,rows,summary=rows_and_summary();calls=[]
    async def restore(config,out): calls.append(('physical',len(config['instances'])))
    async def reset(profiler,instances):
        calls.append(('roles',[i['role'] for i in instances]));return [{'confirmed':True}]*4
    async def sources(profiler,setup): return [{'image_id':setup['image'],'source_files_at_import':setup['source_files']}]*4
    async def refs(*a): return reference
    async def observe(stop,values):
        values.extend([[1,[2520]*8],[2,[1500]*8]]);await stop.wait()
    async def cell(args):
        strategy=smoke.read(args.config)['strategy'];calls.append(('cell',strategy))
        assert args.split=='development' and args.timeout==60
        if strategy==fail_strategy: raise RuntimeError('injected cell failure')
        args.out.mkdir()
        with (args.out/'bench.csv').open('w') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        (args.out/'control.jsonl').write_text('')
        save(args.out/'summary.json',summary)
        return deepcopy(summary)
    for name,value in [('restore_layout',restore),('reset_roles',reset),('engine_sources',sources),
                       ('references',refs),('clock_observer',observe),('run_cell',cell)]:
        monkeypatch.setattr(smoke,name,value)
    if fail_strategy:
        with pytest.raises(RuntimeError,match='smoke incomplete'):
            asyncio.run(smoke.run(tmp_path/'setup',tmp_path/'run'))
    else:
        result=asyncio.run(smoke.run(tmp_path/'setup',tmp_path/'run'))
        assert result['passed'] and result['status']=='controller_smoke_passed'
        assert set(result['cells'])==set(smoke.STRATEGIES)
    report=smoke.read(tmp_path/'run/summary.json')
    assert report['formal_eligible'] is False and report['capacity_certified'] is False
    assert calls[-1]==('roles',['mixed']*4)
    assert len(report['cells'])==(2 if fail_strategy else 6)
