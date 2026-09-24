from copy import deepcopy
import json
from pathlib import Path

import pytest

from pdblend.profile.collection import native_timing_plan_v2 as plan_module
from pdblend.profile.collection import native_timing_capacity as module
from pdblend.profile.collection.native_timing_plan import binding
from test_comparison_acceptance import state
from test_pdblend_native_timing import component_rows


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value));return binding(path)


@pytest.fixture
def plans(tmp_path):
    entries=[];corpora={}
    for index,(model,tp) in enumerate(plan_module.MODEL_TP.items()):
        entries.append(dict(model_id=model,dataset='alpaca',rate_scale=.5,selection_split='tuning',tp=tp,pp=1,
            min_m_instances=4,engine_max_num_seqs=32,frequency_scope=[1500,2520],status='development_enumeration',
            queries=[dict(method='decode_supported',args=[1.,121.+index*5,1500],kwargs={},finite_arguments=True),
                dict(method='step_seconds',args=[20.5,7100.+index*20,2520],kwargs={},finite_arguments=True),
                dict(method='step_seconds',args=[33.1,7000.,2520],kwargs={},finite_arguments=True),
                dict(method='prefill_seconds',args=[75.+index,1500],kwargs={},finite_arguments=True),
                dict(method='prefill_marginal_seconds',args=[0,1500],kwargs={},finite_arguments=True)]))
        corpora[model]={}
        for dataset in plan_module.DATASETS:
            corpus=dict(model_name=model,dataset=dataset,
                calibration=[dict(input_tokens=n+index,output_tokens=64) for n in (16,128,512,4096,7166)],
                tuning=[dict(input_tokens=n+index,output_tokens=32) for n in (32,256,1024,6144)],
                evaluation=[dict(input_tokens=99999,output_tokens=99999)])
            corpora[model][dataset]=put(tmp_path/model/(dataset+'.json'),corpus)
    entries.append(dict(model_id='Qwen2.5-32B-Instruct',dataset='longbench',
        status='blocked_missing_tuning_anchor',queries=[],formal_eligible=False))
    ledger=put(tmp_path/'ledger.json',dict(schema='pdblend-offline-query-ledger/v2',evaluation_read=False,
        min_m_floor_overridden=False,ledgers=entries))
    provenance=put(tmp_path/'provenance.json',dict(evaluation_read=False,outputs={'ledger':ledger}))
    return dict(ledger=ledger,provenance=provenance,corpora=corpora,
        plans={model:plan_module.build_plan(ledger,provenance,refs,model_id=model) for model,refs in corpora.items()})


def test_model_owned_three_topologies_preserve_native_boundaries_and_explicit_query_gaps(plans):
    rows=plans['plans'];assert len(rows)==3
    for model,tp in plan_module.MODEL_TP.items():
        plan=plan_module.validate_plan(rows[model]);assert plan['tp']==tp and plan['resident_instances']==8//tp
        assert not plan['formal_eligible'] and plan['collector_integration_required']
        assert {r['frequency_mhz'] for r in plan['points']}=={1500,2520}
        assert any(r['batch']==32 and r['prompt_tokens']>7000 for r in plan['points'])
        assert all(r['prompt_tokens']+r['output_tokens']<=8192 for r in plan['points'])
        assert all(r['repeats']==3 for r in plan['points'])
        training={(p['role'],p['batch'],p['prompt_tokens']) for p in plan['points'] if p['purpose']=='training'}
        holdout={(p['role'],p['batch'],p['prompt_tokens']) for p in plan['points'] if p['purpose']=='holdout'}
        assert not training&holdout
        assert plan['unsupported_queries'][0]['shape']['batch']==33.1 and not plan['unsupported_queries'][0]['clamped']
        assert plan['zero_increment_identities'] and not plan['query_coverage_complete']
    assert rows['Qwen2.5-32B-Instruct']['blocked_ledger_entries'][0]['dataset']=='longbench'
    shapes=lambda model:{p['prompt_tokens'] for p in rows[model]['points']}
    assert shapes('Qwen2.5-7B-Instruct')!=shapes('Qwen2.5-14B-Instruct')


@pytest.mark.parametrize('kind',['other_model','missing_split','min_m','tp','evaluation','mutated_policy','mutated_holdout'])
def test_plan_rejects_cross_model_or_modified_selection(plans,kind):
    x=plans;model='Qwen2.5-14B-Instruct';refs=deepcopy(x['corpora'][model])
    if kind in ('mutated_policy','mutated_holdout'):
        plan=deepcopy(x['plans'][model])
        if kind=='mutated_policy':plan['capacity_policy']['strict_limit']='trust it'
        else:next(p for p in plan['points'] if p['purpose']=='holdout')['purpose']='training'
        with pytest.raises(ValueError):plan_module.validate_plan(plan)
        return
    if kind=='other_model':refs=x['corpora']['Qwen2.5-7B-Instruct']
    elif kind=='missing_split':
        ref=refs['alpaca'];path=Path(ref['path']);corpus=json.loads(path.read_text());corpus.pop('tuning')
        refs['alpaca']=put(path,corpus)
    else:
        path=Path(x['ledger']['path']);ledger=json.loads(path.read_text())
        row=next(r for r in ledger['ledgers'] if r['model_id']==model)
        if kind=='min_m':row['min_m_instances']=2
        elif kind=='tp':row['tp']=2
        else:row['selection_split']='evaluation'
        x['ledger']=put(path,ledger);p=Path(x['provenance']['path'])
        x['provenance']=put(p,dict(evaluation_read=False,outputs={'ledger':x['ledger']}))
    with pytest.raises(ValueError):plan_module.build_plan(x['ledger'],x['provenance'],refs,model_id=model)


def capacity_fixture(plan,*,total=1024):
    point=dict(next(p for p in plan['points'] if p['batch']==32 and p['purpose']=='training'),repeat=0)
    identity=dict(model_id=plan['model_id'],tp=plan['tp'],pp=1,model_hash='model',tokenizer_hash='tokenizer',
        source_revision='source',image_digest='image',engine_revision='vllm-0.10.1.1',
        gpu_uuids=[f'GPU-{i}' for i in range(plan['tp'])])
    idle=state(plan['tp'],100.,5);idle.update(total_kv_tokens=total,free_kv_tokens=total)
    return point,dict(identity=identity,capability=dict(identity,supported=True,state=deepcopy(idle)),
        capability_received_s=100.,drain=dict(idle,acknowledged=True,drained=True),drain_received_s=100.,observed_s=100.1)


@pytest.mark.parametrize('model',list(plan_module.MODEL_TP))
def test_unsupported_requires_fresh_real_native_capacity_and_reproduced_strict_integer_guard(plans,model):
    plan=plans['plans'][model];point,args=capacity_fixture(plan)
    raw=module.unsupported_capacity_record(plan,point,**args)
    replay=module.audit_unsupported_capacity(raw,plan=plan,identity=args['identity'])
    assert replay['reserved_tokens']>=replay['requested_tokens'] and replay['supported'] is False
    assert raw['client_requests']==[] and raw['sample']==dict(ranks=[]) and raw['timing_measured'] is False
    _,large=capacity_fixture(plan,total=10**7)
    assert module.capacity_decision(plan,point,**large)['supported']
    with pytest.raises(ValueError,match='supported point'):module.unsupported_capacity_record(plan,point,**large)


def test_capacity_threshold_equality_is_unsupported_without_float_rounding(plans):
    plan=plans['plans']['Qwen2.5-7B-Instruct']
    def reserve(p):return p['batch']*((p['prompt_tokens']+p['output_tokens']+15)//16)*16
    point=next(p for p in plan['points'] if reserve(p)%144==0)
    total=reserve(point)*10//9;_,args=capacity_fixture(plan,total=total)
    result=module.capacity_decision(plan,dict(point,repeat=0),**args)
    assert result['lhs_10_reserved']==result['rhs_9_actual_total'] and not result['supported']
    _,args=capacity_fixture(plan,total=total+16)
    assert module.capacity_decision(plan,dict(point,repeat=0),**args)['supported']


@pytest.mark.parametrize('kind',['stale','busy','generation','rank','UUID','identity','lying_total','client',
    'measured','error','cleanup','sampler_error','claimed_decision','unplanned','native128','context4096'])
def test_capacity_label_cannot_hide_other_failure_or_existing_measurement(plans,kind):
    plan=plans['plans']['Qwen2.5-32B-Instruct'];point,args=capacity_fixture(plan)
    raw=module.unsupported_capacity_record(plan,point,**args)
    if kind=='stale':raw['observed_s']=110.
    elif kind=='busy':raw['capability']['state']['running']=['busy']
    elif kind=='generation':raw['capability']['state']['generation']=6
    elif kind=='rank':raw['capability']['state']['ranks'].pop()
    elif kind=='UUID':raw['capability']['gpu_uuids'][0]='GPU-wrong'
    elif kind=='identity':raw['capability']['model_id']='Qwen2.5-7B-Instruct'
    elif kind=='lying_total':raw['capability']['state']['total_kv_tokens']=999999
    elif kind=='client':raw['client_requests']=[dict(error='timeout')]
    elif kind=='measured':raw['start_s']=100.
    elif kind=='error':raw['error']='TimeoutError'
    elif kind=='cleanup':raw['cleanup_errors']=['drain failed']
    elif kind=='sampler_error':raw['sampler_error']='NVML failed'
    elif kind=='claimed_decision':raw['capacity_decision']['rhs_9_actual_total']=0
    elif kind in ('native128','context4096'):
        key,value=('max_num_seqs',128) if kind=='native128' else ('max_model_len',4096)
        raw['capability']['state'][key]=raw['drain'][key]=value
    else:raw['point']['prompt_tokens']+=1
    with pytest.raises(ValueError):module.audit_unsupported_capacity(raw,plan=plan,identity=args['identity'])


def test_partial_domain_never_invents_coefficients_and_fit_hull_uses_measured_rows_only():
    training,holdout=component_rows();partition=dict(training=training,holdout=holdout,
        unsupported=[dict(point_sha256='unmeasured-far-away',batch=32,context_tokens=8192)])
    kwargs=dict(identity=dict(system='pdblend',model_id='Qwen2.5-32B-Instruct',tp=2,pp=1),raw_bindings=[],
        measurement_qualification=dict(qualified=True),limits=dict(mean_relative_error=.1,p95_relative_error=.2,max_relative_error=.25))
    result=module.fit_measured_partition(partition,**kwargs)
    assert result['component_qualified'] and not result['formal_eligible']
    decode=next(r for r in result['component']['models'] if r['role']=='decode')
    assert max(v[1] for v in decode['coverage_vertices'])==.5
    missing=dict(partition,holdout=[r for r in holdout if r['frequency_mhz']==1500])
    result=module.fit_measured_partition(missing,**kwargs)
    assert not result['component_qualified'] and result['component'] is None and result['missing_domains']
    deficient=dict(partition,training=[r for r in training if r['role']!='decode' or r['batch']==1])
    result=module.fit_measured_partition(deficient,**kwargs)
    assert result['component'] is None


def test_partition_requires_every_planned_capacity_receipt_and_never_converts_failure(plans):
    plan=plans['plans']['Qwen2.5-7B-Instruct'];identities={};windows=[]
    for i in range(plan['resident_instances']):
        _,args=capacity_fixture(plan);identity=deepcopy(args['identity']);identity['gpu_uuids']=[f'GPU-{i}']
        identities[f'pd-timing-{i}']=identity
    for i,point in enumerate(plan['points']):
        iid=f'pd-timing-{i%len(identities)}'
        for repeat in range(point['repeats']):
            _,args=capacity_fixture(plan,total=16);args['identity']=identities[iid]
            args['capability'].update(identities[iid])
            raw=module.unsupported_capacity_record(plan,dict(point,repeat=repeat),**args)
            windows.append(dict(instance_id=iid,raw=raw))
    partition=module.partition_windows(plan,windows,identities=identities)
    assert not partition['training'] and not partition['holdout'] and len(partition['unsupported'])==len(windows)
    with pytest.raises(ValueError,match='incomplete'):module.partition_windows(plan,windows[:-1],identities=identities)
    wrong=deepcopy(identities);wrong['pd-timing-1']['gpu_uuids']=wrong['pd-timing-0']['gpu_uuids']
    with pytest.raises(ValueError,match='physical fleet'):module.partition_windows(plan,windows,identities=wrong)
    windows[0]['raw']['status']='failed';windows[0]['raw']['error']='timeout'
    with pytest.raises((ValueError,KeyError)):module.partition_windows(plan,windows,identities=identities)
