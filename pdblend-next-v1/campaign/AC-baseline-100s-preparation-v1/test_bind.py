import copy
import json
from pathlib import Path
import pytest
from bind import ROOT,configuration,read,PROTOCOL
from child_entry import cell_args

@pytest.mark.parametrize('model,strategy',[('14b','mixed'),('14b','distserve'),('14b','ecoserve'),('14b','dynamollm'),('7b','mixed'),('7b','distserve'),('7b','ecoserve'),('7b','dynamollm-resident')])
def test_all_dataset_scale_configs_keep_native_algorithm(model,strategy):
    for dataset in ('alpaca','sharegpt','longbench'):
        record=next(r for r in read(ROOT/'historical-index.json')['records'] if
            (r['model'],r['dataset'],r['strategy'])==(model,dataset,strategy))
        old=read(ROOT/record['template'])
        for scale in (.5,1.,2.):
            new,note=configuration(model,dataset,strategy,scale=scale)
            changed={k for k in set(old)|set(new) if old.get(k)!=new.get(k)}
            assert changed <= {'evaluation_protocol','measurement_window_protocol','experiment_protocol',
                'arrival_window_s','slo_scale','slo_ttft_s','slo_tpot_s','slo_attainment_target'}
            assert new['instances']==old['instances'] and new['output_prior']==old['output_prior']
            assert note['binding_is_live_readiness'] is False

def test_c_cannot_relabel_resident_as_full():
    with pytest.raises(ValueError,match='no historical implementation'):
        configuration('7b','alpaca','dynamollm')

def deployment(model,strategy,dataset):
    cfg,_=configuration(model,dataset,strategy)
    for i,instance in enumerate(cfg['instances']):
        instance.update(id='new'+str(i),container_name='pdb-next-new'+str(i),port=35000+i,kv_port=45000+32*i,url='http://127.0.0.1:'+str(35000+i))
    return dict(model=model,historical_engine_observation_only=True,
        layouts={strategy+':'+dataset:cfg['instances']},topology_source_path_verified=True)

def test_distserve_actual_three_placements_and_rebind():
    p=[]
    for d in ('alpaca','sharegpt','longbench'):
        c,_=configuration('14b',d,'distserve',deployment=deployment('14b','distserve',d))
        p.append([(i['tp'],i['gpus']) for i in c['instances'] if i['role']=='prefill'])
        if d=='longbench':assert [(i['tp'],i['gpus']) for i in c['instances'] if i['role']=='decode']==[(2,[6,7])]
    assert list(map(len,p))==[1,2,5]

@pytest.mark.parametrize('bad',['layout','role','duplicate_port','source','model'])
def test_rebinding_failures(bad):
    dep=deployment('14b','dynamollm','alpaca');layout=dep['layouts']['dynamollm:alpaca']
    if bad=='layout':layout.pop()
    elif bad=='role':layout[0]['role']='decode'
    elif bad=='duplicate_port':layout[1]['kv_port']=layout[0]['port']
    elif bad=='source':dep['topology_source_path_verified']=False
    elif bad=='model':dep['model']='7b'
    with pytest.raises(ValueError):configuration('14b','alpaca','dynamollm',deployment=dep)

def test_dynamo_buckets_remap_without_reclassification():
    old,_=configuration('14b','sharegpt','dynamollm')
    new,_=configuration('14b','sharegpt','dynamollm',deployment=deployment('14b','dynamollm','sharegpt'))
    assert list(new['dynamo_assignments'].values())==list(old['dynamo_assignments'].values())
    assert new['topology_costs']==old['topology_costs']

def test_real_args_keep_actual_controller_scoring_identical(tmp_path):
    c,_=configuration('7b','sharegpt','ecoserve',scale=.5)
    t=dict(protocol_id=PROTOCOL,measurement_schema=3,seed=701,arrival_window_s=100,duration_s=100,split='development',dataset='sharegpt',load='declared')
    cp=tmp_path/'c.json';tp=tmp_path/'t.json';cp.write_text(json.dumps(c));tp.write_text(json.dumps(t))
    a=cell_args(cp,tp,tmp_path/'out',scale=.5)
    assert a.slo_ttft_s==2.5 and a.slo_tpot_s==.075 and a.timeout==120 and a.strategy is None
    c['slo_ttft_s']=5;cp.write_text(json.dumps(c))
    with pytest.raises(ValueError,match='actual Controller SLO'):cell_args(cp,tp,tmp_path/'out',scale=.5)
