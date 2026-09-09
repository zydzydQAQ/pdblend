import copy
import csv
import json
from pathlib import Path
import sys

import pytest
sys.path.insert(0,str(Path(__file__).resolve().parent))
from capacity_certificate import ref,derive_group,raw_measurement,validate
from capacity_executor import durable


def fixture(tmp):
    identity=dict(model_sha256='1'*64,node_sha256='2'*64,source_sha256='3'*64,engine_image='sha256:'+'4'*64,tp=2)
    source=tmp/'source.json';durable(source,{'actual':'source'})
    binding=tmp/'binding.json';durable(binding,dict(identity=identity,files={str(source):ref(source)['sha256']}))
    config=tmp/'config.json';durable(config,dict(slo_ttft_s=1.,slo_tpot_s=.1))
    def result(seed,layout,watts):
        out=tmp/f'r{seed}-{layout}';out.mkdir()
        trace=tmp/f'trace{seed}.json'
        if not trace.exists():durable(trace,dict(schema='capacity-development-trace-v1',split='development',formal_eligible=False,
            seed=seed,demand_domain_sha256='a'*64,duration_s=60.,n_requests=3,
            requests=[dict(prompt_len=2,output_len=2) for _ in range(3)]))
        power=out/'power.csv'
        with power.open('w') as f:
            writer=csv.writer(f);writer.writerow(['t_s']+[f'gpu{g}_w' for g in range(8)])
            writer.writerows([[t]+[watts/8]*8 for t in (0,1,61,62)])
        raw=out/'measurement.json';durable(raw,dict(measurement_valid=True,gpu_indices=list(range(8)),
            power_evidence=dict(power_source_verified=True),measurement_start_s=0.,measurement_end_s=62.,
            energy_j=62*watts,duration_s=62.,artifacts={str(power):ref(power)['sha256']}))
        requests=out/'requests.json';durable(requests,[dict(request_id=f'r{i}',success=1,token_ids_verified=1,
            generated_tokens=2,output_len=2,input_tokens=2,prompt_len=2,ttft_s=.1,tpot_s=.05,slo_ok=1) for i in range(3)])
        path=out/'result.json';durable(path,dict(schema='capacity-load-measurement-v1',complete=True,work_complete=True,
            native_idle=True,resident_groups=[[0,1],[2,3]]+([[4,5]] if layout==3 else []),
            source=dict(capacity_binding=ref(binding),original_binding=ref(source),config=ref(config),host_manifest=ref(source)),
            trace=ref(trace),demand_domain_sha256='a'*64,n_expected=3,n_rows=3,n_good=3,slo_attainment=1.,
            raw_measurement=ref(raw),energy_j=62*watts,offered_rate_rps=.05,actual_arrival_epoch_s=1.,
            artifacts={str(requests):ref(requests)['sha256']}))
        return ref(path)
    return identity,ref(binding),result


def test_three_repetitions_recompute_capacity_and_equal_arrival_window_saving(tmp_path):
    identity,binding,result=fixture(tmp_path)
    two=[result(i,2,100+i) for i in range(3)];three=[result(i,3,160+i) for i in range(3)]
    group=tmp_path/'layout.json';durable(group,dict(schema='capacity-evidence-group-v1',identity=identity,
        capacity_binding=binding,kind='layout',members=two))
    kind,bound,raw=derive_group(ref(group),identity)
    assert kind=='layout' and bound['sustainable_rate_lower_rps']==3/62 and len(raw)==3
    savings=tmp_path/'savings.json';durable(savings,dict(schema='capacity-evidence-group-v1',identity=identity,
        capacity_binding=binding,kind='savings',members=[dict(source=s,target=t) for s,t in zip(three,two)]))
    _,b,_=derive_group(ref(savings),identity)
    assert b['whole_node_saving_lower_w']==60.
    certificate=dict(three_independent_repetitions_per_item=True,evidence_groups=[ref(group)],
        layouts=[bound],transitions=[],savings=[],raw_measurements=raw)
    validate(certificate,identity)
    certificate['layouts'][0]['sustainable_rate_lower_rps']=100
    with pytest.raises(ValueError,match='bounds differ'):
        validate(certificate,identity)


def test_duplicate_repeat_and_mismatched_pair_are_rejected(tmp_path):
    identity,binding,result=fixture(tmp_path)
    two=[result(i,2,100) for i in range(3)];three=[result(i,3,160) for i in range(3)]
    group=tmp_path/'group.json'
    durable(group,dict(schema='capacity-evidence-group-v1',identity=identity,capacity_binding=binding,
        kind='layout',members=[two[0],two[0],two[1]]))
    with pytest.raises(ValueError,match='independent'):
        derive_group(ref(group),identity)
    durable(group,dict(schema='capacity-evidence-group-v1',identity=identity,capacity_binding=binding,
        kind='savings',members=[dict(source=s,target=t) for s,t in zip(three,list(reversed(two)))]))
    with pytest.raises(ValueError,match='matching'):
        derive_group(ref(group),identity)


def test_raw_energy_claim_cannot_override_all_eight_gpu_integral(tmp_path):
    identity,binding,result=fixture(tmp_path)
    path=result(0,2,100)
    loaded=json.loads(Path(path['path']).read_text());raw=loaded['raw_measurement']
    value=json.loads(Path(raw['path']).read_text());value['energy_j']=1;durable(raw['path'],value)
    with pytest.raises(ValueError,match='integral'):
        raw_measurement(ref(raw['path']))


def test_transition_aggregate_requires_three_separate_committed_physical_instances(tmp_path):
    identity,binding,result=fixture(tmp_path)
    members=[]
    for index in range(3):
        load=json.loads(Path(result(index,2,100+index)['path']).read_text())
        raw_path=Path(load['raw_measurement']['path']);raw=json.loads(raw_path.read_text())
        raw['peak_memory_per_gpu_bytes']={str(g):1000+index for g in range(8)};durable(raw_path,raw)
        measurement=dict(raw,receipt=ref(raw_path));transaction=f't{index}';iid=f'actual{index}'
        physical=dict(transaction=transaction,operation='restore',execution_verified=True,
            instance_id=iid,started_s=0.,finished_s=61.,measurement=measurement)
        physical_path=tmp_path/f'physical{index}.json';durable(physical_path,physical)
        inventory=dict(identity=identity,complete=True,transition_inflight=False,
            known_instances={iid:dict(gpus=[4,5],physical_proof={'ordinary':'actual oracle'})},
            events=[dict(kind='physical_commit',transaction=transaction,operation='restore',
                         instance_id=iid,execution_verified=True),
                    dict(kind='transition_measurement',transaction=transaction,receipt=ref(raw_path))])
        inv=tmp_path/f'inventory{index}.json';durable(inv,inventory)
        members.append(dict(result=ref(physical_path),inventory=ref(inv)))
    group=tmp_path/'transition-group.json';value=dict(schema='capacity-evidence-group-v1',identity=identity,
        capacity_binding=binding,kind='transition',operation='restore_cold',gpus=[4,5],members=members)
    durable(group,value);kind,bound,refs=derive_group(ref(group),identity)
    assert kind=='transition' and bound['duration_upper_s']==62.
    assert bound['energy_upper_j']==62*102 and bound['peak_memory_per_gpu_upper_bytes']==1002
    value['members']=[members[0]]*3;durable(group,value)
    with pytest.raises(ValueError,match='separate physical'):
        derive_group(ref(group),identity)
