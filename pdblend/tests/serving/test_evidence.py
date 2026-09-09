import json

from ecopadg.serving.evidence import (REQUIRED_MECHANISMS, evaluate,
    evaluation_matrix, freeze_bundle, formal_freeze_gaps, freeze_files, sha256,common_capacity)
import pytest


def evidence_fixture(tmp_path):
    groups={}
    for kind in ('source','model','profiles','protocol'):
        p=tmp_path/kind
        p.write_text(kind)
        groups[kind]=[p]
    image='sha256:'+'a'*64
    p=tmp_path/'image.json'
    p.write_text(json.dumps([{'Id':image}]))
    groups['image']=[p]
    cells=evaluation_matrix(dict(alpaca=1,sharegpt=1,longbench=1))
    cells.extend(dict(dataset='dynamic',load='changing',seed=s,n_requests=500,
                      split='formal',trace_duration_s=3600) for s in (101,202,303))
    groups['traces']=[];trace_hashes={}
    for c in cells:
        key=(c['dataset'],c['load'],c['seed'])
        path=tmp_path/('-'.join(map(str,key))+'.trace.json')
        path.write_text(json.dumps(dict(c,requests=[{'output_len':64}]*500,duration_s=c.get('trace_duration_s',100))))
        groups['traces'].append(path);trace_hashes[key]=sha256(path)
    proof=dict(passed=True,artifact=str(p),sha256=sha256(p))
    mechanisms={b:{m:proof for m in ms} for b,ms in REQUIRED_MECHANISMS.items()}
    # Synthetic collector artifacts exercise the consumer contract, not hardware.
    from ecopadg.serving.evidence import COLLECTION_COMPONENTS
    registry=tmp_path/'mechanisms.json';registry.write_text(json.dumps(mechanisms))
    collection=tmp_path/'collection.json'
    collection.write_text(json.dumps(dict(complete=True,baseline_mechanisms_complete=True,missing={},
        registry=str(registry),registry_sha256=sha256(registry),
        components={k:dict(passed=True) for k in COLLECTION_COMPONENTS},
        source_files=freeze_files(groups['source']))))
    expected=tmp_path/'expected.json';expected.write_text(json.dumps(cells))
    groups['protocol'].extend([registry,collection,expected])
    bundle=freeze_bundle(groups,image)
    bundle['formal_evidence']=dict(mechanisms=str(registry),mechanism_collection=str(collection),expected_cells=str(expected))
    rows=[]
    for c in cells:
        for system in ('pdblend',*REQUIRED_MECHANISMS):
            rows.append(dict(c,system=system,variant='pdblend-joint' if system=='pdblend' else system,**bundle['identities'],
                trace_sha256=trace_hashes[(c['dataset'],c['load'],c['seed'])],measurement_schema=2,
                n_expected=500,completed=500,generated_tokens=32000,expected_generated_tokens=32000,energy_j=90 if system=='pdblend' else 100,
                power_mode='instant',power_source_id='nvml:field:186:scope:0:mW',
                power_field_id=186,power_source_verified=True,
                slo_attainment=1,gpu_count=8,split='formal',formal_eligible=True,validity='ok',duration_s=3610))
    return rows,cells,mechanisms,bundle


def test_complete_paired_matrix_can_pass_and_emits_strict_json(tmp_path):
    result=evaluate(*evidence_fixture(tmp_path))
    assert result['verdict']=='target_achieved'
    json.dumps(result,allow_nan=False)


@pytest.mark.parametrize('change',[{'power_mode':'average'}, {'power_source_verified':False},
                                 {'power_field_id':185}, {'power_source_id':None}])
def test_matching_historical_or_unverified_power_cannot_be_a_formal_victory(tmp_path,change):
    rows,cells,mechanisms,bundle=evidence_fixture(tmp_path)
    for row in rows: row.update(change)
    assert evaluate(rows,cells,mechanisms,bundle)['verdict']=='evidence_insufficient'


def test_omitting_dynamic_trace_never_becomes_a_victory(tmp_path):
    rows,cells,mechanisms,bundle=evidence_fixture(tmp_path)
    result=evaluate(rows,[c for c in cells if c['dataset']!='dynamic'],mechanisms,bundle)
    assert result['verdict']=='evidence_insufficient'
    assert any('dynamic' in s for s in result['reasons'])


def test_incomplete_seeds_and_nonfinite_energy_do_not_produce_nan_json(tmp_path):
    rows,cells,mechanisms,bundle=evidence_fixture(tmp_path)
    rows=[r for r in rows if r['seed']!=303]
    rows[0]['energy_j']=float('nan')
    result=evaluate(rows,cells,mechanisms,bundle)
    assert result['verdict']=='evidence_insufficient'
    json.dumps(result,allow_nan=False)


def test_single_arbitrary_file_is_not_a_formal_freeze(tmp_path):
    p=tmp_path/'unrelated';p.write_text('x')
    assert formal_freeze_gaps(freeze_files([p]))


def test_matching_rows_cannot_claim_a_different_frozen_image(tmp_path):
    rows,cells,mechanisms,bundle=evidence_fixture(tmp_path)
    for r in rows:
        r['engine_image']='mutable-image-tag:latest'
    result=evaluate(rows,cells,mechanisms,bundle)
    assert result['verdict']=='evidence_insufficient'


def test_same_frozen_trace_cannot_be_relabelled_as_different_loads_or_seeds(tmp_path):
    rows,cells,mechanisms,bundle=evidence_fixture(tmp_path)
    for row in rows: row['trace_sha256']=rows[0]['trace_sha256']
    assert evaluate(rows,cells,mechanisms,bundle)['verdict']=='evidence_insufficient'


def test_common_capacity_requires_confirmed_work_and_an_infeasible_upper_bracket():
    probes=[dict(dataset=d,system='mixed',rate=4,slo_attainment=1,validity='ok',split='calibration')
            for d in ('alpaca','sharegpt','longbench')]
    with pytest.raises(ValueError): common_capacity(probes,required=('mixed',))
    records=[dict(dataset=d,system='mixed',passed=True,capacity_rps=2,infeasible_upper_rps=3,
        confirmation=dict(rate=2,slo_attainment=1,validity='ok',split='calibration',completed=256,n_expected=256))
        for d in ('alpaca','sharegpt','longbench')]
    assert common_capacity(records,required=('mixed',))==dict(alpaca=2,sharegpt=2,longbench=2)
    records[0]['confirmation']['completed']=255
    with pytest.raises(ValueError): common_capacity(records,required=('mixed',))
