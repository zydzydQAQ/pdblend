"""Bounded CPU integration with existing B producers, no proof publication."""
import copy
from pathlib import Path
import pytest
import release as r

def draft():return r.read(r.HERE/'draft-spec.json')

def test_actual_original120_partition_plus_future_eco30_is_not_ready():
    value=draft();rows=r.v1.source_rows(r.read(r.SOURCE),'32b')
    assert len(rows)==150 and len(value['groups'])==5
    assert sum(len(g['cell_ids']) for g in value['groups'] if g['binding'])==120
    assert next(g for g in value['groups'] if g['system']=='ecoserve')['binding'] is None
    with pytest.raises(ValueError):r.p.partition_contract(value['groups'],rows)

@pytest.mark.parametrize('system,last',[('pdblend',False),('mixed',False),('mixed',True),('distserve',False),('dynamollm',False)])
def test_real_original_producer_binding_receipt_and_source(system,last):
    spec=draft();g=next(g for g in spec['groups'] if g['system']==system);reader=r.v1.Reader()
    compatibility,binding=r.p.compatibility(g,reader)
    source=r.p.execution_source(g,binding,reader,spec['source_manifest'],spec['source_sha256'])
    group=dict(g,compatibility=compatibility,execution_source=source)
    rows=[x for x in r.v1.source_rows(r.read(r.SOURCE),'32b') if x['system']==system]
    row=rows[-1 if last else 0];value=r.p.record(row,group,binding,reader,[])
    inv=reader.read(value['execution']['invocation'])
    assert row['cell_id'] in inv['completed'] and value['execution']['invocation_succeeded']
    assert value['binding_sha256']==g['binding']['sha256'] and value['measurement_valid']
    if system=='mixed':assert len(inv['completed'])==(6 if last else 24)
    reader.stable()

@pytest.mark.parametrize('change',['missing_eco','another_model','old_protocol','fake_passed_only'])
def test_no_unqualified_eco_binding(change):
    b={'model':'32b','system':'ecoserve','protocol_id':r.PROTOCOL,'deadline_s':r.DEADLINE,
        'correctness_protocol_id':r.CORRECTNESS,'passed':True}
    if change=='missing_eco':b['system']='mixed'
    if change=='another_model':b['model']='14b'
    if change=='old_protocol':b['correctness_protocol_id']='legacy-single-vs-pair'
    with pytest.raises((ValueError,KeyError,RuntimeError)):r.audit_eco_binding(b)

def test_old_partial_B_supervisor_report_cannot_release():
    actual=r.read(r.C/'B32B-baseline-main-first-sequence-v1/attempt-001/main-proof.json')
    assert actual['baseline_main_completed']==90 and actual['missing_systems']['ecoserve']['main_missing']==30
    with pytest.raises((ValueError,KeyError)):r.proof_contract(actual)

@pytest.mark.parametrize('model',['7b','14b'])
def test_other_model_actual150_not_B_release(model):
    path=r.C/('C7B-main-proof-prepared-v1' if model=='7b' else 'A14B-main-proof-prepared-v1')/'host-main-proof.json'
    if not path.exists():
        path=r.C/'model-main-release-v1'/('actual-C7B-release.json' if model=='7b' else 'actual-A14B-release.json')
        proof=r.read(path)['models'][model]
    else:proof=r.read(path)
    with pytest.raises(ValueError):r.proof_contract(proof)

def test_no_release_of_null_future_decl(tmp_path,monkeypatch):
    monkeypatch.setattr(r,'package_check',lambda:None)
    with pytest.raises((ValueError,KeyError)):
        r.assemble_release(r.HERE/'draft-spec.json',r.sha(r.HERE/'draft-spec.json'),tmp_path/'release.json')
    assert not (tmp_path/'release.json').exists()

def test_explicit_sha_and_model_are_mandatory(tmp_path):
    p=tmp_path/'release.json';p.write_text('{}')
    with pytest.raises(ValueError):r.verify_release(p,'0'*64,expected_model='32b')
    with pytest.raises(ValueError):r.release_contract(dict(model='32b'),expected_model='7b')

def test_actual_eco_new_qualification_preserves_missing_old_header():
    path=r.C/'B32B-ecoserve-qualified-main-v1/binding.json'
    assert r.sha(path)=='87dbf43fcf8076dc1b4bc588e14fad717a2d2b3eeb83cae1514f3fe2d078d119'
    value=r.audit_eco_binding(r.read(path))
    assert value['eligible_systems']=={'ecoserve':True}
    assert value['raw_temporal_exact_header_present'] is False
    assert value['original_mechanism_gate']==dict(ordinary=True,pd=True,temporal=False)

def test_qualification_does_not_allow_changed_eco_policy(tmp_path):
    b=r.read(r.C/'B32B-ecoserve-qualified-main-v1/binding.json')
    cfg=r.read(b['configs']['sharegpt']);cfg['service_dvfs']=not cfg.get('service_dvfs',True)
    path=tmp_path/'changed.json';path.write_text(__import__('json').dumps(cfg))
    b['configs']['sharegpt']=str(path);b['files'][str(path)]=r.sha(path)
    with pytest.raises(ValueError,match='algorithm/profile'):r.audit_eco_binding(b)
