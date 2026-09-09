"""Actual model proofs must gate both scale planning and metadata preparation."""
import copy
import json
from pathlib import Path
import time
import pytest
import contract as c
import reference_map as m

def release(tmp_path,model):
    path=(c.CAMPAIGN/'A14B-main-completion-proof-v1/host-main-proof.json' if model=='14b'
          else c.CAMPAIGN/'C7B-main-proof-prepared-v1/host-main-proof.json')
    proof=c.read(path)
    value=dict(schema=1,kind=c.released.KIND,model=model,protocol_id=c.PROTOCOL,deadline_s=c.DEADLINE,
        models={model:proof},proof_refs={model:dict(path=str(path),sha256=c.released.AUTHORIZED[model][0],
            canonical_sha256=c.released.AUTHORIZED[model][1])},baseline_systems=list(c.barrier.BASELINES),
        main_records=150,baseline_main_records=120,pdblend_main_records=30,global_release=False,
        coordinator_deep_verification=True,created_s=time.time())
    out=tmp_path/'test-model-release.json';out.write_text(json.dumps(value))
    return out,c.sha(out),proof

@pytest.mark.parametrize('model',['14b','7b'])
def test_contract_consumes_only_matching_actual_model_proof(tmp_path,monkeypatch,model):
    path,h,proof=release(tmp_path,model);seen=[]
    group=dict(id='pdb',system='pdblend',datasets=['alpaca'])
    monkeypatch.setattr(c,'check_group',lambda g,p,ma:(seen.append(p) or dict(rows=[],reused=[],pending=[])))
    spec=dict(schema=2,model=model,hostname=proof['hostname'],protocol_id=c.PROTOCOL,deadline_s=c.DEADLINE,
        source=dict(path=proof['source_manifest'],sha256=proof['source_sha256']),groups=[group])
    result=c.check_spec(spec,path,h)
    assert seen==[proof] and result['release']['main_records']==150
    seen.clear();spec['model']='7b' if model=='14b' else '14b'
    with pytest.raises(ValueError,match='this model'):c.check_spec(spec,path,h)
    assert not seen

def test_binding_metadata_cannot_borrow_other_model_release(tmp_path,monkeypatch):
    path,h,proof=release(tmp_path,'14b')
    monkeypatch.setattr(m,'package_check',lambda:None)
    monkeypatch.setattr(c,'released_main_records',lambda *a:pytest.fail('main read before model authorization'))
    with pytest.raises(ValueError,match='this model'):
        m.binding_metadata(dict(model='7b'),{},path,h,[])

def test_original_driver_child_measurement_bytes_unchanged():
    parent=c.CAMPAIGN/'scale-only-continuation-v2'
    for name in ['scale_driver.py','child.py','power_evidence.frozen.py']:
        assert (parent/name).read_bytes()==(c.HERE/name).read_bytes()
