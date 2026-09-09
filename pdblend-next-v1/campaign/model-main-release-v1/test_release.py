"""Use real completed A/C proofs; no fabricated measurement or GPU call."""
import copy
import json
from pathlib import Path
import time
import pytest
import release as r

PROOFS = {'14b':r.HERE.parent/'A14B-main-completion-proof-v1/host-main-proof.json',
          '7b':r.HERE.parent/'C7B-main-proof-prepared-v1/host-main-proof.json'}

def value(model='7b'):
    proof=r.read(PROOFS[model])
    return dict(schema=1,kind=r.KIND,model=model,protocol_id=r.PROTOCOL,deadline_s=r.DEADLINE,
        models={model:proof},proof_refs={model:dict(path=str(PROOFS[model]),sha256=r.AUTHORIZED[model][0],
            canonical_sha256=r.AUTHORIZED[model][1])},baseline_systems=list(r.v1.BASELINES),
        main_records=150,baseline_main_records=120,pdblend_main_records=30,global_release=False,
        coordinator_deep_verification=True,created_s=time.time())

@pytest.mark.parametrize('model',['7b','14b'])
def test_real150_contract_and_deep_original_evidence(model):
    v=value(model)
    assert r.sha(PROOFS[model])==r.AUTHORIZED[model][0]
    assert r.release_contract(v,expected_model=model)==model
    relocation=(r.read(r.HERE.parent/'C7B-main-proof-prepared-v1/root-deep-verification.json')['local_path_maps']
                if model=='7b' else None)
    assert r.per_cell.verify_model_proof(v['models'][model],deep=True,path_map=relocation)==model

@pytest.mark.parametrize('change',['global_kind','global450','another_model','32b','deadline','proof_sha','changed_raw_record','missing_record','future_publication','claim_global'])
def test_scope_and_proof_rejections(change):
    v=value()
    if change=='global_kind':v['kind']='global-main-release-per-cell'
    elif change=='global450':v['main_records']=450
    elif change=='another_model':v['models']['14b']=v['models']['7b']
    elif change=='32b':v['model']='32b'
    elif change=='deadline':v['deadline_s']+=1
    elif change=='proof_sha':v['proof_refs']['7b']['sha256']='a'*64
    elif change=='changed_raw_record':v['models']['7b']['records'][0]['energy_j']+=1
    elif change=='missing_record':v['models']['7b']['records'].pop()
    elif change=='future_publication':v['created_s']=r.DEADLINE
    elif change=='claim_global':v['global_release']=True
    with pytest.raises(ValueError):r.release_contract(v,expected_model='7b')

def test_file_sha_and_cross_model_rejected(tmp_path):
    path=tmp_path/'cpu-test-release.json';path.write_text(json.dumps(value()))
    assert r.verify_release(path,r.sha(path),expected_model='7b')['main_records']==150
    with pytest.raises(ValueError):r.verify_release(path,r.sha(path),expected_model='14b')
    with pytest.raises(ValueError):r.verify_release(path,'a'*64,expected_model='7b')

def test_expired_release_rejected(monkeypatch):
    v=value();monkeypatch.setattr(r.time,'time',lambda:r.DEADLINE)
    with pytest.raises(ValueError):r.release_contract(v)

def test_bad_assembly_never_calls_deep_or_publishes(tmp_path,monkeypatch):
    monkeypatch.setattr(r,'package_check',lambda:None)
    monkeypatch.setattr(r.per_cell,'verify_model_proof',lambda *a,**k:pytest.fail('deep review of unapproved proof'))
    out=tmp_path/'not-created.json'
    with pytest.raises(ValueError):r.assemble_release(PROOFS['7b'],'a'*64,out)
    assert not out.exists()
