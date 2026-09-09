"""CPU-only actual raw readers and fail-closed composition; no new inference."""
import copy
import json
from pathlib import Path
import pytest
import oracle
import qualification as q
import shape_gate as g
import power_source
from test_shape_gate import actual,BASE

def test_actual_sameflag_native_registration_is_stable():
    p=oracle.ATTEMPT/'spec.json'
    inputs=oracle.contract('b4e299bc703e2146582d3180d01d41eddb60dac2b91394033f5d0a1b66e7d4ad',p,oracle.sha(p))
    result=oracle.verify(inputs)
    assert result['full256_exact_original003'] and result['actual_native_config_true']
    assert result['legacy_single_vs_pair_exact'] is False and result['legacy_failure_preserved']
    assert len(result['reference_shape'])==197 and len(result['files'])>100
    assert all(not p.startswith('/tmp/') for p in result['files'])
    assert result==oracle.verify(inputs)

@pytest.mark.parametrize('change',['old_attempt','unknown_package','placeholder'])
def test_wrong_reference_cannot_register(change):
    p=oracle.ATTEMPT/'spec.json';h=oracle.sha(p);package='b4e299bc703e2146582d3180d01d41eddb60dac2b91394033f5d0a1b66e7d4ad'
    if change=='old_attempt':p=oracle.ROOT/'campaign/B32B-native-default-reference-attempt-001/spec.json'
    if change=='unknown_package':package='0'*64
    if change=='placeholder':h='x'*64
    with pytest.raises(RuntimeError):oracle.contract(package,p,h)

def test_no_fixture_eligibility_without_registered_oracle(monkeypatch):
    monkeypatch.setattr(q,'check_package',lambda:{})
    with pytest.raises(RuntimeError,match='unregistered'):q.audit_fresh_gate(BASE/'legacy-correctness',{}, {'passed':True})

@pytest.mark.parametrize('stale',[False,True])
def test_complete_original27_reader_and_host_process_gate(tmp_path,stale):
    reader=g.load_reader();binding=reader.read(BASE/'correctness-binding/binding.json');gate=BASE/'legacy-correctness'
    # Real old inspect used only for the parser fixture, not a fresh qualification.
    inventory=[x['container'] for x in reader.read(gate/'identity.after.json')]
    if stale:inventory[1]['State']['Pid']+=1
    identity=tmp_path/'identity.json';identity.write_text(json.dumps(inventory))
    binding['identity_file']=str(identity);binding['files'][str(identity)]=reader.sha(identity)
    *_,tokens=actual()
    if stale:
        with pytest.raises(RuntimeError,match='another host process'):
            g.inspect_fresh_gate(gate,binding,tokens,power_source.load())
    else:
        value=g.inspect_fresh_gate(gate,binding,tokens,power_source.load())
        assert value['original_gate']['verified']==dict(ordinary=True,pd=True,temporal=False)
        assert value['temporal']['pair_steps']==69 and value['eligible_systems']=={'ecoserve':False}

def test_owner_output_token_mismatch_rejected():
    events=[json.loads(x) for x in (oracle.ATTEMPT/'results/diagnostic-owner.events.jsonl').read_text().splitlines()]
    native=oracle.read(oracle.ATTEMPT/'results/child/full-outputs.json')['token_ids_by_request_uuid']
    oracle.verify_emitted(events,native)
    next(e for e in events if e['kind']=='output')['token_ids'][0]+=1
    with pytest.raises(RuntimeError):oracle.verify_emitted(events,native)
