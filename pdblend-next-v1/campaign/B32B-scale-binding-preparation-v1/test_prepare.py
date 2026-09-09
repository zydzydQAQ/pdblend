import copy
from pathlib import Path
import pytest
import prepare as p

@pytest.mark.parametrize('system',['mixed','dynamollm','distserve','ecoserve'])
def test_real_policy_gate_and_process_compatibility_in_temporary_output(system,tmp_path):
    c,_=p.modules();_,policies,fresh=p.inputs();original=c.reference(policies[system]);value,mode=p.candidate(system,original,fresh,c)
    identity=tmp_path/'identity.json';p.write(identity,p.read(fresh['identity_file']))
    value['identity_file']=str(identity);value['files'][str(identity)]=p.sha(identity)
    binding=tmp_path/'binding.json';p.write(binding,value)
    checked=c.compatible_bindings(policies[system],p.ref(binding),p.DATASETS,mode)
    assert checked['identity_mode']==('same_process' if system=='ecoserve' else 'restarted')
    assert value['configs']==original['configs'] and value['output']==original['output']
    assert value['large_inputs']==original['large_inputs']
    if system!='ecoserve':assert value['mechanism_proof']['verified']=={'ordinary':True,'pd':True,'temporal':False}


def test_new_pid_without_new_startedat_cannot_be_called_restarted(tmp_path):
    c,_=p.modules();_,policies,fresh=p.inputs();main=c.reference(policies['mixed']);value,mode=p.candidate('mixed',main,fresh,c)
    inv=p.read(fresh['identity_file']);old={i['container']['id']:i for i in main['instances']}
    for r in inv:r['State']['StartedAt']=old[r['Id']]['container']['StartedAt']
    for i in value['instances']:i['container']['StartedAt']=old[i['container']['id']]['container']['StartedAt']
    ip=tmp_path/'identity.json';p.write(ip,inv);value['files'][str(ip)]=p.sha(ip);value['identity_file']=str(ip)
    bp=tmp_path/'binding.json';p.write(bp,value)
    with pytest.raises(Exception,match='StartedAt AND host'):c.compatible_bindings(policies['mixed'],p.ref(bp),p.DATASETS,'restarted')


def test_null_release_never_writes_binding_or_future_spec(tmp_path):
    out=tmp_path/'out'
    with pytest.raises(Exception):p.prepare(tmp_path/'missing-release.json',None,out)
    assert not out.exists()


def test_physical_capacity_or_policy_cannot_change():
    c,_=p.modules();_,policies,fresh=p.inputs();main=c.reference(policies['mixed']);changed=copy.deepcopy(fresh);changed['instances'][0]['gpus']=[0]
    with pytest.raises(RuntimeError,match='physical'):p.candidate('mixed',main,changed,c)
    a=p.read(main['configs']['alpaca']);b=copy.deepcopy(a);b['output_prior']=a.get('output_prior',0)+1
    with pytest.raises(Exception,match='scale policy'):c.policy_equal(a,b)
