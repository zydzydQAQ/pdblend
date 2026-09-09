import ast,copy,json,sys
from pathlib import Path
from types import SimpleNamespace
import pytest
import deploy
from test_deployment import group

def test_main_only_does_not_read_unmeasured_scale(tmp_path):
    b,m,out=group(tmp_path)
    for p in (out/'checkpoints').glob('*-scale-*'):p.unlink()
    (out/'invocations/scale.json').unlink()
    result=deploy.terminal_group(b,m,datasets=['alpaca','sharegpt'],phases=['main'])
    assert result['counts']=={'main':20} and len(result['checkpoint_sha256'])==20
    assert result['unrequested_phases']==['scale'] and not result['unrequested_phase_completion_claimed']
    assert not result['reference_checkpoint_sha256']
    with pytest.raises(FileNotFoundError):deploy.terminal_group(b,m,datasets=['alpaca','sharegpt'])

@pytest.mark.parametrize('failure',['main_missing','raw_changed','main_not_terminal','foreign_phase','empty_phase','repeated_phase','duplicate_cell'])
def test_main_only_still_strict(tmp_path,failure):
    b,m,out=group(tmp_path);phases=['main']
    if failure=='main_missing':(out/'checkpoints/alpaca-main-2.json').unlink()
    elif failure=='raw_changed':(out/'raw/alpaca-main-2').write_text('changed')
    elif failure=='main_not_terminal':
        x=deploy.read(out/'invocations/main.json');x['complete']=False;(out/'invocations/main.json').write_text(json.dumps(x))
    elif failure=='foreign_phase':phases=['probe']
    elif failure=='empty_phase':phases=[]
    elif failure=='repeated_phase':phases=['main','main']
    else:
        x=deploy.read(m);x['cells'].append(copy.deepcopy(x['cells'][0]));m.write_text(json.dumps(x))
        y=deploy.read(b);y['files'][str(m)]=deploy.sha(m);b.write_text(json.dumps(y))
    with pytest.raises((RuntimeError,FileNotFoundError)):
        deploy.terminal_group(b,m,datasets=['alpaca','sharegpt'],phases=phases)

def test_scale_only_still_verifies_original_main_reference(tmp_path):
    b,m,out=group(tmp_path)
    r=deploy.terminal_group(b,m,datasets=['alpaca'],phases=['scale'])
    assert r['counts']=={'scale':6} and len(r['reference_checkpoint_sha256'])==6
    (out/'raw/alpaca-main-1').write_text('tampered main raw')
    with pytest.raises(RuntimeError):deploy.terminal_group(b,m,datasets=['alpaca'],phases=['scale'])

def test_previous_phase_bound_into_explicit_actual_binding(tmp_path):
    b,m,out=group(tmp_path);prior=deploy.read(b);prior['model']='14b';b.write_text(json.dumps(prior))
    previous=tmp_path/'distserve.json';v=dict(prior,system='distserve',instances=[dict(container={'name':'actual-retained'})]);deploy.write(previous,v)
    args=SimpleNamespace(model='14b',pdb_binding=b,workloads=m,previous_binding=previous,
      previous_dataset=['alpaca','sharegpt'],previous_phase=['main'],pdb_phase=['main'])
    req,current,pdbphases,phases=deploy.predecessor_requirements(args,prior)
    assert req[1]['binding']==str(previous) and req[1]['system']=='distserve' and req[1]['phases']==['main']
    assert current==previous and phases==pdbphases==['main']
    args.previous_binding=None
    with pytest.raises(RuntimeError):deploy.predecessor_requirements(args,prior)

def test_old_binder_v2_accepts_new_spec_fields_and_same_geometry():
    root=deploy.WORKSPACE/'campaign/AC-baseline-binding-v2';sys.path.insert(0,str(root))
    try:bind=deploy.load('phase_compat_old_binder',root/'bind.py')
    finally:sys.path.remove(str(root))
    spec=dict(model='14b',layout='distserve-longbench',previous_phases=['main'],pdb_phases=['main'],
      instances=[dict(tp=tp,gpus=gpus,native_kind='legacy_sync_put',scheduler_cache_observed=False,image=deploy.IMAGE)
        for tp,gpus in [(1,[i]) for i in range(5)]+[(2,[6,7])]])
    bind.scope(spec);assert bind.selected(spec,None,None)==[]
    assert bind.selected(spec,'distserve',['longbench'])==['longbench']
    with pytest.raises(RuntimeError):bind.selected(spec,'mixed',['longbench'])
    # Old binder DEPLOY constant is unused; real entry/provenance is taken from spec.
    source=ast.parse((root/'bind.py').read_text())
    assert not any(isinstance(n,ast.Name) and n.id=='DEPLOY' and isinstance(n.ctx,ast.Load) for n in ast.walk(source))

def test_heterogeneous_switch_cannot_substitute_mixed_for_distserve20(tmp_path):
    b,m,out=group(tmp_path);prior=dict(deploy.read(b),model='14b');b.write_text(json.dumps(prior))
    previous=tmp_path/'actual-mixed.json';deploy.write(previous,dict(prior,system='mixed',instances=[{'id':'actual'}]))
    args=SimpleNamespace(model='14b',layout='distserve-longbench',pdb_binding=b,workloads=m,previous_binding=previous,
      previous_dataset=['alpaca','sharegpt'],previous_phase=['main'],pdb_phase=['main'])
    with pytest.raises(RuntimeError,match='DistServe'):deploy.predecessor_requirements(args,prior)
