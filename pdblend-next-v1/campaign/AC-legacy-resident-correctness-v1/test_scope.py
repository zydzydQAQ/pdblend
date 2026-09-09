import ast,hashlib
from pathlib import Path
import pytest
from validate import validate_scope,mechanism_gates
P=Path(__file__).resolve().parent;B=P.parent/'B32B-legacy-baseline-correctness-v1'
def binding(model):
    return dict(model=model,instances=[dict(id=str(g),tp=1,gpus=[g],native_kind='legacy_sync_put',container=dict(image='sha256:d11407cd827a43a0dec8ad7d4d7037c97c39bbe93c6f4b4fd951c94e67509a8b')) for g in range(8)])
@pytest.mark.parametrize('model',['7b','14b'])
def test_real_resident_scope(model):validate_scope(binding(model))
def test_heterogeneous_not_mislabelled_same_tp():
    b=binding('14b');b['instances']=b['instances'][:6];b['instances'][-1].update(tp=2,gpus=[6,7])
    with pytest.raises(RuntimeError):validate_scope(b)
def test_checks_are_original_frozen_bytes():
    assert (P/'checks.py').read_bytes()==(B/'checks.py').read_bytes()
    assert hashlib.sha256((P/'checks.py').read_bytes()).hexdigest()=='e0965f922ae42245b275d9c17342c31689290c61bcac5dfaf33a2a352fd29931'
def test_execution_only_model_and_tp_scope_differ():
    def body(path):return next(ast.get_source_segment(path.read_text(),n) for n in ast.parse(path.read_text()).body if isinstance(n,ast.AsyncFunctionDef) and n.name=='execute')
    expected=body(B/'validate.py').replace("and cfg['tp']==2,'static startup budget/TP/model work limit differs'","and cfg['tp']==1,'static startup budget/TP/model work limit differs'").replace("expected_model='/models/Qwen2.5-32B-Instruct'","expected_model='/models/Qwen2.5-'+binding['model'].upper()+'-Instruct'")
    assert ast.dump(ast.parse(expected),include_attributes=False)==ast.dump(ast.parse(body(P/'validate.py')),include_attributes=False)
def test_failed_energy_invalidates_all_mechanisms():
    assert mechanism_gates(dict(ordinary_cross_replica_exact=True,pd_exact_all_declared_pairs=True,cancel_all_tp_ranks=True,temporal_exact=True),False,True)==dict(ordinary=False,pd=False,temporal=False)
