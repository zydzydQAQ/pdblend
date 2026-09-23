import copy
import importlib.util
import json
from pathlib import Path

import pytest

path=Path(__file__).resolve().parents[2]/'scripts/2026-09-23_prepare_quad_profile_wave.py'
spec=importlib.util.spec_from_file_location('quad_profile',path)
quad=importlib.util.module_from_spec(spec);spec.loader.exec_module(quad)


def fixture(tmp_path,model='Qwen2.5-32B-Instruct',tp=2):
    raw=dict(system='pdblend',model_id=model,tp=tp,pp=1,
             kv_capacity_tokens=44416 if tp==2 else 411824,
             prefill=[dict(freq_mhz=f,input_tokens=c,seconds=s,concurrency=1)
                      for f in quad.costing.FREQUENCIES for c,s in ((4096,.5),(7168,.8))])
    raw_path=tmp_path/'raw.json';raw_path.write_text(json.dumps(raw))
    shapes=[(5120,1),(5120,4),(5120,6),(7168,1),(7168,4)] if tp==2 else [
        (5120,1),(5120,4),(5120,8),(5120,60),(7168,1),(7168,4),(7168,8),(7168,45)]
    plan=dict(system='pdblend',model_id=model,tp=tp,pp=1,kv_capacity_tokens=raw['kv_capacity_tokens'],
        training_source=str(raw_path),training_source_sha256=quad.paired.sha256(raw_path),fit_existing_holdout=False,
        training=[dict(freq_mhz=f,context_tokens=c,batch=b,repeats=3,settle_s=2,measure_s=5,max_tokens=1024,
                       purpose='training_extension') for f in quad.costing.FREQUENCIES for c,b in shapes],holdout=[])
    p=tmp_path/'plan.json';p.write_text(json.dumps(plan))
    return p


def test_32b_tp2_explicit_memory_exclusion_and_deferred_b6(tmp_path):
    path=fixture(tmp_path);before=path.read_bytes()
    plan,cost=quad.subset_plan(path,model_id='Qwen2.5-32B-Instruct',tp=2,batches=(1,4))
    assert path.read_bytes()==before
    assert len(plan['training'])==24 and len(plan['deferred_training'])==6
    assert {x['batch'] for x in plan['deferred_training']}=={6}
    assert len(plan['unsupported_requested_points'])==12
    assert all(p['status']=='unsupported_memory' and p['allowed_in_planner'] is False for p in plan['unsupported_requested_points'])
    assert all(p['requested_reservation_tokens']>p['usable_capacity_tokens'] for p in plan['unsupported_requested_points'])
    assert plan['full_training_matrix_complete'] is False
    assert plan['coverage_constraints']['long_context_batch_max']==4
    assert cost['largest_selected_reservation_tokens']==32768
    with pytest.raises(ValueError,match='not in the original'):
        quad.subset_plan(path,model_id='Qwen2.5-32B-Instruct',tp=2,batches=(1,4,8))


def test_7b_tp1_has_36_points_without_promoting_deferred_domain(tmp_path):
    path=fixture(tmp_path,'Qwen2.5-7B-Instruct',1)
    plan,cost=quad.subset_plan(path,model_id='Qwen2.5-7B-Instruct',tp=1,batches=(1,4,8))
    assert len(plan['training'])==36 and len(plan['deferred_training'])==12
    assert plan['unsupported_requested_points']==[]
    assert plan['coverage_constraints']['long_context_batch_max']==8
    assert not cost['actual_batch_runtime_measured'] and not cost['guaranteed_runtime_bound']


def test_role_argv_preserves_runtime_and_replaces_old_profile_mounts():
    template=dict(image_digest='image',argv=['docker','run','--name','old','--gpus','device={lease_gpus}',
        '-v','models:/models:ro','-v','source:/opt/pdblend-src:ro','-v','cohort:/wave:rw',
        '-v','oldplan:/plan/long-context.json:ro','-v','oldraw:/training/raw.json:ro',
        '-e','PDBLEND_SOURCE_SHA256=frozen','-e','PDBLEND_PROFILE_MEMBER=old','image','oldcommand'])
    before=copy.deepcopy(template)
    argv=quad.rewritten_argv(template,name='repair',member='14b-repair',
        mounts=[('/candidate','/candidate'),('/original','/original')],command=['repair_module','--gpus','0'])
    assert template==before
    assert argv[-4:]==['-m','repair_module','--gpus','0']
    assert 'source:/opt/pdblend-src:ro' in argv and 'cohort:/wave:rw' in argv
    assert 'PDBLEND_SOURCE_SHA256=frozen' in argv and 'PDBLEND_PROFILE_MEMBER=14b-repair' in argv
    assert '/candidate:/candidate:ro' in argv and '/original:/original:ro' in argv
    assert not any('oldplan:' in x or 'oldraw:' in x for x in argv)
