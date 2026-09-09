import copy
import importlib.util
from pathlib import Path
import pytest

path=Path(__file__).with_name('prepare.py')
spec=importlib.util.spec_from_file_location('b_eco_plan_tests',path)
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)

def test_actual_original_rows_and_bytes():
    rows, configs, sources=p.inputs()
    assert len(rows)==48 and sum(r['phase']=='main' for r in rows)==30
    assert all(p.sha(path)==h for path,h in sources.items())
    for ds,cfg in configs.items():
        original=p.read(p.C/f'B32B-five-system100-v1/configs/{ds}.ecoserve.json')
        normalized=copy.deepcopy(cfg)
        normalized.pop('controller_source_release');normalized.pop('comparison_system')
        assert normalized==original
        assert cfg['output_prior']==original['output_prior']

@pytest.mark.parametrize('field,value',[('trace_sha256','0'*64),('n_requests',1),('content_pairing_sha256','x'),('slo_ttft_s',999)])
def test_actual_counterpart_or_slo_mismatch(field,value):
    source=p.read(p.SOURCE)
    row=next(r for r in source['cells'] if r['system']=='ecoserve')
    row[field]=value
    with pytest.raises(RuntimeError):p.check_source(source)

def test_scale_reference_cannot_point_to_another_system():
    source=p.read(p.SOURCE)
    row=next(r for r in source['cells'] if r['system']=='ecoserve' and r['phase']=='scale')
    row['reuse_main_cell_id']=next(r['cell_id'] for r in source['cells'] if r['system']=='mixed' and r['phase']=='main')
    with pytest.raises(RuntimeError):p.check_source(source)

def test_plan_never_emits_performance_binding(tmp_path):
    out=tmp_path/'plan';result=p.prepare(out);plan=p.read(out/'plan.json')
    assert result['ready'] is False and plan['actual_binding'] is None and plan['actual_qualification'] is None
    assert plan['actual_release'] is None and not list(out.rglob('binding.json'))
    assert plan['legacy_temporal_exact_must_remain_false'] is True
    with pytest.raises(RuntimeError):p.prepare(out)
