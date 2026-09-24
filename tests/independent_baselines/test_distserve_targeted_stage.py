import importlib.util
import json
from pathlib import Path

import pytest

from pdblend_baselines.distserve.stage_collect import load_point_plan, rows_from_window
from pdblend_baselines.distserve.deployment import execute_deployment


def loader():
    path=Path(__file__).resolve().parents[2]/'scripts/2026-09-23_prepare_distserve_targeted.py'
    spec=importlib.util.spec_from_file_location('targeted_distserve_prepare',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_targeted_points_come_from_calibration_tuning_and_keep_all_protocol_windows(tmp_path):
    module=loader()
    for dataset in ('alpaca','sharegpt','longbench'):
        (tmp_path/(dataset+'.json')).write_text(json.dumps(dict(model_name='Qwen2.5-7B-Instruct',
            calibration=[dict(input_tokens=32,output_tokens=16),dict(input_tokens=7168,output_tokens=512)],
            tuning=[dict(input_tokens=128,output_tokens=32)],
            evaluation=[dict(input_tokens=1,output_tokens=8191)])))
    value=module.point_plan(tmp_path);path=tmp_path/'points.json';path.write_text(json.dumps(value))
    points=load_point_plan(path,'Qwen2.5-7B-Instruct',1)
    assert len(points)==120 and sum(p['repeats'] for p in points)==360
    assert {p['frequency_mhz'] for p in points}=={900,1200,1500,1800,2100,2520}
    assert min(min(p['lengths']) for p in points)==32  # Evaluation is never used to pick shapes.
    assert max(len(p['lengths']) for p in points)==4
    assert all(sum(p['lengths'])<=8192 for p in points if p['role']=='prefill')
    value['selection_split']='evaluation';path.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='calibration/tuning'):load_point_plan(path,'Qwen2.5-7B-Instruct',1)


def test_point_plan_cannot_weaken_repeats_or_invent_frequency(tmp_path):
    point=dict(frequency_mhz=2520,role='decode',purpose='training',lengths=[128],repeats=1)
    value=dict(schema='distserve-targeted-stage-plan-v1',system='distserve',model_id='Qwen2.5-7B-Instruct',
               tp=1,pp=1,selection_split='tuning',evaluation_used_for_selection=False,points=[point])
    path=tmp_path/'points.json';path.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='protocol'):load_point_plan(path,'Qwen2.5-7B-Instruct',1)
    point.update(repeats=3,frequency_mhz=2400);path.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='protocol'):load_point_plan(path,'Qwen2.5-7B-Instruct',1)


@pytest.mark.asyncio
async def test_positional_dispatch_duration_accepts_api_but_rejects_unselected_plan_before_http(tmp_path):
    with pytest.raises(ValueError,match='calibration-only'):
        await execute_deployment({},[],tmp_path/'trace.json',tmp_path/'run',300.)
    assert not (tmp_path/'run').exists()


def test_stage_windows_do_not_accept_short_duration_as_full_decode_calibration():
    raw=dict(status='measured',point=dict(role='decode'),capability=dict(tp=1),settle_s=2,
             start_s=10.,end_s=14.9,sampler_error=None,cleanup_errors=[])
    with pytest.raises(ValueError,match='protocol'):rows_from_window(raw)
