"""Optional timing collection cannot rewrite the power measurement contract."""
import json
from types import SimpleNamespace as NS

import pytest

from pdblend.profile import power_calibration as pc, timing_calibration


@pytest.mark.asyncio
@pytest.mark.parametrize('violation', ['raw', 'model', 'plan', 'package', 'receipt_checksum', 'formal'])
async def test_timing_hook_rejects_mutation_and_unbound_receipts(tmp_path, monkeypatch, violation):
    package=tmp_path/'package';package.mkdir()
    (package/'manifest.json').write_text(json.dumps(dict(model_id='Qwen2.5-32B-Instruct',tp=4,pp=1)))
    for name in ('candidate.json','timing-plan.json'):(package/name).write_text('{}')
    profiler=NS(raw={'decode':[{'power_w':400}]})
    model_data={'power':400};model=NS(to_json=lambda:json.dumps(model_data));plan={'points':[1]}
    async def collect(**kwargs):
        out=kwargs['out'];out.mkdir()
        value=dict(complete=True,timing_passed=True,formal_eligible=False,energy_comparable=False)
        if violation=='raw':profiler.raw['decode'][0]['power_w']=300
        if violation=='model':model_data['power']=300
        if violation=='plan':plan['points'].append(2)
        if violation=='package':(package/'candidate.json').write_text('{"changed":true}')
        if violation=='formal':value['formal_eligible']=True
        (out/'completion.json').write_text(json.dumps(value))
        return dict(value,receipt_sha256='wrong' if violation=='receipt_checksum' else pc.digest(out/'completion.json'))
    monkeypatch.setattr(timing_calibration,'collect_existing',collect)
    with pytest.raises(ValueError):
        await pc.collect_resident_timing(profiler=profiler,client=object(),gpus=[0,1,2,3],package=package,
            out=tmp_path/'timing',power_model=model,power_plan=plan)


def test_timing_hook_requires_model_bound_complete_package(tmp_path):
    (tmp_path/'manifest.json').write_text(json.dumps(dict(model_id='Qwen2.5-7B-Instruct',tp=4,pp=1)))
    with pytest.raises(ValueError,match='32B'):pc.timing_package_binding(tmp_path)
    (tmp_path/'manifest.json').write_text(json.dumps(dict(model_id='Qwen2.5-32B-Instruct',tp=4,pp=1)))
    with pytest.raises(ValueError,match='incomplete'):pc.timing_package_binding(tmp_path)
