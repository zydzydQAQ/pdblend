import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT=Path(__file__).resolve().parents[2]/'scripts/2026-09-23_prepare_power_pair.py'
spec=importlib.util.spec_from_file_location('prepare_power_pair_test',SCRIPT)
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)


def test_predecessor_selection_keeps_all_quad_members_and_excludes_superseded():
    jobs={f'quad-{i}':dict(status='queued',payload=dict(cohort_id='quad-profile-4-2-1-1-x',profile_wave_member=str(i))) for i in range(4)}
    jobs['old']=dict(status='blocked',payload=dict(cohort_id='quad-profile-4-2-1-1-x',profile_wave_member='0'))
    assert helper.after_quad(dict(jobs=jobs))==['quad-0','quad-1','quad-2','quad-3']
    jobs['old']['status']='queued'
    with pytest.raises(ValueError,match='ambiguous'):helper.after_quad(dict(jobs=jobs))


def test_all_original_absolute_inputs_are_readonly_covered(tmp_path):
    package=tmp_path/'package';package.mkdir();nested=package/'comparison.json';nested.write_text('{}')
    original=tmp_path/'original';original.mkdir();raw=original/'raw.json';raw.write_text('{}')
    training=tmp_path/'training.json';training.write_text('{}')
    manifest=dict(inputs={k:dict(path=str(p),sha256=helper.digest(p)) for k,p in
        [('original_raw',raw),('training_raw',training),('training_comparison',nested)]})
    mounts=helper.mounts_for_package(package,manifest)
    assert set(mounts)=={package,original,training}
    for binding in manifest['inputs'].values():
        path=Path(binding['path'])
        assert any(path==m or (m.is_dir() and path.is_relative_to(m)) for m in mounts)
    training.write_text('tampered')
    with pytest.raises(ValueError,match='checksum'):helper.mounts_for_package(package,manifest)


def test_resident_timing_hook_is_bound_and_mounted_only_on_32b_member(tmp_path,monkeypatch):
    from pdblend.profile import timing_calibration
    packages=[tmp_path/'7b',tmp_path/'32b']
    manifests={}
    for package,model in zip(packages,['Qwen2.5-7B-Instruct','Qwen2.5-32B-Instruct']):
        package.mkdir()
        manifest=dict(model_id=model,tp=4,inputs={},candidate_sha256='candidate',plan_sha256='plan',timing_component_passed=False)
        (package/'manifest.json').write_text(json.dumps(manifest));manifests[package]=manifest
    monkeypatch.setattr(helper,'load_package',lambda package:(manifests[package],dict(cpu_scheduling_proxy=dict(combined_proxy_seconds=1)),None))
    timing=tmp_path/'timing';timing.mkdir();original=tmp_path/'original';original.mkdir()
    raw=original/'raw.json';raw.write_text('{}')
    manifest=dict(model_id='Qwen2.5-32B-Instruct',tp=4,pp=1,
                  inputs=dict(original_raw=dict(path=str(raw),sha256=helper.digest(raw))))
    (timing/'manifest.json').write_text(json.dumps(manifest))
    for name in ('candidate.json','timing-plan.json'):(timing/name).write_text('{}')
    checked=[]
    monkeypatch.setattr(timing_calibration,'load_package',lambda path:checked.append(path))
    state=dict(jobs={str(i):dict(status='running',payload=dict(cohort_id='quad-profile-4-2-1-1-current',
                        profile_wave_member=str(i))) for i in range(4)})
    review=helper.build_jobs(packages,state,tmp_path/'src','source',tmp_path/'verification','image',tmp_path/'review',timing)
    first,second=review['jobs']
    assert checked==[timing] and review['wave']['keep_peers_resident_until_all_done'] is True
    assert review['wave']['synchronize_parallel_windows'] is True
    assert '--timing-package' not in first['payload']['argv']
    assert '--timing-package' in second['payload']['argv']
    assert f'{timing}:{timing}:ro' in second['payload']['argv']
    assert f'{original}:{original}:ro' in second['payload']['argv']
    assert 'timing_package_binding' not in first['payload']
    assert second['payload']['timing_package_binding']['files']['manifest.json']==helper.digest(timing/'manifest.json')
    assert second['payload']['depends_on']==first['payload']['depends_on']==['0','1','2','3']
