from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from pdblend.profile.query.versions import VersionError, load_profile
from pdblend.profile.query.compatibility import numerical_sources
from pdblend.profile.query.runtime import RuntimeQualificationError
from pdblend.profile.calibration.runtime_components import audit_runtime
from test_power_override import model


def binding(path):
    return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def test_unified_legacy_selection_records_identity_without_promoting_quality(tmp_path):
    instance=model();instance.model='Qwen2.5-7B-Instruct'
    instance.save(tmp_path/'base.json')
    select=tmp_path/'selection.json'
    select.write_text(json.dumps(dict(kind='pdblend_profile_selection_v1',profile_path='base.json')))
    loaded=load_profile(select,system='pdblend',model_id=instance.model,tp=1)
    assert loaded.model.decode_power_w(8,900,ctx=1000)==instance.decode_power_w(8,900,ctx=1000)
    assert loaded.model.calibration_identity==loaded.identity
    assert loaded.model.calibration_coverage==loaded.coverage
    assert loaded.model.calibration_qualification['formal_eligible'] is False
    assert loaded.manifest_fields()['profile_key']['source']=='explicit_legacy_profile'
    with pytest.raises(VersionError,match='formal qualification'):
        load_profile(select,system='pdblend',model_id=instance.model,tp=1,usage='formal')
    with pytest.raises(VersionError,match='identity mismatch'):
        load_profile(select,system='pdblend',model_id=instance.model,tp=2)
    with pytest.raises(VersionError,match='latest'):
        load_profile(registry='ignored',version_id='latest',system='pdblend',model_id=instance.model,tp=1)


def test_no_nested_or_ambiguous_selection(tmp_path):
    select=tmp_path/'selection.json'
    select.write_text(json.dumps(dict(kind='pdblend_profile_selection_v1',profile_path='selection.json')))
    with pytest.raises(VersionError,match='nested'):
        load_profile(select,system='pdblend',model_id='fixture',tp=1)
    with pytest.raises(VersionError,match='OR'):
        load_profile(select,registry='x',version_id='x',system='pdblend',model_id='fixture',tp=1)


def test_moved_frozen_implementation_requires_its_own_hash_not_alias(tmp_path):
    root=tmp_path/'pdblend'/'profile';(root/'query').mkdir(parents=True)
    alias=root/'power_table.py';alias.write_text('from pdblend.profile.query.power_table import *\n')
    moved=root/'query'/'power_table.py'
    moved.write_text('def context_bounds(spec, batch): return (1, 2)\ndef predict(spec, batch, context): return 1\n')
    manifest={str(alias.relative_to(tmp_path)):binding(alias)['sha256']}
    with pytest.raises(ValueError,match='alias is insufficient'):
        numerical_sources(root,manifest,['power_table.py'])
    manifest[str(moved.relative_to(tmp_path))]=binding(moved)['sha256']
    assert numerical_sources(root,manifest,['power_table.py'])['power_table.py']==moved
    moved.write_text(moved.read_text().replace('return 1','return 999'))
    with pytest.raises(ValueError,match='checksum mismatch'):
        numerical_sources(root,manifest,['power_table.py'])


def test_moved_frozen_numerical_equations_are_compared_with_compiled_queries(tmp_path):
    from pdblend.profile.query import model as module
    from pdblend.profile.query.compatibility import check
    current=Path(module.__file__).parent
    root=tmp_path/'pdblend'/'profile';(root/'query').mkdir(parents=True)
    mapping={'model.py':'model.py','power_table.py':'power_table.py','decode_fit.py':'decode.py'}
    manifest={}
    for entrance,implementation in mapping.items():
        alias=root/entrance;alias.write_text('from elsewhere import *\n')
        moved=root/'query'/implementation;moved.write_bytes((current/implementation).read_bytes())
        for path in (alias,moved):manifest[str(path.relative_to(tmp_path))]=binding(path)['sha256']
    sources=numerical_sources(root,manifest,list(mapping))
    result=check(model(),root,sources=sources)
    assert result['passed'] and result['scalar_checks']>30


@pytest.mark.historical
def test_real_frozen_components_combine_with_reproduced_runtime_measurements(tmp_path):
    registry=Path(__file__).resolve().parents[2]/'results/2026-09-23/calibration-versions-v1/registry.json'
    if not registry.is_file():pytest.skip('historical immutable registry unavailable')
    for i,row in enumerate(json.loads(registry.read_text())['versions']):
        base=Path(row['evidence']['power_candidate']['path'])
        raw=Path(row['original_inputs']['training_raw']['path'])
        receipt=tmp_path/f'runtime-{i}.json'
        result=audit_runtime(base,raw,out=receipt)
        assert result['passed']
        assert not result['independent_runtime_holdout_passed']
        descriptor=tmp_path/f'selection-{i}.json'
        descriptor.write_text(json.dumps(dict(kind='pdblend_profile_selection_v1',registry=str(registry),
            version_id=row['version_id'],runtime_base=dict(profile=binding(base),raw=binding(raw),audit=binding(receipt)))))
        kwargs={k:row[k] for k in ('system','model_id','tp','pp')}
        loaded=load_profile(descriptor,**kwargs)
        loaded.model.require_runtime_components('capacity','static','transfer','clock_transition')
        assert loaded.model.kv_capacity_tokens>0
        assert loaded.model.static_power_w('parked')>0
        assert loaded.model.transfer_seconds(512)>0
        assert loaded.model.freq_switch_s>=0
        assert not loaded.qualification['transition_energy_qualified']
        uncomposed=load_profile(registry=registry,version_id=row['version_id'],**kwargs)
        with pytest.raises(RuntimeQualificationError,match='capacity'):
            _=uncomposed.model.kv_capacity_tokens
        assert loaded.model.step_seconds(32,1200,1500)==uncomposed.model.step_seconds(32,1200,1500)
