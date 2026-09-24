"""New lifecycle requires explicit, byte-bound review; old proof is unchanged."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from pdblend.bench import comparison_ecoserve_inputs as module
from pdblend.bench.comparison_ecoserve_inputs import validate_ecoserve_inputs
from pdblend.bench.resident_session import digest
from test_comparison_ecoserve import eco_fixture, put


def fixture(tmp_path, monkeypatch):
    args = eco_fixture(tmp_path, monkeypatch)
    point, identity = args['point'], args['engine_identity']
    config = json.loads(Path(point['inputs']['system_config']['path']).read_bytes())
    original = json.loads(Path(args['startup_qualification']['source_manifest']['path']).read_bytes())
    directory = tmp_path/'new-source'
    directory.mkdir()
    old_dir = Path(args['startup_qualification']['source_manifest']['path']).parent
    files = deepcopy(original['files'])
    for name in files:
        target = directory/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((old_dir/name).read_bytes())
    checksums = {}
    for name in ('pdblend_baselines/ecoserve/run_native.py',
                 'pdblend_baselines/ecoserve/comparison_lifecycle.py',
                 'pdblend/bench/comparison_ecoserve_lifecycle.py'):
        target = directory/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text('fixture new lifecycle: '+name)
        checksums[name] = files[name] = hashlib.sha256(target.read_bytes()).hexdigest()
    monkeypatch.setattr(module, 'REVIEWED_LIFECYCLE_WRAPPERS', checksums)
    monkeypatch.setattr(module, 'REVIEWED_WRAPPERS', {'pdblend_baselines/ecoserve/run_native.py': 'old-runner'})
    source = put(directory, 'manifest.json', dict(files=files, source_sha256=digest(files)))
    config['eco_comparison_lifecycle'] = point['eco_comparison_lifecycle'] = identity['eco_comparison_lifecycle'] = module.LIFECYCLE_MODE
    point['inputs']['system_config'] = put(tmp_path, 'new-config.json', config)
    point['inputs']['source_manifest'] = source
    review = dict(schema='ecoserve-comparison-lifecycle-review/v1', mode=module.LIFECYCLE_MODE,
        scope='comparison_wrapper_lifecycle_cpu_review', formal_eligible=False, hardware_qualified=False,
        base_runner_sha256='old-runner', protected_mechanism_source_sha256=module.MECHANISM_SOURCE,
        source_files=checksums, checks={k:True for k in module.LIFECYCLE_CHECKS})
    point['inputs']['eco_lifecycle_review'] = put(tmp_path, 'lifecycle-review.json', review)
    monkeypatch.setattr(module, 'LIFECYCLE_REVIEW_SHA256', point['inputs']['eco_lifecycle_review']['sha256'])
    return args, config, review, source


def test_wrapper_review_preserves_core_inheritance_without_promoting_cpu_evidence(tmp_path, monkeypatch):
    args, _, _, source = fixture(tmp_path, monkeypatch)
    result = validate_ecoserve_inputs(args['point'], args['engine_identity'], source_manifest=source)
    assert result['preflight_ready'], result
    assert result['source_continuity']['comparison_lifecycle'] == module.LIFECYCLE_MODE
    assert not result['formal_eligible'] and not result['full_profile_qualified']


@pytest.mark.parametrize('damage', ['missing', 'review_sha', 'review_source', 'review_check',
    'formal', 'hardware', 'point', 'identity', 'config', 'core', 'runner'])
def test_unbound_changed_or_unselected_lifecycle_source_cannot_inherit_qualification(tmp_path, monkeypatch, damage):
    args, config, review, source = fixture(tmp_path, monkeypatch)
    point, identity = args['point'], args['engine_identity']
    if damage == 'missing':point['inputs'].pop('eco_lifecycle_review')
    elif damage == 'review_sha':point['inputs']['eco_lifecycle_review']['sha256'] = '0'*64
    elif damage == 'review_source':review['source_files'] = {}
    elif damage == 'review_check':review['checks']['exact_http_cancellation_causes'] = False
    elif damage == 'formal':review['formal_eligible'] = True
    elif damage == 'hardware':review['hardware_qualified'] = True
    elif damage == 'point':point.pop('eco_comparison_lifecycle')
    elif damage == 'identity':identity.pop('eco_comparison_lifecycle')
    elif damage == 'config':config.pop('eco_comparison_lifecycle')
    elif damage in ('core', 'runner'):
        name = 'pdblend_runtime/serve.py' if damage == 'core' else 'pdblend_baselines/ecoserve/run_native.py'
        (Path(source['path']).parent/name).write_text('changed bytes')
    if damage in ('review_source', 'review_check', 'formal', 'hardware'):
        point['inputs']['eco_lifecycle_review'] = put(tmp_path, 'changed-review.json', review)
    if damage == 'config':point['inputs']['system_config'] = put(tmp_path, 'changed-config.json', config)
    result = validate_ecoserve_inputs(point, identity, source_manifest=source)
    assert not result['preflight_ready'] and 'source.reviewed_compatibility' in result['missing_gates']
