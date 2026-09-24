import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def bootstrap():
    path = Path(__file__).resolve().parents[2] / 'scripts/2026-09-24_host_analysis_bootstrap.py'
    spec = importlib.util.spec_from_file_location('host_analysis_bootstrap', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def snapshot(tmp_path, bootstrap):
    text = 'version = 1\n'
    files = {'pdblend/__init__.py': hashlib.sha256(text.encode()).hexdigest()}
    identity = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    source = tmp_path / 'host-analysis/sources' / identity
    (source / 'pdblend').mkdir(parents=True)
    (source / 'pdblend/__init__.py').write_text(text)
    manifest = source / 'manifest.json'
    manifest.write_text(json.dumps(dict(files=files, source_sha256=identity)))
    (tmp_path / 'host-analysis/active-source.json').write_text(json.dumps(dict(
        schema='matrix-host-analysis-selection/v1',
        source_manifest=dict(path=str(manifest), sha256=bootstrap.sha(manifest)))))
    return source


def test_selects_only_exact_bound_host_snapshot(bootstrap, tmp_path):
    source = snapshot(tmp_path, bootstrap)
    assert bootstrap.resolve_source(tmp_path) == source
    (source / 'pdblend/__init__.py').write_text('version = 2\n')
    with pytest.raises(ValueError, match='inventory changed'):
        bootstrap.resolve_source(tmp_path)


def test_extra_python_file_changes_source_identity(bootstrap, tmp_path):
    source = snapshot(tmp_path, bootstrap)
    (source / 'extra.py').write_text('extra = True\n')
    with pytest.raises(ValueError, match='inventory changed'):
        bootstrap.resolve_source(tmp_path)


def test_manifest_mutation_is_rejected(bootstrap, tmp_path):
    source = snapshot(tmp_path, bootstrap)
    (source / 'manifest.json').write_text('{}')
    with pytest.raises(ValueError, match='manifest changed'):
        bootstrap.resolve_source(tmp_path)


def test_no_selector_leaves_existing_explicit_environment(bootstrap, tmp_path):
    assert bootstrap.resolve_source(tmp_path) is None
    assert bootstrap.activate(['--different-option']) is None
