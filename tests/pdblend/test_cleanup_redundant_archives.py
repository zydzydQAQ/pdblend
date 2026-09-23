import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def cleaner(tmp_path, monkeypatch):
    path = Path(__file__).parents[2] / 'scripts/2026-09-23_cleanup_redundant_archives.py'
    spec = importlib.util.spec_from_file_location('cleanup_archives', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    monkeypatch.setattr(module, 'BASE', tmp_path / 'results')
    monkeypatch.setattr(module, 'CATALOG', {'old': 'old-final'})
    monkeypatch.setattr(module, 'RETAINED', {})
    for name in ('old', 'old-final'):
        directory = tmp_path / 'results' / name
        directory.mkdir(parents=True)
        (directory / 'artifact.json').write_text('{"sample": 1}\n')
    return module


def test_external_large_jsonl_reference_prevents_deletion(cleaner, tmp_path):
    # Reference crosses a streaming chunk boundary after 2 MiB.
    reference = str(tmp_path / 'results/old/artifact.json').encode()
    blocker = tmp_path / 'raw.jsonl'
    blocker.write_bytes(b' ' * (3 * 1024 * 1024 - len(reference) // 2) + reference + b'\n')
    with pytest.raises(SystemExit, match='external references'):
        cleaner.main(['--apply'])
    assert (tmp_path / 'results/old/artifact.json').is_file()


def test_delete_records_checksums_and_repeat_preserves_tombstone(cleaner, tmp_path):
    # Prefix of canonical old-final is not a reference to old.
    (tmp_path / 'current.json').write_text(json.dumps({'path': str(tmp_path / 'results/old-final')}))
    tombstone = tmp_path / 'results/archive/cleanup-test.json'
    assert cleaner.main(['--apply', '--tombstone', str(tombstone)]) == 0
    before = tombstone.read_bytes()
    value = json.loads(before)
    assert value['complete'] and value['deleted_paths_absent']
    assert len(value['entries'][0]['files'][0]['sha256']) == 64
    assert not (tmp_path / 'results/old').exists()
    assert (tmp_path / 'results/old-final/artifact.json').is_file()
    cleaner.main(['--apply', '--tombstone', str(tombstone)])
    assert tombstone.read_bytes() == before


def test_existing_tombstone_blocks_new_delete(cleaner, tmp_path):
    tombstone = tmp_path / 'existing.json'
    tombstone.write_text('keep original')
    with pytest.raises(FileExistsError):
        cleaner.main(['--apply', '--tombstone', str(tombstone)])
    assert (tmp_path / 'results/old/artifact.json').is_file()
    assert tombstone.read_text() == 'keep original'
