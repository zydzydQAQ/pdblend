"""Portable release archives preserve bytes and reject unsafe restoration."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile

import pytest


SCRIPT = Path(__file__).resolve().parents[2] / 'scripts/2026-09-25_release_assets.py'
SPEC = importlib.util.spec_from_file_location('release_assets', SCRIPT)
release = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release)


def pack_fixture(tmp_path):
    project = tmp_path / 'project'
    (project / 'inputs').mkdir(parents=True)
    first = project / 'inputs/a.json'
    second = project / 'inputs/z.sh'
    first.write_bytes(b'{"identity":"original"}\n')
    second.write_bytes(b'#!/bin/sh\nexit 0\n')
    second.chmod(0o755)
    selection = tmp_path / 'selection.json'
    selection.write_text(json.dumps({'paths': ['inputs']}))
    packed = release.pack(project, selection, tmp_path / 'pack')
    return project, first, second, packed


def test_pack_verify_restore_roundtrip_and_idempotence(tmp_path):
    project, first, second, packed = pack_fixture(tmp_path)
    expected = {p.relative_to(project).as_posix(): p.read_bytes() for p in (first, second)}
    manifest = release.verify(packed['archive'])
    assert manifest['project_root'] == str(project)
    assert manifest['files'] == {
        name: {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
        for name, data in expected.items()
    }
    assert release.digest_file(packed['archive']) == packed['sha256']
    first.unlink()
    second.unlink()
    assert release.extract(packed['archive'], project)['restored_files'] == 2
    assert {name: (project / name).read_bytes() for name in expected} == expected
    assert second.stat().st_mode & 0o111
    assert release.extract(packed['archive'], project)['restored_files'] == 0


def test_restore_checks_all_conflicts_before_writing_missing_assets(tmp_path):
    project, first, second, packed = pack_fixture(tmp_path)
    first.unlink()
    second.write_text('unrelated user content')
    with pytest.raises(FileExistsError, match='refusing overwrite'):
        release.extract(packed['archive'], project)
    assert not first.exists()
    assert second.read_text() == 'unrelated user content'


def test_restore_requires_recorded_root(tmp_path):
    _, _, _, packed = pack_fixture(tmp_path)
    other = tmp_path / 'other'
    with pytest.raises(ValueError, match='recorded project root'):
        release.extract(packed['archive'], other)
    assert not other.exists()


@pytest.mark.parametrize('link_parent', [False, True])
def test_restore_rejects_symlink_target_and_parent(tmp_path, link_parent):
    project, first, second, packed = pack_fixture(tmp_path)
    first.unlink()
    outside = tmp_path / 'outside'
    outside.mkdir()
    if link_parent:
        second.unlink()
        first.parent.rmdir()
        first.parent.symlink_to(outside, target_is_directory=True)
    else:
        first.symlink_to(outside / 'victim')
    with pytest.raises(ValueError, match='symlink restoration path'):
        release.extract(packed['archive'], project)
    assert not list(outside.iterdir())


def write_archive(path, members, project, records=None):
    """Construct malicious headers without relying on filesystem traversal."""
    if records is None:
        records = {name: {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
                   for name, data, _ in members}
    manifest = {'schema': 'pdblend-v3-input-pack/v1',
                'project_root': str(project), 'files': records}
    with tarfile.open(path, 'w:gz') as archive:
        for name, data, kind in members:
            member = tarfile.TarInfo(name)
            member.type = kind
            member.size = len(data) if kind == tarfile.REGTYPE else 0
            member.linkname = '../../outside'
            archive.addfile(member, io.BytesIO(data))
        data = json.dumps(manifest).encode()
        member = tarfile.TarInfo('INPUTS-MANIFEST.json')
        member.size = len(data)
        archive.addfile(member, io.BytesIO(data))


@pytest.mark.parametrize('name', ['../outside', '/tmp/outside', 'inputs/../../outside',
                                   'inputs//outside', './outside'])
def test_restore_rejects_path_traversal_before_writing(tmp_path, name):
    project = tmp_path / 'project'
    archive = tmp_path / 'bad.tar.gz'
    write_archive(archive, [(name, b'attack', tarfile.REGTYPE)], project)
    with pytest.raises(ValueError, match='unsafe archive path'):
        release.extract(archive, project)
    assert not project.exists()
    assert not (tmp_path / 'outside').exists()


@pytest.mark.parametrize('kind', [tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.DIRTYPE])
def test_restore_rejects_nonregular_members(tmp_path, kind):
    project = tmp_path / 'project'
    archive = tmp_path / 'bad.tar.gz'
    write_archive(archive, [('inputs/member', b'', kind)], project)
    with pytest.raises(ValueError, match='non-regular'):
        release.extract(archive, project)
    assert not project.exists()


def test_verify_rejects_duplicate_member_and_checksum_mismatch(tmp_path):
    archive = tmp_path / 'bad.tar.gz'
    member = ('inputs/member', b'original', tarfile.REGTYPE)
    write_archive(archive, [member, member], tmp_path / 'project')
    with pytest.raises(ValueError, match='duplicate'):
        release.verify(archive)
    write_archive(archive, [member], tmp_path / 'project',
                  records={'inputs/member': {'bytes': len(member[1]), 'sha256': '0' * 64}})
    with pytest.raises(ValueError, match='contents differ'):
        release.verify(archive)
