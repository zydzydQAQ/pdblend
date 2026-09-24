#!/usr/bin/env python3
"""Pack and verify immutable v3 inputs; restore without overwriting different bytes.

Uses only Python's standard library. Absolute provenance is preserved: restore
the input pack to its recorded project root, then create a NEW experiment queue.
Qwen weights travel separately and are checked with verify-models.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile


def digest_file(path):
    with Path(path).open('rb') as handle:
        return digest_stream(handle)


def digest_stream(handle):
    value = hashlib.sha256()
    for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
        value.update(block)
    return value.hexdigest()


def safe_name(name):
    path = PurePosixPath(name)
    if not name or path.is_absolute() or '..' in path.parts or str(path) != name:
        raise ValueError(f'unsafe archive path: {name}')
    return path


def pack(project, selection, out):
    project = Path(project).resolve()
    data = json.loads(Path(selection).read_text())
    paths = data if isinstance(data, list) else data.get('paths')
    if paths is None:
        paths = [item['path'] for item in data['files']]
        for item in data['files']:
            if digest_file(item['path']) != item['sha256']:
                raise ValueError('selected asset changed: ' + item['path'])
    selected = set()
    for raw in paths:
        path = Path(raw)
        if not path.is_absolute():
            path = project / path
        if path.is_symlink() or not path.resolve().is_relative_to(project):
            raise ValueError(f'input outside project or symlink: {path}')
        if not path.exists():
            raise FileNotFoundError(path)
        for item in path.rglob('*') if path.is_dir() else [path]:
            if item.is_symlink():
                raise ValueError(f'symlink input: {item}')
            if item.is_file():
                selected.add(item.resolve())
    if not selected:
        raise ValueError('empty input selection')
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    archive = out / 'pdblend4-v3-inputs.tar.gz'
    manifest_path = out / 'inputs-manifest.json'
    if archive.exists() or manifest_path.exists():
        raise FileExistsError('use a fresh output directory')
    manifest = dict(schema='pdblend-v3-input-pack/v1', project_root=str(project), files={})
    with tarfile.open(archive, 'x:gz', compresslevel=1, dereference=True) as tar:
        for path in sorted(selected):
            name = str(path.relative_to(project))
            safe_name(name)
            if name == 'INPUTS-MANIFEST.json':
                raise ValueError('reserved archive member')
            before = path.stat()
            record = dict(bytes=before.st_size, sha256=digest_file(path))
            tar.add(path, arcname=name, recursive=False)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
                raise ValueError(f'input changed during pack: {path}')
            manifest['files'][name] = record
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
        tar.add(manifest_path, arcname='INPUTS-MANIFEST.json', recursive=False)
    verify(archive)
    checksum = digest_file(archive)
    (out / (archive.name + '.sha256')).write_text(f'{checksum}  {archive.name}\n')
    return dict(archive=str(archive), sha256=checksum, files=len(selected),
                bytes=sum(v['bytes'] for v in manifest['files'].values()))


def verify(archive):
    actual, manifest = {}, None
    with tarfile.open(archive, 'r:gz') as tar:
        for member in tar:
            safe_name(member.name)
            if not member.isfile() or member.name in actual:
                raise ValueError(f'non-regular or duplicate member: {member.name}')
            handle = tar.extractfile(member)
            if member.name == 'INPUTS-MANIFEST.json':
                if manifest is not None or member.size > 64 * 1024 * 1024:
                    raise ValueError('duplicate or oversized manifest')
                manifest = json.load(handle)
            else:
                actual[member.name] = dict(bytes=member.size, sha256=digest_stream(handle))
    if not manifest or manifest.get('schema') != 'pdblend-v3-input-pack/v1' or actual != manifest.get('files'):
        raise ValueError('archive contents differ from manifest')
    root = Path(manifest['project_root'])
    if not root.is_absolute() or root == Path('/'):
        raise ValueError('invalid recorded project root')
    return manifest


def extract(archive, project):
    manifest = verify(archive)
    project = Path(project).resolve()
    if str(project) != manifest['project_root']:
        raise ValueError('immutable references require recorded project root: ' + manifest['project_root'])
    # Check every conflict before writing any new asset.
    for name, record in manifest['files'].items():
        target = project / name
        if target.is_symlink() or target.resolve() != target:
            raise ValueError(f'symlink restoration path: {target}')
        if target.exists() and (not target.is_file() or digest_file(target) != record['sha256']):
            raise FileExistsError(f'different existing asset, refusing overwrite: {target}')
    count, seen = 0, set()
    with tarfile.open(archive, 'r:gz') as tar:
        for member in tar:
            safe_name(member.name)
            if not member.isfile() or member.name in seen:
                raise ValueError('archive changed after verification: ' + member.name)
            seen.add(member.name)
            if member.name == 'INPUTS-MANIFEST.json':
                if member.size > 64 * 1024 * 1024 or json.load(tar.extractfile(member)) != manifest:
                    raise ValueError('archive manifest changed after verification')
                continue
            record = manifest['files'].get(member.name)
            if record is None or member.size != record['bytes']:
                raise ValueError('archive member changed after verification: ' + member.name)
            target = project / member.name
            if target.is_symlink() or target.resolve() != target:
                raise ValueError(f'symlink restoration path: {target}')
            if target.exists():
                if not target.is_file() or digest_file(target) != record['sha256']:
                    raise FileExistsError(f'destination changed during restoration: {target}')
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            # Never expose a partially copied final file, and never overwrite a
            # destination created concurrently after the conflict preflight.
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(dir=target.parent, prefix='.v3-restore-', delete=False) as handle:
                    temporary = Path(handle.name)
                    shutil.copyfileobj(tar.extractfile(member), handle, 8 * 1024 * 1024)
                if digest_file(temporary) != record['sha256']:
                    raise ValueError(f'extracted asset mismatch: {target}')
                temporary.chmod(0o755 if member.mode & 0o111 else 0o644)
                os.link(temporary, target)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
            count += 1
    if seen != set(manifest['files']) | {'INPUTS-MANIFEST.json'}:
        raise ValueError('archive members changed after verification')
    return dict(verified_files=len(manifest['files']), restored_files=count, project=str(project))


def verify_models(root):
    root = Path(root).resolve()
    reports = []
    for size in ('7B', '14B', '32B'):
        model = root / f'Qwen2.5-{size}-Instruct'
        path = model / 'pdblend-model-manifest.json'
        pinned = Path(__file__).resolve().parents[1] / 'requirements/models' / model.name / path.name
        if digest_file(path) != digest_file(pinned):
            raise ValueError(f'model manifest differs from the v3 release: {path}')
        manifest = json.loads(path.read_text())
        files = manifest['files']
        if not files or manifest['repo_id'] != f'Qwen/Qwen2.5-{size}-Instruct':
            raise ValueError(f'invalid model manifest: {path}')
        names = set()
        for item in files:
            name = str(safe_name(item['path']))
            if name in names:
                raise ValueError('duplicate model member')
            names.add(name)
            file = model / name
            if not file.resolve().is_relative_to(model):
                raise ValueError('model path escapes model directory')
            if file.stat().st_size != item['bytes'] or digest_file(file) != item['sha256']:
                raise ValueError(f'model checksum differs: {file}')
        reports.append(dict(model=model.name, files=len(files), manifest_sha256=digest_file(path), verified=True))
    return dict(schema='pdblend-v3-model-verification/v1', models=reports)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('pack')
    p.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--selection', type=Path, required=True, help='JSON list or object with paths list')
    p.add_argument('--out', type=Path, required=True)
    p = sub.add_parser('verify'); p.add_argument('archive', type=Path)
    p = sub.add_parser('extract'); p.add_argument('archive', type=Path)
    p.add_argument('--project', type=Path, required=True)
    p = sub.add_parser('verify-models'); p.add_argument('--models-root', type=Path, required=True)
    p.add_argument('--out', type=Path)
    args = parser.parse_args()
    if args.command == 'pack':
        result = pack(args.project, args.selection, args.out)
    elif args.command == 'verify':
        result = verify(args.archive)
        result = dict(verified=True, files=len(result['files']), project_root=result['project_root'])
    elif args.command == 'extract':
        result = extract(args.archive, args.project)
    else:
        result = verify_models(args.models_root)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            with args.out.open('x') as handle:
                json.dump(result, handle, indent=2); handle.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
