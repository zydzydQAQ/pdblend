#!/usr/bin/env python3
"""Delete an exact audited allowlist; preserve referenced evidence and tombstones."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import time

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/2026-09-23'
CATALOG = {f'dynamo-functional-job-specs-v{i}': 'dynamo-functional-job-specs-final-v2' for i in range(2, 10)}
CATALOG.update({
    'incremental-profile-wave-v1': 'incremental-profile-wave-v2',
    '32b-power-training-review': '32b-power-training-review-final',
    '7b-tp4-power-training-review': '7b-tp4-power-training-review-final',
    '32b-tp4-timing-overlay-package': '32b-tp4-timing-overlay-package-final',
    'incremental-profile-sources/6ca08b2b7598b41e060c4d63500cafbfdfc94ffaf53efbc3d41482bd89598aed':
        'incremental-profile-sources/ae694160b8e2b3426306e4fb8879b9a2ac28edd2498f583f340f584a89e13a29',
    'power-pair-review': 'power-pair-resident-v2',
    'power-holdout-sources/a99f0b15eed686f4788bee79b5ebc8ea23201bbb8da48ed65a38eaa6577194d7':
        'power-holdout-sources/f7f443a115ccee96bdf455dbe4e4fb8b0f3e8b2efbdbd3d9e70f9c9573f4a49b',
})
RETAINED = {
    'power-pair-resident-v1': 'referenced by resident-v2 historical README',
    'power-holdout-sources/998a6892b1a8b93d13a029173b255037f95836cc317b8cc7faf03dfadebe48c2': 'referenced by retained resident-v1',
    'long-context-followup-v1': 'supersedes provenance reference from long-context-followup-v2/review.json',
    'incremental-profile-wave-v2': 'active queue mount and compiler cache working copy',
    'quad-synchronized-resume-v1': 'queue and raw sample provenance',
    'quad-compiler-cache-archive': 'bound compiler cache source',
}
TEXT_SUFFIXES = {'.json', '.jsonl', '.md', '.py', '.sh', '.txt', '.toml', '.yaml', '.yml', '.csv', '.log'}


def inventory(path):
    if path.is_symlink() or any(x.is_symlink() for x in path.rglob('*')):
        raise ValueError('cleanup target contains a symlink: ' + str(path))
    files = []
    for item in sorted(path.rglob('*')):
        if item.is_file():
            data = item.read_bytes()
            files.append(dict(path=str(item.relative_to(path)), bytes=len(data),
                              sha256=hashlib.sha256(data).hexdigest()))
    tree = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return dict(path=str(path), bytes=sum(f['bytes'] for f in files), tree_sha256=tree, files=files)


def references(targets):
    patterns = [(target, re.compile(re.escape(str(target.relative_to(ROOT)).encode()) + rb'(?![A-Za-z0-9_.-])'))
                for target in targets]
    hits, scanned = [], 0
    if not patterns:
        return hits, scanned
    overlap = max(len(str(p)) for p in targets) + 8
    for path in ROOT.rglob('*'):
        if (not path.is_file() or path.is_symlink() or path.suffix.lower() not in TEXT_SUFFIXES
                or '.git' in path.parts or '__pycache__' in path.parts
                or path.resolve() == Path(__file__).resolve()
                or any(path.is_relative_to(target) for target in targets)
                or path.parent == ROOT / 'results/archive' and path.name.startswith('cleanup-')):
            continue
        scanned += 1
        found, tail = set(), b''
        with path.open('rb') as stream:
            while chunk := stream.read(1024 * 1024):
                data = tail + chunk
                for target, pattern in patterns:
                    if pattern.search(data):
                        found.add(str(target))
                tail = data[-overlap:]
        if found:
            hits.append(dict(path=str(path), targets=sorted(found)))
    return hits, scanned


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--tombstone', type=Path)
    args = parser.parse_args(argv)
    present = [BASE / name for name in CATALOG if (BASE / name).is_dir()]
    for name, replacement in CATALOG.items():
        if (BASE / name).exists() and not (BASE / replacement).is_dir():
            raise RuntimeError('canonical replacement missing: ' + replacement)
    hits, scanned = references(present)
    if hits:
        print(json.dumps(dict(complete=False, references=hits), indent=2))
        raise SystemExit('refusing cleanup: external references remain')
    entries = []
    for path in present:
        row = inventory(path)
        row.update(replacement=str(BASE / CATALOG[str(path.relative_to(BASE))]),
                   refcheck='no external workspace text/JSON/JSONL reference; no file-size cutoff')
        entries.append(row)
    result = dict(schema='cleanup-tombstone-v3', created_at=time.time(), applied=args.apply,
                  entries=entries, total_bytes=sum(x['bytes'] for x in entries),
                  scanned_files=scanned, reference_scope=str(ROOT),
                  retained=[dict(path=str(BASE / name), reason=why) for name, why in RETAINED.items()],
                  complete=not args.apply)
    if args.apply and entries:
        dest = args.tombstone or ROOT / 'results/archive' / f'cleanup-{time.time_ns()}.json'
        dest.parent.mkdir(parents=True, exist_ok=True)
        # Never overwrite a prior cleanup receipt, including an empty rerun.
        with dest.open('x') as stream:
            json.dump(result, stream, indent=2); stream.write('\n')
        for path, recorded in zip(present, entries):
            if inventory(path)['tree_sha256'] != recorded['tree_sha256']:
                raise RuntimeError('cleanup target changed since inventory: ' + str(path))
            shutil.rmtree(path)
        result['complete'] = True
        result['deleted_paths_absent'] = all(not path.exists() for path in present)
        temporary = dest.with_suffix('.tmp')
        temporary.write_text(json.dumps(result, indent=2) + '\n')
        temporary.replace(dest)
        print(json.dumps(dict(tombstone=str(dest), deleted=len(entries), bytes=result['total_bytes'])))
    else:
        print(json.dumps(dict(applied=False, candidates=[x['path'] for x in entries],
                              total_bytes=result['total_bytes'], scanned_files=scanned)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
