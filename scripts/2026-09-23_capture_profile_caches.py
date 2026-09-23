#!/usr/bin/env python3
"""Preserve compiler caches of this queue's owned quad before a safe restart."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); out = args.out.resolve(); out.mkdir(parents=True, exist_ok=False)
    queue = json.loads((ROOT/'results/2026-09-22/three-model/queue.json').read_text())
    rows = {}
    for name, job in queue['jobs'].items():
        payload = job['payload']
        if job['status'] != 'running' or not payload.get('cohort_id', '').startswith('quad-profile-4-2-1-1-'):
            continue
        member = payload['profile_wave_member']; target = out/member
        inspected = json.loads(subprocess.check_output(['docker', 'inspect', payload['container_name']], text=True))[0]
        if inspected['Image'] != payload['image_digest']:
            raise ValueError('cache container image differs from leased payload')
        subprocess.run(['docker', 'cp', payload['container_name']+':/root/.cache/vllm', str(target)], check=True)
        files = {str(path.relative_to(target)): hashlib.sha256(path.read_bytes()).hexdigest()
                 for path in sorted(target.rglob('*')) if path.is_file() and not path.is_symlink()}
        if not files or any(path.is_symlink() for path in target.rglob('*')):
            raise ValueError('cache must contain ordinary files, no symlinks')
        rows[member] = dict(path=str(target), files=files, source_job=name,
            image_digest=payload['image_digest'], model_id=payload['model_id'], tp=payload['tp'], pp=payload['pp'],
            source_sha256=payload['source_sha256'],
            purpose='same pinned engine compiler cache; numerical and sampling gates remain unchanged')
    if len(rows) != 4: raise ValueError('exactly four running owned quad containers required')
    manifest = out/'manifest.json'; manifest.write_text(json.dumps(dict(schema=1, members=rows), indent=2)+'\n')
    print(manifest)


if __name__ == '__main__': main()
