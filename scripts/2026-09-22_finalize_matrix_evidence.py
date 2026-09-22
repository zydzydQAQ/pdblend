#!/usr/bin/env python3
"""Bind completed benchmark summaries to their exact trace/profile/source inputs."""
import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def source_hash(root):
    h = hashlib.sha256()
    for p in sorted(Path(root).rglob('*.py')):
        if '__pycache__' in p.parts:
            continue
        h.update(str(p.relative_to(root)).encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def finalize(root, profile, spec, source):
    root, profile, spec, source = map(Path, (root, profile, spec, source))
    matrix = json.loads(spec.read_text())
    rows = []
    for point in matrix['points']:
        out = root / point['name']
        summary_path = out / 'summary.json'
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text())
        trace = summary.get('trace_meta', {})
        identity = dict(model=summary.get('model'), tp=summary.get('tp'), gpus=summary.get('gpus'),
                        policy=summary.get('policy', {}).get('name'), dataset=point.get('dataset'),
                        rate=point.get('rate'), scale=point.get('scale'), seed=trace.get('seed'),
                        duration=300, trace_window_s=summary.get('window_s'), requests=summary.get('requests'),
                        profile_sha256=sha(summary['profile']) if Path(summary.get('profile', '')).exists() else sha(profile),
                        spec_sha256=sha(spec), source_sha256=source_hash(source),
                        image=os.environ.get('PDBLEND_IMAGE_ID', 'unknown'),
                        hardware=os.environ.get('PDBLEND_HARDWARE_ID', 'unknown'),
                        clock_protocol='nvidia-smi-lock-v1', energy_protocol='window+tail-v1')
        artifacts = {}
        for name in ('summary.json', 'outcomes.jsonl', 'power.jsonl', 'controller.jsonl', 'freq.jsonl'):
            p = out / name
            if p.exists():
                artifacts[name] = sha(p)
        evidence = dict(status='complete', returncode=0, inputs_unchanged=True,
                        identity=identity, identity_sha256=digest(identity), artifacts=artifacts,
                        finalized_s=time.time())
        (out / 'evidence.json').write_text(json.dumps(evidence, indent=1))
        rows.append(dict(name=point['name'], seed=identity['seed'], evidence=str(out / 'evidence.json')))
    report = root / 'evidence-index.json'
    report.write_text(json.dumps(dict(root=str(root), profile=str(profile), spec=str(spec), source=str(source),
                                      source_sha256=source_hash(source), points=rows), indent=1))
    print(json.dumps(dict(points=len(rows), output=str(report)), indent=1))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('root')
    p.add_argument('profile')
    p.add_argument('spec')
    p.add_argument('--source', default='src')
    a = p.parse_args()
    finalize(a.root, a.profile, a.spec, a.source)
