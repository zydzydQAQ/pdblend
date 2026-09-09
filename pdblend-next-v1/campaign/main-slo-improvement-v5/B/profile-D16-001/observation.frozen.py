"""PDB-only TP2 batch16 short512 observations; never edits serving profiles."""
import argparse
import asyncio
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
HELPER = Path('/root/workspace/pdblend-next-v1/campaign/budget-profiling-v2-candidate')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def check():
    m = read(ROOT / 'manifest.json')
    for name, digest in m['files'].items():
        require(sha(ROOT / name) == digest, 'package changed: ' + name)
    for name, digest in m['helper_files'].items():
        require(sha(HELPER / name) == digest, 'frozen profiler changed: ' + name)


def protect():
    files = read(ROOT / 'protected-baselines.json')['files']
    require(len(files) == 91, 'B baseline preservation set differs')
    for name, digest in files.items():
        require(sha(name) == digest, 'historical baseline changed: ' + name)
    return files


def validate_identity(actual, expected):
    for key, value in expected.items():
        require(actual.get(key) == value, 'live B identity differs: ' + key)


def validate_tp2_observation(raw, events):
    ranks = raw['drain']['transfers']
    require(len(ranks) == 2 and all(r.get('buffered_gpu_bytes') == 0 for r in ranks),
            'both actual TP2 rank drains required')
    for state in (raw['runtime_before'], raw['runtime_after_requests'], raw['runtime_drained']):
        require(len(state['scheduler_io']) == 1, 'actual TP2 scheduler owner cache observation required')
        require(state['acknowledged_generations'] == [state['generation']],
                'actual TP2 scheduler owner ACK required')
    ids = {r['request_id'] for r in raw['requests']}
    batch = raw['spec']['batch_size']
    steps = [e for e in events if e.get('prefill') == 0 and e.get('decode') == batch
             and set(e.get('request_ids', [])) == ids]
    require(len(steps) >= 64, 'requested concurrent decode batch was not sustained for64 steps')
    return dict(actual_batch=batch, full_batch_decode_steps=len(steps),
                earliest_step_s=min(e['started_s'] for e in steps),
                latest_step_s=max(e['finished_s'] for e in steps), tp2_rank_count=2)


def arguments():
    return SimpleNamespace(
        engine_config=Path('/root/workspace/pdblend-next-v1/campaign/B32B-engine-v3-candidate-v2/engine-0.json'),
        runtime_dir=Path('/root/workspace/pdblend-next-v1/campaign/B32B-engine-v3-candidate-v2/runtime'), out=ROOT/'results',
        port=33500, container='pdb-v2-nextv3b0', target_gpus=[0, 1],
        input_patterns=[[512]], output_pattern=[256], batches=[16],
        frequencies=[1500, 2520], budgets=[8192], repeats=3, seed=0,
        arrival_offsets=[0.], require_preexisting_decode=False, arrival_lateness_limit_s=None)


if __name__ == '__main__':
    raise SystemExit('CPU spec module only; use the explicit outer run.py entrypoint')
