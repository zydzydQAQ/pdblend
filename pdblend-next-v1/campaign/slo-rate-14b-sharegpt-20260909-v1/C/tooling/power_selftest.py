"""Fresh new-A measurement-only qualification using the original isolated sampler."""
import argparse
import asyncio
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = Path('/root/workspace/pdblend-next-v1/campaign/parallel-rate-20260908-v1')
HOST = Path('/root/workspace/pdblend-next-v1/campaign/slo-rate-14b-sharegpt-20260909-v1/env/runtime')
METER = Path('/root/workspace/pdblend-next-v1/campaign/slo-rate-14b-sharegpt-20260909-v1/env/meter')
IDENTITY = HERE / 'node-identity.json'
ADAPTER = Path('/root/workspace/pdblend-next-v1/campaign/slo-rate-14b-sharegpt-20260909-v1/env/isolated-power/manifest.json')
HOOKS = METER / 'sampler_hooks.py'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ref(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path))


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def validate_identity(identity, hostname, rows):
    assert identity['node'] == 'C'
    assert hostname == identity['actual_hostname'] == 'iZwz9gfq11hx1sbob59yrgZ', 'wrong physical node'
    expected = [(r['index'], r['uuid']) for r in identity['GPUs']]
    assert len(rows) == 8 and [(r['index'], r['uuid']) for r in rows] == expected, 'wrong actual GPU identity'


def source_check():
    checked = {}
    for path, relative in [(HOST / 'manifest.json', HOST), (Path('/root/workspace/pdblend-next-v1/campaign/slo-rate-14b-sharegpt-20260909-v1/env/native-runtime/manifest.slo14.json'), Path('/root/workspace/pdblend-next-v1/campaign/slo-rate-14b-sharegpt-20260909-v1/env/native-runtime')), (METER / 'manifest.json', None), (ADAPTER, None)]:
        manifest = read(path)
        for name, digest in manifest['files'].items():
            file = relative / name if relative else Path(name)
            assert sha(file) == digest, 'changed qualification source: ' + str(file)
            checked[str(file)] = digest
        checked[str(path)] = sha(path)
    for path in [IDENTITY, HOOKS, Path(__file__), METER / 'meter_evidence.py']:
        checked[str(path)] = sha(path)
    return checked


def actual_identity():
    raw = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader,nounits'],
                         check=True, capture_output=True, text=True).stdout
    rows = [dict(index=int(line.split(',')[0]), uuid=line.split(',')[1].strip())
            for line in raw.strip().splitlines()]
    hostname = socket.gethostname()
    validate_identity(read(IDENTITY), hostname, rows)
    return dict(hostname=hostname, GPUs=rows, captured_s=time.time())



NODE='C'
EXPECTED_HOSTNAME='iZwz9gfq11hx1sbob59yrgZ'
