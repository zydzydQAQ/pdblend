#!/usr/bin/env python3
"""Replay one frozen timing invocation in its real image without GPU access."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from pdblend.profile.collection.native_timing_plan import binding


def write_new(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True)
    args = parser.parse_args()
    prepared = args.prepared.resolve()
    jobs = json.loads((prepared/'jobs.json').read_text())
    if len(jobs) != 1:
        raise ValueError('one immutable timing invocation required')
    job = jobs[0]
    source = job['payload']['argv']
    if source[:2] != ['docker', 'run']:
        raise ValueError('expected direct frozen Docker invocation')
    out = prepared/'cpu-preflight'
    out.mkdir(exist_ok=False)
    uuids = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True).splitlines()
    if len(set(uuids)) != 8:
        raise ValueError('eight physical GPU identities required, without CUDA execution')
    replacements = {'{attempt_dir}': str(out), '{lease_gpu_uuids}': ','.join(uuids),
                    '{lease_gpus}': '0,1,2,3,4,5,6,7',
                    '{lease_local_indices}': '0,1,2,3,4,5,6,7', '{lease_port}': '19500'}
    argv = []
    index = 0
    source_image_index = source.index(job['payload']['image_digest'])
    while index < len(source):
        value = source[index]
        if index < source_image_index and value in ('--gpus', '--cap-add'):
            index += 2
            continue
        if value == '--name':
            argv.extend([value, job['job_id']+'-cpu'])
            index += 2
            continue
        for old, new in replacements.items():
            value = value.replace(old, new)
        if value.startswith('PDBLEND_CONCURRENCY_ENVIRONMENT='):
            value = 'PDBLEND_CONCURRENCY_ENVIRONMENT='+str(out/'concurrency-environment.json')
        argv.append(value)
        index += 1
    image_index = argv.index(job['payload']['image_digest'])
    argv[image_index:image_index] = ['-e', 'NVIDIA_VISIBLE_DEVICES=void',
                                    '-e', 'OMP_NUM_THREADS=2', '-e', 'OPENBLAS_NUM_THREADS=1']
    argv.append('--preflight-only')
    write_new(prepared/'cpu-preflight-command.json', dict(argv=argv, hardware_executed=False))
    started = time.time()
    with (prepared/'cpu-preflight.stdout').open('x') as stdout, (prepared/'cpu-preflight.stderr').open('x') as stderr:
        process = subprocess.run(argv, cwd=ROOT, stdout=stdout, stderr=stderr, timeout=180)
    report = dict(status='cpu_preflight_passed' if process.returncode == 0 else 'failed',
                  returncode=process.returncode, hardware_executed=False,
                  started_s=started, finished_s=time.time(),
                  implementation=binding(__file__), jobs=binding(prepared/'jobs.json'),
                  command=binding(prepared/'cpu-preflight-command.json'),
                  stdout=binding(prepared/'cpu-preflight.stdout'), stderr=binding(prepared/'cpu-preflight.stderr'))
    if (out/'native-timing/preflight.json').exists():
        report['receipt'] = binding(out/'native-timing/preflight.json')
    write_new(prepared/'cpu-preflight-review.json', report)
    print(json.dumps({key: report[key] for key in ('status', 'returncode')}))
    return process.returncode


if __name__ == '__main__':
    raise SystemExit(main())
