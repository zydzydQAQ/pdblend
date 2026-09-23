#!/usr/bin/env python3
"""Keep frozen GPU source; separate queue logs from Dynamo transaction logs."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from pdblend.experimentation.lease import GPULeaseQueue


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    original = json.loads(args.spec.read_text())
    result = deepcopy(original)
    queue = GPULeaseQueue(ROOT/'results/2026-09-22/three-model/queue.json')
    state = queue.snapshot()
    old_dynamo = [j for j in result['jobs'] if j['job_id'].startswith('native-dynamo-')]
    if len(old_dynamo) != 3:
        raise ValueError('three original Dynamo jobs required')
    replacements, changes = {}, []
    for job in old_dynamo:
        old = job['job_id']
        current = state['jobs'][old]
        if current['status'] not in ('failed', 'queued'):
            raise ValueError('refusing to replace running/completed Dynamo execution')
        if current['status'] == 'failed':
            leases = [x for x in state['leases'].values() if x['job_id'] == old]
            path = Path(max(leases, key=lambda x:x['claimed_at'])['attempt_dir'])
            if 'refusing to overwrite Dynamo transition artifacts' not in (path/'worker.log').read_text():
                raise ValueError('different failure needs a separate diagnosis')
        elif current['attempts']:
            raise ValueError('queued retry needs explicit inspection')
        new = old+'-outfix'
        job['job_id'] = new
        payload = job['payload']
        argv = payload['argv']
        if argv[argv.index('--out')+1] != '/output':
            raise ValueError('unexpected output directory')
        argv[argv.index('--out')+1] = '/output/dynamo'
        argv[argv.index('--name')+1] = new
        payload.update(container_name=new, required_receipts=['dynamo/completion.json'])
        replacements[old] = new
        changes.append((old, job))
    for job in result['jobs']:
        dependencies = job['payload'].get('depends_on', [])
        if not any(dep in replacements for dep in dependencies):
            continue
        old = job['job_id']
        current = state['jobs'][old]
        if current['status'] != 'queued' or current['attempts']:
            raise ValueError('dependent profile wave already started')
        new = old+'-outfix'
        job['job_id'] = new
        job['payload']['depends_on'] = [replacements.get(dep, dep) for dep in dependencies]
        job['payload']['container_name'] = new
        argv = job['payload']['argv']
        argv[argv.index('--name')+1] = new
        changes.append((old, job))
    result.update(parent_spec=str(args.spec.resolve()),
                  queue_output_fix='Dynamo artifacts live under attempt/dynamo; immutable source unchanged')
    output = args.spec.with_name(args.spec.stem+'-outfix.json')
    output.write_text(json.dumps(result, indent=2)+'\n')
    if not args.prepare_only:
        for old, _ in changes:
            if state['jobs'][old]['status'] == 'queued':
                queue.block(old, reason='Replaced before execution to isolate Dynamo artifacts from queue events.jsonl')
        for _, job in changes:
            queue.enqueue(**job)
    print(json.dumps(dict(spec=str(output), changes=[dict(old=a,new=b['job_id']) for a,b in changes],
                         prepared_only=args.prepare_only), indent=2))


if __name__ == '__main__':
    main()
