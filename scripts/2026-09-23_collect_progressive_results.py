#!/usr/bin/env python3
"""Collect only this frozen campaign; publish development selections after independent audits."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import time
from pathlib import Path

from pdblend.profile.calibration.optimization_profiles import merge_components
from pdblend.profile.query.versions import load_profile


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2, sort_keys=True)+'\n')
    temp.replace(path)


def collect(queue_path, prepared, out):
    review = json.loads((prepared/'review.json').read_text())
    if sha(prepared/'jobs.json') != review['jobs_sha256']:
        raise ValueError('frozen campaign jobs changed')
    jobs = json.loads((prepared/'jobs.json').read_text())
    state = json.loads(queue_path.read_text())
    rows, roots = [], {}
    for spec in jobs:
        key = spec['job_id']
        job = state['jobs'].get(key, {})
        row = dict(job_id=key,status=job.get('status','not_enqueued'))
        attempts = [v for v in state['leases'].values() if v['job_id']==key]
        if attempts:
            latest = max(attempts,key=lambda v:v['attempt'])
            directory = Path(latest['attempt_dir'])
            receipt = directory/'completion.json'
            row['attempt_dir'] = str(directory)
            if receipt.exists():
                data = json.loads(receipt.read_text())
                row.update(receipt=str(receipt),sha256=sha(receipt),complete=data.get('complete',False),
                    components_passed=data.get('components_passed'),functional_passed=data.get('functional_passed'),
                    error=data.get('error'),gpu_phase_metrics=data.get('gpu_phase_metrics'),
                    total_job_gpu_s=data.get('total_job_gpu_s',data.get('occupied_gpu_s')))
                if job.get('status')=='succeeded' and data.get('components_passed'):
                    roots[spec['payload']['cohort_member']] = directory/'components'
        rows.append(row)
    terminal = all(r['status'] in ('succeeded','failed','cancelled') for r in rows)
    result = dict(schema=1,updated_s=time.time(),queue_terminal=terminal,jobs=rows,
        source_sha256=review['source_sha256'],formal_eligible=False,energy_comparable=False,
        scope='native functional receipts and development-only bounded power components',selections=[])
    if terminal:
        all_functional = all(r.get('functional_passed') is True for r in rows if 'online-safety' in r['job_id'])
        result['native_functional_passed'] = all_functional
        for model in ('7b','14b','32b'):
            panels = [roots.get(f'{model}-{frequency}') for frequency in (1500,2520)]
            if not all(panels):
                continue
            union = out/f'{model}-power-union.json'
            if not union.exists():
                merge_components(panels,union)
            manifest = json.loads((prepared/'packages'/f'{model}-1500'/'manifest.json').read_text())
            selection = out/f'{model}-profile-selection.json'
            if not selection.exists():
                write(selection,dict(kind='pdblend_profile_selection_v1',
                    profile_path=manifest['inputs']['base_candidate']['path'],optimization_component=str(union)))
            loaded = load_profile(selection,system='pdblend',model_id=manifest['model_id'],tp=manifest['tp'])
            receipt = out/f'{model}-consumer.json'
            write(receipt,loaded.manifest_fields())
            result['selections'].append(dict(model_id=manifest['model_id'],selection=str(selection),
                selection_sha256=sha(selection),consumer_receipt=str(receipt),consumer_sha256=sha(receipt),
                serving_validation_passed=all_functional))
        result['all_measurements_passed'] = all(r.get('complete') and r['status']=='succeeded' for r in rows)
        result['all_components_passed'] = len(result['selections'])==3
    write(out/'status.json',result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue',type=Path,required=True)
    parser.add_argument('--prepared',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--watch',action='store_true')
    args = parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    with (args.out/'collector.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            result = collect(args.queue,args.prepared,args.out)
            if result['queue_terminal'] or not args.watch:
                print(json.dumps(result,indent=2))
                return
            time.sleep(15.)


if __name__=='__main__':
    main()
