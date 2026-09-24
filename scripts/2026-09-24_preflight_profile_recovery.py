#!/usr/bin/env python3
"""Run the frozen profile collector's CPU-only preflight inside its real image."""
import argparse
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jobs', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    reports = []
    for job in json.loads(args.jobs.read_text()):
        payload = job['payload']
        original = payload['argv']
        boundary = original.index(payload['image_digest'])
        prefix = original[:boundary]
        safe = []
        i = 0
        while i < len(prefix):
            if prefix[i] in ('--gpus', '--cap-add'):
                i += 2
                continue
            if prefix[i] == '--name':
                safe.extend(['--name', 'cpu-preflight-' + prefix[i+1]])
                i += 2
                continue
            if prefix[i] == '-v' and 'pdblend-physical-clock-owners' in prefix[i+1]:
                i += 2
                continue
            safe.append('--network=none' if prefix[i] == '--network=host' else prefix[i])
            i += 1
        target = (args.out/job['job_id']).resolve()
        target.mkdir()
        values = dict(attempt_dir=str(target), lease_gpu_uuids='',
                      lease_local_indices=','.join(map(str, range(8))), lease_port='58000')
        command = [v.format(**values) for v in safe + original[boundary:]] + ['--preflight-only']
        completed = subprocess.run(command, text=True, capture_output=True, timeout=120)
        record = dict(job_id=job['job_id'], command=command, hardware_executed=False,
                      returncode=completed.returncode, stdout=completed.stdout, stderr=completed.stderr)
        (target/'invocation.json').write_text(json.dumps(record, indent=2)+'\n')
        reports.append(dict(job_id=job['job_id'], passed=completed.returncode == 0))
        print(json.dumps(reports[-1]), flush=True)
    summary = dict(passed=all(r['passed'] for r in reports), jobs=reports, hardware_executed=False)
    (args.out/'completion.json').write_text(json.dumps(summary, indent=2)+'\n')
    return 0 if summary['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
