#!/usr/bin/env python3
"""Preserve a completed timing stage before stopping optional layout work."""
import argparse
import json
import subprocess
import time
from pathlib import Path

from pdblend.profile.collection.native_timing_plan import binding
from pdblend.profile.collection.native_timing_stage import capture_operator_stop_request


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--attempt', required=True, type=Path)
    parser.add_argument('--queue', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((args.attempt/'manifest.json').read_text())
    payload = manifest['payload']
    name = payload['container_name']
    assert payload['system'] == 'pdblend' and payload['gpu_count'] == 8
    assert payload['exclusive'] is True and '--layout-energy-plan' in payload['argv']
    stage = args.attempt/'native-timing/timing-stage.json'
    deadline = time.monotonic() + 3600
    while time.monotonic() < deadline:
        queue = json.loads(args.queue.read_text())
        job = queue['jobs'][manifest['job_id']]
        if job['status'] != 'running':
            write(args.out/'result.json', dict(status='already_terminal', job_status=job['status'],
                signal_sent=False, observed_s=time.time()))
            print('Original task ended; no signal sent.', flush=True)
            return
        if stage.exists():
            # The capture API replays the immutable stage and verifies the
            # active original lease before authorizing this precise boundary.
            try:
                request = capture_operator_stop_request(args.attempt, args.queue, args.out/'operator-stop.json')
            except (ValueError, FileNotFoundError) as exc:
                if (args.attempt/'native-timing/completion.json').exists():
                    write(args.out/'result.json', dict(status='natural_completion_race', signal_sent=False,
                        error=str(exc), observed_s=time.time()))
                    return
                raise
            inspected = json.loads(subprocess.check_output(['docker', 'inspect', name], text=True))[0]
            assert inspected['Name'] == '/'+name and inspected['State']['Running'] is True
            assert inspected['Image'] == payload['image_digest']
            assert 'pdblend.profile.collection.native_timing_collect' in inspected['Config']['Cmd']
            expected_source = payload['source_sha256']
            assert 'PDBLEND_SOURCE_SHA256='+expected_source in inspected['Config']['Env']
            current = json.loads(args.queue.read_text())['jobs'][manifest['job_id']]
            assert current['status'] == 'running' and current['lease_id'] == manifest['lease_id']
            sent_s = time.time()
            command = ['docker', 'kill', '--signal=SIGINT', name]
            result = subprocess.run(command, text=True, capture_output=True, timeout=30)
            write(args.out/'result.json', dict(status='signal_sent' if result.returncode == 0 else 'signal_failed',
                signal_sent=result.returncode == 0, signal='SIGINT', request=request,
                stage=binding(stage), requested_s=sent_s, completed_s=time.time(),
                command=command, returncode=result.returncode, stdout=result.stdout, stderr=result.stderr,
                whole_job_success_claimed=False))
            print(json.dumps(dict(status='signal_sent' if result.returncode == 0 else 'signal_failed',
                request=request)), flush=True)
            if result.returncode:
                raise RuntimeError('SIGINT delivery failed; no force-kill fallback is permitted')
            return
        time.sleep(2)
    raise TimeoutError('No completed timing boundary; no signal sent')


if __name__ == '__main__':
    main()
