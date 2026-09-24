#!/usr/bin/env python3
"""Checkpoint a quiesced campaign, interrupt once, verify native cleanup."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue', type=Path, required=True)
    parser.add_argument('--job', required=True)
    parser.add_argument('--session', type=Path, required=True)
    parser.add_argument('--record', type=Path, required=True)
    args = parser.parse_args()
    record = json.loads(args.record.read_text())
    def save(**values):
        record.update(values, updated_at_s=time.time())
        temp = args.record.with_suffix('.tmp')
        temp.write_text(json.dumps(record, indent=2)+'\n')
        os.replace(temp, args.record)
        print(json.dumps({k: v for k, v in values.items() if k in ('phase', 'signal_sent', 'cleanup_verified')}), flush=True)
    if not Path(record['worker_stop_file']).exists():
        raise RuntimeError('worker stop file must be installed before handoff')
    for pid in record['stopped_schedulers']:
        if Path('/proc', str(pid), 'cmdline').exists():
            raise RuntimeError('outer scheduler is still present')
    windows = args.session/'windows'
    unfinished = [p for p in windows.iterdir() if p.is_dir() and not (p/'receipt.json').exists()]
    if len(unfinished) > 1:
        raise RuntimeError('unexpected multiple unfinished windows')
    target = unfinished[0]/'receipt.json' if unfinished else None
    save(phase='waiting_current_window', active_job=args.job, session=str(args.session.resolve()),
         target_receipt=str(target) if target else None)
    deadline = time.monotonic()+1200
    while target and not target.exists() and not (args.session/'completion.json').exists():
        if time.monotonic() > deadline:
            raise TimeoutError('current window did not complete; no signal sent')
        time.sleep(1)
    receipts = []
    for path in sorted(windows.glob('*/receipt.json')):
        value = json.loads(path.read_text())
        if value.get('cleanup_passed') is True:
            receipts.append(dict(path=str(path.resolve()), sha256=sha(path)))
    latest = args.session/'extensions/latest.json'
    checkpoint = dict(path=str(latest.resolve()), sha256=sha(latest)) if latest.exists() else None
    save(phase='checkpoint_committed', completed_receipts=receipts, extension_checkpoint=checkpoint)
    queue = json.loads(args.queue.read_text())
    job = queue['jobs'][args.job]
    container = job['payload']['container_name']
    state = subprocess.run(['docker', 'inspect', '--format', '{{.State.Running}}', container],
                           capture_output=True, text=True)
    if state.returncode == 0 and state.stdout.strip() == 'true':
        if record.get('signal_sent'):
            raise RuntimeError('first SIGINT already recorded; never send another')
        save(phase='native_cleanup', signal_sent='SIGINT_once', signal_at_s=time.time(), container=container)
        subprocess.run(['docker', 'kill', '--signal=SIGINT', container], check=True, capture_output=True)
    deadline = time.monotonic()+600
    while True:
        queue = json.loads(args.queue.read_text())
        job = queue['jobs'][args.job]
        completion = args.session/'completion.json'
        if completion.exists() and job['status'] != 'running':
            break
        if time.monotonic() > deadline:
            save(phase='cleanup_requires_inspection')
            raise TimeoutError('cleanup not acknowledged; no second signal sent')
        time.sleep(1)
    final = json.loads(completion.read_text())
    cleanup = final.get('cleanup', {})
    if cleanup.get('passed') is not True or cleanup.get('process_cleanup_verified') is not True or final.get('cleanup_errors'):
        save(phase='cleanup_failed', session_complete=final.get('complete'), queue_status=job['status'])
        raise RuntimeError('native cleanup did not pass')
    for ref in receipts:
        if sha(ref['path']) != ref['sha256']:
            raise RuntimeError('committed receipt changed during cleanup')
    processes = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid', '--format=csv,noheader'],
                               capture_output=True, text=True, check=True).stdout.strip()
    if processes:
        raise RuntimeError('GPU compute processes remain after cleanup')
    save(phase='handed_off', cleanup_verified=True, completion=dict(path=str(completion.resolve()), sha256=sha(completion)),
         session_complete=final.get('complete'), queue_status=job['status'], compute_processes=[],
         unfinished_windows=[p.name for p in windows.iterdir() if p.is_dir() and not (p/'receipt.json').exists()])


if __name__ == '__main__':
    main()
