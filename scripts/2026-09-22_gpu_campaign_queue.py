#!/usr/bin/env python3
"""Persistent queue CLI. Work payloads use argv arrays and explicit receipts."""
import argparse
import dataclasses
import concurrent.futures
import json
import time
from pathlib import Path
from pdblend.experimentation.lease import GPULeaseQueue, gpu_snapshot
from pdblend.experimentation.worker import run_one


def _worker_loop(db: Path, *, once: bool, idle_exit: bool, stop_file: Path | None = None) -> None:
    """Keep one isolated queue handle replenished until its queue is drained."""
    while True:
        if stop_file is not None and stop_file.exists():
            return
        worked = run_one(GPULeaseQueue(db))
        if once:
            return
        pending = any(job.status in {'queued', 'running'}
                      for job in GPULeaseQueue(db).list_jobs())
        if idle_exit and not pending:
            return
        if not worked:
            time.sleep(5)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--db', type=Path, default=Path('results/2026-09-22/three-model/queue.json'))
    sub = p.add_subparsers(dest='cmd', required=True)
    en = sub.add_parser('enqueue'); en.add_argument('spec', type=Path)
    work = sub.add_parser('worker'); work.add_argument('--once', action='store_true'); work.add_argument('--idle-exit', action='store_true'); work.add_argument('--workers', type=int, default=1); work.add_argument('--stop-file', type=Path)
    sub.add_parser('status'); sub.add_parser('gpus')
    a = p.parse_args(); q = GPULeaseQueue(a.db)
    if a.cmd == 'enqueue':
        for job in json.loads(a.spec.read_text()):
            print(json.dumps(dataclasses.asdict(q.enqueue(**job))))
    elif a.cmd == 'worker':
        if a.workers < 1:
            p.error('--workers must be positive')
        # Each thread owns a queue handle and replenishes its own slot as soon
        # as a job ends; pool.map would wait for the slowest sibling first.
        with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
            futures = [pool.submit(_worker_loop, a.db, once=a.once,
                                   idle_exit=a.idle_exit, stop_file=a.stop_file)
                       for _ in range(a.workers)]
            for future in futures:
                future.result()
    elif a.cmd == 'gpus': print(json.dumps(gpu_snapshot(), indent=2))
    else: print(json.dumps(q.snapshot(), indent=2))


if __name__ == '__main__': main()
