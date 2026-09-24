#!/usr/bin/env python3
"""Run only this matrix's leases through the shared native queue worker."""
import argparse
from pathlib import Path
import time

from pdblend.experimentation.lease import GPULeaseQueue
from pdblend.experimentation.worker import run_one


class MatrixQueue(GPULeaseQueue):
    def __init__(self, path, *, run_id):
        super().__init__(path)
        self.run_id = run_id

    def _deps_ready(self, state, job):
        # Keep the queue and GPU lock global; restrict only job selection.
        # Other tasks cannot be accidentally resumed by this matrix's worker.
        return (job.get('payload', {}).get('run_id') == self.run_id
                and super()._deps_ready(state, job))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db', type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--stop-file', type=Path, required=True)
    args = parser.parse_args()
    queue = MatrixQueue(args.db, run_id=args.run_id)
    while not args.stop_file.exists():
        if not run_one(queue):
            time.sleep(5)


if __name__ == '__main__':
    main()
