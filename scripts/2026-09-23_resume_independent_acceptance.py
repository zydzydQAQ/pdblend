#!/usr/bin/env python3
"""Replace unexecuted staging dependencies, retaining immutable measurements."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

from pdblend.experimentation.lease import GPULeaseQueue


def replacements(groups):
    mapping, output = {}, []
    for jobs in groups:
        for original in jobs:
            spec = copy.deepcopy(original)
            old_id = spec['job_id']
            payload = spec['payload']
            predecessors = [mapping.get(dep, dep) for dep in payload.pop('depends_on', [])]
            payload['depends_on'] = []
            payload['after_terminal'] = predecessors
            payload['supersedes_job_id'] = old_id
            payload['scheduling_revision'] = 'terminal-after-cleanup-v1'
            if 'exact_inputs_sha256' in payload:
                payload['exact_inputs_sha256']['dependencies'] = []
                payload['exact_inputs_sha256']['after_terminal'] = predecessors
            digest = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:12]
            new_id = old_id + '-terminal-' + digest
            spec['job_id'] = new_id
            payload['container_name'] = new_id
            payload['argv'] = [new_id if value == old_id else value for value in payload['argv']]
            mapping[old_id] = new_id
            output.append(spec)
    return output, mapping


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--db', type=Path, default=Path('results/2026-09-22/three-model/queue.json'))
    p.add_argument('--out', type=Path, default=Path('results/2026-09-23/resume-independent-acceptance-v1'))
    p.add_argument('--apply', action='store_true')
    a = p.parse_args()
    paths = [Path('results/2026-09-23/dist-eco-resident-v1/jobs.json'),
             Path('results/2026-09-23/incremental-profile-wave-v2/jobs.json')]
    specs, mapping = replacements([json.loads(path.read_text()) for path in paths])
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / 'jobs.json').write_text(json.dumps(specs, indent=2) + '\n')
    review = {'schema': 1, 'replacement_mapping': mapping, 'source_specs': [str(p.resolve()) for p in paths],
              'source_checksums': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
              'preserved_measurement_sources': True, 'formal_eligible': False, 'applied': False}
    if a.apply:
        queue = GPULeaseQueue(a.db)
        if queue.active_leases():
            raise RuntimeError('drain worker before replacing these unexecuted jobs')
        state = queue.snapshot()
        for old in mapping:
            job = state['jobs'][old]
            if not (job['status'] == 'queued' and job['attempts'] == 0):
                if not (job['status'] == 'cancelled' and job.get('superseded_by') == mapping[old]):
                    raise RuntimeError(f'job already executed or changed: {old}')
        backup = a.out / 'queue-before.json'
        if not backup.exists():
            backup.write_text(json.dumps(state, indent=2) + '\n')
        queue.enqueue_replacements(specs)
        review['applied'] = True
    (a.out / 'receipt.json').write_text(json.dumps(review, indent=2) + '\n')
    print(json.dumps({'jobs': len(specs), 'applied': review['applied'], 'out': str(a.out)}))


if __name__ == '__main__':
    main()
