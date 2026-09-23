#!/usr/bin/env python3
"""Enqueue reviewed 32B request replay after the already resident profile wave."""
import hashlib
import json
from pathlib import Path
from pdblend.experimentation.lease import GPULeaseQueue


def main():
    root = Path(__file__).resolve().parents[1]
    package = root / 'results/2026-09-23/dynamo-32b-reroute-retry-v1'
    review_path = root / 'results/2026-09-23/dynamo-32b-reroute-retry-v1-cpu-review/review.json'
    review = json.loads(review_path.read_text())
    preflight = json.loads((review_path.parent / 'preflight/dynamo/preflight.json').read_text())
    if not review.get('ready') or not preflight.get('ready'):
        raise RuntimeError('pinned-image replay review is not ready')
    for path, expected in review['inputs_sha256'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise RuntimeError('reviewed retry input changed: ' + path)
    queue = GPULeaseQueue(root / 'results/2026-09-22/three-model/queue.json')
    spec = json.loads((package / 'jobs.json').read_text())[0]
    if (spec['job_id'] != review['job_id']
            or spec['payload']['source_sha256'] != review['source_sha256']
            or spec['payload']['image_digest'] != review['image_digest']):
        raise RuntimeError('replay job differs from reviewed identity')
    replacements = json.loads((root / 'results/2026-09-23/resume-independent-acceptance-v1/jobs.json').read_text())
    dependencies = [s['job_id'] for s in replacements if s['job_id'].startswith('incremental-')]
    if len(dependencies) != 4:
        raise RuntimeError('expected complete four-member incremental wave')
    payload = spec['payload']
    payload.update(after_terminal=dependencies, prepare_only=False, execution_ready=True,
                   cpu_review_path=str(review_path),
                   cpu_review_sha256=hashlib.sha256(review_path.read_bytes()).hexdigest())
    spec['priority'] = 400
    out = root / 'results/2026-09-23/dynamo-reroute-enqueued-v1'
    out.mkdir(exist_ok=True)
    encoded = json.dumps([spec], indent=2) + '\n'
    path = out / 'jobs.json'
    if path.exists() and path.read_text() != encoded:
        raise RuntimeError('refusing to overwrite different replay scheduling spec')
    path.write_text(encoded)
    job = queue.enqueue(**spec)
    (out / 'enqueue-receipt.json').write_text(json.dumps(dict(job_id=job.job_id,
        status=job.status, after_terminal=dependencies,
        jobs_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), formal_eligible=False), indent=2) + '\n')
    print(json.dumps(dict(job_id=job.job_id, status=job.status)))


if __name__ == '__main__':
    main()
