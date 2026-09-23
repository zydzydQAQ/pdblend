#!/usr/bin/env python3
"""Read-only raw-evidence audit of one completed GPU holdout lease."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from pdblend.profile.calibration import digest, evaluate_holdout
from pdblend.profile.model import PerfModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue', type=Path,
                        default=ROOT/'results/2026-09-22/three-model/queue.json')
    parser.add_argument('--job', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    queue = json.loads(args.queue.read_text())
    job = queue['jobs'][args.job]
    leases = [lease for lease in queue['leases'].values()
              if lease['job_id'] == args.job and lease['status'] == 'succeeded']
    if job['status'] != 'succeeded' or len(leases) != 1:
        raise ValueError('exactly one successfully completed sampling lease required')
    artifact = Path(leases[0]['attempt_dir'])
    candidate_dir = Path(job['payload']['candidate_dir'])
    manifest = json.loads((candidate_dir/'manifest.json').read_text())
    candidate = candidate_dir/'candidate.json'
    if digest(candidate) != manifest['candidate_sha256']:
        raise ValueError('frozen candidate checksum mismatch')
    raw = json.loads((artifact/'raw.json').read_text())
    if raw.get('holdout_candidate_sha256') != manifest['candidate_sha256']:
        raise ValueError('raw holdout is bound to a different candidate')
    for key in ('system', 'model_id', 'tp', 'pp', 'model_hash', 'tokenizer_hash'):
        if raw.get(key) != manifest.get(key):
            raise ValueError('holdout identity mismatch: '+key)
    model = PerfModel.load(candidate)
    audit = evaluate_holdout(raw, model, artifact, expected_plan=manifest['plan'])
    power = []
    for row in raw.get('decode', []):
        predicted = model.decode_power_w(row['batch'], row['freq_mhz'])
        errors = [abs(predicted/rep['power_w']-1) for rep in row['repeats']]
        power.append(dict(frequency_mhz=row['freq_mhz'], batch=row['batch'],
            context_tokens=row['context_tokens'], predicted_w=predicted,
            measured_w=[rep['power_w'] for rep in row['repeats']],
            relative_errors=errors, mape=statistics.fmean(errors), maximum=max(errors),
            shared_window_failures=row.get('prediction_failures', [])))
    result = dict(job_id=args.job, artifact=str(artifact), calibration=audit,
        power_points=power, raw_sha256=digest(artifact/'raw.json'),
        candidate_sha256=digest(candidate), manifest_sha256=digest(candidate_dir/'manifest.json'),
        audit_source_sha256=digest(ROOT/'src/pdblend/profile/calibration.py'),
        formal_eligible=False, energy_comparable=False,
        scope='profile holdout only; independent system mechanisms and campaign remain separate')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(report=str(args.out), passed=audit['passed'],
        timing_max=audit['timing_max'], mixed_median=audit['mixed_median'],
        failures=audit['failures']), indent=2))


if __name__ == '__main__':
    main()
