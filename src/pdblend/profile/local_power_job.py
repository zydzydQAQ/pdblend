"""Queue adapter for the bounded 14B power and mixed repair panel.

Sampling completion permits later jobs to run. Calibration qualification is
reported separately and the original failed timing receipt remains immutable.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .local_power import Adapter, SCOPE
from .power_calibration import digest, run
from .wave import atomic_json


def queue_receipt(result, out):
    out = Path(out)
    completion = out / 'completion.json'
    if not completion.is_file() or json.loads(completion.read_text()) != result:
        raise ValueError('local completion must exist unchanged')
    complete = result.get('complete') is True and result.get('status') == 'completed'
    receipt = dict(schema=1, status='passed' if complete else 'failed', complete=complete,
        queue_receipt_semantics='measurement_completion_only; calibration_is_reported_separately',
        underlying_completion_sha256=digest(completion), original_completion_unchanged=True,
        formal_eligible=False, energy_comparable=False, full_profile_qualified=False,
        error=result.get('error'))
    if not complete:
        return receipt
    composite = out / 'composite-audit.json'
    if digest(composite) != result.get('composite_audit_sha256'):
        raise ValueError('local composite checksum differs')
    audit = json.loads(composite.read_text())
    if (result.get('scope') != SCOPE or audit.get('validation_scope') != SCOPE
            or result.get('full_profile_qualified') is not False
            or audit.get('full_profile_qualified') is not False
            or result.get('concurrency_qualified') is not True
            or audit.get('concurrency_qualified') is not True):
        raise ValueError('local panel scope or concurrency qualification differs')
    power, original = audit['power']['passed'], audit['timing']['passed']
    repaired = audit.get('repaired_timing', {}).get('passed')
    if (any(type(value) is not bool for value in (power, original, repaired))
            or audit['repaired_timing'].get('complete') is not True
            or audit['repaired_timing'].get('original_failed_rows_preserved') is not True
            or result.get('power_passed') is not power
            or result.get('reused_timing_passed') is not original
            or result.get('repaired_timing_passed') is not repaired
            or result.get('calibration_components_passed') is not (power and repaired)
            or audit.get('calibration_components_passed') is not (power and repaired)):
        raise ValueError('local qualification fields do not match measured component receipts')
    receipt.update(validation_scope=SCOPE, composite_audit_sha256=digest(composite),
        power_passed=power, original_timing_passed=original,
        repaired_timing_passed=repaired, calibration_components_passed=power and repaired,
        qualification_status='passed' if power and repaired else 'failed')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--gpus', type=int, nargs='+', required=True)
    parser.add_argument('--base-port', type=int, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = run(package=args.package, model_path=args.model, gpus=args.gpus,
        base_port=args.base_port, out=args.out, panel_adapter=Adapter('coordinated-wave'))
    receipt = queue_receipt(result, args.out)
    atomic_json(args.out / 'queue-completion.json', receipt)
    print(json.dumps(receipt, indent=2), flush=True)
    raise SystemExit(0 if receipt['complete'] else 1)


if __name__ == '__main__':
    main()
