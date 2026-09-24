"""Compare observed capacity brackets from immutable calibration/tuning ledgers.

This read-only report never executes jobs or grants profile/formal eligibility.
It needs all five converged brackets, one replayable workload family and the
same eight physical GPUs before describing an observed PDblend boundary lead.
"""
from __future__ import annotations

import argparse
import json

from .capacity_workloads import SYSTEMS
from .slo_capacity_receipts import read_capacity_ledger


BASELINES = tuple(system for system in SYSTEMS if system != 'pdblend')


def compare_capacity_ledgers(ledgers):
    """Read ``{system: ledger_path}``; incomplete evidence never implies a lead."""
    if not isinstance(ledgers, dict) or not ledgers or set(ledgers) - set(SYSTEMS):
        raise ValueError('capacity comparison needs named supported system ledgers')
    reports = {system: read_capacity_ledger(path) for system, path in ledgers.items()}
    reasons, reference, policy, raw_evidence = [], None, None, set()
    keys = ('family_sha256', 'family_id', 'gpu_uuids', 'model_hash', 'tokenizer_hash',
            'measurement_protocol_version', 'duration_s', 'output_workload')
    for system in SYSTEMS:
        if system not in reports:
            reasons.append('missing_system:' + system)
            continue
        report = reports[system]
        if report['series']['system'] != system:
            raise ValueError('capacity ledger system differs from comparison assignment')
        state = report['state']
        current_policy = {key: state[key] for key in
            ('slo', 'required_repeats', 'min_requests_per_trial', 'relative_tolerance')}
        if policy is None:
            policy = current_policy
        elif current_policy != policy:
            raise ValueError('capacity comparison SLO or bracket protocol differs')
        if not report['evidence']:
            reasons.append('missing_workload_evidence:' + system)
        else:
            identity = report['evidence'][0]['identity']
            shared = {key: identity[key] for key in keys}
            shared.update(model_id=report['series']['model_id'], dataset=report['series']['dataset'],
                          selection_split=state['selection_split'])
            if reference is None:
                reference = shared
            elif shared != reference:
                raise ValueError('capacity comparison needs the same workload family, model, dataset and eight-GPU hardware')
            overlap = raw_evidence.intersection(state['evidence_ids'])
            if overlap:
                raise ValueError('capacity comparison reuses the same raw measurement across systems')
            raw_evidence.update(state['evidence_ids'])
        if not state['converged']:
            reasons.append('unconverged:' + system + ':' + state['status'])
        if state['failed_upper'] is None:
            reasons.append('missing_failed_upper:' + system)
    pd = reports.get('pdblend', {}).get('state', {})
    checks = {}
    for system in BASELINES:
        other = reports.get(system, {}).get('state', {})
        lower, upper = pd.get('passed_lower'), other.get('failed_upper')
        checks[system] = dict(pdblend_passed_lower=lower, baseline_failed_upper=upper,
            strictly_separated=bool(lower is not None and upper is not None and lower > upper))
        if lower is not None and upper is not None and lower <= upper:
            reasons.append('not_strictly_separated:' + system)
    lead = not reasons and all(row['strictly_separated'] for row in checks.values())
    return dict(schema='pdblend-observed-capacity-comparison/v1',
        observed_boundary_lead=lead,
        status='observed_pdblend_boundary_lead' if lead else 'boundary_lead_not_established',
        reason_codes=reasons, shared_identity=reference, protocol=policy,
        pairwise_bounds=checks,
        systems={system: dict(ledger=report['ledger'], series=report['series'],
            state=report['state'], frozen_evaluation_grid=report['frozen_evaluation_grid'])
            for system, report in reports.items()},
        comparison_scope='observed repeated calibration/tuning SLO brackets only',
        exact_capacity_established=False, formal_eligible=False,
        profile_qualification_promoted=False, hardware_executed=False, jobs_enqueued=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ledger', action='append', required=True, metavar='SYSTEM=PATH',
                        help='one immutable capacity ledger per comparison system')
    args = parser.parse_args(argv)
    try:
        ledgers = {}
        for value in args.ledger:
            system, separator, path = value.partition('=')
            if not separator or not path or system in ledgers:
                raise ValueError('use each --ledger SYSTEM=PATH exactly once')
            ledgers[system] = path
        result = compare_capacity_ledgers(ledgers)
    except (ValueError, TypeError, KeyError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
