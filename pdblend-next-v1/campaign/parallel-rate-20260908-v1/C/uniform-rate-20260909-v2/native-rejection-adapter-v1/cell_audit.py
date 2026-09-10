"""Preserve the full cell audit, then name proven native refusals accurately."""
from pathlib import Path
import sys

ROOT = Path('/root/workspace/pdblend-next-v1/campaign/parallel-rate-20260908-v1')
sys.path.insert(0, str(ROOT / 'common/uniform-rate-20260909-v2'))
import support as p
PARENT = dict(path=str(ROOT / 'common/uniform-rate-20260909-v2/audit_cell_capacity_v3.py'),
    sha256='f5445bff49578542c0cb01d76d7c7109a430182ba3c6f54fe4a0b228878175da')


def classify(observation):
    proof = observation.get('zero_output_diagnosis') or {}
    if proof.get('schema') != 'C-Eco-native128-refusal-and-deadline-audit-v1':
        return observation
    p.need(observation['system'] == 'ecoserve' and observation['model'] == '7b'
        and observation['measurement_host'] == 'C' and observation['work_complete'] is False,
        'native refusal classification is restricted to incomplete C EcoServe')
    p.need(proof['passed'] and proof['independently_recomputed']
        and proof['checkpoint'] == observation['checkpoint']
        and proof['no_unknown_errors'] and proof['cleanup_verified'], 'native proof mismatch')
    existing = observation['baseline_service_failure']
    refused, deadlines = proof['native_rejection_request_ids'], proof['timeout_request_ids']
    p.need(refused and len(set(refused)) == len(refused) and len(set(deadlines)) == len(deadlines)
        and not set(refused).intersection(deadlines)
        and set(refused + deadlines) == set(existing['failed_request_ids'])
        and len(refused) == proof['native_rejections']
        and len(deadlines) == proof['request_timeouts'] == observation['request_timeouts']
        and len(refused) + len(deadlines) == observation['n_expected'] - observation['completed_work_requests'],
        'native refusals and timeouts do not partition the original failure denominator')
    observation['baseline_service_failure'] = dict(existing,
        classification='baseline_explicit_native_admission_queue_full', passed=True,
        independently_recomputed=True, native_queue_evidence=proof,
        native_rejections=len(refused), actual_request_timeouts=len(deadlines),
        native_rejection_request_ids=refused, timeout_request_ids=deadlines,
        native_rejections_are_not_timeouts=True)
    return observation


def audit(checkpoint):
    original = p.load(PARENT, 'uniform_full_cell_native_refusal_parent')
    return classify(original.audit(checkpoint))
