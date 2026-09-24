"""Host analysis only: explicit, immutable equivalence of complete trace events.

No original trace/point/receipt is rewritten. The sole ignored trace field is
boundary_policy (selection provenance); even other metadata must match exactly.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

SCHEMA = 'exact-boundary-trace-equivalence/v1'
REGISTRY = 'frozen-historical-baseline-registry/v1'
IGNORED = ('boundary_policy',)
WORKLOAD = ('model_id', 'dataset', 'scale', 'rate_rps', 'seed', 'duration_s', 'slo',
            'measurement_protocol_version')
HARDWARE = ('image_digest', 'model_hash', 'tokenizer_hash', 'fleet_gpu_uuids',
            'runtime_source_sha256')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                   allow_nan=False).encode()).hexdigest()


def binding(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def load_bound(ref):
    if binding(ref['path'])['sha256'] != ref['sha256']:
        raise ValueError('equivalence artifact checksum differs')
    return json.loads(Path(ref['path']).read_text())


def write_new(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    return binding(path)


def projection(trace):
    if not isinstance(trace.get('requests'), list) or not trace['requests']:
        raise ValueError('complete nonempty request sequence required')
    if trace.get('selection_split') != 'evaluation':
        raise ValueError('only evaluation traces can establish comparison equivalence')
    for name in ('model_id', 'dataset', 'duration_s', 'seed', 'slo', 'rate_rps',
                 'corpus_sha256', 'measurement_protocol_version'):
        if name not in trace:
            raise ValueError('trace missing workload identity: '+name)
    value = {k:v for k,v in trace.items() if k not in IGNORED}
    digest(value)  # Reject every nonfinite number, including within requests.
    return value


def workload(point):
    return dict(**{k:point[k] for k in WORKLOAD},
                hardware={k:point['engine_identity'][k] for k in HARDWARE})


def trace_matches_point(trace, point):
    names = ('model_id', 'dataset', 'rate_rps', 'duration_s', 'seed', 'slo',
             'measurement_protocol_version')
    if digest({k:trace[k] for k in names}) != digest({k:point[k] for k in names}):
        raise ValueError('trace configuration differs from its bound point')


def checked_record(point_ref, receipt_ref, *, load=load_bound):
    point, receipt = load(point_ref), load(receipt_ref)
    if point.get('system') not in ('mixed', 'distserve', 'ecoserve', 'dynamollm'):
        raise ValueError('registry requires a baseline')
    if (receipt.get('point') != point['name'] or receipt.get('point_sha256') != digest(point)
            or receipt.get('artifacts', {}).get('point.json') != point_ref['sha256']):
        raise ValueError('baseline receipt does not bind its exact point')
    result_ref = binding(Path(receipt_ref['path']).parent/'result.json')
    result = load(result_ref)
    if (receipt.get('artifacts', {}).get('result.json') != result_ref['sha256']
            or result != receipt.get('result')):
        raise ValueError('baseline receipt does not bind its exact result')
    metrics = result.get('metrics', {})
    finite = lambda value: type(value) in (int, float) and math.isfinite(value) and value >= 0
    if (any(receipt.get(k) is not True for k in ('recorded_window_complete', 'cleanup_passed',
                                              'measurement_evidence_valid'))
            or not all(finite(metrics.get(k)) for k in ('energy_service_j', 'energy_tail_j', 'service_start_s'))):
        raise ValueError('frozen baseline must have a complete service/tail measurement')
    if metrics.get('duration_s') != point['duration_s']:
        raise ValueError('baseline service duration differs')
    # The receipt binds the trace reference through point.json. Read the large
    # trace only when an actual endpoint needs complete-event equivalence.
    return dict(point=point_ref, receipt=receipt_ref, result=result_ref,
                point_spec=point, service_start_s=metrics['service_start_s'])


def make_registry(candidates, *, history_inventory=None, load=load_bound):
    """Freeze the earliest complete receipt from an explicit candidate inventory.

    Every candidate ref remains bound; no SLO verdict or energy magnitude is a
    selection input. Caller must provide the full known history for each point.
    """
    if history_inventory is not None:
        history = load(history_inventory)
        if sorted(digest(r) for r in history['completed_receipts']) != sorted(digest(c['receipt']) for c in candidates):
            raise ValueError('candidate inventory does not include all predeclared frozen receipts')
    values = [checked_record(v['point'], v['receipt'], load=load) for v in candidates]
    groups = {}
    for row in values:
        p = row['point_spec']
        key = digest(dict(system=p['system'], workload=workload(p), trace=p['trace'],
                          engine=p['engine_identity'], source=p['source_manifest'], inputs=p.get('inputs')))
        groups.setdefault(key, []).append(row)
    entries = []
    for candidates_for_point in groups.values():
        chosen = min(candidates_for_point, key=lambda v:(v['service_start_s'], v['receipt']['sha256']))
        entries.append(dict(point=chosen['point'], receipt=chosen['receipt'],
                            selection_rule='first_complete_service_energy_no_slo_or_energy_selection'))
    return dict(schema=REGISTRY, candidates=candidates, entries=entries, history_inventory=history_inventory,
                scope='explicit_historical_baseline_receipts_only')


def read_registry(ref, *, load=load_bound):
    registry = load(ref)
    if registry.get('schema') != REGISTRY or registry != make_registry(registry['candidates'],
            history_inventory=registry.get('history_inventory'), load=load):
        raise ValueError('historical frozen baseline registry differs')
    return [checked_record(e['point'], e['receipt'], load=load) for e in registry['entries']]


def build_equivalence(old, new_point_ref, *, load=load_bound):
    new = load(new_point_ref); prior = old['point_spec']
    if new.get('system') != 'pdblend' or new.get('experiment_phase') != 'slo_boundary_extension':
        raise ValueError('equivalence target must be an actual adaptive PD point')
    if digest(workload(prior)) != digest(workload(new)):
        raise ValueError('comparison workload/hardware differs')
    left, right = load(prior['trace']), load(new['trace'])
    trace_matches_point(left, prior); trace_matches_point(right, new)
    if digest(projection(left)) != digest(projection(right)):
        raise ValueError('complete request sequence or non-provenance metadata differs')
    policy_ref = new['boundary_policy']; policy = load(policy_ref)
    if (right.get('boundary_policy') != policy_ref or policy.get('run_id') != new.get('run_id')
            or policy.get('model_id') != new['model_id'] or policy.get('revision') != new['revision']):
        raise ValueError('new point does not bind its actual policy')
    return deepcopy(dict(schema=SCHEMA, historical_point=old['point'], historical_receipt=old['receipt'],
        target_point=new_point_ref, target_policy=policy_ref, traces=[prior['trace'], new['trace']],
        ignored_trace_fields=list(IGNORED), comparison_workload=workload(new),
        analysis_trace_identity_sha256=digest(projection(right)),
        request_count=len(right['requests']), full_request_sequence_compared=True))


def read_equivalence(ref, campaign, registry, *, load=load_bound):
    value = load(ref)
    if value.get('schema') != SCHEMA:
        raise ValueError('unknown trace equivalence protocol')
    old = next((r for r in registry if r['point']==value.get('historical_point')
                and r['receipt']==value.get('historical_receipt')), None)
    if old is None or value != build_equivalence(old, value['target_point'], load=load):
        raise ValueError('trace equivalence does not reproduce its bound evidence')
    target = load(value['target_point'])
    if (target.get('run_id') != campaign['run_id']
            or value['target_policy'] not in campaign.get('active_extension_policy_refs', [])):
        raise ValueError('trace equivalence policy is not authorized by this active campaign')
    check_meter_compatibility(old['point_spec'], target, campaign)
    return value



def check_meter_compatibility(prior, target, campaign):
    old_meter = prior['engine_identity']['measurement_source_sha256']
    new_meter = target['engine_identity']['measurement_source_sha256']
    if old_meter != new_meter:
        from pdblend.bench.measurement_compatibility import load_compatibility
        matched = False
        for proof_ref in campaign.get('measurement_compatibility', []):
            sources = load_compatibility(proof_ref)['sources']
            for left, right in (sources, sources[::-1]):
                if (left['source_manifest'] == prior['source_manifest']
                        and right['source_manifest'] == target['source_manifest']
                        and left['measurement_source_sha256'] == old_meter
                        and right['measurement_source_sha256'] == new_meter):
                    matched = True
        if not matched:
            raise ValueError('trace equivalence cannot waive measurement compatibility')

def prepare_reuse(campaign, endpoint_refs, registry_ref, out, *, load=load_bound):
    """Create proof only after immutable target points exist; no GPU source edit."""
    registry = read_registry(registry_ref, load=load); proofs = []
    for endpoint in endpoint_refs:
        target = load(endpoint)
        for old in registry:
            if digest(workload(old['point_spec'])) != digest(workload(target)):
                continue
            proof = build_equivalence(old, endpoint, load=load)
            ref = write_new(Path(out)/(digest(proof)+'.json'), proof)
            read_equivalence(ref, campaign, registry, load=load)
            proofs.append(ref)
    return registry, proofs


def permits(proofs, old_trace, new_trace):
    return any(old_trace == p['traces'][0] and new_trace == p['traces'][1] for p in proofs)


def annotate(rows, proofs):
    """Only validated proofs enter here; raw trace SHA and all metrics stay put."""
    mapping = {}
    for proof, ref in proofs:
        for trace in proof['traces']:
            existing = mapping.setdefault(trace['sha256'], (proof['analysis_trace_identity_sha256'], []))
            if existing[0] != proof['analysis_trace_identity_sha256']:
                raise ValueError('one trace maps to conflicting analysis identities')
            existing[1].append(ref)
    for row in rows:
        row.pop('analysis_trace_identity_sha256', None)
        row.pop('analysis_trace_equivalence_refs', None)
        found = mapping.get(row.get('trace_sha256'))
        if found:
            row['analysis_trace_identity_sha256'] = found[0]
            row['analysis_trace_equivalence_refs'] = deepcopy(found[1])
    return rows


def validate_reuse(entry, target_ref, campaign, registry, proofs, *, load=load_bound):
    """Completion uses the same authority and identity as preparation/export."""
    receipt_ref = entry['receipt']
    point_ref = binding(Path(receipt_ref['path']).parent/'point.json')
    old = checked_record(point_ref, receipt_ref, load=load)
    target = load(target_ref); prior = old['point_spec']
    if (entry['system'] != prior['system'] or entry['dataset'] != prior['dataset']
            or entry['scale'] != target['scale'] or digest(workload(prior)) != digest(workload(target))):
        raise ValueError('reused baseline belongs to another endpoint identity')
    inventory = load(campaign['baseline_energy_gaps'])
    frozen = [r['receipt'] for r in inventory['frozen_baselines']]
    frozen += [r['receipt'] for r in registry]
    if receipt_ref not in frozen:
        raise ValueError('reused baseline receipt was not predeclared frozen')
    check_meter_compatibility(prior, target, campaign)
    refs = entry.get('trace_equivalence_refs', [])
    if prior['trace'] != target['trace']:
        matches = [(proof, ref) for proof, ref in proofs if ref in refs
                   and proof['historical_receipt']==receipt_ref and proof['target_point']==target_ref]
        if len(matches) != 1 or set(map(digest, refs)) != {digest(matches[0][1])}:
            raise ValueError('cross-trace baseline reuse lacks its exact authorized proof')
    elif refs:
        raise ValueError('exact trace reuse cannot carry an unrelated equivalence proof')
    return old
