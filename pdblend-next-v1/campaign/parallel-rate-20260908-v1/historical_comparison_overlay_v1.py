"""Read-only scientific eligibility correction; historical raw records stay intact."""
import copy
import hashlib
import json
from pathlib import Path

AUDIT_SHA = 'dd52da1a85e9e15c377b0be0c23c30c5fa1109f323b0a5bbf439ed034126d8aa'
QUARANTINED = frozenset({
    '32b-alpaca-r4-s701-w100-dynamollm-slo1',
    '32b-sharegpt-r2-s701-w100-dynamollm-slo1',
})


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def need(value, message):
    if not value:
        raise ValueError(message)


def checked(reference, sources):
    path = reference['path']
    need(sha(path) == reference['sha256'], 'changed eligibility evidence: ' + path)
    sources[path] = reference['sha256']
    return json.loads(Path(path).read_text())


def verify(points, audit_reference, sources):
    need(audit_reference['sha256'] == AUDIT_SHA, 'unreviewed historical timing audit')
    audit = checked(audit_reference, sources)
    need(sha(audit['source']['path']) == audit['source']['sha256'], 'timing auditor changed')
    sources[audit['source']['path']] = audit['source']['sha256']
    need(audit['schema'] == 'all-model-original-Dynamo90-arrival-readonly-screen-v2'
         and len(audit['points']) == 90 and audit['original450_unchanged'] is True,
         'wrong original Dynamo timing audit')
    records = {p['cell_id']: p for p in points}
    need(len(records) == len(points), 'duplicate historical cell identity')
    audited = {p['cell_id']: p for p in audit['points']}
    need(len(audited) == 90, 'duplicate timing-audit cell identity')
    excluded = []
    for cell_id in sorted(QUARANTINED):
        point = records[cell_id]
        raw = audited[cell_id]
        need(point['model'] == '32b' and point['system'] == 'dynamollm'
             and point['phase'] == 'main' and point['metrics_verified'] is True,
             'quarantine must refer to its original raw-verified main record')
        cp = checked(raw['checkpoint'], sources)
        checked(raw['receipt'], sources)
        need(point['checkpoint_path'] == raw['checkpoint']['path']
             and point['receipt_path'] == raw['receipt']['path'], 'quarantine execution differs')
        for path, digest in raw['raw_files'].items():
            need(cp['artifacts'].get(path) == digest and sha(path) == digest,
                 'quarantine raw file changed: ' + path)
            sources[path] = digest
        need(point['executed_source']['binding_sha256'] == raw['source']['binding_sha256'],
             'quarantine source identity differs')
        need(all(point[k] == raw[k] for k in ('energy_j', 'n_expected', 'failed_requests',
                                             'request_timeouts', 'slo_attainment', 'work_complete')),
             'quarantine raw metrics differ from the audited observation')
        need(raw['actual_dispatch_lateness_max_s'] > 10
             and raw['controller_max_event_gap_s'] > 10
             and raw['last_actual_dispatch_offset_s'] > 100
             and raw['power_covers_measurement'] is True,
             'diagnosed scheduler stall evidence missing')
        excluded.append(dict(cell_id=cell_id, checkpoint=raw['checkpoint'], receipt=raw['receipt'],
            raw_arithmetic_verified=True, scientific_comparison_eligible=False,
            reason='synchronous admission planning stalled the event loop and changed the declared arrival trace',
            energy_j=point['energy_j'], energy_preserved=True,
            actual_dispatch_lateness_max_s=raw['actual_dispatch_lateness_max_s'],
            controller_max_event_gap_s=raw['controller_max_event_gap_s'],
            last_actual_dispatch_offset_s=raw['last_actual_dispatch_offset_s'],
            power_covers_measurement=True))
    return dict(schema='historical-scientific-comparison-overlay-v1',
        original_raw_records_unchanged=True, timing_audit=audit_reference,
        explicit_diagnosed_cells_only=True, quarantined=excluded,
        raw_energy_not_erased=True, raw_energy_not_reintegrated_by_this_overlay=True,
        applies_to_comparisons_and_figures=True,
        scope='Two diagnosed arrival stalls; all other historical eligibility follows the existing raw verification.')


def comparison_view(points, overlay):
    need(overlay['schema'] == 'historical-scientific-comparison-overlay-v1'
         and {x['cell_id'] for x in overlay['quarantined']} == QUARANTINED,
         'exact diagnosed quarantine set required')
    need(len(overlay['quarantined']) == len(QUARANTINED), 'duplicate quarantine')
    excluded = {x['cell_id']: x for x in overlay['quarantined']}
    view = copy.deepcopy(points)
    need(len({x['cell_id'] for x in view}) == len(view), 'duplicate historical record')
    need(QUARANTINED <= {x['cell_id'] for x in view}, 'missing quarantined raw record')
    for point in view:
        point['raw_arithmetic_verified'] = point['metrics_verified']
        point['scientific_comparison_eligible'] = bool(point['metrics_verified'] and point['cell_id'] not in excluded)
        if point['cell_id'] in excluded:
            correction = excluded[point['cell_id']]
            need(correction['scientific_comparison_eligible'] is False
                 and correction['raw_arithmetic_verified'] is True
                 and point['energy_j'] == correction['energy_j'], 'quarantine energy or eligibility changed')
            # The legacy renderer gates curves and paired ratios on this field.
            # Change only this derived view, never the archived master record.
            point['metrics_verified'] = False
            point['status'] = 'engineering_invalid_arrival_trace'
            point['scientific_exclusion_reason'] = correction['reason']
    return view
