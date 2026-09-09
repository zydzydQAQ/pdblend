"""Label declaration lifecycle without turning unmeasured work into observations."""
from collections import defaultdict


def annotate(points, origins, protocol):
    by_id = {x['cell_id']: x for x in points}
    by_work = defaultdict(list)
    for point in points:
        by_work[(protocol.pair_identity(point), point['repeat'], point.get('arm'))].append(point)
    for point in points:
        entries = origins[point['cell_id']]
        states = [e['status'] for e in entries]
        if point['measurement_valid']:
            scope = 'verified_complete_work' if point['work_complete'] else 'verified_incomplete_work_negative'
        elif point['status'] not in ('unmeasured', 'awaiting_mirror'):
            scope = 'observed_invalid_requires_diagnosis'
        elif any(point['cell_id'] in (s.get('completed') or []) for s in states):
            scope = 'awaiting_raw_verification'
        else:
            losses = [(s.get('first_complete_breach') or {}).get(point['dataset']) for s in states]
            same = by_work[(protocol.pair_identity(point), point['repeat'], point.get('arm'))]
            alternatives = [x for x in same if x['cell_id'] != point['cell_id'] and
                (x['measurement_valid'] or any(e['status']['phase'] == 'running'
                    for e in origins[x['cell_id']]))]
            if any(v is not None and point['rate_rps'] > v for v in losses):
                scope = 'not_required_above_first_loss'
            elif all(s['phase'] in ('stopped_at_boundary', 'complete') for s in states):
                scope = 'superseded_declaration' if alternatives else 'retired_queue_awaiting_final_selection'
            else:
                scope = 'required_queue_pending'
        point['scope_status'] = scope
        point['required_execution'] = scope not in (
            'not_required_above_first_loss', 'superseded_declaration',
            'retired_queue_awaiting_final_selection')
    return points
