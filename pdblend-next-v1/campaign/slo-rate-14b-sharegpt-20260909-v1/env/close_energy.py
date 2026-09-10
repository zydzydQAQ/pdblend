"""CPU-only closeout of distinct primary GPU windows and separate setup costs.

Read small immutable receipts and prior raw-audit results. Never start GPU work,
transfer evidence, replay raw power, infer gap energy, or add overlapping outer
operation energy. A fresh output directory preserves each partial/final closeout.
"""
from pathlib import Path
import argparse
import collections
import hashlib
import json
import math
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import report

EXPECTED = {'A': 21, 'C': 41}
SMALL_LIMIT = 8 * 1024**2


def need(condition, message):
    if not condition:
        raise ValueError(message)


def close(a, b):
    return isinstance(a, (int, float)) and isinstance(b, (int, float)) and math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-5)


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def checked(reference):
    path = Path(reference['path'])
    need(path.stat().st_size <= SMALL_LIMIT, 'small evidence limit exceeded: ' + str(path))
    raw = path.read_bytes()
    need(hashlib.sha256(raw).hexdigest() == reference['sha256'], 'evidence SHA mismatch: ' + str(path))
    return json.loads(raw)


def window(start, end):
    need(finite(start) and finite(end) and end > start, 'invalid energy window')
    return {'start_s': start, 'end_s': end, 'duration_s': end - start}


def deduplicate(rows):
    distinct = {}
    duplicates = 0
    for row in rows:
        key = (row['measurement_host'], row['cell_id'], row.get('engineering_attempt', 1))
        if key in distinct:
            need(row == distinct[key], 'conflicting cell/attempt observations: ' + str(key))
            duplicates += 1
        else:
            distinct[key] = row
    return list(distinct.values()), duplicates


def nonoverlap(entries):
    ordered = sorted(entries, key=lambda e: e['window']['start_s'])
    for left, right in zip(ordered, ordered[1:]):
        need(left['window']['end_s'] <= right['window']['start_s'],
             'same-host windows overlap: ' + left['id'] + ' / ' + right['id'])
    return ordered


def snapshot_inputs(out):
    rows, inputs = [], []
    for node in 'AC':
        path = ROOT / node / 'observations.json'
        raw = path.read_bytes()
        snap = out / 'inputs' / (node + '-observations.json')
        snap.parent.mkdir(parents=True, exist_ok=True)
        snap.write_bytes(raw)
        source = {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest()}
        inputs.append({'source': source, 'snapshot': report.ref(snap)})
        rows.extend(report.normalize(r, node, source) for r in report.records(json.loads(raw)))
    path = ROOT / 'B/reference.json'
    config = json.loads(path.read_text())
    raw = Path(config['source']['path']).read_bytes()
    need(hashlib.sha256(raw).hexdigest() == config['source']['sha256'], 'B immutable source drift')
    snap = out / 'inputs/B-reference-observations.json'
    selected = set(config['cell_ids'])
    selected_rows = [r for r in report.records(json.loads(raw)) if report.flatten(r).get('cell_id') in selected]
    report.save(snap, selected_rows)
    inputs.append({'source': config['source'], 'reference_selector': report.ref(path), 'selected_snapshot': report.ref(snap)})
    rows.extend(report.normalize(r, 'B', config['source'], reference_only=True) for r in selected_rows)
    rows, duplicates = deduplicate(rows)
    return rows, inputs, duplicates, len(selected)


def primary(row):
    s, r = checked(row['summary']), checked(row['receipt'])
    raw_audit = row.get('verification', {}).get('raw', {})
    energy = row['energy_j']
    need(not row.get('report_issues'), 'observation identity or measurement scope mismatch')
    need(row.get('measurement_valid') and row.get('audit_accepted') and row.get('primary_window_only'), 'unqualified primary observation')
    need(row.get('energy_measured_gpu_count') == s.get('gpu_count') == 8, 'primary eight-GPU scope missing')
    need(raw_audit.get('all_eight_gpu_energy_reintegrated'), 'prior raw energy replay missing')
    per_gpu = row.get('energy_per_gpu_j', raw_audit.get('energy_per_gpu_j'))
    need(isinstance(per_gpu, list) and len(per_gpu) == 8 and all(finite(e) and e >= 0 for e in per_gpu), 'invalid per-GPU energy')
    need(finite(energy) and energy >= 0 and close(sum(per_gpu), energy)
         and close(raw_audit.get('primary_energy_j'), energy) and close(s.get('energy_j'), energy), 'primary energy arithmetic/provenance mismatch')
    if row.get('audit_reference'):
        audited = checked(row['audit_reference'])
        need(audited['cell_id'] == row['cell_id'] and close(audited['energy_j'], energy), 'immutable audited observation mismatch')
    need(r['cell_id'] == row['cell_id'] and r['trace_sha256'] == row['trace_sha256'], 'receipt identity mismatch')
    need(s.get('measurement_valid') and s.get('power_source_verified'), 'invalid primary meter')
    w = window(s['measurement_start_s'], s['measurement_end_s'])
    need(close(w['duration_s'], row['measurement_duration_s']), 'primary duration mismatch')
    fixed = s['fixed_window']
    need(fixed.get('arrival_window_s') == row['arrival_window_s'] == 100, 'arrival window is not 100s')
    need(close(w['start_s'], fixed['arrival_epoch_s']) and close(fixed['arrival_window_end_s'], w['start_s'] + 100), 'arrival clock mismatch')
    need(fixed.get('window_observed_complete') and fixed.get('clock_consistent') and s.get('drain_complete') and not s.get('incomplete_drain'), 'arrival/drain incomplete')
    ends = {k: s.get(k) for k in ('client_completion_end_s', 'drain_end_s', 'controls_end_s')}
    need(all(finite(v) and w['start_s'] <= v <= w['end_s'] + 1e-5 for v in ends.values()), 'frozen primary service/control tail outside meter window')
    need(w['end_s'] >= fixed['arrival_window_end_s'], '100s arrivals not covered')
    need(r.get('child_stopped') and r.get('clock_restore_complete') and r.get('measurement_valid')
         and not r.get('outer_cleanup_errors') and finite(r.get('finished_s')), 'outer terminal cleanup not certified')
    outer = window(r['operation_start_s'], r['operation_end_s'])
    need(outer['start_s'] <= w['start_s'] and outer['end_s'] >= w['end_s'] and r['finished_s'] >= outer['end_s'], 'outer/primary containment mismatch')
    outer.update(energy_j=r.get('full_operation_energy_j'), receipt_finished_s=r['finished_s'],
                 added_to_primary=False, incremental_overhead_j=None,
                 note='Overlapping diagnostic meter retained separately; no sum or subtraction against primary.')
    return {'id': row['cell_id'] + '/attempt-' + str(row.get('engineering_attempt', 1)),
            'cell_id': row['cell_id'], 'engineering_attempt': row.get('engineering_attempt', 1),
            'node': row['measurement_host'], 'reference_only': row['reference_only'],
            'system': row['system'], 'rate_rps': row['rate_rps'], 'slo_scale': row['slo_scale'], 'repeat': row.get('repeat', 1),
            'energy_j': energy, 'energy_per_gpu_j': per_gpu, 'gpu_indices': list(range(8)), 'window': w,
            'arrival_window_s': 100, 'arrival_window_end_s': fixed['arrival_window_end_s'],
            'primary_service_terminal_times': ends, 'actual_tail_after_arrival_window_s': w['end_s'] - fixed['arrival_window_end_s'],
            'covers_100s_arrivals_and_actual_drain': True, 'primary_all8_raw_replay_previously_passed': True,
            'source_checks': 'Pinned small summary/receipt/audited metadata and prior raw integration checked; no raw reintegration in this closeout.',
            'summary': row['summary'], 'receipt': row['receipt'], 'audit_reference': row.get('audit_reference'),
            'raw_power': row['raw_power'], 'raw_requests': row['raw_requests'], 'checkpoint': row['checkpoint'],
            'work_complete': row['work_complete'], 'failed_requests': row.get('failed_requests'),
            'outer_operation_diagnostic_only': outer}


def setup(node):
    path = ROOT / node / 'setup-energy-ledger.json'
    if not path.exists():
        return {'status': 'unknown_no_setup_ledger_in_current_scope', 'measured_subtotal_j': None,
                'complete_setup_energy_j': None, 'counted_windows': [], 'preserved_entries': []}
    source = report.ref(path)
    ledger = checked(source)
    counted, evidence, seen_ids, seen_receipts = [], [], set(), set()
    for entry in ledger['entries']:
        need(entry['operation_id'] not in seen_ids, 'duplicate setup operation id')
        seen_ids.add(entry['operation_id'])
        if not entry.get('included_in_measured_setup_subtotal'):
            continue
        need(entry['energy_status'] == 'measured' and entry.get('all8_gpu_coverage')
             and entry.get('gpu_indices') == list(range(8)) and entry.get('measurement_valid'), 'invalid counted setup meter')
        receipt = entry['receipt']
        need(receipt['sha256'] not in seen_receipts, 'duplicate setup receipt counted')
        seen_receipts.add(receipt['sha256'])
        p = Path(receipt['path'])
        if p.exists() and p.stat().st_size <= SMALL_LIMIT:
            state = checked(receipt)
            proof = {'receipt': receipt, 'method': 'small_receipt_SHA_checked'}
        else:
            projection_ref = entry['status_projection']
            projection = checked(projection_ref)
            matches = [r for r in projection['operations'].values() if r['receipt'] == receipt]
            need(len(matches) == 1, 'setup projection does not bind exact receipt')
            record = matches[0]
            need(all('gpu%d_w' % i in record['power_header'].split(',') for i in range(8)), 'projection missing GPU columns')
            state = record['state']
            proof = {'receipt': receipt, 'projection': projection_ref,
                     'method': 'prior_on_host_receipt_SHA_and_scalar_projection_checked_no_large_receipt_read'}
        energy = next((state[k] for k in ('all8_operation_energy_j', 'full_operation_energy_j', 'energy_j') if k in state), None)
        start = state.get('measurement_start_s', state.get('operation_start_s'))
        end = state.get('measurement_end_s', state.get('operation_end_s'))
        need(close(energy, entry['energy_j']) and close(start, entry['window']['start_s'])
             and close(end, entry['window']['end_s']) and state.get('measurement_valid'), 'setup receipt/ledger scalar mismatch')
        counted.append({'id': entry['operation_id'], 'energy_j': energy, 'window': window(start, end),
                        'operation_status': entry['operation_status'], 'receipt': receipt})
        evidence.append(proof)
    counted = nonoverlap(counted)
    subtotal = sum(e['energy_j'] for e in counted)
    need(close(subtotal, ledger['measured_disjoint_setup_subtotal_j']), 'setup subtotal mismatch')
    return {'status': 'known_disjoint_windows_with_unknown_gaps', 'source': source,
            'measured_subtotal_j': subtotal, 'measured_operation_count': len(counted), 'counted_windows': counted,
            'receipt_checks': evidence, 'preserved_entries': ledger['entries'],
            'failed_setup_entries': [e for e in ledger['entries'] if e.get('operation_status') == 'failed'],
            'direct_failed_operation_measured_subtotal_j': sum(e['energy_j'] for e in counted if e['operation_status'] == 'failed'),
            'failed_parent_attempt_energy_j': None,
            'failed_parent_note': 'Failed parent attempts can share already-counted child windows. Preserve child/reuse references; do not add parent figures again.',
            'unknown_entries': [e for e in ledger['entries'] if e['energy_status'] in ('unknown', 'not_measured')],
            'complete_setup_energy_j': None, 'whole_campaign_energy_j': None}


def node_result(node, rows, expected):
    windows, errors = [], []
    for row in rows:
        try:
            windows.append(primary(row))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors.append({'cell_id': row['cell_id'], 'engineering_attempt': row.get('engineering_attempt', 1), 'error': repr(exc)})
    windows = nonoverlap(windows)
    need(len({w['receipt']['sha256'] for w in windows}) == len(windows), 'duplicate primary receipt counted')
    for left, right in zip(windows, windows[1:]):
        need(left['outer_operation_diagnostic_only']['receipt_finished_s'] <= right['window']['start_s'],
             'prior outer cleanup did not finish before next primary window')
    prep = setup(node)
    combined = nonoverlap(windows + prep['counted_windows'])
    gaps = []
    for left, right in zip(combined, combined[1:]):
        if left['window']['end_s'] < right['window']['start_s']:
            gaps.append(dict(window(left['window']['end_s'], right['window']['start_s']),
                             after=left['id'], before=right['id'], accounted_energy_j=None))
    per_system = {}
    for system in report.SYSTEMS:
        group = [w for w in windows if w['system'] == system]
        per_system[system] = {'cell_attempts': len(group), 'primary_energy_j': sum(w['energy_j'] for w in group) if group else None}
    terminal = None
    if node in EXPECTED:
        path = ROOT / node / 'run-002/status.json'
        if path.exists():
            source = report.ref(path)
            state = checked(source)
            terminal = {'source': source, 'complete': bool(state.get('complete') and state.get('five_system_complete')
                         and state.get('finished_s') and not state.get('node_lease_held') and state.get('phase') == 'complete')}
    complete = len(rows) == expected and len(windows) == expected and not errors and (node == 'B' or terminal and terminal['complete'])
    return {'node': node, 'reference_only': node == 'B', 'expected_observations': expected,
            'loaded_distinct_cell_attempts': len(rows), 'verified_primary_windows': len(windows),
            'grid_energy_status': 'complete_selected_reference' if complete and node == 'B' else 'complete_formal_grid' if complete else 'partial',
            'formal_grid_complete': bool(complete) if node != 'B' else None, 'node_terminal': terminal,
            'primary_energy_subtotal_j': sum(w['energy_j'] for w in windows) if windows else None,
            'primary_energy_subtotal_kwh': sum(w['energy_j'] for w in windows) / 3600000 if windows else None,
            'by_system': per_system, 'primary_windows': windows, 'unverified_or_unavailable_observations': errors,
            'same_host_primary_nonoverlap_verified': True, 'outer_cleanup_before_next_primary_verified': True,
            'window_check_scope': 'Verified windows listed here; any unavailable/unverified observations remain explicit and prevent a complete grid claim.',
            'setup': prep, 'setup_and_primary_disjoint_verified': True,
            'selected_accounting_window_gaps': gaps, 'gap_energy_j': None,
            'gap_note': 'Intervals not covered by the selected primary/setup totals. Some overlap diagnostic outer meters, which are intentionally not added. No interpolation or campaign-total inference.',
            'complete_setup_energy_j': None, 'whole_campaign_energy_j': None,
            'accounting_scope': 'Eight-GPU energy only; CPU, wall-plug, unmeasured intervals and unobserved future cells are unknown.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--require-final', action='store_true')
    args = parser.parse_args()
    out = args.out.resolve()
    need(out.is_relative_to(ROOT) and not out.exists(), 'fresh output directory inside this campaign required')
    out.mkdir(parents=True)
    rows, inputs, duplicates, b_expected = snapshot_inputs(out)
    nodes = {node: node_result(node, [r for r in rows if r['measurement_host'] == node], EXPECTED.get(node, b_expected)) for node in 'ABC'}
    final = all(nodes[n]['formal_grid_complete'] for n in 'AC')
    result = {'schema': 'slo14-energy-closeout-v1', 'created_s': time.time(), 'status': 'final_formal_grid' if final else 'partial_formal_grid',
              'new_formal_grid_complete': final, 'expected_new_observations': 62,
              'verified_new_primary_windows': sum(nodes[n]['verified_primary_windows'] for n in 'AC'),
              'duplicate_observation_copies_not_recounted': duplicates,
              'nodes': nodes, 'inputs': inputs, 'tool': report.ref(__file__), 'observation_normalizer': report.ref(report.__file__),
              'raw_power_reintegrated_this_closeout': False, 'no_remote_transfer': True, 'no_gpu_work': True,
              'primary_plus_outer_total_j': None, 'complete_campaign_energy_j': None,
              'cross_host_performance_or_energy_ratios_computed': False,
              'policy': 'Distinct cell/attempt primary windows counted once, confirmation repeats retained. Setup counted independently by unique physical receipt. Parent/child and overlapping outer meters never added. Unknown gaps remain null. B is an immutable historical reference.'}
    report.save(out / 'energy-closeout.json', result)
    lines = ['# GPU energy closeout', '', 'Status: ' + result['status'] + '. New primary windows verified: ' + str(result['verified_new_primary_windows']) + '/62.', '',
             '| Node | Scope | Verified windows | Primary GPU energy (J) | Known setup windows (J) |',
             '|---|---|---:|---:|---:|']
    for n, v in nodes.items():
        a = v['primary_energy_subtotal_j']; b = v['setup']['measured_subtotal_j']
        lines.append('| %s | %s | %s/%s | %s | %s |' % (n, 'existing 1x reference' if n == 'B' else 'new formal grid', v['verified_primary_windows'], v['expected_observations'], f'{a:.6f}' if a is not None else 'unknown', f'{b:.6f}' if b is not None else 'unknown'))
    lines.extend(['', 'Primary windows cover the fixed 100-second arrivals plus actual request drain and the frozen primary control tail. Outer cleanup is checked separately and finishes before the next primary window. Its overlapping energy is neither added nor subtracted.', '',
                  'Known setup costs include failed operations once. Failed parent attempts retain their child/reuse references. Gaps, complete setup energy and complete campaign energy remain unknown; no cross-host ratio is computed.', '',
                  'This closeout checks pinned small evidence and prior independent raw-audit results. It performs no GPU operation, transfer or new raw-power integration. Details and exact receipt references are in energy-closeout.json.'])
    (out / 'README.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'closeout': report.ref(out / 'energy-closeout.json'), 'status': result['status'], 'verified_new_primary_windows': result['verified_new_primary_windows'],
                      'nodes': {n: {k: v[k] for k in ('verified_primary_windows', 'primary_energy_subtotal_j', 'grid_energy_status')} for n, v in nodes.items()}}))
    need(not args.require_final or final, 'final grid not complete; preserved partial closeout, no final claim')


if __name__ == '__main__':
    main()
