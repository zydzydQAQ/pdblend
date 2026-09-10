"""Export measured preparation/fault energy without adding overlapping windows."""
import argparse
import csv
import json
from pathlib import Path
import time

import metrics as m

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def subtract(start, end, exclusions):
    remaining = [(start, end)]
    for left, right in sorted(exclusions):
        following = []
        for a, b in remaining:
            if right <= a or left >= b:
                following.append((a, b))
            else:
                if a < left:
                    following.append((a, left))
                if right < b:
                    following.append((right, b))
        remaining = following
    return remaining


def directories(root):
    return [(node, directory) for node in ('A', 'B', 'C')
            for directory in (root / node).glob('uniform-*') if directory.is_dir()]


def category(path, operation, failed):
    if operation:
        return 'failed_operation' if failed else 'measurement_preparation_and_cleanup'
    if failed:
        return 'failed_preparation_or_qualification'
    text = str(path)
    if 'qualification' in text:
        return 'qualification'
    if 'power-test' in text:
        return 'power_selftest'
    if 'restore' in text or 'restoration' in text:
        return 'restoration'
    return 'preparation_or_calibration'


def audit(path):
    receipt = m.read(path)
    operation = 'operation_start_s' in receipt
    restoration = 'setup_and_correctness_energy_j' in receipt
    qualification_meter = not operation and 'full_operation_energy_j' in receipt
    start = receipt['operation_start_s' if operation else 'measurement_start_s']
    end = receipt['operation_end_s' if operation else 'measurement_end_s']
    m.need(end > start and not receipt.get('sampling_error'), 'unfinished or invalid power sampling')
    m.need(receipt.get('power_evidence', {}).get('power_source_verified') is True,
           'power source has not been verified')
    power = path.parent / ('power/power.csv' if operation or restoration or qualification_meter else 'power.csv')
    expected = receipt.get('artifacts', {}).get(str(power))
    if expected:
        m.need(m.sha(power) == expected, 'changed power artifact')
    with power.open() as stream:
        samples = list(csv.DictReader(stream))
    energy = sum(m.integrate(samples, start, end, [f'gpu{i}_w' for i in range(8)]))
    energy_key = ('setup_and_correctness_energy_j' if restoration else
                  'full_operation_energy_j' if operation or qualification_meter else 'energy_j')
    m.need(m.close(energy, receipt[energy_key]),
           'producer energy differs from raw eight-GPU integral')
    if restoration:
        m.need(receipt.get('complete') is True and receipt.get('clock_restore_complete') is True
               and not receipt.get('errors'), 'restoration has not completed')
    elif qualification_meter:
        # A failed qualification can still have a valid, fully closed power
        # window. Count its measured energy as failed preparation; this grants
        # no qualification or serving capability.
        m.need(receipt.get('finished_s') and not receipt.get('node_lease_held')
               and receipt.get('clock_restore_complete') is True and not receipt.get('cleanup_errors'),
               'qualification meter has not completed cleanup')
        m.need(receipt.get('complete') is True or receipt.get('measurement_valid') is True,
               'failed qualification power window is not independently valid')
    elif not operation:
        m.need(receipt.get('measurement_valid') is True and receipt.get('gpu_indices') == list(range(8)),
               'preparation measurement is not complete and valid')
    failed = bool(receipt.get('error') or receipt.get('outer_cleanup_errors')
                  or (operation and not receipt.get('measurement_valid')))
    if operation:
        runner_state = path.parents[3] / 'status.json'
        if runner_state.exists():
            state = m.read(runner_state)
            failed = failed or any(isinstance(item, dict) and item.get('cell_id') == receipt.get('cell_id')
                                   for item in state.get('failed', []))
    state_ref = None
    if not operation:
        for parent in [path.parent, *path.parents[1:4]]:
            status_path = parent / 'status.json'
            if status_path.exists():
                state = m.read(status_path)
                if state.get('finished_s'):
                    failed = failed or bool(state.get('error') or state.get('passed') is False)
                    state_ref = m.ref(status_path)
                break
    return dict(receipt=m.ref(path), raw_power=m.ref(power), state=state_ref,
                start_s=start, end_s=end, full_window_energy_j=energy,
                category=category(path, operation, failed), operation=operation), samples


def collect(root=ROOT):
    primary = {node: [] for node in ('A', 'B', 'C')}
    candidates = []
    errors = []
    for node, directory in directories(root):
        for path in directory.glob('**/results/cells/*/summary.json'):
            try:
                summary = m.read(path)
                # Exclude even invalid serving windows: fault energy remains a
                # separate diagnostic and must never become preparation energy.
                start, end = summary['measurement_start_s'], summary['measurement_end_s']
                if end > start:
                    primary[node].append((start, end))
            except (OSError, ValueError, KeyError, TypeError):
                continue
        candidates += [(node, path) for path in directory.glob('**/measurement.json')]
        candidates += [(node, path) for path in directory.glob('**/operations/*/receipt.json')]
        for path in directory.glob('**/status.json'):
            try:
                status = m.read(path)
                if ('setup_and_correctness_energy_j' in status or
                        'full_operation_energy_j' in status and 'measurement_start_s' in status):
                    candidates.append((node, path))
            except (OSError, ValueError):
                continue
    audited = []
    for node, path in sorted(set(candidates)):
        try:
            entry, samples = audit(path)
            entry['node'] = 'Anew20260909' if node == 'A' else node
            audited.append((node, entry, samples))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors.append(dict(node=node, path=str(path), error=repr(exc)))
    # Prefer outer operation meters. Any nested transition meter is diagnostic
    # evidence only for the already covered seconds.
    audited.sort(key=lambda v: (v[0], not v[1]['operation'],
                               -(v[1]['end_s'] - v[1]['start_s']), v[1]['receipt']['path']))
    covered = {node: list(windows) for node, windows in primary.items()}
    entries = []
    for node, entry, samples in audited:
        a, b = entry['start_s'], entry['end_s']
        intervals = subtract(a, b, covered[node])
        energies = [sum(m.integrate(samples, x, y, [f'gpu{i}_w' for i in range(8)]))
                    for x, y in intervals]
        entry.update(exclusive_intervals=intervals, exclusive_energy_j=sum(energies),
                     exclusive_duration_s=sum(y-x for x, y in intervals),
                     excluded_overlap_energy_j=entry['full_window_energy_j'] - sum(energies),
                     added_to_rate_curve_energy=False)
        covered[node].append((a, b))
        entries.append(entry)
    return dict(schema='uniform-setup-energy-ledger-v1', updated_s=time.time(), entries=entries,
                pending_or_invalid_evidence=errors,
                semantics='Only observed non-overlapping eight-GPU windows; serving energy excluded. '
                          'Missing/unmetered intervals are not estimated. Nested meters are retained but not added twice.')


def export(result, out):
    out.mkdir(parents=True, exist_ok=True)
    target = out / 'setup-energy-ledger.json'
    temp = target.with_suffix('.tmp')
    temp.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    temp.replace(target)
    target = out / 'setup-energy-ledger.csv'
    temp = target.with_suffix('.tmp')
    fields = ['node', 'category', 'start_s', 'end_s', 'full_window_energy_j',
              'exclusive_duration_s', 'exclusive_energy_j', 'excluded_overlap_energy_j',
              'added_to_rate_curve_energy', 'receipt_path', 'receipt_sha256', 'power_path', 'power_sha256']
    with temp.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for entry in result['entries']:
            writer.writerow(dict(entry, receipt_path=entry['receipt']['path'],
                receipt_sha256=entry['receipt']['sha256'], power_path=entry['raw_power']['path'],
                power_sha256=entry['raw_power']['sha256']))
    temp.replace(target)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=HERE / 'reports/current')
    args = parser.parse_args()
    result = collect()
    export(result, args.out)
    print(json.dumps(dict(entries=len(result['entries']), pending=len(result['pending_or_invalid_evidence']))))
