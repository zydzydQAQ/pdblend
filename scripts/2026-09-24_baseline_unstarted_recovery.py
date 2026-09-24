"""Host-only continuation of never-created baseline windows; no GPU imports."""
from copy import deepcopy
import json
from pathlib import Path

from pdblend.bench import comparison_campaign as cc
from pdblend.bench.resident_session import digest, engine_signature, write_new

REGISTRY_SCHEMA = 'baseline-unstarted-recovery-registry/v1'
TERMINAL = {'succeeded', 'failed', 'cancelled', 'blocked'}


def read(path):
    return json.loads(Path(path).read_text())


def normalized_payload(payload):
    result = deepcopy(payload)
    result.setdefault('depends_on', [])
    result.setdefault('after_terminal', [])
    return result


def group_for(job):
    argv = job['payload']['argv']
    if argv.count('--group') != 1 or '--previous' in argv:
        raise ValueError('baseline recovery requires one immutable group and no previous-session projection')
    path = Path(argv[argv.index('--group') + 1]).resolve()
    group = read(path)
    expected_id = 'comparison-' + group['model_id'].split('-')[1].lower() + '-' + digest(group)[:16]
    if (job['job_id'] != expected_id or job['payload']['session_id'] != group['session_id']
            or engine_signature(group['engine_identity']) != group['engine_signature']):
        raise ValueError('job/group immutable identity differs')
    points = group['points']
    if not points or any(point['system'] == 'pdblend' or point['system'] != job['payload']['system']
                         or point['engine_identity'] != group['engine_identity']
                         or point['revision'] != job['payload']['source_sha256'] for point in points):
        raise ValueError('baseline point/source/engine identity differs')
    return group, cc.binding(path)


def registry_for(runner):
    path = runner.out / 'baseline-unstarted-recovery-registry.json'
    stored = runner.state.get('baseline_unstarted_registry')
    if stored:
        if not path.is_file() or cc.binding(path) != stored:
            raise ValueError('baseline recovery registry changed after adoption')
        registry = cc.load_bound(stored)
    elif path.is_file():
        stored = cc.binding(path)
        registry = cc.load_bound(stored)
        runner.state['baseline_unstarted_registry'] = stored
        runner.emit('baseline_unstarted_registry_bound', registry=stored)
    else:
        return {}
    if registry.get('schema') != REGISTRY_SCHEMA or registry.get('run_id') != runner.package.name:
        raise ValueError('baseline recovery registry belongs to another round')
    entries = registry.get('recoveries', [])
    if len({entry['parent_job_id'] for entry in entries}) != len(entries):
        raise ValueError('ambiguous manual baseline recovery registry')
    return {entry['parent_job_id']: entry for entry in entries}


def session_evidence(runner, queue, job, group):
    """Inspect every attempt; any created window is forever excluded."""
    roots = {Path(lease['attempt_dir']).resolve() / 'session'
             for lease in queue.get('leases', {}).values() if lease['job_id'] == job['job_id']}
    attempt_root = getattr(runner.queue, 'attempts_dir', None)
    if attempt_root:
        roots.update(path.resolve() for path in (Path(attempt_root) / job['job_id']).glob('attempt-*/session'))
    if not roots:
        raise ValueError('terminal baseline job has no verifiable session attempt')
    expected = {point['name']: point for point in group['points']}
    visited, reports = {}, []
    for root in sorted(roots):
        path = root / 'completion.json'
        if not path.is_file():
            raise ValueError('attempt lacks terminal session completion: ' + str(root))
        report = read(path)
        if (report.get('group_sha256') != digest(group) or report.get('session_id') != group['session_id']
                or report.get('engine_signature') != group['engine_signature']
                or report.get('cleanup', {}).get('passed') is not True
                or report.get('cleanup', {}).get('process_cleanup_verified') is not True
                or report.get('cleanup_errors', []) or report.get('cleanup', {}).get('errors', [])):
            raise ValueError('attempt cleanup or immutable group identity is unverified: ' + str(root))
        reports.append(cc.binding(path))
        recorded = {}
        windows = root / 'windows'
        for item in report.get('windows', []):
            receipt_path = Path(item['path']).resolve()
            if (receipt_path.parent.parent != windows.resolve() or receipt_path.name != 'receipt.json'
                    or receipt_path.parent.name not in expected or item.get('point') != receipt_path.parent.name
                    or receipt_path in recorded or cc.binding(receipt_path)['sha256'] != item.get('sha256')):
                raise ValueError('completion contains a foreign, duplicate, or changed window receipt')
            recorded[receipt_path] = item
        for directory in (windows.iterdir() if windows.is_dir() else []):
            if not directory.is_dir() or directory.name not in expected:
                raise ValueError('unexpected window path in baseline attempt')
            point = expected[directory.name]
            item = dict(directory=str(directory), point_sha256=digest(point), reason='created_do_not_repeat')
            point_path = directory / 'point.json'
            if point_path.is_file():
                if read(point_path) != point:
                    raise ValueError('created point changed identity')
                item['point'] = cc.binding(point_path)
            receipt_path = directory / 'receipt.json'
            if receipt_path.is_file():
                receipt = read(receipt_path)
                if receipt.get('point_sha256') != digest(point) or receipt.get('point') != point['name']:
                    raise ValueError('created receipt changed point identity')
                item['receipt'] = cc.binding(receipt_path)
                declared = recorded.get(receipt_path.resolve())
                if declared and declared.get('sha256') != item['receipt']['sha256']:
                    raise ValueError('session completion receipt binding differs')
                item['recorded_window_complete'] = receipt.get('recorded_window_complete', False)
            visited.setdefault(point['name'], []).append(item)
    return dict(completions=reports, visited=visited)


def validate_child(parent, parent_group, child, remaining):
    group, group_ref = group_for(child)
    if group['points'] != remaining:
        raise ValueError('recovery child is not the exact ordered never-started subset')
    if {k: v for k, v in group.items() if k not in ('points', 'session_id')} != {
            k: v for k, v in parent_group.items() if k not in ('points', 'session_id')}:
        raise ValueError('recovery changed resident compatibility')
    a, b = normalized_payload(parent['payload']), normalized_payload(child['payload'])
    for key in ('argv', 'container_name', 'session_id'):
        a.pop(key, None); b.pop(key, None)
    if a != b or child['max_attempts'] != 1:
        raise ValueError('recovery changed frozen execution payload')
    old, new = list(parent['payload']['argv']), list(child['payload']['argv'])
    for flag in ('--group', '--name'):
        old[old.index(flag) + 1] = new[new.index(flag) + 1]
    if old != new:
        raise ValueError('recovery changed launch options')
    return group_ref


def prepare_subset(runner, original_id, parent, group, group_ref, remaining, evidence):
    base = runner.out / 'baseline-unstarted-recoveries' / original_id / parent['job_id']
    path, index = base, 0
    while path.exists():
        marker = path / 'prepared.json'
        if marker.is_file():
            try:
                prepared = read(marker)
            except (ValueError, OSError):
                prepared = None
            if prepared is not None:
                if (prepared.get('parent_job_sha256') != digest(parent)
                        or prepared.get('evidence') != evidence):
                    raise ValueError('existing complete recovery preparation changed authority')
                child = cc.load_bound(prepared['job'])
                validate_child(parent, group, child, remaining)
                return dict(job=prepared['job'], preparation=cc.binding(marker))
        runner.emit('partial_unstarted_recovery_preserved', path=str(path))
        index += 1
        path = base.with_name(base.name + f'-recovery-{index:04d}')
    path.mkdir(parents=True)
    subset = deepcopy(group)
    subset['points'] = deepcopy(remaining)
    subset['session_id'] = 'unstarted-baseline-' + digest(dict(
        parent_job_id=parent['job_id'], parent_group=group_ref, evidence=evidence, points=remaining))[:20]
    write_new(path / 'group.json', subset)
    child = deepcopy(parent)
    child['job_id'] = 'comparison-' + subset['model_id'].split('-')[1].lower() + '-' + digest(subset)[:16]
    child['priority'] = max(3000, parent['priority'])
    child['max_attempts'] = 1
    child['payload']['container_name'] = child['job_id']
    child['payload']['session_id'] = subset['session_id']
    argv = child['payload']['argv']
    argv[argv.index('--group') + 1] = str(path / 'group.json')
    argv[argv.index('--name') + 1] = child['job_id']
    validate_child(parent, group, child, remaining)
    write_new(path / 'job.json', child)
    prepared = dict(schema='automatic-unstarted-baseline-prepared/v1',
        parent_job_id=parent['job_id'], parent_job_sha256=digest(parent), original_job_id=original_id,
        parent_group=group_ref, evidence=evidence, job=cc.binding(path / 'job.json'),
        group=cc.binding(path / 'group.json'), point_sha256s=[digest(point) for point in remaining],
        source_unchanged=True, original_evidence_changed=False)
    write_new(path / 'prepared.json', prepared)
    return dict(job=prepared['job'], preparation=cc.binding(path / 'prepared.json'))


def run_baseline(runner, original):
    registry = registry_for(runner)
    progress = runner.state.setdefault('baseline_unstarted_recoveries', {}).setdefault(
        original['job_id'], dict(original_job_sha256=digest(original), steps={}))
    if progress['original_job_sha256'] != digest(original):
        raise ValueError('original baseline job changed after checkpoint')
    current = original
    while True:
        group, group_ref = group_for(current)
        runner.enqueue(current)
        queue = runner.wait([current['job_id']])
        entry = queue['jobs'][current['job_id']]
        if normalized_payload(entry['payload']) != normalized_payload(current['payload']):
            raise ValueError('queued baseline payload differs from bound job')
        if any(lease['job_id'] == current['job_id'] and lease.get('status') == 'active'
               for lease in queue.get('leases', {}).values()):
            raise ValueError('terminal baseline job still owns an active lease')
        try:
            evidence = session_evidence(runner, queue, current, group)
        except (ValueError, OSError, KeyError) as exc:
            progress.update(status='obstructed', current_job_id=current['job_id'], reason=str(exc))
            runner.emit('baseline_unstarted_recovery_obstructed', original_job_id=original['job_id'], **progress)
            return progress
        remaining = [point for point in group['points'] if point['name'] not in evidence['visited']]
        if not remaining:
            progress.update(status='all_points_attempted', current_job_id=current['job_id'], evidence=evidence)
            runner.emit('baseline_all_points_attempted', original_job_id=original['job_id'], current_job_id=current['job_id'])
            return progress
        if entry['status'] != 'failed' or not evidence['visited']:
            progress.update(status='obstructed', current_job_id=current['job_id'], evidence=evidence,
                            reason='no_progress' if not evidence['visited'] else 'nonfailed_terminal_with_unstarted_points')
            runner.emit('baseline_unstarted_recovery_obstructed', original_job_id=original['job_id'], **progress)
            return progress
        step = progress['steps'].get(current['job_id'])
        manual = registry.get(current['job_id'])
        if step:
            preparation = cc.load_bound(step['preparation'])
            child = cc.load_bound(step['job'])
            if preparation.get('job') != step['job']:
                raise ValueError('checkpoint preparation job binding differs')
            if step.get('adopted_from_registry'):
                if not manual or step['job'] != manual['job'] or step['preparation'] != manual['preparation']:
                    raise ValueError('checkpoint no longer matches the immutable manual registry')
            elif (preparation.get('parent_job_sha256') != digest(current)
                    or preparation.get('parent_group') != group_ref
                    or preparation.get('evidence') != evidence):
                raise ValueError('checkpoint preparation no longer matches the failed parent evidence')
        elif manual:
            preparation = cc.load_bound(manual['preparation'])
            child = cc.load_bound(manual['job'])
            if (preparation.get('original_job_id') != current['job_id']
                    or preparation.get('job') != manual['job']
                    or preparation.get('original_group') != group_ref
                    or preparation.get('original_completion') not in evidence['completions']):
                raise ValueError('manual recovery preparation is not authorized by the failed parent')
            step = dict(job=manual['job'], preparation=manual['preparation'], adopted_from_registry=True)
        else:
            step = prepare_subset(runner, original['job_id'], current, group, group_ref, remaining, evidence)
            child = cc.load_bound(step['job'])
        validate_child(current, group, child, remaining)
        existing = queue['jobs'].get(child['job_id'])
        if existing and normalized_payload(existing['payload']) != normalized_payload(child['payload']):
            raise ValueError('already-queued recovery differs; original group path must be preserved')
        progress['steps'][current['job_id']] = step
        progress.update(status='recovering', current_job_id=child['job_id'])
        runner.emit('baseline_unstarted_recovery_ready', parent_job_id=current['job_id'],
                    child_job_id=child['job_id'], job=step['job'], preparation=step['preparation'])
        current = child
