"""Append-only B scale history and suffix execution with the original run_one.

Old completed checkpoints remain attributed to their original binding and
invocation. A stopped supervisor is never relabelled complete.
"""
import asyncio
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parent
CAMPAIGN = ROOT.parents[1]
SCALE = CAMPAIGN / 'scale-only-continuation-B32B-v1'
BASELINES = ('mixed', 'distserve', 'dynamollm', 'ecoserve')


def require(ok, why):
    if not ok:
        raise RuntimeError(why)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def ref(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path))


def read(path):
    return json.loads(Path(path).read_text())


def fixed(reference):
    require(sha(reference['path']) == reference['sha256'], 'fixed evidence changed: ' + reference['path'])
    return read(reference['path'])


def write(path, value, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, allow_nan=False) + '\n'
    if exclusive:
        with path.open('x') as stream:
            stream.write(text)
    else:
        temp = path.with_suffix(path.suffix + '.tmp')
        temp.write_text(text)
        temp.replace(path)


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def contract():
    manifest = read(SCALE / 'manifest.json')
    for name, digest in manifest['files'].items():
        require(sha(SCALE / name) == digest, 'original scale source changed')
    for path, digest in manifest['dependencies'].items():
        require(sha(path) == digest, 'original scale dependency changed')
    return load(SCALE / 'contract.py', 'priority_scale_history_contract')


def check_prefix(prefix_ref):
    prefix = fixed(prefix_ref)
    require(prefix['schema'] == 'B-clean-stopped-scale-prefix-v2'
            and prefix['complete_prefix'] is True and prefix['all_children_clean'] is True
            and prefix['whole_baseline192_complete'] is False, 'wrong boundary prefix')
    for name in ('scale_status', 'handoff_status', 'boundary_intent'):
        fixed(prefix[name])
    scale, handoff = fixed(prefix['scale_status']), fixed(prefix['handoff_status'])
    require(scale['phase'] == handoff['phase'] == 'stopped' and not scale['complete']
            and scale.get('finished_s') and handoff.get('finished_s'), 'original parents not cleanly stopped')
    for state in (scale, handoff):
        require(not Path('/proc', str(state['pid'])).exists(), 'original parent has not exited')
    require(all(s.get('complete') and s.get('exitcode') == 0 and s.get('verified_new_checkpoint')
                and not s.get('deadline_interrupt_s') and not s.get('unconfirmed_child')
                for s in scale['steps']), 'original measured prefix has a failed child')
    c = contract()
    spec = fixed(prefix['original_spec'])
    release_ref = prefix['original_release']
    fixed(release_ref)
    checked = c.check_spec(spec, release_ref['path'], release_ref['sha256'])
    by_system = {g['system']: g for g in prefix['groups']}
    require(len(by_system) == 5, 'five original system groups required')
    for group in checked['groups']:
        saved = by_system[group['group']['system']]
        require(group['reused'] == saved['completed']
                and [r['cell_id'] for r in group['pending']] == saved['pending_ids'],
                'original prefix changed after boundary; never choose newer/better outcomes')
        require(group['group'].get('scale_binding') == saved['binding'], 'original physical binding changed')
    require(all(g['group']['system'] == 'ecoserve' or not g['pending'] for g in checked['groups']),
            'only original Eco suffix is authorized by this continuation')
    return c, prefix, spec, checked


def compatibility(c, original_ref, current_ref):
    old, new = fixed(original_ref), fixed(current_ref)
    for key in ('protocol_id', 'deadline_s', 'model', 'system', 'hostname', 'large_inputs'):
        require(old[key] == new[key], 'fresh scale changed original ' + key)
    require(new['model'] == '32b' and new['system'] == 'ecoserve', 'only B Eco restart supported')
    c.source_policy(old, new, ['alpaca', 'sharegpt', 'longbench'])
    require(old['output'] != new['output'], 'new suffix must use an independent output')
    before = c.identity_index(Path(original_ref['path']).parent / 'identity.json', old)
    after = c.identity_index(Path(current_ref['path']).parent / 'identity.json', new)
    old_instances = {i['id']: i for i in old['instances']}
    require(set(old_instances) == {i['id'] for i in new['instances']}, 'instance IDs differ')
    for instance in new['instances']:
        previous = old_instances[instance['id']]
        left, right = c.core_instance(previous), c.core_instance(instance)
        left.pop('host_pid', None)
        right.pop('host_pid', None)
        require(left == right, 'original instance policy/source/layout changed')
        cid = instance['container']['id']
        require(before[cid]['State']['Pid'] != after[cid]['State']['Pid']
                and previous['container']['StartedAt'] != instance['container']['StartedAt'],
                'fresh restart needs different real host PID and StartedAt')
    return old, new


def reconcile(prefix_ref, fresh_ref, qualifier):
    c, prefix, spec, checked = check_prefix(prefix_ref)
    eco = next(g for g in checked['groups'] if g['group']['system'] == 'ecoserve')
    _, fresh = compatibility(c, eco['group']['scale_binding'], fresh_ref)
    qualification = qualifier.audit_binding(fresh_ref)
    require(qualification.get('passed') is True, 'fresh Eco qualification raw audit failed')
    old_ids = {p['cell_id'] for g in checked['groups'] for p in g['reused']}
    output = Path(fresh['output'])
    allowed = {r['cell_id'] for r in eco['pending']}
    actual = {p.stem for p in (output / 'checkpoints').glob('*.json')}
    require(actual <= allowed and not actual & old_ids, 'new output duplicates historical or undeclared work')
    completed, pending = [], []
    for row in eco['pending']:
        cid = row['cell_id']
        if cid in actual:
            completed.append(c.verify_existing(row, [fresh_ref], spec['source']['sha256'], required_binding=fresh_ref))
        else:
            require(not (output / 'operations' / cid).exists() and not (output / 'cells' / cid).exists(),
                    'uncheckpointed new attempt retained; no automatic retry')
            pending.append(row)
    return dict(contract=c, prefix=prefix, spec=spec, groups=checked['groups'], fresh=fresh,
                fresh_ref=fresh_ref, historical_ids=sorted(old_ids), completed=completed, pending=pending,
                qualification=qualification)


async def run_scale_suffix(common, fresh_ref, prefix_ref, out, qualifier, on_update=None):
    """Caller holds the actual node lease; all measurements use frozen run_one."""
    import aiohttp
    from ecopadg.measure.backends import PynvmlBackend
    helper = load(CAMPAIGN / 'main-rate-rerun-v1/repeat.py', 'priority_suffix_lease_check')
    helper.require_held_lease()
    out = Path(out)
    require(not out.exists(), 'new suffix execution journal required')
    initial = reconcile(prefix_ref, fresh_ref, qualifier)
    require(not initial['completed'], 'suffix already attempted; no implicit resume')
    binding, rows = initial['fresh'], initial['pending']
    common.validate_binding(binding)
    output = Path(binding['output'])
    require(not output.exists(), 'fresh scale output required')
    out.mkdir(parents=True)
    output.mkdir(parents=True)
    write(out / 'declaration-order.json', rows, True)
    state = dict(schema='B-original-scale-suffix-v2', pid=os.getpid(), phase='running_suffix', complete=False,
                 started_s=time.time(), attempted=[], completed=[], failed=[], remaining=[r['cell_id'] for r in rows],
                 prefix=prefix_ref, fresh_binding=fresh_ref, original_rows_unchanged=True)
    def update():
        state['updated_s'] = time.time()
        write(out / 'status.json', state)
        if on_update:
            on_update(copy.deepcopy(state))
    stopped = False
    def stop():
        nonlocal stopped
        stopped = True
    for sig in (signal.SIGTERM, signal.SIGINT):
        asyncio.get_running_loop().add_signal_handler(sig, stop)
    update()
    try:
        hardware = await asyncio.to_thread(PynvmlBackend, power_mode='instant')
        async with aiohttp.ClientSession(trust_env=False) as session:
            for row in rows:
                require(not stopped and not (ROOT / 'STOP').exists(), 'suffix stopped at cell boundary')
                require(time.time() + 400 < common.GLOBAL_DEADLINE, 'insufficient original measurement reserve')
                helper.require_held_lease()
                current = reconcile(prefix_ref, fresh_ref, qualifier)
                require(row == current['pending'][0], 'next unattempted original row changed')
                common.validate_binding(binding)
                cid = row['cell_id']
                invocation = output / 'invocations' / ('ecoserve-scale-' + str(time.time_ns()) + '.json')
                inv = dict(started_s=time.time(), system='ecoserve', phase='scale', completed=[], complete=False,
                           pid=os.getpid(), protocol_id=common.PROTOCOL, manifest_sha256=current['spec']['source']['sha256'],
                           binding_sha256=fresh_ref['sha256'], selected_datasets=['alpaca', 'sharegpt', 'longbench'],
                           current_cell=cid, continuation_prefix=prefix_ref)
                write(invocation, inv, True)
                state['attempted'].append(cid)
                state['remaining'].remove(cid)
                state['current_cell'] = cid
                update()
                try:
                    receipt = await common.run_one(session, binding, row, output, hardware)
                    receipt_path = output / 'operations' / cid / 'receipt.json'
                    artifacts = {str(p):sha(p) for folder in (receipt_path.parent, output / 'cells' / cid)
                                 for p in folder.rglob('*') if p.is_file()}
                    write(output / 'checkpoints' / (cid + '.json'), dict(row=row, receipt=str(receipt_path),
                          receipt_sha256=sha(receipt_path), completed_s=time.time(), artifacts=artifacts,
                          measurement_valid=True, work_complete=receipt['summary'].get('work_complete')), True)
                    inv.update(complete=True, completed=[cid])
                    state['completed'].append(cid)
                except BaseException as exc:
                    inv['error'] = repr(exc)
                    state['failed'].append(dict(cell_id=cid, error=repr(exc)))
                    raise
                finally:
                    inv['finished_s'] = time.time()
                    write(invocation, inv)
                    update()
                state.pop('current_cell', None)
        verified = reconcile(prefix_ref, fresh_ref, qualifier)
        require(not verified['pending'] and len(verified['completed']) == len(rows), 'suffix domain incomplete')
        state.update(complete=True, phase='suffix_complete')
    except BaseException as exc:
        state.update(phase='needs_attention', error=repr(exc))
        raise
    finally:
        state['finished_s'] = time.time()
        update()
    return state


def aggregate192(prefix_ref, fresh_ref, suffix_status_ref, qualifier):
    status = fixed(suffix_status_ref)
    require(status['complete'] and not status['failed'] and not status['remaining']
            and status.get('finished_s'), 'new suffix not clean terminal')
    history = reconcile(prefix_ref, fresh_ref, qualifier)
    require(not history['pending'], 'scale suffix remains pending')
    prefix = history['prefix']
    c = history['contract']
    release_ref = prefix['original_release']
    c.released.verify_release(release_ref['path'], release_ref['sha256'], expected_model='32b', deep=True)
    released = fixed(release_ref)['models']['32b']['records']
    main = [r['row'] for r in released if r['row']['system'] in BASELINES]
    proofs = [p for g in history['groups'] if g['group']['system'] in BASELINES for p in g['reused']]
    proofs += history['completed']
    all_rows = {r['cell_id']:r for r in fixed(history['spec']['source'])['cells']}
    scale = [all_rows[p['cell_id']] for p in proofs]
    require(len(main) == 120 and len(scale) == len({r['cell_id'] for r in scale}) == 72, 'baseline192 incomplete/duplicate')
    for system in BASELINES:
        for dataset in ('alpaca', 'sharegpt', 'longbench'):
            require(sum(r['system'] == system and r['dataset'] == dataset for r in main) == 10, 'main domain differs')
            selected = [r for r in scale if r['system'] == system and r['dataset'] == dataset]
            require(len(selected) == 6 and {r['slo_scale'] for r in selected} == {.5, 2.}, 'scale domain differs')
    return dict(schema='B-baseline192-append-only-completion-v2', complete=True, observed_s=time.time(),
                verified_baseline_main=120, verified_baseline_scale=72, total=192,
                original_stopped_parents_unchanged=True, original_release=release_ref, prefix=prefix_ref,
                fresh_binding=fresh_ref, suffix_status=suffix_status_ref, scale_checkpoint_proofs=proofs,
                previous_baseline_binding=fresh_ref, poor_slo_does_not_block_observation_completion=True)
