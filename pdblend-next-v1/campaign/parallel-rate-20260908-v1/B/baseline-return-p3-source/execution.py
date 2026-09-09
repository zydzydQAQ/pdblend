"""New ablation executor around the frozen 100-second measurement implementation.

No hardware work occurs at import. The caller must hold the existing node lease
and independently verify that the complete baseline supervisor has terminated.
"""
import asyncio
import copy
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parent
CAMPAIGN = Path('/root/workspace/pdblend-next-v1/campaign')
START_CUTOFF = 1788872770.0400891  # Explicit current user deadline
DELIVERY_CUTOFF = 1788872770.0400891  # Fixed100 original deadline
SOURCE_EXECUTOR = CAMPAIGN / 'five-system-execution-v3/run.py'


def require(ok, why):
    if not ok:
        raise RuntimeError(why)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(data)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def load_common(host):
    host = Path(host)
    sys.path[:0] = [str(host / 'src'), str(host), '/root/workspace/pdblend/.runtime-deps']
    os.environ['PYTHONPATH'] = ':'.join(sys.path[:3])
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    spec = importlib.util.spec_from_file_location('ablation_frozen_100s', SOURCE_EXECUTOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # An explicit, earlier deadline supplied by this new execution wrapper.
    module.GLOBAL_DEADLINE = DELIVERY_CUTOFF
    return module


def check_cell(cell):
    require(cell['seed'] == 701 and cell['arrival_window_s'] == 100 and cell['repeat'] in (1, 2), 'wrong ablation work')
    require(cell['start_cutoff_s'] == START_CUTOFF, 'wrong cutoff')
    for key in ('trace', 'config', 'source_binding', 'source_manifest', 'host_manifest'):
        ref = cell[key]
        require(sha(ref['path']) == ref['sha256'], 'changed declaration input: ' + key)
    row = cell['source_row']
    require(row['trace_sha256'] == cell['trace']['sha256'] and row['n_requests'] == cell['n_requests'], 'work mismatch')
    require((row['slo_ttft_s'], row['slo_tpot_s'], row['slo_scale']) ==
            (cell['slo_ttft_s'], cell['slo_tpot_s'], cell['slo_scale']), 'SLO mismatch')


def execution_row(cell):
    check_cell(cell)
    row = copy.deepcopy(cell['source_row'])
    row.update(cell_id=cell['cell_id'], ablation_arm=cell['arm'], ablation_repeat=cell['repeat'],
               original_cell_id=cell['source_row']['cell_id'], ablation_phase=cell['ablation_phase'])
    return row


async def command(argv, log, timeout=60):
    record = dict(argv=argv, started_s=time.time())
    p = await asyncio.create_subprocess_exec(*argv, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    try:
        stdout, stderr = await asyncio.wait_for(p.communicate(), timeout)
        record.update(exitcode=p.returncode, stdout=stdout.decode(errors='replace'), stderr=stderr.decode(errors='replace'))
        require(p.returncode == 0, 'command failed: ' + record['stderr'][-1500:])
        return record['stdout']
    except BaseException as exc:
        if p.returncode is None:
            p.kill()
            await p.wait()
        record['error'] = repr(exc)
        raise
    finally:
        record['finished_s'] = time.time()
        with Path(log).open('a') as f:
            f.write(json.dumps(record) + '\n')


def target_identity(parent, inventory):
    """Check only previously declared retained containers, never guessed names."""
    by_id = {r['Id']: r for r in inventory}
    require(len(by_id) == len(parent['instances']), 'wrong target container count')
    for i in parent['instances']:
        ref = i['container']
        c = by_id.get(ref['id'])
        require(c and c['Name'].lstrip('/') == ref['name'] and c['Image'] == ref['image'], 'retained target changed')
        require(c['State'].get('Running') is False, 'target unexpectedly already running')
    return by_id


async def fresh_binding(common, session, parent, out):
    """Bind actual newly started processes while retaining frozen policy/source."""
    b = copy.deepcopy(parent)
    b.update(deadline_s=DELIVERY_CUTOFF, output=str(out / 'results'),
             historical_binding_only=False, fresh_ablation_binding=True,
             experiment_scope='measured retained legacy baseline restart for new same-trace rates; fresh27 qualification pending',
             executor_wrapper=str(ROOT / 'execution.py'))
    require(b['hostname'] == socket.gethostname(), 'wrong model host')
    ids = [i['container']['id'] for i in b['instances']]
    inventory = json.loads(await command(['docker', 'inspect', *ids], out / 'commands.jsonl'))
    write(out / 'containers.after.json', inventory)
    by_id = {r['Id']: r for r in inventory}
    for i, old in zip(b['instances'], parent['instances']):
        c = by_id[i['container']['id']]
        require(c['State']['Running'] and c['State']['Pid'] > 0 and c['Image'] == old['container']['image'], 'target not running original image')
        require(c['State']['StartedAt'] != old['container']['StartedAt'], 'fresh restart identity not observed')
        actual = await common.http(session, i, '/provenance')
        for k, v in old['provenance'].items():
            if k == 'pid':
                require(type(actual.get(k)) is int and actual[k] > 0, 'actual engine provenance PID missing')
                continue
            require(actual.get(k) == v, 'original engine source/model/geometry changed: ' + k)
        i['provenance'] = {k: actual[k] for k in old['provenance']}
        i['container']['StartedAt'] = c['State']['StartedAt']
        i['host_pid'] = c['State']['Pid']
        raw = await common.wait_idle(session, i, seconds=20)
        require(not raw.get('error') and not raw.get('runtime_error'), 'fresh engine error')
    b['files'][str(out / 'containers.after.json')] = sha(out / 'containers.after.json')
    for f in (ROOT / 'execution.py', SOURCE_EXECUTOR, SOURCE_EXECUTOR.with_name('child.py')):
        b['files'][str(f)] = sha(f)
    common.validate_binding(b)
    await common.identity(session, b)
    return b


async def bootstrap_idle(common, session, instance):
    """Establish the first real ACK; never synthesize generation-zero readiness."""
    raw = await common.http(session, instance, '/runtime')
    require(raw.get('id') == instance['id'] and type(raw.get('generation')) is int,
            'fresh owner identity/generation missing')
    for key in ('active', 'running', 'waiting', 'kv_allocations', 'transfer_allocations'):
        require(key in raw and not raw[key], 'fresh owner has work or lacks observation: ' + key)
    require(not raw.get('error') and not raw.get('runtime_error'), 'fresh owner reports an error')
    generation = raw['generation'] + 1
    payload = dict(generation=generation, role='mixed', mode='continuous', admit_prefill=True, admit_decode=True)
    result = await common.http(session, instance, '/control', payload)
    after = await common.wait_idle(session, instance, seconds=20)
    require(result.get('generation') == generation and after.get('generation') == generation
            and after.get('acknowledged_generation') == generation, 'fresh control ACK not observed')
    return dict(before=raw, command=payload, after=after)


async def ordinary_gate(common, session, binding, out):
    """Real deterministic short/long output check under the service budget."""
    out.mkdir()
    evidence = dict(started_s=time.time(), passed=False, owned=[], replies=[])
    try:
        controls = await asyncio.gather(*(common.resume(session, i, i.get('service_budget_tokens')) for i in binding['instances']), return_exceptions=True)
        require(not any(isinstance(r, BaseException) for r in controls), 'correctness resume failed after all controls settled')
        for length in (128, 7168):
            answers = []
            for instance in binding['instances']:
                rid = 'ablation-correctness-' + uuid.uuid4().hex
                payload = dict(prompt=([9707, 1879, 13] * (length // 3 + 1))[:length],
                               max_tokens=64, temperature=0, top_p=1, ignore_eos=True, seed=0, stream=False)
                evidence['owned'].append(dict(instance_id=instance['id'], request_id=rid))
                write(out / 'status.json', evidence)
                import aiohttp
                async with session.post(instance['url'] + '/v1/completions', json=payload,
                        headers={'X-Request-Id': rid}, timeout=aiohttp.ClientTimeout(total=120)) as response:
                    text = await response.text()
                    require(response.status == 200, 'correctness request failed: ' + text[:500])
                    reply = json.loads(text)
                ids = reply.get('token_ids')
                require(isinstance(ids, list) and len(ids) == 64 and all(type(x) is int for x in ids), 'missing/duplicated/truncated output work')
                require(reply.get('usage', {}).get('prompt_tokens') == length and reply['usage'].get('completion_tokens') == 64, 'correctness workload changed')
                evidence['replies'].append(dict(instance_id=instance['id'], request_id=rid, prompt_length=length, response=reply))
                evidence['owned'].remove(dict(instance_id=instance['id'], request_id=rid))
                answers.append(ids)
                await common.wait_idle(session, instance)
                write(out / 'status.json', evidence)
            require(all(x == answers[0] for x in answers), 'cross-replica deterministic output differs')
        evidence['passed'] = True
    except BaseException as exc:
        evidence['error'] = repr(exc)
        raise
    finally:
        for request in evidence['owned']:
            i = next(i for i in binding['instances'] if i['id'] == request['instance_id'])
            try:
                await common.http(session, i, '/cancel', dict(request_id=request['request_id']))
            except BaseException as exc:
                evidence.setdefault('cleanup_errors', []).append(repr(exc))
        restored = await asyncio.gather(*(common.restore(session, i) for i in binding['instances']), return_exceptions=True)
        evidence['restoration'] = [dict(error=repr(r)) if isinstance(r, BaseException) else r for r in restored]
        evidence['passed'] = evidence['passed'] and all(isinstance(r, dict) and r.get('complete') for r in restored)
        evidence['finished_s'] = time.time()
        write(out / 'status.json', evidence)
    require(evidence['passed'], 'fresh correctness/cleanup gate failed')
    return evidence


async def restore_core(common, parent, previous, out):
    """Measured stop/start of exact retained containers after the baseline gate.

    There is no delete/recreate, live source patch or result erasure. On failure,
    only this operation's start intents are stopped and no serving successor runs.
    """
    import aiohttp
    from ecopadg.measure.backends import PynvmlBackend
    from ecopadg.measure.power import PowerSampler, trapezoid_energy
    from ecopadg.metrics import clip_power_window
    from ecopadg.serving.measurement import save_raw, power_evidence
    from ecopadg.serving.backend import ClockOwner

    require(time.time() + 900 < START_CUTOFF, 'insufficient time for recovery and a full paired group')
    require(not out.exists(), 'restore attempt already exists; no automatic retry')
    out.mkdir(parents=True)
    status = dict(started_s=time.time(), complete=False, start_intents=[], stop_intents=[], errors=[], gpu_experiments_started=False)
    sampler = None
    binding = None
    started = None
    hardware = await asyncio.to_thread(PynvmlBackend, power_mode='instant')
    async with aiohttp.ClientSession(trust_env=False) as session:
        try:
            # Older baseline binding retains its original deadline during its own verification.
            deadline = common.GLOBAL_DEADLINE
            common.GLOBAL_DEADLINE = previous['deadline_s']
            try:
                common.validate_binding(previous)
                write(out / 'previous.identity.json', await common.identity(session, previous))
            finally:
                common.GLOBAL_DEADLINE = deadline
            ids = [i['container']['id'] for i in parent['instances']]
            retained = json.loads(await command(['docker', 'inspect', *ids], out / 'commands.jsonl'))
            target_identity(parent, retained)
            write(out / 'containers.before.json', retained)
            # Preserve mutable owner-control/event bytes before a retained process restarts.
            for c in retained:
                argv = c['Args']
                require('--config' in argv, 'engine configuration is not explicit')
                config_path = Path(argv[argv.index('--config') + 1])
                cfg = read(config_path)
                expected = next(i for i in parent['instances'] if i['container']['id'] == c['Id'])
                require(cfg['id'] == expected['id'] and cfg['tp'] == expected['tp'], 'engine arguments changed')
                require(parent['files'].get(str(config_path)) == sha(config_path), 'engine startup config not frozen')
                directory = Path(cfg['runtime_dir'])
                for f in directory.glob(cfg['id'] + '*'):
                    if f.is_file():
                        data = f.read_bytes()
                        dest = out / 'owner-before' / cfg['id'] / f.name
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        dest.write_bytes(data)
                        status.setdefault('owner_archives', []).append(dict(path=str(f), archived=str(dest), bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
            sampler = PowerSampler(range(8), interval=.02, backend=hardware, sample_clocks=True)
            sampler.start()
            until = time.monotonic() + 5
            while len(sampler.samples) < 2:
                require(not sampler.error and time.monotonic() < until, 'restore power source unavailable')
                await asyncio.sleep(.02)
            started = time.time()
            for i in previous['instances']:
                raw = await common.wait_idle(session, i)
                proof = await common.http(session, i, '/drain', dict(expected_generation=raw['generation']))
                common.barrier(raw, proof, i)
                write(out / 'drain' / (i['id'] + '.json'), proof)
            for i in previous['instances']:
                cid = i['container']['id']
                status['stop_intents'].append(cid)
                write(out / 'status.json', status)
                await command(['docker', 'stop', '--time', '30', cid], out / 'commands.jsonl', timeout=45)
            running = (await command(['docker', 'ps', '--format', '{{.ID}}'], out / 'commands.jsonl')).split()
            require(not running, 'unrelated container appeared during exclusive recovery')
            gpu_processes = await command(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], out / 'commands.jsonl')
            require(not gpu_processes.strip(), 'unowned GPU processes remain')
            for cid in ids:
                status['start_intents'].append(cid)
                write(out / 'status.json', status)
            starts = await asyncio.gather(*(command(['docker', 'start', cid], out / 'commands.jsonl', timeout=90) for cid in ids), return_exceptions=True)
            require(not any(isinstance(r, BaseException) for r in starts), 'one or more starts failed after all start commands settled')
            until = time.monotonic() + 540
            for i in parent['instances']:
                while True:
                    try:
                        actual = await common.http(session, i, '/provenance', timeout=3)
                        require(all(actual.get(k) == v for k, v in i['provenance'].items() if k != 'pid'), 'fresh endpoint model/source differs')
                        break
                    except Exception:
                        require(time.monotonic() < until, 'retained PDB engine readiness timeout')
                        await asyncio.sleep(1)
                write(out / 'bootstrap' / (i['id'] + '.json'), await bootstrap_idle(common, session, i))
            binding = await fresh_binding(common, session, parent, out)
            write(out / 'binding.json', binding)
            status['correctness'] = await ordinary_gate(common, session, binding, out / 'correctness')
            status['complete'] = True
        except BaseException as exc:
            status['error'] = repr(exc)
            for cid in status['start_intents']:
                try:
                    await command(['docker', 'stop', '--time', '20', cid], out / 'commands.jsonl', timeout=35)
                except BaseException as stop_error:
                    status['errors'].append('owned cleanup: ' + repr(stop_error))
        finally:
            if started is not None:
                try:
                    owner = await asyncio.to_thread(ClockOwner, hardware, tuple(range(8)))
                    await asyncio.wait_for(owner.close(), 15)
                    status['clock_restore_complete'] = True
                except BaseException as exc:
                    status['errors'].append('clock restore: ' + repr(exc))
            ended = time.time()
            if sampler:
                try:
                    await asyncio.sleep(.1)
                    await asyncio.to_thread(sampler.stop)
                    power = out / 'power'
                    power.mkdir()
                    save_raw(power, [], sampler.samples, sampler.utilization_samples,
                             power_source=sampler.power_source, power_metadata=sampler.power_metadata)
                    with (power / 'clocks.csv').open('w') as f:
                        w = csv.writer(f)
                        w.writerow(['t_s'] + [f'gpu{i}_sm_mhz' for i in range(8)])
                        w.writerows([[t, *v] for t, v in sampler.frequency_samples])
                    status['power_evidence'] = power_evidence(sampler.samples, sampler.power_source, sampler.power_metadata)
                    status['sampling_error'] = sampler.error
                    status['setup_and_correctness_energy_j'] = trapezoid_energy(clip_power_window(sampler.samples, started, ended, pad_s=0)) if started else None
                except BaseException as exc:
                    status['errors'].append('setup energy: ' + repr(exc))
            status.update(measurement_start_s=started, measurement_end_s=ended, finished_s=time.time(),
                          energy_scope='eight-GPU setup including ordinary correctness; separate from serving cells')
            status['complete'] = bool(status['complete'] and not status['errors'] and not status.get('sampling_error') and status.get('power_evidence', {}).get('power_source_verified'))
            write(out / 'status.json', status)
    require(status['complete'], 'PDB restoration failed; original and new operation evidence retained')
    return binding


def bound_arm(base, cell, output):
    b = copy.deepcopy(base)
    b['configs'] = {cell['dataset']: cell['config']['path']}
    b['output'] = str(output)
    b['files'][cell['config']['path']] = cell['config']['sha256']
    b['files'][cell['trace']['path']] = cell['trace']['sha256']
    b['ablation'] = dict(arm=cell['arm'], repeat=cell['repeat'], declared_id=cell['cell_id'],
                         policy_diff=cell['config']['policy_diff'], historical_reference_reused=False)
    return b


async def run_cells(common, base, cells, out, stop_path, on_update):
    import aiohttp
    from ecopadg.measure.backends import PynvmlBackend
    hardware = await asyncio.to_thread(PynvmlBackend, power_mode='instant')
    output = out / 'results'
    output.mkdir(parents=True, exist_ok=True)
    state = dict(started_s=time.time(), completed=[], attempted=[], failed=[], remaining=[c['cell_id'] for c in cells], complete=False)
    async with aiohttp.ClientSession(trust_env=False) as session:
        started_groups = set()
        for cell in cells:
            cid = cell['cell_id']
            receipt_path = output / 'operations' / cid / 'receipt.json'
            checkpoint = output / 'checkpoints' / (cid + '.json')
            require(not receipt_path.exists() and not checkpoint.exists(), 'attempt already exists; no silent repeat or replacement')
            if time.time() >= START_CUTOFF or Path(stop_path).exists():
                state['stopped_at_boundary'] = True
                break
            group = (cell['dataset'], cell['rate_rps'], cell['point_kind'])
            if group not in started_groups:
                group_size = sum((r['dataset'], r['rate_rps'], r['point_kind']) == group for r in cells)
                # The inherited wrapper bounds preparation, service+drain and
                # cleanup at 90+220+90 seconds per run. Reserve a complete group.
                if time.time() + group_size * 400 >= START_CUTOFF:
                    state.update(stopped_at_boundary=True, stop_reason='insufficient_time_for_complete_paired_group')
                    break
                started_groups.add(group)
            check_cell(cell)
            binding = bound_arm(base, cell, output)
            binding_path = out / 'bindings' / (cid + '.json')
            write(binding_path, binding)
            common.validate_binding(binding)
            row = execution_row(cell)
            state.update(current_cell=cid, current_arm=cell['arm'])
            state['attempted'].append(cid)
            on_update(state)
            try:
                receipt = await common.run_one(session, binding, row, output, hardware)
                artifacts = {str(f): sha(f) for folder in (receipt_path.parent, output / 'cells' / cid)
                             for f in folder.rglob('*') if f.is_file()}
                write(checkpoint, dict(row=row, declaration=cell, binding=str(binding_path), binding_sha256=sha(binding_path),
                    receipt=str(receipt_path), receipt_sha256=sha(receipt_path), artifacts=artifacts,
                    completed_s=time.time(), measurement_valid=True,
                    work_complete=receipt['summary'].get('work_complete')))
                state['completed'].append(cid)
            except BaseException as exc:
                state['failed'].append(dict(cell_id=cid, error=repr(exc)))
                # A cleanup/identity failure must not be followed by another GPU job.
                state['error'] = repr(exc)
                break
            finally:
                state['remaining'] = [c['cell_id'] for c in cells if c['cell_id'] not in state['attempted']]
                on_update(state)
    state.update(finished_s=time.time(), complete=len(state['completed']) == len(cells))
    state.pop('current_cell', None)
    on_update(state)
    return state
