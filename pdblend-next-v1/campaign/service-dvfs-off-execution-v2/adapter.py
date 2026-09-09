"""A small binding around each unchanged, previously validated node Cell.

Importing/checking this module never creates NVML or performs network/control.
"""
import argparse
import asyncio
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import socket
import sys
import time
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
PHASE = 'service_dvfs_off'


def require(ok, reason):
    if not ok:
        raise RuntimeError(reason)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def immutable(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def file_refs(refs):
    for path, digest in refs.items():
        require(sha(path) == digest, 'immutable dependency changed: ' + path)


def selected_sources(spec):
    refs = {r['reuse_main_cell_id'] for r in spec['cells'] if r['phase'] == 'scale'}
    selected = [r for r in spec['cells'] if r['phase'] == 'main' and r['cell_id'] in refs]
    require(len(refs) == len(selected) == 18, 'exact eighteen original scale1 references required')
    return selected


def candidate_row(source, index, config, digest):
    row = copy.deepcopy(source)
    row.update(sequence=index, phase=PHASE, part=PHASE,
               cell_id=source['cell_id'] + '-service-dvfs-off',
               source_main_cell_id=source['cell_id'], controller_config=str(config),
               policy_config_sha256=digest, execution_status='not_run',
               mechanism_change={'dvfs': False})
    return row


def validate_candidate(root, original):
    binding = read(root / 'binding.json')
    file_refs(binding['source_files'])
    manifest = read(root / 'package-manifest.json')
    file_refs({str(root / p): h for p, h in manifest['files'].items()})
    original.package_check()
    source = read(original.ROOT / 'runspec.json')
    spec = read(root / 'runspec.json')
    config = root / 'inputs/controller.fixed.json'
    expected = dict(read(original.CONFIG), dvfs=False)
    require(read(original.CONFIG).get('dvfs') is True and read(config) == expected,
            'only dvfs True to False is authorized; idle/capacity/budget must be identical')
    require(expected.get('manage_clocks') is True and expected.get('allow_pd') is False,
            'managed continuous mixed service is required')
    require(spec['cells'] == [candidate_row(r, i, config, sha(config))
                             for i, r in enumerate(selected_sources(source), 1)],
            'ablation rows differ from exact original main scale1 rows')
    require(spec['phase'] == PHASE and spec['model'] == source['model']
            and spec['deadline_scope'] == source['deadline_scope']
            and spec['execution_budget'] == source['execution_budget']
            and spec['execute_baselines'] is False and spec['formal_eligible'] is False,
            'phase/deadline/scope differs')
    require(spec['protocol_id'] == source['protocol_id'] and spec['measurement_schema'] == 3,
            'measurement protocol differs')
    file_refs({r['trace']: r['trace_sha256'] for r in spec['cells']})
    return spec


def source_processes(root, proc=Path('/proc')):
    """Only process reads; the lease remains the actual exclusion primitive."""
    found = []
    for path in proc.glob('[0-9]*/cmdline'):
        try:
            args = path.read_bytes().decode().split('\0')
        except (FileNotFoundError, ProcessLookupError):
            continue
        if str(root / 'run.py') in args or str(root / 'child.py') in args or (
                any(x.endswith('/bridge.py') for x in args) and str(root) in args):
            found.append(dict(pid=int(path.parent.name), args=args))
    return found


def terminal_invocation(original, phase, progress):
    paths = sorted((original.ROOT / 'invocations').glob('*.json'))
    matches = [(p, read(p)) for p in paths if read(p).get('selected_phase') == phase]
    require(matches, 'source phase has no actual invocation: ' + phase)
    path, result = matches[-1]
    require(result.get('complete') is True and result.get('phase') in ('finished', 'stopped_by_deadline')
            and result.get('baseline_preservation_verified') is True
            and result.get('checkpointed_cells') == progress['completed']
            and type(result.get('finished_s')) in (float, int)
            and math.isfinite(result['finished_s']) and result['finished_s'] <= time.time()+1,
            'source phase not cleanly terminal: ' + phase)
    if result['phase'] == 'finished':
        require(progress['phase_execution_complete'], 'finite source batch is not a finished whole phase')
    return path, result


def expected_actual(config, row, out):
    return dict(config, slo_ttft_s=row['slo_ttft_s'], slo_tpot_s=row['slo_tpot_s'],
                slo_scale=row['slo_scale'], slo_protocol='per-dataset-slo-v1',
                slo_attainment_target=.9, journal=str(out / 'control.jsonl'))


def terminal_evidence(b):
    """Invoke the original verifier; never manufacture a phase ledger or receipt."""
    original, queue = b.ORIGINAL, b.SOURCE_QUEUE
    require(not (original.ROOT / 'STOP').exists(), 'source STOP is active')
    require(not source_processes(original.ROOT), 'original bridge/runner/child still active')
    source = read(original.ROOT / 'runspec.json')
    ledger_root = Path(source['deadline_scope']['path']).parent / 'phase-ledgers' / source['model']
    files, progress, finished = {}, {}, {}
    for phase in ('main', 'scale'):
        ledger = ledger_root / (phase + '.json')
        require(ledger.is_file(), 'source phase ledger missing; never reset its clock')
        progress[phase] = queue.inspect_queue(original, phase)
        invocation, result = terminal_invocation(original, phase, progress[phase])
        phase_clock = queue.phase_record(original, source, phase)
        require(result['finished_s'] >= phase_clock['started_s'], 'source completion predates its phase')
        if result['phase'] == 'stopped_by_deadline':
            require(queue.cell_limits(source, phase_clock, now=result['finished_s']) is None,
                    'source phase claims deadline stop while another reserved cell still fits')
        finished[phase] = result['finished_s']
        files[str(ledger)] = sha(ledger)
        files[str(invocation)] = sha(invocation)
        for record in progress[phase]['records']:
            path = original.ROOT / 'checkpoints' / phase / f"{record['sequence']:04d}-{record['cell_id']}.json"
            files[str(path)] = sha(path)
            for rel, digest in record['artifacts'].items():
                files[str(original.ROOT / rel)] = digest
    require(finished['main'] <= finished['scale'], 'scale completion precedes main completion')
    available = {r['cell_id']: r for r in progress['main']['records']}
    references = {}
    for source_row in selected_sources(source):
        cid = source_row['cell_id']
        require(cid in available, 'missing verified original scale1 reference: ' + cid)
        record = available[cid]
        path = original.ROOT / 'checkpoints/main' / f"{record['sequence']:04d}-{cid}.json"
        actual = original.ROOT / 'cells' / cid / 'runtime_config.json'
        require(read(actual) == expected_actual(read(original.CONFIG), source_row, actual.parent),
                'original actual Controller config differs: ' + cid)
        files[str(actual)] = sha(actual)
        references[cid] = dict(checkpoint_path=str(path), checkpoint_sha256=sha(path),
                               actual_config_path=str(actual), actual_config_sha256=sha(actual),
                               trace_sha256=source_row['trace_sha256'])
    require(not source_processes(original.ROOT), 'source process restarted during evidence check')
    file_refs(files)
    return dict(schema=1, checked_s=time.time(), phase_finished_s=finished, files=files, references=references,
                main_completed=progress['main']['completed'], scale_completed=progress['scale']['completed'],
                source_phase_ledgers_preserved=True, low_slo_or_failed_work_not_filtered=True)


def bind(root):
    root = Path(root).resolve()
    binding = read(root / 'binding.json')
    file_refs(binding['source_files'])  # Before importing any referenced code.
    source = Path(binding['source_package'])
    sys.path.insert(0, str(source))
    original = load(source / 'run.py', '_dvfs_source_' + binding['model'])
    queue = load(Path(binding['source_queue']), '_dvfs_queue_' + binding['model'])
    execution = load(source / 'execution.py', '_dvfs_cell_' + binding['model'])
    b = SimpleNamespace(**{k: v for k, v in vars(original).items() if not k.startswith('__')})
    b.ROOT, b.CONFIG, b.ORIGINAL, b.SOURCE_QUEUE = root, root / 'inputs/controller.fixed.json', original, queue
    b.CELL, b.BINDING = execution.Cell, binding

    def write(name, value):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(path.suffix + '.tmp')
        temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
        temp.replace(path)

    def package_check():
        return validate_candidate(root, original)

    def frozen_check(full=True):
        package_check()
        source_freeze = original.frozen_check(full)
        freeze = read(root / 'freeze.json')
        require(freeze['package_manifest_sha256'] == sha(root / 'package-manifest.json')
                and freeze['source_freeze_sha256'] == sha(source / 'freeze.json')
                and freeze['policy_config_sha256'] == sha(b.CONFIG)
                and freeze['inventory'] == source_freeze['inventory'], 'ablation freeze changed')
        file_refs(freeze['terminal_evidence']['files'])
        require(not (source / 'STOP').exists(), 'source STOP active')
        return freeze

    b.write, b.package_check, b.frozen_check = write, package_check, frozen_check
    b.cell_args = lambda row, out: SimpleNamespace(config=b.CONFIG, trace=Path(row['trace']), out=out,
        dataset=row['dataset'], load=row['load'], seed=row['seed'], split='development', strategy=None,
        freeze=None, mechanisms=None, slo_ttft_s=row['slo_ttft_s'], slo_tpot_s=row['slo_tpot_s'],
        slo_scale=row['slo_scale'], timeout=120)

    def verify_actual_config(row, out):
        require(read(out / 'runtime_config.json') == expected_actual(read(b.CONFIG), row, out),
                'actual Controller config differs from sole dvfs=False treatment')
        return dict(protocol_id=read(root / 'runspec.json')['protocol_id'],
                    slo_ttft_s=row['slo_ttft_s'], slo_tpot_s=row['slo_tpot_s'], slo_scale=row['slo_scale'],
                    runtime_config_sha256=sha(out / 'runtime_config.json'))

    b.verify_actual_config = verify_actual_config
    return b


async def prepare(b):
    import aiohttp
    b.package_check()
    require(not (b.ROOT / 'freeze.json').exists(), 'prepare is immutable; do not overwrite')
    terminal = await asyncio.to_thread(terminal_evidence, b)
    source_freeze = await asyncio.to_thread(b.ORIGINAL.frozen_check, True)
    spec=read(b.ROOT/'runspec.json');deadline=read(spec['deadline_scope']['path'])
    required=sum(spec['execution_budget'].values())+420
    require(time.time()+required<=deadline['deadline_s'], 'no complete ablation cell fits before original deadline')
    async with aiohttp.ClientSession(trust_env=False) as session:
        before = await b.live(session, expected_inventory=source_freeze['inventory'])
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', read(b.CONFIG)['port']))
    file_refs(terminal['files'])
    b.write('prepare.identity.json', before)
    freeze = copy.deepcopy(source_freeze)
    freeze.update(prepared_s=time.time(), no_control_actions=True,
        package_manifest_sha256=sha(b.ROOT / 'package-manifest.json'),
        source_freeze_sha256=sha(b.ORIGINAL.ROOT / 'freeze.json'),
        policy_config_sha256=sha(b.CONFIG), terminal_evidence=terminal,
        source_contract_sha256=sha(b.ROOT / 'binding.json'), scope=PHASE)
    immutable(b.ROOT / 'freeze.json', freeze)
    return dict(prepared=True, gpu_controls=False, original_main_completed=terminal['main_completed'],
                original_scale_completed=terminal['scale_completed'])


def main(b):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('check', 'prepare', 'run', 'status'), default='check')
    parser.add_argument('--max-cells', type=int, default=1)
    args = parser.parse_args()
    require(args.max_cells > 0, 'positive cell count required')
    q = load(HERE / 'ablation_queue.py', '_ablation_queue_' + b.BINDING['model'])
    if args.phase == 'check':
        b.package_check()
        print(json.dumps(dict(package_valid=True, cells=18, gpu_executed=False, phase=PHASE)))
        return
    if args.phase == 'status':
        if not (b.ROOT/'freeze.json').exists():
            print(json.dumps(dict(prepared=False,cells=18,gpu_executed=False)))
            return
        print(json.dumps(q.inspect_queue(b), indent=2))
        return
    # Never share an inherited lease with an active supervisor: acquire this
    # node's real experiment lock independently and hold it through cleanup.
    require('PDBLEND_NODE_LOCK_FD' not in os.environ, 'inherited lease not accepted for independent ablation')
    from ecopadg.serving.campaign import node_lease

    async def work():
        task = asyncio.current_task()
        interrupted = False
        def stop():
            nonlocal interrupted
            if not interrupted:
                interrupted = True
                task.cancel()
        for sig in (signal.SIGTERM, signal.SIGINT):
            asyncio.get_running_loop().add_signal_handler(sig, stop)
        return await prepare(b) if args.phase == 'prepare' else await q.sweep(b, max_cells=args.max_cells)

    with node_lease() as lease:
        stat=os.fstat(lease.fileno())
        b.LEASE=dict(pid=os.getpid(),fd=lease.fileno(),path=lease.name,
                     inode=stat.st_ino,device=stat.st_dev,inherited=False)
        print(json.dumps(asyncio.run(work()), indent=2))
