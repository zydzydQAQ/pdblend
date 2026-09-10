"""Cold-start adaptation; original native bootstrap, binding and correctness helpers.

The measured startup/cleanup body is retained from the frozen restore helper.
Only the live-predecessor identity/drain stage is replaced by stopped-host proof.
"""
import asyncio
import copy
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import time

from adapters import HERE, OLD, control, p, g, R

helper = g.load(R / 'B/baseline-return-after-external-source-v1/execution.py', 'B32B_resume_cold_helper')
require, read, sha, write = helper.require, helper.read, helper.sha, helper.write
command, target_identity = helper.command, helper.target_identity
bootstrap_idle, fresh_binding, ordinary_gate = helper.bootstrap_idle, helper.fresh_binding, helper.ordinary_gate


def host_processes():
    first = Path('/proc/1/cmdline').read_bytes().split(b'\0')[0]
    require(b'codex-linux-sandbox' not in first, 'host PID namespace required for --run; sandbox /proc is insufficient')
    state = p.read(OLD / 'baseline-pipeline-001/status.json')
    old_pdb = p.read(OLD / 'pdb-performance-001/status.json')
    pids = [state['pid'], old_pdb['pid'], *[c['pid'] for c in state['children']]]
    require(all(not control.r.alive(pid) for pid in pids), 'historical owner or child is still alive')
    require(not state['complete'] and state['phase'] == 'restore' and not state['completed_systems'], 'old interruption shape changed')
    return dict(host_pid_namespace=True, pid1_executable=first.decode(), checked_dead_pids=pids,
                boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())


def validate_inventory(parent, inventory, original_inventory):
    target_identity(parent, inventory)
    original = {c['Id']: c for c in original_inventory}
    for c in inventory:
        require(c['State']['Pid'] == 0 and not c['State'].get('Paused') and not c['State'].get('Restarting'), 'target is not fully stopped')
        old = original[c['Id']]
        for key in ('Id', 'Name', 'Image', 'Path', 'Args', 'Config', 'HostConfig'):
            require(c.get(key) == old.get(key), 'retained Docker setting changed: ' + key)
        norm = lambda row: sorted(json.dumps(v, sort_keys=True) for v in row.get('Mounts', []))
        require(norm(c) == norm(old), 'retained Docker mounts changed')
    return True


def cold_snapshot(parent):
    processes = host_processes()
    def call(argv):
        return subprocess.check_output(argv, text=True, timeout=30)
    ids = [i['container']['id'] for i in parent['instances']]
    inventory = json.loads(call(['docker', 'inspect', *ids]))
    original = p.read(parent['identity_file'])
    validate_inventory(parent, inventory, original)
    running = call(['docker', 'ps', '--no-trunc', '--format', '{{.ID}}']).strip()
    require(not running, 'a container is running; exclusive cold-start requires an idle host')
    compute = call(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits']).strip()
    require(not compute, 'unowned GPU process exists')
    gpu_text = call(['nvidia-smi', '--query-gpu=index,uuid,name,memory.used,utilization.gpu', '--format=csv,noheader,nounits'])
    gpu = list(csv.reader(gpu_text.splitlines(), skipinitialspace=True))
    require(len(gpu) == 8 and {int(r[0]) for r in gpu} == set(range(8)), 'eight GPUs 0..7 required')
    require(all('L20' in r[2] and int(r[3]) == 0 and int(r[4]) == 0 for r in gpu), 'eight idle L20 GPUs required')
    return dict(schema='B32B-resume-cold-preflight-v1', captured_s=time.time(), **processes,
                target_inventory=inventory, running_container_ids=[], gpu_compute_pids=[], gpu_inventory=gpu,
                interrupted_pipeline=p.ref(OLD / 'baseline-pipeline-001/status.json'),
                interrupted_restoration=p.ref(OLD / 'baseline-restoration-001/status.json'),
                historical_commands=p.ref(OLD / 'baseline-restoration-001/commands.jsonl'),
                drain_statement='No live predecessor exists; old drain evidence is historical only.')


async def restore_cold(common, parent, snapshot, out):
    """Original measured start/fresh-binding/correctness path from an idle host.

    All previous containers are already stopped: no live identity or drain is
    claimed. Their actual historical drain evidence remains in the old attempt.
    """
    import aiohttp
    from ecopadg.measure.backends import PynvmlBackend
    from ecopadg.measure.power import PowerSampler, trapezoid_energy
    from ecopadg.metrics import clip_power_window
    from ecopadg.serving.measurement import save_raw, power_evidence
    from ecopadg.serving.backend import ClockOwner

    require(not out.exists(), 'restore attempt already exists; no automatic retry')
    out.mkdir(parents=True)
    status = dict(started_s=time.time(), complete=False, start_intents=[], stop_intents=[], errors=[], gpu_experiments_started=False)
    sampler = None
    binding = None
    started = None
    hardware = await asyncio.to_thread(PynvmlBackend, power_mode='instant')
    async with aiohttp.ClientSession(trust_env=False) as session:
        try:
            write(out / 'cold-start-preflight.json', snapshot)
            status['cold_start_preflight'] = p.ref(out / 'cold-start-preflight.json')
            status['previous_live_drain_executed'] = False
            ids = [i['container']['id'] for i in parent['instances']]
            retained = snapshot['target_inventory']
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

