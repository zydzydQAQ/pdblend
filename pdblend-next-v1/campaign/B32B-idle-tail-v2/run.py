"""One PDB-only B LongBench repeat with frozen host io-v1.2 and unchanged engines."""
import asyncio
from collections import Counter
import hashlib
import inspect
import json
from pathlib import Path
import time
from types import SimpleNamespace

import aiohttp
from ecopadg.serving.campaign import node_lease
from ecopadg.serving.cell import run_cell
from ecopadg.serving.runtime import Controller

ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT.parent / 'B32B-io-v1'
HOST = ROOT.parents[1] / 'releases/io-v1.2-runtime'
NAMES = ['pdb-next-b32q' + str(i) for i in range(4)]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(name, value):
    (ROOT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


async def inspect_containers():
    process = await asyncio.create_subprocess_exec('docker', 'inspect', *NAMES,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    output, error = await asyncio.wait_for(process.communicate(), 20)
    if process.returncode:
        raise RuntimeError(error.decode())
    return json.loads(output)


def verify_inputs():
    manifest = json.loads((ROOT / 'manifest.json').read_text())
    if digest(ROOT / 'run.py') != manifest['runner_sha256']:
        raise RuntimeError('runner identity changed')
    for path, expected in manifest['files'].items():
        if digest(path) != expected:
            raise RuntimeError('frozen source or previous evidence changed: ' + path)
    release = json.loads((HOST / 'manifest.json').read_text())
    if any(digest(HOST / p) != h for p, h in release['files'].items()):
        raise RuntimeError('frozen host source changed')
    if Path(inspect.getfile(Controller)).resolve() != HOST / 'src/ecopadg/serving/runtime.py':
        raise RuntimeError('Controller imported from unexpected host release')
    return manifest


async def main():
    if (ROOT / 'status.json').exists():
        raise RuntimeError('candidate already attempted; refusing evidence overwrite')
    status = dict(phase='preflight', complete=False, started_s=time.time(),
        scope='PDB-only host correction; original LongBench trace; no engine restart or baseline rerun',
        controller_release=str(HOST))
    write('status.json', status)
    try:
        manifest = verify_inputs()
        previous = json.loads((PREVIOUS / 'status.json').read_text())
        if previous.get('complete') is not True or previous.get('phase') != 'finished':
            raise RuntimeError('previous three-cell run must first finish')
        before = await inspect_containers()
        for row in before:
            frozen = manifest['containers'][row['Name'].lstrip('/')]
            if (not row['State']['Running'] or row['Id'] != frozen['Id']
                or row['Image'] != frozen['Image'] or row['State']['StartedAt'] != frozen['StartedAt']):
                raise RuntimeError('frozen B engine was replaced, restarted, or stopped')
        write('deployment-before.json', before)
        async with aiohttp.ClientSession(trust_env=False, timeout=aiohttp.ClientTimeout(total=10)) as session:
            states = {}
            for i in range(4):
                async with session.get(f'http://127.0.0.1:{24300+i}/runtime') as response:
                    if response.status != 200:
                        raise RuntimeError('engine runtime unavailable')
                    state = await response.json()
                if (state.get('error') or state.get('runtime_error')
                    or state.get('generation') != state.get('acknowledged_generation')
                    or any(state.get(k) for k in ('active', 'running', 'waiting',
                        'kv_allocations', 'transfer_allocations', 'transfer_buffered_tensors',
                        'transfer_inflight_receives'))):
                    raise RuntimeError('B engine not fully drained')
                states[i] = state
            write('runtime-before.json', states)
        config = json.loads((PREVIOUS / 'longbench.v1.1.config.json').read_text())
        if (config.get('independent_idle_mixed_on_stale_tail') is not True
            or config.get('park_idle') is not False or config.get('output_prior') != 211
            or config.get('allow_pd') is not False or len(config['instances']) != 4
            or any(i['role'] != 'mixed' or i['tp'] != 2 for i in config['instances'])):
            raise RuntimeError('original four-mixed policy differs')
        config.update(controller_source_release=str(HOST), journal=str(ROOT / 'unused.jsonl'))
        write('longbench.config.json', config)
        probe = Controller(config)
        try:
            if probe.planner.independent_idle_mixed_on_stale_tail is not True:
                raise RuntimeError('Controller planner fallback flag is not enabled')
            write('planner-preflight.json', dict(enabled=True,
                controller_module=inspect.getfile(Controller), output_prior=config['output_prior'],
                park_idle=config['park_idle'], engine_restart=False))
        finally:
            await probe.planning_executor.close()
        status['phase'] = 'pdblend_longbench'
        write('status.json', status)
        args = SimpleNamespace(config=ROOT / 'longbench.config.json',
            trace=PREVIOUS / 'longbench.trace.json', out=ROOT / 'cell-longbench',
            dataset='longbench', load='pilot', seed=11, split='development', strategy=None,
            freeze=None, mechanisms=None, slo_ttft_s=None, slo_tpot_s=None, timeout=120)
        result = await run_cell(args)
        status['cells'] = {'longbench': {k: result.get(k) for k in ('energy_j', 'slo_attainment',
            'work_complete', 'measurement_valid', 'completed', 'n_expected', 'runtime_error')}}
        events = [json.loads(line) for line in (ROOT / 'cell-longbench/control.jsonl').read_text().splitlines()]
        admissions = [e for e in events if e['kind'] == 'admission']
        executed = [e for e in admissions if 'fresh idle mixed fallback;' in e['plan']['reason']]
        write('fallback-execution.json', dict(executed_admissions=len(executed),
            executed_client_request_ids=[e.get('client_request_id') for e in executed],
            admission_reasons=dict(Counter(e['plan']['reason'] for e in admissions)),
            first_input_snapshots=sum(e['kind'] == 'pdb_stale_tail_fallback_snapshot' for e in events)))
        write('request-timing.json', [e for e in events if e['kind'] == 'request_timing'])
        after = await inspect_containers()
        if any(a['Id'] != b['Id'] or a['State']['StartedAt'] != b['State']['StartedAt']
               for a, b in zip(before, after)):
            raise RuntimeError('engine identity or start time changed during measurement')
        write('deployment-after.json', after)
        verify_inputs()
        if not result.get('measurement_valid'):
            raise RuntimeError('invalid measurement retained')
        status.update(phase='finished', complete=True, finished_s=time.time())
    except BaseException as exc:
        status.update(phase='failed', error=repr(exc), finished_s=time.time())
        raise
    finally:
        write('status.json', status)


if __name__ == '__main__':
    with node_lease():
        asyncio.run(main())
