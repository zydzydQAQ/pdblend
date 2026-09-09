"""PDB-only two-TP2 capacity screening, fixed host v1.2 and original traces."""
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
PREVIOUS = ROOT.parent / 'B32B-capacity3-v2'
ORIGINAL = ROOT.parent / 'B32B-io-v1'
HOST = ROOT.parents[1] / 'releases/io-v1.2-runtime'
NAMES = ['pdb-next-b32q' + str(i) for i in range(4)]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(name, value):
    (ROOT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


async def command(*args):
    process = await asyncio.create_subprocess_exec(*args, stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE)
    output, error = await asyncio.wait_for(process.communicate(), 60)
    if process.returncode:
        raise RuntimeError(error.decode())
    return output.decode()


def verify_inputs():
    manifest = json.loads((ROOT / 'manifest.json').read_text())
    if digest(ROOT / 'run.py') != manifest['runner_sha256']:
        raise RuntimeError('runner identity changed')
    for path, expected in {**manifest['files'], **manifest['baseline_reference_hashes']}.items():
        if digest(path) != expected:
            raise RuntimeError('frozen input changed: ' + path)
    release = json.loads((HOST / 'manifest.json').read_text())
    if any(digest(HOST / p) != h for p, h in release['files'].items()):
        raise RuntimeError('frozen host changed')
    if Path(inspect.getfile(Controller)).resolve() != HOST / 'src/ecopadg/serving/runtime.py':
        raise RuntimeError('unexpected host import')
    return manifest


async def main():
    if (ROOT / 'status.json').exists():
        raise RuntimeError('candidate already attempted; refuse to overwrite evidence')
    status = dict(phase='prepare_two_tp2', complete=False, started_s=time.time(),
        scope='PDB-only two-TP2 capacity screening, original per-dataset priors and traces retained',
        active_gpus=list(range(4)), measured_gpus=list(range(8)), controller_release=str(HOST))
    write('status.json', status)
    try:
        manifest = verify_inputs()
        previous = json.loads((PREVIOUS / 'status.json').read_text())
        if not previous.get('complete') or previous.get('phase') != 'finished':
            raise RuntimeError('B3 capacity screening must finish first')
        before = json.loads(await command('docker', 'inspect', *NAMES))
        for row in before:
            frozen = manifest['containers'][row['Name'].lstrip('/')]
            if (row['State']['Running'] != frozen['Running'] or row['Id'] != frozen['Id']
                or row['Image'] != frozen['Image'] or row['State']['StartedAt'] != frozen['StartedAt']):
                raise RuntimeError('frozen B3 resident identity differs')
        write('deployment-before.json', before)
        async with aiohttp.ClientSession(trust_env=False, timeout=aiohttp.ClientTimeout(total=10)) as session:
            states = {}
            for i in range(3):
                async with session.get(f'http://127.0.0.1:{24300+i}/runtime') as response:
                    if response.status != 200:
                        raise RuntimeError('engine runtime unavailable')
                    state = await response.json()
                if (state.get('error') or state.get('runtime_error')
                    or state.get('generation') != state.get('acknowledged_generation')
                    or any(state.get(k) for k in ('active', 'running', 'waiting', 'kv_allocations',
                        'transfer_allocations', 'transfer_buffered_tensors', 'transfer_inflight_receives'))):
                    raise RuntimeError('B3 engine is not completely drained')
                async with session.post(f'http://127.0.0.1:{24300+i}/drain',
                        json={'expected_generation': state['generation']}) as response:
                    response.raise_for_status()
                    proof = await response.json()
                if (proof.get('drained') is not True or proof.get('accepting') is not False
                    or proof.get('generation') != state['generation'] + 1
                    or proof.get('drain_proof_type') != 'synchronous_put_owner_barrier'):
                    raise RuntimeError('fresh owner/rank drain proof unavailable')
                states[i] = dict(runtime=state, drain=proof)
            write('runtime-before.json', states)
        template = json.loads((PREVIOUS / 'longbench.config.json').read_text())
        template['instances'] = template['instances'][:2]
        if (template.get('output_prior') != 211 or template.get('park_idle') is not False
            or template.get('independent_idle_mixed_on_stale_tail') is not True
            or template.get('allow_pd') is not False or template.get('slo_attainment_target') != .9
            or template.get('node_gpus') != list(range(8))
            or [i['gpus'] for i in template['instances']] != [[0, 1], [2, 3]]
            or any(i['role'] != 'mixed' or i['tp'] != 2 for i in template['instances'])):
            raise RuntimeError('two-TP2 fixed policy differs')
        template['journal'] = str(ROOT / 'unused.jsonl')
        write('controller.template.json', template)
        probe = Controller(template)
        try:
            if probe.planner.independent_idle_mixed_on_stale_tail is not True:
                raise RuntimeError('actual planner fallback is disabled')
            write('planner-preflight.json', dict(enabled=True, output_priors=manifest['output_priors'], park_idle=False,
                module=inspect.getfile(Controller), measured_gpus=list(range(8))))
        finally:
            await probe.planning_executor.close()
        stopped = await command('docker', 'stop', '-t', '10', 'pdb-next-b32q2')
        write('removed-residency.json', dict(stopped_preserved=['pdb-next-b32q2'],
            deleted=[], at_s=time.time(), stdout=stopped))
        stopped_state = json.loads(await command('docker', 'inspect', 'pdb-next-b32q2'))[0]
        if stopped_state['State']['Running']:
            raise RuntimeError('removed PDB candidate is still running')
        physical = await command('nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits')
        memory = {int(line.split(',')[0]): int(line.split(',')[1]) for line in physical.strip().splitlines()}
        if any(memory[gpu] != 0 for gpu in (4, 5, 6, 7)):
            raise RuntimeError('removed/unused GPU still has residency')
        write('gpu-residency-after-removal.json', memory)
        status['phase'] = 'pdblend_development_cells'
        write('status.json', status)
        for dataset in ('longbench', 'alpaca', 'sharegpt'):
            config = json.loads((ORIGINAL / (dataset + '.v1.1.config.json')).read_text())
            config.update(instances=template['instances'], controller_source_release=str(HOST),
                journal=str(ROOT / f'unused-{dataset}.jsonl'))
            if config['output_prior'] != manifest['output_priors'][dataset]:
                raise RuntimeError('original per-dataset output prior differs')
            write(dataset + '.config.json', config)
            status['dataset'] = dataset
            write('status.json', status)
            args = SimpleNamespace(config=ROOT / (dataset + '.config.json'),
                trace=ORIGINAL / (dataset + '.trace.json'), out=ROOT / ('cell-' + dataset),
                dataset=dataset, load='pilot', seed=11, split='development', strategy=None,
                freeze=None, mechanisms=None, slo_ttft_s=None, slo_tpot_s=None, timeout=120)
            result = await run_cell(args)
            status.setdefault('cells', {})[dataset] = {k: result.get(k) for k in ('energy_j',
                'slo_attainment', 'energy_per_good_request_j', 'work_complete', 'measurement_valid',
                'completed', 'n_expected', 'runtime_error')}
            events = [json.loads(line) for line in
                (ROOT / ('cell-' + dataset) / 'control.jsonl').read_text().splitlines()]
            admissions = [e for e in events if e['kind'] == 'admission']
            write(dataset + '.fallback-execution.json', dict(
                executed_admissions=sum('fresh idle mixed fallback;' in e['plan']['reason'] for e in admissions),
                admission_reasons=dict(Counter(e['plan']['reason'] for e in admissions))))
            write('status.json', status)
            # Low SLO or incomplete work remains an observed candidate outcome;
            # finish all three requested traces without changing the threshold.
        after = json.loads(await command('docker', 'inspect', *NAMES))
        if any(a['Id'] != b['Id'] or a['State']['StartedAt'] != b['State']['StartedAt']
               for a, b in zip(before, after)):
            raise RuntimeError('engine identity/start time changed during screening')
        write('deployment-after.json', after)
        verify_inputs()
        status.update(phase='finished', complete=True, finished_s=time.time())
    except BaseException as exc:
        status.update(phase='failed', error=repr(exc), finished_s=time.time())
        raise
    finally:
        write('status.json', status)


if __name__ == '__main__':
    with node_lease():
        asyncio.run(main())
