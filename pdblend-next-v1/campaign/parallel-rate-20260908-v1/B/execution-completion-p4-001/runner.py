"""Fresh, exclusive, bounded stages for the versioned improvement campaign."""
import argparse
import asyncio
import copy
import importlib.util
import os
from pathlib import Path
import signal
import socket
import sys
import time
import protocol as p

COMMON = p.REPO / 'campaign/parallel-rate-20260908-v1/common/execution-until-complete-v1/run.py'
COMMON_SHA = '77bcbbb68e20419e5bc469a838c71e1abfa789dc167d901501715fda0ff4a8a9'
DECLARATION_SHA = '653172ee53e6db739ce5b9c144a440cc3e3db50d132dae9eda0bf1a8af5c25e8'

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

def runtime(host):
    host = Path(host)
    paths = [str(host / 'src'), str(host), '/root/workspace/pdblend/.runtime-deps']
    sys.path[:0] = paths
    os.environ['PYTHONPATH'] = ':'.join(paths)
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    p.need(p.sha(COMMON) == COMMON_SHA, 'original measurement executor changed')
    return load(COMMON, 'improvement_original_fixed100')

def release_contract(path):
    release = p.read(path)
    p.need(release['schema'] == 'main-slo-improvement-release-v1' and release['approved'] is True,
           'actual approved release required')
    p.need(release['deadline_s'] is None and release['campaign_lifecycle']=='until_declared_complete_v1' and release['model'] in p.MODELS,
           'wrong model/deadline')
    declaration = p.checked(release['declaration'])
    p.need(release['declaration']['sha256'] == DECLARATION_SHA, 'work declaration changed')
    for file, digest in release['files'].items():
        p.need(p.sha(file) == digest, 'released implementation changed: ' + file)
    base = p.checked(release['binding'])
    p.need(base['model'] == release['model'] and base['system'] == 'pdblend', 'wrong actual binding')
    qualification = p.checked(release['qualification'])
    p.need(qualification.get('passed') is True, 'actual predecessor qualification incomplete')
    p.need(release['cpu_validation']['passed'] is True, 'implementation validation missing')
    cpu_evidence = p.checked(release['cpu_validation']['evidence'])
    p.need(cpu_evidence.get('passed') is True, 'referenced CPU validation failed')
    host_manifest = Path(release['host_release']) / 'manifest.json'
    cpu_model = cpu_evidence['per_model'][release['model']]
    p.need(cpu_model['manifest'] == str(host_manifest) and cpu_model['manifest_sha256'] == p.sha(host_manifest),
           'runtime differs from actual CPU-tested implementation')
    p.need(base['hostname'] == socket.gethostname(), 'release belongs to another host')
    for arm, configs in release['configs'].items():
        p.need(arm in ('fixed2', 'dynamic') and set(configs) == set(p.DATASETS), 'incomplete model-wide config')
        layouts = []
        policies = []
        for reference in configs.values():
            cfg = p.checked(reference)
            p.need(cfg['strategy'].startswith('pdblend') and cfg.get('allow_pd') is False,
                   'mixed PDB-only implementation required')
            p.need(set(cfg.get('node_gpus', ())) == set(range(8)), 'all eight measured GPUs required')
            p.need(cfg.get('admission_round_fairness') is True and cfg.get('pending_admission_capacity_guard') is True
                   and cfg.get('measured_frequency_write_guard_v1') is True
                   and cfg.get('observed_first_admission_frequency_v1') is False
                   and cfg.get('observed_idle_admission_frequency_v2') is True
                   and cfg.get('unconfirmed_retained_admission_deferral_v1') is True,
                   'declared improvement switches must be enabled')
            layouts.append([(i['id'], i['tp'], tuple(i['gpus'])) for i in cfg['instances']])
            policies.append((cfg['strategy'], cfg['profiles']))
        p.need(all(x == layouts[0] for x in layouts) and all(x == policies[0] for x in policies),
               'one instance layout and strategy/profile must apply across all datasets')
    return release, base, declaration

async def measured_ordinary(common, session, binding, hardware, out):
    """Keep every ordinary request, cleanup and its eight-board setup energy."""
    from ecopadg.measure.power import PowerSampler, trapezoid_energy
    from ecopadg.metrics import clip_power_window
    from ecopadg.serving.measurement import save_raw, power_evidence
    from ecopadg.serving.backend import ClockOwner
    executor = load(p.REPO / 'campaign/pdblend-ablation-20260908-v1/execution.py',
                    'improvement_original_ordinary_gate')
    out.mkdir()
    state = dict(passed=False, measured_gpu_count=8, started_s=time.time(), errors=[],
                 energy_scope='setup and ordinary correctness, separate from serving-cell energy')
    sampler = PowerSampler(range(8), interval=.02, backend=hardware, sample_clocks=True)
    sampler.start()
    started = None
    try:
        end_wait = time.monotonic() + 5
        while len(sampler.samples) < 2:
            p.need(not sampler.error and time.monotonic() < end_wait, 'setup power preflight failed')
            await asyncio.sleep(.01)
        started = time.time()
        await common.identity(session, binding)
        state['ordinary'] = await executor.ordinary_gate(common, session, binding, out / 'ordinary')
        await common.identity(session, binding)
        state['passed'] = True
    except BaseException as exc:
        state['errors'].append(repr(exc))
        raise
    finally:
        try:
            clocks = await asyncio.to_thread(ClockOwner, hardware, tuple(range(8)))
            await clocks.close()
            state['clock_restore_complete'] = True
        except BaseException as exc:
            state['errors'].append('clock cleanup: ' + repr(exc))
        ended = time.time()
        await asyncio.sleep(.1)
        await asyncio.to_thread(sampler.stop)
        power = out / 'power'; power.mkdir()
        save_raw(power, [], sampler.samples, sampler.utilization_samples,
                 power_source=sampler.power_source, power_metadata=sampler.power_metadata)
        state.update(measurement_start_s=started, measurement_end_s=ended,
            finished_s=time.time(), sampling_error=sampler.error,
            power_evidence=power_evidence(sampler.samples, sampler.power_source, sampler.power_metadata),
            setup_energy_j=trapezoid_energy(clip_power_window(sampler.samples, started, ended, pad_s=0))
                if started else None)
        state['passed'] = bool(state['passed'] and not state['errors'] and not sampler.error
                              and state['power_evidence']['power_source_verified'])
        p.write(out / 'status.json', state)
    p.need(state['passed'], 'ordinary setup did not complete')
    return state

def point_binding(base, release, cell, output, out):
    binding = copy.deepcopy(base)
    binding.update(host_release=release['host_release'], deadline_s=None,campaign_lifecycle='until_declared_complete_v1', output=str(output),
                   unchanged_pdb_policy=False, formal_eligible=False,
                   improvement=dict(arm=cell['arm'], repeat=cell['repeat'],
                     original_cell_id=cell['original_cell_id'], implementation=release['implementation_id']))
    config_ref = release['configs'][cell['arm']][cell['dataset']]
    config = p.checked(config_ref)
    if cell['arm'] == 'dynamic':
        p.need(release.get('dynamic_qualified') is True, 'actual dynamic qualification required')
        p.need(config.get('capacity_control') is True, 'dynamic arm is not enabled')
        config['capacity_inventory_path'] = str(out / 'inventories' / (cell['cell_id'] + '.json'))
        config_path = out / 'configs' / (cell['cell_id'] + '.json')
        p.write(config_path, config, exclusive=True)
        config_ref = p.ref(config_path)
    binding['configs'] = {cell['dataset']: config_ref['path']}
    binding['files'].update(release['files'])
    for reference in (config_ref, cell['trace'], release['declaration']):
        binding['files'][reference['path']] = reference['sha256']
    return binding

def engineering_gate(receipt):
    summary=receipt['summary']
    errors=[]
    for key in ('measurement_valid','child_stopped','clock_restore_complete'):
        if receipt.get(key) is not True: errors.append('missing '+key)
    if receipt.get('child_exitcode') != 0: errors.append('child exit error')
    for key in ('outer_cleanup_errors','sampling_error'):
        if receipt.get(key): errors.append(key)
    for key in ('runtime_error','sampling_error','outer_cleanup_errors'):
        if summary.get(key): errors.append(key)
    # Preserve every outcome, then stop expansion after any failed request.
    if int(summary.get('admission_rejections') or 0): errors.append('explicit admission rejection')
    if int(summary.get('failed_requests') or 0): errors.append('request failure')
    if int(summary.get('request_timeouts') or 0): errors.append('request timeout')
    if summary.get('work_complete') is not True: errors.append('incomplete prescribed work')
    return dict(passed=not errors,errors=errors,request_timeouts=summary.get('request_timeouts'),
                low_slo_retained=True,work_complete=summary.get('work_complete'))

def selection_skip(cell,state):
    dataset=cell['dataset'];rate=cell['source_row']['rate_rps'];mode=cell['rate_selection']
    bound=state['first_complete_breach'].get(dataset);observed=state['observed_rates'].get(dataset,[])
    if bound is not None and rate>bound:return 'above first complete SLO<0.90'
    if mode=='fill_original_at_or_below_breach' and bound is None and (not observed or rate>max(observed)):
        return 'above successful exploration; further adaptive qualification required'
    if mode=='confirm_only_if_first_repeat_measured' and rate not in observed:return 'first repeat was not selected'
    return None

async def execute(args, state):
    release, base, declaration = release_contract(args.release)
    common = runtime(release['host_release'])
    from ecopadg.serving.campaign import node_lease
    from ecopadg.measure.backends import PynvmlBackend
    import aiohttp
    p.need('PDBLEND_NODE_LOCK_FD' not in os.environ, 'fresh process must acquire its own node lease')
    cells = [c for c in declaration['cells'] if c['model'] == release['model'] and c['stage'] == args.stage]
    p.need(cells and all(c['arm'] in release['configs'] for c in cells), 'unreleased stage')
    p.need(not args.out.exists(), 'fresh stage output required; no implicit replay')
    args.out.mkdir(parents=True)
    output = args.out / 'results'; output.mkdir()
    p.write(args.out / 'release-reference.json', p.ref(args.release), exclusive=True)
    p.write(args.out / 'declaration-order.json', cells, exclusive=True)
    state.update(model=release['model'], stage=args.stage, declared=len(cells),
                 remaining=[c['cell_id'] for c in cells])
    def update():
        state['updated_s'] = time.time()
        p.write(args.out / 'status.json', state)
    update()
    with node_lease():
        state['node_lease_held'] = True
        hardware = await asyncio.to_thread(PynvmlBackend, power_mode='instant')
        async with aiohttp.ClientSession(trust_env=False) as session:
            base['deadline_s'] = None; base['campaign_lifecycle']='until_declared_complete_v1'
            common.validate_binding(base)
            state['ordinary'] = await measured_ordinary(common, session, base, hardware, args.out / 'setup')
            for cell in cells:
                dataset=cell['dataset'];rate=cell['source_row']['rate_rps'];reason=selection_skip(cell,state)
                if reason:
                    state['skipped'].append(dict(cell_id=cell['cell_id'],reason=reason));update();continue
                if args.stop_requested or (p.ROOT / 'STOP').exists():
                    state.update(phase='stopped_at_boundary', stopped_at_boundary=True)
                    break
                release_contract(args.release)
                binding = point_binding(base, release, cell, output, args.out)
                binding_path = args.out / 'bindings' / (cell['cell_id'] + '.json')
                p.write(binding_path, binding, exclusive=True)
                common.validate_binding(binding)
                row = copy.deepcopy(cell['source_row'])
                row.update(cell_id=cell['cell_id'], original_cell_id=cell['original_cell_id'],
                           improvement_arm=cell['arm'], improvement_repeat=cell['repeat'])
                state.update(phase='running', current_cell=cell['cell_id'])
                state['attempted'].append(cell['cell_id']); update()
                try:
                    executor = common
                    if cell['arm'] == 'dynamic':
                        executor = load(p.ROOT / 'dynamic_measurement.py', 'improvement_dynamic_measurement')
                    receipt = await executor.run_one(session, binding, row, output, hardware)
                    receipt_path = output / 'operations' / cell['cell_id'] / 'receipt.json'
                    artifacts = {str(f): p.sha(f) for directory in (receipt_path.parent, output / 'cells' / cell['cell_id'])
                                 for f in directory.rglob('*') if f.is_file()}
                    p.write(output / 'checkpoints' / (cell['cell_id'] + '.json'),
                        dict(row=row, declaration=cell, binding=str(binding_path), binding_sha256=p.sha(binding_path),
                            receipt=str(receipt_path), receipt_sha256=p.sha(receipt_path), artifacts=artifacts,
                            measurement_valid=True, work_complete=receipt['summary'].get('work_complete'),
                            completed_s=time.time(), poor_slo_does_not_trigger_retry=True), exclusive=True)
                    state['completed'].append(cell['cell_id'])
                    gate=engineering_gate(receipt)
                    p.write(args.out/'engineering-gates'/(cell['cell_id']+'.json'),gate,exclusive=True)
                    p.need(gate['passed'],'engineering gate stopped expansion: '+str(gate['errors']))
                    state['observed_rates'].setdefault(dataset,[]).append(rate)
                    if receipt['summary']['slo_attainment']<.90:
                        state['first_complete_breach'][dataset]=min(rate,state['first_complete_breach'].get(dataset,rate))
                except BaseException as exc:
                    # run_one writes its complete bounded cleanup receipt before
                    # raising for an invalid measurement. Preserve a checkpoint
                    # for that failed observation without marking it completed.
                    rp = output/'operations'/cell['cell_id']/'receipt.json'
                    cp = output/'checkpoints'/(cell['cell_id']+'.json')
                    if rp.exists() and not cp.exists():
                        actual = p.read(rp)
                        artifacts = {str(f):p.sha(f) for directory in (rp.parent,output/'cells'/cell['cell_id']) for f in directory.rglob('*') if f.is_file()}
                        p.write(cp,dict(row=row,declaration=cell,binding=str(binding_path),binding_sha256=p.sha(binding_path),receipt=str(rp),receipt_sha256=p.sha(rp),artifacts=artifacts,measurement_valid=actual.get('measurement_valid') is True,work_complete=actual.get('summary',{}).get('work_complete'),execution_error=repr(exc),failed_observation_preserved=True,completed_s=time.time()),exclusive=True)
                    state['failed'].append(dict(cell_id=cell['cell_id'], error=repr(exc)))
                    raise
                finally:
                    state['remaining'] = [c['cell_id'] for c in cells if c['cell_id'] not in state['attempted'] and c['cell_id'] not in {v['cell_id'] for v in state['skipped']}]
                    update()
        state['complete'] = len(state['completed'])+len(state['skipped']) == len(cells) and set(state['first_complete_breach'])==set(p.DATASETS)
        state['adaptive_followup_needed']=[d for d in p.DATASETS if d not in state['first_complete_breach']]
        if state['complete']:
            state['phase'] = 'complete'
    state['node_lease_held'] = False
    update()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--release', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--stage', required=True, choices=('screen_fixed2', 'screen_dynamic', 'confirm_dynamic'))
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args(); args.stop_requested = False
    if not args.run:
        release, _, d = release_contract(args.release)
        print(dict(cpu_only=True, model=release['model'], stage=args.stage,
                   cells=sum(c['model'] == release['model'] and c['stage'] == args.stage for c in d['cells'])))
        return
    p.need(not args.out.exists(), 'fresh stage output required; existing evidence is immutable')
    state = dict(schema=1, pid=os.getpid(), started_s=time.time(), phase='starting', complete=False,
                 attempted=[], completed=[], failed=[],skipped=[],first_complete_breach={},observed_rates={}, automatic_retries=False)
    def stop(_signum, _frame):
        args.stop_requested = True
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stop)
    try:
        asyncio.run(execute(args, state))
    except BaseException as exc:
        state.update(phase='failed', error=repr(exc))
        raise
    finally:
        if args.out.exists():
            state.update(finished_s=time.time(), node_lease_held=False)
            state.pop('current_cell', None)
            p.write(args.out / 'status.json', state)

if __name__ == '__main__':
    main()
