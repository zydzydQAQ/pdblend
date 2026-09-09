"""New original-policy observations using the frozen all-eight-GPU executor.

Caller owns the exclusive node lease and validates the current task handoff.
This module never restores/restarts containers or retries a failed cell.
"""
import asyncio
import copy
import hashlib
import json
import os
from pathlib import Path
import socket
import time

ROOT = Path(__file__).resolve().parent
DECLARATION_SHA = 'c51feb0af47f623dc28200abebeee6d439e7544f4cbac7e42a162549c064e529'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def require(ok, why):
    if not ok:
        raise RuntimeError(why)


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def verify_cells(declarations, model):
    require(sha(declarations) == DECLARATION_SHA, 'Rerun declaration changed')
    cells = [c for c in read(declarations)['cells'] if c['model'] == model]
    require(len(cells) == {'7b':12, '14b':12, '32b':16}[model], 'Wrong declared count')
    for cell in cells:
        require(cell['system'] == 'pdblend' and cell['seed'] == 701 and cell['repeat'] in (1, 2), 'Wrong experiment')
        require(cell['arrival_window_s'] == 100 and cell['slo_scale'] == 1 and not cell['policy_diff'], 'Policy/work altered')
        for name in ('source_binding', 'source_manifest', 'host_manifest', 'trace', 'config', 'original_receipt'):
            ref = cell[name]
            require(sha(ref['path']) == ref['sha256'], 'Frozen source changed: ' + name)
        original = cell['source_row']
        require(original['trace_sha256'] == cell['trace']['sha256'] and original['n_requests'] == cell['n_requests'], 'Trace/count differs')
        require((original['slo_ttft_s'], original['slo_tpot_s']) == (cell['slo_ttft_s'], cell['slo_tpot_s']), 'SLO differs')
    return cells


def verify_policy_binding(base, cell):
    original = read(cell['source_binding']['path'])
    require(base['model'] == cell['model'] and base['hostname'] == socket.gethostname(), 'Wrong node')
    require(base['system'] == 'pdblend' and base['host_release'] == original['host_release'], 'Original host/system differs')
    require({x['id'] for x in base['instances']} == {x['id'] for x in original['instances']}, 'Original instance set differs')
    by = {x['id']: x for x in original['instances']}
    for instance in base['instances']:
        old = by[instance['id']]
        for name in ('gpus', 'tp', 'port', 'kv_port', 'native_kind', 'service_budget_tokens'):
            require(instance.get(name) == old.get(name), 'Original instance policy differs: ' + name)
        for name, value in old['provenance'].items():
            if name != 'pid':
                require(instance['provenance'].get(name) == value, 'Imported source/model differs: ' + name)
        for name in ('id', 'name', 'image'):
            require(instance['container'][name] == old['container'][name], 'Original retained container differs')
    require(sha(cell['config']['path']) == original['files'][cell['config']['path']], 'Original config differs')


def require_held_lease():
    lock = Path('/root/workspace/pdblend/new-results/campaigns/node-experiment.lock')
    expected = lock.stat()
    for descriptor in Path('/proc/self/fd').iterdir():
        try:
            actual = os.fstat(int(descriptor.name))
            if (actual.st_dev, actual.st_ino) != (expected.st_dev, expected.st_ino):
                continue
            info = Path('/proc/self/fdinfo', descriptor.name).read_text()
            if any('FLOCK' in line and 'WRITE' in line for line in info.splitlines() if line.startswith('lock:')):
                return int(descriptor.name)
        except (FileNotFoundError, OSError, ValueError):
            continue
    raise RuntimeError('Caller must hold the actual exclusive node lease')


async def run_repeats(common, base, model, out, declarations=ROOT/'declarations.json', on_update=None):
    import aiohttp
    from ecopadg.measure.backends import PynvmlBackend
    out = Path(out)
    require_held_lease()
    cells = verify_cells(declarations, model)
    for cell in cells:
        verify_policy_binding(base, cell)
    common.validate_binding(base)
    require(not out.exists(), 'Fresh output required; no overwrite or automatic retry')
    out.mkdir(parents=True)
    write(out/'declaration-order.json', cells)
    output = out/'results'
    output.mkdir()
    state = dict(schema=1, model=model, pid=os.getpid(), phase='starting', complete=False,
                 started_s=time.time(), declared=len(cells), attempted=[], completed=[], failed=[],
                 remaining=[c['cell_id'] for c in cells], automatic_retries=False,
                 criterion='E_PDB <= E_baseline AND SLO_PDB >= min(.90,SLO_baseline)',
                 fixed_original_policy=True, independent_arrival_seeds=False,
                 source_declarations=dict(path=str(declarations), sha256=sha(declarations)))
    def update():
        state['updated_s'] = time.time()
        write(out/'status.json', state)
        if on_update:
            on_update(copy.deepcopy(state))
    update()
    try:
        hardware = await asyncio.to_thread(PynvmlBackend, power_mode='instant')
        async with aiohttp.ClientSession(trust_env=False) as session:
            for cell in cells:
                require_held_lease()
                verify_cells(declarations, model)
                require(not (ROOT/'STOP').exists(), 'Rerun STOP at cell boundary')
                require(time.time()+400 < common.GLOBAL_DEADLINE, 'Insufficient full measurement/cleanup time')
                verify_policy_binding(base, cell)
                binding = copy.deepcopy(base)
                binding.update(output=str(output), configs={cell['dataset']:cell['config']['path']},
                               priority_repeat=dict(declared_id=cell['cell_id'], repeat=cell['repeat'],
                               original_cell_id=cell['original_cell_id'], policy_diff={},
                               user_authorized_new_run=True, original_failed_attempts_unchanged=True))
                for file in (declarations, Path(__file__), Path(cell['config']['path']), Path(cell['trace']['path'])):
                    binding['files'][str(file)] = sha(file)
                binding_path = out/'bindings'/(cell['cell_id']+'.json')
                write(binding_path, binding)
                common.validate_binding(binding)
                row = copy.deepcopy(cell['source_row'])
                row.update(cell_id=cell['cell_id'], original_cell_id=cell['original_cell_id'],
                           rerun_repeat=cell['repeat'], rerun_purpose=cell['point_kind'])
                cid = row['cell_id']
                state.update(phase='running', current_cell=cid)
                state['attempted'].append(cid)
                update()
                try:
                    receipt = await common.run_one(session, binding, row, output, hardware)
                    receipt_path = output/'operations'/cid/'receipt.json'
                    artifacts = {str(f):sha(f) for directory in (receipt_path.parent, output/'cells'/cid)
                                 for f in directory.rglob('*') if f.is_file()}
                    write(output/'checkpoints'/(cid+'.json'), dict(row=row, declaration=cell,
                        binding=str(binding_path), binding_sha256=sha(binding_path),
                        receipt=str(receipt_path), receipt_sha256=sha(receipt_path), artifacts=artifacts,
                        measurement_valid=True, work_complete=receipt['summary'].get('work_complete'),
                        completed_s=time.time(), poor_slo_does_not_trigger_retry=True))
                    state['completed'].append(cid)
                except BaseException as exc:
                    state['failed'].append(dict(cell_id=cid, error=repr(exc)))
                    raise
                finally:
                    state['remaining'] = [c['cell_id'] for c in cells if c['cell_id'] not in state['attempted']]
                    update()
        state.update(phase='complete', complete=len(state['completed']) == len(cells))
    except BaseException as exc:
        state.update(phase='failed', error=repr(exc), needs_attention=True)
        raise
    finally:
        state['finished_s'] = time.time()
        state.pop('current_cell', None)
        update()
    return state
