"""Allocate new cell-local inventory paths without changing the qualified policy."""
import copy
import hashlib
from pathlib import Path
import sys
import fresh_support as f
import verify


def prepare_dispatch(*, rows, **kwargs):
    f.need(len(rows) == 1 and rows[0]['system'] == 'pdblend' and rows[0]['dataset'] == 'alpaca'
           and rows[0]['model'] == '14b' and rows[0]['node'] == 'Anew20260909', 'one genuine new-A Alpaca cell required')
    row = rows[0]
    qualification_ref = kwargs['qualification']
    proof = verify.verify(qualification_ref)
    parent = f.checked(qualification_ref)
    binding = f.checked(proof['binding'])
    config = f.read(binding['configs']['alpaca'])
    capacity = f.read(config['capacity_binding_path'])
    release_out = Path(kwargs['out']).resolve()
    cellroot = release_out.parent / 'dynamic-cell'
    f.need(not cellroot.exists(), 'fresh dynamic cell input directory required')
    measurement_out = release_out.parent / 'measurement'
    owner = 'uniformcap' + hashlib.sha256(row['cell_id'].encode()).hexdigest()[:12]
    runtime = str(cellroot / 'runtime')
    inventory = str(cellroot / 'inventory.json')
    journal = str(cellroot / 'unused-controller.jsonl')
    capacity.update(runtime_dir=runtime, owner_id=owner, max_creations=8)
    capref = f.save(cellroot / 'capacity-binding.json', capacity)
    config.update(journal=journal, capacity_binding_path=capref['path'], capacity_binding_sha256=capref['sha256'],
                  capacity_inventory_path=inventory,
                  measurement_window_protocol='per-dataset-slo-five-system-fixed-window-v1', arrival_window_s=100.)
    cfgref = f.save(cellroot / 'config.json', config)
    files = dict(binding['files'])
    files.update(parent['files'])
    files.update(parent['source_files'])
    for reference in (qualification_ref, capref, cfgref, f.ref(__file__), f.ref(f.HERE / 'verify.py')):
        f.add(files, reference)
    binding.update(configs=dict(alpaca=cfgref['path']), files=files, output=str(measurement_out))
    binding_ref = f.save(cellroot / 'binding.json', binding)
    qref = f.save(cellroot / 'qualified.json', dict(schema='new-A-fresh-dynamic-capacity-cell-qualification-v1',
        parent_qualification=qualification_ref, node='Anew20260909', model='14b', dataset='alpaca',
        cell_id=row['cell_id'], binding=binding_ref, config=cfgref, capacity=capref,
        measurement_output=str(measurement_out), owner_id=owner, runtime_dir=runtime,
        inventory_path=inventory, journal=journal, files=files, source_files={str(f.HERE / 'fresh_support.py'): f.sha(f.HERE / 'fresh_support.py')}))
    kwargs.update(qualification=qref, qualification_validator=f.ref(f.HERE / 'verify.py'),
                  measurement_executor=f.ref(f.A / 'dynamic-execution-isolated-power-002/dynamic_measurement.py'))
    shared = f.ROOT / 'C/uniform-rate-20260909-v1'
    sys.path.insert(0, str(shared))
    prepare = f.load(shared / 'prepare_release.py', 'newA_dynamic_original_prepare')
    return prepare.prepare(**kwargs)
