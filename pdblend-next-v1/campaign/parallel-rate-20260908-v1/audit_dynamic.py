"""Check terminal dynamic ownership and inclusion of every transition in main energy."""
import importlib.util
from pathlib import Path
from source_identity import read, sha

ROOT = Path(__file__).resolve().parent
OWNERSHIP = ROOT / 'A/dynamic-execution-until-complete-001/dynamic_ownership.py'
PIN = 'e925f22bbc99b8881891465fe436ef5afaf179abaa03c7ddb3ceecc02b23b0b6'


def inspect(point, checkpoint):
    cp = read(checkpoint)
    binding_path = cp['binding']['path'] if isinstance(cp['binding'], dict) else cp['binding']
    receipt_path = cp['receipt']['path'] if isinstance(cp['receipt'], dict) else cp['receipt']
    binding, receipt = read(binding_path), read(receipt_path)
    config = read(binding['configs'][point['dataset']])
    if config.get('capacity_integration_v1') is not True:
        return {}
    if sha(OWNERSHIP) != PIN:
        raise ValueError('independent dynamic ownership verifier changed')
    spec = importlib.util.spec_from_file_location('report_frozen_dynamic_ownership', OWNERSHIP)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.need(receipt.get('dynamic_inventory_verified') is True, 'producer did not verify dynamic inventory')
    capacity = read(config['capacity_binding_path'])
    path = Path(receipt_path).parent / 'inventory.final.json'
    module.need(cp['artifacts'].get(str(path)) == sha(path), 'terminal inventory not frozen in checkpoint')
    value = module.inventory(path, binding['instances'], child_pid=receipt['child_pid'], identity=capacity['identity'])
    module.validate_terminal(value, binding['instances'])
    artifacts = module.transition_artifacts(value)
    module.need(artifacts == receipt['dynamic_artifacts'], 'dynamic transition references differ')
    summary = receipt['summary']
    transitions = []
    active = set(value['initial_ids'])
    known = value['known_instances']
    peak = len({g for iid in active for g in known[iid]['gpus']})
    for event in value['events']:
        if event['kind'] == 'routing_commit':
            active.difference_update(event.get('removed', []))
            active.update(event.get('added', []))
            peak = max(peak, len({g for iid in active for g in known[iid]['gpus']}))
        if event['kind'] != 'transition_measurement':
            continue
        transition = read(event['receipt']['path'])
        module.need(summary['measurement_start_s'] <= transition['measurement_start_s']
                    <= transition['measurement_end_s'] <= summary['measurement_end_s'],
                    'physical conversion extends outside primary eight-GPU energy window')
        transitions.append(dict(transaction=event['transaction'], receipt=event['receipt'],
            energy_j=transition['energy_j'], duration_s=transition['duration_s'],
            overlaps_primary_energy=True, add_to_primary_energy=False))
    return dict(dynamic_inventory= dict(path=str(path), sha256=sha(path)),
        dynamic_ownership_reverified=True, all_transitions_inside_primary_energy_window=True,
        actual_peak_published_resident_gpu_count=peak, transitions=transitions,
        measured_return_to_initial_layout=True, verifier=dict(path=str(OWNERSHIP), sha256=PIN))
