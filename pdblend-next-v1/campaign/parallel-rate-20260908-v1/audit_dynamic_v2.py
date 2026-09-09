"""Dynamic physical ownership plus exact isolated-observer artifact union."""
import importlib.util
from pathlib import Path
from source_identity import read, sha
import isolated_measurement_audit_v1 as isolated

ROOT = Path(__file__).resolve().parent
OWNERSHIP = ROOT/'A/dynamic-execution-until-complete-001/dynamic_ownership.py'
PIN = 'e925f22bbc99b8881891465fe436ef5afaf179abaa03c7ddb3ceecc02b23b0b6'

def artifact_union(transitions, sampler_artifacts, declared, checkpoint_artifacts):
    union = dict(transitions)
    for path, digest in sampler_artifacts.items():
        isolated.need(path not in union or union[path] == digest, 'conflicting transition/sampler artifact')
        union[path] = digest
    isolated.need(union == declared, 'dynamic transition plus isolated sampler references differ')
    isolated.need(all(checkpoint_artifacts.get(p) == digest and sha(p) == digest for p, digest in union.items()),
                  'dynamic artifacts not frozen in checkpoint')
    return union

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
    spec = importlib.util.spec_from_file_location('report_frozen_dynamic_ownership_v2', OWNERSHIP)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    module.need(receipt.get('dynamic_inventory_verified') is True, 'producer did not verify dynamic inventory')
    capacity = read(config['capacity_binding_path'])
    path = Path(receipt_path).parent/'inventory.final.json'
    module.need(cp['artifacts'].get(str(path)) == sha(path), 'terminal inventory not frozen in checkpoint')
    value = module.inventory(path, binding['instances'], child_pid=receipt['child_pid'], identity=capacity['identity'])
    module.validate_terminal(value, binding['instances'])
    transitions_artifacts = module.transition_artifacts(value)
    adapter = dict(path=str(isolated.ADAPTER), sha256=isolated.ADAPTER_SHA)
    module.need(binding.get('isolated_power_adapter') == receipt.get('measurement_adapter') == adapter
                and binding['files'].get(adapter['path']) == adapter['sha256']
                and binding['files'].get(str(isolated.HOOK)) == isolated.HOOK_SHA
                and binding['files'].get(str(isolated.EXECUTOR)) == isolated.EXECUTOR_SHA,
                'actual isolated sampler implementation is not frozen in binding')
    host_path = str(Path(binding['host_release'])/'manifest.json')
    host = dict(path=host_path, sha256=binding['files'][host_path])
    audited = isolated.audit_samplers(receipt.get('isolated_samplers'), host, artifacts=cp['artifacts'])
    artifact_union(transitions_artifacts, audited['artifacts'], receipt['dynamic_artifacts'], cp['artifacts'])
    operation = Path(receipt_path).parent
    all_raw_roots = {str(Path(p).parent) for p in cp['artifacts']
                     if Path(p).name == 'final-raw.json' and Path(p).parent.name.startswith('sampler-')}
    module.need(all_raw_roots == set(audited['raw_values']), 'undeclared or omitted sampler stream in checkpoint')
    child_roots = {name for name in audited['raw_values'] if operation in Path(name).parents}
    outer_roots = set(audited['raw_values']) - child_roots
    module.need(len(outer_roots) == 1 and child_roots, 'per-cell outer/child sampler ownership differs')
    power_matches = [isolated.match_power_directory(operation/'power', audited, artifacts=cp['artifacts'])]
    primary = operation.parent.parent/'cells'/receipt['cell_id']
    power_matches.append(isolated.match_power_directory(primary, audited, artifacts=cp['artifacts']))
    module.need(power_matches[0]['isolated_directory'] in outer_roots
                and power_matches[1]['isolated_directory'] in child_roots, 'primary/outer observer attribution differs')
    summary = receipt['summary']; transitions = []; active = set(value['initial_ids'])
    known = value['known_instances']; peak = len({g for iid in active for g in known[iid]['gpus']})
    for event in value['events']:
        if event['kind'] == 'routing_commit':
            active.difference_update(event.get('removed', [])); active.update(event.get('added', []))
            peak = max(peak, len({g for iid in active for g in known[iid]['gpus']}))
        if event['kind'] != 'transition_measurement':
            continue
        transition = read(event['receipt']['path'])
        module.need(summary['measurement_start_s'] <= transition['measurement_start_s']
                    <= transition['measurement_end_s'] <= summary['measurement_end_s'],
                    'physical conversion extends outside primary eight-GPU energy window')
        power_matches.append(isolated.match_power_directory(Path(event['receipt']['path']).parent,
                                                            audited, artifacts=cp['artifacts']))
        transitions.append(dict(transaction=event['transaction'], receipt=event['receipt'],
                                energy_j=transition['energy_j'], duration_s=transition['duration_s'],
                                overlaps_primary_energy=True, add_to_primary_energy=False))
    module.need({p['isolated_directory'] for p in power_matches} == set(audited['raw_values']),
                'isolated observer lacks exact original measurement-file counterpart')
    return dict(dynamic_inventory=dict(path=str(path), sha256=sha(path)), dynamic_ownership_reverified=True,
                all_transitions_inside_primary_energy_window=True, actual_peak_published_resident_gpu_count=peak,
                transitions=transitions, measured_return_to_initial_layout=True,
                verifier=dict(path=str(OWNERSHIP), sha256=PIN), measurement_adapter=adapter,
                isolated_sampler_proofs=audited['proofs'], power_stream_matches=power_matches,
                transition_plus_sampler_artifacts_exact=True,
                isolated_measurement_verifier=isolated.ref(Path(isolated.__file__)))
