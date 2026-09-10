"""Verify fresh new-A capacity qualification or one operational cell binding."""
import argparse
import copy
import json
from pathlib import Path
import sys
import fresh_support as f


def verify_fixed(reference, validator):
    sys.path.insert(0, str(Path(validator['path']).parent))
    f.need(f.sha(validator['path']) == validator['sha256'], 'fixed native verifier changed')
    result = f.load(validator['path'], 'fresh_capacity_fixed_verifier').verify(reference)
    f.need(result['passed'] and result['independently_recomputed'], 'fresh fixed native qualification failed')
    return result


def verify(reference):
    q = f.checked(reference)
    if q['schema'] == 'new-A-fresh-dynamic-capacity-cell-qualification-v1':
        parent = verify(q['parent_qualification'])
        old = f.checked(parent['binding'])
        binding = f.checked(q['binding'])
        f.need(q['dataset'] == 'alpaca' and q['node'] == 'Anew20260909' and q['model'] == '14b', 'cell identity differs')
        expected = copy.deepcopy(old)
        expected.update(configs=dict(alpaca=q['config']['path']), files=q['files'], output=q['measurement_output'])
        f.need(binding == expected, 'operational wrapper altered qualified binding')
        config = f.checked(q['config'])
        original_config = f.read(old['configs']['alpaca'])
        cap = f.checked(q['capacity'])
        original_cap = f.read(original_config['capacity_binding_path'])
        desired_cap = copy.deepcopy(original_cap)
        desired_cap.update(runtime_dir=q['runtime_dir'], owner_id=q['owner_id'], max_creations=8)
        f.need(cap == desired_cap and cap['calibration'] == f.checked(q['parent_qualification'])['certificate'],
               'operational capacity wrapper altered control or evidence')
        desired = dict(original_config, journal=q['journal'], capacity_binding_path=q['capacity']['path'],
                       capacity_binding_sha256=q['capacity']['sha256'], capacity_inventory_path=q['inventory_path'],
                       measurement_window_protocol='per-dataset-slo-five-system-fixed-window-v1', arrival_window_s=100.)
        f.need(config == desired and config['capacity_integration_v1'] is True, 'cell control/profile config changed')
        # The original dynamic run_one checks inventory absence immediately
        # before execution. Saved qualifications remain replayable afterward.
        f.need(all(f.sha(p) == h for p, h in q['files'].items()), 'operational file closure changed')
        return dict(passed=True, independently_recomputed=True, qualification=reference, binding=q['binding'],
                    node='Anew20260909', model='14b', datasets=['alpaca'], dynamic_capacity_qualification=True)
    f.need(q['schema'] == 'new-A-fresh-dynamic-capacity-qualification-v1', 'unknown fresh qualification')
    for key in ('files', 'source_files'):
        f.need(q[key] and all(f.sha(p) == h for p, h in q[key].items()), 'qualification evidence/source changed')
    fixed = verify_fixed(q['fixed_qualification'], q['fixed_validator'])
    state = f.checked(q['status'])
    f.need(state['complete'] and not state.get('error') and state['finished_s']
           and not state['node_lease_held'] and f.no_live_pid(state['pid'], state['startticks'])
           and state['stages'] == q['stages'] and state['binding'] == q['binding'], 'producer incomplete or alive')
    f.need([s['mode'] for s in q['stages']] == ['layout_calibration', 'automatic_underload_gate', 'qualification900'],
           'fresh stage closure incomplete')
    import audit
    for stage in q['stages']:
        actual = audit.audit_stage(stage['output'], stage['spec'], stage['mode'])
        f.need(actual == f.checked(stage['audit']), 'independent saved stage audit differs')
    from capacity_certificate import validate
    certificate = f.checked(q['certificate'])
    identity = f.checked(f.checked(q['stages'][0]['spec'])['capacity_binding'])['identity']
    validate(certificate, identity)
    binding = f.checked(q['binding'])
    f.need(binding['instances'] == f.checked(fixed['binding'])['instances']
           and binding['independent_capacity_qualification_granted'] is True
           and binding['host_release'] == str(f.HOST) and binding['model'] == '14b' and binding['system'] == 'pdblend',
           'qualified original instances or source changed')
    return dict(passed=True, independently_recomputed=True, qualification=reference, binding=q['binding'],
                node='Anew20260909', model='14b', datasets=['alpaca'], dynamic_capacity_qualification=True,
                fresh_physical_calibration=True, old_A_results_inherited=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--qualification', type=Path, required=True)
    ap.add_argument('--validator', type=Path)
    ap.add_argument('--fixed-only', action='store_true')
    ap.add_argument('--out', type=Path)
    args = ap.parse_args()
    result = verify_fixed(f.ref(args.qualification), f.ref(args.validator)) if args.fixed_only else verify(f.ref(args.qualification))
    if args.out:
        f.save(args.out, result)
    else:
        print(json.dumps(result))


if __name__ == '__main__':
    main()
