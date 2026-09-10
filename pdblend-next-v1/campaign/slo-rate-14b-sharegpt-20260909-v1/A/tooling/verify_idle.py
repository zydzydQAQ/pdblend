"""Independently replay fresh idle/park/reacquisition plus the fixed native qualification."""
import copy
import json
import math
from pathlib import Path
import sys

import bootstrap as b
import power_selftest as p
import qualify_fixed as q
import verify_fixed


def verify(reference):
    qualification = b.checked(reference)
    assert qualification['schema'] == 'migration-B-fixed14B-idle-qualification-v1'
    assert all(p.sha(path) == digest for kind in ('files', 'source_files')
               for path, digest in qualification[kind].items())
    prior = verify_fixed.verify(qualification['previous_qualification'])
    old_binding = b.checked(prior['binding'])
    binding = b.checked(qualification['binding'])
    state = b.checked(qualification['status'])
    spec = b.checked(qualification['spec'])
    assert spec['previous_qualification'] == qualification['previous_qualification']
    assert state['binding'] == qualification['binding'] and state['passed'] and not state.get('error')
    assert not state['node_lease_held'] and state['finished_s'] and state['measurement']['measurement_valid']
    assert binding['instances'] == old_binding['instances'] and binding['hostname'] == old_binding['hostname']
    assert all(p.sha(path) == digest for path, digest in binding['files'].items())
    out = Path(qualification['status']['path']).parent
    for dataset, path in binding['configs'].items():
        expected = copy.deepcopy(p.read(old_binding['configs'][dataset]))
        expected.update(idle_domain_reacquire_v1=True, idle_domain_reacquire_timeout_s=spec['idle_timeout_s'], journal=str(out / 'unused.jsonl'))
        assert p.read(path) == expected, 'policy changed outside existing idle-domain recovery option'
    sys.path[:0] = [str(p.METER), str(p.HOST / 'src')]
    from capacity_certificate import raw_measurement
    from ecopadg.serving.completion_policy import engine_residual
    raw = raw_measurement(state['measurement']['receipt'])
    assert state['started_s'] <= raw['measurement_start_s'] < raw['measurement_end_s'] <= state['finished_s']
    assert math.isclose(raw['energy_j'], state['measurement']['energy_j'], rel_tol=1e-8, abs_tol=1e-6)
    clocks = p.read(out / 'power/clocks.json')
    old_state = b.checked(b.checked(qualification['previous_qualification'])['status'])
    refs = {(r['instance_id'], r['input_length']): r['request']['output_token_ids'] for r in old_state['reference_cases']}
    instances = {i['id']: i for i in binding['instances']}
    assert {(r['instance_id'], r['cycle']) for r in state['probes']} == {(iid, c) for iid in instances for c in range(3)}
    assert len(state['probes']) == 6
    transitions = []
    for iid, instance in instances.items():
        journal = {r['request_id']: r for r in
                   [json.loads(line) for line in (out / (iid + '.requests.jsonl')).read_text().splitlines()]}
        events = [json.loads(line) for line in (out / (iid + '.control.clock-guard.jsonl')).read_text().splitlines()]
        for probe in [r for r in state['probes'] if r['instance_id'] == iid]:
            request = probe['request']
            q.stream_check(request, 128, refs[(iid, 128)])
            assert journal[request['request_id']] == request
            assert [token for e in request['stream_events'] for token in e['event'].get('token_ids', [])] == request['output_token_ids']
            assert request['token_received_s'][0] - request['dispatch_s'] < p.read(binding['configs']['sharegpt'])['slo_ttft_s']
            assert not engine_residual(probe['native_before'], probe['native_before']['timestamp'])
            assert not engine_residual(probe['native_after'], probe['native_after']['timestamp'])
            start, end = request['token_received_s'][0], request['token_received_s'][-1]
            loaded = [(t, f) for t, f in clocks if start <= t <= end]
            assert len(loaded) >= 2 and loaded[0][0] - start <= .25 and end - loaded[-1][0] <= .25
            assert max(y[0] - x[0] for x, y in zip(loaded, loaded[1:])) <= .25
            assert all(len(f) == 8 and all(any(abs(f[g] - target) <= 15 for target in (900,1500,2100))
                                          for g in instance['gpus']) for _, f in loaded)
            if probe['cycle']:
                assert probe['pre_request_clock_mhz'] > 2115
                actual = [e for e in events if e['kind'] == 'idle_domain_reacquisition'
                          and request['dispatch_s'] <= e['started_s'] < start]
                assert actual and all(e['confirmed'] and e['confirmed_within_idle_recovery_bound']
                    and e['request_deadline_unchanged'] and e['deadline_extended'] is False
                    and e['unprofiled_transition_predicted'] is False
                    and e['idle_domain_reacquire_timeout_s'] == spec['idle_timeout_s'] and e['original_active_settle_timeout_s'] == .3
                    and e['target_mhz'] == 2100 and e['gpus'] == instance['gpus']
                    and not e.get('physical_state_unknown') and not e.get('error') for e in actual)
                for event in actual:
                    last = event['confirmation_observations'][-instance['tp']:]
                    assert len(last) == instance['tp'] and all(abs(r['observed_mhz'] - 2100) <= 15 for r in last)
                transitions.extend(actual)
        assert state['controller_drains'][iid]['drain_complete'] and state['restoration'][iid]['complete']
    terminal = p.read(out / 'isolated-observers-terminal.json')
    assert terminal['complete'] and not terminal['errors']
    hooks=b.load(p.HOOKS,'migration_B_idle_saved_observers')
    dirs={Path(r['directory']) for r in terminal['isolated_samplers']}
    assert hooks.completed_artifacts(dirs,p.ref(p.HOST/'manifest.json'),p.ref(p.ADAPTER))==terminal['artifacts']
    result = dict(prior, binding=qualification['binding'], qualification=reference,
                  idle_domain_reacquire_v1=True, idle_cycles=4, idle_probe_requests=6,
                  idle_setup_energy_j=raw['energy_j'], independently_recomputed=True)
    result['files'].update(qualification['source_files'])
    result['files'].update(qualification['files'])
    result['files'][str(Path(__file__))] = p.sha(__file__)
    return result
