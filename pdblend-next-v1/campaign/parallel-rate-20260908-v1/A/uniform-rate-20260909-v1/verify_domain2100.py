"""Replay native 14B shape, frequency, token, cancellation and energy evidence."""
import copy
import json
from pathlib import Path
import sys

import bootstrap as b
import power_selftest as p
import qualify_domain2100 as q


def verify(reference):
    qualification = b.checked(reference)
    assert qualification['schema'] == 'new-A-fixed14B-domain2100-qualification-v1'
    assert qualification['datasets'] == ['sharegpt', 'longbench']
    assert qualification['dynamic_capacity_qualification'] is False
    for group in ('files', 'source_files'):
        assert all(p.sha(path) == digest for path, digest in qualification[group].items())
    spec = b.checked(qualification['spec'])
    bootstrap = q.validate(spec)
    state = b.checked(qualification['status'])
    binding = b.checked(qualification['binding'])
    assert state['binding'] == qualification['binding'] and state['spec'] == qualification['spec']
    assert state['passed'] and not state.get('error') and not state['cleanup_errors']
    assert state['finished_s'] and not state['node_lease_held'] and state['clock_restore_complete']
    assert state['measurement']['measurement_valid'] and all(v['complete'] for v in state['restoration'].values())
    assert binding['instances'] == bootstrap['instances'] and binding['hostname'] == bootstrap['hostname']
    assert binding['system'] == 'pdblend' and binding['model'] == '14b' and binding['host_release'] == str(p.HOST)
    assert binding['independent_capacity_qualification_granted'] is False
    assert set(binding['configs']) == {'sharegpt', 'longbench'}
    assert all(p.sha(path) == digest for path, digest in binding['files'].items())
    for dataset, template in spec['config_templates'].items():
        expected = copy.deepcopy(b.checked(template))
        actual = p.read(binding['configs'][dataset])
        expected.update(instances=[{k: i[k] for k in ('id', 'tp', 'gpus', 'role', 'url', 'port', 'kv_port', 'container_name')}
                                   for i in bootstrap['instances']], port=34650,
                        journal=str(Path(qualification['status']['path']).parent / 'unused-controller.jsonl'),
                        host_source_release=str(p.HOST), controller_source_release=str(p.HOST), profiles=spec['profile']['path'])
        assert actual == expected, 'control policy differs from qualified template'
    sys.path[:0] = [str(p.METER), str(p.HOST / 'src')]
    from capacity_certificate import raw_measurement, close
    from capacity_backend import cancel_transfer_rows
    from ecopadg.serving.completion_policy import engine_residual
    partial = b.checked(qualification['same_node_partial'])
    previous_raw = raw_measurement(partial['measurement']['receipt'])
    raw = raw_measurement(state['measurement']['receipt'])
    assert state['started_s'] <= raw['measurement_start_s'] < raw['measurement_end_s'] <= state['finished_s']
    assert close(raw['energy_j'], state['measurement']['energy_j'])
    out = Path(qualification['status']['path']).parent
    previous_out = Path(qualification['same_node_partial']['path']).parent
    clocks = sorted(p.read(out / 'power/clocks.json') + p.read(previous_out / 'power/clocks.json'))
    instances = {i['id']: i for i in binding['instances']}
    references = {(row['instance_id'], row['input_length']): row['request'] for row in state['reference_cases']}
    expected_reference_keys = {(iid, shape[1]) for iid in instances for shape in spec['shapes']}
    assert set(references) == expected_reference_keys and len(references) == len(state['reference_cases']) == 8
    journals = {}
    for iid in instances:
        rows = [json.loads(line) for line in (out / (iid + '.requests.jsonl')).read_text().splitlines()]
        prior_journal = previous_out / (iid + '.requests.jsonl')
        if prior_journal.exists():
            rows += [json.loads(line) for line in prior_journal.read_text().splitlines()]
        journals[iid] = {row['request_id']: row for row in rows}
        assert len(journals[iid]) == len(rows)
    for (iid, length), row in references.items():
        q.stream_check(row, length, row['output_token_ids'])
        assert journals[iid][row['request_id']] == row
        old = next((r['response']['token_ids'] for r in p.read(bootstrap['ordinary']['path'])
                    if r['instance_id'] == iid and r['prompt_length'] == length), None)
        assert old is None or old == row['output_token_ids']
    cases = state['shape_cases']
    expected_cases = {(iid, *shape) for iid in instances for shape in spec['shapes']}
    actual_cases = {(case['instance_id'], case['frequency'], case['input_length'], case['batch']) for case in cases}
    assert actual_cases == expected_cases and len(cases) == len(expected_cases) == 82
    requests_seen = set()
    for case in cases:
        iid, length, frequency = case['instance_id'], case['input_length'], case['frequency']
        instance = instances[iid]
        assert len(case['requests']) == case['batch'] == len(case['loaded_clocks'])
        for row, clock in zip(case['requests'], case['loaded_clocks']):
            q.stream_check(row, length, references[(iid, length)]['output_token_ids'])
            assert row['request_id'] not in requests_seen
            requests_seen.add(row['request_id'])
            assert journals[iid][row['request_id']] == row
            assert [token for event in row['stream_events'] for token in event['event'].get('token_ids', [])] == row['output_token_ids']
            assert q.clock_window(clocks, instance['gpus'], frequency, row['token_received_s'][0],
                                  row['token_received_s'][-1]) == clock
        native = case['native_after']
        assert native['id'] == iid and not engine_residual(native, native['timestamp'])
    assert len(state['cancellations']) == 2 and {v['instance_id'] for v in state['cancellations']} == set(instances)
    for row in state['cancellations']:
        evidence = row['evidence']
        rid = evidence['request_id']
        assert evidence['verified'] and evidence['before']['active'] and evidence['before']['running']
        assert evidence['before']['kv_allocations'][rid] > 0 and evidence['cancelled']['cancelled'] == rid
        cancel_transfer_rows(evidence['cancelled']['transfers'], 1)
        assert evidence['response']['status'] >= 400 and 'cancel' in evidence['response']['body'].lower()
        assert not engine_residual(evidence['settled'], evidence['settled']['timestamp'])
    common = b.load(spec['common_executor']['path'], 'newA_saved_native_barrier')
    for iid, instance in instances.items():
        restoration = state['restoration'][iid]
        common.barrier(restoration['before'], restoration['proof'], instance)
        assert not engine_residual(restoration['resumed']['after'], restoration['resumed']['after']['timestamp'])
    terminal = p.read(out / 'isolated-observers-terminal.json')
    assert terminal['complete'] and not terminal['errors']
    hooks = b.load(p.HOOKS, 'newA_fixed_saved_observers')
    dirs = {Path(r['directory']) for r in terminal['isolated_samplers']}
    assert hooks.completed_artifacts(dirs, p.ref(p.HOST / 'manifest.json'), p.ref(p.ADAPTER)) == terminal['artifacts']
    return dict(passed=True, independently_recomputed=True, node='Anew20260909', model='14b',
                binding=qualification['binding'], qualification=reference, host_manifest=p.ref(p.HOST / 'manifest.json'),
                datasets=['sharegpt', 'longbench'], native_shape_cases=82, native_requests=len(requests_seen),
                v3_cancellations=2, raw_energy_recomputed=True, setup_energy_j=raw['energy_j'],
                dynamic_capacity_qualification=False, historical_profile_costs_recalibrated=False,
                files={str(Path(__file__)): p.sha(__file__), str(p.HERE / 'qualify_domain2100.py'): p.sha(p.HERE / 'qualify_domain2100.py')})


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('qualification', type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(p.ref(args.qualification))))
