"""Pure saved revalidation of B32B's fresh retained-instance qualification."""
import csv
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import qualify_pdb_v1 as g
p = g.p


def idle(raw, instance):
    assert raw['id'] == instance['id']
    for key in ('active', 'running', 'waiting', 'transfer_inflight_sends', 'transfer_inflight_receives', 'transfer_buffered_tensors'):
        assert type(raw[key]) is int and raw[key] == 0, key
    assert raw['kv_allocations'] == raw['transfer_allocations'] == {}
    assert raw.get('error') is None and raw.get('runtime_error') is None
    assert raw['generation'] == raw['acknowledged_generation'] > 0
    assert raw['transfer_send_started'] == raw['transfer_send_completed'] and raw['transfer_send_failed'] == 0
    assert raw['transfer_send_healthy'] is True and raw['transfer_send_counters_observed'] is True
    assert raw['transfer_inflight_sends_observed'] is True and raw['scheduler_budget_pending'] is None
    assert raw['scheduler_budget_effective'] == {'max_num_batched_tokens': 8192, 'max_num_seqs': 32}


def verify(reference):
    """No live process, network, GPU, or Docker calls; replay original raw facts."""
    contract = p.checked(reference)
    assert contract['schema'] == 'B32B-ascending-saved-qualification-v1'
    for path, digest in contract['files'].items():
        assert p.sha(path) == digest, path
    spec = p.checked(contract['spec'])
    binding = g.validate(spec)
    state = p.checked(contract['status'])
    assert state['spec'] == contract['spec'] and state['passed'] and state['finished_s']
    assert not state['node_lease_held'] and not state.get('error') and not state['cleanup_errors']
    assert state['clock_restore_complete']
    out = Path(contract['status']['path']).parent
    clocks = p.read(out / 'power/clocks.json')
    ordinary = p.checked(spec['ordinary'])
    expected = {r['instance_id']: r['response']['token_ids'] for r in ordinary['replies'] if r['prompt_length'] == 128}
    cases = state['frequency_cases']
    assert len(cases) == 4 and {(x['instance_id'], x['loaded_clock']['target_mhz']) for x in cases} == {
        (i['id'], f) for i in binding['instances'] for f in (1500, 2520)}
    by_id = {i['id']: i for i in binding['instances']}
    for case in cases:
        instance = by_id[case['instance_id']]
        for row in (case['request'], case['warmup']):
            assert row['success'] and row['done_marker'] and row['http_status'] == 200 and not row.get('error')
            assert row['prompt_token_ids'] == ([9707, 1879, 13] * 43)[:128]
            assert row['output_token_ids'] == expected[instance['id']]
            assert len(row['output_token_ids']) == len(row['token_received_s']) == 64
            assert row['usage']['prompt_tokens'] == 128 and row['usage']['completion_tokens'] == 64
            tokens = [t for event in row['stream_events'] for t in event['event'].get('token_ids', [])]
            assert tokens == row['output_token_ids']
        r = case['request']
        assert g.clock_window(clocks, instance['gpus'], case['loaded_clock']['target_mhz'],
                              r['token_received_s'][0], r['token_received_s'][-1]) == case['loaded_clock']
        idle(case['native_after'], instance)
    g.load(Path(spec['capacity_executor']['path']), 'capacity_executor')
    cb = g.load(Path(spec['capacity_backend']['path']), 'ascending_B_saved_cancel')
    assert len(state['cancellations']) == 2 and {r['instance_id'] for r in state['cancellations']} == set(by_id)
    for item in state['cancellations']:
        instance, c = by_id[item['instance_id']], item['evidence']
        rid = c['request_id']
        assert c['before']['active'] and c['before']['running'] and c['before']['kv_allocations'].get(rid, 0) > 0
        assert c['cancelled']['cancelled'] == rid and c['response']['status'] >= 400 and 'cancel' in c['response']['body'].lower()
        cb.cancel_transfer_rows(c['cancelled']['transfers'], instance['tp'])
        idle(c['settled'], instance)
    host = Path(spec['host_manifest']['path']).parent
    sys.path[:0] = [str(host / 'src'), str(host), '/root/workspace/pdblend/.runtime-deps']
    from ecopadg.serving.measurement import power_evidence
    from ecopadg.measure.power import trapezoid_energy
    from ecopadg.metrics import clip_power_window
    for name in ('ecopadg.serving.measurement', 'ecopadg.measure.power', 'ecopadg.metrics'):
        actual = Path(sys.modules[name].__file__).resolve()
        assert contract['files'].get(str(actual)) == p.sha(actual), 'unbound measurement module'
    with (out / 'power/power.csv').open() as f:
        power = [(float(r['t_s']), [float(r[f'gpu{i}_w']) for i in range(8)]) for r in csv.DictReader(f)]
    metadata = [json.loads(line) for line in (out / 'power/power_metadata.jsonl').read_text().splitlines()]
    m = state['measurement']
    actual_evidence = power_evidence(power, p.read(out / 'power/power_source.json'), metadata)
    assert actual_evidence == m['power_evidence'] and actual_evidence['power_source_verified'] and m['measurement_valid']
    energy = trapezoid_energy(clip_power_window(power, m['measurement_start_s'], m['measurement_end_s'], pad_s=0))
    assert math.isclose(energy, m['energy_j'], rel_tol=1e-10, abs_tol=1e-6)
    before, after = p.read(out / 'identity.before.json'), p.read(out / 'identity.after.json')
    assert len(before) == len(after) == 2
    helper = g.load(g.R / 'B/baseline-return-after-external-source-v1/execution.py', 'ascending_B_saved_original')
    common = helper.load_common(host)
    for instance, left, right in zip(binding['instances'], before, after):
        for value in (left, right):
            container = value['container']
            assert container['Id'] == instance['container']['id'] and container['Image'] == instance['container']['image']
            assert container['State']['StartedAt'] == instance['container']['StartedAt'] and container['State']['Running']
            assert all(value['provenance'][k] == v for k, v in instance['provenance'].items())
            idle(value['runtime'], instance)
        restored = state['restoration'][instance['id']]
        assert restored['complete'] and not restored['errors']
        common.barrier(restored['before'], restored['proof'], instance)
        idle(restored['resumed']['after'], instance)
    return dict(passed=True, independently_recomputed=True, node='B', model='32b',
                binding=spec['binding'], actual_hostname=binding['hostname'],
                profile=spec['profile'], frequencies_mhz=[1500, 2520], tp=2,
                host_manifest=spec['host_manifest'], files=contract['files'], qualification=reference)


if __name__ == '__main__':
    print(json.dumps(verify(p.ref(sys.argv[1]))))
