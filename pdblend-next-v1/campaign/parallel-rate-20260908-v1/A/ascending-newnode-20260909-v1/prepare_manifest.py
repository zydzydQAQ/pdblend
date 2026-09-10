"""Freeze fresh-node 14B work and byte-identical workloads; never launch hardware."""
from __future__ import annotations

import copy
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import importlib.util
import json
from pathlib import Path
import random

HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE.parents[1]
REPO = CAMPAIGN.parents[1]
PARENT = CAMPAIGN / 'common/ascending-rate-execution-v2/release-001/declaration.json'
NODE = 'Anew20260909'
HOSTNAME = 'iZwz9274emxme9019d2sjgZ'
UUIDS = [
    'GPU-3c2e8b60-fd83-bd03-21fe-0f8363a26ea8',
    'GPU-8970ab2f-c6e6-d46f-be88-28c731ca746a',
    'GPU-c0c37c25-f790-5a46-20ee-f1af1032af2e',
    'GPU-84065575-f5fc-cacd-f13b-3d4f657f01b3',
    'GPU-445db269-c202-c48e-200d-4a17731f869b',
    'GPU-7ae5a8a7-183f-3279-7eb8-a44cf7e55317',
    'GPU-7bb03f3c-e196-0094-0fa6-10bc94877dfa',
    'GPU-220ffd7e-2c4a-3ff5-d8fe-92b70445c856',
]
BUS_IDS = ['00000000:89:00.0', '00000000:8A:00.0', '00000000:D1:00.0',
           '00000000:D2:00.0', '00000001:89:00.0', '00000001:8A:00.0',
           '00000001:D1:00.0', '00000001:D2:00.0']
SYSTEMS = ['pdblend', 'mixed', 'distserve', 'dynamollm', 'ecoserve']
DATASETS = ['alpaca', 'sharegpt', 'longbench']


def encode(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n').encode()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def ref(path):
    return {'path': str(path), 'sha256': sha(path)}


def read(path):
    return json.loads(Path(path).read_text())


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fresh_write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(payload)


def save(path, value):
    fresh_write(path, (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode())


def reconstruct_missing(parent_positions):
    """Use the original reader for 14B only; unavailable 7B data is irrelevant."""
    fixed = load(REPO / 'campaign/five-system-fixed-window-v1/generate.py', 'newnode_fixed_window')
    old = fixed.parent_generator()
    spec_path = REPO / 'campaign/A14B-deadline-matrix-v1/source/spec.json'
    spec = old.read(spec_path)
    pool_spec = old.read(old.frozen(spec['frozen_development_pool_declaration']))
    reader = load(old.frozen(pool_spec['development_pool_reader']), 'newnode_original_14b_pool')
    entry = pool_spec['models']['14b']
    policy = old.read(old.frozen(entry['policy_reference']))
    groups = {}
    for dataset, ds in entry['datasets'].items():
        records, pool = reader.load_pool(ds['pool'], model='14b', dataset=dataset)
        order = list(range(len(records)))
        random.Random(pool_spec['sampling_seed']).shuffle(order)
        groups[dataset] = dict(records=records, pool=pool, order=order,
            record_digests=[old.digest(dict(prompt=r['prompt'], input_tokens=r['input_tokens'], output_tokens=r['output_tokens'])) for r in records],
            anchor=Decimal(str(ds['anchor_rps'])), policy=entry['policy_reference'], strategy=policy['strategy'])
    recovered = {}
    for pos in parent_positions:
        if Path(pos['trace']['path']).is_file():
            continue
        _, trace300 = old.build_trace(spec, '14b', pos['dataset'], Decimal(pos['rate_rps_decimal']), 701,
                                     groups[pos['dataset']], pool_spec['sampling_seed'])
        source = pos['workload']['source_300s_trace']
        trace100 = fixed.prefix_trace(trace300, source)
        raw300, raw100 = old.encode(trace300), fixed.encode(trace100)
        assert hashlib.sha256(raw300).hexdigest() == source['sha256']
        assert hashlib.sha256(raw100).hexdigest() == pos['trace']['sha256']
        recovered[pos['position_id']] = (raw300, raw100)
    return recovered


def main():
    assert not (HERE / 'declaration.json').exists(), 'Do not overwrite the frozen new-node declaration'
    parent = read(PARENT)
    assert ref(PARENT) == read(PARENT.parent / 'manifest.json')['declaration']
    parent_positions = [p for p in parent['positions'] if p['node'] == 'A']
    assert len(parent_positions) == 32
    recovered = reconstruct_missing(parent_positions)
    assert len(recovered) == 2
    identity = dict(node=NODE, logical_assignment='A', actual_hostname=HOSTNAME,
        old_A_hostname='iZwz92bdfqihqp38tekqjyZ', old_A_identity_not_equivalent=True,
        public_address='120.79.123.62', private_addresses=['172.16.50.106', '172.16.50.107'],
        ssh_user='root', ssh_host_key_alias='120.79.123.62',
        observed_by_read_only_ssh_on='2026-09-09', observed_gpu_driver='580.126.09',
        GPUs=[dict(index=i, uuid=uuid, pci_bus_id=BUS_IDS[i], name='NVIDIA L20', memory_mib=46068)
              for i, uuid in enumerate(UUIDS)],
        expected_gpu_indices=list(range(8)), whole_node_measurement_required=True,
        hardware_qualification_granted=False)
    save(HERE / 'node-identity.json', identity)
    positions, cells, pairings, aliases = [], [], [], []
    for parent_pos in parent_positions:
        dataset, rate = parent_pos['dataset'], parent_pos['rate_rps_decimal']
        position_id = f'ascending-newnode-v1-{NODE}-14b-{dataset}-r{rate}-s701'
        src100 = parent_pos['trace']
        src300 = parent_pos['workload']['source_300s_trace']
        if parent_pos['position_id'] in recovered:
            raw300, raw100 = recovered[parent_pos['position_id']]
        else:
            raw100, raw300 = Path(src100['path']).read_bytes(), Path(src300['path']).read_bytes()
        assert hashlib.sha256(raw100).hexdigest() == src100['sha256']
        assert hashlib.sha256(raw300).hexdigest() == src300['sha256']
        path100 = HERE / 'traces' / f'14b-{dataset}-r{rate}-s701-w100.json'
        path300 = HERE / 'source300' / f'14b-{dataset}-r{rate}-s701-w300.json'
        fresh_write(path100, raw100)
        fresh_write(path300, raw300)
        trace = json.loads(raw100)
        aliases += [dict(original=src100, local_copy=ref(path100), bytes_unchanged=True),
                    dict(original=src300, local_copy=ref(path300), bytes_unchanged=True)]
        tasks = {}
        for system in SYSTEMS:
            repeats = sorted({task['repeat'] for task in parent_pos['systems'][system]})
            assert len(repeats) == len(parent_pos['systems'][system])
            tasks[system] = []
            for repeat in repeats:
                cid = f'{position_id}-w100-{system}-slo1-repeat{repeat}'
                row = copy.deepcopy(parent_pos['workload'])
                row.update(cell_id=cid, node=NODE, measurement_host=NODE, expected_actual_hostname=HOSTNAME,
                    system=system, repeat=repeat, sequence=len(cells)+1, phase='main', part='main', slo_scale=1.,
                    allowed_slo_scales=[1.], trace=str(path100), trace_path=str(path100),
                    trace_reference=ref(path100), trace_sha256=src100['sha256'],
                    source_300s_trace=ref(path300), original_source_300s_trace=src300,
                    source_parent_position_id=parent_pos['position_id'], source_parent_declaration=ref(PARENT),
                    workload_provenance_only_not_prior_hardware_qualification=True,
                    execution_status='not_run', actual_fixed_window_verified=False,
                    execution_binding_required=True, policy_binding_required=True, controller_config=None,
                    strategy=None, request_hard_timeout_s=120., drain_after_arrival_window_s=120.,
                    fixed_slo_only=True, formal_eligible=False, reused_historical_observation=False,
                    slo_ttft_s=trace['slo']['ttft_s'], slo_tpot_s=trace['slo']['tpot_s'], slo_attainment_target=.9,
                    n_requests=trace['n_requests'], content_pairing_sha256=trace['content_pairing_sha256'])
                cells.append(row)
                tasks[system].append(dict(action='execute', cell_id=cid, repeat=repeat))
        position = dict(position_id=position_id, node=NODE, model='14b', dataset=dataset,
            rate_rps=parent_pos['rate_rps'], rate_rps_decimal=rate, order=parent_pos['order'],
            trace=ref(path100), source300=ref(path300), source_parent_position_id=parent_pos['position_id'],
            systems=tasks, conditional_on_no_lower_rate_cap=True,
            required_pdb_repeats=copy.deepcopy(parent_pos['required_pdb_repeats']))
        positions.append(position)
        for pt in tasks['pdblend']:
            for system in SYSTEMS[1:]:
                matched = [b for b in tasks[system] if b['repeat'] == pt['repeat']]
                reused_r1 = not matched
                if reused_r1:
                    assert len(tasks[system]) == 1 and tasks[system][0]['repeat'] == 1 and pt['repeat'] == 2
                    matched = tasks[system]
                assert len(matched) == 1
                pairings.append(dict(position_id=position_id, pdb_cell_id=pt['cell_id'],
                    baseline_cell_id=matched[0]['cell_id'], baseline_system=system,
                    measurement_host=NODE, expected_actual_hostname=HOSTNAME, trace=ref(path100),
                    baseline_R1_reused_for_PDB_R2=reused_r1,
                    independent_baseline_repeat=not reused_r1,
                    independent_arrival_seed=False, historical_A_result_used=False))
    assert Counter(r['system'] for r in cells) == {'pdblend':40, 'mixed':34, 'distserve':34, 'dynamollm':34, 'ecoserve':35}
    assert len(cells) == len({r['cell_id'] for r in cells}) == 177
    assert len(pairings) == 160 and sum(p['baseline_R1_reused_for_PDB_R2'] for p in pairings) == 23
    declaration = dict(schema='ascending-new-physical-node-14B-declaration-v1',
        created_at_utc=datetime.now(timezone.utc).isoformat(timespec='seconds'),
        parent=ref(PARENT), node_identity=ref(HERE / 'node-identity.json'), node=NODE, model='14b',
        expected_actual_hostname=HOSTNAME, expected_gpu_uuids=UUIDS,
        ready=False, measurements_started=False, physical_qualification_granted=False,
        purpose='Fresh same-node five-system comparison; old A observations remain historical only.',
        authority='Controller selected 177 conditional runs retaining original per-system repeats and explicit R1-to-R2 pairing.',
        counts=dict(rate_positions=32, initial_conditional_runs_upper_bound=177,
                    by_system=dict(Counter(r['system'] for r in cells)),
                    by_dataset=dict(Counter(r['dataset'] for r in cells)),
                    potential_pairings=160, explicit_nonindependent_baseline_R1_reuse_pairings=23),
        counts_are_not_campaign_total_or_completion_percentage_denominator=True,
        historical_A_checkpoint_reuse_count=0, historical_A_capacity_certificate_reuse_allowed=False,
        request_window_s=100., request_hard_timeout_s=120., measurement_cleanup_bound_s=90.,
        arrival_seed=701, sampling_seed=20260907, repeat_semantics='Same seed701 trace repeated; not independent arrival seeds.',
        dataset_order=DATASETS, baseline_system_order=SYSTEMS[1:],
        phase_order=['all_PDB_group_boundaries_on_this_node', 'baselines_batched_by_system_on_this_node'],
        source_rules=copy.deepcopy(parent['rules']),
        additional_rules=dict(exclusive_all_eight_GPU_measurement=True,
            all_prior_A_results_historical_only=True, any_valid_complete_PDB_SLO_below_0_90_caps_higher_rates=True,
            complete_only_already_declared_repeats_at_capped_rate=True,
            unknown_engineering_failure_stops_for_diagnosis_no_automatic_retry=True,
            baselines_only_through_reached_cap_same_actual_host_trace_SLO_seed=True,
            PDB_P12_idle_domain_reacquire_v1=False,
            unchanged_profiles_are_candidates_until_qualified_on_this_physical_node=True,
            immutable_new_actual_binding_required=True, old_hardware_qualification_must_not_be_relabelled=True,
            extension_requires_new_child_declaration=True),
        source_target_manifest=ref(CAMPAIGN / 'hosts/14b-capacity-p12/manifest.json'),
        source_target_profile=ref(REPO / 'campaign/main-slo-improvement-v1/A/long-batch6-profile-001/profiles.development.json'),
        source_aliases=aliases, positions=positions, cells=cells, pairings=pairings)
    save(HERE / 'declaration.json', declaration)
    save(HERE / 'preparation.json', dict(schema='newnode14B-preparation-checklist-v1', ready=False,
        target=ref(HERE / 'node-identity.json'), declaration=ref(HERE / 'declaration.json'),
        image=dict(tag='pdblend-next:io-v3', id='sha256:0bb51d143b7fcaaea2e794dd6e207cf4165a4f21522a2e932a4bd4a117074bc2'),
        model_path='/root/workspace/models/Qwen2.5-14B-Instruct',
        runtime_dependencies='/root/workspace/pdblend/.runtime-deps',
        model_and_image_transfer_owner='root controller; no concurrent child writes',
        candidate_layout=dict(native_kind='v3', tp=1, initial_mixed_instance_count=2,
                              initial_gpu_indices=[6,7], service_budget_tokens=2048, restore_budget_tokens=8192),
        prerequisites=[
            'Verify source package, model shards, image ID and driver/runtime compatibility on the new host.',
            'Bind every native instance to freshly observed container IDs, ports, hostname and GPU UUIDs.',
            'Fresh ordinary output correctness across all datasets; all-eight power evidence and cleanup.',
            'Validate candidate TP1 profile numerical shapes and frequency support/settle/cancellation on the actual host.',
            'Fresh isolated power sampler validation on this node; no inherited success from the old A selftest.',
            'Fresh 14B capacity calibration/certificate and autonomous growth/return if dynamic Alpaca is retained; no old A certificate.',
            'Source/profile/config and phase-specific measurement qualification reviewed before freezing a runnable release.',
            'Each baseline restore preserves its genuine implementation and records its own ordinary and measurement qualification.',
            'Acquire exclusive actual-node all-eight-GPU lease; inspect existing owner before any hardware action.'
        ], hardware_or_GPU_measurement_started=False))
    save(HERE / 'build-validation.json', dict(passed=True, cpu_only=True,
        validation_scope='declaration counts, complete new tasks, trace byte identity, same-node pairing and explicit reuse only',
        declaration=ref(HERE / 'declaration.json'), cells=177, positions=32, pairings=160,
        recomputed_missing_traces=2, all32_trace_hashes_match_parent=True,
        all32_source300_hashes_match_parent=True, old_A_results_used_in_new_pairings=False,
        physical_qualification_granted=False, GPU_work_started=False))
    print(json.dumps(declaration['counts']))


if __name__ == '__main__':
    main()
