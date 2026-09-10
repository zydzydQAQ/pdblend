"""Build immutable uniform-grid declarations with the unchanged frozen sampler.

CLI: python3 generate.py --out release-001
Extension: append_point(declaration_ref, model, dataset, next_rate, new_output_dir)
Outputs declare possible work; only contract.select_group authorizes its order.
"""
from __future__ import annotations
import argparse
import copy
from decimal import Decimal
import hashlib
from pathlib import Path
import contract as c

OLD = c.HERE.parent / 'ascending-rate-execution-v2'
OLD_DECLARATION = dict(path=str(OLD / 'release-001/declaration.json'),
                      sha256='a566e7f203a77abe64aa2f122e66c47b87c44ab0b91f60c3453235ca69c4c32f')
OLD_API = dict(path=str(OLD / 'contract.py'), sha256='4d3438b0e90035054b643bf6464a08ba4c6d7f8f3e3128f53c8adac3b67fc645')
GENERATOR = dict(path=str(c.REPO / 'campaign/five-system-fixed-window-v1/generate.py'),
                 sha256='d554c26f9d9608d8ce443720eb21cb4129d6d209819282df3c5c154274fa5b4d')


def frozen_code(reference, name):
    c.need(c.sha(reference['path']) == reference['sha256'], 'frozen code changed')
    return c.load_module(reference['path'], name)


def write_new(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as f:
        f.write(c.encode(value))


class Builder:
    def __init__(self, out):
        self.out = Path(out).resolve()
        c.need(not self.out.exists(), 'output directory must be new')
        self.old = c.checked(OLD_DECLARATION)
        self.old_api = frozen_code(OLD_API, '_uniform_original_ascending')
        self.generator = frozen_code(GENERATOR, '_uniform_original_100s')
        self.parent = self.generator.parent_generator()
        self.spec_refs = {m: next(w['source_spec'] for w in self.old['new_workloads'] if w['model'] == m)
                          for m in c.MODELS}
        self.specs = {m: c.checked(r) for m, r in self.spec_refs.items()}
        c.need(len({self.parent.digest(s) for s in self.specs.values()}) == 1, 'model source specifications diverged')
        self.groups, self.sampling_seed = self.parent.load_sources(self.specs['7b'])
        self.reconstructed_sources = []
        self.out.mkdir(parents=True)
        self.old_positions = {(p['model'], p['dataset'], c.number(p['rate_rps'])): p for p in self.old['positions']}
        self.old_saved = {r['cell_id']: r for r in self.old['reused_observations']}
        snapshot = c.read(c.HERE.parent / 'uniform-rate-20260909-v1/reports/current/results.json')
        write_new(self.out / 'previous-report-snapshot.json', snapshot)
        self.snapshot_ref = c.ref(self.out / 'previous-report-snapshot.json')
        self.candidates = {}
        for observation in self.old['reused_observations'] + snapshot['observations']:
            m, d = observation['model'], observation['dataset']
            rate = c.number(observation['rate_rps'])
            if observation.get('measurement_host') != c.host(m, d) or not c.on_grid(m, d, rate):
                continue
            if (m, d) == ('14b', 'sharegpt'):
                continue
            self.candidates[observation['checkpoint']['sha256']] = copy.deepcopy(observation)
        # Current PDB workloads determine pairing for newly measured baseline gaps.
        for observation in snapshot['observations']:
            if observation['system'] != 'pdblend' or observation.get('measurement_host') != c.host(observation['model'], observation['dataset']):
                continue
            cp = c.checked(observation['checkpoint'])
            row = copy.deepcopy(cp['row'])
            row.setdefault('trace_path', row['trace'])
            key = row['model'], row['dataset'], c.number(row['rate_rps'])
            self.old_positions[key] = dict(workload=row, original_current_checkpoint=observation['checkpoint'])
        self.audited = {}
        self.hash_cache = {}
        self.metrics = c.load_module(c.HERE / 'metrics.py', '_v2_frozen_historical_metrics')

    def workload(self, model, dataset, rate):
        rate = c.number(rate); key = model, dataset, rate
        old_position = self.old_positions.get(key)
        _, original = self.parent.build_trace(self.specs[model], model, dataset, Decimal(rate), 701,
                                              self.groups[(model, dataset)], self.sampling_seed)
        if old_position is not None:
            w = copy.deepcopy(old_position['workload'])
            trace_ref = dict(path=w['trace_path'], sha256=w['trace_sha256'])
            if Path(trace_ref['path']).exists():
                trace = c.checked(trace_ref)
            else:
                trace = self.generator.prefix_trace(original, w['source_300s_trace'])
                c.need(hashlib.sha256(self.generator.encode(trace)).hexdigest() == trace_ref['sha256'],
                       'missing old 100s trace cannot be reconstructed at its frozen SHA')
                restored_trace = self.out / 'traces' / Path(trace_ref['path']).name
                write_new(restored_trace, trace)
                self.reconstructed_sources.append(dict(original_reference=trace_ref,
                                                        exact_local_copy=c.ref(restored_trace)))
                w.update(trace_path=str(restored_trace), trace=str(restored_trace))
            original_ref = trace['source_300s_trace']
            c.need(hashlib.sha256(self.parent.encode(original)).hexdigest() == original_ref['sha256'],
                   'saved 300s trace SHA differs from the frozen source')
            if Path(original_ref['path']).exists():
                c.need(c.checked(original_ref) == original, 'saved 300s trace differs from the frozen source')
            else:
                # Preserve the old trace's embedded path/hash. The exact missing
                # bytes are reconstructed under the new release, not rewritten
                # into an old campaign by the declaration builder.
                restored = self.out / 'source300' / Path(original_ref['path']).name
                write_new(restored, original)
                self.reconstructed_sources.append(dict(original_reference=original_ref,
                                                        exact_local_copy=c.ref(restored)))
            c.need(self.generator.prefix_trace(original, original_ref) == trace,
                   'saved 100s trace differs from the unchanged original prefix')
        else:
            source_path = self.out / 'source300' / f'{model}-{dataset}-r{rate}-s701-w300.json'
            write_new(source_path, original)
            original_ref = c.ref(source_path)
            trace = self.generator.prefix_trace(original, original_ref)
            path = self.out / 'traces' / f'{model}-{dataset}-r{rate}-s701-w100.json'
            write_new(path, trace)
            w = dict(model=model, dataset=dataset, protocol_id=self.generator.PROTOCOL,
                measurement_schema=3, split='development', load='declared_absolute_rate',
                trace=str(path), trace_path=str(path), trace_sha256=c.sha(path),
                trace_bytes=path.stat().st_size, materialized=True, slo=trace['slo'],
                slo_protocol='per-dataset-slo-v1', source_300s_trace=original_ref,
                source_spec=self.spec_refs[model], generator=GENERATOR,
                parent_generator=c.ref(self.generator.PARENT_GENERATOR),
                sparse_screen=trace['n_requests'] < 30, within_trace_resampling=trace['within_trace_resampling'],
                output_lengths_modified=False, prompts_truncated=False, formal_eligible=False)
        node = c.host(model, dataset)
        w.update(model=model, dataset=dataset, node=node, measurement_host=node,
            workload_id=f'{model}-{dataset}-r{rate}-s701-w100', rate_rps=float(rate), rate_rps_decimal=rate,
            seed=701, arrival_seed=701, sampling_seed=self.sampling_seed, arrival_window_s=100.,
            trace_duration_s=100., request_hard_timeout_s=120.,
            trace_reference=dict(path=w['trace_path'], sha256=w['trace_sha256']),
            slo_ttft_s=c.SLOS[dataset][0], slo_tpot_s=c.SLOS[dataset][1], slo_scale=1., allowed_slo_scales=[1.],
            content_pairing_sha256=trace['content_pairing_sha256'], n_requests=trace['n_requests'],
            n_expected=trace['n_requests'], expected_generated_tokens=sum(r['output_len'] for r in trace['requests']),
            planned_arrival_span_s=trace['planned_arrival_span_s'],
            source_indices_sha256=hashlib.sha256(c.encode(trace['source_pool_indices'])).hexdigest(),
            request_count_warning='one arrival seed; no independent-seed confidence interval')
        c.need((trace['slo']['ttft_s'], trace['slo']['tpot_s']) == c.SLOS[dataset] and
               trace['arrival_window_s'] == 100 and trace['request_hard_timeout_s'] == 120, 'fixed workload changed')
        return w, old_position

    def audit_saved(self, observation, workload):
        digest = observation['checkpoint']['sha256']
        if digest in self.audited:
            return copy.deepcopy(self.audited[digest])
        cp = c.checked(observation['checkpoint'])
        row = copy.deepcopy(cp['row'])
        row.setdefault('node', observation['measurement_host'])
        row.setdefault('measurement_host', observation['measurement_host'])
        row.setdefault('n_requests', workload['n_requests'])
        for path, expected in cp.get('artifacts', {}).items():
            if path not in self.hash_cache:
                self.hash_cache[path] = c.sha(path)
            actual = self.hash_cache[path]
            c.need(actual == expected, 'saved raw artifact changed: ' + path)
        receipt = cp['receipt']
        if isinstance(receipt, str):
            receipt = dict(path=receipt, sha256=cp['receipt_sha256'])
        metrics = self.metrics.audit_receipt(receipt['path'], row, receipt_reference=receipt,
            checkpoint_reference=observation['checkpoint'] if cp.get('binding') else None)
        if cp.get('binding'):
            c.need(self.metrics.checkpoint_host(cp) == observation['measurement_host'], 'physical host differs from retained observation')
        else:
            # Legacy CPs retain physical identity through the pinned original audit declaration.
            c.need(observation['cell_id'] in self.old_saved and
                   self.old_saved[observation['cell_id']]['checkpoint'] == observation['checkpoint'],
                   'legacy checkpoint lacks original physical-host audit membership')
        result = dict(observation)
        result.update(metrics)
        result.update(node=observation['measurement_host'], strict_slo_recomputed=True,
            historical_partial_work_audited=bool(not metrics['work_complete'] and observation['system'] != 'pdblend'),
            historical_outcome_preserved=True, reuse_raw_artifacts_verified=True,
            checkpoint=observation['checkpoint'], independently_recomputed=True,
            trace=row['trace'], trace_path=row['trace'], trace_reference=c.ref(row['trace']),
            generated_token_throughput_tps=metrics['token_throughput_tps'],
            completed_work_throughput_rps=metrics['request_throughput_rps'])
        c.need(c.identity(result) == c.identity(workload), 'saved host/trace/SLO differs')
        if result['system'] == 'pdblend':
            c.need(result['work_complete'] is True, 'incomplete PDB cannot qualify a boundary')
        self.audited[digest] = result
        return copy.deepcopy(result)

    def position(self, model, dataset, rate, sequence=1):
        workload, original = self.workload(model, dataset, rate)
        node = c.host(model, dataset)
        pid = c.position_id(node, model, dataset, rate)
        candidates = [o for o in self.candidates.values() if c.identity(o) == c.identity(workload)]
        tasks, cells, reused = {}, [], []
        for system in c.SYSTEMS:
            saved = sorted([o for o in candidates if o['system'] == system], key=lambda o: o['repeat'])
            c.need(len({o['repeat'] for o in saved}) == len(saved), 'multiple historical outcomes for one repeat')
            observations = [self.audit_saved(o, workload) for o in saved]
            reused.extend(observations)
            tasks[system] = [c.saved_task(o) for o in observations]
            repeats = [o['repeat'] for o in observations]
            if not repeats:
                row = c.row_for(workload, node, system, 1, sequence + len(cells))
                cells.append(row); tasks[system].append(c.new_task(row))
            if system == 'pdblend' and 2 not in repeats:
                row = c.row_for(workload, node, system, 2, sequence + len(cells))
                cells.append(row); tasks[system].append(c.new_task(row))
        supplements = []
        for system in c.SYSTEMS[1:]:
            if (model, dataset, c.number(rate), system) in c.SUPPLEMENTS:
                row = c.row_for(workload, node, system, 1, sequence + len(cells), 'metric_supplement')
                cells.append(row); supplements.append(c.new_task(row))
        pairings = [dict(position_id=pid, pdb=pdb, baseline=baseline, baseline_system=system,
                        trace=workload['trace_reference'], pair_origin='same_host_exact_trace_fixed_slo',
                        baseline_repeat_reused=baseline['repeat'] != pdb['repeat'])
                    for pdb in tasks['pdblend'] for system in c.SYSTEMS[1:]
                    for baseline in tasks[system]]
        position = dict(position_id=pid, node=node, model=model, dataset=dataset,
            order=int(Decimal(c.number(rate)) / c.step(model, dataset)),
            rate_rps=float(rate), rate_rps_decimal=c.number(rate), workload=workload,
            trace=workload['trace_reference'], systems=tasks, metric_supplements=supplements,
            required_pdb_repeats=[t['repeat'] for t in tasks['pdblend'] if not t.get('row', {}).get('conditional')],
            newly_added_coordinate=not candidates, conditional_on_no_lower_rate_cap=True)
        return position, cells, reused, pairings

    def finish(self, positions, cells, reused, pairings, parent=None):
        groups = []
        for model in c.MODELS:
            for dataset in c.DATASETS:
                ps = [p for p in positions if (p['model'], p['dataset']) == (model, dataset)]
                source = next(g for g in self.old['groups'] if (g['model'], g['dataset']) == (model, dataset))
                groups.append(dict(group_id=f'{c.host(model, dataset)}-{model}-{dataset}',
                    node=c.host(model, dataset), model=model, dataset=dataset,
                    position_ids=[p['position_id'] for p in ps], ascending_rates=[p['rate_rps'] for p in ps],
                    start_rate_rps_decimal=c.number(c.step(model, dataset)),
                    rate_step_rps_decimal=c.number(c.step(model, dataset)),
                    pdb_source_contract=copy.deepcopy(source['pdb_source_contract']),
                    migrated_group_requires_fresh_qualification=(model, dataset) == ('14b', 'sharegpt'),
                    actual_execution_binding_required=True))
        reused = list({o['cell_id']: o for o in reused}.values())
        audit = dict(schema='uniform-v2-reuse-raw-audit', slo_threshold_comparison='strict_lt',
            observations=reused, original_report_snapshot=self.snapshot_ref,
            raw_auditor=c.ref(c.HERE / 'metrics.py'),
            artifacts_sha_verified=True, original_producer_fields_preserved=True)
        write_new(self.out / 'reuse-audit.json', audit)
        audit_reference = c.ref(self.out / 'reuse-audit.json')
        for observation in reused:
            observation['audit_reference'] = audit_reference
        declaration = dict(schema=c.SCHEMA, fixed_slo_only=True, completion_scope='five_systems',
            arrival_seed=701, sampling_seed=20260907, arrival_window_s=100., request_hard_timeout_s=120.,
            drain_after_arrival_window_s=120., normal_repeats=1, campaign_deadline_s=None,
            slo_uses_ttlt=False, baseline_saturation_search_enabled=False,
            slo_definition='complete work AND TTFT < fixed threshold AND TPOT < fixed threshold',
            stop_rule='first any valid complete strict PDB attainment below .90; finish one confirmation repeat, never a third normal repeat',
            metric_supplement_policy='one separately labelled collection only where no exact token throughput is audited; preserve original repetitions',
            parent_declaration=parent, frozen_parent=OLD_DECLARATION, frozen_parent_contract=OLD_API,
            frozen_generator=GENERATOR, previous_report_snapshot=self.snapshot_ref, reuse_audit=audit_reference,
            source_specs=self.spec_refs, reconstructed_missing_sources=self.reconstructed_sources,
            groups=groups, positions=positions, workloads=[p['workload'] for p in positions],
            cells=cells, reused_observations=reused, pairings=pairings,
            measurement_window='full 100s arrival and actual drain/control tail; all eight GPUs',
            cross_host_scientific_reuse_forbidden=True, physical_qualification_granted=False)
        declaration['counts'] = dict(rate_positions=len(positions), reused_observations=len(reused),
            declared_new_cells=len(cells),
            normal_missing_by_node={node: sum(r['node'] == node and r['measurement_purpose'] == 'normal'
                and not r['conditional'] for r in cells) for node in sorted(set(c.HOSTS.values()))},
            metric_supplements_by_node={node: sum(r['node'] == node and r['measurement_purpose'] == 'metric_supplement'
                for r in cells) for node in sorted(set(c.HOSTS.values()))})
        c.validate_structure(declaration)
        write_new(self.out / 'declaration.json', declaration)
        reference = c.ref(self.out / 'declaration.json')
        for node in sorted(set(c.HOSTS.values())):
            decisions = {g['group_id']: c.select_group(c.resolve_group(reference, g['model'], g['dataset']), [])
                         for g in groups if g['node'] == node}
            write_new(self.out / f'node-{node}.json', dict(schema='uniform-v2-node-view', node=node,
                declaration=reference, groups=[g for g in groups if g['node'] == node], decisions=decisions))
        manifest = dict(schema='uniform-rate-execution-release-v2', declaration=reference,
            API=c.ref(c.HERE / 'contract.py'), generator=c.ref(__file__), counts=declaration['counts'],
            measurements_started=False, physical_qualification_granted=False,
            files={str(p.relative_to(self.out)): c.sha(p) for p in self.out.rglob('*') if p.is_file()})
        write_new(self.out / 'manifest.json', manifest)
        return manifest


def generate_release(out=None):
    builder = Builder(out or c.HERE / 'release-001')
    positions, cells, saved, pairs = [], [], [], []
    for model in c.MODELS:
        for dataset in c.DATASETS:
            for rate in c.grid(model, dataset, c.INITIAL_LIMITS[model][dataset]):
                p, rs, observations, ps = builder.position(model, dataset, rate, len(cells) + 1)
                positions.append(p); cells.extend(rs); saved.extend(observations); pairs.extend(ps)
    return builder.finish(positions, cells, saved, pairs)


def append_point(declaration_ref, model, dataset, next_rate, out):
    previous = c.load_declaration(declaration_ref)
    group = c.resolve_group(declaration_ref, model, dataset)
    expected = Decimal(group['positions'][-1]['rate_rps_decimal']) + c.step(model, dataset)
    c.need(Decimal(c.number(next_rate)) == expected, 'extension must add exactly one fixed increment')
    builder = Builder(out)
    p, rows, saved, pairs = builder.position(model, dataset, next_rate, len(previous['cells']) + 1)
    positions = previous['positions'] + [p]
    positions.sort(key=lambda x: (c.MODELS.index(x['model']), c.DATASETS.index(x['dataset']), Decimal(x['rate_rps_decimal'])))
    return builder.finish(positions, previous['cells'] + rows, previous['reused_observations'] + saved,
                          previous['pairings'] + pairs, parent=declaration_ref)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=c.HERE / 'release-001')
    args = parser.parse_args()
    result = generate_release(args.out)
    print(c.encode({k: result[k] for k in ('declaration', 'API', 'counts')}).decode())
