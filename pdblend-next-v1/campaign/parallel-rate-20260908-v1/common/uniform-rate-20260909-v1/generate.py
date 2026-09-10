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
        node = c.HOSTS[model]
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

    def position(self, model, dataset, rate, sequence=1):
        w, old = self.workload(model, dataset, rate)
        node = c.HOSTS[model]; pid = c.position_id(node, model, dataset, rate)
        inherited = bool(node in ('B', 'C') and old and
                         all(t['action'] == 'reuse' for ts in old['systems'].values() for t in ts))
        cells = []; reused = []; pairings = []
        if inherited:
            tasks = copy.deepcopy(old['systems'])
            for ts in tasks.values():
                for task in ts:
                    observation = copy.deepcopy(self.old_saved[task['cell_id']])
                    c.need(c.identity(observation) == c.identity(w), 'saved observation host/trace/SLO differs')
                    c.checked(observation['checkpoint'])
                    reused.append(observation)
            pairings = [copy.deepcopy(p) for p in self.old['pairings'] if p['position_id'] == old['position_id']]
            for pair in pairings:
                pair['position_id'] = pid
        else:
            tasks = {s: [] for s in c.SYSTEMS}
            for s in c.SYSTEMS:
                for repeat in (1, 2):
                    row = c.row_for(w, node, s, repeat, sequence + len(cells))
                    cells.append(row); tasks[s].append(c.new_task(row))
            for pt in tasks['pdblend']:
                for s in c.SYSTEMS[1:]:
                    bt = next(t for t in tasks[s] if t['repeat'] == pt['repeat'])
                    pairings.append(dict(position_id=pid, pdb=pt, baseline=bt, baseline_system=s,
                        baseline_repeat_reused=False, pair_origin='new_exact_repeat', trace=w['trace_reference']))
        p = dict(position_id=pid, node=node, model=model, dataset=dataset,
            order=int(Decimal(c.number(rate)) / c.step(model, dataset)), rate_rps=float(rate),
            rate_rps_decimal=c.number(rate), trace=w['trace_reference'], workload=w,
            newly_added_coordinate=not inherited, historical_coordinate_exists=old is not None,
            required_pdb_repeats=[t['repeat'] for t in tasks['pdblend']], systems=tasks,
            new_executions_before_cap_pruning=len(cells), conditional_on_no_lower_rate_cap=True)
        return p, cells, reused, pairings

    def finish(self, positions, cells, reused, pairings, parent=None):
        groups = []
        for m in c.MODELS:
            for d in c.DATASETS:
                ps = [p for p in positions if p['model'] == m and p['dataset'] == d]
                source = next(g for g in self.old['groups'] if g['model'] == m and g['dataset'] == d)
                groups.append(dict(group_id=f'{c.HOSTS[m]}-{m}-{d}', node=c.HOSTS[m], model=m, dataset=d,
                    start_rate_rps_decimal=c.number(c.step(m, d)), rate_step_rps_decimal=c.number(c.step(m, d)),
                    near_boundary_increment_rps=float(c.step(m, d)),
                    ascending_rates=[p['rate_rps'] for p in ps], position_ids=[p['position_id'] for p in ps],
                    pdb_source_contract=copy.deepcopy(source['pdb_source_contract']),
                    historical_reuse_allowed=c.HOSTS[m] in ('B', 'C'), actual_execution_binding_required=True))
        unique_saved = {r['cell_id']: r for r in reused}
        d = dict(schema=c.SCHEMA, fixed_slo_only=True, arrival_seed=701, sampling_seed=20260907,
            arrival_window_s=100., request_hard_timeout_s=120., drain_after_arrival_window_s=120.,
            campaign_deadline_s=None, baseline_saturation_search_enabled=False, slo_uses_ttlt=False,
            slo_definition='complete work AND 0 <= TTFT < threshold AND 0 <= TPOT < threshold; no TTLT gate',
            authorized_user_grid=True, rate_grid_arithmetic='start=step; rate[k]=(k+1)*step using Decimal',
            stop_rule='any valid complete PDB repeat below .90 caps higher rates immediately; finish declared repeats and reached baselines',
            repeats_for_new_points=2, repeat_semantics='two actual repetitions of identical seed701 trace, not independent arrival seeds',
            actual_execution_binding_required=True, measurements_started_by_this_builder=False,
            new_A_old_observation_reuse_forbidden=True, parent_declaration=parent,
            frozen_parent=OLD_DECLARATION, frozen_parent_contract=OLD_API, frozen_generator=GENERATOR,
            source_specs=self.spec_refs, groups=groups, positions=positions,
            reconstructed_missing_sources=self.reconstructed_sources,
            workloads=[p['workload'] for p in positions], cells=cells,
            reused_observations=list(unique_saved.values()), pairings=pairings,
            dynamic_reuse_requires='same-host exact workload/repeat and checkpoint in a pinned arithmetic audit report',
            metrics=['energy_j', 'slo_attainment', 'ttft_avg_s', 'tpot_avg_s',
                     'completed_work_throughput_rps', 'generated_token_throughput_tps', 'gpu_util', 'gpu_util_per_gpu'],
            measurement_window='full 100s arrival epoch and actual drain/control tail; all eight GPUs',
            historical_off_grid_observations_preserved_by_reference=OLD_DECLARATION)
        d['counts'] = dict(rate_positions=len(positions), initial_new_runs_upper_bound=len(cells),
            initial_new_runs_by_node={n: sum(r['node'] == n for r in cells) for n in c.HOSTS.values()},
            reused_observations=len(unique_saved), pairings=len(pairings))
        c.validate_structure(d)
        write_new(self.out / 'declaration.json', d)
        declaration_ref = c.ref(self.out / 'declaration.json')
        for node in c.HOSTS.values():
            write_new(self.out / f'node-{node}.json', dict(schema='uniform-rate-node-view-v1', node=node,
                declaration=declaration_ref, position_ids=[p['position_id'] for p in positions if p['node'] == node],
                new_cell_ids=[r['cell_id'] for r in cells if r['node'] == node]))
        sources = {r['path']: r['sha256'] for r in (OLD_DECLARATION, OLD_API, GENERATOR, *self.spec_refs.values())}
        for reference in self.spec_refs.values():
            c.checked(reference)
        c.checked(OLD_DECLARATION)
        manifest = dict(schema='uniform-rate-execution-release-v1', declaration=declaration_ref,
            API=c.ref(c.HERE / 'contract.py'), generator=c.ref(__file__), counts=d['counts'],
            sources=sources, measurements_started=False, physical_qualification_granted=False,
            files={str(p.relative_to(self.out)): c.sha(p) for p in self.out.rglob('*') if p.is_file()})
        write_new(self.out / 'manifest.json', manifest)
        return manifest


def generate_release(out=None):
    b = Builder(out or c.HERE / 'release-001')
    positions = []; cells = []; reused = []; pairs = []
    for m in c.MODELS:
        for ds in c.DATASETS:
            for rate in c.grid(m, ds, c.INITIAL_LIMITS[m][ds]):
                p, rs, saved, ps = b.position(m, ds, rate, len(cells) + 1)
                positions.append(p); cells.extend(rs); reused.extend(saved); pairs.extend(ps)
    return b.finish(positions, cells, reused, pairs)


def append_point(declaration_ref, model, dataset, next_rate, out):
    """Create a new full declaration by adding exactly the next on-grid point.

Caller must first obtain extension_declaration_required from select_group.
Earlier declarations and source paths are never modified.
"""
    old = c.load_declaration(declaration_ref)
    group = c.resolve_group(declaration_ref, model, dataset)
    expected = Decimal(group['positions'][-1]['rate_rps_decimal']) + c.step(model, dataset)
    c.need(Decimal(c.number(next_rate)) == expected, 'extension must be exactly one fixed increment')
    b = Builder(out)
    p, rows, saved, pairs = b.position(model, dataset, next_rate, len(old['cells']) + 1)
    positions = old['positions'] + [p]
    positions.sort(key=lambda p: (c.MODELS.index(p['model']), c.DATASETS.index(p['dataset']), Decimal(p['rate_rps_decimal'])))
    return b.finish(positions, old['cells'] + rows, old['reused_observations'] + saved,
                    old['pairings'] + pairs, parent=declaration_ref)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=c.HERE / 'release-001')
    args = parser.parse_args()
    print(c.encode(generate_release(args.out)).decode())
