"""CPU-only original-policy plan. No performance binding or executor is created."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
C = HERE.parent
SOURCE = C/'five-system-fixed-window-v1/sources/B32B/manifest.json'
SOURCE_SHA = '4b9494c6b0a38cb9d44dbc490530d88e9a0eec76b44854f6e40db0328bbfe5ed'
HOST = C.parent/'releases/five-system100-B32B-v1-runtime'
HOST_SHA = '158605f4ff65c97028c975760b78c97b4f3241bb8655c05f62746487483291bd'
CONFIGS = {
    'alpaca': '34fc552fce6d8fd2dcaf31ed31c3a5ab9d56bf842a5e95391495b16c68997e41',
    'sharegpt': '9a3c39c0708de47e600155c4710aecef6324c9d77cbc193747863f35abaac98c',
    'longbench': 'ea9af3b79ff4970c8ec6cf11f22a6b1b92d74cb6147d77cd89915e554cf9fa68',
}
SLOS = {'alpaca': (1., .1), 'sharegpt': (5., .15), 'longbench': (15., .2)}
DEADLINE = 1788872770.0400891
PROTOCOL = 'per-dataset-slo-five-system-fixed-window-v1'
CORRECTNESS_PROTOCOL = 'legacy-temporal-default-trajectory-exact-v2'
PLANNED_BINDING = C/'B32B-baseline-main-first-sequence-v1/attempt-001/bindings/ecoserve/binding.json'


def require(ok, why):
    if not ok: raise RuntimeError(why)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''): h.update(block)
    return h.hexdigest()


def read(path): return json.loads(Path(path).read_text())


def check_source(source):
    require(source['model'] == '32b' and source['protocol_id'] == PROTOCOL, 'wrong model/protocol')
    require(source['arrival_window_s'] == 100 and source['arrival_seeds'] == [701]
        and source['request_hard_timeout_s'] == source['drain_after_arrival_window_s'] == 120,
        'original timing/seed changed')
    all_rows = source['cells']
    require(len(all_rows) == 240 and len({r['cell_id'] for r in all_rows}) == 240, 'original240 required')
    rows = [r for r in all_rows if r['system'] == 'ecoserve']
    require(len(rows) == 48, 'original Eco48 required')
    index = {r['cell_id']: r for r in all_rows}
    for dataset, (ttft, tpot) in SLOS.items():
        main = [r for r in rows if r['dataset'] == dataset and r['phase'] == 'main']
        scale = [r for r in rows if r['dataset'] == dataset and r['phase'] == 'scale']
        require(len(main) == 10 and len({r['rate_rps'] for r in main}) == 10 and len(scale) == 6,
            'original ten rates/six scales per dataset required')
        for row in main + scale:
            factor = row['slo_scale']
            require(factor in ((1,) if row['phase'] == 'main' else (.5, 2)), 'wrong SLO phase/scale')
            require(row['arrival_window_s'] == 100 and row['seed'] == row['arrival_seed'] == 701
                and row['slo_ttft_s'] == ttft*factor and row['slo_tpot_s'] == tpot*factor,
                'original per-dataset timing/SLO changed')
            counterparts = [r for r in all_rows if (r['dataset'], r['rate_rps'], r['slo_scale'], r['phase']) ==
                (dataset, row['rate_rps'], factor, row['phase'])]
            require(len(counterparts) == 5 and len({r['system'] for r in counterparts}) == 5, 'five system counterpart missing')
            for field in ('trace_sha256','trace','n_requests','content_pairing_sha256','source_indices_sha256'):
                require(len({r[field] for r in counterparts}) == 1, 'paired offered work changed: '+field)
            if row['phase'] == 'scale':
                parent = index.get(row['reuse_main_cell_id'])
                require(parent and parent['system'] == 'ecoserve' and parent['phase'] == 'main'
                    and parent['trace_sha256'] == row['trace_sha256'] and parent['rate_rps'] == row['rate_rps'],
                    'scale1 reference is not original same-work main')
    return copy.deepcopy(rows)


def configuration(original):
    require(original['strategy'] == 'ecoserve', 'original Eco policy required')
    cfg = copy.deepcopy(original)
    # Exact two additions made by the original bind_baseline_v2.py.
    cfg.update(controller_source_release=str(HOST), comparison_system='ecoserve')
    return cfg


def inputs():
    require(sha(SOURCE) == SOURCE_SHA and sha(HOST/'manifest.json') == HOST_SHA, 'frozen source/host changed')
    files = {str(SOURCE): SOURCE_SHA, str(HOST/'manifest.json'): HOST_SHA}
    configs = {}
    for dataset, digest in CONFIGS.items():
        path = C/f'B32B-five-system100-v1/configs/{dataset}.ecoserve.json'
        require(sha(path) == digest, 'original Eco configuration changed')
        files[str(path)] = digest
        configs[dataset] = configuration(read(path))
    rows = check_source(read(SOURCE))
    for row in rows:
        require(sha(row['trace']) == row['trace_sha256'], 'original trace bytes changed')
        files[row['trace']] = row['trace_sha256']
    for path in (C/'B32B-five-system100-v1/bind_baseline_v2.py',
        C/'B32B-five-system100-baseline-deployment-v1/deployment.json',
        C/'five-system-execution-v3/run.py', C/'five-system-execution-v3/manifest.json'):
        files[str(path)] = sha(path)
    return rows, configs, files


def prepare(out):
    require(not out.exists(), 'new plan output required')
    rows, configs, files = inputs()
    plan = dict(schema=1, kind='unqualified-ecoserve-continuation-plan', model='32b', ready=False,
        actual_binding=None, actual_qualification=None, actual_native_oracle=None, actual_release=None,
        protocol_id=PROTOCOL, proposed_correctness_protocol_id=CORRECTNESS_PROTOCOL, deadline_s=DEADLINE,
        original_full_manifest=dict(path=str(SOURCE), sha256=SOURCE_SHA),
        original_source_rows=rows, main_cells=30, future_scale_cells=18, seed=701,
        original_planned_binding_path=str(PLANNED_BINDING),
        original_planned_output=str(PLANNED_BINDING.parent/'results'),
        new_host_or_engine_source_claimed=False, GPU_or_network_capability=False,
        main_run_only_after_actual_qualification=True, scale_requires_separate_B_main_release=True,
        legacy_temporal_exact_must_remain_false=True,
        correctness_api=dict(module='B32B-temporal-qualification-v2', module_frozen_sha256=None,
            function='audit_fresh_gate(gate_dir, binding, oracle)',
            required_verified=['ordinary','pd','cancel','temporal_native_trajectory_exact','native_cleanup','identity','all8_measurement','clock']),
        future_main_argv=['python3',str(C/'five-system-execution-v3/run.py'),'--manifest',str(SOURCE),
            '--binding',str(PLANNED_BINDING),'--system','ecoserve','--phase','main','--max-cells','30','--run'],
        do_not_run_argv_until_fresh_binding_exists_and_owner_releases=True)
    out.mkdir(parents=True)
    def write(path, value):
        with path.open('x') as stream: json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')
    write(out/'plan.json', plan)
    (out/'configs').mkdir()
    for dataset, cfg in configs.items(): write(out/'configs'/f'{dataset}.json', cfg)
    write(out/'sources.json', files)
    for path, digest in files.items(): require(sha(path) == digest, 'source changed while preparing')
    write(out/'manifest.json', dict(files={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()},
        generator=dict(path=str(Path(__file__).resolve()),sha256=sha(__file__)), ready=False))
    return dict(prepared=str(out),main_cells=30,future_scale_cells=18,ready=False,actual_binding_written=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    if args.out: result = prepare(args.out.resolve())
    else:
        rows, _, _ = inputs()
        result = dict(cpu_check=True, original_cells=len(rows), ready=False, actual_binding_written=False)
    print(json.dumps(result, indent=2))
