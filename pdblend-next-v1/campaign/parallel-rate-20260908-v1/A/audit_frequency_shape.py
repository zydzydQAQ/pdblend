"""Independent actual-shape evidence; this never publishes a profile."""
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]

def require(ok, message):
    if not ok:
        raise ValueError(message)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

def audit(spec_path, micro, context, out):
    spec_path, micro, out = map(lambda p: Path(p).resolve(), (spec_path, micro, out))
    require(not out.exists(), 'append-only evidence output required')
    spec = read(spec_path)
    sources = dict(spec['files'])
    sources[str(spec_path)] = sha(spec_path)
    sources[str(Path(__file__).resolve())] = sha(__file__)
    require(all(sha(p) == h for p, h in sources.items()), 'frozen input changed')
    def used(path):
        path = Path(path).resolve()
        value = sha(path)
        require(str(path) not in sources or sources[str(path)] == value, 'evidence changed during audit')
        sources[str(path)] = value
        return path
    state = read(used(micro / 'status.json'))
    require(state['complete'] and state['cleanup_complete'] and not state['failed']
            and state['node_lease_held'] is False and not Path('/proc/' + str(state['pid'])).exists(),
            'complete work, restoration and released owner required')
    require(read(used(micro / 'spec-reference.json')) == dict(path=str(spec_path), sha256=sha(spec_path)),
            'actual spec differs')
    code = next(Path(p).parent for p in spec['files'] if p.endswith('/run_current.py'))
    sys.path[:0] = [str(code), str(Path(spec['host_release']) / 'src')]
    evidence = load('evidence', code / 'evidence.py')
    export = load('context_export', code / 'context_export.py')
    order = load('source_order', code / 'source_order.py')
    terminal = load('a_2400_terminal', REPO / 'campaign/main-slo-improvement-v1/A/long-batch6-registration-v1/terminal.py')
    used(terminal.__file__)
    from ecopadg.metrics import clip_power_window
    from ecopadg.measure.power import trapezoid_energy
    measurement = read(used(state['measurement']['path']))
    require(sha(state['measurement']['path']) == state['measurement']['sha256'], 'measurement receipt changed')
    for path, h in measurement['artifacts'].items():
        require(sha(used(path)) == h, 'raw power/clock artifact changed')
    power_path = next(p for p in measurement['artifacts'] if Path(p).name == 'power.csv')
    with Path(power_path).open() as handle:
        power = [(float(r['t_s']), [float(r[f'gpu{g}_w']) for g in range(8)]) for r in csv.DictReader(handle)]
    clocks = read(next(p for p in measurement['artifacts'] if Path(p).name == 'clocks.json'))
    if isinstance(clocks, dict):
        clocks = clocks.get('samples', clocks.get('frequency_samples'))
    before, after = [read(used(micro / f'source-order.{phase}.json')) for phase in ('before', 'after')]
    declarations = evidence.point_specs()
    require(len(declarations) == spec['planned_points'] == 3 and
            [p['point_id'] for p in declarations] == [p['point_id'] for p in state['completed']],
            'complete declared three repeats required')
    rows, all_ids, output_hash = [], set(), None
    for declared in declarations:
        path = micro / 'results' / declared['point_id']
        raw = read(used(path / 'raw.json'))
        events = [json.loads(line) for line in used(path / 'events.jsonl').read_text().splitlines() if line]
        row = dict(point_id=declared['point_id'], spec=declared, valid=False)
        try:
            require(raw['spec'] == declared and raw['observation_valid'], 'actual work declaration failed')
            original = evidence.derive_point(raw, events, power, clocks)
            require(original['valid'], 'original full-work/native/clock check failed: ' + str(original))
            ids = {r['request_id'] for r in raw['requests']}
            require(len(ids) == declared['batch_size'] and not ids & all_ids, 'request identities reused')
            all_ids.update(ids)
            value = original['execution']['output_sha256']
            require(output_hash is None or output_hash == value, 'same-prompt full output differs across repeats')
            output_hash = value
            proof = order.validate_pair(before, after, 'nextv3a6',
                measurement_start_s=raw['measurement_start_s'], measurement_end_s=raw['measurement_end_s'],
                contract_path=code / 'source-order-contract.json')
            selected, empty = events, None
            if any(not e.get('request_ids') for e in events):
                selected, empty = terminal.trailing_empty(raw, events)
            late = export.reconstruct(raw, selected, generation=raw['runtime_before']['generation'],
                token_budget=2048, source_order_verified=proof['verified'])
            late = export.attach_power(late, power, clocks, target_gpu=6, frequency_mhz=declared['clock_command_mhz'])
            shared_context = min(v['attention_after_max'] + 1 for v in late['per_request'].values())
            require(shared_context >= context, 'real shared full-batch context does not reach requested bucket')
            prefill = []
            for rid in sorted(ids):
                phase = [e for e in selected if rid in e['request_ids'][:e['prefill']]]
                require(phase, 'missing source-ordered prefill')
                start, end = phase[0]['started_s'], phase[-1]['finished_s']
                values = clip_power_window(power, start, end, pad_s=0)
                prefill.append(dict(request_id=rid, duration_s=end-start,
                    background_decode_max=max(e['decode'] for e in phase),
                    target_power_w=trapezoid_energy([(t, [v[6]]) for t, v in values])/(end-start)))
            require(max(p['background_decode_max'] for p in prefill) == declared['batch_size']-1,
                    'actual new prefill alongside full existing decode batch minus one is missing')
            runs, run = [], []
            for event in selected:
                if event['prefill'] == 0 and event['decode'] == len(ids) and set(event['request_ids']) == ids:
                    run.append(event)
                else:
                    if run:
                        runs.append(run)
                    run = []
            if run:
                runs.append(run)
            decoded = []
            for run in runs:
                if len(run) < 2:
                    continue
                start, end = run[0]['started_s'], run[-1]['finished_s']
                active = [v[6] for t, v in clocks if start <= t <= end]
                require(len(active) >= 3 and all(abs(f-declared['clock_command_mhz']) <= 15 for f in active),
                        'target clock does not cover every full-batch run')
                values = clip_power_window(power, start, end, pad_s=0)
                decoded.append(dict(steps=len(run), start_s=start, end_s=end,
                    finish_spacing_s=[b['finished_s']-a['finished_s'] for a,b in zip(run,run[1:])],
                    actual_clock_min_mhz=min(active), actual_clock_max_mhz=max(active),
                    target_power_w=trapezoid_energy([(t,[v[6]]) for t,v in values])/(end-start),
                    all_eight_gpu_j=trapezoid_energy(values)))
            require(decoded, 'no real full-batch decode span')
            row.update(valid=True, full_work=original, source_order=proof, late_context=late,
                       shared_context=shared_context, prefill=prefill, full_decode=decoded,
                       terminal_empty_event=empty, original_event_count=len(events))
        except Exception as exc:
            row['error'] = repr(exc)
        rows.append(row)
    result = dict(schema='A2400-actual-shape-evidence-v1', passed=all(r['valid'] for r in rows),
        profile_publication_allowed=False, shape_context_limit=context, points=rows, source_sha256=sources,
        limitations='Repeated empirical same-shape envelope; no physical KV contents or hard future bound claimed.')
    require(all(sha(p) == h for p,h in sources.items()), 'source changed at final audit')
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(passed=result['passed'], points=[dict(point_id=r['point_id'], valid=r['valid'],
        shared_context=r.get('shared_context'), error=r.get('error')) for r in rows])))
    return result['passed']

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--spec', required=True)
    parser.add_argument('--micro', required=True)
    parser.add_argument('--context', type=int, required=True)
    parser.add_argument('--out', required=True)
    a=parser.parse_args()
    raise SystemExit(0 if audit(a.spec,a.micro,a.context,a.out) else 1)
