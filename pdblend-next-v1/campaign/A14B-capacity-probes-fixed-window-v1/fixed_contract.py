"""Explicit fixed-window probe contract; old minimum/span gates do not apply."""
from pathlib import Path
import math

PROTOCOL = 'per-dataset-slo-fixed-window-v2'
SLOS = {'alpaca': (1., .1), 'sharegpt': (5., .15), 'longbench': (15., .2)}


def validate_row(b, row):
    b.require(row['phase'] == 'probe' and row['model'] == '14b', 'wrong probe scope')
    b.require(row['seed'] in (701, 1701) and row['slo_scale'] in (.5, 1., 2.), 'wrong seed/scale')
    b.require(row['trace_duration_s'] == 300 and row['protocol_id'] == PROTOCOL, 'wrong window')
    b.require(b.sha(row['trace']) == row['trace_sha256'], 'trace source changed')
    trace = b.read(row['trace'])
    b.require(trace['protocol_id'] == PROTOCOL and trace['measurement_schema'] == 3
              and trace['arrival_window_s'] == trace['duration_s'] == 300, 'wrong trace protocol')
    b.require(trace['dataset'] == row['dataset'] and trace['model'] == '14b'
              and trace['seed'] == row['seed'] and trace['split'] == 'development', 'wrong trace identity')
    b.require(len(trace['requests']) == len(trace['prompts']) == row['n_requests']
              and trace['n_requests'] == row['n_requests'] > 0, 'missing work')
    arrivals = [r['arrival_s'] for r in trace['requests']]
    b.require(arrivals[0] == 0 and arrivals == sorted(arrivals)
              and all(math.isfinite(t) and 0 <= t < 300 for t in arrivals), 'wrong arrivals')
    base = SLOS[row['dataset']]
    b.require((row['slo_ttft_s'], row['slo_tpot_s']) == tuple(v * row['slo_scale'] for v in base),
              'scaled dataset SLO differs')
    source = b.read(row['source_manifest'])
    b.require(b.sha(row['source_manifest']) == row['source_manifest_sha256'], 'generator declaration changed')
    candidates = [r for r in source['cells'] if r['cell_id'] == row['source_cell_id']]
    b.require(len(candidates) == 1 and candidates[0]['trace_sha256'] == row['trace_sha256']
              and candidates[0]['n_requests'] == row['n_requests'], 'generated work differs')


def validate_package(b):
    manifest = b.read(b.ROOT / 'package-manifest.json')
    for path, digest in manifest['files'].items():
        b.require(b.sha(b.ROOT / path) == digest, 'package changed: ' + path)
    spec = b.read(b.ROOT / 'runspec.json')
    b.require(spec['protocol_id'] == PROTOCOL and spec['execute_baselines'] is False
              and spec['formal_eligible'] is False and 0 < len(spec['cells']) <= 10, 'wrong probe declaration')
    for row in spec['cells']:
        validate_row(b, row)
    cfg = b.read(b.CONFIG)
    original = b.read(b.ROOT / 'inputs/controller.source.json')
    original.update(controller_source_release=str(b.HOST), measurement_window_protocol=PROTOCOL,
                    arrival_window_s=300, slo_scale=1)
    b.require(cfg == original, 'probe changes frozen A policy beyond measurement protocol')
    b.require(cfg['strategy'] == 'pdblend-joint' and cfg['allow_pd'] is False
              and cfg['node_gpus'] == list(range(8)), 'wrong policy/hardware')
    return spec
