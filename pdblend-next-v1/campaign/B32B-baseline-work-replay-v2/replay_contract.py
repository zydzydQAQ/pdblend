"""Bind a PDB-only replay to the unchanged original baseline trace bytes."""


def validate(b):
    manifest = b.read(b.ROOT / 'package-manifest.json')
    for rel, digest in manifest['files'].items():
        b.require(b.sha(b.ROOT / rel) == digest, 'replay source changed: ' + rel)
    spec = b.read(b.ROOT / 'runspec.json')
    b.require(spec['baseline_execution'] is False and spec['formal_eligible'] is False, 'wrong replay scope')
    b.require(b.sha(spec['source_replay_plan']) == spec['source_replay_plan_sha256'], 'replay input declaration changed')
    source = b.read(spec['source_replay_plan'])
    refs = {r['dataset']: r for r in source['cells'] if r['model'] == '32b'}
    b.require(len(spec['cells']) == 3 and {r['dataset'] for r in spec['cells']} == set(refs), 'three original datasets required')
    slos = {'alpaca': (1., .1), 'sharegpt': (5., .15), 'longbench': (15., .2)}
    for row in spec['cells']:
        ref = refs[row['dataset']]
        b.require(set(ref['baseline_systems']) == {'mixed', 'distserve', 'ecoserve', 'dynamollm-resident'}, 'original four references missing')
        b.require(row['system'] == 'pdblend' and row['seed'] == 11 and row['n_requests'] == 64,
                  'only original PDB replay requests are authorized')
        b.require(b.sha(row['trace']) == row['trace_sha256'] == ref['trace_sha256'], 'original trace bytes changed')
        trace = b.read(row['trace'])
        b.require(trace['seed'] == 11 and trace['dataset'] == row['dataset']
                  and trace['split'] == row['split'] == 'development'
                  and trace['load'] == row['load'] and len(trace['requests']) == len(trace['prompts']) == 64,
                  'original trace identity or work differs')
        b.require(row['trace_duration_s'] == trace['duration_s'] == ref['arrival_span_s'], 'original arrivals changed')
        b.require((row['slo_ttft_s'], row['slo_tpot_s']) == slos[row['dataset']], 'wrong user dataset SLO')
    cfg = b.read(b.CONFIG)
    old = b.read(b.ROOT / 'inputs/controller.source.json')
    old['controller_source_release'] = str(b.HOST)
    old['profiles'] = str(b.PROFILE)
    b.require(b.sha(b.HOST/'manifest.json') == b.BINDING['host_manifest_sha256'] and b.sha(b.PROFILE) == b.BINDING['profile_sha256'], 'bound host/profile changed')
    b.require(cfg['scheduler_budget_ablation'] == dict(schema_version=1,max_num_batched_tokens=8192,max_num_seqs=32) and cfg['output_prior'] == 211 and cfg['node_gpus'] == list(range(8)), 'uniform budget/prior/energy scope changed')
    b.require(cfg == old and cfg['strategy'] == 'pdblend-joint' and cfg['allow_pd'] is False,
              'B2 replay policy must equal the verified per-model policy')
    b.require('measurement_window_protocol' not in cfg and 'slo_scale' not in cfg,
              'original offered work does not use the new fixed300 window')
    return spec
