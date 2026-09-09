"""CPU-only static-three declaration; no executor, network or synthetic profiles."""
import argparse
import collections
import copy
import csv
import hashlib
import json
import statistics
from pathlib import Path

ROOT = Path('/root/workspace/pdblend-next-v1')
CAM = ROOT / 'campaign'
INPUTS = {}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    path = Path(path).resolve()
    INPUTS[str(path)] = sha(path)
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def verified_point(point):
    cp = read(point['checkpoint_path'])
    assert cp['row']['cell_id'] == point['cell_id']
    assert cp['row']['trace_sha256'] == point['trace_sha256']
    assert sha(cp['receipt']) == cp['receipt_sha256']
    receipt = read(cp['receipt'])
    assert receipt['measurement_valid'] and receipt['child_stopped']
    for path, digest in cp['artifacts'].items():
        assert sha(path) == digest
        INPUTS[path] = digest
    return cp


def timing(cp):
    path = Path(next(p for p in cp['artifacts'] if p.endswith('/control.jsonl')))
    rows = [json.loads(x) for x in path.read_text().splitlines()]
    counts = collections.Counter(r.get('kind') for r in rows)
    routes = collections.Counter(a['decode_id'] for r in rows if r.get('kind') == 'admission'
                                 for a in r['plan']['routes'])
    pairs = [('queue_to_forward', 'queued_s', 'forward_started_s'),
             ('forward_to_first', 'forward_started_s', 'first_token_s'),
             ('queue_to_first_planning', 'queued_s', 'first_planning_s')]
    result = {}
    for label, a, b in pairs:
        values = [r[b]-r[a] for r in rows if r.get('kind') == 'request_timing'
                  and r.get(a) is not None and r.get(b) is not None]
        result[label] = dict(n=len(values), median_s=statistics.median(values),
                             maximum_s=max(values))
    return dict(journal=str(path), journal_sha256=sha(path), kinds=dict(counts),
                actual_admission_routes=dict(routes), segments=result,
                topology_events=sum(v for k, v in counts.items() if 'topology' in k))


def prepare(out):
    out = out.resolve()
    assert not out.exists(), 'new output only'
    binding_path = CAM / 'B32B-five-system100-v1/binding.pdblend.r2.json'
    binding = read(binding_path)
    host = Path(binding['host_release'])
    for rel, digest in read(host/'manifest.json')['files'].items():
        p = host/rel
        assert sha(p) == digest
        INPUTS[str(p)] = digest
    source_path = CAM/'five-system-fixed-window-v1/sources/B32B/manifest.json'
    source = read(source_path)
    snapdir = CAM/'five-system-results-v4/actual-snapshot-005'
    manifest = read(snapdir/'manifest.json')
    assert sha(snapdir/'results.json') == manifest['files']['results.json']
    snapshot = read(snapdir/'results.json')
    old = {ds: read(path) for ds, path in binding['configs'].items()}
    original = old['alpaca']
    assert binding['model'] == '32b' and len(binding['instances']) == 2
    assert all(i['tp'] == 2 for i in binding['instances'])
    assert original['strategy'] == 'pdblend-joint' and original['allow_pd'] is False
    assert original['park_idle'] is False and original['node_gpus'] == list(range(8))
    assert not original.get('topology') and not original.get('retained_weights')
    assert not original['slow_topology'] and not original['dynamic_pools']
    policy = lambda c: {k:v for k,v in c.items() if k not in ('slo_ttft_s','slo_tpot_s')}
    assert all(policy(c) == policy(original) for c in old.values())
    profile = read(original['profiles'])
    assert sha(original['profiles']) == binding['files'][original['profiles']]
    third = dict(id='cap3b2', role='mixed', tp=2, gpus=[4,5], port=34702,
                 kv_port=55764, url='http://127.0.0.1:34702', container_name='pdb-v2-cap3b2')
    layout = original['instances']+[third]
    for ds, config in old.items():
        write(out/'static2'/f'{ds}.json', config)
        candidate = copy.deepcopy(config)
        candidate['instances'] = layout
        assert {k for k in candidate if candidate[k] != config[k]} == {'instances'}
        write(out/'static3'/f'{ds}.json', candidate)
    engine_path = CAM/'B32B-engine-v3-candidate-v2/engine-0.json'
    engine = read(engine_path)
    assert sha(engine_path) == binding['files'][str(engine_path)]
    assert (engine['tp'], engine['max_num_batched_tokens'], engine['max_num_seqs'],
            engine['max_model_len'], engine['retained_weights']) == (2,8192,32,8192,None)
    new_engine = copy.deepcopy(engine)
    new_engine.update(id=third['id'], port=third['port'], kv_port=third['kv_port'],
                      runtime_dir=str(out/'future-runtime'), initial_generation=0,
                      peers={i['id']:dict(host='127.0.0.1',tp=2,kv_port=i['kv_port']) for i in layout})
    write(out/'engine-third.future.json', new_engine)
    inventory_path = CAM/'B32B-engine-v3-candidate-v2/all-containers.after.json'
    inventory = read(inventory_path)
    template = next(i for i in inventory if i['Id'] == binding['instances'][0]['container']['id'])
    env = [e for e in template['Config']['Env'] if not e.startswith('CUDA_VISIBLE_DEVICES=')]
    env.append('CUDA_VISIBLE_DEVICES=4,5')
    engine_source = ROOT/'releases/io-v3-runtime/src/ecopadg/serving/engine.py'
    assert f'PYTHONPATH={engine_source.parents[2]}' in env
    assert sha(engine_source) == binding['files'][str(engine_source)]
    INPUTS[str(engine_source)] = sha(engine_source)
    endpoint = dict(host='127.0.0.1',tp=2,kv_port=third['kv_port'])
    deploy = dict(schema=1, execution_ready=False, future_binding=None,
        actual_new_identity=None, mode='static_three_resident_before_arrivals',
        original_pdb_binding=dict(path=str(binding_path),sha256=sha(binding_path)),
        original_two=[dict(id=i['id'],container=i['container'],future_started_at=None,
                           future_host_pid=None) for i in binding['instances']],
        third=third, third_config=str(out/'engine-third.future.json'),
        image=template['Image'], env=env, host_config=template['HostConfig'],
        entry_argv=['python3',str(engine_source),'--config',str(out/'engine-third.future.json')],
        engine_entry_sha256=sha(engine_source),
        original_engine_configs_unchanged=True, old_container_removal_allowed=False,
        register_new_peer_after_all_three_idle=[dict(instance=i['id'],path='/register-peer',
            json=dict(id=third['id'],peer=endpoint)) for i in layout[:2]],
        prepare_pairs=[dict(instance=a['id'],peers=[b['id'] for b in sorted(layout,key=lambda x:x['id'])[j+1:]])
            for j,a in enumerate(sorted(layout,key=lambda x:x['id'])) if j < len(layout)-1],
        prerequisite='All B baseline main and scale terminal; fresh lease, no surviving serving child; no active container interruption.',
        readiness=['actual GPU ownership/free-memory and port bind checks',
                   'same two retained IDs restarted with fresh StartedAt and hostPID',
                   'new third container actual identity/model source/weights and measured freeKV',
                   'native v3 2-rank send/receive/buffer and owner/cache 8192/32 ACK',
                   'measured all-three ordinary full-output/cancel/cleanup gate',
                   'new isolated output/binding for all three instances; no old identity reuse'])
    write(out/'deployment.future.json', deploy)
    selected=[]; evidence=[]
    for ds in old:
        available=[p for p in snapshot['points'] if p['model']=='32b' and p['system']=='pdblend'
                   and p['phase']=='main' and p['status']=='completed' and p['dataset']==ds]
        for p in sorted(available,key=lambda p:p['rate_rps'])[::len(available)-1]:
            cp=verified_point(p)
            selected.append(dict(dataset=ds, rate_rps=p['rate_rps'], seed=701,
                trace_sha256=p['trace_sha256'], original_row=cp['row'],
                historical_static2_checkpoint=p['checkpoint_path'],
                historical_static2_checkpoint_sha256=sha(p['checkpoint_path'])))
        p=max(available,key=lambda p:p['rate_rps'])
        cp=verified_point(p)
        eco=next(q for q in snapshot['points'] if q['model']=='32b' and q['system']=='ecoserve'
            and q['phase']=='main' and q['dataset']==ds and q['rate_rps']==p['rate_rps'])
        verified_point(eco)
        metrics=('energy_j','energy_per_good_request_j','slo_attainment',
                 'completed_work_requests','n_requests','measurement_start_s','measurement_end_s','gpu_util_per_gpu')
        evidence.append(dict(dataset=ds,rate_rps=p['rate_rps'],
            pdb={k:p.get(k) for k in metrics},eco={k:eco.get(k) for k in metrics},timing=timing(cp)))
    declaration=dict(schema=1,cpu_only=True,gpu_executed=False,execution_ready=False,
        parent_source=dict(path=str(source_path),sha256=sha(source_path)),
        protocol_id=source['protocol_id'], deadline_s=binding['deadline_s'],
        primary_candidate='static3',reference='historical static2; future fresh static2 separately labeled',
        uniform_rule='Three pre-resident mixed TP2, all datasets and rates; original online JointPlanner unchanged.',
        changed_controller_keys=['instances'],profile_sha256=sha(original['profiles']),
        static4=dict(selected=False,reason='No empirical or justified counterfactual evidence that three is insufficient; not an automatic fallback.'),
        fixed_development_selection='Original declared minimum and maximum rate of each dataset, all six outcomes retained; no winner selection.',
        selected_six=selected, rows_original_unchanged=True,arrival_window_s=100,request_timeout_s=120,drain_timeout_s=120,
        energy=dict(primary='all 8 GPUs for complete measurement including failed requests',
                    operation='all 8 GPUs including cleanup; preserve unsuccessful attempts',
                    transitions='independent all8 startup/peer setup/correctness and restoration energy/time; unknown until measured',
                    amortization=None, idle_subtraction=False),
        dynamic_expansion_ready=False,
        dynamic_missing=['enabled controller strategy and topology configuration',
                         'same v3 engine lifecycle entry; original lifecycle uses old source and rm',
                         'complete actual retained-weight cache; binding has none',
                         'measured matching v3 2->3/4 transition time, energy, KV and temporary serving capacity',
                         'capacity-driven PDB proposal when original layout infeasible, with pending deadlines and bounded rollback',
                         'fresh identity registry and new output proof for added replicas'],
        deployment=str(out/'deployment.future.json'),evidence=evidence)
    write(out/'declaration.json',declaration)
    assert all(sha(p)==h for p,h in INPUTS.items()), 'input bytes changed while reading'
    write(out/'input-sha256.json',INPUTS)
    return declaration


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();result=prepare(args.out)
    print(json.dumps(dict(execution_ready=False,candidate='static3',selected=len(result['selected_six']),
                          input_files=len(INPUTS),out=str(args.out.resolve()))))
