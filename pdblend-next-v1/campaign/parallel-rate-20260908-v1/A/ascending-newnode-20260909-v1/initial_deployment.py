"""Explicit new-A deployment authorization around the unchanged measured deployer.

Only --run starts the two initial native services. Initial service deployment is
not dynamic-capacity qualification and never enables the 177-cell queue itself.
"""
import argparse
import asyncio
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REPO = ROOT.parents[1]
ENGINE_SOURCE = REPO / 'releases/io-v3-runtime'
HOST = ROOT / 'hosts/14b-capacity-p12'
COMMON = ROOT / 'common/execution-until-complete-v1'
HELPER = ROOT / 'common/distributed14b-deployment-v1/deploy.py'
RETAINED = Path('/root/workspace/pdblend/new-results/campaigns/three-pool-v2/weights/aeabfaf47f4941e6ba56d5e20d27d055')
IMAGE = 'sha256:0bb51d143b7fcaaea2e794dd6e207cf4165a4f21522a2e932a4bd4a117074bc2'
NODE = 'Anew20260909'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def ref(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_identity(check_host=True):
    identity = read(HERE / 'node-identity.json')
    require(identity['node'] == NODE, 'wrong new-node identity')
    if check_host:
        require(socket.gethostname() == identity['actual_hostname'], 'wrong actual new-A host')
        raw = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'],
                             check=True, capture_output=True, text=True).stdout
        observed = [(int(line.split(',')[0]), line.split(',')[1].strip()) for line in raw.strip().splitlines()]
        require(observed == [(g['index'], g['uuid']) for g in identity['GPUs']], 'new-A GPU UUIDs differ')
    return identity


def validate_spec(spec, *, check_host=True):
    """Replace only old B/C authorization; retain concrete deployment invariants."""
    identity = check_identity(check_host)
    declaration_ref = ref(HERE / 'declaration.json')
    declaration = read(declaration_ref['path'])
    require(declaration['node'] == NODE and declaration['counts']['initial_conditional_runs_upper_bound'] == 177,
            'new-A declaration changed')
    require(declaration['historical_A_checkpoint_reuse_count'] == 0 and
            declaration['historical_A_capacity_certificate_reuse_allowed'] is False, 'old A identity leaked')
    require(spec['protocol_id'] == 'per-dataset-slo-five-system-fixed-window-v1' and spec['model'] == '14b'
            and spec['stage'] == 'pdblend' and spec['node'] == NODE, 'initial new-A PDB stage only')
    require(spec['hostname'] == identity['actual_hostname'], 'new-A physical hostname differs')
    jobs_ref = spec['redistribution_jobs']
    require(sha(jobs_ref['path']) == jobs_ref['sha256'], 'new-A job reference changed')
    jobs = read(jobs_ref['path'])
    require(jobs['parent'] == declaration_ref and jobs['node'] == NODE and jobs['model'] == '14b',
            'deployment not bound to fresh new-A declaration')
    require(jobs['role'] == 'initial_two_for_fresh_dynamic_qualification', 'fixed-two performance not authorized')
    require(spec['files'].get(jobs_ref['path']) == jobs_ref['sha256'], 'jobs not frozen')
    require(spec['host_manifest'] == ref(HOST / 'manifest.json') == jobs['common_controller_manifest'],
            'new-A P12 host source differs')
    require(spec['host_release'] == str(HOST) and spec['common_dir'] == str(COMMON), 'foreign host/measurement executor')
    require(spec['source_entry'] == str(ENGINE_SOURCE / 'src/ecopadg/serving/engine.py'), 'native source differs')
    require([(i['tp'], i['gpus']) for i in spec['instances']] == [(1, [6]), (1, [7])], 'initial TP1 geometry differs')
    names = [i['container_name'] for i in spec['instances']]
    require(len(set(names)) == 2 and all(n.startswith('slo90-anew20260909-initial-') for n in names), 'owned container namespace required')
    require(spec['deployment_budget_s'] == 720 and spec['cleanup_budget_s'] == 120
            and spec['campaign_deadline_s'] is None and spec['remove_containers'] is False
            and spec['preserve_previous_containers'] is True, 'deployment/cleanup/retention policy differs')
    for path, digest in spec['files'].items():
        require(sha(path) == digest, 'frozen deployment source differs: ' + path)
    for path, item in spec['large_inputs'].items():
        require(adapter.adapter.stat_identity(path) == item['stat'], 'target-local retained weights changed')
    for instance in spec['instances']:
        cfg = read(instance['config'])
        require(Path(instance['config']).is_relative_to(HERE / 'deployment-spec-001'), 'config outside new-A namespace')
        require(instance['image'] == IMAGE and cfg['id'] == instance['id'] and cfg['tp'] == 1
                and cfg['model'] == '/models/Qwen2.5-14B-Instruct' and cfg['max_model_len'] == 8192
                and cfg['max_num_seqs'] == 32 and cfg['max_num_batched_tokens'] == 8192,
                'native image/model/budget differs')
        require(instance['native_kind'] == 'v3' and instance['scheduler_cache_count'] == 1
                and instance['service_budget_tokens'] == 2048 and instance['restore_budget_tokens'] == 8192,
                'native scheduling contract differs')
        require(instance['command'] == ['python3', '-m', 'ecopadg.serving.engine', '--config', instance['config']],
                'unexpected native command')
        require('PYTHONPATH=' + str(ENGINE_SOURCE / 'src') in instance['environment'], 'native source import path differs')
        require(instance['expected_provenance']['source_files_at_import'], 'native provenance freeze absent')
    return True


module_spec = importlib.util.spec_from_file_location('newnode_original_measured_deployer', HELPER)
adapter = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(adapter)
# The historical deployer hard-codes B/C authorization. The new validator above
# binds the new child declaration, actual hostname, GPU UUIDs and owned paths.
# Its measured engine lifecycle, ordinary gate, all-eight energy and cleanup are
# reused without modifying any historical source file.
adapter.validate_spec = validate_spec


def prepare():
    identity = check_identity()
    preflight = read(HERE / 'cpu-preflight-result-001.json')
    require(preflight['passed'] and preflight['checks']['node_identity']['hostname'] == identity['actual_hostname'],
            'actual new-A CPU preflight absent')
    root = HERE / 'deployment-spec-001'
    require(not root.exists(), 'fresh deployment spec required')
    retained_manifest = read(RETAINED / 'manifest.json')
    rank = retained_manifest['ranks'][0]
    require(retained_manifest['complete'] and retained_manifest['tp'] == 1 and len(retained_manifest['ranks']) == 1,
            'native TP1 retained manifest required')
    actual_rank_sha = sha(RETAINED / 'rank-0.safetensors')
    require(actual_rank_sha == rank['sha256'], 'retained tensor bytes differ')
    required = {}
    for p in [RETAINED / 'manifest.json', RETAINED / 'rank-0.json', RETAINED / 'rank-0.safetensors',
              *[Path('/root/workspace/models/Qwen2.5-14B-Instruct') / n for n in
                ('config.json', 'tokenizer.json', 'tokenizer_config.json', 'model.safetensors.index.json')]]:
        required[str(p)] = dict(sha256=actual_rank_sha if p.name == 'rank-0.safetensors' else sha(p), size=p.stat().st_size)
    save(root / 'required-inputs.json', dict(schema='newnode14B-source-bytes-only-v1', files=required,
         source_bytes_verified=True, physical_qualification_granted=False))
    running = subprocess.run(['docker', 'ps', '-q'], check=True, capture_output=True, text=True).stdout.split()
    require(not running, 'new-A initial deployment requires no running containers')
    save(root / 'idle-observation.json', dict(hostname=identity['actual_hostname'], observed_s=time.time(),
         node_identity=ref(HERE / 'node-identity.json'), running_containers=running,
         CPU_only=True, actual_GPU_free_memory_and_process_check_repeated_by_measured_deployer=True))
    jobs = dict(parent=ref(HERE / 'declaration.json'), node=NODE, model='14b',
                common_controller_manifest=ref(HOST / 'manifest.json'),
                role='initial_two_for_fresh_dynamic_qualification', fixed_two_performance_authorized=False)
    save(root / 'jobs.json', jobs)
    release = dict(deployment_root=str(root), model_root='/root/workspace/models', retained_weights=str(RETAINED),
        engine_source_release=str(ENGINE_SOURCE), host_releases={'pdblend':str(HOST), 'baselines':str(HOST)},
        common_dir=str(COMMON), required_inputs=ref(root / 'required-inputs.json'),
        verified_large_inputs={str(RETAINED / 'rank-0.safetensors'):dict(sha256=actual_rank_sha,
            stat=adapter.adapter.stat_identity(RETAINED / 'rank-0.safetensors'))},
        container_prefix='slo90-anew20260909-initial', node=NODE, redistribution_jobs=ref(root / 'jobs.json'))
    reference = adapter.prepare_spec('pdblend', release, identity['actual_hostname'], root / 'idle-observation.json')
    spec = read(reference['path'])
    for p in [HERE / 'node-identity.json', HERE / 'declaration.json', Path(__file__), HELPER,
              HERE / 'cpu-preflight-result-001.json', root / 'required-inputs.json']:
        spec['files'][str(p)] = sha(p)
    # A separate immutable successor adds the new-node adapter/proofs to the
    # unchanged generic preparation result; the historical source is untouched.
    save(root / 'frozen-spec.json', spec)
    validate_spec(spec)
    save(root / 'cpu-validation.json', dict(passed=True, cpu_only=True, actual_GPU_work_started=False,
        retained_weights_sha256=actual_rank_sha, spec=ref(root / 'frozen-spec.json'),
        deployment_only_not_dynamic_capacity_or_performance_qualification=True,
        adapter_source=ref(Path(__file__)), unchanged_measured_deployer=ref(HELPER)))
    print(json.dumps(ref(root / 'frozen-spec.json')))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--spec', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if args.prepare:
        require(not args.run and args.spec is None, 'prepare performs no hardware work')
        prepare()
        return
    require(args.spec is not None, 'explicit frozen new-A spec required')
    validate_spec(read(args.spec))
    if not args.run:
        print(json.dumps(dict(passed=True, cpu_only=True, GPU_work_started=False)))
        return
    require(args.out is not None and args.out.is_relative_to(HERE) and not args.out.exists(), 'fresh owned output required')
    require('PDBLEND_NODE_LOCK_FD' not in os.environ, 'fresh process must acquire its own lease')
    adapter.adapter.load_runtime(HOST, COMMON)
    from ecopadg.serving.campaign import node_lease
    async def execute(lease):
        loop = asyncio.get_running_loop()
        task = asyncio.current_task()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, task.cancel)
        return await adapter.execute(args.spec, args.out, run=True, lease=lease)
    with node_lease() as lease:
        result = asyncio.run(execute(lease))
    print(json.dumps(dict(complete=result['complete'], measurement_valid=result['measurement_valid'],
                         binding=result['binding_base'], dynamic_capacity_qualified=False,
                         performance_queue_started=False)))


if __name__ == '__main__':
    main()
