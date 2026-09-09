"""Generate finite, calibration-only baseline campaigns; restore each probe.

Generation is CPU-only and fails closed on missing measurement evidence. The
explicit ``restore`` and ``run`` commands are GPU stages under the node lease.
Predicted configuration candidates never count as calibrated capacities.
"""
import argparse
import asyncio
from collections import Counter, defaultdict
from dataclasses import asdict
import json
import math
from pathlib import Path
import time

from .budget import read_budget
from .baselines import DynamoLLMPolicy
from .dynamo import SHAPES, dominates
from .evidence import sha256
from .frequency import verify_frozen_costs
from .profiles import ProfileStore
from .topology import InstanceSpec, validate_layout


BASELINES = ('mixed', 'mixed_dvfs', 'distserve', 'ecoserve', 'dynamollm')
DATASETS = ('alpaca', 'sharegpt', 'longbench')
MODEL = '/models/Qwen2.5-14B-Instruct'
MODEL_NAME = 'Qwen2.5-14B-Instruct'


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False))


def spec(value):
    return InstanceSpec(value.get('instance_id', value.get('id')), value['tp'],
        tuple(value['gpus']), value['port'], value['kv_port'], value.get('role', 'mixed'))


def physical(value):
    """Roles/generations are restored by the runtime, without process restart."""
    return value.instance_id, value.tp, value.gpus, value.port, value.kv_port


def positive(value, name):
    if not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
        raise ValueError(name + ' must be finite and positive')
    return value


def verify_artifacts(artifacts):
    if not artifacts or any(not Path(p).is_absolute() or sha256(p) != digest
                            for p, digest in artifacts.items()):
        raise ValueError('missing or changed measurement source fingerprint')


def dynamo_pools(records, instance_ids, input_cuts=(255, 1023), output_cuts=(99, 349)):
    """Initialize from calibration histories, retaining all nine logical types.

    Sparse native demand spills to a componentwise larger resident pool; LL
    absorbs rounding and unseen shapes. No future trace/request output is read.
    """
    if not records or not instance_ids:
        raise ValueError('calibration histories and physical instances required')
    policy = DynamoLLMPolicy(input_cuts=input_cuts, output_cuts=output_cuts)
    counts = Counter(policy.classify(r['input_tokens'], r['output_tokens']) for r in records)
    assigned = ['LL']
    for _ in instance_ids[1:]:
        choices = [shape for shape in SHAPES if shape != 'LL' and counts[shape]]
        if not choices:
            assigned.append('LL')
        else:
            assigned.append(max(choices, key=lambda s: (counts[s] / (assigned.count(s)+1), s)))
    mapping = dict(zip(instance_ids, assigned))
    # This explicit carry table describes the single-node fragmentation rule;
    # online resident_reassignment applies its normal 1800-second policy.
    spill = {shape: next(s for s in SHAPES if s in assigned and dominates(s, shape))
             for shape in SHAPES}
    return mapping, dict(counts={s: counts[s] for s in SHAPES}, initial_spill=spill,
                        periods_s=dict(DynamoLLMPolicy.PERIODS))


def valid_transition_output(measurement):
    targets=sorted(measurement['target_tps']);checks=measurement.get('output_checks',[])
    return bool(measurement.get('result',{}).get('committed') and
        (measurement.get('unaffected_requests_completed',0)>0 or
         measurement.get('unaffected_requests_crossing_interval',0)>0) and
        sorted(c.get('tp',0) for c in checks)==targets and all(c.get('matches') for c in checks))


def validated_inputs(manifest):
    """Check existing evidence without synthesizing missing TP/cost points."""
    image = manifest['image']
    if len(image) != 71 or not image.startswith('sha256:'):
        raise ValueError('immutable Docker image SHA256 ID required')
    try:
        int(image[7:], 16)
    except ValueError as exc:
        raise ValueError('invalid image digest') from exc
    profiles = read(manifest['profiles'])
    if (profiles.get('engine_image') != image or profiles.get('model') != MODEL_NAME
            or profiles.get('status') != 'validated_envelope' or any(profiles.get(k) is not True for k in
                ('frequency_commands_verified', 'heldout_calibration_complete',
                 'mixed_interference_measured', 'resident_idle_measured',
                 'instant_prefill_calibration_complete','instant_heldout_calibration_complete'))):
        raise ValueError('certified hardware profiles with held-out/interference/residency evidence required')
    ProfileStore.load(manifest['profiles'])
    artifacts = dict(profiles['certification_artifacts'])
    verify_artifacts(artifacts)
    transfers = read(manifest['transfers'])
    if (not transfers.get('certified') or transfers.get('engine_image') != image or not transfers.get('links')
            or transfers.get('instant_power_costs_verified') is not True
            or transfers.get('receiver_transfer_energy_included') is not True):
        raise ValueError('certified same-image transport costs required')
    verify_artifacts(transfers['certification_artifacts'])
    artifacts.update(transfers['certification_artifacts'])
    search = read(manifest['search'])
    if search.get('status') != 'predicted_candidates_only':
        raise ValueError('expected explicit predicted candidates, never historical results')
    verify_artifacts(search['artifacts'])
    for field in ('profiles', 'transfers', 'interconnect'):
        if search['artifacts'].get(str(Path(manifest[field]).resolve())) != sha256(manifest[field]):
            raise ValueError('configuration search used different ' + field)
    artifacts.update(search['artifacts'])
    template = read(manifest['engine_template'])
    if template.get('model') != MODEL or template.get('max_model_len') != 8192:
        raise ValueError('fixed Qwen2.5-14B/8192-token engine template required')
    costs = []; evidence = []
    for path in manifest['frequency_costs']:
        costs.extend(read(path)); evidence.append(str(Path(path).parent / 'raw.json'))
        artifacts[str(Path(path).resolve())] = sha256(path)
    for path in evidence:
        artifacts[str(Path(path).resolve())] = sha256(path)
    topology_costs = read(manifest['topology_costs'])
    raw_path = Path(manifest['topology_costs']).parent / 'raw.json'
    raw = read(raw_path); digest = sha256(raw_path)
    if (not raw.get('complete') or not raw.get('passed') or raw.get('sampling_error')
            or not raw.get('transactions') or not topology_costs):
        raise ValueError('completed physical transition measurements required')
    from .measurement import power_evidence
    power = power_evidence(raw.get('power_samples',[]),raw.get('power_source'),raw.get('power_metadata'))
    if not power['power_source_verified'] or power['power_mode']!='instant':
        raise ValueError('physical transition costs require verified instantaneous power')
    provenance = [p for field in ('provenance_before', 'provenance_after') for p in raw.get(field, {}).values()]
    if not provenance or any(p.get('image_id') != image or p.get('model') != MODEL
                             or p.get('engine_version') != '0.9.2' for p in provenance):
        raise ValueError('physical transition image mismatch')
    final_instances=[spec(s) for s in raw.get('live_instances',[])]
    validate_layout(final_instances,range(8))
    if not final_instances or {s.instance_id for s in final_instances}!=set(raw.get('provenance_after',{})):
        raise ValueError('physical transition final layout evidence is incomplete')
    for cost in topology_costs:
        positive(cost['duration_upper_s'], 'measured topology duration')
        positive(cost['energy_upper_j'], 'measured topology energy')
        measurements = [r for r in raw['transactions'] if sorted(r['source_tps']) == sorted(cost['source_tps'])
                        and sorted(r['target_tps']) == sorted(cost['target_tps'])]
        if (cost.get('source_sha256') != digest or not measurements
                or cost['duration_upper_s'] < max(r['finished_s']-r['started_s'] for r in measurements)
                or cost['energy_upper_j'] < max(r['total_node_energy_j'] for r in measurements)
                or any(not valid_transition_output(r) for r in measurements)):
            raise ValueError('topology cost missing, below measured bound, or output unverified')
    artifacts[str(raw_path.resolve())] = digest
    weights_manifest = Path(manifest['retained_weights'])/'manifest.json'
    weights = read(manifest['weights_evidence'])
    if not weights.get('passed') or not any(r.get('bit_exact') and
            r.get('manifest_sha256') == sha256(weights_manifest) for r in weights.get('retained',[])):
        raise ValueError('retained weights lack complete original-model equality evidence')
    for path in (weights_manifest,Path(manifest['weights_evidence'])):
        artifacts[str(path.resolve())] = sha256(path)
    for field in ('profiles', 'transfers', 'search', 'interconnect', 'topology_costs', 'engine_template'):
        path = Path(manifest[field]).resolve(); artifacts[str(path)] = sha256(path)
    return profiles, transfers, search, template, costs, evidence, topology_costs, artifacts


def generate(manifest, out):
    """Generate only files; never instantiate a GPU, clock, Docker or HTTP client."""
    out = Path(out).resolve()
    if out.exists():
        raise ValueError('refusing to overwrite a calibration setup')
    profiles, transfers, search, template, costs, evidence, topo_costs, artifacts = validated_inputs(manifest)
    campaign_root = Path(manifest['campaign_root']).resolve()
    budget = read_budget(campaign_root)
    positive(budget['started_s'], 'original campaign start time')
    remaining = budget['remaining_s']
    allocation = positive(manifest['calibration_budget_s'], 'calibration budget')
    if allocation > remaining-60:
        raise ValueError('requested calibration allocation exceeds the effective authorized campaign deadline')
    initial = [spec(s) for s in manifest['initial_instances']]
    validate_layout(initial, range(8))
    max_candidates = manifest.get('max_candidates_per_pair', 1)
    if not isinstance(max_candidates, int) or max_candidates < 1:
        raise ValueError('positive candidate cap required')
    if (manifest.get('target',.99) != .99 or not 1 <= manifest.get('max_trials',7) <= 7
            or not 1 <= manifest.get('probe_requests',64) <= 128):
        raise ValueError('calibration uses a 99% SLO target, at most seven probes, and full confirmation')
    selected = []; seen = set(); pair_counts = Counter(); records = {}
    for entry in manifest['entries']:
        system, dataset, index = entry['system'], entry['dataset'], entry['candidate_index']
        if system not in BASELINES or dataset not in DATASETS or not isinstance(index, int) or index < 0:
            raise ValueError('only independent baselines and calibration dataset candidates are accepted')
        key = (system, dataset, index)
        pair_counts[(system, dataset)] += 1
        if key in seen or pair_counts[(system, dataset)] > max_candidates:
            raise ValueError('duplicate candidate or explicit per-pair budget cap exceeded')
        seen.add(key)
        if dataset not in records:
            corpus_path = Path(manifest['corpus']) / (dataset+'.json')
            if search['artifacts'].get(str(corpus_path.resolve())) != sha256(corpus_path):
                raise ValueError('configuration search used a different corpus')
            # Deliberately project only calibration. Development/formal entries
            # never feed shape classes, initial rates, priors or selection.
            records[dataset] = read(corpus_path)['calibration']
            if len(records[dataset]) < 128:
                raise ValueError('at least 128 independent calibration records required')
        family = 'distserve' if system == 'distserve' else 'mixed'
        choices = search['datasets'][dataset][family]
        if index >= len(choices):
            raise ValueError(f'{system}/{dataset} candidate {index} is unavailable')
        candidate = choices[index]
        positive(candidate['capacity_rps'], 'predicted candidate rate')
        if any(candidate[k] > template.get('max_num_seqs',32)
               for k in ('prefill_batch','decode_batch') if k in candidate) or candidate.get('batch',1)>template.get('max_num_seqs',32):
            raise ValueError('candidate batch exceeds the measured engine sequence limit')
        groups = candidate['gpus']
        degrees = ([candidate['prefill_tp']]*candidate['prefill_count']
                   + [candidate['decode_tp']]*candidate['decode_count']) if family == 'distserve' else [candidate['tp']]*candidate['instance_count']
        if len(groups) != len(degrees) or any(len(g) != tp for g, tp in zip(groups, degrees)):
            raise ValueError('candidate contains an invalid physical layout')
        layout = tuple(sorted((tp, tuple(g)) for tp, g in zip(degrees, groups)))
        selected.append(dict(entry, family=family, candidate=candidate, layout=layout))
    if not selected:
        raise ValueError('explicit nonempty baseline candidate selection required')
    grouped = defaultdict(list)
    for item in selected: grouped[item['layout']].append(item)
    prepare_limit = positive(manifest.get('preparation_stage_limit_s', 600), 'preparation limit')
    cell_limit = positive(manifest.get('calibration_stage_limit_s', 1200), 'calibration limit')
    upper = 60 + len(grouped)*prepare_limit + sum(positive(x.get('limit_s', cell_limit), 'entry limit') for x in selected)
    if upper > allocation:
        raise ValueError(f'stage upper bounds {upper:g}s exceed allocation {allocation:g}s; explicitly trim candidates')
    stage_names = []; stages = []; configs = []; generated = {}; used_candidates = set()
    all_specs = list(initial); frequency_freeze = dict(files=artifacts,
        groups=dict(profiles=list(artifacts)), identities=dict(engine_image=manifest['image']))
    template = dict(template, operation_timeout_s=45, transfer_buffer_bytes=4*1024**3,
        verify_transport=False, validated_tp_pairs=sorted({(t['source_tp'],t['target_tp']) for t in transfers['links']}))
    template_path = out/'engine-template.json'; generated[template_path] = template
    for group_index, (layout, items) in enumerate(grouped.items()):
        instances = [InstanceSpec(f'cal{group_index}_{n}', tp, gpus,
            24000+group_index*128+n, 28000+group_index*128+n*8) for n, (tp, gpus) in enumerate(layout)]
        validate_layout(instances, range(8)); all_specs.extend(instances)
        restore = dict(instances=[asdict(s) for s in instances], initial_instances=[asdict(s) for s in initial],
            image=manifest['image'], engine_template=str(template_path), retained_weights=manifest['retained_weights'],
            ownership_root=str(out))
        restore_path = out/f'layout-{group_index}.restore.json'; generated[restore_path] = restore
        prepare_name = f'calibration-prepare-{group_index}'
        stages.append(dict(name=prepare_name, limit_s=prepare_limit, requires=stage_names[-1:],
            argv=['python3','-m','ecopadg.serving.calibration_setup','restore','--manifest',str(restore_path),
                  '--out',str(out/f'prepare-{group_index}')]))
        stage_names.append(prepare_name)
        # Shared physical engines; Dynamo is last because it can change them.
        for item in sorted(items, key=lambda x: (x['system']=='dynamollm', x['dataset'], BASELINES.index(x['system']), x['candidate_index'])):
            system, dataset = item['system'], item['dataset']; candidate = item['candidate']
            key = f'{system}-{dataset}-{item["candidate_index"]}'
            endpoints = [s.endpoint() for s in instances]
            if system == 'distserve':
                p_groups = {tuple(g) for g in candidate['gpus'][:candidate['prefill_count']]}
                endpoints = [dict(i, role='prefill' if tuple(i['gpus']) in p_groups else 'decode') for i in endpoints]
            outputs = sorted(r['output_tokens'] for r in records[dataset])
            config = dict(strategy=system, port=18080, model_name=MODEL_NAME,
                profiles=str(Path(manifest['profiles']).resolve()), instances=endpoints, node_gpus=list(range(8)),
                slo_ttft_s=manifest.get('slo_ttft_s',5), slo_tpot_s=manifest.get('slo_tpot_s',.1),
                output_prior=outputs[min(len(outputs)-1, math.ceil(.9*len(outputs))-1)],
                max_pending=256, manage_clocks=True, prepare_peers=True, park_idle=True,
                decision_budget_s=positive(manifest.get('decision_budget_s',.01),'planning decision budget'),
                power_mode='instant',
                interconnect=str(Path(manifest['interconnect']).resolve()), allow_unprofiled_fallback=False,
                journal=str(out/'unused-control.jsonl'))
            if system == 'distserve':
                config.update(distserve_prefill_batch=candidate['prefill_batch'], distserve_decode_batch=candidate['decode_batch'],
                    transfers=transfers['links'], transfer_evidence=str(Path(manifest['transfers']).resolve()))
            if system in ('mixed_dvfs','dynamollm'):
                config.update(frequency_costs=costs, frequency_evidence=evidence)
            pool_proof = None
            if system == 'dynamollm':
                mapping, pool_proof = dynamo_pools(records[dataset], [i['id'] for i in endpoints])
                config.update(dynamo_assignments=mapping, dynamo_input_cuts=[255,1023], dynamo_output_cuts=[99,349],
                    topology_costs=topo_costs, retained_weights=manifest['retained_weights'],
                    topology=dict(runtime_dir=str(out/'dynamic-runtime'), image=manifest['image'], engine_template=str(template_path)))
            if system in ('mixed_dvfs','dynamollm'):
                verify_frozen_costs(config, profiles, frequency_freeze)
            config_path = out/(key+'.config.json'); generated[config_path] = config
            calibration = dict(corpus=str(Path(manifest['corpus']).resolve()),
                target=manifest.get('target',.99), max_trials=manifest.get('max_trials',7),
                probe_requests=manifest.get('probe_requests',64), request_timeout_s=manifest.get('request_timeout_s',120),
                entries=[dict(dataset=dataset, config=str(config_path),
                    initial_rate=positive(item.get('initial_rate', candidate['capacity_rps']*.6),'initial rate'))],
                restoration=str(restore_path), input_evidence=str(out/'input-evidence.json'))
            calibration_path = out/(key+'.calibration.json'); generated[calibration_path] = calibration
            name = 'calibrate-'+key
            stages.append(dict(name=name, requires=[stage_names[-1]], limit_s=item.get('limit_s',cell_limit),
                argv=['python3','-m','ecopadg.serving.calibration_setup','run','--manifest',str(calibration_path),
                      '--out',str(out/key)]))
            stage_names.append(name); used_candidates.add((system,dataset,item['candidate_index']))
            configs.append(dict(system=system,dataset=dataset,candidate_index=item['candidate_index'],
                config=str(config_path),result=str(out/key/'summary.json'),physical_gpus=[list(s.gpus) for s in instances],
                idle_unallocated_gpus=sorted(set(range(8))-{g for s in instances for g in s.gpus}),
                predicted_capacity_rps=candidate['capacity_rps'],status='awaiting_independent_calibration',
                dynamo_initialization=pool_proof))
    # Explicitly authorize all generated starting IDs when moving between groups.
    for path, value in generated.items():
        if path.name.endswith('.restore.json'):
            value['initial_instances'] = [asdict(s) for s in all_specs]
    unmeasured = []
    for system in BASELINES:
        family = 'distserve' if system == 'distserve' else 'mixed'
        for dataset in DATASETS:
            for index, candidate in enumerate(search['datasets'][dataset][family]):
                if (system,dataset,index) not in used_candidates:
                    unmeasured.append(dict(system=system,dataset=dataset,candidate_index=index,
                        predicted_capacity_rps=candidate['capacity_rps'],
                        reason='outside explicit budgeted selection; no calibrated result or infeasibility claim'))
    out.mkdir(parents=True)
    for path, value in generated.items(): write(path,value)
    artifacts[str(template_path)] = sha256(template_path)
    for path in generated: artifacts[str(path)] = sha256(path)
    write(out/'input-evidence.json',dict(artifacts=artifacts, purpose='calibration setup inputs, not a formal implementation freeze'))
    stages.append(dict(name='summarize-baseline-calibration',requires=[stage_names[-1]],limit_s=60,gpu=False,
        argv=['python3','-m','ecopadg.serving.calibration_setup','summarize','--manifest',str(out/'selection.json'),
              '--out',str(out/'summary.json')]))
    write(out/'campaign.json',dict(output=str(campaign_root), budget_s=budget['limit_s'],stages=stages))
    result = dict(status='prepared_not_calibrated', selected=configs, unmeasured=unmeasured,
        input_evidence=str(out/'input-evidence.json'),
        original_deadline_s=budget['original_deadline_s'], effective_deadline_s=budget['deadline_s'],
        budget_revision_seq=budget['revision_seq'], authorization_sha256=budget['authorization_sha256'],
        stage_upper_bound_s=upper, allocation_s=allocation, measurement_gpus=list(range(8)),
        startup_and_probe_restoration='separate preparation artifacts; in-run reconfiguration remains in serving energy',
        limitations=['short capacity probes do not certify DynamoLLM slow-cycle mechanisms',
                    'each baseline/candidate is calibrated separately; no PDBlend result sets a load',
                    'unmeasured configurations remain unknown; selection is not a global optimum claim'])
    write(out/'selection.json',result)
    return result


async def docker(*argv):
    process = await asyncio.create_subprocess_exec('docker',*argv,stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.PIPE)
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(),30)
    except BaseException:
        if process.returncode is None: process.kill(); await process.wait()
        raise
    if process.returncode:
        raise RuntimeError(stderr.decode(errors='replace')[-2000:])
    return stdout.decode()


def inspected_instance(raw, manifest):
    """Bind mutable names to this experiment's image, config and GPU allocation."""
    name = raw['Name'].lstrip('/')
    if not name.startswith('pdb-v2-') or raw['Image'] != manifest['image']:
        raise ValueError('unexpected experiment container image or name')
    command = raw['Config']['Cmd']
    if command[:3] != ['python3','-m','ecopadg.serving.engine'] or '--config' not in command:
        raise ValueError('container is not a declared PDBlend engine')
    path = Path(command[command.index('--config')+1]).resolve()
    environment = dict(item.split('=',1) for item in raw['Config']['Env'] if '=' in item)
    visible = environment.get('CUDA_VISIBLE_DEVICES','')
    try:
        devices = [int(g) for g in visible.split(',')]
    except ValueError as exc:
        raise ValueError('container has no explicit node GPU environment') from exc
    config = read(path)
    # Engine templates intentionally identify devices through Docker's CUDA
    # environment; a gpus field is optional and must agree when present.
    value = spec(dict(config,gpus=config.get('gpus',devices)))
    if name != 'pdb-v2-'+value.instance_id:
        raise ValueError('container ID differs from its engine configuration')
    allowed = [spec(s) for s in manifest.get('initial_instances',[])+manifest['instances']]
    known = any(physical(value)==physical(s) for s in allowed)
    if not known and not path.is_relative_to(Path(manifest['ownership_root']).resolve()):
        raise ValueError('undeclared container outside calibration ownership; refusing removal')
    if visible != ','.join(map(str,value.gpus)):
        raise ValueError('container GPU environment differs from its declaration')
    validate_layout([value],range(8))
    return value


def unallocated_gpu_processes(gpus):
    """Real process counters, not power-threshold inference or a logical label."""
    if not gpus: return {}
    import pynvml
    pynvml.nvmlInit()
    try:
        result = {}
        for gpu in gpus:
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu)
            processes = list(pynvml.nvmlDeviceGetComputeRunningProcesses(handle))
            processes += list(pynvml.nvmlDeviceGetGraphicsRunningProcesses(handle))
            result[gpu] = sorted({p.pid for p in processes})
        if any(result.values()):
            raise RuntimeError('unallocated experiment GPUs still host processes: '+str(result))
        return result
    finally:
        pynvml.nvmlShutdown()


def current_engine_sources():
    return {str(p.resolve()):sha256(p) for p in Path(__file__).parent.glob('*.py')}


def engine_settings(config):
    defaults = dict(model=None,max_model_len=8192,max_num_seqs=32,max_num_batched_tokens=8192,
                    gpu_memory_utilization=.85,operation_timeout_s=45,transfer_buffer_bytes=2*1024**3,
                    verify_recompute=False,verify_transport=False,validated_tp_pairs=[[1,1]])
    return {key:config.get(key,value) for key,value in defaults.items()}


async def restore_layout(manifest, out):
    """Restore measured physical instances, leaving roles for runtime.initialize.

    Must run under the experiment node lease. Unknown containers fail closed;
    replacements never borrow a GPU outside the fixed eight-card node.
    """
    import aiohttp
    from .prepare import prepare
    desired = [spec(v) for v in manifest['instances']]
    validate_layout(desired,range(8)); out = Path(out)
    out.mkdir(parents=True,exist_ok=False)
    names = (await docker('ps','-a','--filter','name=^pdb-v2-','--format','{{.Names}}')).splitlines()
    if any(not name.startswith('pdb-v2-') for name in names):
        raise ValueError('unexpected container outside this experiment')
    inspected = json.loads(await docker('inspect',*names)) if names else []
    actual = [inspected_instance(value,manifest) for value in inspected]
    validate_layout(actual,range(8))
    settings = engine_settings(read(manifest['engine_template']))
    stale_template = {value.instance_id for value,raw in zip(actual,inspected)
        if engine_settings(read(raw['Config']['Cmd'][raw['Config']['Cmd'].index('--config')+1])) != settings}
    # Exited engines must restart even when their static allocation matches.
    running = {v['Name'].lstrip('/')[len('pdb-v2-'):] for v in inspected if v['State']['Running']}
    expected_source = await asyncio.to_thread(current_engine_sources)
    source_before = {}; stale = set(stale_template)
    # Every running engine must be drained before between-probe reconstruction.
    async with aiohttp.ClientSession(trust_env=False,timeout=aiohttp.ClientTimeout(total=5)) as session:
        for value in actual:
            if value.instance_id not in running: continue
            async with session.get(value.endpoint()['url']+'/runtime') as response:
                if response.status != 200: raise RuntimeError('cannot verify prior engine drain')
                state = await response.json()
            if any(state.get(k) for k in ('running','waiting','active','transfer_allocations')):
                raise RuntimeError('cannot restore physical layout while requests or KV remain')
            async with session.get(value.endpoint()['url']+'/provenance') as response:
                provenance = await response.json() if response.status==200 else {}
            source_before[value.instance_id] = provenance
            if provenance.get('source_files_at_import') != expected_source:
                stale.add(value.instance_id)
    write(out/'source.before.json',dict(expected=expected_source,instances=source_before,stale=sorted(stale),
        stale_template=sorted(stale_template),force_restart=manifest.get('force_restart',False)))
    keep = [s for s in desired if not manifest.get('force_restart',False) and s.instance_id not in stale
            and s.instance_id in running and any(physical(s)==physical(a) for a in actual)]
    removed = [s for s in actual if not any(physical(s)==physical(k) for k in keep)]
    added = [s for s in desired if s not in keep]
    restore = {k:manifest[k] for k in ('image','engine_template','retained_weights')}
    restore.update(add=[asdict(s) for s in added],remove=[asdict(s) for s in removed],keep=[asdict(s) for s in keep])
    write(out/'restore.json',restore)
    if added or removed:
        result = await prepare(restore,out)
        if not result.get('complete') or result.get('errors') or result.get('sampling_error'):
            raise RuntimeError('between-cell physical restoration incomplete')
    else:
        result = dict(complete=True,changed=False,energy_j=None,
            purpose='already matched and drained; no preparation interval or fabricated zero-energy measurement')
        write(out/'startup.json',result)
    unused = sorted(set(range(8))-{g for s in desired for g in s.gpus})
    proof = await asyncio.to_thread(unallocated_gpu_processes,unused)
    write(out/'unallocated-gpus.json',dict(at_s=time.time(),processes=proof,passed=True))
    source_after = {}
    async with aiohttp.ClientSession(trust_env=False,timeout=aiohttp.ClientTimeout(total=5)) as session:
        for value in desired:
            async with session.get(value.endpoint()['url']+'/provenance') as response:
                if response.status != 200: raise RuntimeError('cannot verify restored engine provenance')
                source_after[value.instance_id] = await response.json()
    write(out/'source.after.json',dict(expected=expected_source,instances=source_after))
    if (await asyncio.to_thread(current_engine_sources) != expected_source or
            any(p.get('source_files_at_import') != expected_source for p in source_after.values())):
        raise RuntimeError('restored engine imports differ from the stable source snapshot')
    return result


async def run(args):
    from .calibration import calibrate
    manifest = read(args.manifest)
    verify_artifacts(read(manifest['input_evidence'])['artifacts'])
    restoration = read(manifest['restoration'])
    async def before_cell(options):
        await restore_layout(restoration, options.out.with_name(options.out.name+'.preparation'))
    return await calibrate(args,before_cell=before_cell)


def summarize(selection, out):
    """Merge explicitly selected hardware calibrations, never predicted rates."""
    from .evidence import common_capacity,freeze_files
    from .calibration import implementation_sources
    if Path(out).exists(): raise ValueError('refusing to overwrite calibration summary')
    verify_artifacts(read(selection['input_evidence'])['artifacts'])
    results = []; gaps = []; snapshots = []; artifact_paths = {}
    for item in selection['selected']:
        path = Path(item['result'])
        if not path.exists():
            gaps.append('missing calibration: '+str(path)); continue
        summary = read(path)
        snapshot_path = path.parent/'source.before.json'
        snapshot = read(snapshot_path)
        snapshots.append(snapshot)
        artifact_paths[str(path.resolve())] = sha256(path)
        artifact_paths[str(snapshot_path.resolve())] = sha256(snapshot_path)
        rows = summary.get('results',[])
        if len(rows)!=1 or any(rows[0].get(k)!=item[k] for k in ('system','dataset','config')):
            raise ValueError('candidate calibration identity differs from explicit selection')
        if not summary.get('source_unchanged'):
            gaps.append('source changed during '+str(path))
        results.extend(rows)
    current = freeze_files(implementation_sources())
    unchanged = bool(snapshots) and all(s==current for s in snapshots)
    if not unchanged: gaps.append('baseline candidates did not run the same current implementation')
    capacities = None
    try: capacities = common_capacity(results)
    except ValueError as exc: gaps.append(str(exc))
    result = dict(passed=unchanged and capacities is not None and not gaps,source_unchanged=unchanged,
        results=results,common_capacity=capacities,gaps=gaps,artifacts=artifact_paths,
        scope='best independently confirmed tested candidate per baseline; omitted configurations remain unknown')
    write(out,result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('generate','restore','run','summarize'))
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    if args.action == 'generate':
        result = generate(read(args.manifest),args.out)
    elif args.action == 'summarize':
        result = summarize(read(args.manifest),args.out)
    else:
        from .campaign import node_lease
        with node_lease():
            result = asyncio.run(restore_layout(read(args.manifest),args.out) if args.action=='restore' else run(args))
    print(json.dumps({k:v for k,v in result.items() if k in ('status','passed','complete','stage_upper_bound_s')},allow_nan=False))


if __name__ == '__main__':
    main()
