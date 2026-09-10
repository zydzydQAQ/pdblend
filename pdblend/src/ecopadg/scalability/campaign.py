"""Resumable scaling campaign: PD split pilots, capacity confirmation, weak load.

One node is held exclusively. Every observation has a new immutable directory;
engineering failures stop the queue and are never replaced automatically.
"""
import argparse
import asyncio
import copy
import json
import socket
from pathlib import Path

from ecopadg.serving.campaign import node_lease

from .artifacts import object_hash, read_json, sha256, write_json
from .capacity import new_search, next_action, pilot_capacity_row, record_observation, weak_load_q
from .preflight import verify_freeze
from .protocol import FORMAL_SCALES, FORMAL_SEEDS, SYSTEMS, SLOS, build_config, protocol_dict
from .run import run_cell
from .workload import build_trace, load_pool


def key(dataset, system, n, p_count=None):
    return f'{dataset}-{system}-n{n}' + (f'-p{p_count}' if p_count is not None else '')


def new_campaign(config_path, pools, root, *, freeze_path=None):
    root = Path(root)
    if set(pools) != set(SLOS):
        raise ValueError('one explicit ShareGPT and one LongBench pool are required')
    config = read_json(config_path)
    # Read and validate every source before emitting a runnable campaign.
    for path in pools.values():
        load_pool(path)
    for dataset, scales in FORMAL_SCALES.items():
        for n in scales:
            for system in SYSTEMS:
                build_config(config, system=system, dataset=dataset, allocated_gpu_ids=list(range(n)),
                             fixed_pd_p_count=1 if system == 'fixed_pd' else None,
                             max_frequency_mhz=2520)
    searches = {}
    splits = {}
    for dataset, scales in FORMAL_SCALES.items():
        for n in sorted(set(scales) | {3, 4}):
            for system in ('pdblend', 'mixed'):
                searches[key(dataset, system, n)] = new_search(system=system, dataset=dataset, n_gpus=n)
            for p_count in range(1, n):
                splits[key(dataset, 'fixed_pd', n, p_count)] = dict(
                    new_search(system='fixed_pd', dataset=dataset, n_gpus=n), p_count=p_count)
    state = dict(schema='pdblend-scalability-campaign-v1', host_id=socket.gethostname(),
        protocol=protocol_dict(), config_path=str(Path(config_path).resolve()),
        config_sha256=sha256(config_path), pools={d: str(Path(p).resolve()) for d,p in pools.items()},
        pool_hashes={d: sha256(p) for d,p in pools.items()},
        freeze_path=str(Path(freeze_path).resolve()) if freeze_path else None,
        freeze_sha256=sha256(freeze_path) if freeze_path else None,
        searches=searches, split_searches=splits, selected_pd={}, weak_q={},
        weak_observations=[], observations=[], status='prepared', blocked_reason=None)
    if freeze_path:
        verify_campaign_inputs(state)
    root.mkdir(parents=True, exist_ok=False)
    write_json(root/'protocol.json', state['protocol'])
    write_json(root/'state.json', state)
    write_json(root/'inputs.json', {k:state[k] for k in ('config_path','config_sha256','pools','pool_hashes')})
    return state


def settle_selections(state):
    state = copy.deepcopy(state)
    grouped = {}
    for name, search in state['split_searches'].items():
        grouped.setdefault((search['dataset'], search['n_gpus']), []).append((name, search))
    for (dataset,n), rows in grouped.items():
        slot = key(dataset, 'fixed_pd', n)
        if slot not in state['searches'] and all(s.get('pilot_interval_complete') for _,s in rows):
            selected_name, selected = min(rows, key=lambda pair:(-pair[1]['lower_rate_rps'],pair[1]['p_count']))
            state['selected_pd'][slot] = dict(p_count=selected['p_count'], selection_search=selected_name,
                selection_rule='maximum pilot lower capacity; tie: smaller prefill count',
                candidates={name:pilot_capacity_row(s) for name,s in rows})
            state['searches'][slot] = copy.deepcopy(selected)
    for dataset in SLOS:
        required = [state['searches'].get(key(dataset,s,n)) for s in SYSTEMS for n in (3,4)]
        if all(s and s.get('pilot_interval_complete') for s in required):
            value = weak_load_q([pilot_capacity_row(s) for s in required], dataset=dataset)
            if dataset in state['weak_q'] and state['weak_q'][dataset] != value:
                raise ValueError('frozen weak-load normalization changed')
            state['weak_q'][dataset] = value
    return state


def next_task(state):
    if state['status'] == 'blocked':
        return dict(action='blocked', reason=state['blocked_reason'])
    state = settle_selections(state)
    for name, search in state['split_searches'].items():
        if not search.get('pilot_interval_complete'):
            return dict(next_action(search), search_key=name, search_group='split_searches',
                        p_count=search['p_count'], purpose='pd_split_selection')
    for name, search in state['searches'].items():
        if search['status'] == 'pilot':
            return dict(next_action(search), search_key=name, search_group='searches',
                p_count=search.get('p_count'), purpose='capacity_pilot')
    candidates = []
    for name, search in state['searches'].items():
        if search['status'] == 'confirm':
            action = next_action(search)
            index = FORMAL_SEEDS.index(action['seed'])
            rotated = SYSTEMS[index % 3:] + SYSTEMS[:index % 3]
            order = (index, action['boundary'], action['dataset'],action['n_gpus'],rotated.index(action['system']))
            candidates.append((order, dict(action, search_key=name, search_group='searches',
                                          p_count=search.get('p_count'), purpose='capacity_confirmation')))
    if candidates:
        return min(candidates,key=lambda item:item[0])[1]
    done = {(r['dataset'],r['system'],r['n_gpus'],r['seed']) for r in state['weak_observations']}
    for index, seed in enumerate(FORMAL_SEEDS):
        for dataset, scales in FORMAL_SCALES.items():
            for n in scales:
                for system in SYSTEMS[index % 3:] + SYSTEMS[:index % 3]:
                    if (dataset,system,n,seed) not in done:
                        p = state['selected_pd'].get(key(dataset,'fixed_pd',n),{}).get('p_count')
                        return dict(action='measure', stage='weak', dataset=dataset, system=system,
                            n_gpus=n, seed=seed, q=state['weak_q'][dataset],
                            rate_rps=n*state['weak_q'][dataset], p_count=p if system=='fixed_pd' else None,
                            purpose='weak_scaling')
    return dict(action='complete', confirmed=all(s.get('confirmed') for s in state['searches'].values()
                if s['n_gpus'] in FORMAL_SCALES[s['dataset']]))


def apply_observation(state, task, observation):
    result = settle_selections(state)
    expected = next_task(result)
    if task != expected:
        raise ValueError('stale task no longer matches the campaign state')
    observation = dict(observation, system=task['system'], dataset=task['dataset'], n_gpus=task['n_gpus'],
                       stage=task['stage'], seed=task['seed'], rate_rps=task['rate_rps'])
    result['observations'].append(observation)
    if task['stage'] == 'weak':
        result['weak_observations'].append(observation)
    else:
        result[task['search_group']][task['search_key']] = record_observation(
            result[task['search_group']][task['search_key']], observation)
    if observation.get('measurement_valid') is not True:
        result.update(status='blocked',blocked_reason=observation.get('error') or observation.get('audit_errors')
                      or 'engineering observation invalid; raw attempt retained')
    else:
        result['status'] = 'running'
    return settle_selections(result)


def verify_campaign_inputs(state):
    if state['host_id'] != socket.gethostname():
        raise ValueError('all GPU scales must run on the frozen host')
    for path,digest in [(state['config_path'],state['config_sha256']),
                        *[(path,state['pool_hashes'][d]) for d,path in state['pools'].items()]]:
        if sha256(path) != digest:
            raise ValueError('campaign input changed: '+path)
    if not state.get('freeze_path'):
        raise ValueError('campaign is prepared but lacks a fresh qualified source freeze')
    if sha256(state['freeze_path']) != state['freeze_sha256']:
        raise ValueError('source freeze changed')
    freeze = read_json(state['freeze_path'])
    errors = verify_freeze(freeze)
    if errors:
        raise ValueError('; '.join(errors[:5]))
    verify_freeze_binding(state, freeze)
    return freeze


def verify_freeze_binding(state, freeze):
    """A valid freeze must belong to these exact prepared inputs."""
    if freeze.get('config_path') != state['config_path']:
        raise ValueError('source freeze belongs to a different campaign config')
    if freeze.get('pools') != state['pools']:
        raise ValueError('source freeze belongs to different dataset pools')
    expected = {state['config_path']: state['config_sha256'],
                **{path: state['pool_hashes'][d] for d, path in state['pools'].items()}}
    if any(freeze.get('files', {}).get(path) != digest for path, digest in expected.items()):
        raise ValueError('source freeze does not hash the prepared inputs')


def bind_freeze(root, freeze_path):
    """Attach qualified inputs before the first observation; preserve the binding."""
    root = Path(root)
    state = read_json(root/'state.json')
    if state.get('observations') or state.get('freeze_path'):
        raise ValueError('freeze can only be bound once before any observations')
    candidate = dict(state, freeze_path=str(Path(freeze_path).resolve()),
                     freeze_sha256=sha256(freeze_path))
    verify_campaign_inputs(candidate)
    write_json(root/'freeze-binding.json', {k: candidate[k]
               for k in ('freeze_path', 'freeze_sha256', 'config_sha256', 'pool_hashes')})
    write_json(root/'state.json', candidate, replace=True)
    return dict(status=candidate['status'], freeze_path=candidate['freeze_path'])


async def execute(root, *, max_cells=None):
    root = Path(root)
    state = read_json(root/'state.json')
    freeze = verify_campaign_inputs(state)
    base = read_json(state['config_path'])
    pools = {d:load_pool(path) for d,path in state['pools'].items()}
    count = 0
    while max_cells is None or count < max_cells:
        if (root/'STOP').exists():
            return dict(status='paused_at_cell_boundary')
        state = settle_selections(state)
        task = next_task(state)
        if task['action'] != 'measure':
            state['status'] = task['action']
            write_json(root/'state.json', state, replace=True)
            return task
        count += 1
        name = f"{len(state['observations'])+1:05d}-{task['dataset']}-{task['system']}-n{task['n_gpus']}-{task['stage']}-s{task['seed']}"
        cell = root/'results'/'gpu'/name
        dispatch = root/'dispatch'/name
        dispatch.mkdir(parents=True, exist_ok=False)
        config = build_config(base, system=task['system'], dataset=task['dataset'],
            allocated_gpu_ids=list(range(task['n_gpus'])),fixed_pd_p_count=task.get('p_count'),max_frequency_mhz=2520)
        trace = build_trace(pools[task['dataset']], dataset=task['dataset'],n_gpus=task['n_gpus'],
            seed=task['seed'],rate_rps=task['rate_rps'],stage=task['stage'])
        warmup = build_trace(pools[task['dataset']], dataset=task['dataset'],n_gpus=task['n_gpus'],
            seed=task['seed'],rate_rps=task['rate_rps'],stage='warmup')
        config['warmup_trace'] = warmup
        manifest = dict(task, allocated_gpu_ids=list(range(task['n_gpus'])), arrival_window_s=600.,
            slo_ttft_s=config['slo_ttft_s'],slo_tpot_s=config['slo_tpot_s'],model=base['model_name'],
            campaign_id=root.name, protocol_sha256=object_hash(state['protocol']))
        write_json(dispatch/'task.json', task)
        write_json(dispatch/'config.json', config)
        write_json(dispatch/'trace.json', trace)
        write_json(dispatch/'manifest.json', manifest)
        try:
            observation = await run_cell(config, trace, manifest, cell, freeze=freeze)
        except Exception as exc:
            observation = dict(measurement_valid=False,capacity_pass=False,error=type(exc).__name__+': '+str(exc),
                               manifest_path=str(cell/'manifest.json'))
            write_json(dispatch/'failure.json', observation)
        observation['raw_directory'] = str(cell.resolve())
        state = apply_observation(state, task, observation)
        write_json(root/'state.json', state, replace=True)
        write_json(root/'report-inputs.json', [str(Path(r['raw_directory'])/'manifest.json')
                   for r in state['observations'] if (Path(r['raw_directory'])/'manifest.json').is_file()],
                   replace=True)
        print(json.dumps(dict(cell=name,measurement_valid=observation['measurement_valid'],
                             capacity_pass=observation.get('capacity_pass'),status=state['status'])),flush=True)
    return dict(status=state['status'], cells_this_invocation=count)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    prepare=sub.add_parser('prepare')
    prepare.add_argument('--config',type=Path,required=True)
    prepare.add_argument('--sharegpt-pool',type=Path,required=True)
    prepare.add_argument('--longbench-pool',type=Path,required=True)
    prepare.add_argument('--freeze',type=Path)
    prepare.add_argument('--root',type=Path,required=True)
    bind=sub.add_parser('bind-freeze')
    bind.add_argument('--root',type=Path,required=True)
    bind.add_argument('--freeze',type=Path,required=True)
    for command in ('next','run'):
        p=sub.add_parser(command)
        p.add_argument('--root',type=Path,required=True)
        if command=='run':
            p.add_argument('--max-cells',type=int)
    args=parser.parse_args()
    if args.command=='prepare':
        state=new_campaign(args.config,{'sharegpt':args.sharegpt_pool,'longbench':args.longbench_pool},
                           args.root,freeze_path=args.freeze)
        result=dict(status=state['status'],root=str(args.root),next=next_task(state))
    elif args.command=='next':
        result=next_task(read_json(args.root/'state.json'))
    elif args.command=='bind-freeze':
        result=bind_freeze(args.root,args.freeze)
    else:
        if args.max_cells is not None and args.max_cells<1:
            parser.error('--max-cells must be positive')
        with node_lease():
            result=asyncio.run(execute(args.root,max_cells=args.max_cells))
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    main()
