"""Prepare immutable inputs for functional, primitive or real-period Dynamo runs."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from .deployment import save, sha
from .policy import PERIODS, SHAPES
from .predictor import model_identity
from .profiles import PaperProfiles
from .validation import MODELS, preflight


def merge_own_profiles(paths, *, model_id):
    points, sources = [], {}
    for path in paths:
        path = Path(path)
        # Reuse the portable-profile validator so merge cannot trust a
        # tampered cell SHA/holdout flag while ignoring raw samples,
        # capabilities, geometry, or fit values.
        from .portable_profile import validate_profile
        raw_value = json.loads(path.read_text())
        source_paths = [row.get('source_profile_path') for row in raw_value.get('points', [])]
        if not source_paths:
            raise ValueError('profile has no source cells')
        source_root = Path(source_paths[0]).resolve().parent.parent
        validate_profile(artifact_root=source_root, recorded_root=source_root,
                         profile=path, allow_profile_outside=True)
        value = raw_value
        if (value.get('system') != 'dynamollm' or value.get('model_id', value.get('model')) != model_id
                or value.get('engine_revision') != 'vllm-0.10.1.1'
                or value.get('independent_profile') is not True
                or value.get('coordinate_system') != 'input_output_batch'):
            raise ValueError('merge requires own same-model V1 profiles with the same coordinates')
        PaperProfiles.load(path)
        for point in value['points']:
            if point.get('tp') not in MODELS[model_id] or point.get('pp', 1) != 1:
                raise ValueError('profile topology is outside this eight-GPU PP1 campaign')
            artifact = Path(point['source_profile_path'])
            if sha(artifact) != point['source_sha256']:
                raise ValueError('own profile cell checksum differs')
            cell = json.loads(artifact.read_text())
            if (cell['identity']['model_id'] != model_id or cell['identity']['system'] != 'dynamollm'
                    or cell['fit']['holdout_passed'] is not True):
                raise ValueError('own profile cell identity or holdout differs')
            points.append(point)
        sources[str(path.resolve())] = sha(path)
    if not points:
        raise ValueError('no measured independent Dynamo points to merge')
    return dict(schema=2, measurement='hardware', system='dynamollm', model_id=model_id, model=model_id,
        engine_revision='vllm-0.10.1.1', coordinate_system='input_output_batch', independent_profile=True,
        points=points, profile_sources=sources, formal_eligible=False, hardware_qualified=False)


def main(argv=None):
    from .profile_v1 import integers
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--profiles', type=Path, nargs='+', required=True)
    parser.add_argument('--predictor', type=Path, required=True)
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--gpus', type=integers, required=True)
    parser.add_argument('--initial-tp', type=int, required=True)
    parser.add_argument('--base-port', type=int, default=19000)
    parser.add_argument('--mode', choices=('functional','primitive','full'), default='functional')
    parser.add_argument('--duration', type=float, default=100)
    parser.add_argument('--weekly-history', type=Path)
    parser.add_argument('--history-mapping', type=Path)
    parser.add_argument('--history-reference-start', type=float)
    parser.add_argument('--shape-demands', type=Path)
    parser.add_argument('--transition-costs', type=Path)
    parser.add_argument('--goldens', type=Path)
    parser.add_argument('--primitive', type=Path)
    parser.add_argument('--functional-profile-stage', type=Path,
                        help='explicit development-only after-start profile stage JSON')
    parser.add_argument('--slo-ttft', type=float, default=5)
    parser.add_argument('--slo-tpot', type=float, default=.15)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError('refusing to overwrite prepared Dynamo run inputs')
    identity = model_identity(args.model)
    model = identity['model']
    if args.initial_tp not in MODELS.get(model, ()) or len(args.gpus) > 8:
        parser.error('legal model-specific TP and <=8 owned GPUs required')
    if args.mode == 'full' and (len(args.gpus) != 8 or args.duration < 1890):
        parser.error('full mechanism run requires eight-GPU budget and >=1890 real seconds')
    merged = merge_own_profiles(args.profiles, model_id=model)
    args.out.mkdir(parents=True)
    profile_path = args.out/'profiles.json'
    save(profile_path, merged)
    profile = PaperProfiles.load(profile_path)
    maximum = max(profile.frequencies(args.initial_tp))
    # Primitive reserves spare GPUs for the explicitly supplied target. Other
    # modes allocate complete initial replicas within their own fixed budget.
    count = 2 if args.mode == 'primitive' else len(args.gpus)//args.initial_tp
    if count < 1 or count*args.initial_tp > len(args.gpus):
        parser.error('initial replicas do not fit the owned group')
    instances = []
    for index in range(count):
        port = args.base_port+index*2
        instances.append(dict(id='dynamo-initial-'+str(index),
            gpus=args.gpus[index*args.initial_tp:(index+1)*args.initial_tp], tp=args.initial_tp, pp=1,
            role='mixed', shape='LL', frequency_mhz=maximum, generation=0,
            port=port, url='http://127.0.0.1:'+str(port)))
    config = dict(system='dynamollm', model_id=model, model_path=str(args.model.resolve()),
        profiles=str(profile_path.resolve()), dynamo_predictor_dir=str(args.predictor.resolve()),
        tokenizer=str(args.model.resolve()), trace=str(args.trace.resolve()), node_gpus=args.gpus,
        legal_tp=list(MODELS[model]), instances=instances, base_port=args.base_port,
        target_port=args.base_port+32, store_port=args.base_port+80,
        slo_ttft_s=args.slo_ttft, slo_tpot_s=args.slo_tpot, dynamo_reference_tp=4,
        periods_s=dict(PERIODS), evidence_class='development', formal_eligible=False)
    if args.functional_profile_stage:
        stage = json.loads(args.functional_profile_stage.read_text())
        if stage.get('enabled') is not True or stage.get('mode', 'functional') != 'functional':
            parser.error('functional profile stage must be explicitly enabled in functional mode')
        config['functional_profile_stage'] = stage
    for option, key in ((args.shape_demands,'dynamo_shape_demands'),
                        (args.transition_costs,'dynamo_transition_costs'),
                        (args.goldens,'goldens'), (args.primitive,'primitive'),
                        (args.history_mapping,'dynamo_history_mapping')):
        if option:
            config[key] = json.loads(option.read_text())
    if args.weekly_history:
        config['dynamo_weekly_history'] = str(args.weekly_history.resolve())
    if args.history_reference_start is not None:
        config['dynamo_history_reference_start_s'] = args.history_reference_start
    for shape, demand in config.get('dynamo_shape_demands', {}).items():
        if shape not in SHAPES or not profile.configurations(**demand):
            raise ValueError('shape demand lacks its own measured feasible configurations: '+shape)
    save(args.out/'config.json', config)
    receipt = preflight(config, mode=args.mode, duration_s=args.duration, seed=701)
    save(args.out/'preflight.json', receipt)
    print(json.dumps(dict(receipt, execution_argv=['python','-m','pdblend_baselines.dynamollm.run_v1',
        '--config',str((args.out/'config.json').resolve()),'--trace',str(args.trace.resolve()),
        '--out',str((args.out/'execution').resolve()),'--duration',str(args.duration),'--seed','701','--mode',args.mode])))
    return 0 if receipt['ready'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
