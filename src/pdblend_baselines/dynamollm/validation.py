"""Independent V1 preflight; no hardware actions or implicit profile fallbacks."""
import json
import math
from pathlib import Path
from .deployment import sha
from .policy import PERIODS


MODELS = {
    'Qwen2.5-7B-Instruct': (1, 2, 4),
    'Qwen2.5-14B-Instruct': (1, 2, 4),
    'Qwen2.5-32B-Instruct': (2, 4),
}


def preflight(config, *, mode='functional', duration_s=100, seed=701):
    """Validate model-bound assets before constructing NVML or any engine."""
    from .predictor import model_identity, verify_checkpoint
    from .profiles import PaperProfiles
    missing = {}
    evidence = {}
    model = config.get('model_id')
    if seed != 701:
        missing['seed'] = 'this campaign uses workload seed 701 only'
    if mode not in ('functional', 'primitive', 'full'):
        missing['mode'] = 'unknown execution mode'
    if model not in MODELS:
        missing['model'] = 'explicit supported Qwen2.5 target required'
    instances = config.get('instances', [])
    ids = [row.get('instance_id', row.get('id')) for row in instances]
    flat = [g for row in instances for g in row.get('gpus', [])]
    node = config.get('node_gpus', [])
    if (not instances or None in ids or len(set(ids)) != len(ids) or len(set(flat)) != len(flat)
            or len(node) > 8 or len(set(node)) != len(node) or not set(flat) <= set(node)):
        missing['placement'] = 'unique identities and nonoverlapping owned GPU groups required'
    for row in instances:
        if (row.get('tp') not in MODELS.get(model, ()) or row.get('pp', 1) != 1
                or len(row.get('gpus', [])) != row.get('tp')):
            missing['placement'] = 'model-specific TP and PP1 required; no 32B TP1 or TP8 P/D'
    identity = None
    try:
        identity = model_identity(config['model_path'])
        if identity['model'] != model:
            raise ValueError('target model directory differs from declared identity')
        evidence['model_identity'] = identity
    except (KeyError, OSError, ValueError, TypeError) as exc:
        missing['model_identity'] = str(exc)
    try:
        directory = config['dynamo_predictor_dir']
        if identity is None:
            raise ValueError('verified target model required before predictor binding')
        verify_checkpoint(directory, expected_model=model,
                          expected_tokenizer_sha256=identity['tokenizer_sha256'])
        evidence['predictor_manifest_sha256'] = sha(Path(directory) / 'manifest.json')
    except (KeyError, OSError, ValueError, TypeError) as exc:
        missing['missing_predictor'] = str(exc)
    try:
        path = Path(config['profiles'])
        value = json.loads(path.read_text())
        key = value.get('profile_key', {})
        if (value.get('system', key.get('system')) != 'dynamollm'
                or value.get('model', value.get('model_id', key.get('model_id'))) != model
                or value.get('engine_revision', key.get('engine_revision')) != 'vllm-0.10.1.1'
                or value.get('independent_profile') is not True):
            raise ValueError('independent model-bound Dynamo V1 profile required')
        profile = PaperProfiles.load(path)
        for row in instances:
            if not profile.frequencies(row['tp']):
                raise ValueError('no independent profile for initial TP' + str(row['tp']))
        evidence['profile_sha256'] = profile.fingerprint
    except (KeyError, OSError, ValueError, TypeError) as exc:
        missing['missing_profile'] = str(exc)
    if mode in ('primitive', 'full'):
        if not config.get('goldens'):
            missing['missing_golden'] = 'same-target-TP native golden receipts required'
        for tp_key, golden in config.get('goldens', {}).items():
            try:
                tp = int(tp_key)
                if (tp not in MODELS.get(model, ()) or golden.get('tp') != tp
                        or golden.get('model_id') != model
                        or golden.get('engine_revision') != 'vllm-0.10.1.1'
                        or golden.get('seed') != 701 or not golden.get('prompt')
                        or not golden.get('token_ids') or len(golden['token_ids']) > 64
                        or sha(golden['source_path']) != golden['source_sha256']):
                    raise ValueError('target golden identity, output or raw source differs')
                evidence['golden_tp' + str(tp)] = golden['source_sha256']
            except (KeyError, OSError, ValueError, TypeError) as exc:
                missing['missing_golden_tp' + str(tp_key)] = str(exc)
    if mode == 'primitive':
        primitive = config.get('primitive', {})
        if (not primitive.get('source_ids') or not primitive.get('target_layout')
                or len(primitive.get('target_shapes', [])) != len(primitive.get('target_layout', []))):
            missing['primitive_plan'] = 'explicit source identities, target GPUs and shapes required'
    if mode == 'full':
        if duration_s < 1890:
            missing['duration'] = 'full hierarchy needs >=1890 real seconds'
        if not config.get('dynamo_shape_demands'):
            missing['missing_shape_profile'] = 'frozen shape demands required'
        if not config.get('dynamo_transition_costs'):
            missing['missing_transition_cost'] = 'own measured V1 transition costs required'
        try:
            _, _, receipt = verified_history(config['dynamo_weekly_history'])
            evidence['weekly_history'] = receipt
        except (KeyError, OSError, ValueError, TypeError) as exc:
            missing['missing_history'] = str(exc)
    return dict(ready=not missing, status='ready' if not missing else 'inconclusive',
                missing_evidence=missing, evidence=evidence, system='dynamollm',
                model_id=model, mode=mode, seed=seed, periods_s=dict(PERIODS),
                hardware_actions_started=False, formal_eligible=False,
                hardware_qualified=False, energy_comparable=False)


def workload_contract(value):
    if (value.get('schema')!=1 or value.get('split')!='mechanism_validation'
            or value.get('controller_reads_future') is not False
            or value.get('production_prediction_accuracy_evaluation') is not False
            or value.get('seed')!=701):
        raise ValueError('independent frozen mechanism workload and no future-controller access required')
    duration=value.get('duration_s')
    if type(duration) not in (int,float) or not math.isfinite(duration) or duration<1890:
        raise ValueError('original-period validation needs at least 1890 real seconds')
    rows=value.get('requests')
    if not isinstance(rows,list) or not rows:raise ValueError('actual frozen mechanism requests required')
    previous=-1.
    for row in rows:
        at=row.get('at_s');n=row.get('input_tokens');o=row.get('output_tokens');prompt=row.get('prompt')
        if (type(at) not in (int,float) or not math.isfinite(at) or not 0<=at<duration or at<previous
                or type(n) is not int or not 1<=n<=7168 or type(o) is not int or not 1<=o<=512
                or n+o>8192 or not isinstance(prompt,list) or len(prompt)!=n
                or any(type(token) is not int or token<0 for token in prompt)):
            raise ValueError('mechanism request time or fixed 8192-context work differs')
        previous=at
    return dict(requests=len(rows),duration_s=duration,seed=701,periods_s=dict(PERIODS),
                controller_reads_future=False,production_prediction_accuracy_evaluation=False)


def verified_history(path):
    """Verify actual author source, prepared counts, and frozen adapter bytes."""
    from .history_adapter import PINNED_ASSETS
    from .policy import WeeklyLoadTemplate
    path=Path(path).resolve();value=json.loads(path.read_text())
    summary_path=path.parent/'summary.json';summary=json.loads(summary_path.read_text())
    source=Path(value['source_path']);expected=PINNED_ASSETS.get(source.name)
    if (expected is None or value.get('source_sha256')!=expected or sha(source)!=expected
            or summary.get('source_sha256')!=expected or summary.get('source_path')!=str(source)
            or summary.get('source_and_implementation_unchanged') is not True
            or not summary.get('usable_independent_history')
            or summary.get('artifacts',{}).get(path.name)!=sha(path)
            or value.get('split')!='calibration' or value.get('evaluation_trace_used') is not False
            or value.get('prompt_corpora_used') is not False or value.get('timestamp_replay_speed')!=1):
        raise ValueError('verified same-service real author history required')
    evidence={str(path):sha(path),str(summary_path):sha(summary_path),str(source):expected}
    for name,expected_code in summary['implementation_files'].items():
        frozen=path.parent/'implementation'/name
        if Path(name).name!=name or sha(frozen)!=expected_code:raise ValueError('frozen history adapter differs')
        evidence[str(frozen)]=expected_code
    provenance=value['provenance']
    if (provenance.get('repository')!='Azure/AzurePublicDataset'
            or provenance.get('revision')!='207bed67dd10090b28ad4f745b2cfd41a11aace4'):
        raise ValueError('official author revision differs')
    existing = provenance.get('source_receipt_kind') == 'existing_file_revalidation'
    receipt=Path(provenance['source_receipt'] if existing else provenance['download_receipt'])
    receipt_sha=provenance.get('source_receipt_sha256' if existing else 'download_receipt_sha256')
    if sha(receipt)!=receipt_sha:
        raise ValueError('author download receipt differs')
    receipt_value=json.loads(receipt.read_text())
    if existing and (receipt_value.get('schema')!='dynamo-existing-source-revalidation-v1'
            or receipt_value.get('operation')!='verify_existing_file'
            or receipt_value.get('downloaded') is not False
            or receipt_value.get('prior_download_receipt_restored') is not False
            or receipt_value.get('repository')!=provenance['repository']
            or receipt_value.get('revision')!=provenance['revision']
            or receipt_value.get('bytes')!=source.stat().st_size
            or receipt_value.get('official_reference_files')!=provenance['official_reference_files']):
        raise ValueError('existing-file revalidation receipt contract differs')
    if (receipt_value.get('sha256')!=expected or receipt_value.get('verified') is not True
            or Path(receipt_value['path']).resolve()!=source.resolve()):raise ValueError('source receipt identity differs')
    evidence[str(receipt)]=sha(receipt)
    for name,expected_reference in provenance['official_reference_files'].items():
        if sha(name)!=expected_reference:raise ValueError('official author reference differs')
        evidence[name]=expected_reference
    total=sum(row['count'] for row in value['records'])
    if total!=value['aggregated_requests'] or total!=summary['rows']:
        raise ValueError('aggregated actual source request count differs')
    template=WeeklyLoadTemplate.fit(value['records'],start_s=value['start_s'],end_s=value['end_s'])
    return value,template,dict(service=provenance['service'],requests=total,evidence=evidence,
        calendar_exposure_assumption=value['temporal_endpoint_assumption'],
        chronological_holdout_available=summary['chronological_holdout']['available'],
        history_mapped_by='UTC weekday/weekend and 300-second time-of-day; one second remains one second')
