"""Audit own raw-backed profile coverage against cached visible-prompt predictions.

This is a CPU inventory, never a fitted capacity model. Evaluation outputs are
used only to check the fixed-work maximum envelope, not to predict or fit it.
No arrival times, admission rates or GPU measurements are generated here.
"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

from .deployment import sha
from .prediction_cache import read_prefix
from .profiles import CoverageError, PaperProfiles
from .portable_profile import validate_profile

FREQUENCIES = (900, 1200, 1500, 1800, 2100, 2520)


def cached_rows(cache: Path, corpus: Path, dataset: str, split: str):
    cache, corpus = Path(cache), Path(corpus)
    completion = json.loads((cache/'completion.json').read_text())
    binding_path = cache/'binding.json'
    binding = json.loads(binding_path.read_text())
    if (completion.get('complete') is not True
            or completion.get('binding_sha256') != sha(binding_path)
            or binding.get('corpus_manifest_sha256') != sha(corpus/'manifest.json')):
        raise ValueError('completed prediction cache binding differs')
    source = corpus/(dataset+'.json')
    manifest = json.loads((corpus/'manifest.json').read_text())
    source_sha = sha(source)
    if source_sha != manifest['dataset_sha256'][dataset]:
        raise ValueError('prediction corpus bytes differ')
    path = cache/(dataset+'-'+split+'.jsonl')
    if completion['artifacts'].get(str(path.resolve())) != sha(path):
        raise ValueError('prediction cache rows checksum differs')
    source_rows = json.loads(source.read_text())[split]
    rows = read_prefix(path, source_rows, sha(binding_path))
    if (len(rows) != len(source_rows) or completion['counts'].get(dataset+'-'+split) != len(rows)
            or any(row['corpus_file_sha256'] != source_sha
                   or row.get('dataset') != dataset or row.get('split') != split
                   or row.get('output_limit_used_for_prediction') is not False
                   or row.get('actual_output_limit') != source_rows[i]['output_tokens']
                   for i, row in enumerate(rows))):
        raise ValueError('complete cache/corpus request inventory differs')
    return binding, rows


def coverage(rows, profiles: PaperProfiles, tp: int):
    memo = {}
    def supported(n, o):
        key = n, o
        if key not in memo:
            errors = {}
            for frequency in FREQUENCIES:
                try:
                    profiles.query(tp, frequency, n, n+o, 1)
                except CoverageError as exc:
                    errors[str(frequency)] = str(exc)
            memo[key] = errors
        return not memo[key]
    prediction_ok = sum(supported(row['input_tokens'],row['predicted_output']) for row in rows)
    # The runtime may raise its output estimate after observing emitted tokens.
    # Checking both ends is necessary, but not sufficient for arbitrary batches
    # or interpolation accuracy, which remain independent gates below.
    full_ok = sum(supported(row['input_tokens'],row['predicted_output'])
                  and supported(row['input_tokens'],max(row['predicted_output'],row['actual_output_limit']))
                  for row in rows)
    failures = Counter(error for errors in memo.values() for error in errors.values())
    return dict(requests=len(rows),batch=1,tp=tp,frequencies=list(FREQUENCIES),
        predicted_shape_supported=prediction_ok,fixed_work_envelope_supported=full_ok,
        all_requests_supported=full_ok==len(rows),
        input_bounds=[min(r['input_tokens'] for r in rows),max(r['input_tokens'] for r in rows)],
        prediction_counts=dict(Counter(r['predicted_output'] for r in rows)),
        output_limit_bounds=[min(r['actual_output_limit'] for r in rows),max(r['actual_output_limit'] for r in rows)],
        unique_queries=len(memo),failure_counts=dict(failures),
        larger_batches_qualified=False,interpolation_holdout_qualified=False,
        parallel_layout_qualified=False,formal_eligible=False)


def inventory(*, model_id, tp, profiles, cache, corpus):
    portable, sources = [], []
    for profile, artifact_root in profiles:
        value = validate_profile(profile=profile,artifact_root=artifact_root,recorded_root='/output')
        if value['model_id'] != model_id or value['tp'] != tp:
            raise ValueError('same-model initial-TP profile required')
        portable.extend(value['points'])
        sources.append(dict(profile=str(Path(profile).resolve()),sha256=sha(profile),
                            receipt=value['portable_profile_receipt']))
    own = PaperProfiles(portable,coordinate_system='input_output_batch')
    datasets = {}
    for dataset in ('alpaca','sharegpt','longbench'):
        for split in ('calibration','evaluation'):
            binding,rows = cached_rows(cache,corpus,dataset,split)
            if binding['model_identity']['model'] != model_id:
                raise ValueError('prediction cache belongs to another model')
            datasets[dataset+'-'+split] = coverage(rows,own,tp)
    return dict(schema='dynamo-own-profile-coverage-inventory-v1',system='dynamollm',
        model_id=model_id,tp=tp,pp=1,raw_revalidated_cells=len(portable),
        source_profiles=sources,prediction_cache_completion_sha256=sha(Path(cache)/'completion.json'),
        datasets=datasets,arrival_times_bound=False,rate_anchor_status='missing_rate_anchor',
        fit_performed=False,holdout_used_for_fitting=False,formal_eligible=False,
        measurement_reuse='raw-backed measurements only; neither old source identity nor hardware UUIDs are rewritten',
        remaining_gates=['uncovered workload shapes','batch > 1 capacity coverage',
                         'independent interpolation holdout','parallel interference qualification',
                         'additional legal TP coverage for ScaleShard',
                         'loaded transition envelope matching real admission bounds',
                         'original stationary-weight retention'])
