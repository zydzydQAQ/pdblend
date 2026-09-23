"""Strict, provenance-preserving merge of frequency-sharded raw profiles."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import statistics
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def merge_raw(paths: list[Path], out: Path) -> dict:
    paths = [Path(p).resolve() for p in paths]
    raws = [json.loads(p.read_text()) for p in paths]
    if len(raws) < 2:
        raise ValueError('at least two shards required')
    first = raws[0]
    required = ('schema', 'model', 'tp', 'kv_capacity_tokens', 'kv_bytes_per_token')
    for key in required:
        if not first.get(key) or any(r.get(key) != first[key] for r in raws):
            raise ValueError(f'missing or inconsistent {key}')
    if first['schema'] != 2:
        raise ValueError('unsupported raw schema')
    for section, keys in (
        ('config', ('decode_repeats', 'decode_settle_s', 'decode_measure_s', 'decode_batches')),
        ('environment', ('image_digest', 'source_hash', 'vllm', 'torch', 'cuda', 'python')),
    ):
        for key in keys:
            value = first.get(section, {}).get(key)
            if value is None or any(r.get(section, {}).get(key) != value for r in raws):
                raise ValueError(f'missing or inconsistent {section}.{key}')
    frequencies = [f for r in raws for f in r['freqs']]
    if len(set(frequencies)) != len(frequencies):
        raise ValueError('overlapping frequency shards (conflicting measurement ownership)')
    result = copy.deepcopy(first)
    result['freqs'] = sorted(frequencies)
    result['config'].pop('base_port', None)
    result['config']['mixed_freqs'] = sorted({x['freq_mhz'] for r in raws for x in r['mixed']})
    result['shards'] = []

    def relocate(value, source):
        if isinstance(value, list):
            return [relocate(v, source) for v in value]
        if not isinstance(value, dict):
            return value
        row = {k: relocate(v, source) for k, v in value.items()}
        if 'samples_file' in row:
            evidence = (source.parent / row['samples_file']).resolve()
            if not evidence.is_file():
                raise ValueError(f'missing evidence: {evidence}')
            row['samples_file'] = os.path.relpath(evidence, out.resolve())
            row['samples_sha256'] = sha256(evidence)
        return row

    for section, keys in (
        ('prefill', ('freq_mhz', 'input_tokens')),
        ('decode', ('freq_mhz', 'context_tokens', 'batch')),
        ('mixed', ('freq_mhz', 'batch', 'chunk_tokens')),
    ):
        seen, rows = set(), []
        for path, raw in zip(paths, raws):
            for row in raw[section]:
                key = tuple(row[k] for k in keys)
                if key in seen or row['freq_mhz'] not in raw['freqs']:
                    raise ValueError(f'conflicting {section} point: {key}')
                seen.add(key)
                rows.append(relocate(row, path))
        result[section] = sorted(rows, key=lambda r: tuple(r[k] for k in keys))
    # Preserve all contributors; the aggregate uses the median of common static points.
    result['static'] = {}
    for key in sorted({k for r in raws for k in r['static']}):
        vals = [r['static'][key] for r in raws if key in r['static']]
        result['static'][key] = {k: statistics.median(v[k] for v in vals if k in v)
                                 for k in {k for v in vals for k in v}}
    result['transfer'] = []
    lengths = set()
    for r in raws:
        for row in r['transfer']:
            if row['input_tokens'] in lengths:
                raise ValueError('duplicate transfer measurements need explicit reconciliation')
            lengths.add(row['input_tokens'])
            result['transfer'].append(copy.deepcopy(row))
    result['freq_switch_s'] = [x for r in raws for x in r['freq_switch_s']]
    uuids = []
    for path, raw in zip(paths, raws):
        env = raw['environment']
        if not isinstance(env.get('gpu_uuids'), list) or not env['gpu_uuids']:
            raise ValueError('missing GPU UUID list')
        uuids.extend(env['gpu_uuids'])
        result['shards'].append(dict(path=os.path.relpath(path, out.resolve()), sha256=sha256(path),
                                    freqs=raw['freqs'], environment=env, config=raw['config'],
                                    gpus=raw['gpus'], static=raw['static'], elapsed_s=raw.get('elapsed_s')))
    if len(set(uuids)) != len(uuids):
        raise ValueError('GPU UUIDs overlap between shards')
    result['gpus'] = sorted(set(uuids))  # global identity; shard.gpus retains local NVML indices
    result['environment']['gpu_uuids'] = sorted(set(uuids))
    result['elapsed_s'] = max(r.get('elapsed_s', 0) for r in raws)
    result['merge'] = dict(static_aggregation='median', elapsed_semantics='maximum shard duration')
    return result
