"""Validate independently measured, model-bound V1 transition costs."""
import json
import math
from pathlib import Path
from .deployment import sha


def validate_cost(row, demand, *, model_id=None):
    envelope = row.get('workload_envelope', {})
    if (row.get('measurement') != 'hardware' or row.get('system') != 'dynamollm'
            or row.get('engine_revision') != 'vllm-0.10.1.1'
            or not row.get('model_id') or (model_id is not None and row['model_id'] != model_id)
            or any(not math.isfinite(row.get(key, 0)) or row.get(key, 0) <= 0
                   for key in ('duration_s', 'energy_j'))
            or demand['input_tokens'] > envelope.get('max_input_tokens', 0)
            or demand['output_tokens'] > envelope.get('max_output_tokens', 0)
            or ('max_batch' in envelope and (type(demand.get('batch')) is not int
                or demand['batch'] < 1 or demand['batch'] > envelope['max_batch']))):
        raise ValueError('independent model-bound V1 transition cost coverage absent')
    path = Path(row['audit_path'])
    if not path.is_file() or sha(path) != row['audit_sha256']:
        raise ValueError('Dynamo transition audit source changed')
    value = json.loads(path.read_text())
    for key in ('system', 'model_id', 'engine_revision', 'source_sha256', 'source_tps',
                'target_tps', 'energy_j', 'duration_s', 'workload_envelope'):
        if row[key] != value[key]:
            raise ValueError('Dynamo cost differs from measured transition: ' + key)
    if not value.get('evidence'):
        raise ValueError('Dynamo transition raw artifacts absent')
    for artifact, expected in value['evidence'].items():
        if sha(artifact) != expected:
            raise ValueError('Dynamo transition raw evidence changed')
    return True
