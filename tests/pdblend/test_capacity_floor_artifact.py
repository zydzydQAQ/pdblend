import hashlib
import json

import pytest

from pdblend.planner.capacity import load_capacity_floors, select_artifact
from pdblend.planner.transitions import identity
from synthetic import synthetic_model


def test_floor_cannot_trust_passed_flag_or_skip_real_gpu_acceptance(tmp_path):
    model = synthetic_model()
    manifest = tmp_path / 'acceptance.json'
    manifest.write_text(json.dumps({'accepted': True, 'stage': 'capacity', 'pairs': []}))
    artifact = tmp_path / 'floor.json'
    artifact.write_text(json.dumps(dict(kind='pdblend_capacity_floor_v1', identity=identity(model),
        stage='capacity', acceptance_manifest={'path': 'acceptance.json',
        'sha256': hashlib.sha256(manifest.read_bytes()).hexdigest()}, floors=[])))
    with pytest.raises(ValueError, match='passed controlled GPU acceptance'):
        load_capacity_floors(artifact, model=model)
    manifest.write_text('{}')
    with pytest.raises(ValueError, match='checksum'):
        load_capacity_floors(artifact, model=model)


def test_artifact_index_requires_explicit_topology_entry(tmp_path):
    index = tmp_path / 'index.json'
    index.write_text(json.dumps({'kind': 'pdblend_optimization_artifact_set_v1',
                                 'profiles': {'tp2-pp1': 'two.json'}}))
    with pytest.raises(ValueError, match='misses tp1'):
        select_artifact(index, synthetic_model())
