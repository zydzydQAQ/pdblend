"""CPU-only contracts for a combined diagnostic that reuses real controls."""
import importlib.util
import json
from pathlib import Path

import pytest

from pdblend.bench.comparison_campaign import binding
from pdblend.bench.resident_session import digest


spec = importlib.util.spec_from_file_location('prepare_combined_recovery',
    Path(__file__).resolve().parents[2]/'scripts/2026-09-24_prepare_combined_recovery_ab.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def observations(tmp_path):
    points, windows = [], []
    for index, case in enumerate(module.CASES):
        point = dict(name=case+'-control-r0', recovery_experiment=dict(case=case, arm='control', repeat=0))
        path = tmp_path/(str(index)+'.json')
        path.write_text(json.dumps(dict(point_sha256=digest(point))))
        points.append(point)
        windows.append(dict(case=case, arm='control', repeat=0, receipt=binding(path),
            point_sha256=digest(point), artifact_valid=True, canonical_metrics_valid=True,
            energy_metrics_valid=True, cleanup_passed=True, measurement_valid=False))
    return dict(points=points), dict(windows=windows)


def test_combined_freezes_reviewed_defaults_without_optional_artifact_mechanisms():
    value = module.combined_options()
    assert value['shield_mode'] == 'budget_aware' and value['slo_routing']
    assert value['safety_recovery'] and value['preserve_overload_capacity']
    assert value['experiment_mode'] == 'adaptive'
    assert (value['shield_sustained_gap_s'], value['shield_stalled_fraction'], value['shield_stalled_min_requests']) == (1., .25, 2)
    assert (value['slo_routing_safety'], value['slo_routing_handoff_floor_s']) == (.85, 0.)
    assert not value['joint_resident'] and not value['capacity_floor_reserve_canonical']
    assert all(value[key] is None for key in module.ARTIFACTS)


def test_reused_controls_retain_original_failed_qualification(tmp_path):
    parent, observed = observations(tmp_path)
    points, refs = module.base_controls(parent, observed)
    assert points == parent['points'] and len(refs) == 4
    assert all(not row['measurement_qualified'] and row['original_qualification_retained'] for row in refs.values())


@pytest.mark.parametrize('fault', ['missing', 'duplicate', 'point', 'artifact', 'cleanup', 'checksum'])
def test_control_reuse_refuses_missing_ambiguous_or_changed_measurements(tmp_path, fault):
    parent, observed = observations(tmp_path)
    row = observed['windows'][0]
    if fault == 'missing': observed['windows'].pop(0)
    elif fault == 'duplicate': observed['windows'].append(row)
    elif fault == 'point': row['point_sha256'] = 'foreign'
    elif fault == 'artifact': row['artifact_valid'] = False
    elif fault == 'cleanup': row['cleanup_passed'] = False
    else: Path(row['receipt']['path']).write_text('{}')
    with pytest.raises(ValueError):
        module.base_controls(parent, observed)
