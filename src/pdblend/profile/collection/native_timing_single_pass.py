"""Explicit single-observation development timing; no formal qualification.

The original model-owned design supplies every unique train/holdout shape.
Only repeat count changes. Old power, wall-clock token timestamps and failed
attempts are never converted into CUDA observations by this module.
"""
from __future__ import annotations

from copy import deepcopy

from .native_timing_audit import need
from .native_timing_plan import digest, read_bound
from .native_timing_plan_v2 import validate_plan as validate_parent

SCHEMA = 'pdblend-native-timing-development-plan/v1'
INPUT_SCHEMA = 'pdblend-native-timing-development-inputs/v1'
COLLECTION_SCHEMA = 'pdblend-native-timing-development-collection/v1'
EVIDENCE_SCHEMA = 'pdblend-native-timing-development-replay-evidence/v1'
REPLAY_SCHEMA = 'pdblend-native-timing-development-replay/v1'
LEVEL = 'single_pass_development'


def is_single_pass(plan):
    return plan.get('schema') == SCHEMA


def from_parent(parent, parent_ref):
    """Derive after the caller has independently validated the bound v2 parent."""
    need(parent.get('schema') == 'pdblend-native-timing-plan/v2'
         and parent.get('model_id') in ('Qwen2.5-7B-Instruct', 'Qwen2.5-14B-Instruct')
         and parent.get('tp') == 1 and parent.get('resident_instances') == 8,
         'single-pass revision is restricted to the model-owned 7B/14B eight-replica design')
    plan = deepcopy(parent)
    shapes = [{k: v for k, v in point.items() if k != 'repeats'} for point in plan['points']]
    need(len({digest(point) for point in shapes}) == len(shapes)
         and all(point['repeats'] == 3 for point in plan['points']),
         'parent must contain unique shapes with the original three-repeat design')
    for point in plan['points']:
        point['repeats'] = 1
    plan.update(schema=SCHEMA, parent_point_plan=deepcopy(parent_ref),
                qualification_level=LEVEL, interference_repeats=0,
                execution_mode='parallel_development_unqualified',
                original_repeats=3, observation_repeats=1,
                original_design_qualified=False, component_qualified=False,
                parallel_qualified=False, collector_integration_required=False,
                required_remaining=['single_pass_raw_replay_and_holdout_diagnostics',
                    'measurement_interference_qualification_not_collected',
                    'independent_repeatability_not_collected',
                    'power_and_runtime_component_qualification',
                    'complete_tuning_query_replay', 'formal_workload_energy'])
    return plan


def build_plan(parent_plan_ref):
    return from_parent(validate_parent(read_bound(parent_plan_ref)), parent_plan_ref)


def validate_plan(plan):
    need(is_single_pass(plan), 'explicit single-pass development schema required')
    expected = build_plan(plan['parent_point_plan'])
    need(digest(plan) == digest(expected),
         'single-pass plan changes bound shapes, holdout, repeats or qualification')
    return plan


def development_qualification(checks, gpu_uuids):
    need(checks == [], 'single-pass development does not collect interference probes')
    need(isinstance(gpu_uuids, list) and len(gpu_uuids) == len(set(gpu_uuids)) == 8
         and all(isinstance(u, str) and u.startswith('GPU-') for u in gpu_uuids),
         'single-pass development requires eight distinct owned physical GPUs')
    return dict(qualified=False, parallel_qualified=False,
                mode='parallel_development_unqualified', checks=[],
                exclusive_fleet_gpu_uuids=list(gpu_uuids), energy_comparable=False,
                qualification_level=LEVEL, interference_measured=False,
                original_design_qualified=False)
