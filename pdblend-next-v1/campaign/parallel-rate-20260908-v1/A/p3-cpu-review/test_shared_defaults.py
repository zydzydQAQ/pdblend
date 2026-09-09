"""Independent A-profile safety checks for the shared p3 superset."""
import json,sys
from pathlib import Path
from types import SimpleNamespace
R=Path('/root/workspace/pdblend-next-v1')
H=R/'campaign/parallel-rate-20260908-v1/hosts/14b-fixed-p3'
sys.path[:0]=[str(H/'src'),'/root/workspace/pdblend/.runtime-deps']
from ecopadg.serving.profiles import ProfileStore
from ecopadg.serving.planner import JointPlanner
from ecopadg.serving.frequency import coverage_recovery,recovery_actions
P=R/'campaign/main-slo-improvement-v1/A/long-batch6-profile-001/profiles.development.json'

def test_a_phase_fallback_absent_remains_disabled_for_all_measured_points_and_misses():
    store=ProfileStore.load(P)
    assert 'mixed_decode_phase_fallback' not in json.loads(P.read_text())
    assert store.mixed_decode_phase_fallback is False
    for point in store.points:
        args=(point.role,point.tp,point.frequency_mhz,point.input_tokens,point.context_tokens,point.batch)
        assert store.lookup_execution_phase(*args) is store.lookup(*args)
        assert store.lookup_execution_phase(*args[:-1],10000) is None

def test_decode_phase_fallback_requires_optin_and_preserves_original_role():
    point=next(p for p in ProfileStore.load(P).points if p.role=='decode')
    args=('mixed',point.tp,point.frequency_mhz,point.input_tokens,point.context_tokens,point.batch)
    assert ProfileStore([point]).lookup_execution_phase(*args) is None
    enabled=ProfileStore([point],mixed_decode_phase_fallback=True)
    assert enabled.lookup_execution_phase(*args) is point and point.role=='decode'
    assert enabled.lookup_execution_phase(*args[:-1],10000) is None

def test_a_recovery_default_retains_maximum_actions_and_expiry_without_coverage_access():
    planner=JointPlanner(ProfileStore.load(P),allow_pd=False)
    assert planner.coverage_aware_recovery is False
    instance=SimpleNamespace(instance_id='actual-a')
    assert coverage_recovery(planner,instance,123.) is None
    actions,expiry,used=recovery_actions(planner,[(instance,True)],123.,124.,maximum=2520)
    assert [(a.instance_id,a.frequency_mhz) for a in actions]==[('actual-a',2520)]
    assert expiry==124. and used is False
