"""Whole baseline replacements with A's actual final first-loss scope."""
import final_baseline_overlay_v2 as previous
import a_eco_scope_v1 as a_scope
from final_baseline_overlay_v2 import eligible, audit_quarantined_original, verify_fresh_identity, required_baselines


def validate(p,selection,originals):
    return a_scope.apply(p,selection,previous.validate(p,selection,originals))


def annotate(point,overlay):
    previous.annotate(point,overlay)
    item=overlay['fresh'].get(point['cell_id'],{})
    if item.get('execution_scope'):
        point.update(execution_scope=item['execution_scope'],execution_release=item['execution_release'])
    if item.get('excluded_above_first_loss'):
        point.update(selected_for_final_comparison=False,required_baseline_execution=False,
            baseline_scope_status='excluded_above_first_complete_PDB_loss',
            exclusion=item['excluded_above_first_loss'])
    return point


def verify_scope(p,overlay,points):
    return a_scope.verify_against_measured(p,overlay,points)
