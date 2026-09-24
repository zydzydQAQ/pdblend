"""Strict timing/runtime composition for the opt-in whole-layout PD revision.

This loader has no prefill/decode energy functions. The legacy profile loader
and PoolPlanner are unaffected. All native components are independently replayed.
"""
from __future__ import annotations
from pathlib import Path
import math
import statistics

from pdblend.profile.collection.native_timing_audit import need,finite
from pdblend.profile.collection.native_timing_plan import binding,digest
from pdblend.profile.collection.native_timing_replay import Resolver
from .native_composition import replay_sources,_translated,_identity,IDENTITY
from .native_timing import NativeTimingOverlay
from pdblend.profile.collection.native_runtime_collect import write_new
from pdblend.profile.collection.native_frequency_domain import FIELDS,identity_frequencies,require_same_domain

TIMING_KIND='pdblend_native_layout_timing_selection_v1'
PROFILE_KIND='pdblend_native_layout_profile_v1'
RUNTIME_REQUIRED=(('clock_1500_to_2520','elapsed_s'),('clock_2520_to_1500','elapsed_s'))


def runtime_required(identity):
    low,high=identity_frequencies(identity)
    return ((f'clock_{low}_to_{high}','elapsed_s'),(f'clock_{high}_to_{low}','elapsed_s'))


class LayoutQualificationUnavailable(ValueError):
    """Complete untampered raw observations failed a required holdout gate."""
    def __init__(self,gate,message,*,evidence):
        super().__init__(message)
        self.gate=gate;self.evidence=evidence


def replay_layout_runtime_component(reference,resolver,identity,sources):
    """Predeclared M4/TP2 subset; never changes the raw global holdout result."""
    from pdblend.profile.collection.native_runtime_audit import replay_runtime
    from pdblend.profile.collection.native_runtime_collect import RUNTIME_PLAN
    need(identity.get('model_id')=='Qwen2.5-32B-Instruct' and (identity.get('tp'),identity.get('pp'))==(2,1),
         'scoped runtime is restricted to native32 M4/TP2, not the full PD profile')
    report=_translated(resolver.read(reference),resolver)
    need(report.get('complete') is True and report.get('safe_restore_passed',report.get('ready_for_timing')) is True,
         'runtime collection or restoration did not finish')
    caps=report['initial_capabilities'];need(len(caps)==4,'M4/TP2 capacity inventory incomplete')
    for cap in caps.values():
        # Native capabilities bind physical engine identity; revision fields
        # belong to the collection component and are verified by raw replay.
        _identity(cap,{k:v for k,v in identity.items() if k not in FIELDS})
        need(cap.get('source_revision') in sources,'runtime source continuity differs')
    audit=replay_runtime(report)
    need(audit.get('raw_components_complete') is True and audit.get('independent_holdout_collected') is True,
         'scoped runtime raw replay incomplete: '+repr(audit.get('errors')))
    require_same_domain(report,identity)
    if 'frequency_domain' in identity:
        _identity(audit['component_identity'],identity)
    capacity=audit['capacity'];need(len(capacity)==4 and all(type(n)is int and n>0 for n in capacity.values()),
                                  'scoped runtime actual native capacity incomplete')
    rows={(r['component'],r['metric']):r for r in audit['holdout_comparisons']};values={};errors=[]
    required=runtime_required(identity)
    for key in required:
        need(key in rows,'required all-M clock runtime component missing')
        row=rows[key];training=row['training'];held=row['heldout']
        need(len(training)==3 and all(finite(v) and v>0 for v in [*training,held]),
             'required clock runtime raw training/holdout missing or nonpositive')
        prediction=statistics.median(training);error=abs(prediction-held)/held
        need(row['training_prediction']==prediction and row['relative_error']==error,
             'clock component independent error replay differs')
        values[key]=prediction;errors.append(error)
    ordered=sorted(errors);summary=dict(mean_relative_error=statistics.mean(errors),
        p95_relative_error=ordered[max(0,math.ceil(.95*len(ordered))-1)],max_relative_error=max(errors))
    if not all(summary[k]<=limit for k,limit in RUNTIME_PLAN['holdout_limits'].items()):
        raise LayoutQualificationUnavailable('scoped_clock_holdout','required all-M clock component holdout failed',
            evidence=dict(runtime=reference,scope='32B_M4_TP2_capacity_and_clock_only',summary=summary,
                          limits=RUNTIME_PLAN['holdout_limits'],original_global_holdout_passed=audit.get('holdout_passed')))
    return dict(capacity_tokens=min(capacity.values()),values=values,audit=audit,
        scoped_runtime_qualified=True,scope='32B_M4_TP2_capacity_and_clock_only',
        consumed_components=[list(k) for k in required],holdout_error_summary=summary,
        original_global_holdout_passed=audit.get('holdout_passed'),global_result_modified=False,
        transfer_cost_consumed=False,off_wake_power_consumed=False,full_runtime_profile_qualified=False)


def freeze_layout_timing_selection(*,identity,timing_ref,runtime_ref,serving_source_manifest,
                                   calibration_source_manifests,out):
    """Bind completed components; replay before returning a usable reference.

    A rejected immutable document remains an honest failed attempt. This does
    not edit legacy component flags or inherit a pilot as qualification.
    """
    reference=write_new(Path(out),dict(kind=TIMING_KIND,identity=identity,timing=timing_ref,runtime=runtime_ref,
        serving_source_manifest=serving_source_manifest,calibration_source_manifests=calibration_source_manifests))
    try:load_layout_timing(reference)
    except LayoutQualificationUnavailable as exc:
        exc.timing_selection=reference
        raise
    return reference


class _RuntimeTimingBase:
    def __init__(self,identity,runtime):
        self.identity=dict(identity);self.system='pdblend';self.model=identity['model_id'];self.tp=identity['tp'];self.pp=1
        self.freqs=identity_frequencies(identity);self.kv_capacity_tokens=runtime['capacity_tokens'];self.profile_key={}
        self.bounded_coverage={'native_timing_actual_hulls':True};self.decode_power_overrides={}
        self.query_qualification={'formal_eligible':False,'scope':'timing_and_runtime_only'}
        self.runtime_components={'capacity':True,'clock_transition':True,'static':False,'transfer':False}
        self.values=runtime['values'];self.freq_switch_s=max(self.values[key] for key in runtime_required(identity))
    def require_runtime_components(self,*names):need(all(self.runtime_components.get(n) for n in names),'missing_profile: native runtime gate absent')
    def transfer_seconds(self,n):raise ValueError('missing_profile: scoped all-M runtime does not qualify P/D transfer')
    def static_power_w(self,state,f=None):
        raise ValueError('missing_profile: whole-layout mean watts does not expose per-state static power')
    def wake_seconds(self,state):
        raise ValueError('missing_profile: scoped all-M runtime does not qualify off/park wake')


class LayoutTimingModel(NativeTimingOverlay):
    def prefill_marginal_seconds(self,n,f):
        self.nearest_freq(f)
        return 0. if n==0 else super().prefill_marginal_seconds(n,f)


def load_layout_timing(reference,*,path_map=()):
    resolver=Resolver(path_map);selection=resolver.read(reference)
    need(selection.get('kind')==TIMING_KIND,'explicit native layout timing selection required')
    ident=selection['identity'];identity_frequencies(ident)
    required=set(IDENTITY)|(set(FIELDS) if 'frequency_domain' in ident else set())
    need(set(ident)==required and ident['system']=='pdblend'
        and ident['model_id']=='Qwen2.5-32B-Instruct' and (ident['tp'],ident['pp'])==(2,1)
        and all(ident.values()),'native layout model-owned identity differs')
    sources=replay_sources(selection,resolver)
    evidence=resolver.read(selection['timing'])
    from pdblend.profile.collection.native_layout_stage import SCHEMA as RESIDENT_SCHEMA
    if evidence.get('schema')=='pdblend-native-terminal-timing-stage-evidence/v1':
        from pdblend.profile.collection.native_timing_stage import replay_terminal_evidence as replay_evidence
    elif evidence.get('schema')==RESIDENT_SCHEMA:
        from pdblend.profile.collection.native_layout_stage import replay_resident_timing as replay_evidence
    else:
        from pdblend.profile.collection.native_timing_replay_v2 import replay_evidence
    timing=replay_evidence(selection['timing'],path_map=path_map);component=timing['component']
    _identity(timing['identity'],ident)
    need(timing['identity']['source_revision'] in sources['calibration_source_revisions'],'layout timing source continuity differs')
    if timing.get('component_qualified') is not True or not isinstance(component,dict) or component.get('component_qualified') is not True:
        raise LayoutQualificationUnavailable('timing_component_holdout','native layout timing component holdout is unqualified',
            evidence=dict(timing=selection['timing'],component_qualified=timing.get('component_qualified'),
                          model_id=ident['model_id'],raw_replay_completed=True))
    runtime=replay_layout_runtime_component(selection['runtime'],resolver,ident,sources['calibration_source_revisions'])
    model=LayoutTimingModel(_RuntimeTimingBase(ident,runtime),timing)
    model.profile_key=dict(identity=ident,layout_timing_selection_sha256=reference['sha256'])
    model.calibration_identity=dict(ident)
    return model,dict(timing_runtime_qualified=True,formal_eligible=False,source_compatibility=sources,
                      timing_selection=reference,identity=ident,runtime_scope={k:v for k,v in runtime.items()
                          if k not in ('values','audit')},resident_stage_only=timing.get('resident_stage_only',False),
                      physical_cleanup_verified=not timing.get('resident_stage_only',False),
                      queue_terminal_verified=not timing.get('resident_stage_only',False))


def replay_layout_profile(reference,*,path_map=()):
    """Compose a bounded static-layout component, not an online PD profile."""
    from .native_layout_model import replay_layout_component,NativeLayoutEnergyModel
    from pdblend.profile.collection.native_layout_energy import validate_layout_plan,layout_revision
    need(not path_map,'layout energy raw replay currently requires its actual immutable mounted paths')
    resolver=Resolver(path_map);profile=resolver.read(reference)
    need(profile.get('kind')==PROFILE_KIND,'explicit layout revision profile required')
    model,timing=load_layout_timing(profile['timing_profile'],path_map=path_map)
    if timing['resident_stage_only']:
        from pdblend.profile.collection.native_layout_stage import verify_final_timing
        need(profile.get('final_timing_evidence'),'resident timing is not an externally qualified completed job')
        verify_final_timing(resolver.read(profile['timing_profile'])['timing'],profile['final_timing_evidence'])
    plan=resolver.read(profile['energy_plan']);validate_layout_plan(plan)
    revision=layout_revision(plan)
    need(profile.get('revision')==revision,'explicit layout revision frequency domain differs')
    require_same_domain(plan,timing['identity'])
    candidate=resolver.read(profile['candidate']);_identity(candidate['identity'],timing['identity'])
    need(candidate['identity']['source_revision'] in timing['source_compatibility']['calibration_source_revisions'],
         'layout power has no verified source continuity')
    selection=resolver.read(profile['selection']);need(selection['timing_profile']==profile['timing_profile'],
                                                       'energy selection uses another timing profile')
    result=replay_layout_component(plan,profile['candidate'],profile['selection'],profile['holdout'],model)
    need(result['component_qualified'] is True,'native layout energy/selected SLO holdout failed')
    # The opt-in algorithm itself is part of the serving identity, not only the
    # common runtime and measurement code checked by replay_sources.
    source=resolver.read(resolver.read(profile['timing_profile'])['serving_source_manifest'])
    modules=('pdblend/planner/native_layout.py','pdblend/profile/query/native_layout_model.py',
             'pdblend/profile/query/native_layout_profile.py','pdblend/profile/collection/native_layout_energy.py')
    source_root=Path(__file__).parents[3]
    need(all(source['files'].get(p)==binding(source_root/p)['sha256'] for p in modules),
         'serving layout revision source differs from the independently replayed implementation')
    qualification=dict(revision=revision,static_layout_component_qualified=True,formal_profile_eligible=False,
        complete_candidate_replay=True,independent_selected_150s_holdout=True,profile=reference,
        exact_scope=dict(model_id=plan['model_id'],tp=2,pp=1,counts={'M':4},gpu_count=8,
            datasets=['alpaca','sharegpt','longbench'],rate_scales=[.25,.5,.75,1.],frequencies_mhz=list(identity_frequencies(plan)),
            arrival_family='poisson',duration_s=150.,routing='least_load_all_M',min_m_instances=4),
        formal_window_accepted=False,static_offline_selection_only=True,
        online_backlog_or_dynamic_layout_qualified=False,
        remaining_gates=['native_online_backlog_domain','native_clock_transition_and_hysteresis_domain',
            'source_bound_online_planner_factory_and_fallback','raw_online_controller_acceptance'],
        component=result,timing_runtime=timing)
    model.profile_key=dict(model.profile_key,layout_profile_sha256=reference['sha256'],revision=revision)
    return model,NativeLayoutEnergyModel(candidate),qualification
