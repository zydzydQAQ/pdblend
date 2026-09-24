"""Strict CPU replay and bounded composition of native PD calibration evidence.

This is a new selection kind. Historical component flags and registries remain
unchanged; neither a pilot nor a safely restored fleet is a qualified profile.
"""
from __future__ import annotations
from copy import deepcopy
import hashlib
import math
from pathlib import Path
import statistics

from pdblend.profile.collection.native_timing_plan import digest,binding
from pdblend.profile.collection.native_timing_replay import Resolver,replay_evidence
from pdblend.profile.collection.native_timing_audit import need,finite
from pdblend.profile.collection.native_frequency_domain import require_same_domain
from pdblend.profile.collection.native_runtime_audit import replay_runtime
from .native_power_components import replay_power
from .native_timing import NativeTimingOverlay
from .index import CurveIndex
from pdblend.profile.collection.native_timing_plan_v2 import MODEL_TP

KIND='pdblend_native_profile_selection_v1'
IDENTITY=('system','model_id','tp','pp','model_hash','tokenizer_hash','image_digest','engine_revision')
GATES=('identity','source_compatibility','timing_raw_and_holdout','runtime_raw_and_holdout',
       'power_training_and_holdout','native_serving_energy_holdout','complete_tuning_query_coverage')


def _translated(value,resolver):
    if isinstance(value,dict):
        result={k:_translated(v,resolver) for k,v in value.items()}
        if {'path','sha256'}<=value.keys():
            path=resolver.path(value['path'])
            need(binding(path)['sha256']==value['sha256'],'native component raw checksum differs')
            result['path']=str(path)
        return result
    if isinstance(value,list):return [_translated(v,resolver) for v in value]
    return value


def _identity(actual,expected):
    require_same_domain(actual,expected)
    need(all(actual.get(k)==v for k,v in expected.items() if k!='system'),
         'actual native component model/image/topology identity differs')


def _source_files(ref,resolver):
    source=resolver.read(ref);files=source['files'];root=resolver.path(ref['path']).parent
    need(files and digest(files)==source['source_sha256'],'native source manifest digest differs')
    for name,sha in files.items():
        path=(root/name).resolve()
        need(path.is_relative_to(root) and binding(path)['sha256']==sha,'immutable native source bytes differ: '+name)
    return source


def replay_sources(selection,resolver):
    target=_source_files(selection['serving_source_manifest'],resolver)
    protected=lambda files:{k:v for k,v in files.items() if k.startswith(('pdblend_runtime/','pdblend/engine/','pdblend/measure/'))
        or k in ('pdblend/bench/comparison_metering.py','pdblend/bench/comparison_metrics.py','pdblend/bench/client.py')}
    wanted=protected(target['files']);need(wanted,'serving native/measurement source inventory empty')
    worker='pdblend/profile/collection/native_timing_worker.py'
    # This known worker only appends true per-request shapes to inherited CUDA
    # events. A different calibration worker requires a new explicit review.
    known=Path(__file__).parents[1]/'collection/native_timing_worker.py'
    worker_sha=binding(known)['sha256'];sources={}
    for ref in selection['calibration_source_manifests']:
        source=_source_files(ref,resolver)
        need(protected(source['files'])==wanted,'native numerical/runtime/measurement bytes differ across components')
        need(source['files'].get(worker)==worker_sha,'unknown calibration worker cannot inherit native equivalence')
        sources[source['source_sha256']]=source
    need(sources,'calibration source lineage missing')
    return dict(serving_source_sha256=target['source_sha256'],calibration_source_revisions=sorted(sources),
                protected_files=wanted,calibration_worker_sha256=worker_sha)


def replay_runtime_component(ref,resolver,identity,sources):
    report=_translated(resolver.read(ref),resolver)
    need(report.get('complete') is True and report.get('safe_restore_passed',report.get('ready_for_timing')) is True,
         'native runtime collection/restore did not complete')
    caps=report['initial_capabilities'];need(len(caps)==8//identity['tp'],'runtime component needs the model-owned full eight-GPU inventory')
    for cap in caps.values():
        _identity(cap,identity)
        need(cap['source_revision'] in sources,'runtime source has no verified lineage')
    audit=replay_runtime(report)
    need(audit['raw_components_complete'],'runtime raw replay failed: '+repr(audit['errors']))
    need(audit.get('independent_holdout_collected') is True and audit.get('holdout_passed') is True,
         'native runtime independent holdout is absent or failed')
    # The old audit's component_qualified=False is retained. Qualification here
    # comes from its reproducible raw/holdout checks plus the new source gates.
    values={(r['component'],r['metric']):r['training_prediction'] for r in audit['holdout_comparisons']}
    needed={(s,'group_power_w') for s in ('active_idle@1500','active_idle@2520','active_idle_reset','L1','off')}
    needed|={(s,'elapsed_s') for s in ('wake','unpark','clock_1500_to_2520','clock_2520_to_1500')}
    needed|={('transfer_'+str(n),'second_output_overhead_s') for n in (512,2048,7168)}
    need(needed<=values.keys(),'native runtime component inventory lacks required static/clock/wake/transfer inputs')
    need(all(finite(values[k]) and values[k]>0 for k in needed),'native runtime training predictor is nonpositive')
    return dict(report=report,audit=audit,values=values,capacity_tokens=min(audit['capacity'].values()))


class NativeRuntimePowerModel:
    def __init__(self,identity,power,runtime):
        self.identity=identity;self.system='pdblend';self.model=identity['model_id'];self.tp=identity['tp'];self.pp=identity['pp']
        self.freqs=(1500,2520);self.power=power;self.runtime=runtime
        self.runtime_components={k:True for k in ('capacity','static','transfer','clock_transition')}
        self.kv_capacity_tokens=runtime['capacity_tokens'];self.profile_key={};self.bounded_coverage={'native_actual_domains':True}
        self.decode_power_overrides={'native':power.candidate};self.decode_overrides={}
        self.query_qualification=dict(native_bounded=True,formal_eligible=False)
        values=runtime['values'];self.freq_switch_s=max(values[(s,'elapsed_s')] for s in ('clock_1500_to_2520','clock_2520_to_1500'))
        self.transfer=CurveIndex((512,2048,7168),[values[('transfer_'+str(n),'second_output_overhead_s')] for n in (512,2048,7168)])
    def require_runtime_components(self,*names):need(all(self.runtime_components.get(n) for n in names),'missing_profile: native runtime component absent')
    def prefill_power_w(self,n,f):
        need(self.power.candidate.get('active_prefill_kernel_power_qualified') is True,
             'missing_profile: whole-request-cycle mean power cannot be multiplied by CUDA prefill time; '
             'an independently qualified active-prefill or complete-cycle predictor is required')
        return self.power.predict('prefill',1,n,f)
    def decode_power_w(self,b,f,*,ctx=None):return self.power.predict('decode',b,ctx,f)
    def decode_power_supported(self,b,c,f):
        try:self.decode_power_w(b,f,ctx=c);return True
        except ValueError:return False
    def static_power_w(self,state,f=None):
        name='active_idle@'+str(f) if state=='active_idle' else ('L1' if state in ('parked','L1') else state)
        key=(name,'group_power_w');need(key in self.runtime['values'],'missing_profile: unknown native static state')
        return self.runtime['values'][key]
    def wake_seconds(self,state):
        if state in ('off','L1','parked'):return self.runtime['values'][('wake' if state=='off' else 'unpark','elapsed_s')]
        if state in ('active_idle','active_idle_reset'):return 0.
        raise ValueError('missing_profile: unknown wake state')
    def transfer_seconds(self,n):return self.transfer.predict(n)


class NativeComposedModel(NativeTimingOverlay):
    @property
    def query_qualification(self):
        return dict(native_bounded=True,native_timing_evidence=self.native_timing_replay['evidence'],
                    formal_eligible=getattr(self,'calibration_qualification',{}).get('formal_eligible',False))
    def prefill_marginal_seconds(self,n,f):
        self.nearest_freq(f)
        # Exactly zero additional tokens is an algebraic identity, not a GPU
        # measurement at length zero or an extension of the sampled domain.
        return 0. if n==0 else super().prefill_marginal_seconds(n,f)


def replay_queries(ref,resolver,model,candidate_sha):
    from .native_query_replay import replay_queries as replay
    return replay(ref,resolver,model,candidate_sha)


def audit_native_profile(selection,*,path_map=()):
    """Return all missing gates; caller-supplied formal flags have no authority."""
    resolver=Resolver(path_map);result=dict(schema='pdblend-native-profile-composition-audit/v1',
        gates={},missing_gates=[],gate_failures={},formal_eligible=False,full_profile_qualified=False,
        energy_comparable=False,old_component_flags_unchanged=True)
    state={}
    def gate(name,fn):
        try:state[name]=fn();result['gates'][name]=True
        except (OSError,ValueError,KeyError,TypeError,RuntimeError,IndexError,AttributeError) as exc:
            result['gates'][name]=False;result['missing_gates'].append(name);result['gate_failures'][name]=str(exc)
    def identity():
        need(selection.get('kind')==KIND,'unknown native profile selection kind')
        value=selection['identity'];need(set(value)==set(IDENTITY) and value['system']=='pdblend'
             and value['tp']==MODEL_TP.get(value['model_id']) and value['pp']==1
             and all(value[k] for k in IDENTITY),'complete model-owned native TP1/TP2 identity required')
        return value
    gate('identity',identity)
    gate('source_compatibility',lambda:replay_sources(selection,resolver))
    def timing():
        evidence=resolver.read(selection['timing'])
        if evidence.get('schema')=='pdblend-native-terminal-timing-stage-evidence/v1':
            from pdblend.profile.collection.native_timing_stage import replay_terminal_evidence as replay_timing
        elif evidence.get('schema')=='pdblend-native-timing-replay-evidence/v2':
            from pdblend.profile.collection.native_timing_replay_v2 import replay_evidence as replay_timing
        else:replay_timing=replay_evidence
        replay=replay_timing(selection['timing'],path_map=path_map);component=replay['component']
        _identity(component['identity'],state['identity'])
        need(component['identity']['source_revision'] in state['source_compatibility']['calibration_source_revisions'],
             'timing source has no verified lineage')
        need(component.get('component_qualified') is True,'native timing raw holdout failed')
        return replay
    gate('timing_raw_and_holdout',timing)
    gate('runtime_raw_and_holdout',lambda:replay_runtime_component(selection['runtime'],resolver,state['identity'],
        state['source_compatibility']['calibration_source_revisions']))
    def power():
        evidence=resolver.read(selection['power'])
        need(evidence.get('schema')!='pdblend-native-power-pilot/v1',
             'power pilot is not a calibrated component: missing training-only candidate, frozen candidate binding, '
             'independent holdout, qualified actual-context domain and serving-energy composition')
        ident=dict(state['identity'],source_revision=evidence.get('identity',{}).get('source_revision'))
        need(ident['source_revision'] in state['source_compatibility']['calibration_source_revisions'],
             'power source has no verified lineage')
        return replay_power(evidence,resolver,ident)
    gate('power_training_and_holdout',power)
    def model():
        if 'model' not in state:
            base=NativeRuntimePowerModel(state['identity'],state['power_training_and_holdout'][1],state['runtime_raw_and_holdout'])
            state['model']=NativeComposedModel(base,state['timing_raw_and_holdout'])
        return state['model']
    candidate={k:selection.get(k) for k in ('identity','serving_source_manifest','calibration_source_manifests','timing','runtime','power')}
    result['candidate_sha256']=digest(candidate)
    gate('complete_tuning_query_coverage',lambda:replay_queries(selection['query_ledger'],resolver,model(),result['candidate_sha256']))
    def energy():
        from .native_serving_holdout import replay_serving_holdout
        return replay_serving_holdout(resolver.read(selection['serving_energy_holdout']),resolver,model(),
            candidate_sha256=result['candidate_sha256'],sources=state['source_compatibility'],
            required_deployments=state['complete_tuning_query_coverage']['chosen_candidates'])
    gate('native_serving_energy_holdout',energy)
    result['formal_eligible']=result['full_profile_qualified']=result['energy_comparable']=all(result['gates'].get(k) is True for k in GATES)
    result['component_summary']={key:deepcopy(value) for key,value in state.items() if key in
        ('source_compatibility','native_serving_energy_holdout','complete_tuning_query_coverage')}
    return result,state.get('model')


def load_native_profile(path,*,system,model_id,tp,pp,usage):
    from .versions import LoadedVersion,VersionError
    resolver=Resolver();selection=resolver.read(binding(path));audit,model=audit_native_profile(selection)
    identity=selection.get('identity',{})
    if any(identity.get(k)!=v for k,v in dict(system=system,model_id=model_id,tp=tp,pp=pp).items()):
        raise VersionError('native composed selection identity differs')
    component_gates=GATES[:5]
    required=GATES if usage=='formal' else component_gates
    missing=[g for g in required if audit['gates'].get(g) is not True]
    if missing or model is None:
        raise VersionError('native profile blocked: '+ '; '.join(g+': '+audit['gate_failures'].get(g,'missing') for g in missing))
    qualification=dict(audit,usage=usage,planner_automatically_wired=True,consumer_loader_available=True)
    key=dict(identity,native_selection_sha256=binding(path)['sha256'],native_candidate_sha256=audit['candidate_sha256'])
    coverage=dict(exact_frequencies=[1500,2520],timing_actual_hulls=model.native_timing_replay['component']['models'],
                  power_actual_nodes=model.base.power.candidate['nodes'],workload_scope=audit['component_summary'].get('native_serving_energy_holdout'))
    model.profile_key=deepcopy(key);model.calibration_identity=deepcopy(identity)
    model.calibration_qualification=deepcopy(qualification);model.calibration_coverage=deepcopy(coverage)
    return LoadedVersion(model,identity,coverage,qualification,key)
