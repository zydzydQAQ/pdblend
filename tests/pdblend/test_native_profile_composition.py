"""Strict native replay rejects pilots, mutated samples, leakage and fake flags."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from pdblend.profile.collection.native_timing_plan import binding
from pdblend.profile.collection.native_timing_replay import Resolver
from pdblend.profile.query.native_power_components import fit_nodes,replay_power,NativePowerTable
from pdblend.profile.query.native_composition import audit_native_profile,KIND,replay_sources
from pdblend.profile.query.native_serving_holdout import replay_serving_holdout
from pdblend.profile.query.versions import load_profile,VersionError
from test_pdblend_native_power import raw_fixture


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
    return binding(path)


def identity():
    return dict(system='pdblend',model_id='Qwen2.5-7B-Instruct',tp=1,pp=1,model_hash='model',
                tokenizer_hash='tokenizer',image_digest='image',engine_revision='vllm-0.10.1.1')


@pytest.mark.parametrize('flags',[{},dict(formal_eligible=True,full_profile_qualified=True,energy_comparable=True)])
def test_empty_or_forged_formal_selection_reports_every_missing_component(flags):
    report,model=audit_native_profile(dict(kind=KIND,**flags))
    assert model is None and not report['formal_eligible']
    assert len(report['missing_gates'])==7


def test_load_profile_native_dispatch_cannot_use_legacy_formal_flag(tmp_path):
    path=tmp_path/'selection.json'
    put(path,dict(kind=KIND,identity=identity(),quality={'formal_eligible':True},formal_eligible=True))
    with pytest.raises(VersionError,match='native profile blocked'):
        load_profile(path,system='pdblend',model_id='Qwen2.5-7B-Instruct',tp=1,usage='formal')


def test_pilot_and_ready_for_timing_do_not_grant_power_qualification():
    pilot=dict(schema='pdblend-native-power-pilot/v1',complete=True,ready_for_timing=True,
               safe_restore_passed=True,formal_eligible=True,component_qualified=True)
    with pytest.raises(ValueError,match='pilot has no training candidate'):
        replay_power(pilot,Resolver(),identity())


def _shift_state(value,delta):
    if isinstance(value,dict):
        for key,item in value.items():
            if key in ('at_s','native_at_s','response_at_s','scheduler_at_s') and isinstance(item,(int,float)):
                value[key]+=delta
            else:_shift_state(item,delta)
    elif isinstance(value,list):
        for item in value:_shift_state(item,delta)


def calibration(tmp_path):
    from pdblend.profile.collection.native_power_audit import audit_power_window
    ident=dict(identity(),source_revision='source');training=[];holdout=[];points=[];refs=[];index=0
    for purpose in ('training','holdout'):
        for f in (1500,2520):
            for role in ('prefill','decode'):
                for length in ((128,512) if purpose=='training' else (256,)):
                    point=dict(role=role,frequency_mhz=f,batch=1,prompt_tokens=length,output_tokens=1 if role=='prefill' else 8192-length,
                        purpose=purpose,seed=9701 if purpose=='training' else 9702,repeats=3)
                    points.append(point)
                    for repeat in range(3):
                        raw=raw_fixture(role);raw['point']=dict(point,repeat=repeat);raw['capability'].update(ident)
                        delta=index*20.;index+=1
                        for key in ('settle_started_s','start_s','end_s'):raw[key]+=delta
                        for row in raw['client_requests']:
                            row['submitted_s']+=delta
                            if 'observed_running_through_s' in row:row['observed_running_through_s']+=delta
                        _shift_state(raw['drain'],delta);_shift_state(raw['service_end_state'],delta)
                        # Cancellation and drain intentionally refer to the same
                        # native object in the fixture; do not shift it twice.
                        for row in raw['sample']['ranks'][0]['samples']:
                            row['at_s']+=delta
                            row['prompt_lengths']=[length]
                            row['context_lengths']=[c+length-128 for c in row['context_lengths']]
                            if role=='prefill':row['scheduled_lengths']=[length]
                        power=raw['power']
                        for key in ('samples','frequency_samples','utilization_samples'):
                            power[key]=[(t+delta,v) for t,v in power[key]]
                        for row in power['power_metadata']:row['read_finished_s']=[t+delta for t in row['read_finished_s']]
                        for _,values in power['frequency_samples']:values[0]=f
                        audit=audit_power_window(raw)
                        target=tmp_path/f'{index}.json';ref=put(target,raw)
                        (training if purpose=='training' else holdout).append((raw,audit,ref));refs.append(ref)
    candidate=fit_nodes(training,ident);candidate_ref=put(tmp_path/'candidate.json',candidate)
    for raw,audit,ref in holdout:
        raw['point']['candidate_sha256']=candidate_ref['sha256']
        updated=put(Path(ref['path']),raw);refs[refs.index(ref)]=updated
    plan=put(tmp_path/'plan.json',dict(schema='pdblend-native-power-calibration-plan/v1',
        evaluation_used_for_selection=False,identity=ident,points=points))
    evidence=dict(schema='pdblend-native-power-calibration-evidence/v1',identity=ident,plan=plan,
        candidate=candidate_ref,candidate_frozen_s=max(r['drain']['response_at_s'] for r,_,_ in training)+1,
        windows=refs)
    return evidence,ident


def test_real_raw_window_refit_and_holdout_can_qualify_component_without_formal_profile(tmp_path):
    evidence,ident=calibration(tmp_path);result,table=replay_power(evidence,Resolver(),ident)
    assert result['component_qualified'] and result['raw_windows']==36
    assert not result['serving_energy_composition_qualified'] and not result['low_fractional_batch_qualified']
    assert table.predict('decode',1,300,1500)==pytest.approx(100)
    with pytest.raises(ValueError,match='low fractional'):table.predict('decode',1.55,300,1500)
    with pytest.raises(ValueError,match='actual measured domain'):table.predict('decode',1,121,1500)


@pytest.mark.parametrize('change',['missing_repeat','candidate','candidate_freeze','identity','purpose','holdout_hash','sample_sha'])
def test_component_flags_never_override_mutation_leakage_or_missing_holdout(tmp_path,change):
    evidence,ident=calibration(tmp_path);evidence.update(formal_eligible=True,component_qualified=True)
    if change=='missing_repeat':evidence['windows'].pop()
    elif change=='candidate_freeze':evidence['candidate_frozen_s']=10000.
    elif change=='candidate':
        value=json.loads(Path(evidence['candidate']['path']).read_text());value['nodes'][0]['power_w']+=1
        evidence['candidate']=put(Path(evidence['candidate']['path']),value)
    else:
        index=-1 if change=='holdout_hash' else 0
        ref=evidence['windows'][index];path=Path(ref['path']);raw=json.loads(path.read_text())
        if change=='identity':raw['capability']['model_hash']='different-model'
        elif change=='purpose':raw['point']['purpose']='coverage_feasibility_pilot'
        elif change=='holdout_hash':raw['point']['candidate_sha256']='not-frozen-candidate'
        elif change=='sample_sha':raw['power']['samples'][10][1][0]=999
        updated=put(path,raw)
        if change!='sample_sha':evidence['windows'][index]=updated
    with pytest.raises(ValueError):replay_power(evidence,Resolver(),ident)


def test_serving_energy_holdout_cannot_be_replaced_by_pure_pilot_or_flags():
    with pytest.raises(ValueError,match='serving-energy holdout required'):
        replay_serving_holdout(dict(schema='pdblend-native-power-pilot/v1',formal_eligible=True),Resolver(),None,
                               candidate_sha256='candidate',sources={})


def test_serving_holdout_rejects_evaluation_even_when_hash_bound(tmp_path):
    plan=put(tmp_path/'plan.json',dict(schema='pdblend-native-serving-energy-plan/v1',
        candidate_sha256='candidate',selection_split='evaluation',seed=9702,evaluation_used_for_selection=True))
    with pytest.raises(ValueError,match='independent predeclared'):
        replay_serving_holdout(dict(schema='pdblend-native-serving-energy-holdout/v1',candidate_sha256='candidate',plan=plan),
                               Resolver(),None,candidate_sha256='candidate',sources={})


def test_request_cycle_prefill_power_cannot_be_multiplied_by_kernel_duration():
    from types import SimpleNamespace
    from pdblend.profile.query.native_composition import NativeRuntimePowerModel
    model=object.__new__(NativeRuntimePowerModel)
    model.power=SimpleNamespace(candidate={'active_prefill_kernel_power_qualified':False})
    with pytest.raises(ValueError,match='cannot be multiplied'):
        model.prefill_power_w(128,1500)


def test_scalar_success_list_cannot_replace_complete_tuning_enumeration(tmp_path):
    from pdblend.profile.query.native_query_replay import replay_queries
    ref=put(tmp_path/'queries.json',dict(schema='pdblend-native-composed-query-ledger/v1',
        formal_eligible=True,ledgers=[dict(queries=[])]))
    with pytest.raises(ValueError,match='full-candidate replay'):
        replay_queries(ref,Resolver(),None,'candidate')


def test_query_logger_preserves_rejected_branches_and_never_swallows_invalid_numbers():
    from pdblend.profile.query.native_query_replay import QueryLog
    class Model:
        def decode_supported(self,b,c,f):return b<=32
        def step_seconds(self,b,c,f):
            if b>32:raise ValueError('missing_profile: native max_num_seqs exceeded')
            return float('nan')
    log=QueryLog(Model())
    assert log.decode_supported(33,121,1500) is False
    with pytest.raises(ValueError,match='missing_profile'):log.step_seconds(33,121,1500)
    assert all(r['unsupported'] for r in log.rows)
    with pytest.raises(RuntimeError,match='invalid numeric'):log.step_seconds(1,121,1500)


def test_unexpected_value_error_cannot_be_classified_as_unsupported_candidate():
    from pdblend.profile.query.native_query_replay import QueryLog
    class Model:
        def step_seconds(self,*args):raise ValueError('corrupt coefficients')
    with pytest.raises(RuntimeError,match='unexpected profile'):
        QueryLog(Model()).step_seconds(1,121,1500)
