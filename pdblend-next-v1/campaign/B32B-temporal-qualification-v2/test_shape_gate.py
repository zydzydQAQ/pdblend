"""Actual historical parsing tests; supplied oracle values are CPU fixtures only."""
import copy
from pathlib import Path
import pytest
import shape_gate as s

BASE=s.ROOT/'campaign/B32B-baseline-sequence-v1/attempt-001'


def actual():
    r=s.load_reader();gate=BASE/'legacy-correctness'
    binding=r.read(BASE/'correctness-binding/binding.json')
    checks=r.read(gate/'checks/checks.json');http=r.lines(gate/'checks/http.jsonl')
    events={i['id']:r.lines(gate/(i['id']+'.control.events.jsonl')) for i in binding['instances']}
    spec=r.read(s.ROOT/'campaign/B32B-temporal-observation-attempt-003/observation-spec.json')
    outputs=r.read(s.ROOT/'campaign/B32B-temporal-observation-attempt-003/results/child/full-outputs.json')['token_ids_by_request_uuid']
    # This deliberately demonstrates parsing, not an independent oracle: no eligibility is issued.
    fixture={label:outputs[row['request_uuid']] for label,row in zip(s.LABELS,spec['requests'][:4])}
    return checks,http,events,binding['instances'],fixture


def test_actual27_shape_and_legacy_failure_preserved():
    inputs=actual();before=copy.deepcopy(inputs)
    result=s.temporal(*inputs)
    assert result['temporal_native_trajectory_exact'] and result['pair_steps']==69
    assert result['owner_nonempty_steps']==197 and not result['legacy_single_vs_pair_exact']
    assert result['legacy_first_differences'][1]['position_one_based']==32
    assert result['is_performance_qualification'] is False and inputs==before


def test_actual_native_falseflag_cannot_register():
    *_,original=actual();r=s.load_reader()
    base=s.ROOT/'campaign/B32B-native-default-reference-attempt-001'
    spec=r.read(base/'observation-spec.json');out=r.read(base/'results/child/full-outputs.json')['token_ids_by_request_uuid']
    native={label:out[row['request_uuid']] for label,row in zip(s.LABELS,spec['requests'])}
    with pytest.raises(RuntimeError,match='does not exactly'):s.exact_reference(native,original)


@pytest.mark.parametrize('change',['one_token','missing_label','short64'])
def test_all_four_reference_outputs_required(change):
    *_,oracle=actual();changed=copy.deepcopy(oracle)
    if change=='one_token':changed['pair192'][63]+=1
    if change=='missing_label':changed.pop('solo96')
    if change=='short64':changed['pair96'].pop()
    with pytest.raises(RuntimeError):s.exact_reference(changed,oracle)


@pytest.mark.parametrize('change',['wrong_shape','decode_paused','hold_allocated','missing_control_http','fake_legacy_pass'])
def test_original_mechanism_negatives(change):
    checks,http,events,instances,oracle=actual();second=instances[1]['id']
    if change=='wrong_shape':
        rid=next(r['request_id'] for r in checks['requests'] if r['label']=='temporal-second')
        next(e for e in events[second] if rid in e.get('request_ids',[]))['tokens']+=1
    if change=='decode_paused':checks['temporal']['held']['admit_decode']=False
    if change=='hold_allocated':
        rid=next(r['request_id'] for r in checks['requests'] if r['label']=='temporal-second')
        checks['temporal']['held']['kv_allocations'][rid]=1
    if change=='missing_control_http':
        command=checks['temporal']['open_prefill']['command'];http=[r for r in http if not (r['route']=='/control' and r['body']==command)]
    if change=='fake_legacy_pass':checks['checks']['temporal_exact']=True
    with pytest.raises(RuntimeError):s.temporal(checks,http,events,instances,oracle)


def test_template_cannot_qualify_any_system():
    template=s.load_reader().read(Path(__file__).parent/'protocol-template.json')
    assert template['ready'] is False and template['registered_oracle'] is None
    assert not template['eligible_systems']['ecoserve'] and template['fresh_binding'] is None
