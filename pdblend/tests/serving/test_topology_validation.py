from copy import deepcopy

import pytest

from ecopadg.serving.topology_validation import background_continuity, measured_costs, validate_transitions


def spec(key, tp, gpus, port):
    return dict(id=key, tp=tp, gpus=gpus, port=port, kv_port=port+100)


def manifest():
    return dict(instances=[spec('pair', 2, [0, 1], 18000), spec('background', 1, [2], 18010)],
        background_instance='background', reference_instances=['pair', 'background'],
        transitions=[dict(remove=['pair'], add=[spec('left', 1, [0], 18200), spec('right', 1, [1], 18210)]),
                     dict(remove=['left', 'right'], add=[spec('merged', 2, [0, 1], 18400)])])


def test_complete_sequence_validated_before_physical_work():
    validate_transitions(manifest())
    invalid = deepcopy(manifest())
    invalid['transitions'][1]['add'][0]['gpus'] = [1, 2]
    with pytest.raises(ValueError, match='overlapping'):
        validate_transitions(invalid)


def test_cannot_remove_background_or_hide_existing_instance_by_id():
    invalid = manifest()
    invalid['transitions'][0]['remove'].append('background')
    with pytest.raises(ValueError, match='background'):
        validate_transitions(invalid)
    invalid = manifest()
    invalid['transitions'][0]['add'][0]['id'] = 'background'
    with pytest.raises(ValueError, match='overlapping'):
        validate_transitions(invalid)


def test_missing_ordinary_target_reference_rejected():
    invalid = manifest()
    invalid['reference_instances'] = ['background']
    with pytest.raises(ValueError, match='reference'):
        validate_transitions(invalid)


def test_remove_then_add_uses_cached_weights_and_free_gpus():
    value = manifest()
    value['retained_weights'] = '/cached/weights'
    value['transitions'] += [dict(remove=['merged'], add=[]),
        dict(remove=[], add=[spec('added', 1, [0], 18500)])]
    validate_transitions(value)
    value['transitions'][-1]['add'][0]['gpus'] = [2]
    with pytest.raises(ValueError, match='overlapping'):
        validate_transitions(value)


def test_empty_change_and_add_without_weights_rejected():
    value = manifest()
    value['transitions'].append(dict(remove=[], add=[]))
    with pytest.raises(ValueError, match='empty'):
        validate_transitions(value)
    value['transitions'][-1]['add'] = [spec('added', 1, [3], 18500)]
    with pytest.raises(ValueError, match='retained weights'):
        validate_transitions(value)


def test_repeated_costs_enclose_every_observation_not_last_or_average():
    rows=[dict(source_tps=[2],target_tps=[1,1],started_s=0,finished_s=40,total_node_energy_j=1000),
          dict(source_tps=[2],target_tps=[1,1],started_s=50,finished_s=100,total_node_energy_j=900),
          dict(source_tps=[1],target_tps=[],started_s=101,finished_s=102,total_node_energy_j=400)]
    costs=measured_costs(rows,'raw')
    assert len(costs)==2
    assert costs[0]['duration_upper_s']==60 and costs[0]['energy_upper_j']==1200
    assert costs[1]['target_tps']==[] and costs[1]['source_sha256']=='raw'


def test_background_request_can_cross_a_fast_delete_without_extending_its_cost():
    rows=[dict(started_s=1.,finished_s=3.,passed=True),
          dict(started_s=3.1,finished_s=4.,passed=True),
          dict(started_s=1.5,finished_s=2.,passed=False)]
    assert background_continuity(rows,1.8,2.1)==rows[:1]
    assert not background_continuity(rows,4.1,4.2)
